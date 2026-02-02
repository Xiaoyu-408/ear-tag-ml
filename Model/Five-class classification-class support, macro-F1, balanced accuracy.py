import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# ======================================================
# 0. Reproducibility
# ======================================================
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

# ======================================================
# 1. Paths and parameters
# ======================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "Bootstrap_Metrics")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 100
STEP_SIZE   = 150

N_BOOT     = 500
CONF_LEVEL = 0.95

LABEL_NAMES = ["0", "1", "2", "3", "4"]

# ======================================================
# 2. Feature extraction (11 engineered features)
# ======================================================
def extract_aligned_features(window_df, b_stats, stage_ref_mean):
    feats = {}
    rel = {}

    for col in ["pH", "K+", "Ca2+"]:
        v = window_df[col].values
        if len(v) < 3 or np.std(v) < 1e-12:
            return None

        mean_v = np.mean(v)
        rel_shift = mean_v - b_stats[col]["mean"]
        rel[col] = rel_shift

        feats[f"{col}_rel_m"]     = rel_shift
        feats[f"{col}_trend_off"] = mean_v - stage_ref_mean[col]
        feats[f"{col}_cv"]        = np.std(v) / (mean_v + 1e-6)

    feats["pH_K_ratio"]    = rel["pH"]   / (abs(rel["K+"])  + 1e-6)
    feats["Ca_vs_pH_sync"] = rel["Ca2+"] / (abs(rel["pH"]) + 1e-6)

    return feats

# ======================================================
# 3. Build feature table (multiclass)
# ======================================================
def build_feature_table():
    rows = []

    for i in range(1, 13):
        path = os.path.join(BASE_DIR, f"{i}.xlsx")
        if not os.path.exists(path):
            continue

        df = pd.read_excel(path).iloc[:, [0, 2, 3, 4]]
        df.columns = ["Day_Label", "pH", "K+", "Ca2+"]
        df["Day_Label"] = df["Day_Label"].astype(str).str.lower().str.strip()

        day0 = df[df["Day_Label"] == "day0"]
        b_stats = {c: {"mean": day0[c].mean()} for c in ["pH", "K+", "Ca2+"]}
        stage_ref = df.iloc[:500][["pH", "K+", "Ca2+"]].mean().to_dict()

        pig_id = "Pig_1" if i <= 4 else ("Pig_2" if i <= 8 else "Pig_3")
        label_type = (i - 1) % 4 + 1

        for s in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
            feats = extract_aligned_features(
                df.iloc[s:s + WINDOW_SIZE], b_stats, stage_ref
            )
            if feats is None:
                continue

            feats["Pig_ID"] = pig_id
            feats["Label"] = label_type if df.iloc[s]["Day_Label"] != "day0" else 0
            rows.append(feats)

    return pd.DataFrame(rows)

# ======================================================
# 4. Train model (Pig1 + Pig2)
# ======================================================
def train_model(train_df):
    X = train_df.drop(columns=["Label", "Pig_ID"])
    y = train_df["Label"].values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    w = dict(zip(np.unique(y), weights))

    if 2 in w:
        w[2] *= 2.0

    sample_w = np.array([w[i] for i in y])

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=8,
        reg_lambda=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )

    model.fit(X, y, sample_weight=sample_w)
    return model, X.columns

# ======================================================
# 5. Bootstrap evaluation (Excel only)
# ======================================================
def bootstrap_metrics(model, X_test, y_test):
    records = []
    n = len(y_test)

    for _ in range(N_BOOT):
        idx = np.random.choice(n, n, replace=True)
        y_true = y_test[idx]
        y_pred = model.predict(X_test.iloc[idx])

        records.append({
            "Accuracy": accuracy_score(y_true, y_pred),
            "Macro_F1": f1_score(y_true, y_pred, average="macro"),
            "Balanced_Acc": balanced_accuracy_score(y_true, y_pred)
        })

    df = pd.DataFrame(records)

    summary = pd.DataFrame({
        "Metric": df.columns,
        "Mean": df.mean().values,
        "CI_lower": df.quantile(0.025).values,
        "CI_upper": df.quantile(0.975).values
    })

    return summary, df

# ======================================================
# 6. Main workflow
# ======================================================
if __name__ == "__main__":

    print("Building feature table...")
    df = build_feature_table()

    train_df = df[df["Pig_ID"].isin(["Pig_1", "Pig_2"])]
    test_df  = df[df["Pig_ID"] == "Pig_3"]

    print("Training model...")
    model, feat_cols = train_model(train_df)

    X_test = test_df[feat_cols]
    y_test = test_df["Label"].values

    # ---------------------
    # Per-class support (Excel)
    # ---------------------
    support_df = (
        pd.DataFrame({"Class": y_test})
        .value_counts()
        .rename("Support")
        .reset_index()
    )
    support_df.to_excel(
        os.path.join(OUT_DIR, "PerClass_Support.xlsx"),
        index=False
    )

    # ---------------------
    # Bootstrap metrics (Excel)
    # ---------------------
    print("Running bootstrap evaluation...")
    summary_df, raw_boot = bootstrap_metrics(model, X_test, y_test)

    summary_df.to_excel(
        os.path.join(OUT_DIR, "Bootstrap_Metrics_95CI.xlsx"),
        index=False
    )
    raw_boot.to_excel(
        os.path.join(OUT_DIR, "Bootstrap_Metrics_AllRuns.xlsx"),
        index=False
    )

    print("\nBootstrap evaluation completed.")
    print(summary_df)
    print(f"\nOutputs saved to: {OUT_DIR}")
