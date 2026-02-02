import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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
OUT_DIR  = os.path.join(os.path.dirname(__file__), "Domain_Shift")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 100
STEP_SIZE   = 150

LABEL_NAMES = ["0", "1", "2", "3", "4"]

# ======================================================
# 2. Feature extraction
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
# 3. Build feature table (retain Day for filtering)
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
            feats["Label"]  = label_type if df.iloc[s]["Day_Label"] != "day0" else 0
            feats["Day"]    = df.iloc[s]["Day_Label"]
            rows.append(feats)

    return pd.DataFrame(rows)

# ======================================================
# 4. Train model (Pig1 + Pig2)
# ======================================================
def train_model(train_df):
    X = train_df.drop(columns=["Label", "Pig_ID", "Day"])
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
# 5. Metric evaluation
# ======================================================
def evaluate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred)
    }

# ======================================================
# 6. Main workflow (Excel only, no day0)
# ======================================================
if __name__ == "__main__":

    print("Building features...")
    df = build_feature_table()

    train_df = df[df["Pig_ID"].isin(["Pig_1", "Pig_2"])]
    test_df  = df[df["Pig_ID"] == "Pig_3"]

    print("Training model...")
    model, feat_cols = train_model(train_df)

    # -------- Exclude day0 from evaluation --------
    test_df = test_df[test_df["Day"] != "day0"]

    records = []

    domains = {
        "Induction (day7+14)": test_df[test_df["Day"].isin(["day7", "day14"])]
    }

    for name, subset in domains.items():
        if len(subset) == 0:
            continue

        y_pred = np.argmax(
            model.predict_proba(subset[feat_cols]),
            axis=1
        )
        metrics = evaluate_metrics(subset["Label"].values, y_pred)
        metrics["Domain"] = name
        records.append(metrics)

    perf_df = pd.DataFrame(records)

    perf_df.to_excel(
        os.path.join(OUT_DIR, "Domain_Shift_Performance.xlsx"),
        index=False
    )

    print("\nDomain shift results (day0 excluded):")
    print(perf_df)
    print(f"\nOutputs saved to: {OUT_DIR}")
