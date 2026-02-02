import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from sklearn.utils.class_weight import compute_class_weight

# ======================================================
# 0. Reproducibility
# ======================================================
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

# ======================================================
# 1. Global configuration
# ======================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "SHAP_CI_PER_CLASS_ONLY")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE  = 100
STEP_SIZE    = 150
N_BOOTSTRAP  = 500

FEATURE_COLS = [
    "pH_rel_m", "pH_trend_off", "pH_cv",
    "K+_rel_m", "K+_trend_off", "K+_cv",
    "Ca2+_rel_m", "Ca2+_trend_off", "Ca2+_cv",
    "pH_K_ratio", "Ca_vs_pH_sync"
]

# ======================================================
# 2. Feature extraction (11 engineered features)
# ======================================================
def extract_aligned_features(window_df, b_stats, stage_ref):
    feats = {}
    rel = {}

    for col in ["pH", "K+", "Ca2+"]:
        v = window_df[col].values
        if len(v) < 3 or np.std(v) < 1e-12:
            return None

        mean_v = np.mean(v)
        rel[col] = mean_v - b_stats[col]["mean"]

        feats[f"{col}_rel_m"]     = rel[col]
        feats[f"{col}_trend_off"] = mean_v - stage_ref[col]
        feats[f"{col}_cv"]        = np.std(v) / (mean_v + 1e-6)

    feats["pH_K_ratio"]    = rel["pH"]   / (abs(rel["K+"])  + 1e-6)
    feats["Ca_vs_pH_sync"] = rel["Ca2+"] / (abs(rel["pH"]) + 1e-6)

    return feats

# ======================================================
# 3. Build feature table
# ======================================================
def build_feature_table():
    rows = []

    for i in range(1, 13):
        path = os.path.join(BASE_DIR, f"{i}.xlsx")
        if not os.path.exists(path):
            continue

        df = pd.read_excel(path).iloc[:, [0, 2, 3, 4]]
        df.columns = ["Day", "pH", "K+", "Ca2+"]
        df["Day"] = df["Day"].astype(str).str.lower().str.strip()

        day0 = df[df["Day"] == "day0"]
        b_stats = {c: {"mean": day0[c].mean()} for c in ["pH", "K+", "Ca2+"]}
        stage_ref = df.iloc[:500][["pH", "K+", "Ca2+"]].mean().to_dict()

        pig_id = "Pig_1" if i <= 4 else ("Pig_2" if i <= 8 else "Pig_3")
        label_type = (i - 1) % 4 + 1

        for s in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
            feats = extract_aligned_features(
                df.iloc[s:s + WINDOW_SIZE],
                b_stats,
                stage_ref
            )
            if feats is None:
                continue

            feats["Pig_ID"] = pig_id
            feats["Label"]  = label_type if df.iloc[s]["Day"] != "day0" else 0
            rows.append(feats)

    return pd.DataFrame(rows)

# ======================================================
# 4. Train model and compute SHAP (single run)
# ======================================================
def train_and_get_shap(df):
    train_df = df[df["Pig_ID"].isin(["Pig_1", "Pig_2"])]

    X = train_df[FEATURE_COLS]
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

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)   # list[n_class]

    return shap_vals, FEATURE_COLS

# ======================================================
# 5. Bootstrap CI for a single class
# ======================================================
def bootstrap_ci(shap_matrix, feature_names):
    boot = []

    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(
            shap_matrix.shape[0],
            shap_matrix.shape[0],
            replace=True
        )
        boot.append(np.abs(shap_matrix[idx]).mean(axis=0))

    arr = np.vstack(boot)

    return (
        pd.DataFrame({
            "Feature": feature_names,
            "Mean_abs_SHAP": arr.mean(axis=0),
            "CI_lower": np.percentile(arr, 2.5, axis=0),
            "CI_upper": np.percentile(arr, 97.5, axis=0),
        })
        .sort_values("Mean_abs_SHAP", ascending=False)
    )

# ======================================================
# 6. Main workflow (per-class Excel output only)
# ======================================================
if __name__ == "__main__":

    print("Building feature table...")
    df_feat = build_feature_table()

    print("Training model and computing SHAP values...")
    shap_vals_all, feature_names = train_and_get_shap(df_feat)

    for cid, sv in enumerate(shap_vals_all):
        print(f"Computing bootstrap CI for class {cid}...")
        ci_df = bootstrap_ci(sv, feature_names)

        ci_df.to_excel(
            os.path.join(OUT_DIR, f"SHAP_CI_Class_{cid}.xlsx"),
            index=False
        )

    print("\nPer-class SHAP CI tables generated successfully.")
    print(f"Outputs saved to: {OUT_DIR}")
