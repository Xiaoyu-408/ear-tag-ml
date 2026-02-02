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

WINDOW_SIZE = 100
STEP_SIZE   = 150

LABEL_NAMES = [
    "Normal",
    "Acid load",
    "Ketoacid load",
    "Alkali load",
    "Calcium depletion"
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "SHAP_Fingerprint_Data")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# 2. Feature extraction
# ======================================================
def extract_aligned_features(window_df, b_stats, stage_ref_mean):
    feats = {}
    rel_means = {}

    for col in ["pH", "K+", "Ca2+"]:
        conc = window_df[col].values
        if len(conc) < 3 or np.std(conc) < 1e-12:
            return None

        curr_mean = np.mean(conc)
        rel_m = curr_mean - b_stats[col]["mean"]
        rel_means[col] = rel_m

        feats[f"{col}_rel_m"]     = rel_m
        feats[f"{col}_trend_off"] = curr_mean - stage_ref_mean[col]
        feats[f"{col}_cv"]        = np.std(conc) / (curr_mean + 1e-6)

    feats["pH_K_ratio"]    = rel_means["pH"]   / (abs(rel_means["K+"])  + 1e-6)
    feats["Ca_vs_pH_sync"] = rel_means["Ca2+"] / (abs(rel_means["pH"]) + 1e-6)

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
        df.columns = ["Day_Label", "pH", "K+", "Ca2+"]
        df["Day_Label"] = df["Day_Label"].astype(str).str.lower().str.strip()

        day0 = df[df["Day_Label"] == "day0"]
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
            feats["Label"]  = label_type if df.iloc[s]["Day_Label"] != "day0" else 0
            rows.append(feats)

    return pd.DataFrame(rows)

# ======================================================
# 4. Train model and compute SHAP (multi-class)
# ======================================================
def train_and_compute_shap(df):
    train_df = df[df["Pig_ID"].isin(["Pig_1", "Pig_2"])]

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
        n_jobs=1,
        eval_metric="mlogloss"
    )

    model.fit(X, y, sample_weight=sample_w)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)  # list[n_class]

    return shap_vals, X.columns

# ======================================================
# 5. Compute per-class mean(|SHAP|)
# ======================================================
def compute_mean_abs_shap(shap_vals, feature_names):
    records = []

    for cid, cname in enumerate(LABEL_NAMES):
        mean_abs = np.abs(shap_vals[cid]).mean(axis=0)

        for f, v in zip(feature_names, mean_abs):
            records.append({
                "Class": cname,
                "Feature": f,
                "Mean_abs_SHAP": v
            })

    return pd.DataFrame(records)

# ======================================================
# 6. Main entry (Excel output only)
# ======================================================
if __name__ == "__main__":

    df_feat = build_feature_table()

    shap_vals, feature_names = train_and_compute_shap(df_feat)

    mean_abs_df = compute_mean_abs_shap(shap_vals, feature_names)

    out_path = os.path.join(
        OUT_DIR,
        "SHAP_Fingerprint_meanAbs.xlsx"
    )
    mean_abs_df.to_excel(out_path, index=False)

    print("Mean absolute SHAP values (fingerprint data) generated.")
    print(f"Output saved to: {out_path}")
    print("\nPreview:")
    print(mean_abs_df.head())
