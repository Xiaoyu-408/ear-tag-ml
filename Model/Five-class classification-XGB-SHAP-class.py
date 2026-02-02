import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

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
N_REPEAT    = 30
CONF_LEVEL  = 0.95

OUT_DIR = os.path.join(os.path.dirname(__file__), "SHAP_FINAL_NO_DECISION")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_NAMES = ["0", "1", "2", "3", "4"]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14

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
                df.iloc[s:s + WINDOW_SIZE], b_stats, stage_ref
            )
            if feats is None:
                continue

            feats["Pig_ID"] = pig_id
            feats["Label"] = label_type if df.iloc[s]["Day_Label"] != "day0" else 0
            rows.append(feats)

    return pd.DataFrame(rows)

# ======================================================
# 4. Train model and compute SHAP
# ======================================================
def train_and_get_shap(df, seed=42):
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
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss"
    )

    model.fit(X, y, sample_weight=sample_w)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    return shap_vals, X

# ======================================================
# 5. SHAP confidence intervals
# ======================================================
def compute_shap_ci(df):
    all_runs = []

    for i in range(N_REPEAT):
        shap_vals, X = train_and_get_shap(df, seed=100 + i)
        shap_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        all_runs.append(shap_abs.mean(axis=0))

    arr = np.vstack(all_runs)

    return pd.DataFrame({
        "Feature": X.columns,
        "Mean_abs_SHAP": arr.mean(axis=0),
        "CI_lower": np.percentile(arr, 2.5, axis=0),
        "CI_upper": np.percentile(arr, 97.5, axis=0),
    }).sort_values("Mean_abs_SHAP", ascending=False)

# ======================================================
# 6. Main workflow
# ======================================================
if __name__ == "__main__":

    print("Building features...")
    df_feat = build_feature_table()

    print("Computing SHAP confidence intervals...")
    ci_df = compute_shap_ci(df_feat)
    ci_df.to_excel(
        os.path.join(OUT_DIR, "SHAP_meanAbs_CI.xlsx"),
        index=False
    )

    feature_order = ci_df["Feature"].tolist()

    # ----------------------
    # Overall SHAP beeswarm
    # ----------------------
    shap_vals, X = train_and_get_shap(df_feat, seed=42)

    idx = [X.columns.get_loc(f) for f in feature_order]
    shap_all = np.mean(shap_vals, axis=0)[:, idx]
    X_all = X.iloc[:, idx]

    shap.summary_plot(shap_all, X_all, sort=False, show=False)
    plt.title("SHAP Beeswarm (Overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_SHAP_Beeswarm_All.png"), dpi=300)
    plt.show()

    # ----------------------
    # Per-class SHAP beeswarm
    # ----------------------
    for cid in range(len(shap_vals)):
        shap.summary_plot(shap_vals[cid], X, show=False)
        plt.title(f"SHAP Beeswarm (Class {cid})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"03_SHAP_Beeswarm_Class_{cid}.png"),
            dpi=300
        )
        plt.show()

    # ----------------------
    # Sankey data (Excel only)
    # ----------------------
    records = []
    for cid, sv in enumerate(shap_vals):
        mean_abs = np.mean(np.abs(sv), axis=0)
        for f, v in zip(X.columns, mean_abs):
            records.append({
                "Class": f"Class {cid}",
                "Feature": f,
                "MeanAbsSHAP": v
            })

    sankey_df = pd.DataFrame(records)
    sankey_df.to_excel(
        os.path.join(OUT_DIR, "04_SHAP_Sankey_Data.xlsx"),
        index=False
    )

    print("\nAll SHAP analyses completed.")
    print(f"Outputs saved to: {OUT_DIR}")
