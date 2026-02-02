import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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

OUT_DIR = os.path.join(os.path.dirname(__file__), "FIGURES_BINARY_SHAP")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 100
STEP_SIZE   = 150

LABEL_NAMES = ["Normal", "Abnormal"]

plt.rcParams["font.family"] = "Arial"

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

    return feats, rel

# ======================================================
# 3. Build feature table (binary classification)
# ======================================================
def build_feature_table():
    rows = []

    PH_TH = 0.08
    K_TH  = 0.2
    CA_TH = 0.1

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

        for s in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
            out = extract_aligned_features(
                df.iloc[s:s + WINDOW_SIZE], b_stats, stage_ref
            )
            if out is None:
                continue

            feats, rel = out
            feats["Pig_ID"] = pig_id

            if (
                abs(rel["pH"])   < PH_TH and
                abs(rel["K+"])   < K_TH  and
                abs(rel["Ca2+"]) < CA_TH
            ):
                feats["Label"] = 0
            else:
                feats["Label"] = 1

            rows.append(feats)

    return pd.DataFrame(rows)

# ======================================================
# 4. Training, LOAO evaluation, and SHAP analysis
# ======================================================
def train_test_shap_LOAO(df):

    train_df = df[df["Pig_ID"].isin(["Pig_1", "Pig_2"])]
    test_df  = df[df["Pig_ID"] == "Pig_3"]

    X_train = train_df.drop(columns=["Label", "Pig_ID"])
    y_train = train_df["Label"].values
    X_test  = test_df.drop(columns=["Label", "Pig_ID"])
    y_test  = test_df["Label"].values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    w = dict(zip(np.unique(y_train), weights))
    sample_w = np.array([w[y] for y in y_train])

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train, sample_weight=sample_w)

    # ---------------------
    # Confusion matrix
    # ---------------------
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )
    plt.title("Binary classification (LOAO: Pig3)", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "Fig_CM_Binary_LOAO_Pig3.png"),
        dpi=600
    )
    plt.close()

    # ---------------------
    # SHAP analysis
    # ---------------------
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)

    plt.figure(figsize=(9, 14))
    shap.summary_plot(shap_vals, X_test, show=False)

    ax = plt.gca()
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=16)

    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=16)

    ax.set_xlabel("SHAP value", fontsize=18)
    ax.set_ylabel("Feature", fontsize=18)
    plt.title("SHAP beeswarm (Binary classification)", fontsize=20)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "Fig_SHAP_Beeswarm_Binary.png"),
        dpi=600
    )
    plt.close()

    # ---------------------
    # Mean |SHAP| with 95% CI
    # ---------------------
    mean_abs = np.abs(shap_vals).mean(axis=0)

    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.choice(len(shap_vals), len(shap_vals), replace=True)
        boot.append(np.abs(shap_vals[idx]).mean(axis=0))
    boot = np.vstack(boot)

    ci_low  = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)

    shap_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_abs_SHAP": mean_abs,
        "CI_95_low": ci_low,
        "CI_95_high": ci_high
    }).sort_values("Mean_abs_SHAP", ascending=False)

    shap_df.to_excel(
        os.path.join(OUT_DIR, "SHAP_Overall_Feature_Contribution_Binary.xlsx"),
        index=False
    )

# ======================================================
# 5. Main entry
# ======================================================
if __name__ == "__main__":

    print("Building feature table...")
    df_feat = build_feature_table()

    print("Training, evaluation, and SHAP analysis...")
    train_test_shap_LOAO(df_feat)
