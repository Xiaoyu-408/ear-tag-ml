import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
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

WINDOW_SIZE = 100
STEP_SIZE   = 150

LABEL_NAMES = ["Normal", "Abnormal"]

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ======================================================
# 2. Feature extraction (identical to multiclass setting)
# ======================================================
def extract_aligned_features(window_df, b_stats, stage_ref_mean):
    feats = {}
    rel = {}

    for col in ["pH", "K+", "Ca2+"]:
        v = window_df[col].values
        if len(v) < 3 or np.std(v) < 1e-12:
            return None

        mean_v = np.mean(v)
        rel_m = mean_v - b_stats[col]["mean"]
        rel[col] = rel_m

        feats[f"{col}_rel_m"]     = rel_m
        feats[f"{col}_trend_off"] = mean_v - stage_ref_mean[col]
        feats[f"{col}_cv"]        = np.std(v) / (mean_v + 1e-6)

    feats["pH_K_ratio"]    = rel["pH"]  / (abs(rel["K+"])  + 1e-6)
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
# 4. Train and test (LOAO: Pig3)
# ======================================================
def train_and_test_LOAO(df):
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

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )
    plt.title("Binary physiological classification (Normal vs Abnormal, LOAO: Pig3)")
    plt.tight_layout()
    plt.show()

# ======================================================
# 5. Main
# ======================================================
if __name__ == "__main__":
    print("Building feature table...")
    df_feat = build_feature_table()

    print("\nLabel distribution:")
    print(df_feat["Label"].value_counts(normalize=True))

    print("\nTraining and testing binary model (LOAO)")
    train_and_test_LOAO(df_feat)
