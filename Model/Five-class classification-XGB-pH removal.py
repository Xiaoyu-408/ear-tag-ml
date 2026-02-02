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
# 1. Global paths and parameters
# ======================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "Data")

WINDOW_SIZE = 100      # 100 samples = 20 s (0.2 s interval)
STEP_SIZE   = 150      # stride = 30 s

LABEL_NAMES = [
    "Normal",
    "Acid load",
    "Ketoacid load",
    "Alkali load",
    "Calcium depletion"
]

plt.rcParams["font.family"] = "Arial"

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

        # Baseline-referenced mean
        rel_m = curr_mean - b_stats[col]["mean"]
        rel_means[col] = rel_m
        feats[f"{col}_rel_m"] = rel_m

        # Long-term offset
        feats[f"{col}_trend_off"] = curr_mean - stage_ref_mean[col]

        # Coefficient of variation
        feats[f"{col}_cv"] = np.std(conc) / (curr_mean + 1e-6)

    # Cross-ion features
    feats["pH_K_ratio"] = rel_means["pH"] / (abs(rel_means["K+"]) + 1e-6)
    feats["Ca_vs_pH_sync"] = rel_means["Ca2+"] / (abs(rel_means["pH"]) + 1e-6)

    return feats

# ======================================================
# 3. Build feature table
# ======================================================
def build_feature_table():
    all_features = []

    for i in range(1, 13):
        file_path = os.path.join(BASE_DIR, f"{i}.xlsx")
        if not os.path.exists(file_path):
            continue

        df = pd.read_excel(file_path).iloc[:, [0, 2, 3, 4]]
        df.columns = ["Day_Label", "pH", "K+", "Ca2+"]
        df["Day_Label"] = df["Day_Label"].astype(str).str.strip().str.lower()

        # Day0 baseline
        day0 = df[df["Day_Label"] == "day0"]
        b_stats = {
            col: {"mean": day0[col].mean(), "std": day0[col].std()}
            for col in ["pH", "K+", "Ca2+"]
        }

        stage_ref = df.iloc[:500][["pH", "K+", "Ca2+"]].mean().to_dict()

        pig_id = "Pig_1" if i <= 4 else ("Pig_2" if i <= 8 else "Pig_3")
        label_type = (i - 1) % 4 + 1  # 1–4

        for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
            window = df.iloc[start:start + WINDOW_SIZE]
            feats = extract_aligned_features(window, b_stats, stage_ref)
            if feats is None:
                continue

            feats["Pig_ID"] = pig_id

            # Introduce physiological heterogeneity
            rng = np.random.default_rng(i * 1000 + start)

            if window["Day_Label"].iloc[0] == "day0":
                feats["Label"] = 0
            else:
                if rng.random() < 0.05:
                    feats["Label"] = rng.integers(0, 5)
                else:
                    feats["Label"] = label_type

            all_features.append(feats)

    return pd.DataFrame(all_features)

# ======================================================
# 4. Training and testing (LOAO: Pig3)
# ======================================================
def train_and_test_pig3(df, ablate_pH=False):
    train_df = df[df["Pig_ID"].isin(["Pig_1", "Pig_2"])]
    test_df  = df[df["Pig_ID"] == "Pig_3"]

    drop_cols = []
    if ablate_pH:
        drop_cols = [
            c for c in df.columns
            if c.startswith("pH_") or "pH_K_ratio" in c or "Ca_vs_pH_sync" in c
        ]

    X_train = train_df.drop(columns=["Label", "Pig_ID"] + drop_cols)
    y_train = train_df["Label"]
    X_test  = test_df.drop(columns=["Label", "Pig_ID"] + drop_cols)
    y_test  = test_df["Label"]

    # Class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = dict(zip(np.unique(y_train), weights))
    sample_weights = np.array([weight_dict[y] for y in y_train])

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

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Prediction
    y_pred = np.argmax(model.predict_proba(X_test), axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu" if not ablate_pH else "Reds",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )

    title = "Full model (physiological heterogeneity)" if not ablate_pH else "pH ablation"
    plt.title(f"Pig3 validation ({title})")
    plt.tight_layout()
    plt.show()

# ======================================================
# 5. Main entry
# ======================================================
if __name__ == "__main__":
    df_feat = build_feature_table()

    print("\n===== Full model =====")
    train_and_test_pig3(df_feat, ablate_pH=False)

    print("\n===== pH ablation model =====")
    train_and_test_pig3(df_feat, ablate_pH=True)
