import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_class_weight

# ======================================================
# 0. Reproducibility
# ======================================================
np.random.seed(42)

# ======================================================
# 1. Paths & constants
# ======================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "Calibration")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 100
STEP_SIZE   = 150
N_BINS      = 10

CLASS_NAMES = [
    "Normal",
    "Acid load",
    "Ketoacid load",
    "Alkali load",
    "Calcium depletion"
]

# ======================================================
# 2. Feature extraction
# ======================================================
def extract_aligned_features(window_df, b_stats, stage_ref):
    feats = {}
    rel = {}

    for col in ["pH", "K+", "Ca2+"]:
        v = window_df[col].values
        if len(v) < 3 or np.std(v) < 1e-12:
            return None

        m = np.mean(v)
        rel[col] = m - b_stats[col]["mean"]

        feats[f"{col}_rel_m"]     = rel[col]
        feats[f"{col}_trend_off"] = m - stage_ref[col]
        feats[f"{col}_cv"]        = np.std(v) / (m + 1e-6)

    feats["pH_K_ratio"]    = rel["pH"]   / (abs(rel["K+"]) + 1e-6)
    feats["Ca_vs_pH_sync"] = rel["Ca2+"] / (abs(rel["pH"]) + 1e-6)

    return feats

# ======================================================
# 3. Build feature table from raw Excel files
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
        label_type = (i - 1) % 4 + 1  # 1–4

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
            feats["Day"]    = df.iloc[s]["Day"]

            rows.append(feats)

    return pd.DataFrame(rows)

# ======================================================
# 4. Train model (Pig1 + Pig2)
# ======================================================
def train_model(df):
    train_df = df[df["Pig_ID"].isin(["Pig_1", "Pig_2"])]

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
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X, y, sample_weight=sample_w)
    return model, X.columns

# ======================================================
# 5. Calibration + Brier (five curves per plot)
# ======================================================
def compute_calibration(df, model, feat_cols, domain_name):
    X = df[feat_cols]
    y_true = df["Label"].values
    prob = model.predict_proba(X)

    records = []

    plt.figure(figsize=(4, 4))

    for cid, cname in enumerate(CLASS_NAMES):
        y_bin = (y_true == cid).astype(int)
        p_c = prob[:, cid]

        frac_pos, mean_pred = calibration_curve(
            y_bin, p_c, n_bins=N_BINS, strategy="uniform"
        )

        brier = brier_score_loss(y_bin, p_c)

        # Plot five curves
        plt.plot(mean_pred, frac_pos, marker="o", label=cname)

        # Save bin-level data
        for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos), start=1):
            records.append({
                "Domain": domain_name,
                "Class": cname,
                "Bin": i,
                "Mean_Pred_Prob": mp,
                "Frac_Positive": fp,
                "Brier": brier
            })

    # Reference line
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Prediction probability")
    plt.ylabel("Observed proportion")
    plt.title(domain_name)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"Calibration_{domain_name}.png"),
        dpi=300
    )
    plt.close()

    return records

# ======================================================
# 6. Main
# ======================================================
if __name__ == "__main__":

    print("▶ Building feature table")
    df = build_feature_table()

    print("▶ Training model (Pig1 + Pig2)")
    model, feat_cols = train_model(df)

    test_df = df[df["Pig_ID"] == "Pig_3"]

    all_records = []

    domain_def = {
        "day7":  ["day7"],
        "day14": ["day14"],
        "all":   ["day0", "day7", "day14"]
    }

    for domain, days in domain_def.items():
        sub = test_df[test_df["Day"].isin(days)]
        if len(sub) == 0:
            continue

        print(f"▶ Calibration for {domain}")
        all_records += compute_calibration(
            sub,
            model,
            feat_cols,
            domain
        )

    out_df = pd.DataFrame(all_records)
    out_path = os.path.join(OUT_DIR, "Calibration_Brier_RawData.xlsx")
    out_df.to_excel(out_path, index=False)

    print("\n✅ Five-class calibration + Brier finished")
    print(f"📂 Output saved to: {out_path}")
