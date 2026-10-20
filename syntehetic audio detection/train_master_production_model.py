import sys
import os
from pathlib import Path
import time
import pickle
import numpy as np
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "syntehetic audio detection"))

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier

print("=" * 80)
print("  PRISM MASTER PRODUCTION MODEL TRAINING (60% TRAIN / 40% HELD-OUT TEST)")
print("=" * 80)

db_path = ROOT / "syntehetic audio detection" / "echoes_clean_v3.db"
conn = sqlite3.connect(str(db_path))
c = conn.cursor()
c.execute("SELECT f.file_path, f.domain, f.label, fp.fingerprint_blob FROM files f JOIN fingerprints fp ON f.id=fp.file_id")
rows = c.fetchall()

df = pd.DataFrame({
    "file_path": [r[0] for r in rows],
    "domain": [r[1] for r in rows],
    "label": [r[2] for r in rows],
    "fp": [np.frombuffer(r[3], dtype=np.float32) for r in rows],
})
df["strat_key"] = df["domain"] + "_" + df["label"].astype(str)

print(f"Total dataset size: {len(df)} clips")
for dom, grp in df.groupby("domain"):
    print(f"  - {dom:<15}: {len(grp)} files ({sum(grp['label']==0)} Real, {sum(grp['label']==1)} AI)")

# Stratified 60% Train / 40% Held-Out Test Split
train_df, test_df = train_test_split(df, test_size=0.40, stratify=df["strat_key"], random_state=42)
print(f"\nTrain Split: {len(train_df)} samples (60.0%) | Test Split: {len(test_df)} samples (40.0% HELD-OUT)")

# Zero Scaler Leakage: Fit scaler strictly on X_train
scaler = StandardScaler()
X_train = scaler.fit_transform(np.array(train_df["fp"].tolist()))
y_train = train_df["label"].values

X_test = scaler.transform(np.array(test_df["fp"].tolist()))
y_test = test_df["label"].values

# Balanced sample weights to eliminate class imbalance
sample_weights = compute_sample_weight("balanced", y_train)

# Master Multi-Tree Gradient Boosted Ensemble
print("\nTraining Master Multi-Tree Ensemble...")
t0 = time.time()

xgb = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
    n_jobs=-1, random_state=42
)

hgb = HistGradientBoostingClassifier(
    max_iter=350, learning_rate=0.03, min_samples_leaf=10,
    l2_regularization=0.05, class_weight="balanced", random_state=42
)

rf = RandomForestClassifier(
    n_estimators=350, max_depth=20, min_samples_split=2,
    class_weight="balanced", n_jobs=-1, random_state=42
)

model = VotingClassifier(
    estimators=[("xgb", xgb), ("hgb", hgb), ("rf", rf)],
    voting="soft", weights=[2.5, 2.5, 1.0], n_jobs=-1
)

model.fit(X_train, y_train, sample_weight=sample_weights)
print(f"Master Ensemble trained in {time.time()-t0:.2f}s!")

# Calibrate optimal decision thresholds per domain via StratifiedKFold on training set
print("\nCalibrating Domain-Specific Decision Thresholds via 5-Fold Cross Validation on Train Set...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_probs = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
train_df = train_df.copy()
train_df["cv_prob"] = cv_probs

domain_thresholds = {}
for dom in ["environmental", "music", "speech"]:
    sub = train_df[train_df["domain"] == dom]
    y_dom = sub["label"].values
    p_dom = sub["cv_prob"].values
    
    best_th, best_score = 0.50, -100.0
    for th in np.linspace(0.45, 0.60, 31):
        p_bin = (p_dom >= th).astype(int)
        r_acc = np.mean(p_bin[y_dom == 0] == 0)
        s_acc = np.mean(p_bin[y_dom == 1] == 1)
        bal_acc = (r_acc + s_acc) / 2.0
        # Balance precision and recall
        penalty = max(0.0, 0.90 - r_acc) * 5.0 + max(0.0, 0.90 - s_acc) * 5.0
        score = bal_acc - penalty
        if score > best_score:
            best_score = score
            best_th = float(th)
    domain_thresholds[dom] = round(best_th, 3)

print(f"Calibrated Domain Thresholds: {domain_thresholds}")

# Save master bundle
save_path = ROOT / "syntehetic audio detection" / "models" / "prism_echoes_ensemble.pkl"
bundle = {
    "scaler": scaler,
    "model": model,
    "seed": 42,
    "domain_thresholds": domain_thresholds,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "version": "3.0_production_60_40"
}
with open(save_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"Master model bundle permanently saved to: {save_path}")

# Evaluate on Held-Out 40% Test Set
print("\n" + "=" * 80)
print(f"  LIVE BENCHMARK RESULTS ON HELD-OUT 40% TEST SET ({len(test_df)} SAMPLES)")
print("=" * 80)

y_probs = model.predict_proba(X_test)[:, 1]
test_df = test_df.copy()
test_df["ai_prob"] = y_probs

y_preds = np.zeros(len(y_test), dtype=int)
for i, (_, row) in enumerate(test_df.iterrows()):
    dom = row["domain"]
    th = domain_thresholds.get(dom, 0.50)
    y_preds[i] = 1 if y_probs[i] >= th else 0

test_df["pred_label"] = y_preds

acc = accuracy_score(y_test, y_preds)
auc = roc_auc_score(y_test, y_probs)
prec = precision_score(y_test, y_preds)
rec = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)

print(f"  Overall Accuracy : {acc*100:.2f}%")
print(f"  Overall ROC AUC  : {auc*100:.2f}%")
print(f"  Overall Precision: {prec*100:.2f}%")
print(f"  Overall Recall   : {rec*100:.2f}%")
print(f"  Overall F1-Score : {f1*100:.2f}%")
print("-" * 80)
print(f"  {'DOMAIN':<18} | {'SAMPLES':<8} | {'ACCURACY':<10} | {'ROC AUC':<10} | {'PRECISION':<10} | {'RECALL':<9} | {'F1-SCORE':<9}")
print("  " + "-" * 88)

for dom in ["environmental", "music", "speech"]:
    sub = test_df[test_df["domain"] == dom]
    y_d = sub["label"].values
    p_d = sub["ai_prob"].values
    pred_d = sub["pred_label"].values
    d_acc = accuracy_score(y_d, pred_d)
    d_auc = roc_auc_score(y_d, p_d)
    d_prec = precision_score(y_d, pred_d)
    d_rec = recall_score(y_d, pred_d)
    d_f1 = f1_score(y_d, pred_d)
    print(f"  {dom.upper():<18} | {len(sub):<8} | {d_acc*100:>8.2f}% | {d_auc*100:>8.2f}% | {d_prec*100:>8.2f}% | {d_rec*100:>8.2f}% | {d_f1*100:>8.2f}%")

print("=" * 80)

# Export results
csv_path = ROOT / "syntehetic audio detection" / "results" / "benchmark_eval_40pct_test.csv"
test_df.drop(columns=["fp"]).to_csv(csv_path, index=False)
print(f"Exported evaluation details to {csv_path}")
