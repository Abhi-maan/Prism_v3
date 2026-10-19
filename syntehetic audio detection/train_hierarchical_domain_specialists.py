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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier

print("=" * 85)
print("  PRISM HIERARCHICAL DOMAIN-SPECIALIST PRODUCTION MODEL TRAINING")
print("  (60% TRAIN / 40% HELD-OUT TEST ACROSS ALL THREE DOMAINS)")
print("=" * 85)

# Load clean labeled data from both databases
combined_rows = []
seen_fnames = set()

for db_file in ["echoes_clean_v3.db", "echoes_fingerprints.db"]:
    db_p = ROOT / "syntehetic audio detection" / db_file
    if db_p.exists():
        conn = sqlite3.connect(str(db_p))
        c = conn.cursor()
        c.execute("""
            SELECT f.file_path, f.domain, f.label, fp.fingerprint_blob
            FROM files f
            JOIN fingerprints fp ON f.id=fp.file_id
            WHERE f.label IN (0, 1) AND f.file_path NOT LIKE 'C:\\Users\\nishe%'
        """)
        for r in c.fetchall():
            fname = Path(r[0]).name.lower()
            if fname not in seen_fnames:
                seen_fnames.add(fname)
                combined_rows.append(r)
        conn.close()

df = pd.DataFrame({
    "file_path": [r[0] for r in combined_rows],
    "domain": [r[1] for r in combined_rows],
    "label": [r[2] for r in combined_rows],
    "fp": [np.frombuffer(r[3], dtype=np.float32) for r in combined_rows],
})
df["strat_key"] = df["domain"] + "_" + df["label"].astype(str)

print(f"Total dataset size: {len(df)} clips")
for dom, grp in df.groupby("domain"):
    print(f"  - {dom:<15}: {len(grp)} files ({sum(grp['label']==0)} Real, {sum(grp['label']==1)} AI)")

# Global Stratified 60% Train / 40% Held-Out Test Split
train_df, test_df = train_test_split(df, test_size=0.40, stratify=df["strat_key"], random_state=42)
print(f"\nTrain Split: {len(train_df)} samples (60.0%) | Test Split: {len(test_df)} samples (40.0% HELD-OUT)")

domain_bundles = {}
metrics_report = {}

def build_ensemble(pos_weight=1.0):
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
        scale_pos_weight=pos_weight,
        n_jobs=-1, random_state=42
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=350, learning_rate=0.03, min_samples_leaf=10,
        l2_regularization=0.05, class_weight="balanced", random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=350, max_depth=18, min_samples_split=2,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    return VotingClassifier(
        estimators=[("xgb", xgb), ("hgb", hgb), ("rf", rf)],
        voting="soft", weights=[2.5, 2.5, 1.0], n_jobs=-1
    )

for dom in ["environmental", "music", "speech"]:
    print(f"\n--- Training Domain Specialist: {dom.upper()} ---")
    tr_sub = train_df[train_df["domain"] == dom]
    te_sub = test_df[test_df["domain"] == dom]
    
    X_tr = np.array(tr_sub["fp"].tolist())
    y_tr = tr_sub["label"].values
    X_te = np.array(te_sub["fp"].tolist())
    y_te = te_sub["label"].values
    
    # Zero Scaler Leakage: Fit strictly on X_tr
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    sw = compute_sample_weight("balanced", y_tr)
    pos_w = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
    
    ens = build_ensemble(pos_weight=pos_w)
    t0 = time.time()
    ens.fit(X_tr_sc, y_tr, sample_weight=sw)
    print(f"  {dom.upper()} ensemble trained in {time.time()-t0:.2f}s!")
    
    probs_te = ens.predict_proba(X_te_sc)[:, 1]
    
    # Determine best threshold on training set via 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = np.zeros(len(y_tr))
    for tr_idx, val_idx in cv.split(X_tr_sc, y_tr):
        m_cv = build_ensemble(pos_weight=pos_w)
        m_cv.fit(X_tr_sc[tr_idx], y_tr[tr_idx], sample_weight=sw[tr_idx])
        cv_probs[val_idx] = m_cv.predict_proba(X_tr_sc[val_idx])[:, 1]
    
    best_th, best_score = 0.50, -100.0
    for th in np.linspace(0.46, 0.56, 21):
        preds_cv = (cv_probs >= th).astype(int)
        r_acc = np.mean(preds_cv[y_tr == 0] == 0)
        s_acc = np.mean(preds_cv[y_tr == 1] == 1)
        bal = (r_acc + s_acc) / 2.0
        score = bal - max(0.0, 0.90 - r_acc)*5.0 - max(0.0, 0.90 - s_acc)*5.0
        if score > best_score:
            best_score = score
            best_th = float(th)
    
    best_th = round(best_th, 3)
    preds_te = (probs_te >= best_th).astype(int)
    
    acc = accuracy_score(y_te, preds_te)
    auc = roc_auc_score(y_te, probs_te)
    prec = precision_score(y_te, preds_te)
    rec = recall_score(y_te, preds_te)
    f1 = f1_score(y_te, preds_te)
    
    print(f"  {dom.upper()} 40% Test (N={len(y_te)}):")
    print(f"    Accuracy  = {acc*100:.2f}%  (Threshold = {best_th:.3f})")
    print(f"    ROC AUC   = {auc*100:.2f}%")
    print(f"    Precision = {prec*100:.2f}%")
    print(f"    Recall    = {rec*100:.2f}%")
    print(f"    F1-Score  = {f1*100:.2f}%")
    
    domain_bundles[dom] = {
        "scaler": sc,
        "model": ens,
        "threshold": best_th,
    }
    metrics_report[dom] = {
        "samples": len(y_te), "acc": acc, "auc": auc, "prec": prec, "rec": rec, "f1": f1, "th": best_th
    }

# Global fallback model (trained across all domains)
print("\n--- Training Master Global Model ---")
X_tr_all = np.array(train_df["fp"].tolist())
y_tr_all = train_df["label"].values
X_te_all = np.array(test_df["fp"].tolist())
y_te_all = test_df["label"].values

sc_all = StandardScaler()
X_tr_all_sc = sc_all.fit_transform(X_tr_all)
X_te_all_sc = sc_all.transform(X_te_all)

sw_all = compute_sample_weight("balanced", y_tr_all)
pos_w_all = float(np.sum(y_tr_all == 0) / max(np.sum(y_tr_all == 1), 1))

ens_all = build_ensemble(pos_weight=pos_w_all)
ens_all.fit(X_tr_all_sc, y_tr_all, sample_weight=sw_all)

probs_te_all = ens_all.predict_proba(X_te_all_sc)[:, 1]
preds_te_all = (probs_te_all >= 0.50).astype(int)

acc_all = accuracy_score(y_te_all, preds_te_all)
auc_all = roc_auc_score(y_te_all, probs_te_all)
print(f"  GLOBAL 40% Test (N={len(y_te_all)}): Accuracy={acc_all*100:.2f}%, ROC AUC={auc_all*100:.2f}%")

# Save master production bundle
save_path = ROOT / "syntehetic audio detection" / "models" / "prism_echoes_ensemble.pkl"
master_bundle = {
    "version": "4.0_hierarchical_domain_specialists",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "scaler": sc_all,
    "model": ens_all,
    "domain_bundles": domain_bundles,
    "domain_thresholds": {dom: b["threshold"] for dom, b in domain_bundles.items()},
    "metrics": metrics_report,
}
with open(save_path, "wb") as f:
    pickle.dump(master_bundle, f)

print(f"\nMaster model bundle permanently saved to: {save_path}")

print("\n" + "=" * 90)
print("  FINAL ACCREDITED 40% HELD-OUT TEST METRICS BY DOMAIN")
print("=" * 90)
print(f"  {'DOMAIN':<18} | {'SAMPLES':<8} | {'ACCURACY':<10} | {'ROC AUC':<10} | {'PRECISION':<10} | {'RECALL':<9} | {'F1-SCORE':<9}")
print("  " + "-" * 88)
for dom in ["environmental", "music", "speech"]:
    m = metrics_report[dom]
    print(f"  {dom.upper():<18} | {m['samples']:<8} | {m['acc']*100:>8.2f}% | {m['auc']*100:>8.2f}% | {m['prec']*100:>8.2f}% | {m['rec']*100:>8.2f}% | {m['f1']*100:>8.2f}%")
print("=" * 90)
