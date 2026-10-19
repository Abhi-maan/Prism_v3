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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from echoes_pipeline import extract_uniform_fingerprint

print("=" * 80)
print("  PRISM PRODUCTION MULTI-DOMAIN SPECIALIST RETRAINING (REGULARIZED V5)")
print("  Integrating Royalty-Free Music Benchmark + Anti-Overfitting Regularization")
print("=" * 80)

# 1. Connect to database
db_path = ROOT / "syntehetic audio detection" / "echoes_fingerprints.db"
conn = sqlite3.connect(str(db_path))
c = conn.cursor()
c.execute("""
    SELECT f.file_path, f.label, fp.fingerprint_blob
    FROM files f
    JOIN fingerprints fp ON f.id=fp.file_id
""")
db_rows = c.fetchall()
conn.close()

# 2. Extract multi-window fingerprints for Dispatch soundtrack
dispatch_dir = Path(r"D:\Games\Dispatch\Dispatch - Official Soundtrack\MP3")
dispatch_files = sorted(list(dispatch_dir.glob("*.mp3")))
print(f"Extracting multi-window fingerprints for {len(dispatch_files)} Dispatch tracks...")
dispatch_fps = []
offsets = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0]

for f in dispatch_files:
    for off in offsets:
        try:
            res = extract_uniform_fingerprint(str(f), offset_sec=off)
            dispatch_fps.append(res["fingerprint"])
        except Exception:
            pass

print(f"Extracted {len(dispatch_fps)} Dispatch representations.")

# 3. Categorize database items strictly by ground-truth path
music_real = list(dispatch_fps) # Start with modern commercial music
music_ai = []

speech_real = []
speech_ai = []

env_real = []
env_ai = []

for p_str, label, blob in db_rows:
    p_lower = p_str.lower()
    vec = np.frombuffer(blob, dtype=np.float32)
    
    # Music routing
    if ("genres" in p_lower or "gtzan" in p_lower or "royaltyfree" in p_lower) and label == 0:
        music_real.append(vec)
    elif ("tta" in p_lower or "ata" in p_lower) and label == 1:
        music_ai.append(vec)
        
    # Speech routing
    elif ("librispeech" in p_lower or "ljspeech" in p_lower) and label == 0:
        speech_real.append(vec)
    elif "wavefake" in p_lower and label == 1:
        speech_ai.append(vec)
        
    # Environmental routing
    elif ("esc-50" in p_lower or "esc50" in p_lower) and label == 0:
        env_real.append(vec)
    elif "desed" in p_lower and label == 1:
        env_ai.append(vec)

# If env_ai is small, load clean env from echoes_clean_v3
db_v3 = ROOT / "syntehetic audio detection" / "echoes_clean_v3.db"
if db_v3.exists():
    conn3 = sqlite3.connect(str(db_v3))
    c3 = conn3.cursor()
    c3.execute("SELECT f.file_path, f.domain, f.label, fp.fingerprint_blob FROM files f JOIN fingerprints fp ON f.id=fp.file_id WHERE f.domain='environmental' AND f.label=1")
    for r in c3.fetchall():
        env_ai.append(np.frombuffer(r[3], dtype=np.float32))
    conn3.close()

print(f"\nExpanded & Balanced Dataset Summary:")
print(f"  MUSIC        : {len(music_real)} Real (Dispatch={len(dispatch_fps)}, GTZAN+RoyaltyFree={len(music_real)-len(dispatch_fps)}), {len(music_ai)} AI")
print(f"  SPEECH       : {len(speech_real)} Real, {len(speech_ai)} AI")
print(f"  ENVIRONMENTAL: {len(env_real)} Real, {len(env_ai)} AI")

def train_specialist(name, X_real, X_ai, calib_threshold=0.60):
    print(f"\n--- Training Domain Specialist: {name.upper()} ---")
    X = np.vstack([X_real, X_ai])
    y = np.array([0] * len(X_real) + [1] * len(X_ai))
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    pos_w = len(X_real) / max(len(X_ai), 1)
    
    # 1. Regularized XGBoost (L1 + L2 regularization, min_child_weight to prevent tiny leaves)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80, eval_metric="logloss",
        gamma=1.5, reg_alpha=0.2, reg_lambda=1.5, min_child_weight=4,
        scale_pos_weight=pos_w,
        n_jobs=-1, random_state=42
    )
    
    # 2. Regularized HistGradientBoosting (Robust continuous feature partitioner)
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, min_samples_leaf=20,
        l2_regularization=0.20, class_weight="balanced", random_state=42
    )
    
    # 3. Regularized RandomForest (Depth 9, min_samples_leaf 8 to kill the +18% memorization gap)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=9, min_samples_leaf=8, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    
    # Weighted voting prioritising top generalisers
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("hgb", hgb), ("rf", rf)],
        voting="soft", weights=[2.5, 3.5, 1.0], n_jobs=-1
    )
    
    t0 = time.time()
    ensemble.fit(X_tr_sc, y_tr)
    fit_time = time.time() - t0
    
    # Evaluate Train vs Test Generalization Gap
    tr_probs = ensemble.predict_proba(X_tr_sc)[:, 1]
    tr_preds = (tr_probs >= calib_threshold).astype(int)
    tr_acc = accuracy_score(y_tr, tr_preds) * 100
    
    te_probs = ensemble.predict_proba(X_te_sc)[:, 1]
    preds = (te_probs >= calib_threshold).astype(int)
    te_acc = accuracy_score(y_te, preds) * 100
    auc = roc_auc_score(y_te, te_probs) * 100
    gap = tr_acc - te_acc
    
    print(f"  Fit Time: {fit_time:.1f}s")
    print(f"  TRAIN Acc: {tr_acc:.2f}% | TEST Acc: {te_acc:.2f}% | Generalization Gap: {gap:+.2f}%")
    print(f"  ROC-AUC  : {auc:.2f}% (Threshold: {calib_threshold:.2f})")
    
    # Sub-model test
    rf_tr_acc = accuracy_score(y_tr, (ensemble.named_estimators_['rf'].predict_proba(X_tr_sc)[:, 1] >= calib_threshold).astype(int)) * 100
    rf_te_acc = accuracy_score(y_te, (ensemble.named_estimators_['rf'].predict_proba(X_te_sc)[:, 1] >= calib_threshold).astype(int)) * 100
    print(f"    -> Regularized RF: Train={rf_tr_acc:.2f}% | Test={rf_te_acc:.2f}% | Gap={rf_tr_acc-rf_te_acc:+.2f}%")
    
    return {
        "scaler": sc,
        "model": ensemble,
        "threshold": calib_threshold,
        "accuracy": te_acc,
        "auc": auc
    }

# Train specialists
music_bundle = train_specialist("music", music_real, music_ai, calib_threshold=0.65)
speech_bundle = train_specialist("speech", speech_real, speech_ai, calib_threshold=0.55)
env_bundle = train_specialist("environmental", env_real, env_ai, calib_threshold=0.50)

# Build general fallback model (balanced across all three domains)
print("\n--- Training Master General Ensemble ---")
all_real = np.vstack([music_real[:1000], speech_real[:1000], env_real[:1000]])
all_ai = np.vstack([music_ai[:1000], speech_ai[:1000], env_ai[:1000]])
gen_bundle = train_specialist("general", all_real, all_ai, calib_threshold=0.55)

production_bundle = {
    "scaler": gen_bundle["scaler"],
    "model": gen_bundle["model"],
    "domain_thresholds": {
        "music": 0.65,
        "speech": 0.55,
        "environmental": 0.50,
        "general": 0.55
    },
    "domain_bundles": {
        "music": music_bundle,
        "speech": speech_bundle,
        "environmental": env_bundle,
        "general": gen_bundle
    },
    "feature_dim": 240,
    "version": "prism_v5_regularized_royaltyfree",
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
}

out_path = ROOT / "syntehetic audio detection" / "models" / "prism_echoes_ensemble.pkl"
with open(out_path, "wb") as f:
    pickle.dump(production_bundle, f)

print(f"\nSuccessfully saved production bundle to {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.2f} MB)")

# Sync to Prism-main
prism_out = Path(r"D:\claude\Prism-main\Prism-main\syntehetic audio detection\models\prism_echoes_ensemble.pkl")
if prism_out.parent.exists():
    with open(prism_out, "wb") as pf:
        pickle.dump(production_bundle, pf)
    print(f"Synced model bundle to {prism_out}")

# Copy this script to Prism-main
prism_script = Path(r"D:\claude\Prism-main\Prism-main\syntehetic audio detection\train_clean_production_specialists.py")
with open(prism_script, "w", encoding="utf-8") as ps:
    with open(__file__, "r", encoding="utf-8") as src:
        ps.write(src.read())
print(f"Synced script to {prism_script}")
