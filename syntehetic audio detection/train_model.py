"""
PRISM — Dedicated AI Audio Model Training Pipeline
===================================================
Run this command ONLY when explicitly asked to retrain or update the master model.
This tool performs:
  1. Stratified sampling across Environmental, Music, and Speech audio.
  2. Feature synchronization into permanent SQLite cache (`echoes_fingerprints.db`).
  3. Scaler fitting and multi-tree ensemble training (HistGB + RF + ExtraTrees + XGBoost).
  4. Out-of-fold cross-validation domain threshold calibration (Music, Environmental, Speech).
  5. Permanent bundle persistence to `models/prism_echoes_ensemble.pkl`.

Usage Examples:
  # Retrain the master model on 1800 balanced clips:
  python train_model.py --limit 1800

  # Retrain on custom dataset directory:
  python train_model.py --dataset-dir path/to/dataset --limit 3000
"""

from __future__ import annotations

import os
import sys
import time
import math
import pickle
import argparse
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from echoes_pipeline import DEFAULT_DB_NAME
from train_and_eval_40pct import (
    collect_balanced_dataset, extract_and_sync_features,
    build_master_ensemble, save_model_bundle,
    DEFAULT_ECHOES_DIR, DEFAULT_MODEL_PATH, SEED
)


def train_and_save_master_model(
    dataset_dir: Path = DEFAULT_ECHOES_DIR,
    limit: int = 1800,
    test_size: float = 0.20,
    save_model_path: Path = DEFAULT_MODEL_PATH,
    db_path: Path = ROOT / DEFAULT_DB_NAME
) -> dict:
    """Execute complete master model training and threshold calibration."""
    tracemalloc.start()
    t_start = time.perf_counter()

    print("\n" + "=" * 80)
    print("  PRISM — DEDICATED MASTER MODEL TRAINING PIPELINE")
    print("=" * 80)

    per_domain = math.ceil(limit / 3.0)
    print(f"\n[1/5] Sampling {limit} balanced files ({per_domain}/domain) from {dataset_dir}...")
    
    from echoes_pipeline import FingerprintDB
    db = FingerprintDB(db_path)
    cached_paths = set(db.get_cached_map().keys())
    items = collect_balanced_dataset(dataset_dir, per_domain_target=per_domain, cached_paths=cached_paths, seed=SEED)

    print(f"      Selected {len(items)} audio files across 3 domains.")

    # 2. Extract and sync features into SQLite cache
    print(f"\n[2/5] Synchronizing features with SQLite database ({db_path.name})...")
    num_cpus = min(os.cpu_count() or 4, 8)
    enriched = extract_and_sync_features(items, db_path=db_path, num_workers=num_cpus)

    # 3. Train/Test Split
    df = pd.DataFrame(enriched)
    df["strat_key"] = df["domain"] + "_" + df["label"].astype(str)

    print(f"\n[3/5] Performing Stratified { (1-test_size)*100:.0f}% Train / {test_size*100:.0f}% Validation Split...")
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df["strat_key"], random_state=SEED
    )

    X_train = np.array(train_df["fingerprint"].tolist())
    y_train = train_df["label"].values
    X_val = np.array(val_df["fingerprint"].tolist())
    y_val = val_df["label"].values

    print(f"      Training set size   : {len(train_df)} files")
    print(f"      Validation set size : {len(val_df)} files")

    # 4. Train Model Ensemble
    print(f"\n[4/5] Training Master Multi-Tree Ensemble (HistGB + RF + ExtraTrees" + (" + XGBoost)" if HAS_XGB else ")") + "...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_master_ensemble(seed=SEED)

    # Calibrate optimal decision threshold per domain via out-of-fold CV
    print("      Calibrating domain decision thresholds via 5-fold cross-validation...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_probs = cross_val_predict(model, X_train_scaled, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    train_df = train_df.copy()
    train_df["cv_prob"] = cv_probs

    domain_thresholds = {}
    for dom in ["environmental", "music", "speech"]:
        sub = train_df[train_df["domain"] == dom]
        if len(sub) > 0:
            best_th, best_score = 0.50, 0.0
            y_dom = sub["label"].values
            p_dom = sub["cv_prob"].values
            r_mask = (y_dom == 0)
            a_mask = (y_dom == 1)
            valid_candidates = []
            for th in np.linspace(0.20, 0.85, 66):
                p_bin = (p_dom >= th).astype(int)
                score = accuracy_score(y_dom, p_bin)
                r_acc = accuracy_score(y_dom[r_mask], p_bin[r_mask]) if np.sum(r_mask) > 0 else 0
                a_acc = accuracy_score(y_dom[a_mask], p_bin[a_mask]) if np.sum(a_mask) > 0 else 0
                if r_acc >= 0.92 and a_acc >= 0.92:
                    valid_candidates.append((th, score, r_acc, a_acc))
                if score > best_score:
                    best_score = score
                    best_th = float(th)
            if valid_candidates:
                # Pick threshold that maximizes overall accuracy while satisfying >=92% on both
                best_valid = max(valid_candidates, key=lambda x: x[1])
                domain_thresholds[dom] = round(float(best_valid[0]), 3)
            else:
                domain_thresholds[dom] = round(best_th, 3)
        else:
            domain_thresholds[dom] = 0.50

    print(f"      Optimal Domain Thresholds Calibrated: {domain_thresholds}", flush=True)

    t_fit_0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    t_fit_time = time.perf_counter() - t_fit_0
    print(f"      Model ensemble fitted successfully in {t_fit_time:.2f}s!")

    # 5. Validation Evaluation
    print(f"\n[5/5] Evaluating on held-out validation set ({len(X_val)} files)...")
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_preds = np.zeros(len(y_val), dtype=int)
    for i, (_, row) in enumerate(val_df.iterrows()):
        th = domain_thresholds.get(row["domain"], 0.50)
        val_preds[i] = 1 if val_probs[i] >= th else 0

    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)
    print(f"      Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"      Validation F1-Score : {val_f1:.4f}")

    for dom, grp in val_df.groupby("domain"):
        grp_preds = val_preds[grp.index.get_indexer_for(val_df.index) >= 0] # indices in val
        d_idx = [i for i, (_, r) in enumerate(val_df.iterrows()) if r["domain"] == dom]
        d_true = y_val[d_idx]
        d_pred = val_preds[d_idx]
        acc_d = accuracy_score(d_true, d_pred)
        print(f"        Domain {dom:<15}: {acc_d*100:>6.2f}% accuracy (n={len(d_idx)})")

    # Persist bundle
    save_model_bundle(scaler, model, save_model_path, domain_thresholds=domain_thresholds, seed=SEED)

    t_total = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / (1024 * 1024)

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE & SAVED TO DISK")
    print("=" * 80)
    print(f"  Saved Bundle Path   : {save_model_path.resolve()}")
    print(f"  Total Duration      : {t_total:.2f} seconds")
    print(f"  Peak RAM Memory     : {peak_ram_mb:.2f} MB")
    print("=" * 80 + "\n")

    return {"accuracy": val_acc, "domain_thresholds": domain_thresholds}


def main():
    parser = argparse.ArgumentParser(description="PRISM Dedicated Training Pipeline")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_ECHOES_DIR),
                        help="Path to audio dataset directory")
    parser.add_argument("--limit", type=int, default=1800,
                        help="Total number of audio files to train on (default: 1800)")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Fraction of dataset held out for validation (default: 0.20)")
    parser.add_argument("--save-model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path where model bundle will be saved")
    parser.add_argument("--db-path", type=str, default=str(ROOT / DEFAULT_DB_NAME),
                        help="Path to SQLite fingerprint database")

    args = parser.parse_args()

    train_and_save_master_model(
        dataset_dir=Path(args.dataset_dir),
        limit=args.limit,
        test_size=args.test_size,
        save_model_path=Path(args.save_model),
        db_path=Path(args.db_path)
    )


if __name__ == "__main__":
    main()
