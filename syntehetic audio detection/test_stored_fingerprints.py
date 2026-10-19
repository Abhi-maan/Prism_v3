"""
PRISM -- Stored Audio Fingerprints Benchmark & AI Detection Tester
===================================================================
Tests pre-extracted, permanently cached audio fingerprints directly
from the SQLite database (echoes_fingerprints.db) with ZERO audio
feature re-extraction overhead.

Usage:
    # Run test on 3,000 stored fingerprinted files (default balanced):
    python test_stored_fingerprints.py --limit 3000

    # Run test with stratified sampling:
    python test_stored_fingerprints.py --limit 3000 --sampling stratified

    # Save detailed per-file predictions to custom CSV:
    python test_stored_fingerprints.py --limit 3000 --output-csv results/stored_fingerprints_test_3000.csv

    # Evaluate all 5,139 labeled fingerprints in the database:
    python test_stored_fingerprints.py --limit 0
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import random
import sqlite3
import argparse
import tracemalloc
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()
DEFAULT_DB_PATH = ROOT / "echoes_fingerprints.db"
DEFAULT_MODEL_PATH = ROOT / "models" / "prism_echoes_ensemble.pkl"
DEFAULT_OUTPUT_CSV = ROOT / "results" / "stored_fingerprints_test_3000.csv"


def query_stored_fingerprints(
    db_path: Path,
    limit: int = 3000,
    sampling_mode: str = "balanced",
    seed: int = 42
) -> Tuple[List[dict], float, int]:
    """
    Query already fingerprinted files and their 240-dim feature vectors
    directly from SQLite database cache.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path.resolve()}")

    t0 = time.perf_counter()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Query files that have ground-truth labels (0 = real, 1 = ai)
    query = """
        SELECT f.id, f.file_path, f.domain, f.label,
               fp.fingerprint_blob, fp.vocoder_score, fp.gan_score,
               fp.diffusion_score, fp.anomaly_score, fp.artifact_score
        FROM files f
        JOIN fingerprints fp ON f.id = fp.file_id
        WHERE f.label IN (0, 1)
    """
    cur.execute(query)
    rows = cur.fetchall()
    t_query = (time.perf_counter() - t0) * 1000.0  # ms
    total_in_db = len(rows)
    conn.close()

    if total_in_db == 0:
        raise ValueError("No labeled fingerprints found in database (labels 0 or 1).")

    rng = random.Random(seed)

    # Determine subset to test
    if limit <= 0 or limit >= total_in_db:
        selected_rows = list(rows)
        rng.shuffle(selected_rows)
    elif sampling_mode == "balanced":
        real_rows = [r for r in rows if r["label"] == 0]
        synth_rows = [r for r in rows if r["label"] == 1]
        half = limit // 2
        
        n_real = min(half, len(real_rows))
        n_synth = min(limit - n_real, len(synth_rows))
        if n_synth < (limit - n_real):
            n_real = min(limit - n_synth, len(real_rows))
            
        rng.shuffle(real_rows)
        rng.shuffle(synth_rows)
        selected_rows = real_rows[:n_real] + synth_rows[:n_synth]
        rng.shuffle(selected_rows)
    elif sampling_mode == "stratified":
        domain_groups: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            domain_groups.setdefault(r["domain"], []).append(r)
        selected_rows = []
        for d, grp in domain_groups.items():
            k = int(round(limit * len(grp) / total_in_db))
            rng.shuffle(grp)
            selected_rows.extend(grp[:k])
        if len(selected_rows) < limit:
            remaining = [r for r in rows if r not in set(selected_rows)]
            rng.shuffle(remaining)
            selected_rows.extend(remaining[:limit - len(selected_rows)])
        elif len(selected_rows) > limit:
            selected_rows = selected_rows[:limit]
        rng.shuffle(selected_rows)
    else:  # random
        rows_list = list(rows)
        selected_rows = rng.sample(rows_list, min(limit, len(rows_list)))

    # Parse records
    parsed_records = []
    for r in selected_rows:
        fp_arr = np.frombuffer(r["fingerprint_blob"], dtype=np.float32)
        parsed_records.append({
            "id": r["id"],
            "file_path": r["file_path"],
            "domain": r["domain"],
            "label": int(r["label"]),
            "fingerprint": fp_arr,
            "vocoder_score": float(r["vocoder_score"]),
            "gan_score": float(r["gan_score"]),
            "diffusion_score": float(r["diffusion_score"]),
            "anomaly_score": float(r["anomaly_score"]),
            "artifact_score": float(r["artifact_score"]),
        })

    return parsed_records, t_query, total_in_db


def load_model_ensemble(model_path: Path) -> Tuple[dict, float]:
    """Load pre-trained model ensemble and domain decision thresholds."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")
    t0 = time.perf_counter()
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    t_load = (time.perf_counter() - t0) * 1000.0  # ms
    return bundle, t_load


def evaluate_fingerprints(
    records: List[dict],
    bundle: dict
) -> Tuple[Dict[str, float], pd.DataFrame, float, float]:
    """Run model inference on pre-extracted feature vectors and calculate metrics."""
    scaler = bundle["scaler"]
    model = bundle["model"]
    domain_thresholds = bundle.get("domain_thresholds", {
        "environmental": 0.62,
        "music": 0.35,
        "speech": 0.65
    })

    # Prepare feature matrix
    X = np.array([r["fingerprint"] for r in records], dtype=np.float32)
    y_true = np.array([r["label"] for r in records], dtype=int)
    domains = np.array([r["domain"] for r in records])

    # Timing: Scaling and Model Inference
    t0 = time.perf_counter()
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    # Domain-adaptive thresholding
    preds = np.zeros(len(records), dtype=int)
    applied_thresholds = np.zeros(len(records), dtype=float)
    for i, (p, d) in enumerate(zip(probs, domains)):
        th = domain_thresholds.get(d, 0.50)
        applied_thresholds[i] = th
        preds[i] = 1 if p >= th else 0

    t_inference = (time.perf_counter() - t0) * 1000.0  # ms
    ms_per_file = t_inference / len(records) if len(records) > 0 else 0.0

    # Calculate Global Metrics
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Per-domain metrics
    domain_stats = {}
    for dom in np.unique(domains):
        mask = (domains == dom)
        y_dom = y_true[mask]
        p_dom = preds[mask]
        dom_acc = accuracy_score(y_dom, p_dom) if len(y_dom) > 0 else 0.0
        dom_prec = precision_score(y_dom, p_dom, zero_division=0) if len(y_dom) > 0 else 0.0
        dom_rec = recall_score(y_dom, p_dom, zero_division=0) if len(y_dom) > 0 else 0.0
        dom_f1 = f1_score(y_dom, p_dom, zero_division=0) if len(y_dom) > 0 else 0.0
        dom_cm = confusion_matrix(y_dom, p_dom, labels=[0, 1])
        d_tn, d_fp, d_fn, d_tp = dom_cm.ravel()
        d_fpr = d_fp / (d_fp + d_tn) if (d_fp + d_tn) > 0 else 0.0
        domain_stats[dom] = {
            "count": int(np.sum(mask)),
            "real": int(np.sum(y_dom == 0)),
            "synthetic": int(np.sum(y_dom == 1)),
            "threshold": domain_thresholds.get(dom, 0.50),
            "accuracy": dom_acc,
            "precision": dom_prec,
            "recall": dom_rec,
            "f1": dom_f1,
            "fpr": d_fpr,
            "tn": int(d_tn),
            "fp": int(d_fp),
            "fn": int(d_fn),
            "tp": int(d_tp),
        }

    metrics = {
        "total": len(records),
        "real_count": int(np.sum(y_true == 0)),
        "synth_count": int(np.sum(y_true == 1)),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mean_ai_prob_synth": float(np.mean(probs[y_true == 1])) if np.sum(y_true == 1) > 0 else 0.0,
        "mean_ai_prob_real": float(np.mean(probs[y_true == 0])) if np.sum(y_true == 0) > 0 else 0.0,
        "domain_stats": domain_stats,
    }

    # Build detailed DataFrame
    df = pd.DataFrame({
        "file_id": [r["id"] for r in records],
        "file_path": [r["file_path"] for r in records],
        "domain": domains,
        "ground_truth": ["Synthetic" if y == 1 else "Real" for y in y_true],
        "label": y_true,
        "ai_probability": probs,
        "threshold_applied": applied_thresholds,
        "prediction": ["Synthetic" if p == 1 else "Real" for p in preds],
        "pred_label": preds,
        "correct": (preds == y_true),
        "vocoder_score": [r["vocoder_score"] for r in records],
        "gan_score": [r["gan_score"] for r in records],
        "diffusion_score": [r["diffusion_score"] for r in records],
        "anomaly_score": [r["anomaly_score"] for r in records],
        "artifact_score": [r["artifact_score"] for r in records],
    })

    return metrics, df, t_inference, ms_per_file


def run_benchmark(
    limit: int = 3000,
    sampling: str = "balanced",
    db_path: Path = DEFAULT_DB_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_csv: Optional[Path] = DEFAULT_OUTPUT_CSV,
    seed: int = 42
) -> None:
    """Orchestrates loading, inference, and pretty printing in terminal format."""
    tracemalloc.start()
    t_wall_start = time.perf_counter()

    print("\n" + "=" * 80)
    print("  PRISM -- STORED AUDIO FINGERPRINT TEST BENCHMARK")
    print("=" * 80)

    # 1. Query Database
    print(f"\n[1/3] Querying SQLite Fingerprint Database ({db_path.name})...")
    records, t_query, total_db_count = query_stored_fingerprints(
        db_path=db_path, limit=limit, sampling_mode=sampling, seed=seed
    )
    q_ms_per_file = t_query / len(records) if len(records) > 0 else 0.0
    print(f"      Total Labeled Fingerprints in DB : {total_db_count:,}")
    print(f"      Files Selected for Evaluation    : {len(records):,}")
    print(f"      Sampling Mode                    : {sampling.capitalize()}")
    print(f"      SQLite DB Query Time             : {t_query:.2f} ms ({q_ms_per_file:.3f} ms / file)")
    print(f"      Query Throughput                 : {len(records) / (t_query / 1000.0):,.1f} files/sec")

    # 2. Load Model Ensemble
    print(f"\n[2/3] Loading Pre-Trained Model Ensemble ({model_path.name})...")
    bundle, t_load = load_model_ensemble(model_path)
    model_type = type(bundle["model"]).__name__
    domain_thresholds = bundle.get("domain_thresholds", {})
    print(f"      Model Ensemble Type              : {model_type}")
    print(f"      Model Loading Time               : {t_load:.2f} ms ({t_load/1000:.2f} s)")
    print(f"      Domain Decision Thresholds       : {domain_thresholds}")

    # 3. Model Inference & Evaluation
    print(f"\n[3/3] Running Inference on {len(records):,} Stored 240-dim Fingerprint Vectors...")
    metrics, df, t_inf, ms_per_file = evaluate_fingerprints(records, bundle)
    inf_throughput = len(records) / (t_inf / 1000.0) if t_inf > 0 else 0.0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_total_wall = (time.perf_counter() - t_wall_start) * 1000.0

    # Optional CSV Export
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n[EXPORT] Detailed results saved to: {output_csv}")

    # Terminal Report Display
    print("\n" + "=" * 80)
    print(f"                     BENCHMARK RESULTS SUMMARY ({metrics['total']:,} FILES)")
    print("=" * 80)

    print("\n  [LATENCY & TIMING BREAKDOWN]")
    print(f"  +--------------------------------+--------------------+----------------------+")
    print(f"  | Pipeline Stage                 | Duration           | Per-File Latency     |")
    print(f"  +--------------------------------+--------------------+----------------------+")
    print(f"  | SQLite DB Retrieval (240-dim)  | {t_query:>10.2f} ms   | {q_ms_per_file:>10.3f} ms/file |")
    print(f"  | Model Ensemble Loading         | {t_load:>10.2f} ms   |             --       |")
    print(f"  | Scaling & Model Inference      | {t_inf:>10.2f} ms   | {ms_per_file:>10.3f} ms/file |")
    print(f"  | Metric Calculation & Thresholds| {(t_total_wall - t_query - t_load - t_inf):>10.2f} ms   |             --       |")
    print(f"  +--------------------------------+--------------------+----------------------+")
    print(f"  | Total Wall Clock Execution     | {t_total_wall:>10.2f} ms   | {t_total_wall/metrics['total']:>10.3f} ms/file |")
    print(f"  | Peak Memory Allocation (RAM)   | {peak_mem / (1024*1024):>10.2f} MB   |             --       |")
    print(f"  | Real-Time Inference Throughput | {inf_throughput:>10.1f} fps  | (files / second)     |")
    print(f"  +--------------------------------+--------------------+----------------------+")

    print("\n  [DETECTION ACCURACY & CLASSIFICATION PERFORMANCE]")
    print(f"  * Total Files Tested      : {metrics['total']:,} ({metrics['real_count']:,} Real, {metrics['synth_count']:,} Synthetic)")
    print(f"  * Overall Accuracy        : {metrics['accuracy']:.4%}  ({int(round(metrics['accuracy'] * metrics['total'])):,} / {metrics['total']:,} correct)")
    print(f"  * Precision (AI Detected) : {metrics['precision']:.4%}")
    print(f"  * Recall / Sensitivity    : {metrics['recall']:.4%}  (True Synthetic catch rate)")
    print(f"  * Specificity (TNR)       : {metrics['specificity']:.4%}  (True Real preservation rate)")
    print(f"  * False Positive Rate     : {metrics['fpr']:.4%}  (Real misclassified as AI)")
    print(f"  * False Negative Rate     : {metrics['fnr']:.4%}  (AI missed as Real)")
    print(f"  * F1-Score                : {metrics['f1']:.4%}")

    print("\n  [CONFUSION MATRIX]")
    print(f"  +------------------------+------------------------+------------------------+")
    print(f"  |                        | Predicted REAL         | Predicted SYNTHETIC    |")
    print(f"  +------------------------+------------------------+------------------------+")
    print(f"  | Actual REAL            | TN: {metrics['tn']:<18,}| FP: {metrics['fp']:<18,}|")
    print(f"  | Actual SYNTHETIC       | FN: {metrics['fn']:<18,}| TP: {metrics['tp']:<18,}|")
    print(f"  +------------------------+------------------------+------------------------+")

    print("\n  [PER-DOMAIN PERFORMANCE BREAKDOWN]")
    print(f"  +---------------+-------+-------+-------+-----------+----------+----------+----------+--------+")
    print(f"  | Domain        | Total | Real  | Synth | Threshold | Accuracy | Precision| Recall   | FPR    |")
    print(f"  +---------------+-------+-------+-------+-----------+----------+----------+----------+--------+")
    for dom, dstat in sorted(metrics["domain_stats"].items()):
        print(f"  | {dom:<13} | {dstat['count']:>5} | {dstat['real']:>5} | {dstat['synthetic']:>5} | {dstat['threshold']:>9.2f} | {dstat['accuracy']:>7.2%} | {dstat['precision']:>7.2%} | {dstat['recall']:>7.2%} | {dstat['fpr']:>5.2%} |")
    print(f"  +---------------+-------+-------+-------+-----------+----------+----------+----------+--------+")

    print("\n  [CONFIDENCE & PROBABILITY CALIBRATION]")
    print(f"  * Mean AI Probability on Synthetic Audio : {metrics['mean_ai_prob_synth']:.4f}")
    print(f"  * Mean AI Probability on Authentic Audio : {metrics['mean_ai_prob_real']:.4f}")
    print(f"  * Separation Margin                     : {metrics['mean_ai_prob_synth'] - metrics['mean_ai_prob_real']:.4f}")

    print("\n  [SAMPLE PREDICTIONS (FIRST 10 TESTED FILES)]")
    print(f"  {'Filename':<35} | {'Domain':<13} | {'True Label':<10} | {'AI Prob':<8} | {'Pred':<10} | {'Status'}")
    print("  " + "-" * 90)
    for idx, row in df.head(10).iterrows():
        fname = Path(row["file_path"]).name
        if len(fname) > 33:
            fname = fname[:30] + "..."
        status = "OK" if row["correct"] else "MISSED"
        print(f"  {fname:<35} | {row['domain']:<13} | {row['ground_truth']:<10} | {row['ai_probability']:>7.4f} | {row['prediction']:<10} | {status}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PRISM -- Fast Evaluation of Stored Audio Fingerprints from SQLite"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=3000,
        help="Number of stored fingerprints to test (default: 3000, set to 0 for all labeled)"
    )
    parser.add_argument(
        "--sampling", "-s", type=str, default="balanced",
        choices=["balanced", "stratified", "random"],
        help="Sampling strategy: balanced (50/50 real/synthetic), stratified (matches DB domain mix), random"
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH,
        help="Path to SQLite fingerprint database (default: echoes_fingerprints.db)"
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help="Path to pre-trained model bundle (default: models/prism_echoes_ensemble.pkl)"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
        help="Destination CSV path for detailed prediction logs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for repeatable sampling (default: 42)"
    )

    args = parser.parse_args()
    run_benchmark(
        limit=args.limit,
        sampling=args.sampling,
        db_path=args.db_path,
        model_path=args.model_path,
        output_csv=args.output_csv,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
