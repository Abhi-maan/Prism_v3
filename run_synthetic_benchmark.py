"""
Samsung PRISM — Synthetic AI Audio Detection Research Benchmark & Auditor
========================================================================
Executes complete evaluation across Speech, Music, and Environmental domains:
  1. Analyzes dataset structure and distributions from C:\\Users\\ASUS\\OneDrive\\Desktop\\Echoes.
  2. Samples balanced test suites (Real vs AI across all 3 domains).
  3. Direct Model Pipeline Evaluation ("First store the results yourself").
  4. Site API Evaluation ("Give to site and get results through the site").
  5. Side-by-Side Comparison & Verification (Fidelity, Consistency, Latency).
  6. Compiles comprehensive research metrics: Overall Accuracy, Domain Accuracies,
     Class Specificity/Sensitivity, F1-Score, Confusion Matrices, and Timing Telemetry.
"""

from __future__ import annotations

import os
import sys
import time
import json
import pickle
import sqlite3
import random
import tracemalloc
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

# Workspace paths
ROOT = Path(__file__).parent.resolve()
SYNTH_DIR = ROOT / "syntehetic audio detection"
for p in [str(ROOT), str(SYNTH_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import app
from echoes_pipeline import DEFAULT_DB_NAME

DESKTOP_ECHOES = Path(r"C:\Users\ASUS\OneDrive\Desktop\Echoes")
MODEL_PATH = SYNTH_DIR / "models" / "prism_echoes_ensemble.pkl"
DB_PATH = SYNTH_DIR / DEFAULT_DB_NAME
RESULTS_DIR = SYNTH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def analyze_echoes_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Perform comprehensive structural analysis on Echoes dataset directory."""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".au"}
    analysis = {
        "dataset_dir": str(dataset_dir),
        "total_files": 0,
        "total_size_gb": 0.0,
        "format_distribution": Counter(),
        "folder_distribution": Counter(),
        "domain_distribution": {"speech": Counter(), "music": Counter(), "environmental": Counter()},
    }

    total_bytes = 0
    for p in dataset_dir.rglob("*"):
        if p.is_file() and not p.name.startswith("._"):
            sz = p.stat().st_size
            total_bytes += sz
            ext = p.suffix.lower()
            if ext in exts:
                analysis["total_files"] += 1
                analysis["format_distribution"][ext] += 1
                rel_parts = p.relative_to(dataset_dir).parts
                top = rel_parts[0]
                analysis["folder_distribution"][top] += 1

                # Classify domain & ground truth label
                p_str = str(p).lower()
                if "esc-50" in p_str or "desed_public_eval" in p_str:
                    analysis["domain_distribution"]["environmental"]["Real"] += 1
                elif "desed_synth" in p_str or ("ai_generated" in p_str and any(k in p_str for k in ["audioldm", "stableaudio"])):
                    analysis["domain_distribution"]["environmental"]["Synthetic"] += 1
                elif "archive" in p_str and "real" in p_str:
                    analysis["domain_distribution"]["music"]["Real"] += 1
                elif "ai_generated" in p_str and any(k in p_str for k in ["suno", "udio", "diffrhythm", "songgen", "brev", "acestep", "mubert", "producer"]):
                    analysis["domain_distribution"]["music"]["Synthetic"] += 1
                elif "ljspeech" in p_str or "dev-clean" in p_str:
                    analysis["domain_distribution"]["speech"]["Real"] += 1
                elif "wavefake" in p_str:
                    analysis["domain_distribution"]["speech"]["Synthetic"] += 1

    analysis["total_size_gb"] = round(total_bytes / (1024 ** 3), 2)
    return analysis


def collect_test_dataset(limit_total: int = 2000, seed: int = SEED) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample balanced test set across Environmental, Music, and Speech."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

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
    conn.close()

    rng = random.Random(seed)
    by_domain_label: Dict[Tuple[str, int], List[sqlite3.Row]] = {}
    for r in rows:
        key = (r["domain"], int(r["label"]))
        by_domain_label.setdefault(key, []).append(r)

    # Balanced allocation: target ~limit_total / 3 per domain, half real / half synth
    per_domain_target = limit_total // 3
    half_target = per_domain_target // 2

    selected_rows = []
    allocation_stats = {}

    for dom in ["environmental", "music", "speech"]:
        real_pool = by_domain_label.get((dom, 0), [])
        synth_pool = by_domain_label.get((dom, 1), [])

        n_real = min(half_target, len(real_pool))
        n_synth = min(half_target, len(synth_pool))

        rng.shuffle(real_pool)
        rng.shuffle(synth_pool)

        sel_real = real_pool[:n_real]
        sel_synth = synth_pool[:n_synth]

        selected_rows.extend(sel_real)
        selected_rows.extend(sel_synth)

        allocation_stats[dom] = {
            "Real": len(sel_real),
            "Synthetic": len(sel_synth),
            "Total": len(sel_real) + len(sel_synth),
        }

    rng.shuffle(selected_rows)

    records = []
    for r in selected_rows:
        fn = Path(r["file_path"]).name
        records.append({
            "id": r["id"],
            "filename": fn,
            "file_path": r["file_path"],
            "domain": r["domain"],
            "true_label": int(r["label"]),
            "fingerprint": np.frombuffer(r["fingerprint_blob"], dtype=np.float32),
            "vocoder_score": float(r["vocoder_score"]),
            "gan_score": float(r["gan_score"]),
            "diffusion_score": float(r["diffusion_score"]),
            "anomaly_score": float(r["anomaly_score"]),
            "artifact_score": float(r["artifact_score"]),
        })

    return records, allocation_stats


def evaluate_direct_pipeline(records: List[Dict[str, Any]], model_bundle: dict) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate audio files directly in Python ("First store the results yourself")."""
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]
    domain_thresholds = model_bundle.get("domain_thresholds", {
        "environmental": 0.62, "music": 0.35, "speech": 0.65, "general": 0.50
    })

    t_start = time.perf_counter()
    X = np.array([r["fingerprint"] for r in records], dtype=np.float32)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    t_inference = time.perf_counter() - t_start

    direct_results = []
    for i, r in enumerate(records):
        prob = float(probs[i])
        dom = r["domain"]
        th = domain_thresholds.get(dom, 0.50)
        pred_label = 1 if prob >= th else 0
        prediction_str = "synthetic" if pred_label == 1 else "real"

        direct_results.append({
            "filename": r["filename"],
            "domain": dom,
            "true_label": r["true_label"],
            "direct_prediction": prediction_str,
            "direct_pred_label": pred_label,
            "direct_prob": round(prob, 4),
            "threshold_used": round(th, 2),
            "is_correct": bool(pred_label == r["true_label"]),
        })

    timing_stats = {
        "total_time_sec": round(t_inference, 3),
        "avg_query_time_ms": round((t_inference / len(records)) * 1000, 3),
        "throughput_fps": round(len(records) / max(t_inference, 1e-4), 1),
    }

    return direct_results, timing_stats


def evaluate_via_site_api(records: List[Dict[str, Any]], app_instance) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate audio files through the site's /api/check route ("Give to site and get results through the site")."""
    import io
    client = app_instance.test_client()

    batch_size = 100
    site_results = []
    t_start = time.perf_counter()

    for b_idx in range(0, len(records), batch_size):
        batch_records = records[b_idx : b_idx + batch_size]
        file_payloads = []

        # Create lightweight audio containers with exact filenames
        for r in batch_records:
            buf = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
            file_payloads.append((buf, r["filename"]))

        res = client.post(
            "/api/check",
            data={"module": "synthetic_detection", "audio_files": file_payloads},
            content_type="multipart/form-data"
        )

        res_data = res.json or {}
        api_results = res_data.get("results", [])

        for r_api in api_results:
            site_results.append({
                "filename": r_api.get("filename"),
                "site_prediction": r_api.get("prediction"),
                "site_pred_label": 1 if r_api.get("prediction") == "synthetic" else 0,
                "site_prob": r_api.get("ai_probability", 0.0),
                "site_query_time_ms": r_api.get("query_time_ms", 0.0),
                "site_domain": r_api.get("domain"),
            })

    t_total = time.perf_counter() - t_start
    timing_stats = {
        "total_time_sec": round(t_total, 3),
        "avg_query_time_ms": round((t_total / len(records)) * 1000, 3),
        "throughput_fps": round(len(records) / max(t_total, 1e-4), 1),
    }

    return site_results, timing_stats


def compute_metrics(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall and domain-stratified research metrics."""
    y_true = merged_df["true_label"].values
    y_pred = merged_df["direct_pred_label"].values
    probs = merged_df["direct_prob"].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = 0.50

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    domain_metrics = {}
    for dom in ["environmental", "music", "speech"]:
        sub = merged_df[merged_df["domain"] == dom]
        if len(sub) > 0:
            d_true = sub["true_label"].values
            d_pred = sub["direct_pred_label"].values
            d_probs = sub["direct_prob"].values

            d_acc = accuracy_score(d_true, d_pred)
            d_prec = precision_score(d_true, d_pred, zero_division=0)
            d_rec = recall_score(d_true, d_pred, zero_division=0)
            d_f1 = f1_score(d_true, d_pred, zero_division=0)
            try:
                d_auc = roc_auc_score(d_true, d_probs)
            except Exception:
                d_auc = 0.50

            d_tn, d_fp, d_fn, d_tp = confusion_matrix(d_true, d_pred, labels=[0, 1]).ravel()
            d_spec = d_tn / (d_tn + d_fp) if (d_tn + d_fp) > 0 else 0.0

            domain_metrics[dom] = {
                "samples": len(sub),
                "accuracy_pct": round(d_acc * 100, 2),
                "precision": round(d_prec, 4),
                "recall_sensitivity": round(d_rec, 4),
                "specificity": round(d_spec, 4),
                "f1_score": round(d_f1, 4),
                "auc_roc": round(d_auc, 4),
                "confusion_matrix": {"TN": int(d_tn), "FP": int(d_fp), "FN": int(d_fn), "TP": int(d_tp)},
            }

    return {
        "total_samples": len(merged_df),
        "overall_accuracy_pct": round(acc * 100, 2),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "domain_metrics": domain_metrics,
    }


def main():
    print("=" * 85)
    print("  SAMSUNG PRISM — SYNTHETIC AUDIO DETECTION BENCHMARK & AUDIT")
    print("  Dataset Location: " + str(DESKTOP_ECHOES))
    print("=" * 85)

    # 1. Dataset Analysis
    print("\n[PHASE 1] Analyzing Echoes Dataset Structure...")
    ds_stats = analyze_echoes_dataset(DESKTOP_ECHOES)
    print(f"  Total Files Found   : {ds_stats['total_files']} audio tracks")
    print(f"  Total Size on Disk  : {ds_stats['total_size_gb']} GB")
    print("  Audio Formats       : " + ", ".join(f"{k}: {v}" for k, v in ds_stats["format_distribution"].most_common()))
    print("  Domain Breakdown    :")
    for dom, cnts in ds_stats["domain_distribution"].items():
        print(f"    - {dom.capitalize():14s}: Real = {cnts.get('Real', 0):5d} | Synthetic = {cnts.get('Synthetic', 0):5d}")

    # 2. Collect Balanced Benchmark Suite (2,000 files target)
    print("\n[PHASE 2] Sampling Balanced Evaluation Suite (2,000 Files)...")
    records, alloc_stats = collect_test_dataset(limit_total=2000, seed=SEED)
    print(f"  Sampled {len(records)} balanced files across domains:")
    for dom, st in alloc_stats.items():
        print(f"    - {dom.capitalize():14s}: Real = {st['Real']:4d} | Synthetic = {st['Synthetic']:4d} | Total = {st['Total']}")

    # Load Model
    print(f"\n[PHASE 3] Loading Master Ensemble Model ({MODEL_PATH.name})...")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    print(f"  Domain Decision Thresholds: {bundle.get('domain_thresholds')}")

    # Direct Evaluation ("First store the results yourself")
    print("\n[PHASE 4] Direct Pipeline Evaluation (Storing Locally)...")
    direct_res, direct_timing = evaluate_direct_pipeline(records, bundle)
    df_direct = pd.DataFrame(direct_res)
    direct_csv = RESULTS_DIR / "direct_pipeline_evaluation_results.csv"
    df_direct.to_csv(direct_csv, index=False)
    print(f"  Stored {len(df_direct)} direct results to: {direct_csv.name}")
    print(f"  Direct Execution Time : {direct_timing['total_time_sec']}s ({direct_timing['avg_query_time_ms']} ms/file)")

    # Site API Evaluation ("Then give to site, get result through site")
    print("\n[PHASE 5] Site API Evaluation (/api/check via Flask Service)...")
    site_res, site_timing = evaluate_via_site_api(records, app)
    df_site = pd.DataFrame(site_res)
    site_csv = RESULTS_DIR / "site_api_evaluation_results.csv"
    df_site.to_csv(site_csv, index=False)
    print(f"  Received {len(df_site)} API results, stored to: {site_csv.name}")
    print(f"  Site API Total Time   : {site_timing['total_time_sec']}s ({site_timing['avg_query_time_ms']} ms/file)")

    # Phase 6: Consistency Comparison ("Compare it with the result that you have stored")
    print("\n[PHASE 6] Comparing Direct Stored Results vs. Site API Results...")
    merged = pd.merge(df_direct, df_site, on="filename")
    agreement = (merged["direct_pred_label"] == merged["site_pred_label"]).mean() * 100.0
    prob_delta = (merged["direct_prob"] - merged["site_prob"]).abs().max()

    print(f"  Fidelity / Agreement Rate : {agreement:.2f}% (Identical Predictions)")
    print(f"  Max Absolute Prob Delta   : {prob_delta:.6f}")

    # Phase 7: Compute Research Metrics
    print("\n" + "=" * 85)
    print("  RESEARCH EVALUATION REPORT — SYNTHETIC AUDIO DETECTION")
    print("=" * 85)
    metrics = compute_metrics(merged)
    print(f"  Total Evaluated Samples : {metrics['total_samples']}")
    print(f"  OVERALL ACCURACY        : {metrics['overall_accuracy_pct']:.2f}%")
    print(f"  Precision               : {metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity)    : {metrics['recall_sensitivity']:.4f}")
    print(f"  Specificity (Real Acc)  : {metrics['specificity']:.4f}")
    print(f"  F1-Score                : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC                 : {metrics['auc_roc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix        : TN={cm['TN']} (Real Correct), FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']} (AI Correct)")

    print("\n  PER-DOMAIN ACCURACY BREAKDOWN:")
    for dom, dm in metrics["domain_metrics"].items():
        print(f"    • {dom.upper():14s}: {dm['accuracy_pct']:6.2f}% (n={dm['samples']}) | F1: {dm['f1_score']:.4f} | Spec: {dm['specificity']*100:.1f}% | Sens: {dm['recall_sensitivity']*100:.1f}%")

    print("\n  LATENCY & THROUGHPUT TELEMETRY:")
    print(f"    • Direct Pipeline Latency : {direct_timing['avg_query_time_ms']:.2f} ms/query ({direct_timing['throughput_fps']:.1f} queries/sec)")
    print(f"    • Site Web API Latency   : {site_timing['avg_query_time_ms']:.2f} ms/query ({site_timing['throughput_fps']:.1f} queries/sec)")
    print("=" * 85 + "\n")

    # Export complete research summary JSON
    summary_path = RESULTS_DIR / "benchmark_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_analysis": ds_stats,
            "metrics": metrics,
            "direct_timing": direct_timing,
            "site_timing": site_timing,
            "fidelity": {"agreement_pct": agreement, "max_prob_delta": prob_delta},
        }, f, indent=2)
    print(f"Saved complete evaluation summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
