"""
PRISM — Held-Out Test Set Verification Runner
==============================================
Runs an objective evaluation on the 100% held-out test audio files
(from `C:\\Users\\ASUS\\OneDrive\\Desktop\\Echoes_Test_Dataset`)
that were NEVER seen during model training.

Usage:
  # Quick test on 100 random held-out files:
  python run_heldout_test.py --limit 100

  # Test 200 files in music domain:
  python run_heldout_test.py --domain music --limit 200

  # Test all held-out files:
  python run_heldout_test.py --limit 0
"""

import os
import sys
import time
import pickle
import argparse
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.resolve()
SYNTH_DIR = ROOT / "syntehetic audio detection"
sys.path.insert(0, str(SYNTH_DIR))
sys.path.insert(0, str(ROOT))

from echoes_pipeline import extract_uniform_fingerprint

DEFAULT_TEST_DIR = Path(r"C:\Users\ASUS\OneDrive\Desktop\Echoes_Test_Dataset")
DEFAULT_MANIFEST = DEFAULT_TEST_DIR / "manifest_test_dataset.csv"
DEFAULT_MODEL_PATH = SYNTH_DIR / "models" / "prism_echoes_ensemble.pkl"

def main():
    parser = argparse.ArgumentParser(description="Evaluate PRISM model on 100% held-out test data.")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Path to manifest_test_dataset.csv")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to prism_echoes_ensemble.pkl")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "music", "speech", "environmental"], help="Filter by domain")
    parser.add_argument("--limit", type=int, default=100, help="Number of files to test (0 for all)")
    parser.add_argument("--output-csv", type=str, default=None, help="Save per-file results to CSV")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    # 1. Load model bundle
    print("=" * 80)
    print("  PRISM HELD-OUT DATASET OBJECTIVE ACCURACY VERIFICATION")
    print("=" * 80)
    print(f"Loading production ensemble from: {args.model}")
    t0 = time.perf_counter()
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    print(f"Model loaded in {(time.perf_counter() - t0)*1000:.1f}ms. Version: {bundle.get('version', 'unknown')}")

    domain_bundles = bundle.get("domain_bundles", {})
    fallback_model = bundle.get("model")
    fallback_scaler = bundle.get("scaler")

    # 2. Read test manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if args.domain == "all" or r["domain"] == args.domain:
                rows.append(r)

    print(f"Total available test samples (Domain: {args.domain}): {len(rows)}")

    # Sample balanced subset if limit specified
    if args.limit > 0 and len(rows) > args.limit:
        import random
        rng = random.Random(42)
        # Stratify by label
        reals = [r for r in rows if int(r["label"]) == 0]
        ais = [r for r in rows if int(r["label"]) == 1]
        half = args.limit // 2
        selected = rng.sample(reals, min(half, len(reals))) + rng.sample(ais, min(half, len(ais)))
        rng.shuffle(selected)
        eval_rows = selected
    else:
        eval_rows = rows

    print(f"Evaluating {len(eval_rows)} files...\n")

    correct = 0
    total = 0
    latencies = []
    results = []

    domain_stats = {}

    for i, r in enumerate(eval_rows, 1):
        fpath = r["test_file_path"]
        ground_truth = int(r["label"])
        dom = r["domain"]
        cat = r["category"]

        if not Path(fpath).exists():
            continue

        t_start = time.perf_counter()
        try:
            fp_res = extract_uniform_fingerprint(fpath)
            vec = fp_res["fingerprint"]

            # Select domain specialist or fallback
            if dom in domain_bundles:
                d_b = domain_bundles[dom]
                sc = d_b["scaler"]
                mdl = d_b["model"]
                thresh = d_b["threshold"]
            else:
                sc = fallback_scaler
                mdl = fallback_model
                thresh = 0.50

            feat = sc.transform([vec])
            prob = float(mdl.predict_proba(feat)[0, 1])
            pred = 1 if prob >= thresh else 0
        except Exception as e:
            print(f"[{i}/{len(eval_rows)}] Error processing {Path(fpath).name}: {e}")
            continue

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(elapsed_ms)

        is_correct = (pred == ground_truth)
        if is_correct:
            correct += 1
        total += 1

        # Domain stats
        if dom not in domain_stats:
            domain_stats[dom] = {"correct": 0, "total": 0}
        domain_stats[dom]["total"] += 1
        if is_correct:
            domain_stats[dom]["correct"] += 1

        pred_str = "AI" if pred == 1 else "REAL"
        gt_str = "AI" if ground_truth == 1 else "REAL"
        status_str = "PASS" if is_correct else "FAIL"

        results.append({
            "filename": Path(fpath).name,
            "domain": dom,
            "category": cat,
            "ground_truth": gt_str,
            "predicted": pred_str,
            "ai_probability": round(prob, 4),
            "status": status_str,
            "latency_ms": round(elapsed_ms, 2)
        })

        if i <= 15 or i % 10 == 0 or i == len(eval_rows):
            print(f"[{i:3d}/{len(eval_rows)}] {Path(fpath).name[:30]:30s} | Dom: {dom:12s} | GT: {gt_str:4s} | Pred: {pred_str:4s} (p={prob:.3f}) | {status_str:4s} | {elapsed_ms:.1f}ms")

    # Summary
    acc = (correct / total * 100) if total > 0 else 0.0
    avg_lat = np.mean(latencies) if latencies else 0.0

    print("\n" + "=" * 80)
    print("  FINAL ACCURACY & LATENCY SUMMARY ON HELD-OUT TEST DATA")
    print("=" * 80)
    print(f"Total Evaluated : {total}")
    print(f"Overall Accuracy: {correct}/{total} ({acc:.2f}%)")
    print(f"Average Latency : {avg_lat:.2f} ms/sample")
    print("-" * 80)
    for dom, s in domain_stats.items():
        d_acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
        print(f"  {dom.upper():15s}: {s['correct']}/{s['total']} ({d_acc:.2f}%)")
    print("=" * 80)

    if args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "domain", "category", "ground_truth", "predicted", "ai_probability", "status", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed per-sample audit report saved to: {out_p}")

if __name__ == "__main__":
    main()
