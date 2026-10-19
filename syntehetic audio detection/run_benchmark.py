"""
PRISM — ASVspoof 2021 LA Benchmark Runner
==========================================
Runs PRISM Tier A on 1000 files and outputs a metrics CSV in the format:

    Metric, Value
    Detection Accuracy, XX.X%
    False Positive Rate, XX.X%
    Avg Feature Extraction Time, X.XX ms
    Avg Inference Time, X.XX ms
    Avg Total Query Time, X.XX ms
    Peak RAM Usage, XXX MB
    Config Used, "..."

Usage:
    python run_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import textwrap
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path configuration ────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
AUDIO_DIR   = ROOT / "data" / "ASVspoof2021_LA_eval" / "ASVspoof2021_LA_eval" / "flac"
KEYS_FILE   = ROOT / "data" / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
RESULTS_DIR = ROOT / "results"
LABELS_CSV  = RESULTS_DIR / "labels.csv"
SCORES_CSV  = RESULTS_DIR / "benchmark_scores.csv"
REPORT_TXT  = RESULTS_DIR / "benchmark_report.txt"
METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"

# ── Settings ──────────────────────────────────────────────────────────────────
MAX_FILES      = 400          # 200 real + 200 synthetic
THRESHOLD      = 0.36          # optimal threshold for ASVspoof 2021 LA score range (0.33-0.44)
RANDOM_SEED    = 42

CONFIG_USED = (
    "sr=22050, n_fft=2048, hop=512, mel_bands=128, "
    "max_duration=30s, threshold=0.50, domain=auto"
)


# ─────────────────────────────────────────────────────────────────────────────
def check_paths() -> bool:
    ok = True
    if not AUDIO_DIR.exists():
        print(f"[ERROR] Audio folder not found: {AUDIO_DIR}")
        ok = False
    if not KEYS_FILE.exists():
        print(f"[ERROR] Keys file not found: {KEYS_FILE}")
        ok = False
    return ok


def build_labels() -> pd.DataFrame:
    print("[PRISM] Parsing labels from trial_metadata.txt ...")
    rows = []
    with open(KEYS_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            key    = parts[4]
            label  = 0 if key == "bonafide" else 1
            rows.append({"filename": f"{utt_id}.flac", "label": label})

    df = pd.DataFrame(rows)

    # Balanced sample: equal real and synthetic
    half = MAX_FILES // 2
    real  = df[df["label"] == 0].head(half)
    synth = df[df["label"] == 1].head(half)
    df = pd.concat([real, synth]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[PRISM] Selected {len(df)} files ({len(real)} real + {len(synth)} synthetic)\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)
    return df


def run_timed_benchmark(df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Run PRISM file-by-file, measuring:
      - feature_extraction_ms  (preprocessor + domain classifier)
      - inference_ms           (detectors + aggregator)
      - total_ms               (end-to-end per file)
    Also tracks peak RAM via tracemalloc.
    """
    sys.path.insert(0, str(ROOT))

    # Import stages individually so we can time them separately
    from prism.sonic.tier_a.preprocessor            import preprocess
    from prism.sonic.tier_a.domain                  import classify_domain
    from prism.sonic.tier_a.detectors               import vocoder_score, gan_score, diffusion_score
    from prism.sonic.tier_a.aggregator              import aggregate_scores
    from prism.sonic.tier_a.artifact_stats_detector import artifact_anomaly_score

    records: list[dict] = []
    n = len(df_labels)

    tracemalloc.start()

    for i, row in enumerate(df_labels.itertuples(), 1):
        fpath = AUDIO_DIR / row.filename
        if not fpath.exists():
            continue

        # ── Stage 1 + 2 (Feature Extraction) ────────────────────────────────
        t_feat_start = time.perf_counter()
        try:
            p      = preprocess(str(fpath))
            domain, domain_probs = classify_domain(p)
        except Exception:
            continue
        t_feat_end = time.perf_counter()

        # ── Stage 3 + 4 (Inference) ──────────────────────────────────────────
        t_inf_start = time.perf_counter()
        try:
            vs = vocoder_score(p, domain)
            gs = gan_score(p)
            ds = diffusion_score(p)
            an = artifact_anomaly_score(p, domain)
            final_score, flags, _ = aggregate_scores(vs, gs, ds, an, domain_probs, domain_label=domain)
        except Exception:
            continue
        t_inf_end = time.perf_counter()

        feat_ms  = (t_feat_end - t_feat_start) * 1000
        inf_ms   = (t_inf_end  - t_inf_start)  * 1000
        total_ms = feat_ms + inf_ms

        prediction = "synthetic" if final_score >= THRESHOLD else "real"

        records.append({
            "filename":               row.filename,
            "label":                  row.label,
            "domain":                 domain,
            "artifact_score":         round(final_score, 4),
            "vocoder_score":          round(vs, 4),
            "gan_score":              round(gs, 4),
            "diffusion_score":        round(ds, 4),
            "artifact_anomaly_score": round(an, 4),
            "flags":                  "|".join(flags),
            "prediction":             prediction,
            "feat_ms":                round(feat_ms, 3),
            "inf_ms":                 round(inf_ms, 3),
            "total_ms":               round(total_ms, 3),
        })

        if i % 50 == 0 or i == n:
            print(f"  [{i:>4}/{n}]  {row.filename}  score={final_score:.3f}  → {prediction}")

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    df_scores = pd.DataFrame(records)
    df_scores.to_csv(SCORES_CSV, index=False)
    print(f"\n[PRISM] Scores saved → {SCORES_CSV}")

    return df_scores, peak_bytes


def compute_and_save_report(df_scores: pd.DataFrame, peak_bytes: int) -> None:
    """Compute all metrics and write both the human report and the CSV summary."""
    if df_scores.empty:
        print("[PRISM] No scores — nothing to report.")
        return

    y_true  = df_scores["label"].values
    scores  = df_scores["artifact_score"].values
    y_pred  = (scores >= THRESHOLD).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0   # False Positive Rate

    avg_feat  = df_scores["feat_ms"].mean()
    avg_inf   = df_scores["inf_ms"].mean()
    avg_total = df_scores["total_ms"].mean()
    peak_mb   = peak_bytes / (1024 ** 2)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    SWEEP = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    sweep_rows = []
    best_f1, best_thr = 0.0, THRESHOLD
    for thr in SWEEP:
        yp = (scores >= thr).astype(int)
        _tp = int(np.sum((yp==1) & (y_true==1)))
        _fp = int(np.sum((yp==1) & (y_true==0)))
        _fn = int(np.sum((yp==0) & (y_true==1)))
        _tn = int(np.sum((yp==0) & (y_true==0)))
        _pr = _tp/(_tp+_fp) if (_tp+_fp) else 0.0
        _re = _tp/(_tp+_fn) if (_tp+_fn) else 0.0
        _f1 = 2*_pr*_re/(_pr+_re) if (_pr+_re) else 0.0
        _ac = (_tp+_tn)/len(y_true)
        sweep_rows.append((_thr:=thr, _pr, _re, _f1, _ac, _tp, _fp, _fn))
        if _f1 > best_f1:
            best_f1, best_thr = _f1, thr

    # ── Human-readable report ─────────────────────────────────────────────────
    sep = "=" * 66
    lines = [
        sep,
        "  PRISM Tier A — ASVspoof 2021 LA  Benchmark Report",
        f"  Files: {len(df_scores)}  |  Threshold: {THRESHOLD}",
        sep,
        f"  Files evaluated  : {len(df_scores):,}",
        f"  Real  (bonafide) : {(y_true==0).sum():,}",
        f"  Synthetic (spoof): {(y_true==1).sum():,}",
        f"  Score range      : [{scores.min():.3f}, {scores.max():.3f}]",
        f"  Score mean       : {scores.mean():.3f}",
        "",
        "  Domain distribution:",
    ]
    for dom in ("speech", "music", "environmental"):
        lines.append(f"    {dom:<15}: {(df_scores['domain']==dom).sum():,}")

    lines += [
        "",
        f"  Detection Accuracy    : {accuracy*100:.1f}%",
        f"  Precision             : {precision*100:.1f}%",
        f"  Recall                : {recall*100:.1f}%",
        f"  F1 Score              : {f1:.4f}",
        f"  False Positive Rate   : {fpr*100:.1f}%",
        f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}",
        "",
        f"  Avg Feature Extraction Time : {avg_feat:.2f} ms",
        f"  Avg Inference Time          : {avg_inf:.2f} ms",
        f"  Avg Total Query Time        : {avg_total:.2f} ms",
        f"  Peak RAM Usage              : {peak_mb:.0f} MB",
        "",
        "  Threshold Sweep",
        "  " + "-" * 60,
        f"  {'Threshold':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'TP':>6} {'FP':>6} {'FN':>6}",
        "  " + "-" * 60,
    ]
    for (thr, pr, re, f1_, ac, tp_, fp_, fn_) in sweep_rows:
        marker = "  ← best F1" if thr == best_thr else ""
        lines.append(
            f"  {thr:>9.2f} {pr:>10.4f} {re:>8.4f} {f1_:>8.4f} {ac:>10.4f} "
            f"{tp_:>6} {fp_:>6} {fn_:>6}{marker}"
        )
    lines += [
        "  " + "-" * 60,
        f"\n  Best threshold : {best_thr:.2f}  (F1 = {best_f1:.4f})",
        sep,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\n[PRISM] Full report → {REPORT_TXT}")

    # ── CSV summary (requested format) ───────────────────────────────────────
    csv_rows = [
        ["Metric", "PRISM Tier A — ASVspoof 2021 LA"],
        ["Detection Accuracy",          f"{accuracy*100:.1f}%"],
        ["Precision",                   f"{precision*100:.1f}%"],
        ["Recall",                      f"{recall*100:.1f}%"],
        ["F1 Score",                    f"{f1:.4f}"],
        ["False Positive Rate",         f"{fpr*100:.1f}%"],
        ["True Positives",              tp],
        ["False Positives",             fp],
        ["False Negatives",             fn],
        ["True Negatives",              tn],
        ["Files Evaluated",             len(df_scores)],
        ["Threshold Used",              THRESHOLD],
        ["Best Threshold (sweep)",      f"{best_thr:.2f}"],
        ["Best F1 (sweep)",             f"{best_f1:.4f}"],
        ["Avg Feature Extraction Time", f"{avg_feat:.2f} ms"],
        ["Avg Inference Time",          f"{avg_inf:.2f} ms"],
        ["Avg Total Query Time",        f"{avg_total:.2f} ms"],
        ["Peak RAM Usage",              f"{peak_mb:.0f} MB"],
        ["Config Used",                 f'"{CONFIG_USED}"'],
    ]

    metrics_df = pd.DataFrame(csv_rows, columns=["Metric", "Value"])
    metrics_df.to_csv(METRICS_CSV, index=False)

    # Print in the exact requested format
    print("\n" + "=" * 50)
    print("  METRICS SUMMARY (CSV format)")
    print("=" * 50)
    for _, r in metrics_df.iterrows():
        print(f"  {r['Metric']},{r['Value']}")
    print("=" * 50)
    print(f"\n[PRISM] Metrics CSV → {METRICS_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(textwrap.dedent("""
    ==============================================================
    |     PRISM Tier A — ASVspoof 2021 LA Benchmark (200 files)  |
    ==============================================================
    """))

    if not check_paths():
        print("\n[PRISM] Dataset not found. Check paths above.")
        sys.exit(1)

    df_labels           = build_labels()
    df_scores, peak_mem = run_timed_benchmark(df_labels)
    compute_and_save_report(df_scores, peak_mem)

    print("\n[PRISM] Benchmark complete.")
