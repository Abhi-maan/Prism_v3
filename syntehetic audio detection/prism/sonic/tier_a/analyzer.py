"""
PRISM — Tier A | Main Entrypoint: Analyzer (v2)
================================================
Orchestrates all pipeline stages and provides batch-processing utilities.

Public API
----------
analyze_file(file_path, threshold=0.5, *, segment_mode=False,
             window_sec=4.0, hop_sec=2.0, top_k=3) -> dict
    Full analysis for a single audio file.  Returns:
        {
            "file_path":              str,
            "domain":                 str,          # hard label for logging
            "domain_probs":           dict,         # soft probabilities (NEW)
            "artifact_score":         float,
            "vocoder_score":          float,
            "gan_score":              float,
            "diffusion_score":        float,
            "artifact_anomaly_score": float,        # NEW
            "sub_scores":             dict,
            "flags":                  list[str],
            "suspicious_windows":     list[dict],   # non-empty in segment_mode
            "prediction":             str,
        }

analyze_dataset(folder_path, output_csv, threshold=0.5, *,
                ground_truth_csv=None, verbose=True,
                segment_mode=False, window_sec=4.0, hop_sec=2.0) -> pd.DataFrame
    Batch-analyses all audio files under *folder_path*.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessor           import preprocess, preprocess_windows
from .domain                 import classify_domain
from .detectors              import vocoder_score, gan_score, diffusion_score
from .aggregator             import aggregate_scores, aggregate_windowed
from .artifact_stats_detector import artifact_anomaly_score

# ─── Supported audio extensions ───────────────────────────────────────────────
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ─── Default decision threshold ───────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.5

# ─── Parameter sweep thresholds ───────────────────────────────────────────────
SWEEP_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]


def analyze_file(
    file_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    segment_mode: bool = False,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
    top_k: int = 3,
) -> dict:
    """Run the full PRISM Tier-A pipeline on a single audio file.

    Parameters
    ----------
    file_path:
        Path to the audio file.
    threshold:
        Decision boundary for "real" vs "synthetic" (default 0.5).
    segment_mode:
        When True, scores are computed on sliding windows and pooled.
        Surfaces top-K most-suspicious window timestamps.
    window_sec:
        Window length in seconds (segment_mode only, default 4.0).
    hop_sec:
        Window hop in seconds (segment_mode only, default 2.0).
    top_k:
        Number of top suspicious windows to return (segment_mode only).

    Returns
    -------
    dict
        Full result dictionary (see module docstring).
    """
    file_path = str(file_path)

    if segment_mode:
        return _analyze_windowed(
            file_path, threshold, window_sec=window_sec, hop_sec=hop_sec, top_k=top_k
        )

    # ── Whole-file mode ───────────────────────────────────────────────────────
    # Stage 1 — Preprocess
    p = preprocess(file_path)

    # Stage 2 — Domain classification (returns hard label + soft probs)
    domain, domain_probs = classify_domain(p)

    # Stage 3 — Artifact detectors
    v_score = vocoder_score(p, domain)
    g_score = gan_score(p)
    d_score = diffusion_score(p)
    a_score = artifact_anomaly_score(p, domain)

    # Stage 4 — Soft-mixture aggregation
    final_score, flags, suspicious = aggregate_scores(
        v_score, g_score, d_score, a_score, domain_probs, domain_label=domain
    )

    prediction = "synthetic" if final_score >= threshold else "real"

    return {
        "file_path":              file_path,
        "domain":                 domain,
        "domain_probs":           domain_probs,
        "artifact_score":         round(final_score, 4),
        "vocoder_score":          round(v_score, 4),
        "gan_score":              round(g_score, 4),
        "diffusion_score":        round(d_score, 4),
        "artifact_anomaly_score": round(a_score, 4),
        "sub_scores": {
            "vocoder":   round(v_score, 4),
            "gan":       round(g_score, 4),
            "diffusion": round(d_score, 4),
            "anomaly":   round(a_score, 4),
        },
        "flags":              flags,
        "suspicious_windows": suspicious,
        "prediction":         prediction,
    }


def analyze_dataset(
    folder_path: str | Path,
    output_csv: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    ground_truth_csv: str | Path | None = None,
    verbose: bool = True,
    segment_mode: bool = False,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
) -> pd.DataFrame:
    """Batch-analyse all audio files in *folder_path*.

    Parameters
    ----------
    folder_path:
        Root directory to search for audio files (recursive).
    output_csv:
        Destination path for the results CSV.
    threshold:
        Primary decision threshold (default 0.5).
    ground_truth_csv:
        Optional CSV with columns ``filename`` and ``label`` (0=real, 1=synthetic).
    verbose:
        Print progress to stdout.
    segment_mode:
        Enable windowed scoring for each file.
    window_sec, hop_sec:
        Window parameters (segment_mode only).

    Returns
    -------
    pd.DataFrame
        Results dataframe (also written to *output_csv*).
    """
    folder_path = Path(folder_path)
    output_csv  = Path(output_csv)

    audio_files = [
        p for p in folder_path.rglob("*")
        if p.suffix.lower() in _AUDIO_EXTENSIONS
    ]

    if not audio_files:
        print(f"[PRISM] No audio files found under: {folder_path}")
        return pd.DataFrame()

    if verbose:
        print(f"[PRISM] Found {len(audio_files)} audio file(s) — starting analysis …")

    rows: list[dict] = []
    for i, fpath in enumerate(audio_files, start=1):
        if verbose:
            print(f"  [{i:>4}/{len(audio_files)}] {fpath.name}", end=" … ", flush=True)
        try:
            result = analyze_file(
                fpath, threshold=threshold,
                segment_mode=segment_mode, window_sec=window_sec, hop_sec=hop_sec,
            )
            rows.append({
                "filename":               fpath.name,
                "domain":                 result["domain"],
                "domain_speech_prob":     round(result["domain_probs"].get("speech", 0.0), 4),
                "domain_music_prob":      round(result["domain_probs"].get("music", 0.0), 4),
                "domain_env_prob":        round(result["domain_probs"].get("environmental", 0.0), 4),
                "artifact_score":         result["artifact_score"],
                "vocoder_score":          result["vocoder_score"],
                "gan_score":              result["gan_score"],
                "diffusion_score":        result["diffusion_score"],
                "artifact_anomaly_score": result["artifact_anomaly_score"],
                "flags":                  "|".join(result["flags"]),
                "suspicious_windows":     str(result["suspicious_windows"]) if result["suspicious_windows"] else "",
                "prediction":             result["prediction"],
            })
            if verbose:
                print(f'{result["artifact_score"]:.3f} → {result["prediction"]}')
        except Exception as exc:
            if verbose:
                print(f"ERROR — {exc}")
            rows.append({
                "filename":               fpath.name,
                "domain":                 "error",
                "domain_speech_prob":     -1.0,
                "domain_music_prob":      -1.0,
                "domain_env_prob":        -1.0,
                "artifact_score":         -1.0,
                "vocoder_score":          -1.0,
                "gan_score":              -1.0,
                "diffusion_score":        -1.0,
                "artifact_anomaly_score": -1.0,
                "flags":                  "error",
                "suspicious_windows":     "",
                "prediction":             "error",
            })

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n[PRISM] Results written → {output_csv}")

    if ground_truth_csv is not None:
        _run_threshold_sweep(df, ground_truth_csv, verbose=verbose)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _analyze_windowed(
    file_path: str,
    threshold: float,
    *,
    window_sec: float,
    hop_sec: float,
    top_k: int,
) -> dict:
    """Run the pipeline in sliding-window mode (memory-bounded)."""
    # We need domain probs from the whole file for blending — use first window
    # as a quick estimate, then stream the rest
    domain: str = "environmental"
    domain_probs: dict[str, float] = {"speech": 1/3, "music": 1/3, "environmental": 1/3}
    first_window_done = False

    def _scores_generator():
        nonlocal domain, domain_probs, first_window_done
        for t_start, p in preprocess_windows(file_path, window_sec, hop_sec):
            if not first_window_done:
                domain, domain_probs = classify_domain(p)
                first_window_done = True
            v = vocoder_score(p, domain)
            g = gan_score(p)
            d = diffusion_score(p)
            a = artifact_anomaly_score(p, domain)
            yield t_start, v, g, d, a

    final_score, flags, suspicious = aggregate_windowed(
        _scores_generator(), domain_probs, top_k=top_k
    )

    prediction = "synthetic" if final_score >= threshold else "real"

    # Get sub-scores from last window for reporting (already inside generator)
    return {
        "file_path":              file_path,
        "domain":                 domain,
        "domain_probs":           domain_probs,
        "artifact_score":         round(final_score, 4),
        "vocoder_score":          round(0.0, 4),   # pooled; individual not tracked here
        "gan_score":              round(0.0, 4),
        "diffusion_score":        round(0.0, 4),
        "artifact_anomaly_score": round(0.0, 4),
        "sub_scores":             {},
        "flags":                  flags,
        "suspicious_windows":     suspicious,
        "prediction":             prediction,
    }


def _run_threshold_sweep(
    df: pd.DataFrame,
    ground_truth_csv: str | Path,
    *,
    verbose: bool = True,
) -> None:
    """Print precision/recall/F1/accuracy for each sweep threshold."""
    gt     = pd.read_csv(ground_truth_csv)
    merged = df.merge(gt, on="filename", how="inner")

    if merged.empty:
        print("[PRISM] ⚠ No matching filenames between results and ground truth.")
        return

    y_true = merged["label"].values.astype(int)
    scores = merged["artifact_score"].values

    header = f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    sep    = "-" * len(header)

    if verbose:
        print(f"\n[PRISM] Threshold Sweep Results\n{sep}\n{header}\n{sep}")

    for thr in SWEEP_THRESHOLDS:
        y_pred = (scores >= thr).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        if verbose:
            print(
                f"{thr:>10.2f} {precision:>10.4f} {recall:>8.4f} "
                f"{f1:>8.4f} {accuracy:>10.4f}"
            )

    if verbose:
        print(sep)
