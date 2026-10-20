"""
PRISM — Tier A | Stage 4: Domain-Weighted Score Aggregator (v2)
================================================================
Fuses four detector scores into a single synthetic-probability score
using **soft domain blending** instead of hard domain switching.

Changes from v1
---------------
* Accepts ``domain_probs`` dict (probability distribution over domains)
  instead of a hard domain string — weights are blended proportionally.
* 4th detector term: ``anomaly_s`` (from artifact_stats_detector.py).
* Weights loaded from ``config/fusion_weights.json`` at import time;
  falls back to hardcoded defaults if the file is missing.
* ``aggregate_windowed()`` helper for streaming segment-level scoring.

Public API
----------
aggregate_scores(vocoder_s, gan_s, diffusion_s, anomaly_s,
                 domain_probs, *, domain_label="",
                 windowed_scores=None, pooling="max", top_k=3)
    -> tuple[float, list[str], list[dict]]

aggregate_windowed(scores_iter, domain_probs, *, pooling="max", top_k=3)
    -> tuple[float, list[str], list[dict]]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parents[3]   # project root
_WEIGHTS_FILE = _ROOT / "config" / "fusion_weights.json"

# ─── Hardcoded fallback weights (4 detectors) ─────────────────────────────────
_DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "speech":        {"vocoder": 0.45, "gan": 0.20, "diffusion": 0.15, "anomaly": 0.20},
    "music":         {"vocoder": 0.15, "gan": 0.35, "diffusion": 0.25, "anomaly": 0.25},
    "environmental": {"vocoder": 0.10, "gan": 0.25, "diffusion": 0.40, "anomaly": 0.25},
}

# ─── Flag triggers ────────────────────────────────────────────────────────────
_ARTIFACT_THRESHOLD = 0.6
_UNCERTAIN_LOW      = 0.35
_UNCERTAIN_HIGH     = 0.55


def _load_weights() -> dict[str, dict[str, float]]:
    """Load fusion weights from JSON, falling back to hardcoded defaults."""
    if _WEIGHTS_FILE.exists():
        try:
            with open(_WEIGHTS_FILE, "r") as fh:
                data = json.load(fh)
            # Strip comment keys and validate
            weights = {k: v for k, v in data.items() if not k.startswith("_")}
            required = {"speech", "music", "environmental"}
            required_det = {"vocoder", "gan", "diffusion", "anomaly"}
            if required.issubset(weights) and all(
                required_det.issubset(weights[d]) for d in required
            ):
                return weights
        except Exception:
            pass
    return _DEFAULT_WEIGHTS


# Load once at import time
_WEIGHTS = _load_weights()


def reload_weights() -> None:
    """Re-read config/fusion_weights.json (call after tune_fusion_weights.py)."""
    global _WEIGHTS
    _WEIGHTS = _load_weights()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_scores(
    vocoder_s: float,
    gan_s: float,
    diffusion_s: float,
    anomaly_s: float,
    domain_probs: dict[str, float] | str,
    *,
    domain_label: str = "",
    windowed_scores: list[tuple[float, float, float, float, float]] | None = None,
    pooling: str = "max",
    top_k: int = 3,
) -> tuple[float, list[str], list[dict]]:
    """Compute the blended final score, flags, and (optionally) top-K windows.

    Parameters
    ----------
    vocoder_s, gan_s, diffusion_s, anomaly_s:
        Individual detector scores ∈ [0, 1].
    domain_probs:
        Either a ``dict[str, float]`` probability distribution over
        ``{speech, music, environmental}`` (preferred), or a plain string
        domain label for backward compatibility.
    domain_label:
        Hard domain label used only for flag text and logging.
    windowed_scores:
        Optional list of ``(t_start, v, g, d, a)`` tuples from segment
        mode.  When provided, windowed pooling is performed and this
        function calls ``aggregate_windowed`` internally.
    pooling:
        ``"max"`` (default) or ``"mean"`` — window pooling strategy.
    top_k:
        Number of most-suspicious windows to surface in the output.

    Returns
    -------
    tuple[float, list[str], list[dict]]
        ``(final_score, flags, suspicious_windows)``
        *suspicious_windows* is a list of ``{"t_start": float, "score": float}``
        dicts, sorted by score descending.  Empty list when not in segment mode.
    """
    # ── Backward-compat: accept a plain string domain label ───────────────────
    if isinstance(domain_probs, str):
        domain_label = domain_probs if not domain_label else domain_label
        domain_probs = _label_to_uniform_probs(domain_probs)

    # ── Windowed mode ─────────────────────────────────────────────────────────
    if windowed_scores:
        return aggregate_windowed(
            iter(windowed_scores), domain_probs, pooling=pooling, top_k=top_k
        )

    # ── Blended weight computation ────────────────────────────────────────────
    w = _blend_weights(domain_probs)

    final_score = (
        w["vocoder"]   * float(vocoder_s)
        + w["gan"]     * float(gan_s)
        + w["diffusion"] * float(diffusion_s)
        + w["anomaly"] * float(anomaly_s)
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    # ── Flags ─────────────────────────────────────────────────────────────────
    flags = _build_flags(vocoder_s, gan_s, diffusion_s, anomaly_s, final_score)

    return final_score, flags, []


def aggregate_windowed(
    scores_iter: Iterator[tuple[float, float, float, float, float]],
    domain_probs: dict[str, float],
    *,
    pooling: str = "max",
    top_k: int = 3,
) -> tuple[float, list[str], list[dict]]:
    """Pool per-window scores into a single file-level result.

    Parameters
    ----------
    scores_iter:
        Generator / iterable of ``(t_start, vocoder_s, gan_s, diffusion_s,
        anomaly_s)`` tuples, consumed one at a time (memory-bounded).
    domain_probs:
        Probability distribution over ``{speech, music, environmental}``.
    pooling:
        ``"max"`` keeps the worst window's score (catches short fakes).
        ``"mean"`` averages all windows.
    top_k:
        How many top suspicious windows to return.

    Returns
    -------
    tuple[float, list[str], list[dict]]
        ``(final_score, flags, suspicious_windows)``
    """
    w = _blend_weights(domain_probs)
    window_records: list[tuple[float, float]] = []   # (t_start, blended_score)

    pooled_v, pooled_g, pooled_d, pooled_a = [], [], [], []

    for t_start, v, g, d, a in scores_iter:
        # Blended score for this window
        ws = (
            w["vocoder"]   * float(v)
            + w["gan"]     * float(g)
            + w["diffusion"] * float(d)
            + w["anomaly"] * float(a)
        )
        window_records.append((float(t_start), float(np.clip(ws, 0.0, 1.0))))
        pooled_v.append(float(v))
        pooled_g.append(float(g))
        pooled_d.append(float(d))
        pooled_a.append(float(a))

    if not window_records:
        return 0.0, [], []

    # ── Pool detector scores ──────────────────────────────────────────────────
    pool_fn = np.max if pooling == "max" else np.mean
    final_score = float(np.clip(
        w["vocoder"]   * pool_fn(pooled_v)
        + w["gan"]     * pool_fn(pooled_g)
        + w["diffusion"] * pool_fn(pooled_d)
        + w["anomaly"] * pool_fn(pooled_a),
        0.0, 1.0,
    ))

    flags = _build_flags(
        pool_fn(pooled_v), pool_fn(pooled_g),
        pool_fn(pooled_d), pool_fn(pooled_a),
        final_score,
    )

    # ── Top-K suspicious windows ──────────────────────────────────────────────
    sorted_windows = sorted(window_records, key=lambda x: x[1], reverse=True)
    suspicious = [
        {"t_start": t, "score": round(s, 4)}
        for t, s in sorted_windows[:top_k]
    ]

    return final_score, flags, suspicious


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _blend_weights(domain_probs: dict[str, float]) -> dict[str, float]:
    """Compute detector weights as a probability-weighted mixture of domain profiles."""
    detectors = ("vocoder", "gan", "diffusion", "anomaly")
    blended = {det: 0.0 for det in detectors}
    for domain, prob in domain_probs.items():
        profile = _WEIGHTS.get(domain, _WEIGHTS["environmental"])
        for det in detectors:
            blended[det] += prob * profile.get(det, 0.0)
    return blended


def _label_to_uniform_probs(label: str) -> dict[str, float]:
    """Convert a hard label string to a one-hot probability dict."""
    domains = ("speech", "music", "environmental")
    label = label if label in domains else "environmental"
    return {d: (1.0 if d == label else 0.0) for d in domains}


def _build_flags(
    vocoder_s: float,
    gan_s: float,
    diffusion_s: float,
    anomaly_s: float,
    final_score: float,
) -> list[str]:
    flags: list[str] = []
    if vocoder_s > _ARTIFACT_THRESHOLD:
        flags.append("vocoder_artifact")
    if gan_s > _ARTIFACT_THRESHOLD:
        flags.append("gan_artifact")
    if diffusion_s > _ARTIFACT_THRESHOLD:
        flags.append("diffusion_artifact")
    if anomaly_s > _ARTIFACT_THRESHOLD:
        flags.append("anomaly_artifact")
    if _UNCERTAIN_LOW < final_score < _UNCERTAIN_HIGH:
        flags.append("uncertain")
    return flags
