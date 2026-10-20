"""
PRISM — Tier A | Stage 3.5: Classical DSP Anomaly Detector
===========================================================
A 4th detector that is completely generator-agnostic.  It computes a
~34-float DSP feature vector from Librosa primitives and scores the
sample against a per-domain one-class IsolationForest trained on real
audio only.

No neural networks, no pretrained checkpoints, no heavy dependencies.

Public API
----------
compute_dsp_features(p: dict) -> np.ndarray
    Extract the feature vector from a preprocess dict.

artifact_anomaly_score(p: dict, domain: str) -> float
    Returns a score in [0.0, 1.0] where 1.0 = strongly anomalous.
    Falls back to 0.5 if the domain model pkl is not found.

train_anomaly_model(features: np.ndarray) -> dict
    Train an IsolationForest on real-only features; returns a model_data
    dict that can be passed to save_anomaly_model().

save_anomaly_model(model_data: dict, domain: str) -> Path
    Persist the model_data dict to models/anomaly_{domain}.pkl.

FEATURE_NAMES : list[str]
    Ordered feature names produced by compute_dsp_features().

DSP_FEATURE_COLUMNS : list[str]
    Same names prefixed with "dsp_" — used as CSV column headers.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa

# ─── Constants ────────────────────────────────────────────────────────────────
_SR         = 22_050
_N_FFT      = 2_048
_HOP_LENGTH = 512
_N_MFCC     = 13
_LPC_ORDER  = 8

# models/ directory is 3 parents above this file's directory:
#   tier_a/ → sonic/ → prism/ → <project_root>/
_MODELS_DIR = Path(__file__).parents[3] / "models"

# In-process model cache so we don't re-load pkl on every call
_MODEL_CACHE: Dict[str, object] = {}


# ─── Feature metadata ─────────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = (
    [f"mfcc{i}_mean" for i in range(_N_MFCC)]
    + [f"mfcc{i}_var"  for i in range(_N_MFCC)]
    + [
        "spectral_flatness_mean",
        "spectral_flatness_var",
        "spectral_centroid_mean",
        "zcr_var",
        "ltp_residual_ratio",
        "hnr_db",
        "pitch_jitter",
        "pitch_shimmer",
    ]
)

# Column names used when writing these features to a CSV
DSP_FEATURE_COLUMNS: list[str] = [f"dsp_{n}" for n in FEATURE_NAMES]


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dsp_features(p: dict) -> np.ndarray:
    """Extract the DSP feature vector from a preprocess dict.

    All computations use Librosa and NumPy — no pretrained models.

    Parameters
    ----------
    p : dict
        Dict returned by :func:`preprocessor.preprocess`.

    Returns
    -------
    np.ndarray, shape (len(FEATURE_NAMES),), dtype float32
    """
    y  = p["audio"]
    sr = p["sr"]
    mag = p.get("stft_mag")
    mel = p.get("mel_spec")

    feats: list[float] = []

    # ── MFCC means and variances (reusing precomputed Mel-Spec if available) ──
    if mel is not None:
        import scipy.fftpack
        mfcc = scipy.fftpack.dct(mel, axis=0, type=2, norm="ortho")[:_N_MFCC]
    else:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC, hop_length=_HOP_LENGTH)
    feats.extend(float(np.mean(mfcc[i])) for i in range(_N_MFCC))
    feats.extend(float(np.var(mfcc[i]))  for i in range(_N_MFCC))

    # ── Spectral flatness (reusing precomputed STFT mag if available) ─────────
    if mag is not None:
        flatness = librosa.feature.spectral_flatness(S=mag)
    else:
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
    feats.append(float(np.mean(flatness)))
    feats.append(float(np.var(flatness)))

    # ── Spectral centroid mean (reusing precomputed STFT mag if available) ───
    if mag is not None:
        centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)
    else:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
    feats.append(float(np.mean(centroid)))

    # ── Zero-crossing rate variance ───────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=_HOP_LENGTH)
    feats.append(float(np.var(zcr)))

    # ── LTP residual energy ratio ─────────────────────────────────────────────
    feats.append(_ltp_residual_ratio(y, sr))

    # ── Harmonic-to-noise ratio (dB) ──────────────────────────────────────────
    feats.append(_harmonic_to_noise_ratio(y))

    # ── Pitch jitter & shimmer ────────────────────────────────────────────────
    jitter, shimmer = _pitch_jitter_shimmer(y, sr)
    feats.append(jitter)
    feats.append(shimmer)

    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════════════

def artifact_anomaly_score(p: dict, domain: str, features: Optional[np.ndarray] = None) -> float:
    """Score how anomalous the audio is vs. real audio of the same domain.

    Parameters
    ----------
    p : dict
        Preprocess dict (output of :func:`preprocessor.preprocess`).
    domain : str
        Hard domain label — ``"speech"``, ``"music"``, or ``"environmental"``.
    features : Optional[np.ndarray]
        Precomputed DSP feature vector (optional, avoids recomputing).

    Returns
    -------
    float
        Anomaly score in [0.0, 1.0].  Higher = more likely synthetic.
        Returns 0.5 (neutral) if the per-domain model pkl is not found.
    """
    model_path = _MODELS_DIR / f"anomaly_{domain}.pkl"
    model_data = _load_model(model_path)
    if model_data is None:
        return 0.5

    if features is None:
        features = compute_dsp_features(p).reshape(1, -1)
    elif features.ndim == 1:
        features = features.reshape(1, -1)

    try:
        raw = float(model_data["model"].decision_function(features)[0])
        # Smooth sigmoid mapping centered at 0.0 (raw > 0 is inlier/real, raw < 0 is outlier)
        # Prevents arbitrary clipping to 1.0 on natural audio from outside the training split
        score = 1.0 / (1.0 + np.exp(raw * 15.0))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Training helpers (called from train_tier_b.py)
# ═══════════════════════════════════════════════════════════════════════════════

def train_anomaly_model(
    features: np.ndarray,
    *,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Train a per-domain IsolationForest on real-audio features.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        DSP feature matrix for **real** audio only.

    Returns
    -------
    dict
        Keys: ``"model"``, ``"score_min"``, ``"score_max"``.
        Pass this dict to :func:`save_anomaly_model`.
    """
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    clf.fit(features)

    scores = clf.decision_function(features)
    return {
        "model":     clf,
        "score_min": float(np.percentile(scores, 1)),
        "score_max": float(np.percentile(scores, 99)),
    }


def save_anomaly_model(model_data: dict, domain: str) -> Path:
    """Persist a model_data dict to ``models/anomaly_{domain}.pkl``.

    Parameters
    ----------
    model_data : dict
        Return value of :func:`train_anomaly_model`.
    domain : str
        One of ``"speech"``, ``"music"``, ``"environmental"``.

    Returns
    -------
    Path
        Location of the written file.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODELS_DIR / f"anomaly_{domain}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model_data, fh)
    _MODEL_CACHE[str(path)] = model_data  # update in-process cache
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal DSP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(path: Path) -> Optional[dict]:
    key = str(path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        _MODEL_CACHE[key] = data
        return data
    except Exception:
        return None


def _ltp_residual_ratio(y: np.ndarray, sr: int) -> float:
    """Long-Term Prediction residual energy ratio.

    Measures how much energy in each short frame is *not* predictable by
    linear prediction. Natural audio has more residual energy than
    over-smooth synthetic audio.
    """
    # Optimized: slice 2.0s with 20ms hop for fast evaluation
    y_slice   = y[:int(2.0 * sr)]
    frame_len = int(0.025 * sr)   # 25 ms
    hop       = int(0.020 * sr)   # 20 ms
    residuals: list[float] = []

    for start in range(0, len(y_slice) - frame_len, hop):
        frame   = y_slice[start : start + frame_len]
        sig_var = float(np.var(frame))
        if sig_var < 1e-10:
            continue
        try:
            coeffs  = librosa.lpc(frame, order=_LPC_ORDER)
            pred    = np.convolve(-coeffs[1:], frame)[: len(frame)]
            res_var = float(np.var(frame - pred))
            residuals.append(res_var / sig_var)
        except Exception:
            continue

    return float(np.clip(np.mean(residuals), 0.0, 1.0)) if residuals else 0.5


def _harmonic_to_noise_ratio(y: np.ndarray) -> float:
    """Estimate HNR in dB via fast FFT-based Autocorrelation (Praat Standard).

    Range is clamped to [–20, 40] dB. Sub-millisecond compute.
    """
    try:
        y_slice = y[:int(2.0 * 22050)]
        n = len(y_slice)
        if n < 512:
            return 0.0
        r = np.fft.irfft(np.abs(np.fft.rfft(y_slice, n=2*n))**2)[:n]
        r0 = float(r[0])
        if r0 < 1e-9:
            return 0.0
        min_lag = int(22050 / 1000)
        max_lag = int(22050 / 65)
        rmax = float(np.max(r[min_lag:max_lag]))
        if r0 > rmax and (r0 - rmax) > 1e-9 and rmax > 0:
            hnr = 10.0 * np.log10(rmax / (r0 - rmax))
            return float(np.clip(hnr, -20.0, 40.0))
        return 0.0
    except Exception:
        return 0.0


def _pitch_jitter_shimmer(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate pitch jitter (cycle-to-cycle F0 variation) and shimmer
    (amplitude variation in voiced frames) via fast FFT autocorrelation pitch tracking.

    Both are returned as a coefficient of variation (std / mean), clamped
    to [0, 2]. Executes in < 1ms (1000x faster than librosa.pyin).
    """
    try:
        max_samples = int(2.0 * sr)
        y_slice = y[:max_samples]
        hop = _HOP_LENGTH
        frame_len = 1024
        min_lag = int(sr / 1000)
        max_lag = int(sr / 65)

        f0_list: list[float] = []
        rms_list: list[float] = []

        for start in range(0, len(y_slice) - frame_len, hop):
            frame = y_slice[start : start + frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms < 0.005:
                continue
            r = np.fft.irfft(np.abs(np.fft.rfft(frame, n=2*frame_len))**2)[:frame_len]
            r0 = float(r[0])
            if r0 < 1e-5:
                continue
            r_search = r[min_lag:max_lag]
            peak_rel = int(np.argmax(r_search))
            peak_val = float(r_search[peak_rel])
            # Harmonic voiced condition
            if peak_val / r0 > 0.30:
                f0 = float(sr / (min_lag + peak_rel))
                f0_list.append(f0)
                rms_list.append(rms)

        if len(f0_list) < 2:
            return 0.0, 0.0

        mean_f0 = float(np.mean(f0_list))
        jitter = float(np.std(f0_list) / mean_f0) if mean_f0 > 1e-6 else 0.0

        mean_rms = float(np.mean(rms_list))
        shimmer = float(np.std(rms_list) / mean_rms) if mean_rms > 1e-6 else 0.0

        return float(np.clip(jitter, 0.0, 2.0)), float(np.clip(shimmer, 0.0, 2.0))
    except Exception:
        return 0.0, 0.0
