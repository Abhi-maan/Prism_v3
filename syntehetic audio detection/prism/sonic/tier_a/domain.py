"""
PRISM — Tier A | Stage 2: Domain Classifier (Soft-Probability Edition)
=======================================================================
Rule-based spectral domain classifier.  No ML — purely heuristic.

Public API
----------
classify_domain(preprocess_dict: dict) -> tuple[str, dict[str, float]]
    Returns (hard_label, probability_distribution).

    hard_label  : "speech" | "music" | "environmental"   (for logging/UI)
    probability_distribution :
        {"speech": float, "music": float, "environmental": float}
        Values are non-negative and sum to 1.0.

    The probability distribution is derived by converting each feature's
    distance from each domain threshold into a soft confidence score, then
    normalising.  Soft routing allows ``aggregator.py`` to blend weight
    profiles proportionally instead of hard-switching, so a misclassified
    or mixed-content file no longer fully commits to the wrong profile.

classify_domain_label(preprocess_dict: dict) -> str
    Convenience wrapper — returns only the hard label (backward-compat).

Decision features
-----------------
  ZCR_mean              : speech has higher ZCR than music/env
  spectral_centroid_mean: speech peaks in a lower Hz range
  spectral_rolloff_mean : music extends to higher frequencies
  MFCC[1] variance      : music has more spectral variation over time
"""

from __future__ import annotations

import numpy as np
import librosa

# ─── Thresholds ───────────────────────────────────────────────────────────────
# Calibrated for wideband 22kHz audio (LibriSpeech / WaveFake / LJSpeech).
# LibriSpeech measured: ZCR≈0.04-0.10, centroid≈1200-2200Hz, rolloff≈2400-3500Hz
ZCR_SPEECH_THRESH         = 0.04    # speech ZCR lower bound
CENTROID_SPEECH_THRESH_HZ = 3_500.0 # speech centroid upper bound (Hz)
ROLLOFF_MUSIC_THRESH_HZ   = 4_000.0 # music rolloff lower bound (Hz)
MFCC_VAR_MUSIC_THRESH     = 50.0    # music MFCC[1] variance lower bound
ROLLOFF_NARROWBAND_THRESH = 7_000.0 # narrowband/telephony rolloff upper bound

_N_MFCC     = 13
_HOP_LENGTH = 512
_N_FFT      = 2_048


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def classify_domain(p: dict) -> tuple[str, dict[str, float]]:
    """Classify the audio domain with a soft probability distribution.

    Parameters
    ----------
    p : dict
        Dict returned by :func:`preprocessor.preprocess`.

    Returns
    -------
    tuple[str, dict[str, float]]
        ``(hard_label, probs)`` where *hard_label* is ``"speech"``,
        ``"music"``, or ``"environmental"`` and *probs* is a dict of
        floats summing to 1.0.
    """
    y  = p["audio"]
    sr = p["sr"]

    mag = p.get("stft_mag")
    mel = p.get("mel_spec")

    # ── Feature extraction (reusing precomputed STFT & Mel representations) ──
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=_HOP_LENGTH)
    zcr_mean: float = float(np.mean(zcr))

    if mag is not None and mag.shape[0] > 0:
        centroid_mean: float = float(np.mean(librosa.feature.spectral_centroid(S=mag, sr=sr)))
        # Vectorized fast 85% spectral rolloff (~0.2ms vs 12ms)
        cumsum_roll = np.cumsum(mag, axis=0)
        idx_roll = np.argmax(cumsum_roll >= (0.85 * cumsum_roll[-1:, :]), axis=0)
        freq_grid = np.linspace(0, sr / 2, mag.shape[0])
        rolloff_mean: float = float(np.mean(freq_grid[idx_roll]))
        flatness_mean: float = float(np.mean(librosa.feature.spectral_flatness(S=mag)))
        freqs = np.linspace(0, sr / 2, mag.shape[0])
        tot_e = float(np.sum(mag ** 2)) + 1e-9
        sub_bass = float(np.sum(mag[freqs < 180] ** 2) / tot_e)
    else:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        centroid_mean = float(np.mean(centroid))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        rolloff_mean = float(np.mean(rolloff))
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        flatness_mean = float(np.mean(flatness))
        sub_bass = 0.0

    if mel is not None and mel.shape[0] > 1:
        import scipy.fftpack
        mfcc = scipy.fftpack.dct(mel, axis=0, type=2, norm="ortho")[:2]
        mfcc1_var: float = float(np.var(mfcc[1]))
    else:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC, hop_length=_HOP_LENGTH)
        mfcc1_var = float(np.var(mfcc[1]))

    # ── Soft confidence scores (non-negative) ─────────────────────────────────
    probs = _compute_probs(zcr_mean, centroid_mean, rolloff_mean, mfcc1_var, flatness_mean, sub_bass=sub_bass)
    hard_label = max(probs, key=probs.__getitem__)

    return hard_label, probs


def classify_domain_label(p: dict) -> str:
    """Return only the hard domain label (backward-compatibility wrapper)."""
    label, _ = classify_domain(p)
    return label


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_probs(
    zcr_mean: float,
    centroid_mean: float,
    rolloff_mean: float,
    mfcc1_var: float,
    flatness_mean: float = 0.01,
    sub_bass: float = 0.0,
) -> dict[str, float]:
    """Convert acoustic features into a normalized probability distribution."""
    # Environmental confidence is proportional to spectral flatness
    env_raw = float(np.clip((flatness_mean - 0.02) / 0.12, 0.0, 1.0))

    # Speech vs Music
    # Human speech exhibits high MFCC1 temporal variance (alternating vowels/consonants/pauses) and vocal formant energy.
    # Music exhibits continuous instrumentation, polyphonic harmony, and heavier sub-bass (>0.20).
    if mfcc1_var >= 1000.0 and centroid_mean >= 1400.0:
        speech_raw, music_raw = 0.88, 0.12
    elif sub_bass >= 0.20:
        music_raw, speech_raw = 0.88, 0.12
    elif mfcc1_var >= 2000.0:
        speech_raw, music_raw = 0.88, 0.12
    elif zcr_mean >= 0.10 and centroid_mean >= 1500.0 and sub_bass < 0.15:
        speech_raw, music_raw = 0.85, 0.15
    else:
        music_raw, speech_raw = 0.85, 0.15

    if env_raw > 0.4:
        speech_raw *= 0.25
        music_raw *= 0.25
    else:
        env_raw = 0.08

    speech_raw = max(speech_raw, 0.05)
    music_raw  = max(music_raw,  0.05)
    env_raw    = max(env_raw,    0.05)

    total = speech_raw + music_raw + env_raw
    if total < 1e-9:
        return {"speech": 1 / 3, "music": 1 / 3, "environmental": 1 / 3}

    return {
        "speech":        speech_raw / total,
        "music":         music_raw  / total,
        "environmental": env_raw    / total,
    }
