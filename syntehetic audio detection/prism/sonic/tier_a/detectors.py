"""
PRISM — Tier A | Stage 3: Artifact Detectors
==============================================
Three parallel heuristic detectors, each returning a float in [0, 1].
Higher score = more likely synthetic.

Public API
----------
vocoder_score(p: dict, domain: str) -> float
    Targets HiFi-GAN, WaveGlow, WaveNet-style vocoders.

gan_score(p: dict) -> float
    Targets MelGAN, EnCodec, GAN-based music synthesisers.

diffusion_score(p: dict) -> float
    Targets AudioLDM, Stable Audio, diffusion-based synthesisers.
"""

from __future__ import annotations

import numpy as np
import librosa

# ─── Shared constants ────────────────────────────────────────────────────────
_SR         = 22_050
_N_FFT      = 2_048
_HOP_LENGTH = 512

# ─── Empirical normalisation thresholds ──────────────────────────────────────
_BANDING_VAR_THRESHOLD    = 3.0    # dB variance between adjacent mel bands
_PHASE_INCOH_THRESHOLD    = 0.8    # rad/s
_NOISE_FLOOR_VAR_THRESH   = 0.5    # energy variance for diffusion floor test
_ENERGY_COV_REAL_MIN      = 0.4    # real audio CoV lower bound
_ENERGY_COV_SYNTH_MAX     = 0.2    # diffusion CoV upper bound


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR A — Vocoder
# ═══════════════════════════════════════════════════════════════════════════════

def vocoder_score(p: dict, domain: str = "general") -> float:
    """Detect vocoder (HiFi-GAN / WaveGlow / WaveNet) artifacts.

    Parameters
    ----------
    p:
        Preprocess dict.
    domain:
        Domain string from the classifier ("speech" | "music" | "environmental").

    Returns
    -------
    float
        Synthetic probability in [0, 1].
    """
    banding_thresh = 0.15 if domain == "speech" else (0.35 if domain == "music" else 0.25)
    banding = _spectral_banding_index(p, threshold=banding_thresh)

    if domain == "speech":
        formant = _formant_smoothness(p)
        score   = 0.5 * banding + 0.5 * formant
    else:
        score = banding

    return float(np.clip(score, 0.0, 1.0))


def _spectral_banding_index(p: dict, threshold: float = 0.25) -> float:
    """Measure regular variance between adjacent mel-band means normalized by dynamic range.

    Vocoders produce an unnaturally high mel-band variance relative to overall energy spread.
    Normalizing by band_means variance prevents polyphonic music from saturating the detector.
    """
    mel_db: np.ndarray = p["mel_spec"]          # (N_MELS, T)
    band_means = mel_db.mean(axis=1)            # (N_MELS,)
    diffs      = np.diff(band_means)            # (N_MELS-1,)
    var_diffs  = float(np.var(diffs))
    var_bm     = float(np.var(band_means)) + 1e-6
    ratio      = var_diffs / var_bm
    return float(np.clip(ratio / threshold, 0.0, 1.0))


def _formant_smoothness(p: dict) -> float:
    """LPC-based formant trajectory smoothness (speech only).

    Real speech has large frame-to-frame formant deltas (std > 40 Hz).
    Vocoders produce over-smooth formant tracks (std < 20 Hz).

    Score = max(0, (40 - delta_std) / 40)
    """
    y  = p["audio"]
    sr = p["sr"]

    frame_length  = int(0.025 * sr)   # 25 ms
    hop           = _HOP_LENGTH
    lpc_order     = 12

    # Collect first-formant estimates across frames
    formant_tracks: list[list[float]] = [[], [], []]

    for start in range(0, len(y) - frame_length, hop):
        frame = y[start : start + frame_length]
        frame = frame * np.hanning(len(frame))  # window

        try:
            lpc_coeffs = librosa.lpc(frame, order=lpc_order)
        except Exception:
            continue

        # Roots of the LPC polynomial → resonant frequencies
        roots = np.roots(lpc_coeffs)
        # Keep roots in upper half of z-plane (positive imaginary part)
        roots = roots[np.imag(roots) >= 0]
        angles = np.angle(roots)
        freqs  = sorted(angles * (sr / (2 * np.pi)))
        freqs  = [f for f in freqs if 200 < f < 4000]

        for i in range(3):
            if i < len(freqs):
                formant_tracks[i].append(freqs[i])

    if not formant_tracks[0]:
        # Fallback: cannot compute formants → assume vocoder
        return 0.7

    delta_stds: list[float] = []
    for track in formant_tracks:
        if len(track) > 1:
            deltas = np.diff(track)
            delta_stds.append(float(np.std(deltas)))

    if not delta_stds:
        return 0.7

    mean_delta_std = float(np.mean(delta_stds))
    score = max(0.0, (40.0 - mean_delta_std) / 40.0)
    return float(np.clip(score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR B — GAN
# ═══════════════════════════════════════════════════════════════════════════════

def gan_score(p: dict, domain: str = "general") -> float:
    """Detect GAN (MelGAN / EnCodec) artifacts.

    Parameters
    ----------
    p:
        Preprocess dict.
    domain:
        Audio domain ("speech" | "music" | "environmental" | "general").

    Returns
    -------
    float
        Synthetic probability in [0, 1].
    """
    checkerboard = _hf_checkerboard(p)
    phase_thresh = 0.55 if domain == "speech" else (0.75 if domain == "music" else 0.65)
    phase_incoh  = _phase_incoherence(p, threshold=phase_thresh)
    score        = 0.5 * checkerboard + 0.5 * phase_incoh
    return float(np.clip(score, 0.0, 1.0))


def _hf_checkerboard(p: dict) -> float:
    """Detect regular high-frequency checkerboard patterns (GAN upsampling).

    Monotonic autocorrelation decay is natural for real audio. A true checkerboard
    pattern produces periodic secondary peaks or alternating negative polarity.
    """
    mag: np.ndarray = p["stft_mag"]    # (F, T)
    sr  = p["sr"]

    hf_bin = max(1, int(8_000 / (sr / 2) * (mag.shape[0] - 1)))
    hf_subband = mag[hf_bin:, :]       # (F_hf, T)

    if hf_subband.size == 0:
        return 0.0

    hf_energy = hf_subband.mean(axis=0)   # (T,)
    hf_energy -= hf_energy.mean()

    if len(hf_energy) < 6:
        return 0.0

    n  = len(hf_energy)
    ac = np.correlate(hf_energy, hf_energy, mode="full")
    ac = ac[n - 1:]                   # positive lags
    ac = ac / (ac[0] + 1e-10)        # normalise

    # Check for checkerboard oscillation / non-monotonic secondary peaks
    sec_peaks = [max(0.0, ac[k] - ac[k-1]) for k in range(2, min(len(ac), 5))]
    cb = max(sec_peaks) * 3.0 if sec_peaks else 0.0
    if ac[1] < 0:
        cb = max(cb, -ac[1] * 2.0)
    return float(np.clip(cb, 0.0, 1.0))


def _phase_incoherence(p: dict, threshold: float = 0.65) -> float:
    """Measure temporal phase irregularity (phase acceleration/jerk) on active harmonics.

    Natural audio exhibits smooth instantaneous frequency trajectories across consecutive frames.
    Synthetic audio introduces erratic frame-to-frame phase discontinuities.
    """
    mag: np.ndarray = p["stft_mag"]
    phase = p.get("stft_phase")
    if phase is None:
        phase = np.angle(p["stft_full"][:, :mag.shape[1]])
    else:
        phase = phase[:, :mag.shape[1]]

    # 2nd derivative of phase wrapped to [-pi, pi]; mathematically identical to unwrap + diff(diff)
    d2 = np.diff(phase, n=2, axis=1)
    dIF_wrap = np.abs(np.angle(np.exp(1j * d2)))

    mag_active = mag[:, 2:]
    thresh = np.percentile(mag_active, 85)
    mask = mag_active >= thresh
    if np.sum(mask) == 0:
        return 0.2
    val = float(np.mean(dIF_wrap[mask]))
    return float(np.clip((val - 0.70) / max(threshold, 1e-6), 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR C — Diffusion
# ═══════════════════════════════════════════════════════════════════════════════

def diffusion_score(p: dict) -> float:
    """Detect diffusion-model (AudioLDM / Stable Audio) artifacts.

    Parameters
    ----------
    p:
        Preprocess dict.

    Returns
    -------
    float
        Synthetic probability in [0, 1].
    """
    flat_noise  = _flat_noise_floor(p)
    energy_cons = _energy_consistency(p)
    score       = 0.5 * flat_noise + 0.5 * energy_cons
    return float(np.clip(score, 0.0, 1.0))


def _flat_noise_floor(p: dict) -> float:
    """Detect unnaturally flat noise floor in the lowest mel bands.

    Diffusion models tend to generate very consistent background noise,
    unlike real recordings where the noise floor fluctuates with room
    acoustics and microphone noise.
    """
    mel_db: np.ndarray = p["mel_spec"]   # (N_MELS, T)

    # Lowest 10 mel bands
    n_low = min(10, mel_db.shape[0])
    low_bands = mel_db[:n_low, :]        # (10, T)

    # Noise floor estimate = median energy per frame across low bands
    noise_floor = np.median(low_bands, axis=0)   # (T,)
    variance    = float(np.var(noise_floor))

    score = max(0.0, 1.0 - variance / _NOISE_FLOOR_VAR_THRESH)
    return float(np.clip(score, 0.0, 1.0))


def _energy_consistency(p: dict) -> float:
    """Detect over-smooth RMS energy envelope (diffusion signature).

    Real recordings have highly variable energy (CoV > 0.4).
    Diffusion models tend to produce unnaturally steady output (CoV < 0.2).

    Score = max(0, (0.4 - CoV) / 0.4)
    """
    y  = p["audio"]
    sr = p["sr"]

    window_samples = int(0.100 * sr)   # 100 ms windows
    hop            = window_samples    # non-overlapping

    rms_values: list[float] = []
    for start in range(0, len(y) - window_samples + 1, hop):
        frame = y[start : start + window_samples]
        rms_values.append(float(np.sqrt(np.mean(frame ** 2))))

    if len(rms_values) < 2:
        return 0.0

    rms_arr = np.array(rms_values)
    mean    = float(np.mean(rms_arr))
    if mean < 1e-10:
        return 0.0

    cov   = float(np.std(rms_arr) / mean)
    score = max(0.0, (_ENERGY_COV_REAL_MIN - cov) / _ENERGY_COV_REAL_MIN)
    return float(np.clip(score, 0.0, 1.0))
