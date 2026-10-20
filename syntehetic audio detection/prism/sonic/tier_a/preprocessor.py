"""
PRISM — Tier A | Stage 1: Preprocessor (v2)
============================================
Loads an audio file and computes core spectral representations used by
all downstream detectors.

Public API
----------
preprocess(file_path: str | Path) -> dict
    Returns a dict with keys:
        audio, sr, stft_mag, stft_phase, mel_spec, stft_full, file_path

preprocess_windows(file_path, window_sec=4.0, hop_sec=2.0)
    Generator: yields (t_start_sec, preprocess_dict) for each window.
    Audio is loaded once; one window dict is live at a time — memory-bounded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import librosa

# ─── Constants ────────────────────────────────────────────────────────────────
SR                = 22_050   # target sample rate
DURATION          = 10.0     # max audio length for whole-file mode (seconds)
                             # 10 s chosen as the calibrated optimum:
                             #   • energy_consistency  → 100 RMS windows  (vs 50 at 5 s)
                             #   • flat_noise_floor    → ~428 STFT frames (vs ~212 at 5 s)
                             #   • phase_incoherence   → ~428 frames      (vs ~212 at 5 s)
                             #   • hf_checkerboard     → 428-pt series    (vs 212 at 5 s)
                             #   • formant_smoothness  → capped at 7 s internally (unaffected)
                             #   • anomaly detectors   → capped at 5 s/3 s internally (unaffected)
                             # Estimated cost vs 5 s: +17% query time, +6% peak RAM.
DURATION_WINDOWED = None     # preprocess_windows loads the full file (no cap)
N_FFT             = 2_048
HOP_LENGTH        = 512
N_MELS            = 128
FMIN              = 20
FMAX              = 8_000

# Pre-compiled static Mel filterbank (eliminates redundant ~16ms construction per query)
_MEL_BASIS = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)


def preprocess(
    file_path: str | Path,
    offset_sec: float | None = None,
    duration_sec: float = 4.0,
) -> dict:
    """Load *file_path* and return all spectral representations.

    Parameters
    ----------
    file_path:
        Path to any audio format supported by librosa (wav, mp3, flac, ogg …).
    offset_sec:
        Optional float offset in seconds to start reading from. If None,
        automatically bypasses lead-in silence/intro for tracks longer than 4.5s.
    duration_sec:
        Length of audio window to load (default 4.0s for calibrated uniform model).

    Returns
    -------
    dict
        ``audio``      – mono waveform, float32, shape (N,)
        ``sr``         – sample rate (22 050)
        ``stft_full``  – complex STFT, shape (N_FFT//2+1, T)
        ``stft_mag``   – STFT magnitude, shape (N_FFT//2+1, T)
        ``stft_phase`` – STFT phase in radians, shape (N_FFT//2+1, T)
        ``mel_spec``   – mel-spectrogram in dB, shape (N_MELS, T)
        ``file_path``  – str path of the source file
    """
    # Fast load & normalise (soundfile + soxr with fallback to librosa)
    start_offset = 0.0 if offset_sec is None else max(0.0, float(offset_sec))
    load_dur = max(1.0, float(duration_sec))
    try:
        import soundfile as sf
        import soxr
        with sf.SoundFile(str(file_path)) as sfile:
            orig_sr = sfile.samplerate
            tot_frames = sfile.frames
            tot_dur = tot_frames / orig_sr

            # Automatic active region detection if offset_sec not explicitly given
            if offset_sec is None and tot_dur > 4.5:
                search_dur = min(tot_dur, 6.0)
                search_frames = int(search_dur * orig_sr)
                probe_data = sfile.read(frames=search_frames, dtype="float32")
                if probe_data.ndim > 1:
                    probe_data = np.mean(probe_data, axis=1)
                hop = int(1.0 * orig_sr)
                win = int(min(load_dur, tot_dur) * orig_sr)
                best_rms, best_off = -1.0, 0.0
                for st in range(0, len(probe_data) - win + 1, hop):
                    chunk = probe_data[st : st + win]
                    rms = float(np.mean(chunk ** 2))
                    if rms > best_rms:
                        best_rms = rms
                        best_off = st / orig_sr
                start_offset = best_off

            start_frame = max(0, min(int(start_offset * orig_sr), max(0, tot_frames - 1)))
            sfile.seek(start_frame)

            max_frames = int(load_dur * orig_sr)
            data = sfile.read(frames=max_frames, dtype="float32")
            if len(data) == 0:
                raise ValueError("No frames read via soundfile")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if orig_sr != SR:
            data = soxr.resample(data, orig_sr, SR, quality="soxr_mq")
        y = data[:int(load_dur * SR)]
        # Peak normalize to 0.95
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-4:
            y = (y / peak) * 0.95
        sr = SR
    except Exception:
        try:
            y, sr = librosa.load(file_path, sr=SR, mono=True, offset=start_offset, duration=load_dur)
        except Exception:
            try:
                import subprocess
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = [
                    ffmpeg_exe, "-v", "error", "-ss", str(start_offset), "-t", str(load_dur),
                    "-i", str(file_path), "-ar", str(SR), "-ac", "1", "-f", "s16le", "-"
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if len(proc.stdout) > 0:
                    y = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = SR
                else:
                    raise ValueError(f"Failed to decode audio via ffmpeg: {file_path}")
            except Exception as ffmpeg_err:
                raise ValueError(f"Could not load audio file {file_path}: {ffmpeg_err}")

        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-4:
            y = (y / peak) * 0.95

    if len(y) == 0:
        raise ValueError(f"Audio file appears to be empty: {file_path}")

    return _build_preprocess_dict(y, sr, str(file_path))


def preprocess_windows(
    file_path: str | Path,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
) -> Generator[Tuple[float, dict], None, None]:
    """Stream sliding-window preprocess dicts from an audio file.

    Audio is loaded **once** in its entirety; spectral representations
    are computed for **one window at a time** and yielded immediately.
    At most one window's STFT/mel-spec is held in memory simultaneously,
    satisfying the memory-bounded constraint.

    Parameters
    ----------
    file_path:
        Path to the audio file.
    window_sec:
        Duration of each window in seconds (default 4.0 s).
    hop_sec:
        Step between window starts in seconds (default 2.0 s — 50% overlap).

    Yields
    ------
    tuple[float, dict]
        ``(t_start_seconds, preprocess_dict)`` — one per window.
        *t_start_seconds* is the offset of the window start from the
        beginning of the file.
    """
    # Load full audio once (waveform only — not all spectrograms)
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    if len(y) == 0:
        return

    window_samples = int(window_sec * sr)
    hop_samples    = int(hop_sec   * sr)

    # Need at least half a window to be worth scoring
    min_window = window_samples // 2

    for start in range(0, len(y), hop_samples):
        frame = y[start : start + window_samples]

        if len(frame) < min_window:
            break   # remaining audio too short

        # Zero-pad the last (partial) window so all dicts have the same shape
        if len(frame) < window_samples:
            frame = np.pad(frame, (0, window_samples - len(frame)), mode="constant")

        t_start = start / sr
        yield t_start, _build_preprocess_dict(frame, sr, str(file_path))


# ─── Internal helper ──────────────────────────────────────────────────────────

def _build_preprocess_dict(y: np.ndarray, sr: int, file_path: str) -> dict:
    """Compute all spectral representations from a waveform array."""
    D: np.ndarray = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag   = np.abs(D)
    phase = np.angle(D)

    # Ultra-fast dot product with pre-compiled mel filterbank (~1.4ms vs 16.2ms)
    mel_power = np.dot(_MEL_BASIS, mag**2)
    ref = float(np.max(mel_power)) if mel_power.size > 0 else 1.0
    mel_db = (10.0 * np.log10(np.maximum(1e-10, mel_power / (ref + 1e-12)))).astype(np.float32)

    return {
        "audio":      y,
        "sr":         sr,
        "stft_full":  D,
        "stft_mag":   mag,
        "stft_phase": phase,
        "mel_spec":   mel_db,
        "file_path":  file_path,
    }
