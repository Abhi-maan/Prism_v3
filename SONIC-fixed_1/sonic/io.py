"""Shared audio loading used identically for indexing and querying, so preprocessing
never diverges between the two paths.

Deliberately implemented on soundfile + numpy only, NOT librosa: librosa
transitively requires numba and scikit-learn, and profiling showed that the
first real use of librosa.load/librosa.stft lazily JIT-compiles librosa's
internal numba-accelerated routines, adding a one-time ~100-150MB RSS tax to
the process regardless of dataset size - enough on its own to blow the
<120MB peak-RAM budget. soundfile (libsndfile) decodes wav/mp3 directly with a
tiny footprint, and resampling is done with a plain FFT resample (the same
algorithm scipy.signal.resample uses) so no extra heavy compiled dependency is
pulled into the serving path. librosa is still used offline, in
augment/modifications.py only, for pitch_shift/time_stretch test-data
generation - that is a benchmark-authoring tool, not part of the deployed
retrieval system, so its footprint doesn't count against the serving budget.
"""

import numpy as np
import soundfile as sf

from sonic.config import ANALYSIS_SR


def _next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _resample_fft(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """FFT resample (same algorithm as scipy.signal.resample), padded to a
    multiple of orig_sr before transforming. Since standard audio sample rates
    (16000, 22050, 44100) are highly composite numbers, padding to a multiple
    of orig_sr guarantees that both the input and output padded lengths are
    highly composite, keeping numpy's FFT on the fast O(n log n) path while
    mathematically preserving the exact frequency ratio.
    """
    if orig_sr == target_sr or y.size == 0:
        return y.astype(np.float32)
    n = y.size
    new_n = max(1, int(round(n * target_sr / orig_sr)))

    n_pad_in = int(np.ceil(n / orig_sr) * orig_sr)
    y_padded = np.pad(y, (0, n_pad_in - n))
    Y = np.fft.rfft(y_padded)

    n_pad_out = int(round(n_pad_in * target_sr / orig_sr))
    n_freq_new = n_pad_out // 2 + 1

    if n_freq_new <= Y.size:
        Y2 = Y[:n_freq_new]
    else:
        Y2 = np.zeros(n_freq_new, dtype=Y.dtype)
        Y2[: Y.size] = Y

    y2_padded = np.fft.irfft(Y2, n=n_pad_out)
    y2_padded *= n_pad_out / n_pad_in
    return y2_padded[:new_n].astype(np.float32)


def load_audio(path: str, sr: int = ANALYSIS_SR, max_duration: float | None = None,
                offset: float = 0.0) -> np.ndarray:
    """Load an audio file as mono float32 at the shared analysis sample rate,
    peak-normalized to [-1, 1]. Returns an empty array for zero-length input.

    `max_duration`/`offset` (seconds) are converted to a native-sample-rate
    frame range and passed to soundfile so the decoder never reads more of the
    file than needed - this is what keeps query-time latency bounded for the
    long tail of long recordings (see fingerprint/landmarks.py's
    QUERY_MAX_DURATION_SEC). Indexing always uses the full file
    (max_duration=None) so a bounded query can still match any position within
    a long indexed track.
    """
    try:
        info = sf.info(path)
        native_sr = info.samplerate
        start_frame = int(offset * native_sr)
        frames = int(max_duration * native_sr) if max_duration is not None else -1

        y, _ = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)

        y = _resample_fft(y, native_sr, sr)
    except Exception:
        try:
            import librosa
            y, _ = librosa.load(path, sr=sr, mono=True, offset=offset)
            if max_duration is not None:
                y = y[: int(max_duration * sr)]
        except Exception:
            y = np.array([], dtype=np.float32)

    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak
    return y


def trim_leading_silence(y: np.ndarray, sr: int, rel_threshold: float = 0.15,
                          frame_sec: float = 0.1) -> int:
    """Returns the sample index where content starts (first frame whose RMS
    exceeds rel_threshold * the array's peak frame RMS), or 0 if the whole
    array is already active or too short to frame. Used to skip a quiet/pause
    opening before taking a bounded query window - see query_window() below."""
    frame_len = max(1, int(frame_sec * sr))
    n_frames = y.size // frame_len
    if n_frames < 2:
        return 0
    frames = y[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak_rms = rms.max()
    if peak_rms <= 1e-8:
        return 0
    above = np.nonzero(rms > rel_threshold * peak_rms)[0]
    if above.size == 0:
        return 0
    return int(above[0]) * frame_len


QUERY_SILENCE_REL_DB = -45.0
QUERY_SILENCE_MIN_SKIP_SEC = 0.3
QUERY_SILENCE_LOOKAHEAD_SEC = 30.0


def _starts_silent(y: np.ndarray, sr: int, rel: float, frame_sec: float = 0.1) -> bool:
    frame_len = max(1, int(frame_sec * sr))
    if y.size < frame_len:
        return False
    n = y.size // frame_len
    rms = np.sqrt(np.mean(y[: n * frame_len].reshape(n, frame_len).astype(np.float64) ** 2, axis=1))
    peak = rms.max()
    return bool(peak <= 1e-8 or rms[0] <= rel * peak)


def skip_leading_silence(y: np.ndarray, sr: int = ANALYSIS_SR, target_samples: int | None = None,
                          silence_rel_db: float = QUERY_SILENCE_REL_DB,
                          min_skip_sec: float = QUERY_SILENCE_MIN_SKIP_SEC) -> np.ndarray:
    """Drops a leading near-digital-silence section (frames below
    silence_rel_db relative to the loudest 100 ms frame) and returns at most
    target_samples samples from where content starts, re-peak-normalised.
    The threshold is deliberately far lower than the old 0.15 (-16 dB)
    heuristic that config.py notes hurt speech: -45 dB only catches padding /
    true silence, not soft speech onsets. Arrays that already start with
    content are returned unchanged (same object, then capped)."""
    rel = 10.0 ** (silence_rel_db / 20.0)
    out = y
    if y.size and _starts_silent(y, sr, rel):
        start = trim_leading_silence(y, sr, rel_threshold=rel)
        if start >= int(min_skip_sec * sr):
            out = y[start:]
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            if target_samples is not None:
                out = out[:target_samples]
                peak = float(np.max(np.abs(out))) if out.size else 0.0
            if peak > 1e-8:
                out = (out / peak).astype(np.float32)
            return out
    return out if target_samples is None else out[:target_samples]


def load_query_window(path: str, target_duration: float, sr: int = ANALYSIS_SR,
                       lookahead_sec: float = QUERY_SILENCE_LOOKAHEAD_SEC,
                       silence_rel_db: float = QUERY_SILENCE_REL_DB,
                       min_skip_sec: float = QUERY_SILENCE_MIN_SKIP_SEC) -> np.ndarray:
    """Query window of at most target_duration seconds that starts where
    content starts. Decodes only target_duration first; only when that window
    begins with near-silence (see skip_leading_silence) is a longer span
    (target_duration + lookahead_sec) decoded and the leading silence
    skipped. Recordings that start with content cost exactly what
    load_audio(max_duration=target_duration) costs and give the same array."""
    y = load_audio(path, sr=sr, max_duration=target_duration)
    if y.size == 0:
        return y
    rel = 10.0 ** (silence_rel_db / 20.0)
    if not _starts_silent(y, sr, rel):
        return y
    y_long = load_audio(path, sr=sr, max_duration=target_duration + lookahead_sec)
    return skip_leading_silence(y_long, sr, int(target_duration * sr), silence_rel_db, min_skip_sec)
