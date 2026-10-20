"""Synthetic test audio shared by the unit tests (no dataset / DB needed)."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SONIC_ROOT = Path(__file__).resolve().parent.parent
PRISM_ROOT = SONIC_ROOT.parent
for p in (str(PRISM_ROOT), str(SONIC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

SR = 22050


def melody(seed: int, seconds: float = 8.0, sr: int = SR) -> np.ndarray:
    """A random 'tune': harmonic notes (5 partials, decaying envelope) of random
    pitch and length, plus a little filtered noise texture. Different seeds give
    clearly different spectro-temporal patterns."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float64)
    t = 0
    while t < n:
        dur = int(rng.uniform(0.12, 0.45) * sr)
        f0 = 110.0 * 2 ** (rng.integers(0, 36) / 12.0)
        tt = np.arange(min(dur, n - t)) / sr
        env = np.exp(-tt * rng.uniform(2.0, 7.0)) * (1 - np.exp(-tt * 200))
        note = sum((0.6 ** k) * np.sin(2 * np.pi * f0 * (k + 1) * tt + rng.uniform(0, 6.28)) for k in range(5))
        y[t:t + tt.size] += env * note * rng.uniform(0.4, 1.0)
        t += int(dur * rng.uniform(0.6, 1.0))
    noise = rng.standard_normal(n)
    kern = np.ones(64) / 64
    y += 0.05 * np.convolve(noise, kern, mode="same") * (1 + np.sin(2 * np.pi * rng.uniform(0.3, 2) * np.arange(n) / sr))
    y /= np.max(np.abs(y)) + 1e-9
    return y.astype(np.float32)


def write(path: Path, y: np.ndarray, sr: int = SR) -> Path:
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr, subtype="FLOAT")
    return path
