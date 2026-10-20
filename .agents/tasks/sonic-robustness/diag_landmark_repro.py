"""Diagnostic (no DB load): is a 5 s query window's landmark set reproducible
inside the full-file landmark set that ingestion stores? Isolates the effect of
(a) window length (global per-file peak budget / percentile threshold),
(b) peak normalisation of the window, (c) stored-vs-fresh extraction drift."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SONIC-fixed_1"))
import numpy as np

from sonic.io import load_audio
from sonic.fingerprint.landmarks import stft_magnitude, build_landmarks_from_magnitude
from sonic.config import ANALYSIS_SR

ECH = Path.home() / "Downloads" / "Echoes"
FILES = [
    ("music-200s", Path.home() / "Downloads" / "Chaleya Jawan 320 Kbps.mp3", ROOT / "sonic_db_store/fingerprints/fp_28408.npz"),
]


def lm(y):
    mag, ft = stft_magnitude(y)
    return build_landmarks_from_magnitude(mag, ft)


def frac_in(q, ref):
    return float(np.isin(q, ref).mean()) if q.size else float("nan")


def main():
    import json
    man = {e["filename"]: e for e in json.load(open(ROOT / "sonic_db_store/manifest.json"))["files"]}
    for name in ["blues.00000.au", "rock.00063.au", "jazz.00010.au", "1-100032-A-0.wav", "4-195305-A-31.wav", "LJ001-0001.wav"]:
        e = man[name]
        lp = ECH / e["path"].split("Echoes\\", 1)[1].replace("\\", "/")
        FILES.append((name, lp, ROOT / "sonic_db_store/fingerprints" / e["fp_file"]))
    for label, path, npz in FILES:
        full = load_audio(str(path))
        st = np.load(npz)["hash"]
        L_full = lm(full)
        q5 = load_audio(str(path), max_duration=5.0)          # what query() does
        L_q5 = lm(q5)
        L_q5_fullnorm = lm(full[: 5 * ANALYSIS_SR])            # same samples, full-file normalisation
        print(f"{label:22s} dur={full.size/ANALYSIS_SR:6.1f}s stored={st.size:6d} fresh_full={L_full.hash.size:6d} "
              f"| q5 hashes={L_q5.hash.size:4d} in stored={frac_in(L_q5.hash, st):.1%} in fresh_full={frac_in(L_q5.hash, L_full.hash):.1%} "
              f"| q5(fullnorm) in fresh_full={frac_in(L_q5_fullnorm.hash, L_full.hash):.1%} "
              f"| fresh_full in stored={frac_in(L_full.hash, st):.1%}")


if __name__ == "__main__":
    main()
