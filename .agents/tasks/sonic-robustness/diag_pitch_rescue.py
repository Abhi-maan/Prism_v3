"""Diagnostic: would re-enabling the (currently hard-disabled) pitch-hypothesis
rescue in pipeline.query_array recover pitch-shifted queries? Runs an in-memory
copy of query_array with `run_pitch_rescue = True` (the source file is NOT
modified) on +/-2 and +4 st edits, and also estimates the true pitch/tempo
change in the user's own two edited files."""
import inspect
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SONIC-fixed_1"))
import numpy as np
import librosa

import sonic_db_manager as sdm
from sonic.config import ANALYSIS_SR
from sonic.retrieval import pipeline as P

ECH = Path.home() / "Downloads" / "Echoes"


def lp(p):
    return ECH / p.split("Echoes\\", 1)[1].replace("\\", "/")


def dom(p):
    return "environment" if "ESC-50" in p else "music" if "genres" in p else "speech" if ("LJSpeech" in p or "dev-clean" in p) else None


def user_file_shift():
    dl = Path.home() / "Downloads"
    o, _ = librosa.load(str(dl / "Chaleya Jawan 320 Kbps.mp3"), sr=ANALYSIS_SR, offset=60, duration=30)
    for name in ("Chaleya_Jawan_pitch_shift_plus_2_semitones.mp3", "Chaleya_Jawan_speed_plus_5_percent.mp3"):
        d_full = librosa.get_duration(path=str(dl / name))
        ratio = 200.4125 / d_full
        q, _ = librosa.load(str(dl / name), sr=ANALYSIS_SR, offset=60 / ratio, duration=30 / ratio)
        # long-term average log-frequency spectrum, then best semitone lag
        def ltas(y):
            C = np.abs(librosa.cqt(y, sr=ANALYSIS_SR, bins_per_octave=36, n_bins=36 * 7, fmin=55))
            return np.log1p(C.mean(axis=1))
        a, b = ltas(o), ltas(q)
        lags = range(-18, 19)
        sc = [np.corrcoef(a[max(0, -l):len(a) - max(0, l)], b[max(0, l):len(b) - max(0, -l)])[0, 1] for l in lags]
        best = list(lags)[int(np.argmax(sc))] / 3.0
        print(f"user file {name}: duration {d_full:.1f}s (tempo ratio {ratio:.3f}), estimated pitch shift {best:+.2f} st")


def main():
    user_file_shift()
    src = inspect.getsource(P.query_array).replace("run_pitch_rescue = False", "run_pitch_rescue = True")
    ns = dict(vars(P))
    exec(src, ns)
    qa_rescue = ns["query_array"]

    db = sdm.SONICDatabaseManager(store_dir=ROOT / "sonic_db_store")
    while db.domain_index is None:
        time.sleep(0.5)
    idx = db.domain_index
    rng = random.Random(7)
    by_dom = defaultdict(list)
    for e in db.manifest:
        d = dom(e["path"])
        if d and lp(e["path"]).exists():
            by_dom[d].append(e)
    agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0.0, 0.0])
    for d in ("environment", "music", "speech"):
        for e in rng.sample(by_dom[d], 12):
            y, _ = librosa.load(str(lp(e["path"])), sr=ANALYSIS_SR, mono=True)
            for st in (-2, 2, 4):
                yy = librosa.effects.pitch_shift(y=y, sr=ANALYSIS_SR, n_steps=st)
                yy = yy / (np.abs(yy).max() + 1e-9)
                yy = yy[: int(idx.query_max_duration_sec * ANALYSIS_SR)]
                t0 = time.perf_counter(); r0 = P.query_array(yy, idx); t1 = time.perf_counter()
                r1 = qa_rescue(yy, idx); t2 = time.perf_counter()
                a = agg[(d, st)]
                a[0] += 1
                a[1] += bool(r0.ranked_ids and r0.ranked_ids[0] == e["file_id"])
                a[2] += r0.file_id == e["file_id"]
                a[3] += bool(r1.ranked_ids and r1.ranked_ids[0] == e["file_id"])
                a[4] += r1.file_id == e["file_id"]
                a[5] += (t1 - t0) * 1000; a[6] += (t2 - t1) * 1000
    print("domain       st   n | rescue OFF rank1 / accepted | rescue ON rank1 / accepted | ms off/on")
    for (d, st), a in sorted(agg.items()):
        print(f"{d:12s} {st:+d} {a[0]:3d} | {a[1]:3d} / {a[2]:3d} | {a[3]:3d} / {a[4]:3d} | {a[5]/a[0]:.0f}/{a[6]/a[0]:.0f}")


if __name__ == "__main__":
    main()
