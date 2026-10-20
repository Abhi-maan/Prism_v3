"""Diagnostic (read-only on the real store): for a sample of local DB tracks per
domain, query the original, a +2 st pitch shift and a +5 % time-stretch (as
files, through sonic_db.query_file = exactly what /api/check Module 1 calls),
and record per-stage evidence for the true match:

  primary landmark rank / weighted score / gate, coarse+fine sub-fp vote rank,
  pipeline final rank, accept gate, app _classify_match decision.
"""
import json
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SONIC-fixed_1"))
import numpy as np
import librosa
import soundfile as sf

import sonic_db_manager as sdm
from sonic.config import ANALYSIS_SR
from sonic.io import load_audio
from sonic.fingerprint.landmarks import stft_magnitude, build_landmarks_from_magnitude
from sonic.fingerprint.binary_embed import sub_fingerprints_from_magnitude, fine_sub_fingerprints_from_magnitude
from sonic.retrieval import pipeline as P

ECH = Path.home() / "Downloads" / "Echoes"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def lp(p):
    return ECH / p.split("Echoes\\", 1)[1].replace("\\", "/") if "Echoes\\" in p else Path(p)


def dom(p):
    return "environment" if "ESC-50" in p else "music" if "genres" in p else "speech" if ("LJSpeech" in p or "dev-clean" in p) else None


def rank(fid, ids):
    ids = list(ids)
    return ids.index(fid) + 1 if fid in ids else None


def stage_trace(idx, path, fid):
    y = load_audio(path, max_duration=idx.query_max_duration_sec)
    mag, ft = stft_magnitude(y)
    c = idx.landmark_index.query(build_landmarks_from_magnitude(mag, ft), top_k=P.PRIMARY_POOL_K)
    tc = next((x for x in c if x.file_id == fid), None)
    conf, _ = P._is_confident(c[0] if c else None, c[1] if len(c) > 1 else None, idx)
    sub = sub_fingerprints_from_magnitude(mag, ft)
    v = idx.subfp_index.vote(np.stack([idx.binary_embed.binarize(x) for _t, x in sub]), P.SUBFP_TOP_K_PER_WINDOW, P.SUBFP_FINAL_TOP_K) if sub else []
    fine = fine_sub_fingerprints_from_magnitude(mag, ft)
    fv = idx.subfp_fine_index.vote(np.stack([idx.binary_embed.binarize(x) for _t, x in fine]), P.SUBFP_TOP_K_PER_WINDOW, P.SUBFP_FINAL_TOP_K) if fine else []
    res = P.query_array(y, idx)
    return {
        "prim_rank": rank(fid, [x.file_id for x in c]), "prim_true_w": round(tc.weighted_score, 1) if tc else 0.0,
        "prim_top_w": round(c[0].weighted_score, 1) if c else 0.0, "prim_conf": conf,
        "coarse_rank": rank(fid, [f for f, _ in v]), "fine_rank": rank(fid, [f for f, _ in fv]),
        "final_rank": rank(fid, res.ranked_ids), "pipe_fid": res.file_id, "stage": res.stage_used,
    }


def main():
    db = sdm.SONICDatabaseManager(store_dir=ROOT / "sonic_db_store")
    t0 = time.time()
    while db.domain_index is None:
        time.sleep(0.5)
    print(f"[index ready {time.time()-t0:.0f}s]", flush=True)
    idx = db.domain_index
    rng = random.Random(1)
    by_dom = defaultdict(list)
    for e in db.manifest:
        d = dom(e["path"])
        if d and lp(e["path"]).exists():
            by_dom[d].append(e)
    tracks = [("music", db._manifest_by_id[28408] | {"_local": str(Path.home() / "Downloads/Chaleya Jawan 320 Kbps.mp3")})]
    for d in ("environment", "music", "speech"):
        for e in rng.sample(by_dom[d], N):
            tracks.append((d, e | {"_local": str(lp(e["path"]))}))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for d, e in tracks:
            y, _ = librosa.load(e["_local"], sr=ANALYSIS_SR, mono=True)
            conds = {"original": None,
                     "pitch+2": librosa.effects.pitch_shift(y=y, sr=ANALYSIS_SR, n_steps=2),
                     "speed+5": librosa.effects.time_stretch(y=y, rate=1.05)}
            for cname, yy in conds.items():
                if yy is None:
                    q = e["_local"]
                else:
                    q = str(Path(td) / f"{cname}_{Path(e['filename']).stem}.wav")
                    sf.write(q, yy, ANALYSIS_SR)
                tr = stage_trace(idx, q, e["file_id"])
                app = db.query_file(q)
                tr.update(domain=d, file=e["filename"], cond=cname,
                          app_matched=bool(app["matched"]), app_correct=bool(app["matched"] and app["matched_filename"] == e["filename"]),
                          app_type=app["match_type"], app_ms=app["query_time_ms"])
                rows.append(tr)
                print(json.dumps(tr), flush=True)
    out = ROOT / ".agents/tasks/sonic-robustness/diag_probe_rows.json"
    json.dump(rows, open(out, "w"), indent=1)
    print("\nSUMMARY (domain, cond): app_correct | primary rank1 | primary confident | coarse rank<=20 | fine rank1 | final rank1 | pipe_fid==true")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["domain"], r["cond"])].append(r)
    for k, rs in sorted(agg.items()):
        n = len(rs)
        f = lambda c: sum(1 for r in rs if c(r))
        print(f"  {k[0]:12s} {k[1]:9s} n={n:3d} app={f(lambda r: r['app_correct'])} prim1={f(lambda r: r['prim_rank']==1)} "
              f"primconf={f(lambda r: r['prim_conf'])} coarse={f(lambda r: r['coarse_rank'] is not None)} fine1={f(lambda r: r['fine_rank']==1)} "
              f"final1={f(lambda r: r['final_rank']==1)} accepted_true={f(lambda r: r['pipe_fid'] is not None and r['final_rank']==1)}")


if __name__ == "__main__":
    main()
