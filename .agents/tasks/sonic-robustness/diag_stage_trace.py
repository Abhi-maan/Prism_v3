"""Diagnostic (read-only): trace a query through every SONIC stage against the
REAL reference DB and report the rank/score of the true match at each stage,
then the app's final decision (sonic_db.query_file -> _classify_match).

Usage:
  .venv/bin/python .agents/tasks/sonic-robustness/diag_stage_trace.py TRUE_FILE_ID QUERY [QUERY ...]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SONIC-fixed_1"))

import numpy as np

import sonic_db_manager as sdm
from sonic.config import ANALYSIS_SR, QUERY_MAX_DURATION_SEC
from sonic.io import load_audio
from sonic.fingerprint.landmarks import stft_magnitude, build_landmarks_from_magnitude
from sonic.fingerprint.binary_embed import (
    band_energy_vector_from_magnitude, sub_fingerprints_from_magnitude,
    fine_sub_fingerprints_from_magnitude, SUBFP_N_BANDS, SUBFP_FINE_N_BANDS,
)
from sonic.retrieval import pipeline as P


def rank_of(fid, ranked):
    try:
        return ranked.index(fid) + 1
    except ValueError:
        return None


def trace(idx, path, true_fid):
    y = load_audio(path, max_duration=idx.query_max_duration_sec)
    mag, ft = stft_magnitude(y, ANALYSIS_SR)
    q_lm = build_landmarks_from_magnitude(mag, ft, ANALYSIS_SR)
    cands = idx.landmark_index.query(q_lm, top_k=P.PRIMARY_POOL_K)
    prim_ids = [c.file_id for c in cands]
    top = cands[0] if cands else None
    second = cands[1] if len(cands) > 1 else None
    conf, margin = P._is_confident(top, second, idx)
    true_c = next((c for c in cands if c.file_id == true_fid), None)
    out = {
        "query_sec": round(y.size / ANALYSIS_SR, 2),
        "n_query_hashes": int(q_lm.hash.size),
        "primary": {
            "true_rank": rank_of(true_fid, prim_ids),
            "true_wscore": round(true_c.weighted_score, 2) if true_c else None,
            "true_offset_conf": round(true_c.offset_confidence, 3) if true_c else None,
            "top_fid": top.file_id if top else None,
            "top_wscore": round(top.weighted_score, 2) if top else None,
            "top_offset_conf": round(top.offset_confidence, 3) if top else None,
            "confident_gate": conf, "margin": round(margin, 3),
        },
    }
    # secondary
    code = idx.binary_embed.binarize(band_energy_vector_from_magnitude(mag, ANALYSIS_SR))
    shortlist = idx.binary_hnsw.search(code, top_k=P.SECONDARY_SHORTLIST_K)
    sub = sub_fingerprints_from_magnitude(mag, ft, ANALYSIS_SR)
    votes = idx.subfp_index.vote(np.stack([idx.binary_embed.binarize(v) for _t, v in sub]),
                                 top_k_per_window=P.SUBFP_TOP_K_PER_WINDOW, final_top_k=P.SUBFP_FINAL_TOP_K) if sub else []
    fine = fine_sub_fingerprints_from_magnitude(mag, ft, ANALYSIS_SR)
    fvotes = idx.subfp_fine_index.vote(np.stack([idx.binary_embed.binarize(v) for _t, v in fine]),
                                       top_k_per_window=P.SUBFP_TOP_K_PER_WINDOW, final_top_k=P.SUBFP_FINAL_TOP_K) if fine else []
    out["secondary"] = {
        "global_hnsw_true_rank": rank_of(true_fid, [f for f, _ in shortlist]),
        "subfp_coarse_true_rank": rank_of(true_fid, [f for f, _ in votes]),
        "subfp_coarse_top": [(int(f), round(float(s), 1)) for f, s in votes[:3]],
        "subfp_coarse_true_score": next((round(float(s), 1) for f, s in votes if f == true_fid), None),
        "subfp_fine_true_rank": rank_of(true_fid, [f for f, _ in fvotes]),
        "subfp_fine_top": [(int(f), round(float(s), 1)) for f, s in fvotes[:3]],
    }
    t0 = time.perf_counter()
    res = P.query_array(y, idx)
    out["pipeline"] = {
        "file_id": res.file_id, "stage": res.stage_used, "confident": res.confident,
        "wscore": round(res.weighted_score, 2), "offset_conf": round(res.offset_confidence, 3),
        "margin": round(res.margin, 3), "ranked": res.ranked_ids[:5],
        "true_rank_final": rank_of(true_fid, list(res.ranked_ids)),
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return out, res


def corr_check(qpath, mpath):
    yq = load_audio(qpath, sr=ANALYSIS_SR, max_duration=QUERY_MAX_DURATION_SEC)
    yd = load_audio(mpath, sr=ANALYSIS_SR, max_duration=QUERY_MAX_DURATION_SEC)
    n = min(len(yq), len(yd))
    c = float(np.dot(yq[:n], yd[:n]) / (np.linalg.norm(yq[:n]) * np.linalg.norm(yd[:n]) + 1e-12))
    return round(c, 4)


def main():
    true_fid = int(sys.argv[1])
    queries = sys.argv[2:]
    t0 = time.time()
    db = sdm.SONICDatabaseManager(store_dir=ROOT / "sonic_db_store")
    while db.domain_index is None:
        time.sleep(0.5)
        if time.time() - t0 > 1800:
            raise SystemExit("index load timeout")
    print(f"[index ready in {time.time()-t0:.1f}s]", flush=True)
    entry = db._manifest_by_id[true_fid]
    print("true entry:", entry["filename"], entry["path"])
    for q in queries:
        out, res = trace(db.domain_index, q, true_fid)
        app = db.query_file(q)
        out["app_query_file"] = {k: app.get(k) for k in ("matched", "matched_filename", "match_type", "score", "confident", "stage_used", "query_time_ms")}
        if res.file_id is not None and res.file_id in db._manifest_by_id:
            mp = db._manifest_by_id[res.file_id]["path"]
            out["app_query_file"]["classify_waveform_corr"] = corr_check(q, mp) if Path(mp).exists() else "matched path missing -> classify returns res.confident"
        print("\n=== ", Path(q).name)
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
