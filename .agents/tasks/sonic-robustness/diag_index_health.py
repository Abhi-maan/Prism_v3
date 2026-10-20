"""Diagnostic (read-only on the real store): index health + consistency checks.

1. Landmark index stats (stop-hash pruning at 28k scale).
2. Per-file surviving-hash ratio for the user's ingested track (fid 28408) and samples.
3. Archive (unified_archive.npz + hnsw_index.faiss) vs manifest / per-file .npz consistency.
4. Stored .npz vs fresh extraction from the local original (ingest/query config drift).
5. Exact-duplicate self-query through sonic_db.query_file for local sample tracks per domain.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SONIC-fixed_1"))
import numpy as np

import sonic_db_manager as sdm
from sonic.io import load_audio
from sonic.fingerprint.landmarks import stft_magnitude, build_landmarks_from_magnitude

ECHOES = Path.home() / "Downloads" / "Echoes"


def local_path(p):
    if "Echoes\\" in p:
        return ECHOES / p.split("Echoes\\", 1)[1].replace("\\", "/")
    return Path(p)


def domain_of(p):
    if "ESC-50" in p:
        return "environment"
    if "genres" in p:
        return "music"
    if "LJSpeech" in p or "dev-clean" in p:
        return "speech"
    return "other"


def main():
    t0 = time.time()
    db = sdm.SONICDatabaseManager(store_dir=ROOT / "sonic_db_store")
    while db.domain_index is None:
        time.sleep(0.5)
    idx = db.domain_index
    lidx = idx.landmark_index
    print(f"[index ready {time.time()-t0:.0f}s] files={lidx.n_files}")

    total_before = sum(int(v) for v in lidx.n_hashes_per_file.values())
    print(f"1) landmark hashes before prune={total_before:,} after prune={lidx.sorted_hash.size:,} "
          f"kept={lidx.sorted_hash.size/total_before:.1%}")
    u, cnt = np.unique(lidx.sorted_hash, return_counts=True)
    print(f"   distinct kept hashes={u.size:,}; posting len p50={np.median(cnt):.0f} p90={np.percentile(cnt,90):.0f} max={cnt.max()}")

    manifest = db.manifest
    by_id = {e["file_id"]: e for e in manifest}
    lid_counts = np.bincount(lidx.local_ids, minlength=lidx.n_files)

    def surv(fid):
        loc = lidx.file_id_to_local[fid]
        return int(lid_counts[loc]), int(lidx.n_hashes_per_file.get(fid, 0))

    s, n = surv(28408)
    print(f"2) fid 28408 (Chaleya, 200s): surviving hashes {s}/{n} = {s/max(n,1):.1%}")
    rng = random.Random(0)
    ratios = Counter()
    for dom in ("environment", "music", "speech", "other"):
        fids = [e["file_id"] for e in manifest if domain_of(e["path"]) == dom]
        smp = rng.sample(fids, min(200, len(fids)))
        r = [surv(f)[0] / max(surv(f)[1], 1) for f in smp]
        print(f"   {dom:12s} surviving-hash ratio mean={np.mean(r):.1%} p10={np.percentile(r,10):.1%}")

    # Query-hash survival for the exact original
    q = str(Path.home() / "Downloads" / "Chaleya Jawan 320 Kbps.mp3")
    y = load_audio(q, max_duration=idx.query_max_duration_sec)
    mag, ft = stft_magnitude(y)
    qlm = build_landmarks_from_magnitude(mag, ft)
    in_idx = np.isin(qlm.hash, u)
    stored = np.load(ROOT / "sonic_db_store" / "fingerprints" / "fp_28408.npz")
    in_stored = np.isin(qlm.hash, stored["hash"])
    in_stored_kept = np.isin(qlm.hash, lidx.get_file_hashes(28408))
    print(f"   exact-original 5s query: {qlm.hash.size} hashes; in full stored fp_28408={in_stored.mean():.1%}; "
          f"present anywhere in index={in_idx.mean():.1%}; in 28408's surviving postings={in_stored_kept.mean():.1%}")

    # 3) archive consistency
    arc = np.load(ROOT / "sonic_db_store" / "unified_archive.npz")
    a_ids = arc["file_ids"]
    print(f"3) archive file_ids={a_ids.size} manifest={len(manifest)} contiguous={bool(np.all(a_ids==np.arange(a_ids.size)))} "
          f"missing_in_archive={sorted(set(by_id)-set(a_ids.tolist()))[:5]}")
    print(f"   archive keys={list(arc.keys())}")
    import faiss
    fidx = faiss.read_index_binary(str(ROOT / "sonic_db_store" / "hnsw_index.faiss"))
    print(f"   faiss ntotal={fidx.ntotal} d={fidx.d}")
    # compare raw_vec binarization and per-file hash counts with npz for samples
    n_hashes_arc = dict(zip(a_ids.tolist(), arc["n_hashes"].tolist()))
    mism_n, mism_vec, mism_faiss = 0, 0, 0
    sample = rng.sample(a_ids.tolist(), 300)
    codes_arc = np.packbits((arc["raw_vecs"][sample] > 0).astype(np.uint8), axis=1) if arc["raw_vecs"].ndim == 2 else None
    for k, fid in enumerate(sample):
        with np.load(ROOT / "sonic_db_store" / "fingerprints" / by_id[fid]["fp_file"]) as d:
            if int(d["hash"].size) != int(n_hashes_arc[fid]):
                mism_n += 1
            if not np.allclose(d["raw_vec"], arc["raw_vecs"][fid], atol=1e-5):
                mism_vec += 1
        # FAISS self-search: stored code of fid should be its own nearest neighbour (distance 0)
        code = idx.binary_embed.binarize(arc["raw_vecs"][fid])[None, :]
        D, I = fidx.search(code, 1)
        if a_ids[I[0, 0]] != fid and D[0, 0] != 0:
            mism_faiss += 1
    print(f"   300-sample: n_hashes mismatch archive-vs-npz={mism_n}, raw_vec mismatch={mism_vec}, faiss self-NN wrong(not dist0)={mism_faiss}")

    # 4) config drift: fresh extraction vs stored npz for local originals
    drift = []
    for dom in ("environment", "music", "speech"):
        cands = [e for e in manifest if domain_of(e["path"]) == dom and local_path(e["path"]).exists()]
        for e in rng.sample(cands, 5):
            fp = db.extract_fingerprint(local_path(e["path"]))
            with np.load(ROOT / "sonic_db_store" / "fingerprints" / e["fp_file"]) as d:
                same_h = np.array_equal(np.sort(fp["hash"]), np.sort(d["hash"]))
                jacc = np.intersect1d(fp["hash"], d["hash"]).size / max(np.union1d(fp["hash"], d["hash"]).size, 1)
                drift.append((dom, e["filename"], same_h, round(jacc, 3), fp["subfp"].shape, d["subfp"].shape,
                              bool(np.allclose(fp["raw_vec"], d["raw_vec"], atol=1e-4))))
    print("4) fresh-extract vs stored (domain, file, identical_hashes, jaccard, subfp shape new/stored, raw_vec equal):")
    for row in drift:
        print("   ", row)

    # 5) exact-duplicate self-query via the app path
    print("5) exact duplicate self-query via sonic_db.query_file (local originals):")
    for dom in ("environment", "music", "speech"):
        cands = [e for e in manifest if domain_of(e["path"]) == dom and local_path(e["path"]).exists()]
        ok = prim = 0
        smp = rng.sample(cands, 40)
        for e in smp:
            r = db.query_file(str(local_path(e["path"])))
            res_ok = r["matched"] and r["matched_filename"] == e["filename"]
            ok += res_ok
            prim += r["stage_used"] == "primary"
        print(f"   {dom:12s} correct {ok}/{len(smp)}  primary-stage-confident {prim}/{len(smp)}")


if __name__ == "__main__":
    main()
