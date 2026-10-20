"""Module 1 (Duplicate & Edited Audio Detection, SONIC) query-latency vs reference-DB size.

Measures how per-query latency of the app's Module 1 query path grows as the
reference DB grows. Every timed query goes through
``app.run_duplicate_detection([path], "auto")``, the function POST /api/check
calls for module=duplicate_detection, in-process. Retrieval and decision logic
are not re-implemented here.

Reference DB at each size
  * size == real DB size (28,409): the app's own attached index
    (sonic_db_store, attached by the app at import, read-only).
  * larger sizes: the real archive + delta fingerprints plus N - 28,409
    *distractor* entries, assembled in memory with the app's own
    ``SONICDatabaseManager._assemble_domain_index`` (same classes, HNSW M=32,
    efSearch 64, one unified "general" index exactly like the app). faiss
    threads are raised only while building and restored for querying, as
    ``_rebuild_index`` does. Nothing is written to sonic_db_store; the
    manifest change is in memory only.

Distractors (audio-derived, NOT random bits, NOT copies)
  Each distractor copies the (domain, duration) of a real DB entry, so the
  domain mix and length distribution, and with them windows and hashes per
  file, match the real DB. Its audio is a concatenation of segments. Each
  segment is a random 3-15 s excerpt of a real local source clip of that
  domain (GTZAN / ESC-50 / LJSpeech + LibriSpeech dev-clean), processed as
  follows:
    time-reversed, varispeed by r in [0.72,0.86] U [1.18,1.38] (pitch AND
    tempo move together, ~+-2.6..5.7 st), mixed with a different source clip
    of the same domain at -3..-9 dB.
  The result is real-sounding material with realistic landmark and
  sub-fingerprint statistics, yet no copy of any DB entry within the module's
  envelope. The query tracks and the sources of the reversed-ESC negatives
  are excluded from the source pools, so the correct answer stays
  unambiguous. Fingerprints come from
  ``SONICDatabaseManager.extract_fingerprint``, the app's ingest function, run
  on a temporary float WAV.

Query mix (same files at every size)
  (a) duplicate: unmodified copies of DB tracks (benchmark-20 track selection, seed 42)
  (b) edited: pitch +-2/+-4 st, speed +-5/+-20 %, MP3 64 kbps, noise SNR 20 dB,
      trim start 10 s, time-shift pad 20 s (benchmark-20 make_edit)
  (c) negative: not in DB (benchmark-20 build_negatives)
  Correct = matched AND matched_filename == true filename; for negatives any
  match is a false positive.

Stage breakdown: exclusive-time wrappers installed on the module's
functions at runtime (no source edits). The wrappers add roughly 0.1-0.3 ms
per query.

Usage (from prism_v2/, live server stopped to free RAM):
  .venv/bin/python SONIC-fixed_1/scripts/21_module1_latency_scale_benchmark.py \
      --sizes 28409,50000,75000,100000 --out-json /tmp/m1_latency.json < /dev/null
  # HTTP overhead at the real DB, against the running server:
  .venv/bin/python SONIC-fixed_1/scripts/21_module1_latency_scale_benchmark.py \
      --http-url http://127.0.0.1:8060 --out-json /tmp/m1_latency_http.json < /dev/null
"""
from __future__ import annotations

import argparse
import functools
import gc
import importlib.util
import json
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SONIC_ROOT = Path(__file__).resolve().parent.parent
PRISM_ROOT = SONIC_ROOT.parent
for _p in (str(PRISM_ROOT), str(SONIC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STORE = PRISM_ROOT / "sonic_db_store"
DOWNLOADS = Path.home() / "Downloads"
SR = 22050
EDIT_CONDITIONS = ["pitch_-4st", "pitch_-2st", "pitch_+2st", "pitch_+4st",
                   "speed_-20pct", "speed_-5pct", "speed_+5pct", "speed_+20pct",
                   "mp3_64kbps", "noise_snr20db", "trim_start_10s", "timeshift_pad_20s"]
CATS = ("duplicate", "edited", "negative")


def load_bench20():
    spec = importlib.util.spec_from_file_location(
        "bench20", str(SONIC_ROOT / "scripts" / "20_module1_robustness_benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_manifest():
    with open(STORE / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)["files"]


def rss_gb():
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def proc_rusage_info():
    """macOS proc_pid_rusage(RUSAGE_INFO_V0): physical footprint (includes pages the
    kernel compressed, unlike RSS) and process page-ins. {} elsewhere."""
    try:
        import ctypes
        import ctypes.util

        class RUsageInfoV0(ctypes.Structure):
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16), ("ri_user_time", ctypes.c_uint64),
                        ("ri_system_time", ctypes.c_uint64), ("ri_pkg_idle_wkups", ctypes.c_uint64),
                        ("ri_interrupt_wkups", ctypes.c_uint64), ("ri_pageins", ctypes.c_uint64),
                        ("ri_wired_size", ctypes.c_uint64), ("ri_resident_size", ctypes.c_uint64),
                        ("ri_phys_footprint", ctypes.c_uint64), ("ri_proc_start_abstime", ctypes.c_uint64),
                        ("ri_proc_exit_abstime", ctypes.c_uint64)]
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        info = RUsageInfoV0()
        if libc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
            return {}
        return {"footprint_gb": round(info.ri_phys_footprint / 1e9, 2), "pageins": int(info.ri_pageins)}
    except Exception:
        return {}


def vm_counters():
    """System-wide paging counters (macOS vm_stat) to flag swapping during a phase."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except Exception:
        return {}
    c = {}
    for k in ("Pageins", "Swapins", "Decompressions"):
        m = re.search(rf"^{k}:\s+(\d+)", out, re.M)
        if m:
            c[k.lower()] = int(m.group(1))
    return c


# ----------------------------------------------------------------------------- query set
def prepare_queries(b20, manifest, args, qdir: Path):
    import librosa
    echoes = Path(args.echoes_root)
    tracks, _pools = b20.select_tracks(manifest, echoes, args.dup_per_domain, args.seed)
    by_dom = defaultdict(list)
    for t in tracks:
        by_dom[t["domain"]].append(t)
    edit_tracks = list(by_dom.get("user_upload", []))
    for d in ("environment", "music", "speech"):
        edit_tracks += by_dom.get(d, [])[: args.edit_per_domain]
    edit_ids = {t["entry"]["file_id"] for t in edit_tracks}

    rng = np.random.default_rng(args.seed)
    queries = []
    for ti, t in enumerate(tracks):
        e = t["entry"]
        tdir = qdir / f"t{ti}"
        tdir.mkdir(parents=True, exist_ok=True)
        y = None
        conds = ["duplicate"] + (EDIT_CONDITIONS if e["file_id"] in edit_ids else [])
        for c in conds:
            kind, param = b20.ALL_CONDITIONS[c]
            if kind != "duplicate" and y is None:
                y, _ = librosa.load(str(t["local"]), sr=SR, mono=True)
                y = y.astype(np.float32)
            try:
                qp = b20.make_edit(kind, param, y if y is not None else np.zeros(SR, np.float32),
                                   t["local"], tdir, f"{c}_{ti}", rng)
            except Exception as ex:
                print(f"  ! generation failed {c} {e['filename']}: {ex}", flush=True)
                qp = None
            if qp is None:
                continue
            queries.append({"cat": "duplicate" if c == "duplicate" else "edited", "cond": c,
                            "domain": t["domain"], "path": str(qp), "true_filename": e["filename"],
                            "true_file_id": e["file_id"]})
        print(f"[queries] track {ti + 1}/{len(tracks)} {t['domain']:11s} {e['filename'][:40]}", flush=True)
    ndir = qdir / "neg"
    ndir.mkdir(parents=True, exist_ok=True)
    negs = b20.build_negatives(manifest, echoes, args.negatives_per_source, args.seed, ndir)
    for n in negs:
        queries.append({"cat": "negative", "cond": n["source"], "domain": "negative", "path": str(n["path"]),
                        "true_filename": None, "true_file_id": None})
    # sources that must not be used to make distractors
    exclude = {str(t["local"]) for t in tracks}
    esc_by_stem = {Path(e["filename"]).stem: e for e in manifest if "ESC-50" in e["path"]}
    for n in negs:
        if n["source"] == "esc50_time_reversed":
            stem = Path(n["path"]).stem[len("neg_rev_"):]
            e = esc_by_stem.get(stem)
            lp = b20.resolve_local(e["path"], echoes) if e else None
            if lp is not None:
                exclude.add(str(lp))
    return queries, exclude, len(tracks), len(edit_tracks)


# ----------------------------------------------------------------------------- distractors
def template_domain(path: str) -> str:
    p = path.replace("\\", "/")
    if "ESC-50" in p or "DESED" in p:
        return "environment"
    if "LJSpeech" in p or "dev-clean" in p or "wavefake" in p:
        return "speech"
    return "music"  # GTZAN, AI-generated music (TTA/ATA), user songs


def source_pools(b20, manifest, echoes: Path, exclude: set):
    pools = defaultdict(list)
    for e in manifest:
        d = b20.domain_of(e["path"])
        if d is None:
            continue
        lp = b20.resolve_local(e["path"], echoes)
        if lp is None or str(lp) in exclude:
            continue
        pools[d].append(str(lp))
    return {d: sorted(v) for d, v in pools.items()}


def _synth_distractor(dur_sec, pool, rng, get):
    target = max(int(dur_sec * SR), SR)
    parts, n = [], 0
    while n < target:
        a = get(pool[int(rng.integers(len(pool)))])
        if a.size < SR // 2:
            continue
        seg_len = min(a.size, int(rng.uniform(3.0, 15.0) * SR))
        st = int(rng.integers(0, a.size - seg_len + 1))
        seg = a[st:st + seg_len][::-1]
        r = float(rng.uniform(0.72, 0.86) if rng.random() < 0.5 else rng.uniform(1.18, 1.38))
        pos = np.arange(0.0, seg.size - 1, r)
        seg = np.interp(pos, np.arange(seg.size), seg).astype(np.float32)
        b = get(pool[int(rng.integers(len(pool)))])
        if b.size >= SR // 2:
            reps = int(np.ceil(seg.size / b.size)) + 1
            bt = np.tile(b, reps)
            o = int(rng.integers(0, b.size))
            bw = bt[o:o + seg.size]
            ps, pb = float(np.sqrt(np.mean(seg ** 2))) + 1e-9, float(np.sqrt(np.mean(bw ** 2))) + 1e-9
            seg = seg + bw * (ps / pb) * 10 ** (rng.uniform(-9.0, -3.0) / 20.0)
        parts.append(seg)
        n += seg.size
    y = np.concatenate(parts)[:target]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def _gen_chunk(job):
    """Worker: synthesise + fingerprint one chunk of distractors with the app's ingest function."""
    chunk_id, specs, pools, out_path, seed, work_dir = job
    for p in (str(PRISM_ROOT), str(SONIC_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import soundfile as sf
    from sonic.io import load_audio
    from sonic_db_manager import SONICDatabaseManager

    rng = np.random.default_rng(seed)
    cache: dict = {}

    def get(path):
        y = cache.get(path)
        if y is None:
            if len(cache) > 256:
                cache.pop(next(iter(cache)))
            y = load_audio(path, sr=SR).astype(np.float32)
            cache[path] = y
        return y

    t0 = time.time()
    hashes, times, n_h, raw, sc, s_n, fc, f_n, durs = [], [], [], [], [], [], [], [], []
    wav = Path(work_dir) / f"_d{chunk_id}.wav"
    for dom, dur in specs:
        y = _synth_distractor(dur, pools[dom], rng, get)
        sf.write(str(wav), y, SR, subtype="FLOAT")
        fp = SONICDatabaseManager.extract_fingerprint(wav)
        hashes.append(fp["hash"].astype(np.uint32))
        times.append(fp["anchor_time"].astype(np.float32))
        n_h.append(fp["hash"].size)
        raw.append(np.asarray(fp["raw_vec"], np.float32).reshape(-1))
        s = np.asarray(fp["subfp"], np.uint8).reshape(-1, 8)
        f = np.asarray(fp["subfp_fine"], np.uint8)
        f = f.reshape(-1, 24) if f.size else np.empty((0, 24), np.uint8)
        sc.append(s)
        s_n.append(s.shape[0])
        fc.append(f)
        f_n.append(f.shape[0])
        durs.append(fp["duration_sec"])
    wav.unlink(missing_ok=True)
    np.savez(out_path, hashes=np.concatenate(hashes), times=np.concatenate(times),
             n_hashes=np.array(n_h, np.int32), raw_vecs=np.stack(raw), subfp=np.concatenate(sc),
             subfp_n=np.array(s_n, np.int32), fine=np.concatenate(fc), fine_n=np.array(f_n, np.int32),
             dur=np.array(durs, np.float32))
    return chunk_id, len(specs), time.time() - t0, float(np.sum(durs))


def generate_distractors(b20, manifest, exclude, n_needed, args, ddir: Path):
    """Generates (or reuses, if already present) chunks covering n_needed distractors."""
    ddir.mkdir(parents=True, exist_ok=True)
    pools = source_pools(b20, manifest, Path(args.echoes_root), exclude)
    rng = random.Random(f"{args.seed}|distractor-templates")
    order = list(range(len(manifest)))
    rng.shuffle(order)
    specs = []
    for j in range(n_needed):
        e = manifest[order[j % len(order)]]
        specs.append((template_domain(e["path"]), float(min(e.get("duration_sec") or 5.0, args.max_distractor_sec))))
    cs = args.chunk_size
    jobs = []
    for c in range(0, n_needed, cs):
        out = ddir / f"chunk_{c // cs:05d}.npz"
        if out.exists():
            continue
        jobs.append((c // cs, specs[c:c + cs], pools, str(out), args.seed * 100003 + c // cs, str(ddir)))
    stats = {"pool_sizes": {d: len(v) for d, v in pools.items()},
             "template_domains": dict(Counter(s[0] for s in specs)),
             "n": n_needed, "generated_chunks": len(jobs), "cpu_sec": 0.0, "audio_sec": 0.0}
    if not jobs:
        return stats
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_gen_chunk, j) for j in jobs]
        for f in as_completed(futs):
            cid, n, sec, audio = f.result()
            stats["cpu_sec"] += sec
            stats["audio_sec"] += audio
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[distractors] {done}/{len(jobs)} chunks ({time.time() - t0:.0f}s)", flush=True)
    stats["wall_sec"] = round(time.time() - t0, 1)
    return stats


def chunk_files(ddir: Path):
    return sorted(ddir.glob("chunk_*.npz"))


# ----------------------------------------------------------------------------- scaled index
def assemble_scaled(manifest, n_total, ddir: Path):
    """Real archive + delta + (n_total - n_real) distractors -> app DomainIndex (+ manifest entries)."""
    import faiss
    from sonic_db_manager import SONICDatabaseManager as M

    arch = np.load(STORE / "unified_archive.npz")
    a_fids = arch["file_ids"].astype(np.int64)
    a_set = set(a_fids.tolist())
    delta = [e for e in manifest if e["file_id"] not in a_set and (STORE / "fingerprints" / e["fp_file"]).exists()]
    n_real = len(a_fids) + len(delta)
    n_dist = n_total - n_real
    if n_dist < 0:
        raise SystemExit(f"size {n_total} < real DB size {n_real}")
    dz = []
    have = 0
    for p in chunk_files(ddir):
        if have >= n_dist:
            break
        z = dict(np.load(p))
        take = min(len(z["n_hashes"]), n_dist - have)
        dz.append((z, take))
        have += take
    if have < n_dist:
        raise SystemExit(f"only {have} distractors available, need {n_dist}")
    dlt = []
    for e in delta:
        with np.load(STORE / "fingerprints" / e["fp_file"]) as d:
            dlt.append({k: d[k] for k in ("hash", "anchor_time", "raw_vec", "subfp", "subfp_fine")})

    def dist_parts(z, take):
        nh = z["n_hashes"][:take]
        sn, fn = z["subfp_n"][:take], z["fine_n"][:take]
        return nh, int(nh.sum()), sn, int(sn.sum()), fn, int(fn.sum())

    n_hash = int(arch["hashes"].shape[0]) + sum(int(d["hash"].size) for d in dlt) + \
        sum(dist_parts(z, t)[1] for z, t in dz)
    n_sub = int(arch["subfp_codes"].shape[0]) + sum(int(d["subfp"].shape[0]) for d in dlt) + \
        sum(dist_parts(z, t)[3] for z, t in dz)
    n_fine = int(arch["subfp_fine_codes"].shape[0]) + sum(int(d["subfp_fine"].shape[0]) for d in dlt) + \
        sum(dist_parts(z, t)[5] for z, t in dz)

    hashes = np.empty(n_hash, np.uint32)
    lids = np.empty(n_hash, np.int32)
    times = np.empty(n_hash, np.float32)
    s_fids = np.empty(n_sub, np.int64)
    s_codes = np.empty((n_sub, 8), np.uint8)
    f_fids = np.empty(n_fine, np.int64)
    f_codes = np.empty((n_fine, 24), np.uint8)
    raw_vecs = np.empty((n_total, 256), np.float32)
    n_hashes = np.empty(n_total, np.int32)
    file_ids = np.empty(n_total, np.int64)

    na = len(a_fids)
    h0 = arch["hashes"].shape[0]
    hashes[:h0] = arch["hashes"]
    lids[:h0] = arch["lids"]
    times[:h0] = arch["times"]
    s0 = arch["subfp_codes"].shape[0]
    s_fids[:s0] = arch["subfp_fids"]
    s_codes[:s0] = arch["subfp_codes"]
    f0 = arch["subfp_fine_codes"].shape[0]
    f_fids[:f0] = arch["subfp_fine_fids"]
    f_codes[:f0] = arch["subfp_fine_codes"]
    raw_vecs[:na] = arch["raw_vecs"]
    n_hashes[:na] = arch["n_hashes"]
    file_ids[:na] = a_fids
    del arch
    hp, sp, fp, li = h0, s0, f0, na
    for e, d in zip(delta, dlt):
        h = d["hash"].size
        hashes[hp:hp + h] = d["hash"]
        lids[hp:hp + h] = li
        times[hp:hp + h] = d["anchor_time"]
        hp += h
        s = d["subfp"].reshape(-1, 8)
        s_codes[sp:sp + len(s)] = s
        s_fids[sp:sp + len(s)] = e["file_id"]
        sp += len(s)
        f = d["subfp_fine"].reshape(-1, 24) if d["subfp_fine"].size else np.empty((0, 24), np.uint8)
        f_codes[fp:fp + len(f)] = f
        f_fids[fp:fp + len(f)] = e["file_id"]
        fp += len(f)
        raw_vecs[li] = d["raw_vec"].reshape(-1)
        n_hashes[li] = h
        file_ids[li] = e["file_id"]
        li += 1
    sp_real = sp
    next_fid = max(int(e["file_id"]) for e in manifest) + 1
    dist_entries = []
    j = 0
    for z, take in dz:
        nh, th, sn, ts, fn, tf = dist_parts(z, take)
        hashes[hp:hp + th] = z["hashes"][:th]
        times[hp:hp + th] = z["times"][:th]
        lids[hp:hp + th] = np.repeat(np.arange(li, li + take, dtype=np.int32), nh)
        hp += th
        fids = np.arange(next_fid + j, next_fid + j + take, dtype=np.int64)
        s_codes[sp:sp + ts] = z["subfp"][:ts]
        s_fids[sp:sp + ts] = np.repeat(fids, sn)
        sp += ts
        f_codes[fp:fp + tf] = z["fine"][:tf]
        f_fids[fp:fp + tf] = np.repeat(fids, fn)
        fp += tf
        raw_vecs[li:li + take] = z["raw_vecs"][:take]
        n_hashes[li:li + take] = nh
        file_ids[li:li + take] = fids
        for k in range(take):
            dist_entries.append({"file_id": int(fids[k]), "filename": f"distractor_{j + k:07d}.wav",
                                 "path": f"/nonexistent/m1_latency_distractors/distractor_{j + k:07d}.wav",
                                 "category": "audio", "duration_sec": float(z["dur"][k]),
                                 "fp_file": "", "source_folder": "latency-benchmark-distractors"})
        li += take
        j += take
    del dz, dlt
    gc.collect()
    assert hp == n_hash and sp == n_sub and fp == n_fine and li == n_total
    stats = {"n_files": n_total, "n_real": n_real, "n_distractors": n_dist, "landmark_hashes_raw": n_hash,
             "subfp_windows": n_sub, "fine_windows": n_fine,
             # realism check: per-file fingerprint statistics, real entries vs distractors
             "real_hashes_per_file": round(float(n_hashes[:n_real].mean()), 1),
             "real_windows_per_file": round(sp_real / n_real, 1)}
    if n_dist:
        stats["distractor_hashes_per_file"] = round(float(n_hashes[n_real:].mean()), 1)
        stats["distractor_windows_per_file"] = round((n_sub - sp_real) / n_dist, 1)

    prev = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(max(prev, os.cpu_count() or 1))
    t0 = time.perf_counter()
    try:
        idx = M._assemble_domain_index(file_ids, n_hashes, hashes, lids, times, raw_vecs, None,
                                       s_fids, s_codes, f_fids, f_codes)
    finally:
        faiss.omp_set_num_threads(prev)
    stats["build_sec"] = round(time.perf_counter() - t0, 1)
    stats["faiss_query_threads"] = faiss.omp_get_max_threads()
    stats["landmark_hashes_kept"] = int(idx.landmark_index.sorted_hash.size)
    del hashes, lids, times, raw_vecs, s_fids, s_codes, f_fids, f_codes
    gc.collect()
    return idx, dist_entries, stats


# ----------------------------------------------------------------------------- stage profiler
class StageProfiler:
    """Exclusive (self) time per wrapped function, nesting-aware, single-threaded use."""

    def __init__(self):
        self.reset()
        self.last_result = None

    def reset(self):
        self.excl = defaultdict(float)
        self.incl = defaultdict(float)
        self.stack = []

    def wrap(self, name, fn):
        prof = self

        @functools.wraps(fn)
        def w(*a, **k):
            prof.stack.append(0.0)
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t
                child = prof.stack.pop()
                prof.excl[name] += dt - child
                prof.incl[name] += dt
                if prof.stack:
                    prof.stack[-1] += dt
        return w


def install_profiler():
    import sonic.retrieval.pipeline as pl
    import sonic_db_manager as sdm
    from sonic.index.landmark_index import LandmarkIndex
    from sonic.index.subfingerprint_index import SubFingerprintIndex

    prof = StageProfiler()
    for name, attr in [("decode", "load_query_window"), ("feature", "stft_magnitude"),
                       ("feature", "build_landmarks_from_magnitude"), ("primary_label", "_label_evidence"),
                       ("escalation_other", "_verify_escalation"), ("vote", "_vote_offsets"),
                       ("align", "_coarse_align"), ("align", "_finish_align"), ("reverse_check", "_reverse_evidence")]:
        setattr(pl, attr, prof.wrap(name, getattr(pl, attr)))
    pl._QueryCodes.get = prof.wrap("hyp_codes", pl._QueryCodes.get)
    pl._QueryCodes.__init__ = prof.wrap("hyp_codes", pl._QueryCodes.__init__)
    pl._HypCodes.fine = prof.wrap("hyp_codes", pl._HypCodes.fine)
    LandmarkIndex.query = prof.wrap("landmark_search", LandmarkIndex.query)
    SubFingerprintIndex.search_matches = prof.wrap("hnsw_search", SubFingerprintIndex.search_matches)

    orig_query = sdm.query

    def query_capture(*a, **k):
        r = orig_query(*a, **k)
        prof.last_result = r
        return r
    sdm.query = prof.wrap("pipeline_other", query_capture)
    return prof


STAGE_KEYS = ["decode", "feature", "landmark_search", "primary_label", "hyp_codes", "hnsw_search", "vote",
              "align", "reverse_check", "escalation_other", "pipeline_other"]


# ----------------------------------------------------------------------------- run + stats
def pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def lat_summary(v):
    a = np.asarray(v, float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": round(float(a.mean()), 1), "p50": pct(a, 50), "p95": pct(a, 95),
            "p99": pct(a, 99), "max": round(float(a.max()), 1)}


def run_queries(app_mod, prof, queries, order, warmup):
    """warmup < 0: one full untimed pass over every query first (pages the whole
    working set of the index back in after the build's memory peak)."""
    rows = []
    for i in (order if warmup < 0 else order[:warmup]):
        app_mod.run_duplicate_detection([Path(queries[i]["path"])], "auto")
    for n, i in enumerate(order):
        q = queries[i]
        if prof is not None:
            prof.reset()
            prof.last_result = None
        t0 = time.perf_counter()
        body = app_mod.run_duplicate_detection([Path(q["path"])], "auto")
        wall = (time.perf_counter() - t0) * 1000
        res = (body.get("results") or [{}])[0]
        matched = bool(res.get("matched"))
        row = {"i": i, "cat": q["cat"], "cond": q["cond"], "domain": q["domain"], "wall_ms": round(wall, 2),
               "matched": matched, "matched_filename": res.get("matched_filename"),
               "correct": (not matched) if q["cat"] == "negative" else bool(matched and res.get("matched_filename") == q["true_filename"]),
               "match_type": res.get("match_type"), "stage_used": res.get("stage_used")}
        if prof is not None:
            st = {k: round(prof.excl.get(k, 0.0) * 1000, 3) for k in STAGE_KEYS}
            pipeline_ms = prof.incl.get("pipeline_other", 0.0) * 1000
            st["app_overhead"] = round(wall - pipeline_ms, 3)
            row["stages"] = st
            r = prof.last_result
            if r is not None:
                row["escalated"] = bool(r.escalated)
                row["n_hypotheses"] = int((r.timing_ms or {}).get("n_hypotheses", 0))
                row["stage_used"] = r.stage_used
        rows.append(row)
        if (n + 1) % 100 == 0:
            print(f"    {n + 1}/{len(order)} queries", flush=True)
    return rows


def summarize_rows(rows):
    out = {}
    groups = {c: [r for r in rows if r["cat"] == c] for c in CATS}
    groups["all"] = rows
    for g, rs in groups.items():
        if not rs:
            continue
        s = {"latency_ms": lat_summary([r["wall_ms"] for r in rs])}
        if g == "negative":
            s["false_positives"] = sum(r["matched"] for r in rs)
            s["fpr"] = round(s["false_positives"] / len(rs), 4)
        elif g in ("duplicate", "edited"):
            s["top1_correct"] = sum(r["correct"] for r in rs)
            s["top1_acc"] = round(s["top1_correct"] / len(rs), 4)
            s["wrong_match"] = sum(1 for r in rs if r["matched"] and not r["correct"])
            s["distractor_match"] = sum(1 for r in rs if str(r.get("matched_filename") or "").startswith("distractor_"))
        if g == "negative":
            s["distractor_match"] = sum(1 for r in rs if str(r.get("matched_filename") or "").startswith("distractor_"))
        if rs and "stages" in rs[0]:
            s["stages_mean_ms"] = {k: round(float(np.mean([r["stages"][k] for r in rs])), 2)
                                   for k in STAGE_KEYS + ["app_overhead"]}
        if rs and "escalated" in rs[0]:
            s["escalated_frac"] = round(float(np.mean([r.get("escalated", False) for r in rs])), 3)
            s["mean_hypotheses"] = round(float(np.mean([r.get("n_hypotheses", 0) for r in rs])), 2)
        out[g] = s
    per_cond = {}
    for c in sorted({r["cond"] for r in rows if r["cat"] == "edited"}):
        rs = [r for r in rows if r["cond"] == c]
        per_cond[c] = {"n": len(rs), "acc": round(sum(r["correct"] for r in rs) / len(rs), 4),
                       "mean_ms": round(float(np.mean([r["wall_ms"] for r in rs])), 1)}
    out["edited_per_condition"] = per_cond
    return out


def landmark_posting_stats(lidx, qlms):
    tot = []
    for lm in qlms:
        if lm.hash.size == 0:
            continue
        lo = np.searchsorted(lidx.sorted_hash, lm.hash, side="left")
        hi = np.searchsorted(lidx.sorted_hash, lm.hash, side="right")
        tot.append(int((hi - lo).sum()))
    return round(float(np.mean(tot)), 1) if tot else None


def query_landmarks(queries):
    from sonic.io import load_query_window
    from sonic.fingerprint.landmarks import stft_magnitude, build_landmarks_from_magnitude
    out = []
    for q in queries:
        y = load_query_window(q["path"], 5.0)
        mag, ft = stft_magnitude(y, SR)
        out.append(build_landmarks_from_magnitude(mag, ft, SR))
    return out


def landmark_micro(sizes, qlms, reps=3):
    """LandmarkIndex.query cost as n_files grows, real 28k posting lists, n_files padded."""
    import psutil
    from sonic.index.landmark_index import LandmarkIndex
    from sonic_db_manager import SONICDatabaseManager as M  # noqa: F401  (same package state)
    arch = np.load(STORE / "unified_archive.npz")
    base = LandmarkIndex()
    fids = arch["file_ids"].astype(np.int64)
    base.file_id_to_local = {int(f): i for i, f in enumerate(fids)}
    base.n_hashes_per_file = {int(f): int(n) for f, n in zip(fids, arch["n_hashes"])}
    base.build_from_flat(arch["hashes"], arch["lids"], arch["times"])
    del arch
    out = {}
    for n in sizes:
        need = n * base.n_bins * 8 * 1.3
        avail = psutil.virtual_memory().available
        if need > 0.7 * avail:
            out[str(n)] = {"skipped": f"needs ~{need / 1e9:.1f} GB transient, {avail / 1e9:.1f} GB available"}
            continue
        base.n_files = n
        base.local_to_file_id = np.concatenate([fids, np.full(max(0, n - fids.size), -1, np.int64)])
        ts = []
        lquery = getattr(LandmarkIndex.query, "__wrapped__", LandmarkIndex.query)  # bypass the profiler
        for _ in range(reps):
            for lm in qlms:
                t = time.perf_counter()
                lquery(base, lm, top_k=20)  # PRIMARY_POOL_K, as in query_array
                ts.append((time.perf_counter() - t) * 1000)
        out[str(n)] = {"mean_ms": round(float(np.mean(ts)), 2), "p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95),
                       "hist_bytes_per_query": int(n * base.n_bins * 8)}
        print(f"[landmark micro] n_files={n}: mean {out[str(n)]['mean_ms']} ms", flush=True)
    return out


# ----------------------------------------------------------------------------- fit / extrapolate
def _fit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else 1.0
    return [float(coef[0]), float(coef[1])], r2


def extrapolate(size_results, micro, targets):
    """Per category: mean(N) = flat part + landmark(N) + rest(N).
    landmark(N): linear in n_files (fit on measured sizes; micro-benchmark is the cross-check).
    rest(N) = everything else (decode, features, escalation incl. HNSW, decision):
    fitted a + b ln N (central) and a * N^k (power law, upper bound)."""
    ns = np.array([r["size"] for r in size_results], float)
    out = {"targets": targets, "categories": {}}
    if len(ns) < 2:
        return out
    for g in list(CATS) + ["all"]:
        if any(g not in r["summary"] for r in size_results):
            continue
        tot = np.array([r["summary"][g]["latency_ms"]["mean"] for r in size_results])
        p95 = np.array([r["summary"][g]["latency_ms"]["p95"] for r in size_results])
        lm = np.array([r["summary"][g]["stages_mean_ms"]["landmark_search"] for r in size_results])
        rest = tot - lm
        (lm_a, lm_b), lm_r2 = _fit(ns, lm)
        (ra, rb), r_r2 = _fit(np.log(ns), rest)
        (pa, pk), p_r2 = _fit(np.log(ns), np.log(np.maximum(rest, 1e-3)))
        (ta, tb), t_r2 = _fit(np.log(ns), tot)
        (qa, qb), q_r2 = _fit(np.log(ns), p95)
        (qla, qlb), ql_r2 = _fit(ns, p95)
        pred = {}
        for t in targets:
            lm_t = lm_a + lm_b * t
            log_t = ra + rb * np.log(t)
            pow_t = float(np.exp(pa) * t ** pk)
            pred[str(t)] = {"landmark_ms": round(lm_t, 1), "rest_log_ms": round(log_t, 1),
                            "rest_power_ms": round(pow_t, 1),
                            "mean_central_ms": round(lm_t + log_t, 1), "mean_upper_ms": round(lm_t + max(log_t, pow_t), 1),
                            "mean_naive_loglinear_ms": round(ta + tb * np.log(t), 1),
                            "p95_loglinear_ms": round(qa + qb * np.log(t), 1),
                            "p95_linear_ms": round(qla + qlb * t, 1)}
        out["categories"][g] = {
            "fit": {"landmark_linear": {"a_ms": round(lm_a, 3), "b_ms_per_file": lm_b, "r2": round(lm_r2, 3)},
                    "rest_log": {"a_ms": round(ra, 2), "b_ms_per_lnN": round(rb, 2), "r2": round(r_r2, 3)},
                    "rest_power": {"k": round(pk, 3), "r2": round(p_r2, 3)},
                    "total_log": {"a_ms": round(ta, 2), "b_ms_per_lnN": round(tb, 2), "r2": round(t_r2, 3)}},
            "pred": pred}
    if micro:
        ok = {int(k): v["mean_ms"] for k, v in micro.items() if "mean_ms" in v}
        if len(ok) >= 2:
            xs = np.array(sorted(ok), float)
            (ma, mb), mr2 = _fit(xs, np.array([ok[int(x)] for x in xs]))
            out["landmark_micro_fit"] = {"a_ms": round(ma, 3), "b_ms_per_file": mb, "r2": round(mr2, 4),
                                         "pred": {str(t): round(ma + mb * t, 1) for t in targets}}
    return out


def to_markdown(meta, size_results, extra):
    L = [f"# Module 1 latency vs reference-DB size ({meta['started']})", "",
         f"- Query path: {meta['query_path']}; machine: {meta['machine']}",
         f"- Queries per size: {meta['n_queries']} ({meta['query_counts']})", "",
         "| DB files | category | n | mean ms | p50 | p95 | p99 | top-1 acc / FPR | escalated | RAM GB (footprint, else RSS) | build s |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in size_results:
        for g in list(CATS) + ["all"]:
            s = r["summary"].get(g)
            if not s:
                continue
            l = s["latency_ms"]
            acc = (f"FPR {100 * s['fpr']:.1f}% ({s['false_positives']}/{l['n']})" if g == "negative"
                   else f"{100 * s['top1_acc']:.1f}%" if "top1_acc" in s else "-")
            L.append(f"| {r['size']:,} | {g} | {l['n']} | {l['mean']} | {l['p50']} | {l['p95']} | {l['p99']} | {acc} | "
                     f"{s.get('escalated_frac', '-')} | {r.get('footprint_gb_after_build') or r['rss_gb_after_build']} | {r['build']['build_sec']} |")
    L += ["", "Stage means (ms, exclusive):", "",
          "| DB files | category | " + " | ".join(STAGE_KEYS + ["app_overhead"]) + " |",
          "|---:|---|" + "---:|" * (len(STAGE_KEYS) + 1)]
    for r in size_results:
        for g in list(CATS) + ["all"]:
            s = r["summary"].get(g)
            if s and "stages_mean_ms" in s:
                L.append(f"| {r['size']:,} | {g} | " + " | ".join(str(s["stages_mean_ms"][k]) for k in STAGE_KEYS + ["app_overhead"]) + " |")
    ex = extra.get("extrapolation", {})
    if ex.get("categories"):
        L += ["", "Extrapolation (NOT measured): mean ms = landmark(linear in N) + rest(a + b ln N); "
                  "upper = landmark + max(log, power-law) rest", "",
              "| category | " + " | ".join(f"{int(t):,} central / upper" for t in ex["targets"]) + " |",
              "|---|" + "---:|" * len(ex["targets"])]
        for g, v in ex["categories"].items():
            L.append(f"| {g} | " + " | ".join(f"{v['pred'][str(t)]['mean_central_ms']} / {v['pred'][str(t)]['mean_upper_ms']}"
                                             for t in ex["targets"]) + " |")
    if extra.get("landmark_micro"):
        L += ["", "Landmark-stage micro-benchmark (real 28k postings, n_files padded):", ""]
        for k, v in extra["landmark_micro"].items():
            L.append(f"- n_files {int(k):,}: {v}")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- HTTP mode
def http_mode(args, b20, manifest, qdir):
    import requests
    queries, _excl, _nt, _ne = prepare_queries(b20, manifest, args, qdir)
    base = args.http_url.rstrip("/")
    t0 = time.time()
    while True:
        try:
            st = requests.get(f"{base}/api/database/status", timeout=10).json()
            if st.get("attached"):
                break
        except Exception:
            pass
        if time.time() - t0 > 900:
            raise SystemExit("server DB not attached")
        time.sleep(2)
    cache = PRISM_ROOT / "uploads" / "latest_check_result.json"
    backup = qdir / "latest_check_result.json.bak"
    had = cache.exists()
    if had:
        shutil.copyfile(cache, backup)
    order = list(range(len(queries)))
    random.Random(args.seed).shuffle(order)
    rows = []
    sess = requests.Session()
    warm = order if args.warmup < 0 else order[:args.warmup]
    try:
        for i in warm + order:
            q = queries[i]
            p = Path(q["path"])
            t = time.perf_counter()
            with open(p, "rb") as fh:
                r = sess.post(f"{base}/api/check", data={"module": "duplicate_detection", "domain": "auto"},
                              files={"audio_files": (p.name, fh)}, timeout=120)
            wall = (time.perf_counter() - t) * 1000
            body = r.json()
            res = (body.get("results") or [{}])[0]
            rows.append({"i": i, "cat": q["cat"], "cond": q["cond"], "domain": q["domain"],
                         "wall_ms": round(wall, 2), "server_query_ms": res.get("query_time_ms"),
                         "server_total_ms": body.get("total_time_ms"), "matched": bool(res.get("matched")),
                         "matched_filename": res.get("matched_filename"),
                         "correct": (not res.get("matched")) if q["cat"] == "negative"
                         else bool(res.get("matched") and res.get("matched_filename") == q["true_filename"])})
    finally:
        if had:
            shutil.copyfile(backup, cache)
        elif cache.exists():
            cache.unlink()
    rows = rows[len(warm):]
    summ = summarize_rows(rows)
    for g in list(CATS) + ["all"]:
        rs = rows if g == "all" else [r for r in rows if r["cat"] == g]
        if rs:
            summ[g]["server_query_ms"] = lat_summary([r["server_query_ms"] for r in rs if r["server_query_ms"] is not None])
    return {"mode": "http", "url": base, "db_count": st.get("count"), "summary": summ, "rows": rows}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="28409,50000,75000,100000",
                    help="comma-separated DB sizes (files). The real DB size uses the app's own index; "
                         "larger sizes add distractors.")
    ap.add_argument("--dup-per-domain", type=int, default=34, help="duplicate-query tracks per domain (+ user song)")
    ap.add_argument("--edit-per-domain", type=int, default=10, help="tracks per domain that get the 12 edits")
    ap.add_argument("--negatives-per-source", type=int, default=34)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=-1,
                    help="untimed warm-up queries per size; -1 (default) = one full pass over the query set")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--max-distractor-sec", type=float, default=600.0)
    ap.add_argument("--echoes-root", default=str(DOWNLOADS / "Echoes"))
    ap.add_argument("--tmp-dir", default="/tmp/m1_latency_scale")
    ap.add_argument("--keep-tmp", action="store_true", help="keep queries + distractor cache in --tmp-dir")
    ap.add_argument("--landmark-micro-sizes", default="28409,100000,250000,500000,1000000,2000000")
    ap.add_argument("--extrapolate-to", default="250000,500000,1000000,2000000")
    ap.add_argument("--no-stage-profile", action="store_true")
    ap.add_argument("--http-url", default=None, help="HTTP mode: POST the query set to a running server instead")
    ap.add_argument("--out-json", default=str(SONIC_ROOT / "results" / "module1_latency_scale" / "latency.json"))
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    qdir = Path(tempfile.mkdtemp(prefix="queries_", dir=str(tmp)))
    b20 = load_bench20()
    manifest = read_manifest()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    machine = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    machine += f", {os.cpu_count()} cores, {int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout) / 2**30:.0f} GiB"

    try:
        if args.http_url:
            res = http_mode(args, b20, manifest, qdir)
            res["meta"] = {"started": started, "machine": machine}
            json.dump(res, open(out_json, "w"), indent=1)
            print(json.dumps({g: res["summary"][g]["latency_ms"] for g in list(CATS) + ["all"] if g in res["summary"]}, indent=1))
            print(f"wrote {out_json}")
            return

        sizes = sorted(int(s) for s in args.sizes.split(",") if s)
        real_n = len(manifest)
        queries, exclude, n_tracks, n_edit_tracks = prepare_queries(b20, manifest, args, qdir)
        counts = dict(Counter(q["cat"] for q in queries))
        print(f"[queries] {len(queries)} total {counts}", flush=True)

        dstats = None
        ddir = tmp / "distractors"
        if max(sizes) > real_n:
            dstats = generate_distractors(b20, manifest, exclude, max(sizes) - real_n, args, ddir)
            print(f"[distractors] {dstats}", flush=True)

        t_imp = time.time()
        import app as app_mod  # attaches the real DB in a background thread
        import faiss
        db = app_mod.sonic_db
        while not db.get_status()["attached"]:
            if time.time() - t_imp > 3600:
                raise SystemExit("reference DB did not attach")
            time.sleep(0.5)
        attach_sec = round(time.time() - t_imp, 1)
        print(f"[app] imported + DB attached in {attach_sec}s ({len(db.manifest)} files), "
              f"faiss query threads {faiss.omp_get_max_threads()}", flush=True)
        prof = None if args.no_stage_profile else install_profiler()
        qlms = query_landmarks(queries)

        order = list(range(len(queries)))
        random.Random(args.seed).shuffle(order)
        real_manifest = list(db.manifest)
        size_results = []
        for n in sizes:
            print(f"\n=== size {n:,} ===", flush=True)
            if n == real_n:
                idx = db.domain_index
                build = {"n_files": n, "n_real": n, "n_distractors": 0, "build_sec": attach_sec,
                         "note": "app's own DB attach (import + archive/delta attach)",
                         "landmark_hashes_kept": int(idx.landmark_index.sorted_hash.size),
                         "subfp_windows": int(idx.subfp_index.window_file_ids.size)}
                dist_entries = []
            else:
                with db._lock:
                    db.domain_index = None
                    db.manifest = list(real_manifest)
                gc.collect()
                idx, dist_entries, build = assemble_scaled(real_manifest, n, ddir)
                with db._lock:
                    db.manifest = real_manifest + dist_entries
                    db.domain_index = idx
                    db._refresh_manifest_lookup()
            gc.collect()
            rss_after = round(rss_gb(), 2)
            print(f"[size {n:,}] build {build['build_sec']}s, RSS {rss_after} GB, peak {peak_rss_gb():.2f} GB", flush=True)
            fp_after = proc_rusage_info()
            mean_post = landmark_posting_stats(idx.landmark_index, qlms)
            tw, pw0, vw0 = time.time(), proc_rusage_info(), vm_counters()
            for i in (order if args.warmup < 0 else order[:args.warmup]):
                app_mod.run_duplicate_detection([Path(queries[i]["path"])], "auto")
            pw1, vw1 = proc_rusage_info(), vm_counters()
            warm = {"sec": round(time.time() - tw, 1), "proc_pageins": pw1.get("pageins", 0) - pw0.get("pageins", 0),
                    **{f"system_{k}": vw1.get(k, 0) - vw0.get(k, 0) for k in vw1}}
            print(f"[size {n:,}] warm-up {warm}", flush=True)
            ru0, vm0, pi0 = resource.getrusage(resource.RUSAGE_SELF), vm_counters(), proc_rusage_info()
            t0 = time.time()
            rows = run_queries(app_mod, prof, queries, order, 0)
            ru1, vm1, pi1 = resource.getrusage(resource.RUSAGE_SELF), vm_counters(), proc_rusage_info()
            summ = summarize_rows(rows)
            size_results.append({
                "size": n, "build": build, "rss_gb_after_build": rss_after, "peak_rss_gb_so_far": round(peak_rss_gb(), 2),
                "footprint_gb_after_build": fp_after.get("footprint_gb"),
                "footprint_gb_after_queries": pi1.get("footprint_gb"), "warmup_phase": warm,
                "index_nbytes_estimate_gb": round(idx.nbytes() / 1e9, 3),
                "landmark_postings_per_query_mean": mean_post, "query_phase_sec": round(time.time() - t0, 1),
                "query_phase_paging": {"proc_majflt": ru1.ru_majflt - ru0.ru_majflt,
                                       "proc_minflt": ru1.ru_minflt - ru0.ru_minflt,
                                       "proc_pageins": pi1.get("pageins", 0) - pi0.get("pageins", 0),
                                       **{f"system_{k}": vm1.get(k, 0) - vm0.get(k, 0) for k in vm1}},
                "summary": summ, "rows": rows})
            for g in list(CATS) + ["all"]:
                s = summ[g]
                extra = (f"FPR {s['fpr']}" if g == "negative" else f"acc {s.get('top1_acc')}") if g != "all" else ""
                print(f"  {g:9s} {s['latency_ms']}  {extra}  esc {s.get('escalated_frac')}", flush=True)
            json.dump({"partial": True, "size_results": size_results}, open(out_json, "w"), indent=1)
            del idx
        # free the last scaled index before the landmark micro-benchmark
        with db._lock:
            db.domain_index = None
            db.manifest = list(real_manifest)
        gc.collect()
        micro = landmark_micro([int(s) for s in args.landmark_micro_sizes.split(",") if s],
                               [qlms[i] for i, q in enumerate(queries) if q["cat"] in ("duplicate", "negative")])
        targets = [int(s) for s in args.extrapolate_to.split(",") if s]
        extra = {"landmark_micro": micro, "extrapolation": extrapolate(size_results, micro, targets)}
        meta = {"started": started, "machine": machine, "query_path": "app.run_duplicate_detection (in-process)",
                "sizes": sizes, "real_db_files": real_n, "n_queries": len(queries), "query_counts": counts,
                "n_tracks": n_tracks, "n_edit_tracks": n_edit_tracks, "edit_conditions": EDIT_CONDITIONS,
                "seed": args.seed, "warmup": args.warmup, "stage_profile": prof is not None,
                "distractors": dstats, "app_attach_sec": attach_sec}
        json.dump({"meta": meta, "size_results": size_results, **extra}, open(out_json, "w"), indent=1)
        md = to_markdown(meta, size_results, extra)
        out_md.write_text(md, encoding="utf-8")
        print("\n" + md)
        print(f"wrote {out_json} and {out_md}")
    finally:
        shutil.rmtree(qdir, ignore_errors=True)
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
