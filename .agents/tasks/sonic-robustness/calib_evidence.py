"""Calibration / experiment harness for the Module 1 verification stage.

One process, one DB attach (SONICDatabaseManager on the real store, read-only).
Generates DEV positives (seed 7 by default, never the held-out seed 42) with the
benchmark's own edit generator, plus >= 500 calibration negatives that are
disjoint from the benchmark's seed-42 negatives, then for every named config
runs sonic.retrieval.pipeline.query() with acceptance thresholds wide open and
early exit off, dumping per-candidate evidence to experiments/<config>.json.

--sweep <dump.json> grid-searches the acceptance thresholds on a dump and
prints dev accuracy at 0 calibration false positives.

Usage (from prism_v2/):
  .venv/bin/python .agents/tasks/sonic-robustness/calib_evidence.py --configs baseline --per-domain 4 --negatives 60
  .venv/bin/python .agents/tasks/sonic-robustness/calib_evidence.py --sweep .agents/tasks/sonic-robustness/experiments/baseline.json
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import itertools
import json
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "SONIC-fixed_1")]
EXP_DIR = HERE / "experiments"
SR = 22050
MAX_QUERY_FILE_SEC = 40.0  # queries read <= 5 s (+30 s silence lookahead), so longer files are truncated

_spec = importlib.util.spec_from_file_location("bench", ROOT / "SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

# name -> (RetrievalParams overrides, extra settings)
CONFIGS = {
    "baseline": ({}, {}),
    "win8": ({}, {"window_sec": 8.0}),
    "win10": ({}, {"window_sec": 10.0}),
    "hop8": ({"query_hop_frames": 8}, {}),
    "hop26": ({"query_hop_frames": 26}, {}),
    "ef16": ({}, {"ef_search": 16}),
    "ef128": ({}, {"ef_search": 128}),
    "ef256": ({}, {"ef_search": 256}),
    "topk50": ({"subfp_top_k": 50}, {}),
    "topn10": ({"verify_top_n": 10}, {}),
    "fine_identity": ({"fine_search": "identity"}, {}),
    "maxd20": ({"vote_max_dist": 20}, {}),
    "maxd24": ({"vote_max_dist": 24}, {}),
    "tempo_alt": ({"tempo_hyps": (0.85, 0.92, 1.08, 1.17)}, {}),
    "tempo_none": ({"tempo_hyps": ()}, {}),
    "tempo6": ({"tempo_hyps": (0.8, 0.9, 0.95, 1.05, 1.1, 1.2)}, {}),
    "vhop4": ({"verify_hop_frames": 4, "query_hop_frames": 16}, {}),
    "pitch_no3": ({"pitch_hyps": (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)}, {}),
    "joint": ({"joint_hyps": None}, {}),  # filled from JOINT_VARISPEED_HYPOTHESES
    "final": ({}, {"locked": True}),
}

OPEN = dict(accept_coarse_ber=1.0, accept_fine_ber=1.0, strong_fine_ber=-1.0, margin_ber=0.0, min_aligned_windows=1,
            min_contrast=None, min_contrast_long=None, single_window_ber=1.0)


def benchmark_negative_names(manifest, echoes_root: Path, n_per_source=40, seed=42):
    """Replays B.build_negatives' RNG to get the held-out negatives' source names (no files written)."""
    import librosa
    names = {e["filename"] for e in manifest}
    rng = random.Random(f"{seed}|neg")
    tc = sorted(p for p in (echoes_root / "_negatives").rglob("*.flac") if p.name not in names)
    rng.shuffle(tc)
    used_tc = {p.name for p in tc[:n_per_source]}
    seg_paths = []
    for p in sorted(B.DOWNLOADS.glob("*.mp3")):
        if p.name in names or "Chaleya" in p.name:
            continue
        try:
            total = librosa.get_duration(path=str(p))
        except Exception:
            continue
        for k, off in enumerate(np.arange(5.0, max(total - 30.0, 5.0), 30.0)):
            seg_paths.append((p, float(off), k))
    rng.shuffle(seg_paths)
    music = [s for s in seg_paths if "talk" not in s[0].name.lower()][:n_per_source]
    talk = [s for s in seg_paths if "talk" in s[0].name.lower()][: max(1, n_per_source // 2)]
    used_segs = {(str(p), k) for p, _o, k in music + talk}
    esc = sorted((e for e in manifest if "ESC-50" in e["path"]), key=lambda e: e["file_id"])
    rng.shuffle(esc)
    used_esc = {e["filename"] for e in esc[:n_per_source]}
    return used_tc, used_segs, used_esc, seg_paths


def build_calib_negatives(manifest, echoes_root: Path, out_dir: Path, n_total: int, seed: int = 7):
    import librosa
    used_tc, used_segs, used_esc, seg_paths = benchmark_negative_names(manifest, echoes_root)
    names = {e["filename"] for e in manifest}
    rng = random.Random(f"{seed}|calibneg")
    scale = n_total / 500.0
    n_ls, n_gt, n_esc, n_esc2, n_seg, n_ed = (max(1, int(round(x * scale))) for x in (200, 80, 80, 40, 40, 30))
    negs = []
    tc = sorted(p for p in (echoes_root / "_negatives").rglob("*.flac") if p.name not in names and p.name not in used_tc)
    rng.shuffle(tc)
    for p in tc[:n_ls]:
        negs.append({"source": "librispeech_test_clean", "path": p, "y": None})
    gt = sorted((e for e in manifest if "genres" in e["path"]), key=lambda e: e["file_id"])
    rng.shuffle(gt)
    k = 0
    for e in gt:
        if k >= n_gt:
            break
        lp = B.resolve_local(e["path"], echoes_root)
        if lp is None:
            continue
        y, _ = librosa.load(str(lp), sr=SR, mono=True)
        negs.append({"source": "gtzan_time_reversed", "y": y[::-1].copy(), "name": f"rev_{Path(e['filename']).stem}",
                     "src_fid": int(e["file_id"])})
        k += 1
    esc = sorted((e for e in manifest if "ESC-50" in e["path"] and e["filename"] not in used_esc), key=lambda e: e["file_id"])
    rng.shuffle(esc)
    k = 0
    for e in esc:
        if k >= n_esc + n_esc2:
            break
        lp = B.resolve_local(e["path"], echoes_root)
        if lp is None:
            continue
        y, _ = librosa.load(str(lp), sr=SR, mono=True)
        y = y[::-1].copy()
        if k < n_esc:
            negs.append({"source": "esc50_time_reversed", "y": y, "name": f"rev_{Path(e['filename']).stem}",
                         "src_fid": int(e["file_id"])})
        else:
            negs.append({"source": "esc50_reversed_2s", "y": y[: 2 * SR], "name": f"rev2_{Path(e['filename']).stem}",
                         "src_fid": int(e["file_id"])})
        k += 1
    segs = [s for s in seg_paths if (str(s[0]), s[2]) not in used_segs]
    rng.shuffle(segs)
    for p, off, kk in segs[:n_seg]:
        y, _ = librosa.load(str(p), sr=SR, mono=True, offset=off, duration=30.0)
        if y.size >= SR:
            negs.append({"source": "local_mp3_segment_unused", "y": y, "name": f"seg{kk}_{p.stem[:30]}"})
    # edited negatives: the hypothesis grid itself creates FP opportunities
    base_ls = [n for n in negs if n["source"] == "librispeech_test_clean"][:n_ed]
    base_gt = [n for n in negs if n["source"] == "gtzan_time_reversed"][:n_ed]
    for n in base_ls + base_gt:
        y = n["y"] if n["y"] is not None else librosa.load(str(n["path"]), sr=SR, mono=True)[0]
        for kind, prm in (("pitch", 2), ("pitch", -2), ("pitch", 4), ("pitch", -4), ("speed", -20), ("speed", 20)):
            if kind == "pitch":
                z = librosa.effects.pitch_shift(y=y[: int(MAX_QUERY_FILE_SEC * SR)], sr=SR, n_steps=float(prm))
            else:
                z = librosa.effects.time_stretch(y=y[: int(MAX_QUERY_FILE_SEC * SR)], rate=1.0 + prm / 100.0)
            negs.append({"source": f"edited_neg_{kind}", "y": z.astype(np.float32), "src_fid": n.get("src_fid"),
                         "name": f"{kind}{prm:+d}_{n.get('name') or Path(n['path']).stem}"})
    out = []
    for i, n in enumerate(negs):
        if n.get("y") is None:
            out.append({"source": n["source"], "path": n["path"]})
            continue
        dst = out_dir / f"neg{i}_{n['name'][:50]}.wav"
        sf.write(str(dst), np.asarray(n["y"][: int(MAX_QUERY_FILE_SEC * SR)], dtype=np.float32), SR, subtype="FLOAT")
        out.append({"source": n["source"], "path": dst, "src_fid": n.get("src_fid")})
    return out


def build_positives(manifest, echoes_root: Path, out_dir: Path, per_domain: int, seed: int, conditions, varispeed):
    import librosa
    tracks, _ = B.select_tracks(manifest, echoes_root, per_domain, seed)
    rng = np.random.default_rng(seed)
    pos = []
    for ti, t in enumerate(tracks):
        y, _ = librosa.load(str(t["local"]), sr=SR, mono=True)
        y = y.astype(np.float32)
        tdir = out_dir / f"t{ti}"
        tdir.mkdir()
        conds = list(conditions) + (["varispeed_-20pct", "varispeed_-10pct", "varispeed_-5pct", "varispeed_+5pct",
                                     "varispeed_+10pct", "varispeed_+20pct"] if varispeed else [])
        for c in conds:
            if c.startswith("varispeed"):
                p = int(c.split("_")[1].replace("pct", ""))
                z = librosa.resample(y, orig_sr=SR, target_sr=int(round(SR / (1 + p / 100.0))))
                qp = tdir / f"q_{c}_{ti}.wav"
                sf.write(str(qp), z[: int(MAX_QUERY_FILE_SEC * SR)].astype(np.float32), SR, subtype="FLOAT")
            else:
                kind, param = B.CONDITIONS[c]
                yy = y
                if kind in ("pitch", "speed", "noise", "gain", "highpass", "shift_pad") and y.size > 60 * SR:
                    yy = y[: int(60 * SR)]  # long user track: edits only need the first minute
                qp = B.make_edit(kind, param, yy, t["local"], tdir, f"{c}_{ti}", rng)
            if qp is None:
                continue
            pos.append({"condition": c, "domain": t["domain"], "true_fid": int(t["entry"]["file_id"]), "path": qp})
        print(f"  generated track {ti + 1}/{len(tracks)}", flush=True)
    return pos


def compact(res, true_fid=None, keep=6):
    evs = sorted(res.candidate_evidence.items(), key=lambda kv: kv[1].get("score", 9))
    top = [{"fid": int(f), **{k: ev.get(k) for k in ("coarse_ber", "fine_ber", "score", "n_aligned", "semitones", "tempo", "offset_sec", "contrast", "reverse_score")}}
           for f, ev in evs[:keep]]
    true_ev = None
    if true_fid is not None and true_fid in res.candidate_evidence:
        ev = res.candidate_evidence[true_fid]
        true_ev = {k: ev.get(k) for k in ("coarse_ber", "fine_ber", "score", "n_aligned", "semitones", "tempo", "contrast", "reverse_score")}
    return {"confident": res.confident, "primary_fid": (res.ranked_ids[0] if res.confident and res.ranked_ids else None),
            "accepted_fid": res.file_id, "top": top, "true_ev": true_ev,
            "verify_ms": round(res.timing_ms.get("verify_ms", 0.0), 2), "n_hyp": res.timing_ms.get("n_hypotheses")}


def run_configs(args):
    from sonic.retrieval import pipeline as P
    from sonic.fingerprint.binary_embed import JOINT_VARISPEED_HYPOTHESES
    import sonic_db_manager as M

    t0 = time.time()
    db = M.SONICDatabaseManager()
    while db.domain_index is None:
        time.sleep(0.5)
    idx = db.domain_index
    print(f"[attach {time.time() - t0:.0f}s, {idx.landmark_index.n_files} files]", flush=True)
    echoes = Path(args.echoes_root)
    tmp = Path(tempfile.mkdtemp(prefix="m1_calib_"))
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        conds = [c for c in args.conditions.split(",") if c] or list(B.CONDITIONS)
        pos = build_positives(db.manifest, echoes, tmp, args.per_domain, args.seed, conds, args.varispeed)
        (tmp / "neg").mkdir()
        negs = build_calib_negatives(db.manifest, echoes, tmp / "neg", args.negatives)
        print(f"[positives {len(pos)}, negatives {len(negs)}]", flush=True)
        base_params = idx.params
        base_win = idx.query_max_duration_sec
        base_ef = (idx.subfp_index.ef_search, idx.subfp_fine_index.ef_search)
        for name in args.configs.split(","):
            over, extra = CONFIGS[name]
            over = dict(over)
            if "joint_hyps" in over and over["joint_hyps"] is None:
                over["joint_hyps"] = JOINT_VARISPEED_HYPOTHESES
            if not extra.get("locked"):
                over.update(OPEN)
            idx.params = dataclasses.replace(base_params, **over)
            idx.query_max_duration_sec = extra.get("window_sec", base_win)
            ef = extra.get("ef_search")
            if ef:
                idx.subfp_index.set_ef_search(ef)
                idx.subfp_fine_index.set_ef_search(ef)
            rows, nrows = [], []
            tc = time.time()
            for q in pos:
                t1 = time.perf_counter()
                res = P.query(str(q["path"]), idx)
                rows.append({"condition": q["condition"], "domain": q["domain"], "true_fid": q["true_fid"],
                             "wall_ms": round((time.perf_counter() - t1) * 1000, 2), **compact(res, q["true_fid"])})
            for n in negs:
                t1 = time.perf_counter()
                res = P.query(str(n["path"]), idx)
                nrows.append({"source": n["source"], "file": Path(n["path"]).name, "src_fid": n.get("src_fid"),
                              "wall_ms": round((time.perf_counter() - t1) * 1000, 2), **compact(res)})
            out = EXP_DIR / f"{name}.json"
            json.dump({"config": name, "overrides": {k: v for k, v in over.items()}, "extra": extra,
                       "seed": args.seed, "rows": rows, "neg_rows": nrows}, open(out, "w"), indent=0, default=str)
            print(f"[{name}] {len(rows)}+{len(nrows)} queries in {time.time() - tc:.0f}s -> {out}", flush=True)
            if extra.get("locked"):
                fp = sum(r["accepted_fid"] is not None for r in nrows)
                acc = np.mean([r["accepted_fid"] == r["true_fid"] for r in rows if r["condition"] != "duplicate"])
                print(f"[{name}] locked thresholds: dev edited acc {100 * acc:.1f}%  calibration FPs {fp}/{len(nrows)}")
            idx.params = base_params
            idx.query_max_duration_sec = base_win
            idx.subfp_index.set_ef_search(base_ef[0])
            idx.subfp_fine_index.set_ef_search(base_ef[1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- sweep
SHORT = (4, 0.15)  # pipeline short_alignment_windows / short_alignment_fine_ber (E7); (0, 1.0) = off
LONG_CONTRAST = 0.04  # pipeline min_contrast_long (E8): contrast floor when n >= contrast_below_windows; None = off


def _ok(c, P):
    """Mirrors pipeline._passes (BER thresholds, 1-window tier, contrast, time-reversal check)."""
    ac, af, rv, ct, nct, s1 = P
    n = c["n_aligned"]
    if rv is not None and c.get("reverse_score") is not None and c["reverse_score"] - c["score"] < rv:
        return False
    if n >= 2:
        if c["coarse_ber"] > ac or c["fine_ber"] > af:
            return False
        floor = ct if n < nct else LONG_CONTRAST
        if floor is not None and (c.get("contrast") is None or c["contrast"] < floor):
            return False
        if n < SHORT[0] and c["fine_ber"] > SHORT[1]:
            return False
        return True
    if n == 1 and s1 is not None:
        return c["coarse_ber"] <= s1 and c["fine_ber"] <= 2 * s1
    return False


def decide(row, P):
    """Simulates _verify_escalation's decision on dumped evidence (primary fast path kept)."""
    if row["confident"]:
        return row["accepted_fid"]
    if not row["top"]:
        return None
    return row["top"][0]["fid"] if _ok(row["top"][0], P) else None  # lowest combined BER, as in the pipeline


# (accept_coarse_ber, accept_fine_ber, reverse_margin, min_contrast, contrast_below_windows, single_window_ber)
GRID = dict(coarse=(0.20, 0.22, 0.24, 0.25, 0.26, 0.28), fine=(0.21, 0.23, 0.24, 0.25, 0.26, 0.27, 0.29),
            reverse=(None, 0.0, 0.02), contrast=(None, 0.04, 0.06, 0.08), contrast_below=(6, 8, 10, 99),
            single=(None, 0.02, 0.04, 0.05, 0.06))
NAMES = ("coarse<=", "fine<=", "reverse>=", "contrast>=", "contrast if n<", "1-window<=")


def sweep(path, exclude_own=False, locked=None):
    d = json.load(open(path))
    rows, nrows = d["rows"], d["neg_rows"]

    def is_fp(r, P):
        f = decide(r, P)
        if f is None:
            return False
        return not (exclude_own and r.get("src_fid") is not None and f == r["src_fid"])

    def acc(P):
        corr = [decide(r, P) == r["true_fid"] for r in rows if r["condition"] != "duplicate"]
        return float(np.mean(corr)) if corr else 0.0

    print(f"# sweep {Path(path).name}: {len(rows)} positives, {len(nrows)} negatives, exclude_own={exclude_own}")
    if locked is None:
        res = []
        for P in itertools.product(*GRID.values()):
            if P[3] is None and P[4] != 99:
                continue
            fp = sum(is_fp(r, P) for r in nrows)
            if fp <= 1:
                res.append((fp, acc(P), P))
        zero = sorted([r for r in res if r[0] == 0], key=lambda r: -r[1])
        print("| " + " | ".join(NAMES) + " | dev edited acc | calib FP |")
        print("|" + "---:|" * (len(NAMES) + 2))
        for fp, a, P in zero[:10]:
            print("| " + " | ".join(str(x) for x in P) + f" | {100 * a:.1f}% | {fp} |")
        for fp, a, P in sorted([r for r in res if r[0] == 1], key=lambda r: -r[1])[:3]:
            print(f"  (1 FP) {P} acc {100 * a:.1f}%")
        if not zero:
            print("no zero-FP setting")
            return
        best = zero[0][2]
        cs, fs = GRID["coarse"], GRID["fine"]
        stricter = lambda v, g: g[max(0, g.index(v) - 1)]
        locked = (stricter(best[0], cs), stricter(best[1], fs)) + best[2:5] + (
            None if best[5] is None else stricter(best[5], GRID["single"]),)
        print(f"\nbest 0-FP point {best}; locked (one grid step stricter on every BER threshold): {locked}")
    P = tuple(locked)
    per = defaultdict(list)
    for r in rows:
        per[r["condition"]].append(decide(r, P) == r["true_fid"])
    print("\n| condition | n | dev acc |\n|---|---:|---:|")
    for c, v in per.items():
        print(f"| {c} | {len(v)} | {100 * np.mean(v):.1f}% |")
    nfp = defaultdict(int)
    for r in nrows:
        f = decide(r, P)
        if f is not None:
            nfp[r["source"] + ("(own source)" if r.get("src_fid") == f else "")] += 1
    w = [r["wall_ms"] for r in rows + nrows]
    print(f"\nlocked {P}: dev edited acc {100 * acc(P):.1f}% | calibration accepts {sum(nfp.values())}/{len(nrows)} "
          f"{dict(nfp)} | wrong matches {sum(1 for r in rows if decide(r, P) not in (None, r['true_fid']))} | "
          f"latency (thresholds open) p50 {np.percentile(w, 50):.0f} ms, p95 {np.percentile(w, 95):.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="baseline")
    ap.add_argument("--per-domain", type=int, default=12)
    ap.add_argument("--negatives", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--varispeed", action="store_true")
    ap.add_argument("--echoes-root", default=str(B.DOWNLOADS / "Echoes"))
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--locked", default=None, help="evaluate one setting, e.g. '(0.24,0.25,0.0,0.08,10,0.05)'")
    ap.add_argument("--exclude-own", action="store_true",
                    help="do not count a time-reversed/edited negative matched to its own source DB file as FP")
    args = ap.parse_args()
    if args.sweep:
        sweep(args.sweep, args.exclude_own, eval(args.locked) if args.locked else None)
    else:
        run_configs(args)


if __name__ == "__main__":
    main()
