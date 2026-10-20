"""Module 1 ("Duplicate & Edited Audio Detection") end-to-end robustness benchmark.

Exercises the SAME code path the web app uses for Module 1: every query is a
real audio file on disk passed to ``app.run_duplicate_detection([path], "auto")``
(what POST /api/check with module=duplicate_detection calls), or - with
``--via-http`` - an actual multipart POST to /api/check through Flask's test
client. Nothing in the retrieval / decision logic is re-implemented here, so
the numbers reflect what a user of the web UI sees.

Reference DB: the app's real store (prism_v2/sonic_db_store, opened read-only
in practice: queries never write to it). Test tracks are DB entries whose
original source audio exists locally (ESC-50 / GTZAN / LJSpeech / LibriSpeech
dev-clean under ~/Downloads/Echoes, plus any user-ingested track whose stored
path exists, e.g. the uploaded "Chaleya Jawan" song). Tracks whose filename is
not unique in the manifest are skipped so "correct" is unambiguous.

Modifications (both ends of every required range):
  pitch shift +-1/+-2/+-4 st (librosa phase-vocoder pitch_shift, tempo kept)
  speed +-5/+-10/+-20 %      (librosa time_stretch, pitch kept; rate = 1 +- p)
  MP3 64 / 128 kbps          (real .mp3 file, ffmpeg libmp3lame)
  white noise SNR 10 / 20 dB
  gain +-3 / +-6 dB          (float WAV, no clipping - research convention)
  high-pass 150 / 200 Hz     (4th-order Butterworth, zero-phase)
  trim 3 / 10 s from start, trim 3 / 10 s from end
  time shift 5 / 20 s: cut (query starts N s into the track) and pad (N s of
                            leading silence prepended)
  + the unmodified file ("duplicate") and non-DB negative queries.
Trim / cut conditions need the track to keep >= 2 s; ineligible tracks are
skipped for that condition and the per-condition n is reported.

Negatives (must NOT be reported as a match): LibriSpeech test-clean
utterances (speech; none are in the DB), 30 s segments of local music/speech
mp3s that are not in the DB, and time-reversed ESC-50 clips (synthetic hard
negatives for environmental audio, reported separately).

Edited queries are written as float WAV at 22050 Hz mono (except MP3) under a
temporary directory that is removed at the end. Uploaded file names never equal
the DB filename (the app has a same-filename shortcut).

Usage (from prism_v2/):
  .venv/bin/python SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick
  .venv/bin/python SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py            # full
  ... --out-json X.json --out-md X.md --per-domain 34 --negatives-per-source 40
  ... --conditions pitch_+2st,speed_+5pct    # subset while iterating
  ... --via-http                             # go through POST /api/check
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SONIC_ROOT = Path(__file__).resolve().parent.parent
PRISM_ROOT = SONIC_ROOT.parent
for p in (str(PRISM_ROOT), str(SONIC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

SR = 22050
MIN_REMAINING_SEC = 2.0
DOWNLOADS = Path.home() / "Downloads"

# name -> (kind, param)
CONDITIONS = {
    "duplicate": ("duplicate", None),
    **{f"pitch_{s:+d}st": ("pitch", s) for s in (-4, -2, -1, 1, 2, 4)},
    **{f"speed_{p:+d}pct": ("speed", p) for p in (-20, -10, -5, 5, 10, 20)},
    "mp3_64kbps": ("mp3", 64), "mp3_128kbps": ("mp3", 128),
    "noise_snr10db": ("noise", 10), "noise_snr20db": ("noise", 20),
    **{f"gain_{g:+d}db": ("gain", g) for g in (-6, -3, 3, 6)},
    "highpass_150hz": ("highpass", 150), "highpass_200hz": ("highpass", 200),
    "trim_start_3s": ("trim_start", 3), "trim_start_10s": ("trim_start", 10),
    "trim_end_3s": ("trim_end", 3), "trim_end_10s": ("trim_end", 10),
    "timeshift_cut_5s": ("shift_cut", 5), "timeshift_cut_20s": ("shift_cut", 20),
    "timeshift_pad_5s": ("shift_pad", 5), "timeshift_pad_20s": ("shift_pad", 20),
}
# Optional (--with-varispeed): resample-style speed change, pitch moves with tempo.
# Reported under overall["extra"], NOT pooled into edited_accuracy, so the
# pooled number stays comparable with the BEFORE baseline.
EXTRA_CONDITIONS = {f"varispeed_{p:+d}pct": ("varispeed", p) for p in (-20, -10, -5, 5, 10, 20)}
GROUP_OF = {"duplicate": "Duplicate", "varispeed": "Varispeed (extra)", "pitch": "Pitch shift", "speed": "Speed change", "mp3": "Compression",
            "noise": "Noise", "gain": "Gain", "highpass": "High-pass", "trim_start": "Trim",
            "trim_end": "Trim", "shift_cut": "Time shift", "shift_pad": "Time shift"}


ALL_CONDITIONS = {**CONDITIONS, **EXTRA_CONDITIONS}


# ----------------------------------------------------------------------------- audio edits
def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_edit(kind, param, y, src_path: Path, out_dir: Path, stem: str, rng: np.random.Generator):
    """Returns a path to the edited query file, or None if the track is ineligible."""
    import librosa
    from scipy.signal import butter, sosfiltfilt

    dur = y.size / SR
    out = out_dir / f"q_{stem}.wav"
    if kind == "duplicate":
        dst = out_dir / f"q_{stem}{src_path.suffix}"
        shutil.copyfile(src_path, dst)
        return dst
    if kind == "pitch":
        z = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=float(param))
    elif kind == "speed":
        z = librosa.effects.time_stretch(y=y, rate=1.0 + param / 100.0)
    elif kind == "varispeed":
        z = librosa.resample(y, orig_sr=SR, target_sr=int(round(SR / (1.0 + param / 100.0))))
    elif kind == "mp3":
        wav = out_dir / f"tmp_{stem}.wav"
        mp3 = out_dir / f"q_{stem}.mp3"
        sf.write(str(wav), y, SR, subtype="PCM_16")
        subprocess.run([_ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                        "-b:a", f"{param}k", str(mp3)], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        wav.unlink(missing_ok=True)
        return mp3
    elif kind == "noise":
        noise = rng.standard_normal(y.size).astype(np.float32)
        ps = float(np.mean(y ** 2)) + 1e-12
        pn = float(np.mean(noise ** 2)) + 1e-12
        z = y + noise * np.sqrt(ps / (10 ** (param / 10)) / pn)
    elif kind == "gain":
        z = y * (10 ** (param / 20))
    elif kind == "highpass":
        if y.size < 64:
            return None
        sos = butter(4, param / (SR / 2), btype="highpass", output="sos")
        z = sosfiltfilt(sos, y)
    elif kind in ("trim_start", "shift_cut"):
        if dur - param < MIN_REMAINING_SEC:
            return None
        z = y[int(param * SR):]
    elif kind == "trim_end":
        if dur - param < MIN_REMAINING_SEC:
            return None
        z = y[: y.size - int(param * SR)]
    elif kind == "shift_pad":
        z = np.concatenate([np.zeros(int(param * SR), dtype=np.float32), y])
    else:
        raise ValueError(kind)
    z = np.asarray(z, dtype=np.float32)
    if not np.all(np.isfinite(z)) or z.size < int(0.5 * SR):
        return None
    sf.write(str(out), z, SR, subtype="FLOAT")
    return out


# ----------------------------------------------------------------------------- data sourcing
def domain_of(path: str) -> str | None:
    if "ESC-50" in path:
        return "environment"
    if "genres" in path:
        return "music"
    if "LJSpeech" in path or "dev-clean" in path:
        return "speech"
    return None


def resolve_local(path: str, echoes_root: Path) -> Path | None:
    if "Echoes\\" in path:
        p = echoes_root / path.split("Echoes\\", 1)[1].replace("\\", "/")
        return p if p.exists() else None
    p = Path(path)
    return p if p.exists() else None


def select_tracks(manifest, echoes_root: Path, per_domain: int, seed: int):
    names = Counter(e["filename"] for e in manifest)
    pools = defaultdict(list)
    user_tracks = []
    for e in manifest:
        if names[e["filename"]] > 1:
            continue
        lp = resolve_local(e["path"], echoes_root)
        if lp is None:
            continue
        d = domain_of(e["path"])
        if d is None:
            user_tracks.append({"domain": "user_upload", "entry": e, "local": lp})
        else:
            pools[d].append({"domain": d, "entry": e, "local": lp})
    tracks = list(user_tracks)
    for d in ("environment", "music", "speech"):
        pool = sorted(pools[d], key=lambda t: t["entry"]["file_id"])
        random.Random(f"{seed}|{d}").shuffle(pool)
        tracks.extend(pool[:per_domain])
    return tracks, {d: len(v) for d, v in pools.items()} | {"user_upload": len(user_tracks)}


def build_negatives(manifest, echoes_root: Path, n_per_source: int, seed: int, out_dir: Path):
    import librosa
    names = {e["filename"] for e in manifest}
    rng = random.Random(f"{seed}|neg")
    negs = []
    # 1. LibriSpeech test-clean (speech, disjoint speakers from dev-clean)
    tc = sorted(p for p in (echoes_root / "_negatives").rglob("*.flac") if p.name not in names)
    rng.shuffle(tc)
    for p in tc[:n_per_source]:
        dst = out_dir / f"neg_{p.name}"
        shutil.copyfile(p, dst)
        negs.append({"source": "librispeech_test_clean", "path": dst})
    # 2. 30 s segments of local mp3s that are not in the DB (music + one long talk)
    seg_paths = []
    for p in sorted(DOWNLOADS.glob("*.mp3")):
        if p.name in names or "Chaleya" in p.name:  # Chaleya = DB track (+ the user's edits of it)
            continue
        try:
            total = librosa.get_duration(path=str(p))
        except Exception:
            continue
        for k, off in enumerate(np.arange(5.0, max(total - 30.0, 5.0), 30.0)):
            seg_paths.append((p, float(off), k))
    rng.shuffle(seg_paths)
    music_segs = [s for s in seg_paths if "talk" not in s[0].name.lower()][:n_per_source]
    talk_segs = [s for s in seg_paths if "talk" in s[0].name.lower()][: max(1, n_per_source // 2)]
    for p, off, k in music_segs + talk_segs:
        y, _ = librosa.load(str(p), sr=SR, mono=True, offset=off, duration=30.0)
        if y.size < SR:
            continue
        dst = out_dir / f"neg_seg{k}_{p.stem[:40]}.wav"
        sf.write(str(dst), y, SR, subtype="FLOAT")
        src = "local_speech_mp3_segment" if "talk" in p.name.lower() else "local_music_mp3_segment"
        negs.append({"source": src, "path": dst})
    # 3. time-reversed ESC-50 clips (environmental hard negatives)
    esc = sorted((e for e in manifest if "ESC-50" in e["path"]), key=lambda e: e["file_id"])
    rng.shuffle(esc)
    for e in esc[:n_per_source]:
        lp = resolve_local(e["path"], echoes_root)
        if lp is None:
            continue
        y, _ = librosa.load(str(lp), sr=SR, mono=True)
        dst = out_dir / f"neg_rev_{Path(e['filename']).stem}.wav"
        sf.write(str(dst), y[::-1].copy(), SR, subtype="FLOAT")
        negs.append({"source": "esc50_time_reversed", "path": dst})
    return negs


# ----------------------------------------------------------------------------- query via app
class AppQuery:
    def __init__(self, via_http: bool):
        import app as prism_app
        self.app = prism_app
        self.via_http = via_http
        self.client = prism_app.app.test_client() if via_http else None

    def wait_ready(self, timeout=3600):
        t0 = time.time()
        while not self.app.sonic_db.get_status()["attached"]:
            if time.time() - t0 > timeout:
                raise SystemExit("reference DB did not attach")
            time.sleep(0.5)
        return time.time() - t0

    def __call__(self, path: Path) -> tuple[dict, float]:
        t0 = time.perf_counter()
        if self.via_http:
            data = {"module": "duplicate_detection", "domain": "auto",
                    "audio_files": (io.BytesIO(path.read_bytes()), path.name)}
            body = self.client.post("/api/check", data=data, content_type="multipart/form-data").get_json()
            if body.get("job_id"):
                while True:
                    j = self.client.get(f"/api/check/job/{body['job_id']}").get_json()
                    if j["status"] in ("completed", "error", "cancelled"):
                        body = j.get("result") or {}
                        break
                    time.sleep(0.2)
        else:
            body = self.app.run_duplicate_detection([path], "auto")
        wall = (time.perf_counter() - t0) * 1000
        res = (body.get("results") or [{}])[0]
        return res, wall


# ----------------------------------------------------------------------------- stats / report
def lat_stats(v):
    if not v:
        return {"mean_ms": None, "p95_ms": None}
    a = np.asarray(v, dtype=float)
    return {"mean_ms": round(float(a.mean()), 1), "p95_ms": round(float(np.percentile(a, 95)), 1)}


def summarize(rows, neg_rows):
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    conds = {}
    for c in list(CONDITIONS) + list(EXTRA_CONDITIONS):
        rs = by_cond.get(c, [])
        if not rs:
            continue
        per_dom = {}
        for d in sorted({r["domain"] for r in rs}):
            sub = [r for r in rs if r["domain"] == d]
            per_dom[d] = {"n": len(sub), "correct": sum(r["correct"] for r in sub),
                          "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4)}
        conds[c] = {
            "group": GROUP_OF[ALL_CONDITIONS[c][0]], "n": len(rs), "correct": sum(r["correct"] for r in rs),
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
            "wrong_match": sum(1 for r in rs if r["matched"] and not r["correct"]),
            "match_types": dict(Counter(str(r["match_type"]) for r in rs if r["correct"])),
            "per_domain": per_dom, **lat_stats([r["wall_ms"] for r in rs]),
        }
    edited = [r for r in rows if r["condition"] != "duplicate" and r["condition"] not in EXTRA_CONDITIONS]
    extra = [r for r in rows if r["condition"] in EXTRA_CONDITIONS]
    neg_src = defaultdict(list)
    for r in neg_rows:
        neg_src[r["source"]].append(r)
    negs = {
        "n": len(neg_rows), "false_positives": sum(r["matched"] for r in neg_rows),
        "fpr": round(sum(r["matched"] for r in neg_rows) / len(neg_rows), 4) if neg_rows else None,
        "per_source": {s: {"n": len(v), "false_positives": sum(r["matched"] for r in v),
                           "fpr": round(sum(r["matched"] for r in v) / len(v), 4)} for s, v in neg_src.items()},
        **lat_stats([r["wall_ms"] for r in neg_rows]),
    }
    overall = {
        "edited_n": len(edited), "edited_correct": sum(r["correct"] for r in edited),
        "edited_accuracy": round(sum(r["correct"] for r in edited) / len(edited), 4) if edited else None,
        **lat_stats([r["wall_ms"] for r in rows + neg_rows]),
    }
    if extra:
        overall["extra"] = {"conditions": sorted({r["condition"] for r in extra}), "n": len(extra),
                            "correct": sum(r["correct"] for r in extra),
                            "accuracy": round(sum(r["correct"] for r in extra) / len(extra), 4)}
    return conds, negs, overall


def to_markdown(meta, conds, negs, overall):
    doms = ["environment", "music", "speech", "user_upload"]
    L = [f"# Module 1 robustness benchmark ({meta['mode']})", "",
         f"- Run: {meta['started']} | git: {meta.get('git_rev','?')} | path: {meta['query_path']}",
         f"- DB: {meta['db_count']} entries | tracks: {meta['n_tracks']} ({meta['tracks_per_domain']})",
         f"- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).",
         "",
         "| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, s in conds.items():
        cells = []
        for d in doms:
            pd = s["per_domain"].get(d)
            cells.append(f"{pd['correct']}/{pd['n']}" if pd else "-")
        L.append(f"| {c} | {s['group']} | {s['n']} | {100*s['accuracy']:.1f}% | " + " | ".join(cells)
                 + f" | {s['wrong_match']} | {s['mean_ms']} | {s['p95_ms']} |")
    if overall.get("extra"):
        x = overall["extra"]
        L += ["", f"Extra (not pooled): {', '.join(x['conditions'])}: {x['correct']}/{x['n']} = {100*x['accuracy']:.1f}%"]
    L += ["", f"Overall edited accuracy (all modified conditions pooled): "
              f"{overall['edited_correct']}/{overall['edited_n']} = {100*(overall['edited_accuracy'] or 0):.1f}%  "
              f"| latency mean {overall['mean_ms']} ms, p95 {overall['p95_ms']} ms", "",
          "| Negatives (not in DB) | n | false positives | FPR |", "|---|---:|---:|---:|"]
    for s, v in negs["per_source"].items():
        L.append(f"| {s} | {v['n']} | {v['false_positives']} | {100*v['fpr']:.1f}% |")
    if negs["n"]:
        L.append(f"| **all** | {negs['n']} | {negs['false_positives']} | {100*negs['fpr']:.1f}% |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sample (8 tracks/domain, 8 negatives/source)")
    ap.add_argument("--per-domain", type=int, default=None, help="tracks per domain (full default 34)")
    ap.add_argument("--negatives-per-source", type=int, default=None, help="full default 40")
    ap.add_argument("--conditions", default="", help="comma-separated subset of condition names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--echoes-root", default=str(DOWNLOADS / "Echoes"))
    ap.add_argument("--via-http", action="store_true", help="POST /api/check via Flask test client")
    ap.add_argument("--with-varispeed", action="store_true",
                    help="add varispeed_+-5/10/20pct (resample speed change); reported separately, not pooled")
    ap.add_argument("--local-source-paths", action="store_true",
                    help="IN-MEMORY ONLY: point legacy manifest paths at the local Echoes copies, so the "
                         "app's 'source file exists' decision branch is exercised (manifest on disk untouched)")
    ap.add_argument("--out-json", default=str(SONIC_ROOT / "results" / "module1_robustness" / "benchmark.json"))
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    per_domain = args.per_domain or (8 if args.quick else 34)
    n_neg = args.negatives_per_source or (8 if args.quick else 40)
    cond_names = [c for c in args.conditions.split(",") if c] or list(CONDITIONS)
    if args.with_varispeed:
        cond_names += [c for c in EXTRA_CONDITIONS if c not in cond_names]
    unknown = set(cond_names) - set(ALL_CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    echoes_root = Path(args.echoes_root)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    import librosa

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    q = AppQuery(args.via_http)
    ready_s = q.wait_ready()
    db = q.app.sonic_db
    print(f"[DB attached in {ready_s:.0f}s: {len(db.manifest)} entries]", flush=True)

    if args.local_source_paths:
        n_remap = 0
        for e in db.manifest:  # in-memory only; nothing here calls _save_manifest()
            lp = resolve_local(e["path"], echoes_root)
            if lp is not None and str(lp) != e["path"]:
                e["path"] = str(lp)
                n_remap += 1
        print(f"[in-memory path remap: {n_remap} entries]")

    tracks, pool_sizes = select_tracks(db.manifest, echoes_root, per_domain, args.seed)
    print(f"[tracks: {len(tracks)}; eligible pool sizes {pool_sizes}]", flush=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="module1_bench_"))
    rows, neg_rows = [], []
    try:
        np_rng = np.random.default_rng(args.seed)
        t_start = time.time()
        for ti, t in enumerate(tracks):
            e = t["entry"]
            y, _ = librosa.load(str(t["local"]), sr=SR, mono=True)
            y = y.astype(np.float32)
            tdir = tmp_root / f"t{ti}"
            tdir.mkdir()
            for c in cond_names:
                kind, param = ALL_CONDITIONS[c]
                try:
                    qp = make_edit(kind, param, y, t["local"], tdir, f"{c}_{ti}", np_rng)
                except Exception as ex:  # generation failure is reported, not hidden
                    print(f"  ! generation failed {c} {e['filename']}: {ex}")
                    qp = None
                if qp is None:
                    continue
                res, wall = q(qp)
                matched = bool(res.get("matched"))
                rows.append({
                    "condition": c, "domain": t["domain"], "true_file_id": e["file_id"],
                    "true_filename": e["filename"], "duration_sec": round(y.size / SR, 2),
                    "matched": matched, "matched_filename": res.get("matched_filename"),
                    "correct": bool(matched and res.get("matched_filename") == e["filename"]),
                    "match_type": res.get("match_type"), "score": res.get("score"),
                    "stage_used": res.get("stage_used"), "app_query_ms": res.get("query_time_ms"),
                    **{k: res[k] for k in ("edit_hypothesis", "similarity", "verification") if res.get(k) is not None},
                    "wall_ms": round(wall, 2),
                })
                qp.unlink(missing_ok=True)
            shutil.rmtree(tdir, ignore_errors=True)
            done = sum(r["correct"] for r in rows)
            print(f"[{ti+1}/{len(tracks)}] {t['domain']:11s} {e['filename'][:40]:40s} "
                  f"running acc {done}/{len(rows)}  ({time.time()-t_start:.0f}s)", flush=True)

        ndir = tmp_root / "neg"
        ndir.mkdir()
        for n in build_negatives(db.manifest, echoes_root, n_neg, args.seed, ndir):
            res, wall = q(n["path"])
            neg_rows.append({"source": n["source"], "file": n["path"].name, "matched": bool(res.get("matched")),
                             "matched_filename": res.get("matched_filename"), "match_type": res.get("match_type"),
                             "score": res.get("score"), "wall_ms": round(wall, 2)})
        print(f"[negatives: {len(neg_rows)}, false positives {sum(r['matched'] for r in neg_rows)}]", flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    conds, negs, overall = summarize(rows, neg_rows)
    try:
        rev = subprocess.run(["git", "-C", str(PRISM_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PRISM_ROOT), "status", "--porcelain", "--", "app.py",
                                "sonic_db_manager.py", "SONIC-fixed_1/sonic"], capture_output=True, text=True).stdout.strip()
        rev += "+uncommitted-changes" if dirty else ""
    except Exception:
        rev = "?"
    meta = {
        "mode": ("quick" if args.quick else "full") if not (args.per_domain or args.negatives_per_source or args.conditions) else "custom",
        "started": started, "git_rev": rev,
        "query_path": "POST /api/check (Flask test client)" if args.via_http else "app.run_duplicate_detection",
        "local_source_paths": args.local_source_paths, "seed": args.seed,
        "db_count": len(db.manifest), "n_tracks": len(tracks),
        "tracks_per_domain": dict(Counter(t["domain"] for t in tracks)), "eligible_pool_sizes": pool_sizes,
        "conditions": cond_names, "negatives_per_source": n_neg, "runtime_sec": round(time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S")), 1),
    }
    json.dump({"meta": meta, "conditions": conds, "negatives": negs, "overall": overall,
               "rows": rows, "negative_rows": neg_rows}, open(out_json, "w"), indent=1)
    md = to_markdown(meta, conds, negs, overall)
    out_md.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
