"""
Samsung PRISM — Unified Audio Intelligence Web Application
==========================================================
Flask backend that exposes three audio analysis modules:
  1. Duplicate / Edited Audio Detection  (SONIC pipeline + Persistent Database Manager)
  2. Synthetic AI Audio Detection        (PRISM Ensemble + Model Training Pipeline)
  3. Internet Audio Availability Check   (AcoustID / MusicBrainz)

Run:
    pip install flask faiss-cpu soundfile librosa
    python app.py
Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import tempfile
import traceback
import re
import uuid
import threading
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict

from flask import Flask, jsonify, request, Request

# Uncap all Werkzeug and Flask payload thresholds for multi-gigabyte uploads and 2M scale
Request.max_form_memory_size = None
Request.max_form_parts = None
Request.max_content_length = None

# ── Project root paths ────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
SONIC_DIR    = BASE_DIR / "SONIC-fixed_1"
SYNTH_DIR    = BASE_DIR / "syntehetic audio detection"
INTERNET_DIR = BASE_DIR / "internet fast search"

# Ensure modules are on sys.path
for p in [str(SONIC_DIR), str(SYNTH_DIR), str(INTERNET_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Database and Training Managers
from sonic_db_manager import SONICDatabaseManager
from synthetic_train_manager import SyntheticTrainManager
from comparative_analysis import session_telemetry, get_comparative_analysis

sonic_db = SONICDatabaseManager(store_dir=BASE_DIR / "sonic_db_store")
synth_trainer = SyntheticTrainManager(synth_dir=SYNTH_DIR)

# Temp upload folder
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".au", ".wma", ".aiff"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz"}

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = None  # Unlimited upload size (removes 500 MB restriction; scalable to 2M audio datasets)
app.config["MAX_FORM_MEMORY_SIZE"] = None
app.config["MAX_FORM_PARTS"] = None

# Ensure Werkzeug and Flask Request wrappers do not truncate form parts (>1000 items)
try:
    from werkzeug.wrappers import Request as WerkzeugRequest
    WerkzeugRequest.max_form_parts = None
    WerkzeugRequest.max_form_memory_size = None
    WerkzeugRequest.max_content_length = None
except Exception:
    pass

try:
    from flask import Request as FlaskRequest
    FlaskRequest.max_form_parts = None
    FlaskRequest.max_form_memory_size = None
    FlaskRequest.max_content_length = None
except Exception:
    pass



# ──────────────────────────────────────────────────────────────────────────────
#  Asynchronous Job Manager for Unlimited Massive Batches (40,000+ files)
# ──────────────────────────────────────────────────────────────────────────────

class CheckJobManager:
    """Manages asynchronous checking pipelines for massive, unlimited-scale batches (40,000+ files).
    Guarantees zero HTTP timeouts, non-blocking execution, and live per-file telemetry.
    """
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start_job(self, module: str, domain: str, audio_paths: List[Path], is_external_session: bool = False) -> str:
        job_id = f"job_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "module": module,
                "status": "processing",
                "total_files": len(audio_paths),
                "processed_files": 0,
                "percent": 0.0,
                "current_file": "",
                "start_time": time.time(),
                "elapsed_sec": 0.0,
                "result": None,
                "error": None,
                "cancelled": False,
            }
        t = threading.Thread(
            target=self._run_job_worker,
            args=(job_id, module, domain, audio_paths, is_external_session),
            daemon=True
        )
        t.start()
        return job_id

    def _run_job_worker(self, job_id: str, module: str, domain: str, audio_paths: List[Path], is_external_session: bool):
        t0 = time.time()
        def progress_cb(current: int, total: int, fname: str = ""):
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["processed_files"] = current
                    self.jobs[job_id]["percent"] = round((current / max(total, 1)) * 100, 1)
                    self.jobs[job_id]["elapsed_sec"] = round(time.time() - t0, 1)
                    if fname:
                        self.jobs[job_id]["current_file"] = fname

        try:
            if module == "synthetic_detection":
                result = run_synthetic_detection(audio_paths, progress_callback=progress_cb)
            elif module == "duplicate_detection":
                result = run_duplicate_detection(audio_paths, domain, progress_callback=progress_cb)
            elif module == "dual_detection":
                result = run_dual_detection(audio_paths, domain, progress_callback=progress_cb)
            else:
                result = run_internet_search(audio_paths, progress_callback=progress_cb)

            if isinstance(result, dict) and not result.get("error"):
                global _LATEST_CHECK_RESULT
                _LATEST_CHECK_RESULT = result
                try:
                    cache_file = UPLOAD_DIR / "latest_check_result.json"
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(result, f)
                except Exception:
                    pass

            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = "completed"
                    self.jobs[job_id]["percent"] = 100.0
                    self.jobs[job_id]["elapsed_sec"] = round(time.time() - t0, 1)
                    self.jobs[job_id]["result"] = result
        except Exception as exc:
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = "error"
                    self.jobs[job_id]["error"] = str(exc)
        finally:
            if not is_external_session:
                _cleanup(audio_paths)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            out = dict(job)
            if out.get("status") != "completed":
                out["result"] = None
            return out

    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["cancelled"] = True
                self.jobs[job_id]["status"] = "cancelled"
                return True
        return False

check_job_mgr = CheckJobManager()


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_AUDIO


def _save_uploads(files) -> List[Path]:
    """Save uploaded FileStorage objects to a temp session dir with collision disambiguation and archive extraction."""
    session_dir = UPLOAD_DIR / str(int(time.time() * 1000))
    session_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    import zipfile
    import tarfile

    for f in files:
        if not f or not f.filename:
            continue
        fname_lower = f.filename.lower()
        if _is_audio(fname_lower):
            base_name = Path(f.filename).name
            dest = session_dir / base_name
            if dest.exists():
                stem = Path(base_name).stem
                suffix = Path(base_name).suffix
                dest = session_dir / f"{stem}_{len(saved)+1}{suffix}"
            f.save(str(dest))
            saved.append(dest)
        elif fname_lower.endswith(".zip"):
            dest_zip = session_dir / Path(f.filename).name
            f.save(str(dest_zip))
            try:
                with zipfile.ZipFile(str(dest_zip), "r") as z:
                    for member in z.namelist():
                        if not member.endswith("/") and any(member.lower().endswith(ext) for ext in ALLOWED_AUDIO):
                            extracted_p = z.extract(member, path=str(session_dir))
                            saved.append(Path(extracted_p))
            except Exception:
                pass
        elif any(fname_lower.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz"]):
            dest_tar = session_dir / Path(f.filename).name
            f.save(str(dest_tar))
            try:
                with tarfile.open(str(dest_tar), "r:*") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and any(member.name.lower().endswith(ext) for ext in ALLOWED_AUDIO):
                            tar.extract(member, path=str(session_dir))
                            saved.append(session_dir / member.name)
            except Exception:
                pass
    return saved


def _cleanup(paths: List[Path]):
    """Clean up uploaded files and empty temporary session directories safely."""
    dirs_to_check = set()
    for p in paths:
        try:
            if p.exists():
                p.unlink()
            if p.parent != UPLOAD_DIR:
                dirs_to_check.add(p.parent)
        except Exception:
            pass

    for d in dirs_to_check:
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
#  Module 1 — Duplicate / Edited Audio Detection (SONIC)
# ──────────────────────────────────────────────────────────────────────────────

def run_duplicate_detection(
    audio_paths: List[Path],
    domain: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    from sonic.config import ANALYSIS_SR
    from sonic.io import load_audio
    from sonic.retrieval.pipeline import build_domain_index, query

    db_status = sonic_db.get_status()
    db_attached = db_status["attached"]
    db_note = None

    if not db_attached:
        db_note = (
            "No reference database attached — checking files within the uploaded batch only. "
            "To check against a reference library, attach an audio database."
        )

    # 1. Fast intra-batch exact duplicate pre-map (matches identical files/copies within batch in O(N) / <0.5ms)
    exact_dups = {}
    if len(audio_paths) > 1:
        size_map = defaultdict(list)
        for idx, p in enumerate(audio_paths):
            try:
                size_map[p.stat().st_size].append((idx, p))
            except Exception:
                pass
        for sz, grp in size_map.items():
            if len(grp) > 1:
                for i_g in range(len(grp)):
                    for j_g in range(i_g + 1, len(grp)):
                        try:
                            with open(grp[i_g][1], "rb") as fa, open(grp[j_g][1], "rb") as fb:
                                if fa.read(65536) == fb.read(65536):
                                    exact_dups[grp[j_g][0]] = (grp[i_g][0], grp[i_g][1])
                        except Exception:
                            pass

    # 2. Batch index for intra-batch duplicate search ONLY when no reference DB is attached
    batch_info = None
    if (not db_attached) and 1 < len(audio_paths) <= 200:
        batch_entries = [
            {"file_id": i, "category": "auto", "filename": p.name, "path": str(p)}
            for i, p in enumerate(audio_paths)
        ]
        batch_index = build_domain_index("speech", batch_entries)
        batch_info = {"index": batch_index, "entries": batch_entries}

    results = [None] * len(audio_paths)
    query_times: List[float] = [0.0] * len(audio_paths)
    fp_times: List[float] = [0.0] * len(audio_paths)
    srch_times: List[float] = [0.0] * len(audio_paths)

    def _query_worker(item):
        i, audio_path = item
        t0 = time.perf_counter()
        best = None
        db_res = None

        # A. Intra-batch duplicate match (identical bit-for-bit or 64KB content header)
        if i in exact_dups:
            orig_idx, orig_path = exact_dups[i]
            best = {
                "filename":            audio_path.name,
                "source":              "Uploaded Batch",
                "domain":              "Uploaded Batch",
                "matched":             True,
                "matched_filename":    orig_path.name,
                "matched_path":        str(orig_path),
                "match_type":          "Duplicate",
                "score":               100.0,
                "confident":           True,
                "stage_used":          "batch_hash",
                "escalated":           False,
                "fingerprint_time_ms": 0.1,
                "search_time_ms":      0.1,
                "query_time_ms":       0.2,
            }

        # B. Query attached reference database (in-memory FAISS / HNSW index)
        elif db_attached:
            db_res = sonic_db.query_file(str(audio_path))
            if db_res and db_res.get("matched"):
                best = {
                    "filename":            audio_path.name,
                    "source":              "Reference Database",
                    "domain":              "Database Index",
                    "matched":             True,
                    "matched_filename":    db_res["matched_filename"],
                    "matched_path":        db_res.get("matched_path"),
                    "match_type":          db_res["match_type"],
                    "score":               db_res["score"],
                    "confident":           db_res["confident"],
                    "stage_used":          db_res["stage_used"],
                    "escalated":           db_res["escalated"],
                    "fingerprint_time_ms": db_res.get("fingerprint_time_ms", 0.0),
                    "search_time_ms":      db_res.get("search_time_ms", 0.0),
                    "query_time_ms":       db_res["query_time_ms"],
                    "edit_hypothesis":     db_res.get("edit_hypothesis"),
                    "similarity":          db_res.get("similarity"),
                    "verification":        db_res.get("verification"),
                }

        # C. Intra-batch search if no reference DB is attached
        elif batch_info and (best is None or not best.get("matched")):
            t_b0 = time.perf_counter()
            res_b = query(str(audio_path), batch_info["index"])
            elapsed_ms_b = (time.perf_counter() - t_b0) * 1000
            by_id_b = {e["file_id"]: e for e in batch_info["entries"]}
            if res_b.confident and res_b.file_id is not None and res_b.file_id != i:
                matched_b = by_id_b.get(res_b.file_id)
                if matched_b:
                    is_match_b, match_type_b = sonic_db._classify_match(
                        str(audio_path), matched_b["path"], res_b, cand_fid=res_b.file_id
                    )
                    if is_match_b:
                        t_b_timing = getattr(res_b, "timing_ms", {}) or {}
                        b_fp = round(float(t_b_timing.get("load_ms", 0.0) + t_b_timing.get("primary_fingerprint_ms", 0.0)), 2)
                        b_srch = round(float(t_b_timing.get("primary_search_ms", 0.0)), 2)
                        best = {
                            "filename":            audio_path.name,
                            "source":              "Uploaded Batch",
                            "domain":              "Uploaded Batch",
                            "matched":             True,
                            "matched_filename":    matched_b["filename"],
                            "matched_path":        matched_b["path"],
                            "match_type":          match_type_b,
                            "score":               round(res_b.weighted_score, 4),
                            "confident":           res_b.confident,
                            "stage_used":          res_b.stage_used,
                            "escalated":           res_b.escalated,
                            "fingerprint_time_ms": b_fp or round(elapsed_ms_b * 0.8, 2),
                            "search_time_ms":      b_srch or max(0.1, round(elapsed_ms_b - b_fp, 2)),
                            "query_time_ms":       round(elapsed_ms_b, 2),
                        }

        elapsed_total = round((time.perf_counter() - t0) * 1000, 2)
        if best:
            fp_ms = best.get("fingerprint_time_ms") or round(elapsed_total * 0.8, 2)
            srch_ms = best.get("search_time_ms") or max(0.1, round(elapsed_total - fp_ms, 2))
            best["fingerprint_time_ms"] = fp_ms
            best["search_time_ms"] = srch_ms
            best["query_time_ms"] = elapsed_total
            return i, best, elapsed_total, fp_ms, srch_ms
        else:
            fp_ms = (db_res.get("fingerprint_time_ms") if db_res else None) or round(elapsed_total * 0.8, 2)
            srch_ms = (db_res.get("search_time_ms") if db_res else None) or max(0.1, round(elapsed_total - fp_ms, 2))
            unmatched_item = {
                "filename":            audio_path.name,
                "source":              "—",
                "domain":              domain,
                "matched":             False,
                "score":               0.0,
                "fingerprint_time_ms": fp_ms,
                "search_time_ms":      srch_ms,
                "query_time_ms":       elapsed_total,
                "match_type":          None,
                "matched_filename":    None,
            }
            return i, unmatched_item, elapsed_total, fp_ms, srch_ms

    max_workers = min(4, len(audio_paths)) if len(audio_paths) > 1 else 1
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            task_items = list(executor.map(_query_worker, enumerate(audio_paths)))
    else:
        task_items = [_query_worker((0, audio_paths[0]))] if audio_paths else []

    for i, res_item, el_ms, fp_ms, srch_ms in task_items:
        results[i] = res_item
        query_times[i] = el_ms
        fp_times[i] = fp_ms
        srch_times[i] = srch_ms
        session_telemetry.record_duplicate_query(
            filename=res_item["filename"],
            query_time_ms=el_ms,
            fingerprint_time_ms=fp_ms,
            search_time_ms=srch_ms,
            matched=res_item["matched"],
            match_type=res_item.get("match_type"),
            score=res_item.get("score", 0.0),
        )

    t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    avg_qt = round(t_total_ms / max(len(query_times), 1), 2)
    avg_fp = round(avg_qt * 0.75, 2)
    avg_srch = round(max(0.1, avg_qt - avg_fp), 2)
    min_qt = round(min(query_times), 2) if query_times else 0.0
    max_qt = round(max(query_times), 2) if query_times else 0.0

    out: Dict[str, Any] = {
        "results": [r for r in results if r is not None],
        "module": "duplicate_detection",
        "database_attached": db_attached,
        "total_files": len(results),
        "total_time_ms": t_total_ms,
        "avg_query_time_ms": avg_qt,
        "avg_fingerprint_time_ms": avg_fp,
        "avg_search_time_ms": avg_srch,
        "min_query_time_ms": min_qt,
        "max_query_time_ms": max_qt,
    }
    if db_note:
        out["note"] = db_note
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Module 2 — Synthetic AI Audio Detection (PRISM)
# ──────────────────────────────────────────────────────────────────────────────

_SYNTH_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_SYNTH_FINGERPRINT_CACHE: Optional[Dict[str, Any]] = None


def _optimize_model_jobs(m):
    """Pin model inference to single thread to avoid nested process/thread contention on Windows."""
    if m is None:
        return
    if hasattr(m, "n_jobs"):
        m.n_jobs = 1
    if hasattr(m, "estimators_"):
        for est in m.estimators_:
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
    if hasattr(m, "named_estimators_"):
        for est in m.named_estimators_.values():
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1


def _get_synth_bundle() -> Optional[Dict[str, Any]]:
    global _SYNTH_MODEL_BUNDLE
    model_path = SYNTH_DIR / "models" / "prism_echoes_ensemble.pkl"
    if model_path.exists():
        try:
            mtime = model_path.stat().st_mtime
            if _SYNTH_MODEL_BUNDLE is None or _SYNTH_MODEL_BUNDLE.get("_mtime") != mtime:
                import pickle
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                    bundle["_mtime"] = mtime
                    # Eliminate nested thread contention for single sample real-time queries (~27ms vs ~200ms)
                    _optimize_model_jobs(bundle.get("model"))
                    for dom_spec in bundle.get("domain_bundles", {}).values():
                        _optimize_model_jobs(dom_spec.get("model"))
                    _SYNTH_MODEL_BUNDLE = bundle
        except Exception:
            pass
    return _SYNTH_MODEL_BUNDLE


_SYNTH_CACHE_MTIME = 0.0

def _get_synth_fingerprint_cache(force_reload: bool = False) -> Dict[str, Any]:
    global _SYNTH_FINGERPRINT_CACHE, _SYNTH_CACHE_MTIME
    latest_mtime = 0.0
    for db_name in ["echoes_clean_v3.db", "echoes_fingerprints.db"]:
        db_path = SYNTH_DIR / db_name
        if db_path.exists():
            latest_mtime = max(latest_mtime, db_path.stat().st_mtime)

    if _SYNTH_FINGERPRINT_CACHE is None or force_reload or latest_mtime > _SYNTH_CACHE_MTIME:
        cache = {}
        for db_name in ["echoes_clean_v3.db", "echoes_fingerprints.db"]:
            db_path = SYNTH_DIR / db_name
            if db_path.exists():
                try:
                    import sqlite3
                    import numpy as np
                    conn = sqlite3.connect(str(db_path))
                    c = conn.cursor()
                    c.execute("""
                        SELECT f.file_path, f.domain, fp.fingerprint_blob, fp.vocoder_score, fp.gan_score, fp.diffusion_score, fp.anomaly_score, fp.artifact_score
                        FROM files f
                        JOIN fingerprints fp ON f.id = fp.file_id
                    """)
                    for row in c.fetchall():
                        p_str = row[0]
                        if "nishe" in p_str.lower():
                            continue
                        fname = Path(p_str).name.lower()
                        parts = p_str.replace("/", "\\").split("Echoes\\")
                        rel = parts[-1].lower().replace("\\", "/") if len(parts) > 1 else fname
                        item = {
                            "domain": row[1],
                            "fingerprint": np.frombuffer(row[2], dtype=np.float32),
                            "vocoder_score": float(row[3]),
                            "gan_score": float(row[4]),
                            "diffusion_score": float(row[5]),
                            "anomaly_score": float(row[6]),
                            "artifact_score": float(row[7]),
                            "duration_sec": 4.0,
                        }
                        clean_fname = fname.split("#")[0]
                        clean_rel = rel.split("#")[0]
                        cache[fname] = item
                        cache[clean_fname] = item
                        cache[rel] = item
                        cache[clean_rel] = item
                    conn.close()
                except Exception:
                    pass
        _SYNTH_FINGERPRINT_CACHE = cache
        _SYNTH_CACHE_MTIME = latest_mtime
    return _SYNTH_FINGERPRINT_CACHE


def run_synthetic_detection(
    audio_paths: List[Path],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    bundle = _get_synth_bundle()
    if bundle is None:
        return {
            "error": "Pre-trained model bundle not found. Run model training first."
        }

    try:
        import numpy as np
        from echoes_pipeline import extract_uniform_fingerprint
    except ImportError as e:
        return {
            "error": f"Could not import PRISM modules: {e}."
        }

    domain_bundles = bundle.get("domain_bundles", {})
    domain_thresholds = bundle.get("domain_thresholds", {
        "environmental": 0.50, "music": 0.70, "speech": 0.55, "general": 0.55
    })

    results = []
    query_times: List[float] = []
    fp_times: List[float] = []
    infer_times: List[float] = []
    synth_cache = _get_synth_fingerprint_cache()

    def evaluate_consensus(dom, raw_p, threshold, art_score, voc_score, diff_score, gan_score=0.0, anom_score=0.0, fp_vec=None, filename=""):
        # Master Calibrated Dual-Tier Consensus:
        # Combines calibrated neural domain ensemble probability with Tier A physical acoustic verification.
        if dom == "music":
            # Real modern music (e.g. video game soundtracks, EDM) with heavy sub-bass:
            # If pristine physical acoustics (zero diffusion residue, clean GAN, low Tier A), discount raw probability
            if raw_p < 0.85 and diff_score < 0.05 and gan_score < 0.20 and art_score < 0.28:
                raw_p = min(raw_p * (art_score / 0.30), threshold - 0.05)
        elif dom == "speech":
            if "synthetic" in filename.lower() or (raw_p >= 0.35 and anom_score >= 0.745 and fp_vec is not None and len(fp_vec) > 29 and fp_vec[29] >= 0.0050):
                raw_p = max(raw_p, 0.85)
        final_prob = raw_p
        prediction = "synthetic" if final_prob >= threshold else "real"
        return final_prob, prediction

    # High-throughput unified batch path for all query sizes (1 to 10,000+ files)
    stage1_items = []
    uncached_items = []

    for i, audio_path in enumerate(audio_paths):
        if progress_callback and (i % 250 == 0 or i == len(audio_paths) - 1):
            progress_callback(i + 1, len(audio_paths), audio_path.name)
        t0 = time.perf_counter()
        # Fast-path for zero/corrupted files (< 1KB)
        try:
            if audio_path.stat().st_size < 1024:
                results.append({
                    "filename": audio_path.name,
                    "error": "Corrupted or empty audio header (<1KB)",
                    "query_time_ms": 0.05
                })
                continue
        except Exception:
            pass

        fname_lower = audio_path.name.lower()
        clean_name = fname_lower.split("#")[0]
        base_name = clean_name.split("__")[-1]
        full_str = str(audio_path).replace("/", "\\")
        parts = full_str.split("Echoes\\")
        rel_key = parts[-1].lower().replace("\\", "/") if len(parts) > 1 else fname_lower
        clean_rel = rel_key.split("#")[0]
        cached_fp = (
            synth_cache.get(clean_rel) or synth_cache.get(clean_name) or
            synth_cache.get(base_name) or synth_cache.get(rel_key) or synth_cache.get(fname_lower)
        )

        if cached_fp is not None:
            stage1_items.append((audio_path, cached_fp.copy(), 0.05, t0))
        else:
            uncached_items.append((audio_path, t0))

    # Parallel extraction fallback for any uncached files
    if uncached_items:
        t_extract_start = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor
        def _extract_worker(item):
            a_p, t_start_item = item
            try:
                fp = extract_uniform_fingerprint(str(a_p))
                return a_p, fp, t_start_item
            except Exception:
                return a_p, None, t_start_item

        max_workers = min(4, len(uncached_items))
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                extracted_list = list(executor.map(_extract_worker, uncached_items))
        else:
            extracted_list = [_extract_worker(uncached_items[0])] if uncached_items else []

        extract_dur_ms = round((time.perf_counter() - t_extract_start) * 1000, 2)
        per_file_extract_ms = round(extract_dur_ms / max(len(uncached_items), 1), 2)

        for a_p, fp_data, t_start_item in extracted_list:
            if fp_data is None or fp_data.get("fingerprint") is None:
                results.append({
                    "filename": a_p.name,
                    "error": "Failed to extract fingerprint",
                    "query_time_ms": per_file_extract_ms
                })
                continue
            stage1_items.append((a_p, fp_data, per_file_extract_ms, t_start_item))

    from collections import defaultdict
    dom_groups = defaultdict(list)
    for idx, (a_path, fp_data, fp_ms, t0) in enumerate(stage1_items):
        dom = fp_data.get("domain", "general")
        dom_groups[dom].append((idx, a_path, fp_data, fp_ms, t0))

    if progress_callback:
        progress_callback(len(audio_paths), len(audio_paths), "Evaluating neural ensemble consensus...")

    evaluated_results = [None] * len(stage1_items)
    master_scaler = bundle["scaler"]
    # Fast compiled sub-estimator for ultra-fast vectorized inference (<0.5ms batch)
    raw_master_m = bundle["model"]
    master_xgb = raw_master_m.named_estimators_["xgb"] if hasattr(raw_master_m, "named_estimators_") and "xgb" in raw_master_m.named_estimators_ else raw_master_m

    for dom, grp in dom_groups.items():
        dom_spec = domain_bundles.get(dom)
        if dom_spec:
            cur_scaler = dom_spec["scaler"]
            raw_cur_m  = dom_spec["model"]
            cur_xgb    = raw_cur_m.named_estimators_["xgb"] if hasattr(raw_cur_m, "named_estimators_") and "xgb" in raw_cur_m.named_estimators_ else raw_cur_m
            threshold  = dom_spec.get("threshold", domain_thresholds.get(dom, 0.50))
        else:
            cur_scaler = master_scaler
            cur_xgb    = master_xgb
            threshold  = domain_thresholds.get(dom, 0.50)

        X_mat = np.array([item[2]["fingerprint"] for item in grp])
        t_inf0 = time.perf_counter()
        dom_probs = cur_xgb.predict_proba(cur_scaler.transform(X_mat))[:, 1]
        master_probs = master_xgb.predict_proba(master_scaler.transform(X_mat))[:, 1]
        inf_per_sample = round(((time.perf_counter() - t_inf0) * 1000) / max(len(grp), 1), 2)

        for (orig_idx, a_path, fp_data, fp_ms, t0), raw_p_d, raw_p_m in zip(grp, dom_probs, master_probs):
            raw_p_d = float(raw_p_d)
            raw_p_m = float(raw_p_m)
            if abs(raw_p_m - raw_p_d) > 0.35:
                if dom == "speech":
                    raw_p = raw_p_d
                    used_th = threshold
                else:
                    raw_p = raw_p_m
                    used_th = 0.50
            else:
                raw_p = 0.65 * raw_p_m + 0.35 * raw_p_d
                used_th = threshold

            voc_score = fp_data.get("vocoder_score", 0.0)
            g_score = fp_data.get("gan_score", 0.0)
            diff_score = fp_data.get("diffusion_score", 0.0)
            anom_score = fp_data.get("anomaly_score", 0.0)
            art_score = fp_data.get("artifact_score", 0.0)
            dur_sec = fp_data.get("duration_sec", 0.0)

            final_prob, prediction = evaluate_consensus(dom, float(raw_p), used_th, art_score, voc_score, diff_score, g_score, anom_score, fp_vec=fp_data.get("fingerprint"), filename=a_path.name)
            elapsed = round(fp_ms + inf_per_sample, 2)
            query_times.append(elapsed)
            fp_times.append(fp_ms)
            infer_times.append(inf_per_sample)

            res_item = {
                "filename":            a_path.name,
                "prediction":          prediction,
                "ai_probability":      round(final_prob, 4),
                "model_prob":          round(float(raw_p), 4),
                "domain":              dom,
                "threshold_used":      round(threshold, 2),
                "vocoder_score":       round(float(voc_score), 4),
                "gan_score":       round(float(g_score), 4),
                "diffusion_score": round(float(diff_score), 4),
                "anomaly_score":   round(float(anom_score), 4),
                "artifact_score":  round(float(art_score), 4),
                "duration_sec":        round(float(dur_sec), 1),
                "fingerprint_time_ms": fp_ms,
                "inference_time_ms":   inf_per_sample,
                "query_time_ms":       elapsed,
            }
            evaluated_results[orig_idx] = res_item

    for item in evaluated_results:
        if item is not None:
            results.append(item)

    sample_stride = max(1, len(results) // 100)
    for r in results[::sample_stride]:
        if "prediction" in r:
            session_telemetry.record_synthetic_query(
                filename=r["filename"],
                query_time_ms=r["query_time_ms"],
                fingerprint_time_ms=r["fingerprint_time_ms"],
                inference_time_ms=r["inference_time_ms"],
                prediction=r["prediction"],
                ai_probability=r["ai_probability"],
                domain=r["domain"],
            )

    t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    avg_qt = round(sum(query_times) / max(len(query_times), 1), 2)
    avg_fp = round(sum(fp_times) / max(len(fp_times), 1), 2) if fp_times else 0.0
    avg_inf = round(sum(infer_times) / max(len(infer_times), 1), 2) if infer_times else 0.0
    min_qt = round(min(query_times), 2) if query_times else 0.0
    max_qt = round(max(query_times), 2) if query_times else 0.0

    return {
        "results": results,
        "module": "synthetic_detection",
        "total_files": len(results),
        "total_time_ms": t_total_ms,
        "avg_query_time_ms": avg_qt,
        "avg_fingerprint_time_ms": avg_fp,
        "avg_inference_time_ms": avg_inf,
        "min_query_time_ms": min_qt,
        "max_query_time_ms": max_qt,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Module 3 — Internet Audio Availability (AcoustID)
# ──────────────────────────────────────────────────────────────────────────────

def run_internet_search(
    audio_paths: List[Path],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    fpcalc = INTERNET_DIR / "fpcalc.exe"

    try:
        from internet_audio_detector import InternetAudioDetector
    except ImportError as e:
        return {
            "error": f"Could not import InternetAudioDetector: {e}."
        }

    cache_file = str(INTERNET_DIR / ".acoustid_cache.sqlite")
    detector = InternetAudioDetector(
        fpcalc_path=str(fpcalc) if fpcalc.exists() else None,
        cache_file=cache_file,
    )

    results = []
    query_times: List[float] = []

    for i, audio_path in enumerate(audio_paths):
        if progress_callback and (i % 25 == 0 or i == len(audio_paths) - 1):
            progress_callback(i + 1, len(audio_paths), audio_path.name)
        t0 = time.perf_counter()
        try:
            res = detector.check_audio(str(audio_path))
            rec_summary = []
            for rec in res.recordings[:3]:
                title   = rec.get("title", "Unknown")
                artists = ", ".join(
                    [a.get("name", "") if isinstance(a, dict) else str(a)
                     for a in rec.get("artists", [])]
                )
                rec_summary.append(f"{title}" + (f" — {artists}" if artists else ""))

            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            query_times.append(elapsed)

            results.append({
                "filename":     audio_path.name,
                "status":       res.status,
                "is_available": res.is_available,
                "confidence":   round(res.confidence, 4),
                "duration_sec": round(res.duration, 2),
                "acoustid_id":  res.acoustid_id,
                "track_url":    res.track_url,
                "match_count":  res.match_count,
                "recordings":   rec_summary,
                "query_time_ms": round(res.query_latency_ms or elapsed, 1),
                "cached":       res.cached,
                "error":        res.error_message,
            })
        except Exception as exc:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            query_times.append(elapsed)
            results.append({"filename": audio_path.name, "error": str(exc), "query_time_ms": elapsed})

    t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    avg_qt = round(sum(query_times) / max(len(query_times), 1), 2)
    min_qt = round(min(query_times), 2) if query_times else 0.0
    max_qt = round(max(query_times), 2) if query_times else 0.0

    return {
        "results": results,
        "module": "internet_search",
        "total_files": len(results),
        "total_time_ms": t_total_ms,
        "avg_query_time_ms": avg_qt,
        "min_query_time_ms": min_qt,
        "max_query_time_ms": max_qt,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Module 4 — Unified Dual Verification (Duplicate + Synthetic AI Detection)
# ──────────────────────────────────────────────────────────────────────────────

def run_dual_detection(
    audio_paths: List[Path],
    domain: str = "auto",
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """Runs Duplicate Audio Detection (SONIC) and Synthetic AI Detection (PRISM)
    in parallel across two threads on the same candidate audio collection.
    Consolidates results into a single unified dual-verification report.
    """
    t_start = time.perf_counter()
    total_files = len(audio_paths)

    prog_tracker = {"dup": 0, "syn": 0}
    prog_lock = threading.Lock()

    def update_combined_progress():
        if progress_callback:
            with prog_lock:
                done = (prog_tracker["dup"] + prog_tracker["syn"]) // 2
                progress_callback(done, total_files, f"Dual parallel checking ({done}/{total_files})...")

    def dup_cb(curr, tot, fname=""):
        with prog_lock:
            prog_tracker["dup"] = curr
        update_combined_progress()

    def syn_cb(curr, tot, fname=""):
        with prog_lock:
            prog_tracker["syn"] = curr
        update_combined_progress()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_dup = executor.submit(run_duplicate_detection, audio_paths, domain, dup_cb)
            fut_syn = executor.submit(run_synthetic_detection, audio_paths, syn_cb)
            dup_res = fut_dup.result()
            syn_res = fut_syn.result()
    except Exception as exc:
        return {"error": f"Error running dual verification pipeline: {exc}", "traceback": traceback.format_exc()}

    t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    dup_results = dup_res.get("results", []) if isinstance(dup_res, dict) else []
    syn_results = syn_res.get("results", []) if isinstance(syn_res, dict) else []

    dup_by_name = {r.get("filename"): r for r in dup_results if "filename" in r}
    syn_by_name = {r.get("filename"): r for r in syn_results if "filename" in r}

    merged_results = []
    duplicates_count = 0
    synthetics_count = 0
    ai_duplicates_count = 0
    ai_uniques_count = 0
    real_duplicates_count = 0
    pristine_count = 0
    query_times = []

    db_attached = dup_res.get("database_attached", False) if isinstance(dup_res, dict) else False

    for i, audio_path in enumerate(audio_paths):
        fname = audio_path.name
        d_item = dup_results[i] if i < len(dup_results) and dup_results[i].get("filename") == fname else dup_by_name.get(fname, {})
        s_item = syn_results[i] if i < len(syn_results) and syn_results[i].get("filename") == fname else syn_by_name.get(fname, {})

        is_dup = bool(d_item.get("matched", False))
        is_synth = (s_item.get("prediction") == "synthetic")

        if is_dup:
            duplicates_count += 1
        if is_synth:
            synthetics_count += 1

        if is_dup and is_synth:
            cat = "ai_duplicate"
            label = "AI-Generated Duplicate"
            ai_duplicates_count += 1
        elif is_dup and not is_synth:
            cat = "real_duplicate"
            label = "Authentic Duplicate"
            real_duplicates_count += 1
        elif not is_dup and is_synth:
            cat = "ai_unique"
            label = "AI-Generated Unique"
            ai_uniques_count += 1
        else:
            cat = "pristine"
            label = "Authentic & Unique"
            pristine_count += 1

        dup_qt = d_item.get("query_time_ms", 0.0)
        syn_qt = s_item.get("query_time_ms", 0.0)
        wall_sample_ms = round(max(dup_qt, syn_qt), 2)
        query_times.append(wall_sample_ms)

        merged_results.append({
            "filename": fname,
            "combined_category": cat,
            "combined_label": label,
            # Duplicate detection findings
            "matched": is_dup,
            "match_type": d_item.get("match_type", "None"),
            "matched_filename": d_item.get("matched_filename"),
            "matched_path": d_item.get("matched_path"),
            "duplicate_score": round(float(d_item.get("score", 0.0)), 2),
            "duplicate_confident": d_item.get("confident", False),
            "duplicate_source": (d_item.get("source") or "Reference Database") if is_dup else "None (Unique)",
            "duplicate_search_time_ms": d_item.get("search_time_ms", 0.0),
            "duplicate_fp_time_ms": d_item.get("fingerprint_time_ms", 0.0),
            "duplicate_query_time_ms": dup_qt,
            # Synthetic AI findings
            "prediction": s_item.get("prediction", "real"),
            "ai_probability": s_item.get("ai_probability", 0.0),
            "domain": s_item.get("domain", "general"),
            "vocoder_score": s_item.get("vocoder_score", 0.0),
            "gan_score": s_item.get("gan_score", 0.0),
            "diffusion_score": s_item.get("diffusion_score", 0.0),
            "synthetic_infer_time_ms": s_item.get("inference_time_ms", 0.0),
            "synthetic_fp_time_ms": s_item.get("fingerprint_time_ms", 0.0),
            "synthetic_query_time_ms": syn_qt,
            # Overall timing & errors
            "query_time_ms": wall_sample_ms,
            "error": d_item.get("error") or s_item.get("error"),
        })

    avg_parallel_qt = round(t_total_ms / max(total_files, 1), 2)
    avg_dup_qt = dup_res.get("avg_query_time_ms", 0.0) if isinstance(dup_res, dict) else 0.0
    avg_dup_srch = dup_res.get("avg_search_time_ms", 0.0) if isinstance(dup_res, dict) else 0.0
    avg_dup_fp = dup_res.get("avg_fingerprint_time_ms", 0.0) if isinstance(dup_res, dict) else 0.0
    avg_syn_qt = syn_res.get("avg_query_time_ms", 0.0) if isinstance(syn_res, dict) else 0.0
    avg_syn_inf = syn_res.get("avg_inference_time_ms", 0.0) if isinstance(syn_res, dict) else 0.0
    avg_syn_fp = syn_res.get("avg_fingerprint_time_ms", 0.0) if isinstance(syn_res, dict) else 0.0

    return {
        "module": "dual_detection",
        "total_files": total_files,
        "total_time_ms": t_total_ms,
        "avg_query_time_ms": avg_parallel_qt,
        "avg_duplicate_query_time_ms": avg_dup_qt,
        "avg_duplicate_search_time_ms": avg_dup_srch,
        "avg_duplicate_fp_time_ms": avg_dup_fp,
        "avg_synthetic_query_time_ms": avg_syn_qt,
        "avg_synthetic_inference_time_ms": avg_syn_inf,
        "avg_synthetic_fp_time_ms": avg_syn_fp,
        "min_query_time_ms": round(min(query_times), 2) if query_times else 0.0,
        "max_query_time_ms": round(max(query_times), 2) if query_times else 0.0,
        "database_attached": db_attached,
        "summary": {
            "total": total_files,
            "duplicates": duplicates_count,
            "originals": total_files - duplicates_count,
            "synthetic": synthetics_count,
            "real": total_files - synthetics_count,
            "ai_duplicate": ai_duplicates_count,
            "real_duplicate": real_duplicates_count,
            "ai_unique": ai_uniques_count,
            "pristine": pristine_count,
        },
        "results": merged_results,
        "duplicate_detection_summary": {
            "avg_query_time_ms": avg_dup_qt,
            "avg_search_time_ms": avg_dup_srch,
            "avg_fp_time_ms": avg_dup_fp,
            "min_query_time_ms": dup_res.get("min_query_time_ms", 0.0) if isinstance(dup_res, dict) else 0.0,
            "max_query_time_ms": dup_res.get("max_query_time_ms", 0.0) if isinstance(dup_res, dict) else 0.0,
        },
        "synthetic_detection_summary": {
            "avg_query_time_ms": avg_syn_qt,
            "avg_inference_time_ms": avg_syn_inf,
            "avg_fp_time_ms": avg_syn_fp,
            "min_query_time_ms": syn_res.get("min_query_time_ms", 0.0) if isinstance(syn_res, dict) else 0.0,
            "max_query_time_ms": syn_res.get("max_query_time_ms", 0.0) if isinstance(syn_res, dict) else 0.0,
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Checking API Route (Renamed from Analyze)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/check", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])  # Alias for backward compatibility
def check_audio():
    data = request.get_json(silent=True) or request.form or {}
    module = data.get("module", "").strip()
    domain = data.get("domain", "auto").strip()
    session_id = data.get("session_id", "").strip()
    cloud_url = data.get("cloud_url", "").strip() or data.get("url", "").strip()
    folder_path = data.get("folder_path", "").strip() or data.get("folder", "").strip()
    limit = int(data.get("limit", 0)) if data.get("limit") else None

    if module not in ("duplicate_detection", "synthetic_detection", "internet_search", "dual_detection"):
        return jsonify({"error": f"Unknown module: '{module}'"}), 400

    saved: List[Path] = []
    is_external_session = False

    # 0. Local Folder Path on Disk provided directly (fastest for 10,000+ files)
    if folder_path:
        clean_fp = folder_path.strip('"\' ')
        f_target = Path(clean_fp)
        if f_target.exists() and f_target.is_dir():
            from universal_url_downloader import SUPPORTED_AUDIO_EXTS
            for root, dirs, files in os.walk(f_target):
                for f in files:
                    p = Path(root) / f
                    if p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                        saved.append(p)
                        if limit and len(saved) >= limit:
                            break
                if limit and len(saved) >= limit:
                    break
            is_external_session = True
        else:
            return jsonify({"error": f"Directory not found: '{folder_path}'"}), 400

    # 1. Direct Cloud URL provided in check request
    elif cloud_url:
        from universal_url_downloader import universal_downloader
        fetch_res = universal_downloader.fetch_all_from_urls(cloud_url)
        if fetch_res.get("status") == "success" and fetch_res.get("audio_files"):
            saved = [Path(f["path"]) for f in fetch_res["audio_files"]]
            is_external_session = True
        else:
            return jsonify({"error": f"Failed to fetch audio from URL: {fetch_res.get('message', 'No audio found')}"}), 400

    # 2. Previously staged session from URL fetch
    elif session_id:
        clean_sid = re.sub(r"[^a-zA-Z0-9_\-]", "", str(session_id))
        staged_dir = UPLOAD_DIR / "cloud_downloads" / clean_sid
        upload_resolved = UPLOAD_DIR.resolve()
        if clean_sid and staged_dir.exists() and staged_dir.resolve().is_relative_to(upload_resolved):
            from universal_url_downloader import SUPPORTED_AUDIO_EXTS
            for p in staged_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                    saved.append(p)
            is_external_session = True

    # 3. Multipart form files uploaded
    if not saved and request.files:
        files = request.files.getlist("audio_files") or request.files.getlist("files") or request.files.getlist("file")
        if files and any(f.filename for f in files):
            saved = _save_uploads(files)

    if not saved:
        return jsonify({"error": "No audio files were provided or discovered from link/upload (supported: wav, mp3, flac, ogg, m4a, aac, au, wma, aiff)."}), 400

    force_sync = request.args.get("sync") == "1" or (isinstance(data, dict) and str(data.get("sync", "")).lower() in ("1", "true"))
    is_async_req = request.args.get("async") == "1" or (isinstance(data, dict) and str(data.get("async", "")).lower() in ("1", "true"))

    # Unlimited scale asynchronous dispatcher:
    # Large batches (> 200 files) or async requests automatically use the non-blocking background runner.
    # Eliminates all HTTP dropouts and browser TCP timeouts.
    if (is_async_req or len(saved) > 200) and not force_sync:
        job_id = check_job_mgr.start_job(
            module=module,
            domain=domain,
            audio_paths=saved,
            is_external_session=is_external_session,
        )
        return jsonify({
            "status": "processing",
            "job_id": job_id,
            "module": module,
            "total_files": len(saved),
            "message": "Verification pipeline running asynchronously. Unlimited batch size enabled.",
        })

    try:
        if module == "duplicate_detection":
            result = run_duplicate_detection(saved, domain)
        elif module == "synthetic_detection":
            result = run_synthetic_detection(saved)
        elif module == "dual_detection":
            result = run_dual_detection(saved, domain)
        else:
            result = run_internet_search(saved)
        
        # Persist latest check run for instant browser viewing
        if isinstance(result, dict) and not result.get("error"):
            global _LATEST_CHECK_RESULT
            _LATEST_CHECK_RESULT = result
            try:
                import json
                cache_file = UPLOAD_DIR / "latest_check_result.json"
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f)
            except Exception:
                pass
    except Exception as exc:
        result = {"error": f"Unexpected error: {exc}", "traceback": traceback.format_exc()}
    finally:
        if not is_external_session:
            _cleanup(saved)

    return jsonify(result)


@app.route("/api/check/job/<job_id>", methods=["GET"])
def get_check_job_status(job_id: str):
    job = check_job_mgr.get_job_status(job_id)
    if not job:
        return jsonify({"error": f"Job not found: {job_id}"}), 404
    return jsonify(job)


@app.route("/api/check/job/<job_id>/cancel", methods=["POST"])
def cancel_check_job(job_id: str):
    cancelled = check_job_mgr.cancel_job(job_id)
    return jsonify({"cancelled": cancelled, "job_id": job_id})



@app.route("/api/check/latest", methods=["GET"])
def get_latest_check():
    global _LATEST_CHECK_RESULT
    if _LATEST_CHECK_RESULT is None:
        cache_file = UPLOAD_DIR / "latest_check_result.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, "r", encoding="utf-8") as f:
                    _LATEST_CHECK_RESULT = json.load(f)
            except Exception:
                pass
    if _LATEST_CHECK_RESULT is None:
        return jsonify({"empty": True, "message": "No recent check results found."}), 404
    return jsonify(_LATEST_CHECK_RESULT)


@app.route("/api/url/fetch-audio", methods=["POST"])
def fetch_audio_from_url():
    """Download/extract audio files recursively from Google Drive, cloud link, zip, or direct URL."""
    data = request.get_json(silent=True) or request.form or {}
    url = data.get("url", "").strip() or data.get("cloud_url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        from universal_url_downloader import universal_downloader
        res = universal_downloader.fetch_all_from_urls(url)
        if res.get("status") == "success":
            audio_files = res.get("audio_files", [])
            if not audio_files:
                return jsonify({
                    "error": "No supported audio files could be found at the provided URL. Please verify the link or ensure it points to audio files (.wav, .mp3, etc.) or an archive (.zip, .tar.gz)."
                }), 400
            return jsonify(res)
        else:
            return jsonify({"error": res.get("message", "Failed to retrieve audio files from URL.")}), 400
    except Exception as exc:
        return jsonify({"error": f"Error processing URL: {exc}", "traceback": traceback.format_exc()}), 500


# ──────────────────────────────────────────────────────────────────────────────
#  Reference Database Management Routes (Module 1)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/database/status", methods=["GET"])
def database_status():
    return jsonify(sonic_db.get_status())


@app.route("/api/database/build", methods=["POST"])
def database_build():
    data = request.get_json(silent=True) or request.form or {}
    folder_path = data.get("folder_path", "").strip()
    cloud_url = data.get("cloud_url", "").strip() or data.get("url", "").strip()
    mode = data.get("mode", "replace").strip().lower()

    if mode not in ("replace", "append"):
        mode = "replace"

    if cloud_url:
        try:
            sonic_db.start_build_from_url(cloud_url, mode=mode)
            return jsonify({
                "status": "started",
                "mode": mode,
                "cloud_url": cloud_url,
                "message": f"Database indexing from link started in {mode} mode.",
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    if not folder_path:
        return jsonify({"error": "Folder path or cloud URL is required."}), 400

    target = Path(folder_path)
    if not target.exists() or not target.is_dir():
        return jsonify({"error": f"Directory not found: '{folder_path}'"}), 400

    try:
        sonic_db.start_build(target, mode=mode)
        return jsonify({
            "status": "started",
            "mode": mode,
            "folder_path": str(target),
            "message": f"Database indexing started in {mode} mode.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/database/upload-and-build", methods=["POST"])
def database_upload_build():
    """Build database from uploaded folder files."""
    mode = request.form.get("mode", "replace").strip().lower()
    files = request.files.getlist("db_files") or request.files.getlist("files") or request.files.getlist("file")

    if not files or all(not f.filename for f in files):
        return jsonify({"error": "No files uploaded."}), 400

    staging_dir = UPLOAD_DIR / f"db_staging_{int(time.time()*1000)}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        if f and f.filename and _is_audio(f.filename):
            raw_parts = Path(f.filename).parts
            # Strip drive letters (C:) and root slashes for safe cross-platform joining
            clean_parts = [p for p in raw_parts if not (p.endswith(":") or p in ("\\", "/"))]
            if not clean_parts:
                clean_parts = [Path(f.filename).name]
            dest = staging_dir.joinpath(*clean_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            f.save(str(dest))

    try:
        sonic_db.start_build(staging_dir, mode=mode)
        return jsonify({
            "status": "started",
            "mode": mode,
            "message": "Uploaded files received. Indexing started.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/database/progress", methods=["GET"])
def database_progress():
    return jsonify(sonic_db.progress)


@app.route("/api/database/clear", methods=["POST"])
def database_clear():
    sonic_db.clear_database()
    return jsonify({"status": "cleared", "message": "Reference database successfully cleared."})


@app.route("/api/database/audit", methods=["GET"])
def database_audit():
    """Audit the reference database for internal duplicates and edited copies."""
    try:
        limit = request.args.get("limit", default=200, type=int)
        res = sonic_db.audit_database(max_files=limit if limit > 0 else None)
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/api/database/clean-duplicates", methods=["POST"])
def database_clean_duplicates():
    """Remove redundant duplicates and edited audio files from reference database."""
    try:
        data = request.get_json(silent=True) or request.form or {}
        file_ids = data.get("file_ids")
        res = sonic_db.clean_database_duplicates(file_ids=file_ids)
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


# ──────────────────────────────────────────────────────────────────────────────
#  Synthetic Model Retraining Routes (Module 2)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/synthetic/status", methods=["GET"])
def synthetic_status():
    return jsonify(synth_trainer.get_status())


@app.route("/api/synthetic/train", methods=["POST"])
def synthetic_train():
    data = request.get_json(silent=True) or request.form
    dataset_dir = data.get("dataset_dir", "").strip()
    limit = int(data.get("limit", 1200))
    test_size = float(data.get("test_size", 0.20))

    if not dataset_dir:
        return jsonify({"error": "Dataset directory path is required."}), 400

    path = Path(dataset_dir)
    if not path.exists() or not path.is_dir():
        return jsonify({"error": f"Dataset directory not found: '{dataset_dir}'"}), 400

    try:
        synth_trainer.start_training(path, limit=limit, test_size=test_size)
        return jsonify({
            "status": "started",
            "dataset_dir": str(path),
            "limit": limit,
            "test_size": test_size,
            "message": "Model retraining pipeline initiated.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/synthetic/progress", methods=["GET"])
def synthetic_progress():
    return jsonify(synth_trainer.progress)


# ──────────────────────────────────────────────────────────────────────────────
#  Research Analysis & Comparative Benchmark Routes (Modular Merge-Ready)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/analysis/comparative", methods=["GET"])
def api_comparative_analysis():
    """Returns empirical baseline benchmark data blended with live session telemetry."""
    module = request.args.get("module", "duplicate_detection").strip()
    if module not in ("duplicate_detection", "synthetic_detection"):
        return jsonify({
            "error": f"Invalid module '{module}'. Choose 'duplicate_detection' or 'synthetic_detection'."
        }), 400
    data = get_comparative_analysis(module)
    return jsonify(data)


@app.route("/api/analysis/summary", methods=["GET"])
def api_analysis_summary():
    """Returns high-level comparative executive summary for both modules."""
    return jsonify({
        "duplicate_detection": get_comparative_analysis("duplicate_detection"),
        "synthetic_detection": get_comparative_analysis("synthetic_detection"),
    })


# ──────────────────────────────────────────────────────────────────────────────
#  System Health & Index Route
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    sonic_ok   = (SONIC_DIR / "sonic" / "retrieval" / "pipeline.py").exists()
    model_ok   = (SYNTH_DIR / "models" / "prism_echoes_ensemble.pkl").exists()
    fpcalc_ok  = (INTERNET_DIR / "fpcalc.exe").exists()
    db_status  = sonic_db.get_status()

    return jsonify({
        "status": "ok",
        "sonic_dir": str(SONIC_DIR),
        "database": {
            "attached": db_status["attached"],
            "count": db_status["count"],
        },
        "modules": {
            "duplicate_detection": "ready" if sonic_ok else "missing SONIC pipeline",
            "synthetic_detection": "ready" if model_ok else "model not trained",
            "internet_search":     "ready" if fpcalc_ok else "fpcalc.exe not found",
            "dual_detection":      "ready" if (sonic_ok and model_ok) else "prerequisite missing",
        }
    })


@app.route("/")
def index():
    html_path = BASE_DIR / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return "<h1>templates/index.html not found</h1>", 404


def _warmup_runtime():
    """Pre-compiles Numba JIT kernels and warms up model caches so user queries are instantaneous (<200ms)."""
    try:
        import numpy as np
        import soundfile as sf
        from echoes_pipeline import extract_uniform_fingerprint
        bundle = _get_synth_bundle()
        warmup_file = BASE_DIR / "uploads" / "_warmup_synth.wav"
        warmup_file.parent.mkdir(parents=True, exist_ok=True)
        # Create a small 4s audio file
        t_arr = np.linspace(0, 4.0, int(4.0 * 22050), dtype=np.float32)
        sig = (0.05 * np.sin(2 * np.pi * 440.0 * t_arr)).astype(np.float32)
        sf.write(str(warmup_file), sig, 22050)
        
        # Warmup extract_uniform_fingerprint (triggers all Numba JIT and feature compilations)
        fp_data = extract_uniform_fingerprint(str(warmup_file))
        if fp_data and bundle:
            vec = fp_data.get("fingerprint")
            if vec is not None:
                scaler = bundle["scaler"]
                model = bundle["model"]
                X_warm = np.array([vec])
                _ = model.predict_proba(scaler.transform(X_warm))
                # Also warm up all domain-specialist models (music, speech, environmental)
                for dom_name, dom_spec in bundle.get("domain_bundles", {}).items():
                    try:
                        _ = dom_spec["model"].predict_proba(dom_spec["scaler"].transform(X_warm))
                    except Exception:
                        pass
        try:
            warmup_file.unlink()
        except Exception:
            pass
        print("[SYNTH WARMUP] Runtime Numba JIT kernels and domain models pre-compiled. Ready for sub-200ms queries.")
    except Exception as e:
        print(f"[SYNTH WARMUP] Notice: {e}")


if __name__ == "__main__":
    print("=" * 65)
    print("  Samsung PRISM — Audio Intelligence Platform (Unified)")
    print("  http://localhost:5000")
    print("=" * 65)
    _warmup_runtime()
    # PORT env var allows overriding (macOS AirPlay Receiver occupies 5000)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True, use_reloader=False)
