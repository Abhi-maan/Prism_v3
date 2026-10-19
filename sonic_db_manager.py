"""
SONIC Reference Database Manager
================================
Provides persistent reference database storage, recursive audio scanning,
neural audio fingerprint extraction, active progress tracking, and index
assembly for Duplicate and Edited Audio Detection.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np

# Ensure SONIC package is on path
SONIC_ROOT = Path(__file__).parent / "SONIC-fixed_1"
if str(SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SONIC_ROOT))

from sonic.config import ANALYSIS_SR, QUERY_MAX_DURATION_SEC
from sonic.io import load_audio
from sonic.fingerprint.landmarks import stft_magnitude, build_landmarks_from_magnitude
from sonic.fingerprint.binary_embed import (
    band_energy_vector_from_magnitude,
    sub_fingerprints_from_magnitude,
    fine_sub_fingerprints_from_magnitude,
    BinaryEmbedIndex,
)
from sonic.index.landmark_index import LandmarkIndex
from sonic.index.binary_hnsw_index import BinaryHNSWIndex
from sonic.index.subfingerprint_index import SubFingerprintIndex
from sonic.retrieval.pipeline import DomainIndex, query, build_domain_index


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".au", ".wma", ".aiff"}


class SONICDatabaseManager:
    """Manages the reference fingerprint database for duplicate audio detection."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or (Path(__file__).parent / "sonic_db_store")
        self.manifest_path = self.store_dir / "manifest.json"
        self.fp_dir = self.store_dir / "fingerprints"

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.fp_dir.mkdir(parents=True, exist_ok=True)

        self.manifest: List[Dict[str, Any]] = []
        self.domain_index: Optional[DomainIndex] = None
        self._lock = threading.Lock()
        self._cached_duration: float = 0.0
        self._cached_sources: List[str] = []

        # Active progress tracker
        self.progress: Dict[str, Any] = {
            "active": False,
            "mode": "idle",  # "replace" or "append"
            "total": 0,
            "current": 0,
            "percent": 0.0,
            "current_file": "",
            "message": "Idle",
            "error": None,
            "start_time": 0.0,
            "elapsed_sec": 0.0,
        }

        # Initialize existing database asynchronously in background so Flask starts instantly
        threading.Thread(target=self._load_and_init, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    #  State & Manifest
    # ──────────────────────────────────────────────────────────────────────────

    def _load_and_init(self) -> None:
        """Load manifest from disk and rebuild DomainIndex if files exist."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.manifest = data.get("files", [])
                if self.manifest:
                    self._cached_duration = round(sum(f.get("duration_sec", 0.0) for f in self.manifest), 1)
                    self._cached_sources = sorted(list(set(f.get("source_folder", "Unknown") for f in self.manifest)))
                    self._manifest_by_id = {e["file_id"]: e for e in self.manifest}
                    print(f"[SONIC DB] Loading existing database with {len(self.manifest)} files...")
                    self._rebuild_index()
                    print(f"[SONIC DB] Database index ready ({len(self.manifest)} items).")
            except Exception as e:
                print(f"[SONIC DB] Error loading existing database: {e}")
                self.manifest = []
                self.domain_index = None

    def _save_manifest(self) -> None:
        """Persist current manifest to disk."""
        self._cached_duration = round(sum(f.get("duration_sec", 0.0) for f in self.manifest), 1)
        self._cached_sources = sorted(list(set(f.get("source_folder", "Unknown") for f in self.manifest)))
        data = {
            "count": len(self.manifest),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": self.manifest,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            # Compact JSON serialization without indent for high performance at 2M scale
            json.dump(data, f, separators=(",", ":"))

    def get_status(self) -> Dict[str, Any]:
        """Return live health and metadata of the reference database in O(1) time."""
        with self._lock:
            attached = len(self.manifest) > 0 and self.domain_index is not None
            last_updated = "Never"
            if self.manifest_path.exists():
                last_updated = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(self.manifest_path.stat().st_mtime)
                )

            return {
                "attached": attached,
                "count": len(self.manifest),
                "total_duration_sec": self._cached_duration,
                "last_updated": last_updated,
                "store_dir": str(self.store_dir),
                "sources": self._cached_sources,
                "sample_files": [f["filename"] for f in self.manifest[:20]],
                "progress": dict(self.progress),
            }

    def start_build_from_url(self, url: str, mode: str = "replace") -> None:
        """Start asynchronous database indexing from a cloud URL or dataset archive."""
        if self.progress["active"]:
            raise RuntimeError("Database build is already in progress.")
        t = threading.Thread(
            target=self._build_url_worker,
            args=(url, mode),
            daemon=True,
        )
        t.start()

    def _build_url_worker(self, url: str, mode: str) -> None:
        t_start = time.perf_counter()
        with self._lock:
            self.progress.update({
                "active": True,
                "mode": mode,
                "total": 0,
                "current": 0,
                "percent": 0.0,
                "current_file": "Connecting to cloud dataset URL...",
                "message": "Fetching dataset from cloud URL (Drive/ZIP/Link)...",
                "error": None,
                "start_time": t_start,
                "elapsed_sec": 0.0,
            })
        try:
            from universal_url_downloader import universal_downloader

            def _prog_cb(info):
                with self._lock:
                    pct = min(40.0, float(info.get("percent", 0.0)) * 0.40)
                    self.progress.update({
                        "percent": round(pct, 1),
                        "current_file": info.get("current_file", "Downloading..."),
                        "message": f"Cloud download: {info.get('current_file', '')}",
                    })

            fetch_res = universal_downloader.fetch_all_from_urls(url, progress_cb=_prog_cb)
            if fetch_res.get("status") != "success" or not fetch_res.get("audio_files"):
                raise ValueError(fetch_res.get("message", "No audio files discovered from URL."))

            target_dir = Path(fetch_res["target_dir"])
            self._build_worker(target_dir, mode)
        except Exception as exc:
            with self._lock:
                self.progress.update({
                    "active": False,
                    "error": str(exc),
                    "message": f"Cloud URL build failed: {exc}",
                })

    # ──────────────────────────────────────────────────────────────────────────
    #  File Scanning & Fingerprint Extraction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def scan_audio_files(folder_path: Path | str) -> List[Path]:
        """Recursively scan a directory for all audio files in supported formats."""
        root = Path(folder_path)
        if not root.exists() or not root.is_dir():
            return []

        found: List[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                found.append(p)
        return sorted(found, key=lambda x: str(x).lower())

    @staticmethod
    def extract_fingerprint(audio_path: str | Path) -> Dict[str, Any]:
        """Extract SONIC neural fingerprints and landmarks from a single audio file.
        Uses exact SONIC pipeline: STFT -> Landmarks -> Band Energy -> Multi-window Sub-fingerprints.
        """
        y = load_audio(str(audio_path), sr=ANALYSIS_SR)
        duration = float(len(y) / ANALYSIS_SR) if len(y) > 0 else 0.0

        mag, frame_times = stft_magnitude(y, ANALYSIS_SR)
        landmarks = build_landmarks_from_magnitude(mag, frame_times, ANALYSIS_SR)
        raw_vec = band_energy_vector_from_magnitude(mag)

        embed_idx = BinaryEmbedIndex()
        subfp_codes = []
        for _start_time, sv in sub_fingerprints_from_magnitude(mag, frame_times):
            subfp_codes.append(embed_idx.binarize(sv))

        subfp_fine_codes = []
        for _start_time, sv in fine_sub_fingerprints_from_magnitude(mag, frame_times):
            subfp_fine_codes.append(embed_idx.binarize(sv))

        return {
            "duration_sec": duration,
            "hash": landmarks.hash,
            "anchor_time": landmarks.anchor_time,
            "raw_vec": raw_vec,
            "subfp": np.stack(subfp_codes) if subfp_codes else np.empty((0, 8), dtype=np.uint8),
            "subfp_fine": np.stack(subfp_fine_codes) if subfp_fine_codes else np.empty((0, 8), dtype=np.uint8),
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Database Building (Append or Replace)
    # ──────────────────────────────────────────────────────────────────────────

    def start_build(self, folder_path: str | Path, mode: str = "replace") -> None:
        """Start database building in a dedicated background thread."""
        if self.progress["active"]:
            raise RuntimeError("Database building is already in progress.")

        t = threading.Thread(
            target=self._build_worker,
            args=(Path(folder_path), mode),
            daemon=True,
        )
        t.start()

    def _build_worker(self, folder_path: Path, mode: str) -> None:
        t_start = time.perf_counter()
        with self._lock:
            self.progress.update({
                "active": True,
                "mode": mode,
                "total": 0,
                "current": 0,
                "percent": 0.0,
                "current_file": "Scanning directory...",
                "message": f"Scanning directory '{folder_path}' recursively...",
                "error": None,
                "start_time": t_start,
                "elapsed_sec": 0.0,
            })

        try:
            audio_files = self.scan_audio_files(folder_path)
            if not audio_files:
                raise ValueError(f"No audio files found in '{folder_path}' (supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTS))})")

            # Determine existing files and next file ID
            with self._lock:
                if mode == "replace":
                    # Clear out older database
                    self.manifest = []
                    self.domain_index = None
                    if self.fp_dir.exists():
                        shutil.rmtree(self.fp_dir)
                    self.fp_dir.mkdir(parents=True, exist_ok=True)
                    existing_paths = set()
                    next_file_id = 0
                else:  # append / merge over older database
                    existing_paths = {str(Path(f["path"]).resolve()) for f in self.manifest}
                    next_file_id = max([f["file_id"] for f in self.manifest], default=-1) + 1

            # Filter out files already in database (for append mode)
            files_to_index = [f for f in audio_files if str(f.resolve()) not in existing_paths]
            
            # If appending and active database index is present, filter out duplicates of existing database files
            skipped_existing: List[Dict[str, str]] = []
            if mode == "append" and self.domain_index is not None and files_to_index:
                with self._lock:
                    self.progress.update({
                        "message": "Checking incoming files against existing database for duplicates...",
                    })
                unique_candidates = []
                for f in files_to_index:
                    db_match = self.query_file(f)
                    if db_match and db_match.get("matched"):
                        skipped_existing.append({
                            "filename": f.name,
                            "matched_with": db_match.get("matched_filename") or "Existing file",
                            "match_type": db_match.get("match_type") or "Duplicate",
                        })
                    else:
                        unique_candidates.append(f)
                files_to_index = unique_candidates

            if not files_to_index and mode == "append":
                with self._lock:
                    self.progress.update({
                        "active": False,
                        "current": len(audio_files),
                        "total": len(audio_files),
                        "percent": 100.0,
                        "current_file": "Done",
                        "message": f"All incoming files were already in database or matched existing records ({len(skipped_existing)} duplicates/edits skipped).",
                    })
                return

            total_count = len(files_to_index)
            with self._lock:
                self.progress.update({
                    "total": total_count,
                    "current": 0,
                    "message": f"Extracting SONIC neural fingerprints (0 / {total_count})...",
                })

            from concurrent.futures import ThreadPoolExecutor, as_completed

            num_workers = min(8, os.cpu_count() or 4)
            folder_name_str = str(folder_path.name or folder_path)

            def _process_one(item: Tuple[int, Path]) -> Optional[Dict[str, Any]]:
                idx, fpath = item
                cur_id = next_file_id + idx
                fname = fpath.name
                try:
                    fp = self.extract_fingerprint(fpath)
                    npz_filename = f"fp_{cur_id}.npz"
                    npz_path = self.fp_dir / npz_filename

                    with open(npz_path, "wb") as f_out:
                        np.savez_compressed(
                            f_out,
                            hash=fp["hash"],
                            anchor_time=fp["anchor_time"],
                            raw_vec=fp["raw_vec"],
                            subfp=fp["subfp"],
                            subfp_fine=fp["subfp_fine"],
                        )

                    return {
                        "file_id": cur_id,
                        "filename": fname,
                        "path": str(fpath.resolve()),
                        "category": "audio",
                        "duration_sec": fp["duration_sec"],
                        "fp_file": npz_filename,
                        "source_folder": folder_name_str,
                        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                except Exception as ex:
                    print(f"[SONIC DB] Warning: Failed to fingerprint '{fname}': {ex}")
                    return None

            items = list(enumerate(files_to_index))
            completed_count = 0
            new_entries: List[Dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_process_one, item) for item in items]
                for fut in as_completed(futures):
                    entry = fut.result()
                    if entry is not None:
                        new_entries.append(entry)
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == total_count:
                        with self._lock:
                            elapsed = time.perf_counter() - t_start
                            pct = round((completed_count / max(total_count, 1)) * 100, 1)
                            fname = entry["filename"] if entry else ""
                            self.progress.update({
                                "current": completed_count,
                                "percent": pct,
                                "current_file": fname,
                                "elapsed_sec": round(elapsed, 1),
                                "message": f"Fingerprinting [{completed_count}/{total_count}]: {fname}",
                            })

            new_entries.sort(key=lambda x: x["file_id"])

            # Intra-batch deduplication: filter out duplicate/edited copies within this new batch (for batches <= 300)
            accepted_entries: List[Dict[str, Any]] = []
            skipped_batch: List[Dict[str, str]] = []
            if 1 < len(new_entries) <= 300:
                with self._lock:
                    self.progress.update({
                        "percent": 96.0,
                        "current_file": "Deduplicating",
                        "message": "Checking incoming batch for internal duplicates and edits...",
                    })
                batch_entries = [
                    {"file_id": e["file_id"], "category": "auto", "filename": e["filename"], "path": e["path"]}
                    for e in new_entries
                ]
                batch_idx = build_domain_index("speech", batch_entries)
                skipped_batch_ids = set()

                for e in new_entries:
                    fid = e["file_id"]
                    if fid in skipped_batch_ids:
                        continue
                    accepted_entries.append(e)

                    try:
                        res_b = query(e["path"], batch_idx, top_k=5)
                        for cand_fid in (res_b.ranked_ids or []):
                            if cand_fid != fid and cand_fid not in skipped_batch_ids:
                                cand_entry = next((x for x in new_entries if x["file_id"] == cand_fid), None)
                                if cand_entry:
                                    is_match, match_type = self._classify_match(e["path"], cand_entry["path"], res_b)
                                    if is_match and match_type in ("Duplicate", "Edited Audio"):
                                        skipped_batch_ids.add(cand_fid)
                                        skipped_batch.append({
                                            "filename": cand_entry["filename"],
                                            "matched_with": e["filename"],
                                            "match_type": match_type,
                                        })
                                        # Delete redundant fingerprint file
                                        npz_path = self.fp_dir / cand_entry["fp_file"]
                                        if npz_path.exists():
                                            try:
                                                npz_path.unlink()
                                            except Exception:
                                                pass
                    except Exception as ex_b:
                        print(f"[SONIC DB] Intra-batch cross check warning for '{e['filename']}': {ex_b}")
            else:
                accepted_entries = new_entries

            # Update manifest and rebuild index with unique accepted entries
            with self._lock:
                self.progress.update({
                    "current": total_count,
                    "percent": 99.0,
                    "current_file": "Finalizing",
                    "message": "Assembling unified neural index from stored fingerprints...",
                })
                self.manifest.extend(accepted_entries)
                self._save_manifest()
                self._rebuild_index()

                t_total = time.perf_counter() - t_start
                total_skipped = len(skipped_existing) + len(skipped_batch)
                summary_details = f" ({total_skipped} redundant copies skipped: {len(skipped_existing)} against DB, {len(skipped_batch)} in batch)" if total_skipped > 0 else ""
                self.progress.update({
                    "active": False,
                    "percent": 100.0,
                    "current": total_count,
                    "current_file": "Complete",
                    "elapsed_sec": round(t_total, 1),
                    "message": f"Successfully indexed {len(accepted_entries)} unique audio files in {t_total:.1f}s!{summary_details}",
                })
                print(f"[SONIC DB] Build complete: {len(self.manifest)} total files in database{summary_details}.")


        except Exception as exc:
            import traceback
            traceback.print_exc()
            with self._lock:
                self.progress.update({
                    "active": False,
                    "error": str(exc),
                    "message": f"Error during indexing: {exc}",
                })

    def _rebuild_index(self) -> None:
        """Assembles DomainIndex directly from saved .npz fingerprints in memory."""
        if not self.manifest:
            self.domain_index = None
            return

        archive_path = self.store_dir / "unified_archive.npz"
        faiss_path = self.store_dir / "hnsw_index.faiss"

        # Auto-recombine multi-part archive chunks if monolithic archive is missing
        if not archive_path.exists():
            parts = sorted(list(self.store_dir.glob("unified_archive.part*.bin")))
            if parts:
                try:
                    print(f"[SONIC DB] Restoring unified database archive from {len(parts)} parts...")
                    with open(archive_path, "wb") as out_f:
                        for p in parts:
                            with open(p, "rb") as in_f:
                                shutil.copyfileobj(in_f, out_f)
                    print("[SONIC DB] Unified archive restored successfully.")
                except Exception as e:
                    print(f"[SONIC DB] Error restoring archive parts: {e}")

        manifest_fids = {e["file_id"] for e in self.manifest}
        if archive_path.exists() and faiss_path.exists():
            try:
                import faiss
                data = np.load(archive_path)
                file_ids = data["file_ids"]
                if len(file_ids) == len(self.manifest) and set(file_ids) == manifest_fids:
                    n_files = len(file_ids)
                    lidx = LandmarkIndex()
                    lidx.file_id_to_local = {fid: i for i, fid in enumerate(file_ids)}
                    lidx.local_to_file_id = file_ids
                    lidx.n_files = n_files
                    lidx.n_hashes_per_file = dict(zip(file_ids, data["n_hashes"]))
                    lidx.build_from_flat(data["hashes"], data["lids"], data["times"])

                    bidx = BinaryEmbedIndex()
                    bidx.build(file_ids, data["raw_vecs"])

                    hidx = BinaryHNSWIndex()
                    hidx.index = faiss.read_index_binary(str(faiss_path))
                    hidx.file_ids = file_ids

                    sidx = SubFingerprintIndex()
                    sidx.build(data["subfp_fids"], data["subfp_codes"])

                    sidx_fine = SubFingerprintIndex()
                    sidx_fine.build(data["subfp_fine_fids"], data["subfp_fine_codes"])

                    self.domain_index = DomainIndex(
                        category="general",
                        landmark_index=lidx,
                        binary_embed=bidx,
                        binary_hnsw=hidx,
                        subfp_index=sidx,
                        subfp_fine_index=sidx_fine,
                        query_max_duration_sec=QUERY_MAX_DURATION_SEC,
                    )
                    return
            except Exception as e:
                print(f"[SONIC DB] Unified archive load fallback: {e}")

        file_ids = [e["file_id"] for e in self.manifest]
        embed_idx = BinaryEmbedIndex()

        lidx = LandmarkIndex()
        lidx.file_id_to_local = {fid: i for i, fid in enumerate(file_ids)}
        lidx.local_to_file_id = np.array(file_ids, dtype=np.int64)
        lidx.n_files = len(file_ids)

        all_hashes: List[np.ndarray] = []
        all_lids: List[np.ndarray] = []
        all_times: List[np.ndarray] = []
        all_raw_vecs: List[np.ndarray] = []

        subfp_fids_chunks: List[np.ndarray] = []
        subfp_code_chunks: List[np.ndarray] = []

        subfp_fine_fids_chunks: List[np.ndarray] = []
        subfp_fine_code_chunks: List[np.ndarray] = []

        for e in self.manifest:
            fid = e["file_id"]
            local = lidx.file_id_to_local[fid]
            npz_path = self.fp_dir / e["fp_file"]

            if not npz_path.exists():
                continue

            with np.load(npz_path) as data:
                h = data["hash"]
                t = data["anchor_time"]
                rv = data["raw_vec"]
                s_codes = data["subfp"]
                sf_codes = data["subfp_fine"]

                n = h.size
                lidx.n_hashes_per_file[fid] = int(n)
                if n > 0:
                    all_hashes.append(h)
                    all_lids.append(np.full(n, local, dtype=np.int32))
                    all_times.append(t)

                all_raw_vecs.append(rv)

                if s_codes.shape[0] > 0:
                    subfp_fids_chunks.append(np.full(s_codes.shape[0], fid, dtype=np.int64))
                    subfp_code_chunks.append(s_codes)

                if sf_codes.shape[0] > 0:
                    subfp_fine_fids_chunks.append(np.full(sf_codes.shape[0], fid, dtype=np.int64))
                    subfp_fine_code_chunks.append(sf_codes)

        # Build Landmark Index
        if all_hashes:
            lidx.build_from_flat(
                np.concatenate(all_hashes),
                np.concatenate(all_lids),
                np.concatenate(all_times),
            )
        else:
            lidx.build_from_flat(
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
            )

        # Build Binary Embed Index
        bidx = BinaryEmbedIndex()
        if all_raw_vecs:
            bidx.build(file_ids, np.stack(all_raw_vecs))

        # Build Binary HNSW Index
        hidx = BinaryHNSWIndex()
        if bidx.codes.shape[0] > 0:
            hidx.build(file_ids, bidx.codes)

        # Build SubFingerprint Indices
        sidx = SubFingerprintIndex()
        if subfp_code_chunks:
            sidx.build(np.concatenate(subfp_fids_chunks), np.concatenate(subfp_code_chunks))

        sidx_fine = SubFingerprintIndex()
        if subfp_fine_code_chunks:
            sidx_fine.build(np.concatenate(subfp_fine_fids_chunks), np.concatenate(subfp_fine_code_chunks))

        self.domain_index = DomainIndex(
            category="general",
            landmark_index=lidx,
            binary_embed=bidx,
            binary_hnsw=hidx,
            subfp_index=sidx,
            subfp_fine_index=sidx_fine,
            query_max_duration_sec=QUERY_MAX_DURATION_SEC,
        )

    def clear_database(self) -> None:
        """Completely detach and remove the reference database."""
        with self._lock:
            self.manifest = []
            self.domain_index = None
            if self.fp_dir.exists():
                shutil.rmtree(self.fp_dir)
            self.fp_dir.mkdir(parents=True, exist_ok=True)
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            print("[SONIC DB] Reference database cleared.")

    # ──────────────────────────────────────────────────────────────────────────
    #  Query / Duplicate Search
    # ──────────────────────────────────────────────────────────────────────────

    def query_file(self, query_audio_path: str | Path) -> Optional[Dict[str, Any]]:
        """Search query audio against the attached reference database."""
        with self._lock:
            if self.domain_index is None or not self.manifest:
                return None
            idx = self.domain_index

        t0 = time.perf_counter()
        res = query(str(query_audio_path), idx)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        matched_entry = self._manifest_by_id.get(res.file_id) if (res.file_id is not None and hasattr(self, '_manifest_by_id') and self._manifest_by_id) else None
        if matched_entry is None and res.file_id is not None:
            matched_entry = next((e for e in self.manifest if e["file_id"] == res.file_id), None)

        is_matched = False
        match_type = None

        if matched_entry:
            is_matched, match_type = self._classify_match(str(query_audio_path), matched_entry["path"], res)

        timing = getattr(res, "timing_ms", {}) or {}
        fp_ms = round(float(timing.get("load_ms", 0.0) + timing.get("primary_fingerprint_ms", 0.0) + timing.get("secondary_fingerprint_ms", 0.0)), 2)
        srch_ms = round(float(timing.get("primary_search_ms", 0.0) + timing.get("secondary_search_ms", 0.0) + timing.get("subfp_vote_ms", 0.0)), 2)
        if srch_ms <= 0.0:
            srch_ms = max(0.1, round(elapsed_ms - fp_ms, 2))
        if fp_ms <= 0.0:
            fp_ms = max(0.1, round(elapsed_ms - srch_ms, 2))

        return {
            "source": "Reference Database",
            "matched": is_matched,
            "matched_filename": matched_entry["filename"] if is_matched else None,
            "matched_path": matched_entry["path"] if is_matched else None,
            "match_type": match_type,
            "score": round(res.weighted_score, 4),
            "confident": res.confident,
            "stage_used": res.stage_used,
            "escalated": res.escalated,
            "fingerprint_time_ms": fp_ms,
            "search_time_ms": srch_ms,
            "query_time_ms": round(elapsed_ms, 2),
            "timing_breakdown": timing,
        }

    @classmethod
    def _classify_match(cls, qpath: str, mpath: str, res: Any) -> tuple[bool, Optional[str]]:
        """
        Carefully classifies whether the query audio is:
        1. Exact Duplicate: (True, "Duplicate")
        2. Edited Audio: (True, "Edited Audio")
        3. No Match / Different Audio: (False, None)
        """
        try:
            m_path = Path(mpath)
            q_path = Path(qpath)
            if not m_path.exists():
                return (True, "Duplicate") if getattr(res, "confident", False) else (False, None)

            # Metadata fast check: Exact same filename / stem with landmark confirmation
            w_score = getattr(res, "weighted_score", 0.0)
            is_conf = getattr(res, "confident", False)
            p_conf = getattr(res, "primary_confidence", 0.0)
            off_conf = getattr(res, "offset_confidence", 0.0)
            margin = getattr(res, "margin", 0.0)

            if m_path.name.lower() == q_path.name.lower() and (is_conf or w_score >= 10.0 or off_conf >= 0.50):
                return (True, "Duplicate")

            # High-confidence landmark alignment fast-path (bypasses redundant audio reload)
            if is_conf and w_score >= 35.0 and off_conf >= 0.60:
                return (True, "Duplicate")

            # If filenames differ, require minimum landmark retrieval evidence first
            if not is_conf and w_score < 10.0:
                return (False, None)

            yq = load_audio(qpath, sr=ANALYSIS_SR, max_duration=QUERY_MAX_DURATION_SEC)
            yd = load_audio(mpath, sr=ANALYSIS_SR, max_duration=QUERY_MAX_DURATION_SEC)
            if yq.size == 0 or yd.size == 0:
                return (False, None)

            n = min(len(yq), len(yd))
            norm_q = float(np.linalg.norm(yq[:n]))
            norm_d = float(np.linalg.norm(yd[:n]))
            if norm_q < 1e-6 or norm_d < 1e-6:
                corr = 0.0
            else:
                corr = float(np.dot(yq[:n], yd[:n]) / (norm_q * norm_d))
            diff = float(np.mean(np.abs(yq[:n] - yd[:n])))

            # 1. Exact or High-Fidelity Duplicate Match (WAV, MP3 re-encoded, resampled, format copy)
            if corr >= 0.65 or (diff < 0.05 and corr >= 0.45):
                return (True, "Duplicate")

            if is_conf and off_conf >= 0.70 and corr >= 0.45:
                return (True, "Duplicate")

            # 2. Genuine Edited Audio (pitch/tempo shift, noise addition, EQ)
            # Requires genuine shared landmarks AND non-zero acoustic correlation
            has_primary_edit = (is_conf and w_score >= 35.0 and p_conf >= 0.08 and off_conf >= 0.18 and corr >= 0.15)
            has_secondary_edit = (w_score >= 80.0 and margin >= 0.30 and corr >= 0.18)
            has_noise_edit = (w_score >= 10.0 and off_conf >= 0.12 and corr >= 0.20)

            if has_primary_edit or has_secondary_edit or has_noise_edit:
                return (True, "Edited Audio")

            # 3. Open-world negative / unrelated audio (different song)
            return (False, None)
        except Exception:
            return (False, None)

    @classmethod
    def _classify_match_type(cls, qpath: str, mpath: str, res: Any = None) -> Optional[str]:
        """Legacy helper for match type string."""
        if res is not None:
            is_matched, mtype = cls._classify_match(qpath, mpath, res)
            return mtype if is_matched else None
        try:
            if not Path(mpath).exists():
                return "Duplicate"
            yq = load_audio(qpath, sr=ANALYSIS_SR, max_duration=QUERY_MAX_DURATION_SEC)
            yd = load_audio(mpath, sr=ANALYSIS_SR, max_duration=QUERY_MAX_DURATION_SEC)
            if yq.size == 0 or yd.size == 0:
                return None
            n = min(yq.size, yd.size)
            len_ratio = abs(yq.size - yd.size) / max(yq.size, yd.size, 1)
            diff = float(np.mean(np.abs(yq[:n] - yd[:n])))
            if diff < 1e-4 and len_ratio < 0.01:
                return "Duplicate"
            if diff < 0.20:
                return "Edited Audio"
            return None
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Database Audit & De-duplication
    # ──────────────────────────────────────────────────────────────────────────

    def audit_database(self, max_files: Optional[int] = 200) -> Dict[str, Any]:
        """
        Audits all files in the reference database to detect internal duplicates 
        and edited audio copies. Returns duplicate pairs and statistics.
        """
        with self._lock:
            if not self.manifest or self.domain_index is None:
                return {
                    "total_files": 0,
                    "audited_files": 0,
                    "duplicate_count": 0,
                    "edited_count": 0,
                    "clean": True,
                    "duplicates": [],
                    "message": "No reference database is attached."
                }
            manifest_copy = list(self.manifest)
            idx = self.domain_index

        target_files = manifest_copy[:max_files] if (max_files and len(manifest_copy) > max_files) else manifest_copy
        by_id = {e["file_id"]: e for e in manifest_copy}
        duplicates: List[Dict[str, Any]] = []
        already_flagged: set[int] = set()

        for entry in target_files:
            fid = entry["file_id"]
            if fid in already_flagged:
                continue

            fpath = entry.get("path")
            if not fpath or not Path(fpath).exists():
                continue

            try:
                res = query(str(fpath), idx, top_k=10)
                for cand_fid in (res.ranked_ids or []):
                    if cand_fid == fid or cand_fid in already_flagged:
                        continue

                    other_entry = by_id.get(cand_fid)
                    if not other_entry or not Path(other_entry.get("path", "")).exists():
                        continue

                    is_match, match_type = self._classify_match(str(fpath), other_entry["path"], res)
                    if is_match and match_type in ("Duplicate", "Edited Audio"):
                        canonical = entry if entry["file_id"] < other_entry["file_id"] else other_entry
                        redundant = other_entry if entry["file_id"] < other_entry["file_id"] else entry

                        if redundant["file_id"] not in already_flagged:
                            already_flagged.add(redundant["file_id"])
                            duplicates.append({
                                "canonical_id": canonical["file_id"],
                                "canonical_filename": canonical["filename"],
                                "canonical_path": canonical["path"],
                                "redundant_id": redundant["file_id"],
                                "redundant_filename": redundant["filename"],
                                "redundant_path": redundant["path"],
                                "match_type": match_type,
                                "score": round(res.weighted_score, 4),
                            })
            except Exception as e:
                print(f"[SONIC DB] Audit error on file '{entry.get('filename')}': {e}")

        dup_count = sum(1 for d in duplicates if d["match_type"] == "Duplicate")
        edit_count = sum(1 for d in duplicates if d["match_type"] == "Edited Audio")

        is_sampled = (max_files and len(manifest_copy) > max_files)
        sample_note = f" (audited sample of {len(target_files)} files out of {len(manifest_copy)} total)" if is_sampled else ""
        msg = f"Audit complete{sample_note}: Found {dup_count} duplicate(s) and {edit_count} edited audio(s)." if duplicates else f"Audit complete{sample_note}: Database is 100% clean with no internal duplicates."

        return {
            "total_files": len(manifest_copy),
            "audited_files": len(target_files),
            "duplicate_count": dup_count,
            "edited_count": edit_count,
            "clean": len(duplicates) == 0,
            "duplicates": duplicates,
            "message": msg
        }

    def clean_database_duplicates(self, file_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Removes redundant duplicate / edited files from the database manifest and deletes their .npz files.
        If file_ids is None, runs audit_database() and removes all detected redundant files.
        """
        with self._lock:
            if not self.manifest:
                return {"status": "error", "message": "No database attached."}

        if file_ids is None:
            audit_res = self.audit_database()
            file_ids = [d["redundant_id"] for d in audit_res.get("duplicates", [])]

        if not file_ids:
            return {
                "status": "success",
                "removed_count": 0,
                "remaining_count": len(self.manifest),
                "message": "No duplicates to clean."
            }

        ids_to_remove = set(file_ids)
        removed_entries: List[Dict[str, Any]] = []

        with self._lock:
            new_manifest = []
            for e in self.manifest:
                if e["file_id"] in ids_to_remove:
                    removed_entries.append(e)
                    npz_path = self.fp_dir / e.get("fp_file", "")
                    if npz_path.exists():
                        try:
                            npz_path.unlink()
                        except Exception:
                            pass
                else:
                    new_manifest.append(e)

            self.manifest = new_manifest
            self._save_manifest()
            self._rebuild_index()

        return {
            "status": "success",
            "removed_count": len(removed_entries),
            "remaining_count": len(self.manifest),
            "removed_files": [e["filename"] for e in removed_entries],
            "message": f"Successfully removed {len(removed_entries)} duplicate/edited file(s) from database."
        }
