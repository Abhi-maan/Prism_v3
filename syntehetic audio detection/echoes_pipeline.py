"""
PRISM — Echoes Unified Audio Fingerprinting & AI Audio Detection Pipeline
===========================================================================
- Permanent Fingerprint Caching: Stores ONLY raw uniform fingerprints and DSP
  metadata in SQLite DB (echoes_fingerprints.db). Skip audio feature extraction
  if file is already cached in database.
- Dynamic AI Detection & Search: Re-evaluates accuracy dynamically every run
  without saving static accuracy metrics in DB.
- Extensive System Reporting: Monitors RAM usage, CPU execution time, fingerprint
  vector size, DB disk footprint, and prints top AI-generated audio search results.

Usage CLI:
    python echoes_pipeline.py --dataset-dir C:\\Users\\nishe\\Downloads\\Echoes --limit 1000
    python echoes_pipeline.py --action search --top 10
"""

from __future__ import annotations

import os
import sys
import time
import math
import hashlib
import sqlite3
import argparse
import pickle
import warnings
import tracemalloc
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ── Import PRISM Tier A modules ───────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from prism.sonic.tier_a.preprocessor import preprocess
from prism.sonic.tier_a.domain import classify_domain
from prism.sonic.tier_a.detectors import vocoder_score, gan_score, diffusion_score
from prism.sonic.tier_a.aggregator import aggregate_scores
from prism.sonic.tier_a.artifact_stats_detector import artifact_anomaly_score, compute_dsp_features

# ── Constants ─────────────────────────────────────────────────────────────────
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".au"}
DEFAULT_DB_NAME = "echoes_fingerprints.db"


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extractor (Uniform Fingerprint Converter)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_uniform_fingerprint(fpath: str, offset_sec: float | None = None) -> dict:
    """Extract a uniform numerical fingerprint from an audio file.

    Combines:
      1. PRISM Artifact Scores (Vocoder, GAN, Diffusion, Anomaly, Aggregate Tier A)
      2. Domain Probabilities (Speech, Music, Environmental)
      3. Acoustic DSP Features (MFCCs, Deltas, Spectral Centroid, Bandwidth,
         Flatness, Rolloff, HNR, ZCR statistics)

    Returns a dict containing scalar metadata and a float32 1D NumPy vector.
    """
    # Stage 1: PRISM Preprocess (load exact 4.0s calibrated window)
    p = preprocess(fpath, offset_sec=offset_sec, duration_sec=4.0)
    sr = p["sr"]
    max_samples = int(4.0 * sr)
    if len(p["audio"]) > max_samples:
        p["audio"] = p["audio"][:max_samples]
        n_frames = int(max_samples / 512)
        p["stft_mag"] = p["stft_mag"][:, :n_frames]
        p["stft_phase"] = p["stft_phase"][:, :n_frames]
        p["mel_spec"] = p["mel_spec"][:, :n_frames]
        if "stft_full" in p and p["stft_full"] is not None:
            p["stft_full"] = p["stft_full"][:, :n_frames]
    
    audio = p["audio"]
    # Standardize quantization noise floor to eliminate MP3 vs lossless container bias
    np.random.seed(42)
    dither = np.random.normal(0, 1e-5, size=len(audio)).astype(np.float32)
    audio = audio + dither
    p["audio"] = audio

    # Stage 2: Domain Classification
    domain, domain_probs = classify_domain(p)
    p_speech = float(domain_probs.get("speech", 0.0))
    p_music = float(domain_probs.get("music", 0.0))
    p_env = float(domain_probs.get("environmental", 0.0))

    # Stage 4: Base DSP primitives (computed once and reused)
    base_dsp = compute_dsp_features(p)

    # Stage 3: Artifact Detector Scores (pass precomputed base_dsp to avoid double computation)
    vs = float(vocoder_score(p, domain))
    gs = float(gan_score(p, domain=domain))
    ds = float(diffusion_score(p))
    an = float(artifact_anomaly_score(p, domain, features=base_dsp))
    tier_a_score, flags, _ = aggregate_scores(vs, gs, ds, an, domain_probs, domain_label=domain)

    # Reuse precomputed STFT magnitude from preprocess (avoids redundant STFT computation)
    mag_4s = p["stft_mag"]

    # Stage 5: Physical acoustic dynamics & Energy distribution (from precomputed mag_4s)
    dur = float(len(audio) / sr)
    rms = librosa.feature.rms(S=mag_4s)[0]
    min_rms = float(np.min(rms)) if len(rms) > 0 else 0.0
    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0
    std_rms = float(np.std(rms)) if len(rms) > 0 else 0.0
    max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    crest_factor = max_rms / (mean_rms + 1e-6)
    noise_floor_ratio = min_rms / (mean_rms + 1e-6)
    silence_frac = float(np.mean(rms < 0.01 * (max_rms + 1e-6))) if len(rms) > 0 else 0.0

    # Multi-percentile Spectral Rolloffs (fast single cumsum)
    cumsum_4s = np.cumsum(mag_4s, axis=0)
    tot_roll = cumsum_4s[-1:, :]
    freq_grid = np.linspace(0, sr / 2, mag_4s.shape[0])[:, None]
    def _fast_rolloff(pct: float) -> float:
        idx = np.argmax(cumsum_4s >= (pct * tot_roll), axis=0)
        return float(np.mean(freq_grid[idx]))

    r25 = _fast_rolloff(0.25)
    r50 = _fast_rolloff(0.50)
    r75 = _fast_rolloff(0.75)
    r85 = _fast_rolloff(0.85)
    r90 = _fast_rolloff(0.90)
    r95 = _fast_rolloff(0.95)
    r99 = _fast_rolloff(0.99)

    # Flatness, Centroid, Bandwidth, ZCR (reusing mag_4s)
    flat_arr = librosa.feature.spectral_flatness(S=mag_4s)[0]
    flat_mean = float(np.mean(flat_arr))
    flat_std = float(np.std(flat_arr))
    flat_min = float(np.min(flat_arr)) if len(flat_arr) > 0 else 0.0

    cent_arr = librosa.feature.spectral_centroid(S=mag_4s, sr=sr)[0]
    cent_mean = float(np.mean(cent_arr))
    cent_std = float(np.std(cent_arr))

    bw_arr = librosa.feature.spectral_bandwidth(S=mag_4s, sr=sr)[0]
    bw_mean = float(np.mean(bw_arr))
    bw_std = float(np.std(bw_arr))

    zcr_arr = librosa.feature.zero_crossing_rate(y=audio, hop_length=512)[0]
    zcr_mean = float(np.mean(zcr_arr))
    zcr_std = float(np.std(zcr_arr))

    # STFT sub-bands & flux
    mag = p["stft_mag"]
    freqs = np.linspace(0, sr / 2, mag.shape[0])
    tot_e = np.sum(mag**2) + 1e-9
    sub0 = float(np.sum(mag[freqs < 200]**2) / tot_e)
    sub1 = float(np.sum(mag[(freqs >= 200) & (freqs < 1000)]**2) / tot_e)
    sub2 = float(np.sum(mag[(freqs >= 1000) & (freqs < 2500)]**2) / tot_e)
    sub3 = float(np.sum(mag[(freqs >= 2500) & (freqs < 5000)]**2) / tot_e)
    sub4 = float(np.sum(mag[(freqs >= 5000) & (freqs < 8000)]**2) / tot_e)
    sub5 = float(np.sum(mag[freqs >= 8000]**2) / tot_e)
    hf_7k = float(np.sum(mag[freqs >= 7000]**2) / tot_e)
    hf_9k = float(np.sum(mag[freqs >= 9000]**2) / tot_e)

    diff = np.diff(mag, axis=1)
    flux_mean = float(np.mean(np.sqrt(np.sum(diff**2, axis=0)))) if diff.shape[1] > 0 else 0.0
    flux_std = float(np.std(np.sqrt(np.sum(diff**2, axis=0)))) if diff.shape[1] > 0 else 0.0

    # MFCCs (20), Deltas (20), Delta-Deltas (20) (fast scipy DCT from precomputed mel_spec)
    if "mel_spec" in p and p["mel_spec"] is not None:
        import scipy.fftpack
        mfccs = scipy.fftpack.dct(p["mel_spec"], axis=0, type=2, norm="ortho")[:20]
    else:
        mel_mfcc = librosa.feature.melspectrogram(S=mag_4s**2, sr=sr, n_fft=2048, hop_length=512)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_mfcc), sr=sr, n_mfcc=20)
    w = min(9, mfccs.shape[1])
    if w % 2 == 0:
        w -= 1
    if w >= 3:
        deltas = librosa.feature.delta(mfccs, width=w)
        delta2 = librosa.feature.delta(mfccs, order=2, width=w)
    else:
        deltas = np.zeros_like(mfccs)
        delta2 = np.zeros_like(mfccs)

    mfcc_m = np.mean(mfccs, axis=1)
    mfcc_s = np.std(mfccs, axis=1)
    delta_m = np.mean(deltas, axis=1)
    delta_s = np.std(deltas, axis=1)
    delta2_m = np.mean(delta2, axis=1)
    delta2_s = np.std(delta2, axis=1)

    # Spectral contrast (7) (reusing mag_4s)
    contrast = librosa.feature.spectral_contrast(S=mag_4s, sr=sr)
    cont_m = np.mean(contrast, axis=1)
    cont_s = np.std(contrast, axis=1)

    # Chroma (12) (reusing mag_4s with tuning=0.0 to bypass expensive pitch deviation estimate)
    chroma = librosa.feature.chroma_stft(S=mag_4s**2, sr=sr, tuning=0.0)
    chroma_m = np.mean(chroma, axis=1)
    chroma_s = np.std(chroma, axis=1)

    # Pairwise contrasts
    vs_gs = vs - gs
    gs_ds = gs - ds
    max_vgd = max(vs, gs, ds)
    vs_an = vs - an
    gs_an = gs - an
    ds_an = ds - an

    meta_vector = np.array([
        vs, gs, ds, an, tier_a_score,
        vs_gs, gs_ds, max_vgd, vs_an, gs_an, ds_an,
        p_speech, p_music, p_env,
        dur, min_rms, mean_rms, std_rms, max_rms, crest_factor, noise_floor_ratio, silence_frac,
        r25, r50, r75, r85, r90, r95, r99,
        flat_mean, flat_std, flat_min,
        cent_mean, cent_std, bw_mean, bw_std, zcr_mean, zcr_std,
        sub0, sub1, sub2, sub3, sub4, sub5, hf_7k, hf_9k,
        flux_mean, flux_std
    ], dtype=np.float32)

    full_fingerprint = np.concatenate([
        meta_vector, base_dsp.astype(np.float32),
        mfcc_m.astype(np.float32), mfcc_s.astype(np.float32),
        delta_m.astype(np.float32), delta_s.astype(np.float32),
        delta2_m.astype(np.float32), delta2_s.astype(np.float32),
        cont_m.astype(np.float32), cont_s.astype(np.float32),
        chroma_m.astype(np.float32), chroma_s.astype(np.float32)
    ])

    return {
        "file_path": str(fpath),
        "domain": domain,
        "duration_sec": round(dur, 2),
        "vocoder_score": round(vs, 4),
        "gan_score": round(gs, 4),
        "diffusion_score": round(ds, 4),
        "anomaly_score": round(an, 4),
        "artifact_score": round(tier_a_score, 4),
        "fingerprint": full_fingerprint,
    }


def compute_file_hash(fpath: str) -> str:
    """Fast hash based on file size, mtime, and path suffix."""
    st = os.stat(fpath)
    key = f"{fpath}:{st.st_size}:{st.st_mtime}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def infer_label_from_path(fpath: str) -> int:
    """Infer ground truth label from filepath (0=real, 1=ai/synth, -1=unknown)."""
    import re
    p = Path(fpath)
    parts = [part.lower() for part in p.parts]
    p_str = str(fpath).lower().replace("\\", "/")

    # Check synthetic paths first (e.g. WaveFake vocoders trained on LJSpeech)
    if "ai_generated" in parts or "ai_generated" in p_str:
        return 1
    if any(part in ["tta", "ata", "wavefake", "elevenlabs", "audioldm", "diffrhythm", "songgen", "producer", "acestep", "brev", "mubert", "stableaudio", "suno", "udio"] for part in parts):
        return 1
    if re.search(r"[/_\\;](tta|ata|wavefake|elevenlabs|audioldm|diffrhythm|songgen|producer|acestep|brev|mubert|stableaudio|suno|udio)[/_\\.]", p_str):
        return 1

    # Check real explicit paths
    if "desed_public_eval" in parts or "desed_public_eval" in p_str:
        return 0
    if "esc-50" in parts or "esc-50" in p_str:
        return 0
    if ("ljspeech" in parts or "ljspeech-1.1" in parts or "ljspeech" in p_str) and "wavefake" not in p_str:
        return 0
    if "dev-clean" in parts or "librispeech" in parts:
        return 0
    if ("music_data" in parts and "real" in parts) or ("real" in parts and "archive" in parts):
        return 0
    if any(genre in parts for genre in ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]) and "ai_generated" not in parts:
        return 0
    if "desed_synth_soundbank" in parts or "desed_synth" in p_str:
        # Verified from license_eval.tsv: authentic user recordings from Freesound.org
        return 0

    # Check general real keywords
    if any(k in parts for k in ["real", "human", "original", "gtzan", "esc-50", "urbansound"]):
        return 0
    if any(k in parts for k in ["fake", "synth", "synthetic", "deepfake"]):
        return 1

    return -1


# ═══════════════════════════════════════════════════════════════════════════════
#  SQLite Permanent Database Cache Engine
# ═══════════════════════════════════════════════════════════════════════════════

class FingerprintDB:
    """SQLite Database manager storing ONLY raw audio fingerprints permanently."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_NAME):
        self.db_path = Path(db_path)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create tables for file metadata and fingerprint vectors ONLY."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    domain TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    file_id INTEGER UNIQUE NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    fingerprint_blob BLOB NOT NULL,
                    vocoder_score REAL NOT NULL,
                    gan_score REAL NOT NULL,
                    diffusion_score REAL NOT NULL,
                    anomaly_score REAL NOT NULL,
                    artifact_score REAL NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def delete_file_record(self, file_path: str) -> None:
        """Delete a file and its associated fingerprint from the database."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
            conn.commit()
        finally:
            conn.close()

    def clear_all(self) -> None:
        """Clear all cached fingerprints and file metadata from database."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM fingerprints;")
            cursor.execute("DELETE FROM files;")
            conn.commit()
        finally:
            conn.close()

    def get_cached_map(self) -> Dict[str, dict]:
        """Returns map of file_path -> row dict for all stored fingerprints."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            query = """
                SELECT f.id, f.file_path, f.file_hash, f.file_size, f.mtime, f.domain, f.label,
                       fp.fingerprint_blob, fp.vocoder_score, fp.gan_score, fp.diffusion_score,
                       fp.anomaly_score, fp.artifact_score
                FROM files f
                JOIN fingerprints fp ON f.id = fp.file_id
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            cache = {}
            for r in rows:
                fp_arr = np.frombuffer(r["fingerprint_blob"], dtype=np.float32)
                cache[r["file_path"]] = {
                    "id": r["id"],
                    "file_path": r["file_path"],
                    "file_hash": r["file_hash"],
                    "file_size": r["file_size"],
                    "mtime": r["mtime"],
                    "domain": r["domain"],
                    "label": r["label"],
                    "fingerprint": fp_arr,
                    "vocoder_score": r["vocoder_score"],
                    "gan_score": r["gan_score"],
                    "diffusion_score": r["diffusion_score"],
                    "anomaly_score": r["anomaly_score"],
                    "artifact_score": r["artifact_score"],
                }
            return cache
        finally:
            conn.close()

    def insert_fingerprints_batch(self, records: List[dict]) -> None:
        """Insert a batch of extracted fingerprint records in a single transaction."""
        if not records:
            return
        
        file_rows = []
        for rec in records:
            fpath = rec["file_path"]
            fhash = compute_file_hash(fpath)
            st = os.stat(fpath)
            label = rec.get("label", infer_label_from_path(fpath))
            file_rows.append((fpath, fhash, st.st_size, st.st_mtime, rec["domain"], label))

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO files (file_path, file_hash, file_size, mtime, domain, label)
                VALUES (?, ?, ?, ?, ?, ?)
            """, file_rows)
            
            paths = [rec["file_path"] for rec in records]
            placeholders = ",".join(["?"] * len(paths))
            cursor.execute(f"SELECT file_path, id FROM files WHERE file_path IN ({placeholders})", paths)
            path_to_id = {row["file_path"]: row["id"] for row in cursor.fetchall()}

            fp_rows = []
            for rec in records:
                file_id = path_to_id[rec["file_path"]]
                fp_blob = rec["fingerprint"].astype(np.float32).tobytes()
                fp_rows.append((
                    file_id, fp_blob, rec["vocoder_score"], rec["gan_score"],
                    rec["diffusion_score"], rec["anomaly_score"], rec["artifact_score"]
                ))

            cursor.executemany("""
                INSERT OR REPLACE INTO fingerprints 
                (file_id, fingerprint_blob, vocoder_score, gan_score, diffusion_score, anomaly_score, artifact_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, fp_rows)
            conn.commit()
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Incremental Processing Engine
# ═══════════════════════════════════════════════════════════════════════════════

def find_audio_files(dataset_dir: str | Path) -> List[str]:
    """Find all supported audio files in target directory recursively.
    Interleaves real and AI-generated audio files for balanced sampling.
    """
    p = Path(dataset_dir)
    if not p.exists():
        return []
    
    real_files = []
    ai_files = []
    other_files = []

    for root, _, files in os.walk(p):
        for f in files:
            if Path(f).suffix.lower() in AUDIO_EXTENSIONS and not f.startswith("._"):
                full_path = os.path.join(root, f)
                lbl = infer_label_from_path(full_path)
                if lbl == 0:
                    real_files.append(full_path)
                elif lbl == 1:
                    ai_files.append(full_path)
                else:
                    other_files.append(full_path)

    real_files.sort()
    ai_files.sort()
    other_files.sort()

    interleaved = []
    max_len = max(len(real_files), len(ai_files))
    for i in range(max_len):
        if i < len(real_files):
            interleaved.append(real_files[i])
        if i < len(ai_files):
            interleaved.append(ai_files[i])

    interleaved.extend(other_files)
    return interleaved if interleaved else sorted(real_files + ai_files + other_files)


def load_labels_from_csv(labels_csv: str | Path) -> Dict[str, int]:
    """Load ground truth labels mapping filename/filepath -> int (0=real, 1=ai)."""
    p = Path(labels_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    label_map = {}
    col_path = "file_path" if "file_path" in df.columns else ("filename" if "filename" in df.columns else df.columns[0])
    col_label = "label" if "label" in df.columns else df.columns[1]

    for _, row in df.iterrows():
        val_path = str(row[col_path])
        val_lbl = int(row[col_label])
        label_map[val_path] = val_lbl
        label_map[Path(val_path).name] = val_lbl
    return label_map


def _process_file_worker(fpath: str) -> Optional[dict]:
    """Worker function to process single audio file safely."""
    try:
        return extract_uniform_fingerprint(fpath)
    except Exception as exc:
        print(f"  [SKIP/ERROR] {Path(fpath).name}: {exc}", file=sys.stderr, flush=True)
        return None


def run_incremental_pipeline(
    dataset_dir: str | Path,
    limit: Optional[int] = None,
    db_path: str | Path = DEFAULT_DB_NAME,
    num_workers: int = 4,
    clear_cache: bool = False,
    force_reextract: bool = False,
    labels_csv: Optional[str | Path] = None,
) -> Tuple[List[dict], float, float]:
    """Run incremental dataset fingerprinting with hash verification and cache safety."""
    db = FingerprintDB(db_path)
    
    if clear_cache:
        print(f"[ECHOES] Clearing all database cache entries in {db.db_path.name}...", flush=True)
        db.clear_all()

    all_files = find_audio_files(dataset_dir)
    if not all_files:
        print(f"[ECHOES] No audio files found in: {dataset_dir}", flush=True)
        return list(db.get_cached_map().values()), 0.0, 0.0

    if limit is not None and limit > 0:
        all_files = all_files[:limit]

    print(f"\n[ECHOES] Targeted dataset files count: {len(all_files)}", flush=True)
    
    t_db_0 = time.perf_counter()
    cached_map = db.get_cached_map()
    t_db_load = time.perf_counter() - t_db_0

    # Load custom labels CSV if provided
    csv_labels = load_labels_from_csv(labels_csv) if labels_csv else {}

    # File hash verification & stale cache invalidation
    new_files = []
    stale_count = 0
    valid_cached_count = 0

    for fpath in all_files:
        curr_hash = compute_file_hash(fpath)
        if fpath in cached_map:
            cached_rec = cached_map[fpath]
            # Check if file has been modified or force_reextract is requested
            if force_reextract or cached_rec.get("file_hash") != curr_hash:
                stale_count += 1
                db.delete_file_record(fpath)
                new_files.append(fpath)
            else:
                valid_cached_count += 1
        else:
            new_files.append(fpath)

    if stale_count > 0:
        print(f"[DATASET CHANGED DETECTED] {stale_count} cached audio files were modified or updated on disk.", flush=True)
        print(f"[ECHOES] Invalidated {stale_count} stale cache records. Re-extracting fresh fingerprints...", flush=True)

    print(f"[ECHOES] Database Cache Status: {valid_cached_count} clean files loaded from SQLite DB ({db.db_path.name}).", flush=True)
    print(f"[ECHOES] New/Modified files to convert: {len(new_files)} files.", flush=True)

    t_extract_time = 0.0
    if new_files:
        print(f"[ECHOES] Processing {len(new_files)} audio files into uniform 106-dim fingerprints...", flush=True)
        t0 = time.perf_counter()
        processed_count = 0
        batch_buffer = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, os.cpu_count() or 4)) as executor:
            futures = {executor.submit(_process_file_worker, f): f for f in new_files}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                processed_count += 1
                if res is not None:
                    batch_buffer.append(res)
                
                if len(batch_buffer) >= 20 or processed_count == len(new_files):
                    if batch_buffer:
                        db.insert_fingerprints_batch(batch_buffer)
                        batch_buffer.clear()
                    print(f"  Converted & stored [{processed_count:>4}/{len(new_files)}] audio files...", flush=True)
                    
        t_extract_time = time.perf_counter() - t0
        print(f"[ECHOES] Converted & saved {len(new_files)} fingerprints in {t_extract_time:.2f}s!", flush=True)

    # Reload all requested records from updated DB cache
    updated_map = db.get_cached_map()
    requested_records = []
    for f in all_files:
        if f in updated_map:
            rec = dict(updated_map[f])
            # Override label if in csv_labels
            f_name = Path(f).name
            if f in csv_labels:
                rec["label"] = csv_labels[f]
            elif f_name in csv_labels:
                rec["label"] = csv_labels[f_name]
            requested_records.append(rec)

    print(f"[ECHOES] Ready! Loaded {len(requested_records)} uniform fingerprints from database.", flush=True)
    return requested_records, t_extract_time, t_db_load


# ═══════════════════════════════════════════════════════════════════════════════
#  AI Detection Classifier & Dynamic Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class AIAudioClassifier:
    """Train, evaluate, save, and predict using ensemble classifier on loaded fingerprints."""

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed
        self.scaler = StandardScaler()
        self.model = None

    def save_model(self, model_path: str | Path) -> None:
        """Save fitted scaler and classifier model bundle to disk."""
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scaler": self.scaler,
            "model": self.model,
            "seed": self.seed,
            "feature_dim": 240,
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        print(f"[ECHOES] Trained AI detection model saved successfully to: {p.resolve()}", flush=True)

    def load_model(self, model_path: str | Path) -> bool:
        """Load trained scaler and classifier model bundle from disk."""
        p = Path(model_path)
        if not p.exists():
            print(f"[ERROR] Pre-trained model bundle not found: {p}", flush=True)
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.seed = data.get("seed", 42)
        print(f"[ECHOES] Successfully loaded pre-trained model bundle from: {p.resolve()}", flush=True)
        return True

    def train_and_eval(self, records: List[dict], save_path: Optional[str | Path] = None) -> dict:
        """Dynamically evaluate detection performance on requested records."""
        valid_recs = [r for r in records if r["label"] in (0, 1)]
        if len(valid_recs) < 10:
            print("[ECHOES] Not enough labeled samples to train/evaluate classifier (need at least 10 real/ai files).", flush=True)
            return {}

        X = np.array([r["fingerprint"] for r in valid_recs])
        y = np.array([r["label"] for r in valid_recs])

        real_cnt = int(np.sum(y == 0))
        ai_cnt = int(np.sum(y == 1))

        if len(np.unique(y)) < 2:
            only_class = int(y[0])
            class_name = "AI-Generated" if only_class == 1 else "Real"
            print(f"[ECHOES Classifier] All {len(y)} files belong to single class ({class_name}).", flush=True)
            return {"accuracy": 1.0, "predictions": y, "probs": y.astype(float)}

        X_scaled = self.scaler.fit_transform(X)

        estimators = [
            ("et", ExtraTreesClassifier(n_estimators=350, max_depth=18, min_samples_split=2, n_jobs=-1, random_state=self.seed)),
            ("rf", RandomForestClassifier(n_estimators=350, max_depth=18, min_samples_split=2, n_jobs=-1, random_state=self.seed)),
            ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, random_state=self.seed)),
        ]
        if HAS_XGB:
            estimators.insert(0, ("xgb", XGBClassifier(
                n_estimators=350, max_depth=5, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
                n_jobs=-1, random_state=self.seed, verbosity=0
            )))
            model_name = "PRISM-Echoes Master Ensemble (XGBoost + ExtraTrees + RF + HistGB)"
        else:
            estimators.append(("svc", SVC(kernel="rbf", C=10.0, probability=True, random_state=self.seed)))
            model_name = "PRISM-Echoes Master Ensemble (ExtraTrees + RF + HistGB + SVC)"

        clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

        n_splits = min(5, min(real_cnt, ai_cnt))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        t_ml_0 = time.perf_counter()
        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv, n_jobs=-1)
        clf.fit(X_scaled, y)
        self.model = clf
        t_ml_time = time.perf_counter() - t_ml_0

        y_prob = clf.predict_proba(X_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Category-wise / Per-Domain Metrics Calculation
        domains = np.array([r.get("domain", "unknown") for r in valid_recs])
        domain_metrics = {}
        for dom in np.unique(domains):
            mask = (domains == dom)
            if np.sum(mask) == 0:
                continue
            y_dom_true = y[mask]
            y_dom_pred = y_pred[mask]
            dom_acc = accuracy_score(y_dom_true, y_dom_pred) if len(y_dom_true) > 0 else 0.0
            dom_prec = precision_score(y_dom_true, y_dom_pred, zero_division=0)
            dom_rec = recall_score(y_dom_true, y_dom_pred, zero_division=0)
            dom_f1 = f1_score(y_dom_true, y_dom_pred, zero_division=0)
            domain_metrics[dom] = {
                "count": int(np.sum(mask)),
                "real_count": int(np.sum(y_dom_true == 0)),
                "ai_count": int(np.sum(y_dom_true == 1)),
                "accuracy": dom_acc,
                "precision": dom_prec,
                "recall": dom_rec,
                "f1": dom_f1,
            }

        # Ground-Truth Dataset Source Metrics
        dataset_sources = np.array([r.get("dataset_source", "default_dataset") for r in valid_recs])
        dataset_metrics = {}
        for ds in np.unique(dataset_sources):
            mask = (dataset_sources == ds)
            if np.sum(mask) == 0:
                continue
            ds_true = y[mask]
            ds_pred = y_pred[mask]
            dataset_metrics[ds] = {
                "count": int(np.sum(mask)),
                "real_count": int(np.sum(ds_true == 0)),
                "ai_count": int(np.sum(ds_true == 1)),
                "accuracy": accuracy_score(ds_true, ds_pred) if len(ds_true) > 0 else 0.0,
            }

        for r, prob, pred in zip(valid_recs, y_prob, y_pred):
            r["ai_probability"] = float(prob)
            r["prediction"] = "synthetic" if pred == 1 else "real"

        if save_path:
            self.save_model(save_path)

        return {
            "model_name": model_name,
            "total_samples": len(y),
            "real_samples": real_cnt,
            "ai_samples": ai_cnt,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "domain_metrics": domain_metrics,
            "dataset_metrics": dataset_metrics,
            "ml_time_sec": t_ml_time,
        }

    def predict_records(self, records: List[dict]) -> dict:
        """Run direct inference using pre-trained model on records without retraining."""
        if self.model is None:
            raise RuntimeError("Model is not loaded or trained. Call load_model() or train_and_eval() first.")

        if not records:
            return {}

        X = np.array([r["fingerprint"] for r in records])
        
        t_0 = time.perf_counter()
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, "predict_proba") else y_pred.astype(float)
        t_infer = time.perf_counter() - t_0

        for r, prob, pred in zip(records, y_prob, y_pred):
            r["ai_probability"] = float(prob)
            r["prediction"] = "synthetic" if pred == 1 else "real"

        valid_recs = [r for r in records if r.get("label") in (0, 1)]
        eval_res = {"inference_time_sec": t_infer, "total_samples": len(records)}

        if valid_recs:
            y_true = np.array([r["label"] for r in valid_recs])
            y_pred_valid = np.array([1 if r["prediction"] == "synthetic" else 0 for r in valid_recs])
            
            acc = accuracy_score(y_true, y_pred_valid)
            prec = precision_score(y_true, y_pred_valid, zero_division=0)
            rec = recall_score(y_true, y_pred_valid, zero_division=0)
            f1 = f1_score(y_true, y_pred_valid, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_valid, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # Domain breakdown
            domains = np.array([r.get("domain", "unknown") for r in valid_recs])
            domain_metrics = {}
            for dom in np.unique(domains):
                mask = (domains == dom)
                if np.sum(mask) == 0:
                    continue
                dom_true = y_true[mask]
                dom_pred = y_pred_valid[mask]
                domain_metrics[dom] = {
                    "count": int(np.sum(mask)),
                    "real_count": int(np.sum(dom_true == 0)),
                    "ai_count": int(np.sum(dom_true == 1)),
                    "accuracy": accuracy_score(dom_true, dom_pred) if len(dom_true) > 0 else 0.0,
                    "precision": precision_score(dom_true, dom_pred, zero_division=0),
                    "recall": recall_score(dom_true, dom_pred, zero_division=0),
                    "f1": f1_score(dom_true, dom_pred, zero_division=0),
                }

            eval_res.update({
                "model_name": type(self.model).__name__,
                "labeled_samples": len(y_true),
                "real_samples": int(np.sum(y_true == 0)),
                "ai_samples": int(np.sum(y_true == 1)),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fpr": fpr,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "domain_metrics": domain_metrics,
            })

        return eval_res


# ═══════════════════════════════════════════════════════════════════════════════
#  Extensive Resource & Benchmark Profiler
# ═══════════════════════════════════════════════════════════════════════════════

def print_extensive_report(
    records: List[dict],
    eval_res: dict,
    extract_time_sec: float,
    db_load_sec: float,
    total_pipeline_sec: float,
    peak_ram_mb: float,
    db_path: str | Path = DEFAULT_DB_NAME,
    top_k_search: int = 10,
) -> None:
    """Print an extensive benchmark report detailing CPU, RAM, Fingerprints, and AI Detection."""
    db_file = Path(db_path)
    db_size_mb = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0.0

    sample_fp = records[0]["fingerprint"] if records else np.zeros(106, dtype=np.float32)
    fp_dim = len(sample_fp)
    fp_bytes_per_sample = sample_fp.nbytes
    total_fp_raw_mb = (fp_bytes_per_sample * len(records)) / (1024 * 1024) if records else 0.0

    # CPU & RAM Metrics
    if HAS_PSUTIL:
        proc = psutil.Process()
        curr_ram_mb = proc.memory_info().rss / (1024 * 1024)
        cpu_usage_pct = psutil.cpu_percent(interval=0.1)
        system_ram = psutil.virtual_memory()
        total_ram_gb = system_ram.total / (1024**3)
        available_ram_gb = system_ram.available / (1024**3)
    else:
        curr_ram_mb = 0.0
        cpu_usage_pct = 0.0
        total_ram_gb = 0.0
        available_ram_gb = 0.0

    print("\n" + "=" * 80, flush=True)
    print("  EXTENSIVE SYSTEM BENCHMARK & AUDIO FINGERPRINT REPORT", flush=True)
    print("=" * 80, flush=True)

    # 1. Dataset & Storage Profile
    print(f"\n[1] DATASET & FINGERPRINT DATABASE PROFILE", flush=True)
    print("-" * 60, flush=True)
    print(f"  Target Audio Files Evaluated  : {len(records)} files", flush=True)
    print(f"  Database Path                 : {db_file.resolve()}", flush=True)
    print(f"  SQLite Database File Size     : {db_size_mb:.2f} MB", flush=True)
    print(f"  Fingerprint Vector Dimension  : {fp_dim} dimensions (float32)", flush=True)
    print(f"  Memory Per Fingerprint        : {fp_bytes_per_sample} Bytes ({fp_bytes_per_sample/1024:.2f} KB)", flush=True)
    print(f"  Total Raw Fingerprint Memory  : {total_fp_raw_mb:.2f} MB", flush=True)

    # 2. Hardware & Resource Usage
    print(f"\n[2] SYSTEM RESOURCE USAGE (CPU & MEMORY)", flush=True)
    print("-" * 60, flush=True)
    print(f"  Peak Process RAM Memory Usage : {peak_ram_mb:.2f} MB", flush=True)
    if HAS_PSUTIL:
        print(f"  Current Process Memory (RSS)  : {curr_ram_mb:.2f} MB", flush=True)
        print(f"  Current CPU Usage             : {cpu_usage_pct:.1f}%", flush=True)
        print(f"  Total System RAM              : {total_ram_gb:.2f} GB (Available: {available_ram_gb:.2f} GB)", flush=True)

    # 3. Execution Timings & Performance
    print(f"\n[3] EXECUTION TIMINGS & PERFORMANCE BENCHMARK", flush=True)
    print("-" * 60, flush=True)
    print(f"  SQLite DB Query / Load Time   : {db_load_sec*1000:.2f} ms", flush=True)
    print(f"  Feature Extraction Time (New) : {extract_time_sec:.2f} s", flush=True)
    if len(records) > 0 and extract_time_sec > 0:
        avg_extract_ms = (extract_time_sec * 1000) / len(records)
        print(f"  Feature Extraction Per File   : {avg_extract_ms:.2f} ms / file", flush=True)
    if "ml_time_sec" in eval_res:
        print(f"  ML Classification Eval Time   : {eval_res['ml_time_sec']*1000:.2f} ms", flush=True)
    if "inference_time_sec" in eval_res:
        print(f"  ML Model Inference Time       : {eval_res['inference_time_sec']*1000:.2f} ms", flush=True)
    print(f"  Total Pipeline Execution Time : {total_pipeline_sec:.2f} s", flush=True)
    if len(records) > 0:
        avg_ms_per_audio = (total_pipeline_sec * 1000) / len(records)
        print(f"  Average Total Time Per Audio  : {avg_ms_per_audio:.2f} ms / file", flush=True)

    # 4. AI Detection Model Performance
    if eval_res and "accuracy" in eval_res:
        print(f"\n[4] DYNAMIC AI AUDIO DETECTION METRICS ({eval_res.get('model_name', 'Ensemble')})", flush=True)
        print("-" * 75, flush=True)
        samples_cnt = eval_res.get('total_samples', eval_res.get('labeled_samples', len(records)))
        print(f"  Total Labeled Audio Samples   : {samples_cnt} (Real: {eval_res.get('real_samples', 0)}, AI: {eval_res.get('ai_samples', 0)})", flush=True)
        print(f"  Overall Detection Accuracy    : {eval_res['accuracy']*100:.2f}%", flush=True)
        print(f"  Precision (AI Detection)      : {eval_res['precision']*100:.2f}%", flush=True)
        print(f"  Recall / Sensitivity          : {eval_res['recall']*100:.2f}%", flush=True)
        print(f"  F1 Score                      : {eval_res['f1']:.4f}", flush=True)
        print(f"  False Positive Rate (FPR)     : {eval_res['fpr']*100:.2f}%", flush=True)
        print(f"  Confusion Matrix              : TP={eval_res['tp']}  FP={eval_res['fp']}  FN={eval_res['fn']}  TN={eval_res['tn']}", flush=True)

        # 4b. Category-wise / Per-Domain Accuracy Breakdown
        if "domain_metrics" in eval_res and eval_res["domain_metrics"]:
            print(f"\n[4b] ACCURACY BY CATEGORY / AUDIO DOMAIN", flush=True)
            print("-" * 75, flush=True)
            print(f"  {'Category / Domain':<20} {'Samples (Real/AI)':<22} {'Accuracy':<12} {'Precision':<12} {'Recall':<10} {'F1-Score'}", flush=True)
            print("-" * 75, flush=True)
            for dom, m in eval_res["domain_metrics"].items():
                counts_str = f"n={m['count']} ({m['real_count']} R / {m['ai_count']} AI)"
                print(f"  {dom:<20} {counts_str:<22} {m['accuracy']*100:>6.2f}%     {m['precision']*100:>6.2f}%     {m['recall']*100:>5.2f}%     {m['f1']:.4f}", flush=True)
            print("-" * 75, flush=True)

        # 4c. Accuracy by Dataset Source
        if "dataset_metrics" in eval_res and eval_res["dataset_metrics"]:
            print(f"\n[4c] ACCURACY BY GROUND-TRUTH DATASET SOURCE", flush=True)
            print("-" * 70, flush=True)
            print(f"  {'Dataset Source':<45} {'Samples (Real/AI)':<20} {'Accuracy'}", flush=True)
            print("-" * 70, flush=True)
            for ds, m in eval_res["dataset_metrics"].items():
                counts_str = f"n={m['count']} ({m['real_count']} R / {m['ai_count']} AI)"
                print(f"  {ds:<45} {counts_str:<20} {m['accuracy']*100:>6.2f}%", flush=True)
            print("-" * 70, flush=True)

    # 5. Top AI Generated Audio Clips
    if records:
        print(f"\n[5] TOP {top_k_search} MOST CONFIDENT AI-GENERATED AUDIO CLIPS IN DATASET", flush=True)
        print("-" * 80, flush=True)
        print(f"{'Filename':<35} {'Domain':<12} {'AI Prob':<10} {'Artifact Score':<15} {'Prediction'}", flush=True)
        print("-" * 80, flush=True)
        sorted_recs = sorted(records, key=lambda r: (r.get("ai_probability", 0.0), r.get("artifact_score", 0.0)), reverse=True)[:top_k_search]
        for r in sorted_recs:
            fname = Path(r["file_path"]).name
            fname_clean = fname.encode("ascii", errors="ignore").decode("ascii")
            prob = r.get("ai_probability", 0.0)
            pred = r.get("prediction", "synthetic" if prob >= 0.5 else "real")
            art_sc = r.get("artifact_score", 0.0)
            print(f"{fname_clean[:34]:<35} {r.get('domain', 'unknown'):<12} {prob:<10.4f} {art_sc:<15.4f} {pred}", flush=True)
        print("-" * 80, flush=True)

    print("=" * 80 + "\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tracemalloc.start()
    t_pipeline_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="PRISM Echoes — Audio Fingerprinting & AI Detection Pipeline")
    parser.add_argument("--dataset-dir", type=str, default=r"data",
                        help="Path to audio dataset directory.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_NAME,
                        help="Path to persistent SQLite database cache file.")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Limit number of dataset files to process.")
    parser.add_argument("--action", type=str, choices=["process", "search", "classify"], default="process",
                        help="Action to perform.")
    parser.add_argument("--top", type=int, default=10,
                        help="Top K files for search.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear all database cache entries before running.")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Force re-extraction of fingerprints for all target files.")
    parser.add_argument("--save-model", type=str, default=r"models/prism_echoes_ensemble.pkl",
                        help="Path to save the trained model bundle.")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load a pre-trained model bundle for direct inference.")
    parser.add_argument("--labels-csv", type=str, default=None,
                        help="Optional CSV file mapping filenames/file_paths to labels (0=real, 1=ai).")

    args = parser.parse_args()

    records, extract_time, db_load_time = run_incremental_pipeline(
        dataset_dir=args.dataset_dir,
        limit=args.limit,
        db_path=args.db_path,
        clear_cache=args.clear_cache,
        force_reextract=args.force_reextract,
        labels_csv=args.labels_csv,
    )

    clf = AIAudioClassifier()
    eval_res = {}

    if args.load_model and Path(args.load_model).exists():
        if clf.load_model(args.load_model):
            eval_res = clf.predict_records(records)
    elif records:
        eval_res = clf.train_and_eval(records, save_path=args.save_model)

    t_pipeline_end = time.perf_counter()
    total_pipeline_sec = t_pipeline_end - t_pipeline_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / (1024 * 1024)

    print_extensive_report(
        records=records,
        eval_res=eval_res,
        extract_time_sec=extract_time,
        db_load_sec=db_load_time,
        total_pipeline_sec=total_pipeline_sec,
        peak_ram_mb=peak_ram_mb,
        db_path=args.db_path,
        top_k_search=args.top,
    )


if __name__ == "__main__":
    main()

