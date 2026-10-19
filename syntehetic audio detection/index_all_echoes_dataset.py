import sys
import os
import time
import hashlib
import sqlite3
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "syntehetic audio detection"))

from echoes_pipeline import extract_uniform_fingerprint, infer_label_from_path

DB_PATH = ROOT / "syntehetic audio detection" / "echoes_fingerprints.db"
ECHOES_DIR = Path(r"C:\Users\ASUS\OneDrive\Desktop\Echoes")
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".au", ".wma", ".aiff"}

def extract_worker(path_str):
    try:
        if os.path.getsize(path_str) < 1024:
            return None
        res = extract_uniform_fingerprint(path_str)
        if res and res.get("fingerprint") is not None:
            return {
                "file_path": path_str,
                "domain": res["domain"],
                "fingerprint": res["fingerprint"].astype(np.float32),
                "vocoder_score": float(res.get("vocoder_score", 0.0)),
                "gan_score": float(res.get("gan_score", 0.0)),
                "diffusion_score": float(res.get("diffusion_score", 0.0)),
                "anomaly_score": float(res.get("anomaly_score", 0.0)),
                "artifact_score": float(res.get("artifact_score", 0.0)),
            }
    except Exception:
        pass
    return None

def compute_file_hash(fpath: str) -> str:
    st = os.stat(fpath)
    key = f"{fpath}:{st.st_size}:{st.st_mtime}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def main():
    print("=" * 80)
    print("  FAST PARALLEL INDEXER FOR ECHOES DATASET (8 WORKERS)")
    print(f"  Target DB: {DB_PATH}")
    print(f"  Echoes Dir: {ECHOES_DIR}")
    print("=" * 80)

    # 1. Connect to DB and check existing file_paths
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT f.file_path FROM files f JOIN fingerprints fp ON f.id = fp.file_id")
    existing_paths = set(r[0].lower().replace("/", "\\") for r in c.fetchall())
    conn.close()
    print(f"Already indexed in DB: {len(existing_paths)} files")

    # 2. Discover all audio files on disk
    all_audio = []
    for root, dirs, files in os.walk(ECHOES_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in ALLOWED_EXTS:
                all_audio.append(p)
    print(f"Total audio files found on disk: {len(all_audio)}")

    # 3. Filter for missing files
    missing = []
    for p in all_audio:
        norm = str(p).lower().replace("/", "\\")
        if norm not in existing_paths:
            try:
                if p.stat().st_size >= 1024:
                    missing.append(str(p))
            except Exception:
                pass
    print(f"Files needing indexing: {len(missing)}")

    if not missing:
        print("All audio files are already fully indexed!")
        return

    # Sort so that non-LJSpeech files are processed first
    missing.sort(key=lambda x: (1 if "ljspeech" in x.lower() else 0, x))

    workers = 8
    print(f"Starting parallel extraction using {workers} CPU workers...")
    t_start = time.perf_counter()

    chunk_size = 200
    total_processed = 0
    total_saved = 0

    with Pool(processes=workers) as pool:
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i:i + chunk_size]
            t_chunk_start = time.perf_counter()
            results = pool.map(extract_worker, chunk)
            valid = [r for r in results if r is not None]

            # Ingest into SQLite
            if valid:
                conn = sqlite3.connect(str(DB_PATH))
                c = conn.cursor()
                file_rows = []
                for r in valid:
                    fpath = r["file_path"]
                    fhash = compute_file_hash(fpath)
                    st = os.stat(fpath)
                    label = infer_label_from_path(fpath)
                    file_rows.append((fpath, fhash, st.st_size, st.st_mtime, r["domain"], label))

                c.executemany("""
                    INSERT OR REPLACE INTO files (file_path, file_hash, file_size, mtime, domain, label)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, file_rows)

                paths = [r["file_path"] for r in valid]
                placeholders = ",".join(["?"] * len(paths))
                c.execute(f"SELECT file_path, id FROM files WHERE file_path IN ({placeholders})", paths)
                path_to_id = {row[0]: row[1] for row in c.fetchall()}

                fp_rows = []
                for r in valid:
                    fid = path_to_id.get(r["file_path"])
                    if fid is not None:
                        fp_blob = r["fingerprint"].tobytes()
                        fp_rows.append((
                            fid, fp_blob, r["vocoder_score"], r["gan_score"],
                            r["diffusion_score"], r["anomaly_score"], r["artifact_score"]
                        ))

                c.executemany("""
                    INSERT OR REPLACE INTO fingerprints 
                    (file_id, fingerprint_blob, vocoder_score, gan_score, diffusion_score, anomaly_score, artifact_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, fp_rows)
                conn.commit()
                conn.close()
                total_saved += len(valid)

            total_processed += len(chunk)
            chunk_time = time.perf_counter() - t_chunk_start
            rate = len(chunk) / max(chunk_time, 0.001)
            elapsed = time.perf_counter() - t_start
            remaining = len(missing) - total_processed
            eta_sec = remaining / max(rate, 0.001)
            print(f"[{total_processed}/{len(missing)}] Indexed +{len(valid)} ({rate:.1f} files/sec | Elapsed: {elapsed:.0f}s | ETA: {eta_sec:.0f}s)")

    print(f"\nDone! Ingested {total_saved} files into {DB_PATH.name} in {time.perf_counter() - t_start:.1f}s.")

if __name__ == "__main__":
    main()
