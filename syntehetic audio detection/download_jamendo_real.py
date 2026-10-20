import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "jamendo_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fetch track list from HF valid.tsv
url = "https://huggingface.co/datasets/rkstgr/mtg-jamendo/resolve/main/valid.tsv"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=15) as r:
    lines = r.read().decode("utf-8", errors="ignore").splitlines()

tracks = []
for line in lines[1:]:
    parts = line.split("\t")
    if len(parts) >= 8:
        tid = parts[0]
        dur = float(parts[3]) if parts[3] else 0.0
        if 45.0 <= dur <= 240.0:
            tracks.append(tid)

print(f"Candidate tracks: {len(tracks)}")
# We need 250 diverse tracks
target_count = 250

def download_track(tid):
    dest = OUT_DIR / f"jamendo_{tid}.mp3"
    if dest.exists() and dest.stat().st_size > 100000:
        return True, tid, 0.0
    
    t0 = time.time()
    url = f"https://prod-1.storage.jamendo.com/?trackid={tid}&format=mp31&from=app-devsite"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        # Validate audio integrity
        info = sf.info(str(dest))
        if info.duration < 10.0 or info.samplerate < 22050:
            dest.unlink(missing_ok=True)
            return False, tid, time.time() - t0
        return True, tid, time.time() - t0
    except Exception as e:
        dest.unlink(missing_ok=True)
        return False, tid, time.time() - t0

completed = 0
failed = 0
t_start = time.time()

with ThreadPoolExecutor(max_workers=12) as executor:
    futures = {executor.submit(download_track, tid): tid for tid in tracks[:400]}
    for fut in as_completed(futures):
        success, tid, elapsed = fut.result()
        if success:
            completed += 1
            if completed % 25 == 0:
                print(f"Downloaded {completed}/{target_count} tracks ({time.time()-t_start:.1f}s)...", flush=True)
        else:
            failed += 1
        if completed >= target_count:
            break

print(f"Finished! Successfully downloaded {completed} modern CC tracks to {OUT_DIR} (failed={failed}, total_time={time.time()-t_start:.1f}s)")
