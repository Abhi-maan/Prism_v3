"""Reproduction through the exact web path: Flask test client -> POST /api/check
(module=duplicate_detection, domain=auto, multipart audio_files, same form the
UI sends), against the real reference DB. Uses the user's own edited files.
Also calls app.run_duplicate_detection directly for the same files."""
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import app as prism_app  # noqa: E402

DL = Path.home() / "Downloads"
FILES = [DL / "Chaleya Jawan 320 Kbps.mp3",
         DL / "Chaleya_Jawan_pitch_shift_plus_2_semitones.mp3",
         DL / "Chaleya_Jawan_speed_plus_5_percent.mp3"]
KEYS = ("filename", "matched", "matched_filename", "match_type", "score", "confident", "stage_used", "query_time_ms")


def main():
    t0 = time.time()
    while not prism_app.sonic_db.get_status()["attached"]:
        time.sleep(0.5)
    print(f"[DB attached after {time.time()-t0:.0f}s] count={prism_app.sonic_db.get_status()['count']}")
    client = prism_app.app.test_client()

    print("\n--- Web path: one upload per request (POST /api/check, module=duplicate_detection) ---")
    for f in FILES:
        data = {"module": "duplicate_detection", "domain": "auto",
                "audio_files": (io.BytesIO(f.read_bytes()), f.name)}
        r = client.post("/api/check", data=data, content_type="multipart/form-data")
        body = r.get_json()
        if body.get("job_id"):
            while True:
                j = client.get(f"/api/check/job/{body['job_id']}").get_json()
                if j["status"] in ("completed", "error", "cancelled"):
                    body = j.get("result") or j
                    break
                time.sleep(0.5)
        for res in body.get("results", []):
            print(json.dumps({k: res.get(k) for k in KEYS}))

    print("\n--- Web path: both edited files in one request (as a 2-file UI batch) ---")
    data = {"module": "duplicate_detection", "domain": "auto",
            "audio_files": [(io.BytesIO(f.read_bytes()), f.name) for f in FILES[1:]]}
    body = client.post("/api/check", data=data, content_type="multipart/form-data").get_json()
    for res in body.get("results", []):
        print(json.dumps({k: res.get(k) for k in KEYS}))

    print("\n--- Direct call: app.run_duplicate_detection([path], 'auto') ---")
    with tempfile.TemporaryDirectory() as td:
        for f in FILES:
            p = Path(td) / f.name
            shutil.copy(f, p)
            out = prism_app.run_duplicate_detection([p], "auto")
            print(json.dumps({k: out["results"][0].get(k) for k in KEYS}))


if __name__ == "__main__":
    main()
