#!/usr/bin/env python3
"""
Benchmark Script: Testing Internet Audio Availability on 100 Diverse Audio Files
==============================================================================
Tests the Chromaprint + AcoustID lookup pipeline across 5 diverse audio categories:
  1. GTZAN Real Music Tracks (40 files across 10 genres)
  2. DESED YouTube Video Audio Clips (20 files)
  3. AI-Generated Synthetic Tracks (20 files across 6 AI generators)
  4. LibriSpeech Spoken Audio (10 files)
  5. ESC-50 Environmental Recordings (10 files)

Total: 100 files
Outputs:
  - Console report with per-file status & confidence scores
  - Per-category breakdown & summary statistics
  - internet_availability_benchmark_results.csv
  - internet_availability_benchmark_report.json
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from internet_audio_detector import InternetAudioDetector, AudioAvailabilityResult


def collect_100_test_files(base_dir: Path) -> List[Dict[str, str]]:
    """Select a deterministic, balanced set of 100 audio files from available datasets."""
    test_files = []

    # 1. GTZAN Music: 40 files (4 from each of 10 genres)
    gtzan_dir = base_dir / "real" / "archive" / "genres"
    genres = ["blues", "rock", "pop", "disco", "reggae", "classical", "country", "hiphop", "jazz", "metal"]
    if gtzan_dir.is_dir():
        for g in genres:
            g_path = gtzan_dir / g
            if g_path.is_dir():
                # Pick 4 deterministic files per genre
                files = sorted([f for f in g_path.iterdir() if f.suffix.lower() in (".au", ".wav", ".mp3")])
                for f in files[:4]:
                    test_files.append({
                        "category": f"Music (GTZAN {g.capitalize()})",
                        "expected_type": "Real Music",
                        "path": str(f)
                    })

    # 2. DESED YouTube clips: 20 files
    desed_dir = base_dir / "DESED_public_eval" / "audio" / "eval" / "public"
    if desed_dir.is_dir():
        files = sorted([f for f in desed_dir.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".flac")])
        for f in files[:20]:
            test_files.append({
                "category": "YouTube / Domestic (DESED)",
                "expected_type": "Real YouTube Audio",
                "path": str(f)
            })

    # 3. AI-Generated audio: 20 files across multiple generators
    ai_dir = base_dir / "ai_generated"
    if ai_dir.is_dir():
        ai_collected = []
        for root, _, fnames in os.walk(ai_dir):
            for fn in sorted(fnames):
                if fn.lower().endswith((".mp3", ".wav", ".flac")):
                    full_p = os.path.join(root, fn)
                    # Determine generator name from directory path
                    parts = Path(full_p).parts
                    gen = parts[-2] if len(parts) >= 2 else "ai"
                    ai_collected.append((gen, full_p))
                    if len(ai_collected) >= 60:
                        break
            if len(ai_collected) >= 60:
                break
        
        # Sample 20 across available generators
        step = max(1, len(ai_collected) // 20)
        for gen, path in ai_collected[::step][:20]:
            test_files.append({
                "category": f"AI-Generated ({gen})",
                "expected_type": "Synthetic AI Audio",
                "path": path
            })

    # 4. LibriSpeech speech: 10 files
    librispeech_dir = base_dir / "real" / "dev-clean"
    if librispeech_dir.is_dir():
        speech_files = []
        for root, _, fnames in os.walk(librispeech_dir):
            for fn in sorted(fnames):
                if fn.lower().endswith((".flac", ".wav", ".mp3")):
                    speech_files.append(os.path.join(root, fn))
                    if len(speech_files) >= 10:
                        break
            if len(speech_files) >= 10:
                break
        for f in speech_files[:10]:
            test_files.append({
                "category": "Speech (LibriSpeech)",
                "expected_type": "Real Speech Audio",
                "path": f
            })

    # 5. ESC-50 environmental: 10 files
    esc50_dir = base_dir / "ESC-50" / "ESC-50-master" / "audio"
    if esc50_dir.is_dir():
        files = sorted([f for f in esc50_dir.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".flac")])
        for f in files[:10]:
            test_files.append({
                "category": "Environmental (ESC-50)",
                "expected_type": "Real Sound Effect",
                "path": str(f)
            })

    return test_files[:100]


def run_benchmark():
    base_dir = Path(__file__).resolve().parent
    print(f"Collecting 100 test files from {base_dir}...", flush=True)
    sample_manifest = collect_100_test_files(base_dir)

    if len(sample_manifest) < 100:
        print(f"Warning: Only collected {len(sample_manifest)} files. Need 100.")
    else:
        print(f"Successfully collected exactly {len(sample_manifest)} test files across 5 domains.")

    # Initialize Detector
    detector = InternetAudioDetector(
        min_score_threshold=0.5,
        cache_file=".acoustid_cache.sqlite",
        rate_limit_interval=0.34
    )

    print("\n" + "=" * 80)
    print("  COMMENCING 100-FILE INTERNET AUDIO AVAILABILITY BENCHMARK (AcoustID API)")
    print("=" * 80)
    print(f"{'Idx':<4} {'Category':<26} {'Status':<11} {'Score':<7} {'AcoustID ID / Metadata':<32}")
    print("-" * 80)

    results_data = []
    category_stats = {}
    t_start_total = time.time()

    for idx, item in enumerate(sample_manifest, 1):
        cat = item["category"]
        expected = item["expected_type"]
        path = item["path"]
        base_fn = os.path.basename(path)

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "available": 0, "not_found": 0, "error": 0, "scores": []}

        # Run detection
        res = detector.check_audio(path)
        category_stats[cat]["total"] += 1

        if res.is_available:
            category_stats[cat]["available"] += 1
            category_stats[cat]["scores"].append(res.confidence)
            status_disp = "AVAILABLE"
        elif res.status == "NOT_FOUND":
            category_stats[cat]["not_found"] += 1
            status_disp = "NOT FOUND"
        else:
            category_stats[cat]["error"] += 1
            status_disp = "ERROR"

        # Format metadata for display
        if res.recordings:
            rec = res.recordings[0]
            artist_names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in rec.get("artists", [])]
            artists = ", ".join(filter(None, artist_names))
            meta_disp = f"'{rec.get('title')}' by {artists}" if artists else f"'{rec.get('title')}'"
        elif res.acoustid_id:
            meta_disp = f"UUID: {res.acoustid_id}"
        elif res.error_message:
            meta_disp = f"Err: {res.error_message[:28]}"
        else:
            meta_disp = f"No web match"

        # Truncate display string if needed
        if len(meta_disp) > 31:
            meta_disp = meta_disp[:28] + "..."

        score_disp = f"{res.confidence:.3f}" if res.confidence > 0 else "0.000"
        cached_flag = "*" if res.cached else " "

        print(f"[{idx:03d}] {cat:<26} {status_disp:<11} {score_disp:<6}{cached_flag} {meta_disp:<32}", flush=True)

        # Store for CSV & JSON
        row = {
            "index": idx,
            "category": cat,
            "expected_type": expected,
            "file_name": base_fn,
            "file_path": path,
            "is_available": res.is_available,
            "confidence": res.confidence,
            "status": res.status,
            "duration": round(res.duration, 2),
            "acoustid_id": res.acoustid_id or "",
            "track_url": res.track_url or "",
            "recordings_count": len(res.recordings),
            "recording_summary": res.to_dict().get("recording_summary", ""),
            "cached": res.cached,
            "latency_ms": round(res.query_latency_ms, 1),
            "error_message": res.error_message or ""
        }
        results_data.append(row)

    total_time = time.time() - t_start_total

    # Compute overall statistics
    total_files = len(results_data)
    total_available = sum(1 for r in results_data if r["is_available"])
    total_not_found = sum(1 for r in results_data if r["status"] == "NOT_FOUND")
    total_errors = sum(1 for r in results_data if r["status"] == "ERROR")
    total_cached = sum(1 for r in results_data if r["cached"])

    available_scores = [r["confidence"] for r in results_data if r["is_available"]]
    avg_conf = (sum(available_scores) / len(available_scores)) if available_scores else 0.0

    print("\n" + "=" * 80)
    print("                    100-FILE BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Files Tested:               {total_files}")
    print(f"Found on Internet (Available):   {total_available} ({total_available/total_files*100:.1f}%)")
    print(f"Not Found on Internet:            {total_not_found} ({total_not_found/total_files*100:.1f}%)")
    print(f"Errors encountered:               {total_errors} ({total_errors/total_files*100:.1f}%)")
    print(f"Cache Hits:                       {total_cached}")
    print(f"Average Confidence (when match):  {avg_conf:.4f}")
    print(f"Total Benchmark Time:             {total_time:.2f}s")
    print("-" * 80)
    print(f"{'Category Breakdown':<35} {'Total':<7} {'Available':<12} {'Rate':<8} {'Avg Score':<10}")
    print("-" * 80)

    # Group categories by broader domain for clear presentation
    domain_groups = {
        "GTZAN Real Commercial Music": [c for c in category_stats if "Music" in c],
        "DESED YouTube Video Clips": [c for c in category_stats if "YouTube" in c],
        "AI-Generated Synthetic Audio": [c for c in category_stats if "AI-Generated" in c],
        "LibriSpeech Spoken Audio": [c for c in category_stats if "Speech" in c],
        "ESC-50 Environmental Audio": [c for c in category_stats if "Environmental" in c]
    }

    for group_name, cats in domain_groups.items():
        g_total = sum(category_stats[c]["total"] for c in cats)
        g_avail = sum(category_stats[c]["available"] for c in cats)
        g_scores = [s for c in cats for s in category_stats[c]["scores"]]
        g_avg = (sum(g_scores) / len(g_scores)) if g_scores else 0.0
        g_rate = (g_avail / g_total * 100.0) if g_total > 0 else 0.0
        print(f"{group_name:<35} {g_total:<7} {g_avail:<12} {g_rate:>5.1f}%   {g_avg:>7.3f}")

    print("=" * 80)

    # Save to CSV
    csv_file = base_dir / "internet_availability_benchmark_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index", "category", "expected_type", "file_name", "is_available",
            "confidence", "status", "duration", "acoustid_id", "track_url",
            "recordings_count", "recording_summary", "cached", "latency_ms",
            "error_message", "file_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_data:
            writer.writerow(r)
    print(f"\n[+] Saved detailed CSV results to: {csv_file}")

    # Save to JSON
    json_file = base_dir / "internet_availability_benchmark_report.json"
    report_dict = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_files": total_files,
            "threshold": 0.5,
            "execution_time_seconds": round(total_time, 2)
        },
        "summary": {
            "total_files": total_files,
            "available_count": total_available,
            "available_pct": round(total_available / total_files * 100.0, 2),
            "not_found_count": total_not_found,
            "not_found_pct": round(total_not_found / total_files * 100.0, 2),
            "error_count": total_errors,
            "avg_available_confidence": round(avg_conf, 4)
        },
        "results": results_data
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[+] Saved structured JSON report to: {json_file}")


if __name__ == "__main__":
    run_benchmark()
