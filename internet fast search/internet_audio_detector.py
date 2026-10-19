#!/usr/bin/env python3
"""
Internet Audio Availability Detector
====================================
Component for verifying whether a given audio recording exists on the internet
using Chromaprint acoustic fingerprints and the AcoustID / MusicBrainz API.

Part of a 3-model audio intelligence architecture:
1. Duplicate Audio Detection (Local / Neural / HNSW)
2. AI-Generated Audio Detection (Synthetic Artifact Classifier)
3. Internet Audio Availability Detection (AcoustID / MusicBrainz Web Service)
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("InternetAudioDetector")

# Default public application key for AcoustID (from registered application)
DEFAULT_ACOUSTID_API_KEY = "1vOwZtEn"
ACOUSTID_API_URL = "https://api.acoustid.org/v2/lookup"


@dataclass
class AudioAvailabilityResult:
    """Result returned by the InternetAudioDetector."""
    file_path: str
    is_available: bool
    confidence: float
    acoustid_id: Optional[str] = None
    track_url: Optional[str] = None
    duration: float = 0.0
    fingerprint: Optional[str] = None
    fingerprint_length: int = 0
    recordings: List[Dict[str, Any]] = field(default_factory=list)
    match_count: int = 0
    status: str = "UNKNOWN"  # "AVAILABLE", "NOT_FOUND", "ERROR"
    error_message: Optional[str] = None
    query_latency_ms: float = 0.0
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary suitable for JSON/DataFrame serialization."""
        d = asdict(self)
        # Add summary string for easy display
        if self.is_available:
            rec_summary = []
            for r in self.recordings[:2]:
                title = r.get("title", "Unknown")
                artists = r.get("artists", [])
                artist_names = ", ".join([a.get("name", "") if isinstance(a, dict) else str(a) for a in artists]) if artists else ""
                rec_summary.append(f"{title} ({artist_names})" if artist_names else title)
            d["recording_summary"] = "; ".join(rec_summary) if rec_summary else f"AcoustID UUID {self.acoustid_id}"
        else:
            d["recording_summary"] = "None"
        return d

    def to_feature_vector(self) -> Dict[str, Any]:
        """Extract numeric and categorical features for combining with Model 1 & Model 2."""
        return {
            "web_is_available": 1.0 if self.is_available else 0.0,
            "web_confidence": float(self.confidence),
            "web_match_count": int(self.match_count),
            "web_has_musicbrainz": 1.0 if len(self.recordings) > 0 else 0.0,
            "web_recordings_count": len(self.recordings),
            "audio_duration_sec": float(self.duration),
        }


class InternetAudioDetector:
    """
    Detects whether an audio file is available on the internet by computing
    Chromaprint acoustic fingerprints and querying the AcoustID lookup service.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        fpcalc_path: Optional[str] = None,
        min_score_threshold: float = 0.5,
        cache_file: Optional[str] = ".acoustid_cache.sqlite",
        rate_limit_interval: float = 0.34,  # AcoustID allows max 3 req/sec
        max_retries: int = 3,
        timeout: float = 12.0,
        user_agent: str = "EchoesAudioVerifier/1.0",
    ):
        """
        Initialize the detector.

        Args:
            api_key: AcoustID client API key. Defaults to registered key or env var ACOUSTID_API_KEY.
            fpcalc_path: Path to Chromaprint fpcalc executable. If None, auto-detected.
            min_score_threshold: Minimum match score (0.0 to 1.0) to consider audio available online.
            cache_file: Path to SQLite cache file to persist API responses and avoid duplicate lookups.
            rate_limit_interval: Minimum seconds between HTTP requests (default 0.34s = ~3 req/s).
            max_retries: Maximum retries on network failures or rate limits with exponential backoff.
            timeout: Request timeout in seconds.
            user_agent: HTTP User-Agent header.
        """
        self.api_key = api_key or os.environ.get("ACOUSTID_API_KEY") or DEFAULT_ACOUSTID_API_KEY
        self.min_score_threshold = min_score_threshold
        self.rate_limit_interval = rate_limit_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.user_agent = user_agent
        self.last_request_time = 0.0

        # Locate fpcalc binary
        self.fpcalc_path = self._resolve_fpcalc(fpcalc_path)
        if self.fpcalc_path:
            os.environ["FPCALC"] = str(self.fpcalc_path)
            logger.info("Using fpcalc binary: %s", self.fpcalc_path)
        else:
            logger.warning("No fpcalc executable found. Fingerprinting will require system PATH.")

        # HTTP session for keep-alive connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        })

        # Initialize SQLite cache
        self.cache_file = cache_file
        if self.cache_file:
            self._init_cache()

    def _resolve_fpcalc(self, explicit_path: Optional[str]) -> Optional[str]:
        """Auto-detect fpcalc executable across project directories and PATH."""
        candidates = []
        if explicit_path:
            candidates.append(explicit_path)

        env_fpcalc = os.environ.get("FPCALC")
        if env_fpcalc:
            candidates.append(env_fpcalc)

        # Check current working directory and module directory
        cur_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        candidates.extend([
            str(cur_dir / "fpcalc.exe"),
            str(cur_dir / "fpcalc"),
            str(cur_dir / "tools" / "fpcalc.exe"),
            str(cur_dir.parent / "Binary-HNSW-Chromaprint-main" / "Binary-HNSW-Chromaprint-main" / "tools" / "fpcalc.exe"),
            r"C:\Codes\Projects In Download Folder\Echoes\fpcalc.exe",
            r"C:\Codes\Projects In Download Folder\Binary-HNSW-Chromaprint-main\Binary-HNSW-Chromaprint-main\tools\fpcalc.exe"
        ])

        for path_str in candidates:
            p = Path(path_str)
            if p.is_file():
                try:
                    res = subprocess.run([str(p), "-v"], capture_output=True, text=True, timeout=3)
                    if res.returncode == 0:
                        return str(p.resolve())
                except Exception:
                    pass

        # Try system PATH
        import shutil
        sys_fpcalc = shutil.which("fpcalc")
        if sys_fpcalc:
            return sys_fpcalc

        return None

    def _init_cache(self):
        """Set up SQLite caching database."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS acoustid_lookups (
                        cache_key TEXT PRIMARY KEY,
                        file_path TEXT,
                        duration REAL,
                        fingerprint TEXT,
                        response_json TEXT,
                        status TEXT,
                        created_at REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning("Could not initialize SQLite cache: %s", e)
            self.cache_file = None

    def _get_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached API response if present."""
        if not self.cache_file:
            return None
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT response_json FROM acoustid_lookups WHERE cache_key = ?", (cache_key,))
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
        except Exception as e:
            logger.debug("Cache read error: %s", e)
        return None

    def _set_cache(self, cache_key: str, file_path: str, duration: float, fingerprint: str, response: Dict[str, Any], status: str):
        """Store API response in SQLite cache."""
        if not self.cache_file:
            return
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO acoustid_lookups 
                    (cache_key, file_path, duration, fingerprint, response_json, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (cache_key, file_path, duration, fingerprint, json.dumps(response), status, time.time()))
                conn.commit()
        except Exception as e:
            logger.debug("Cache write error: %s", e)

    def compute_fingerprint(self, audio_path: str) -> Tuple[float, str]:
        """
        Compute Chromaprint fingerprint and duration for an audio file.

        Args:
            audio_path: Path to the audio file (.mp3, .wav, .flac, .au, .ogg, etc.)

        Returns:
            Tuple of (duration_in_seconds, fingerprint_base64_string)
        """
        p = Path(audio_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Method 1: Direct fpcalc execution with -json
        if self.fpcalc_path and os.path.exists(self.fpcalc_path):
            cmd = [self.fpcalc_path, "-json", str(p)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                duration = float(data.get("duration", 0.0))
                fingerprint = str(data.get("fingerprint", ""))
                return duration, fingerprint
            else:
                logger.debug("fpcalc failed with code %d: %s", proc.returncode, proc.stderr)

        # Method 2: pyacoustid wrapper fallback
        try:
            import acoustid
            duration, fp_bytes = acoustid.fingerprint_file(str(p))
            if isinstance(fp_bytes, bytes):
                fingerprint = fp_bytes.decode("ascii", errors="ignore")
            else:
                fingerprint = str(fp_bytes)
            return float(duration), fingerprint
        except Exception as e:
            raise RuntimeError(f"Failed to generate fingerprint for {audio_path}: {e}")

    def _rate_limit_wait(self):
        """Enforce AcoustID API rate limit (default 3 requests per second)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_interval:
            sleep_time = self.rate_limit_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def lookup_fingerprint(
        self,
        fingerprint: str,
        duration: float,
        file_path: str = ""
    ) -> AudioAvailabilityResult:
        """
        Query AcoustID API with a pre-computed Chromaprint fingerprint and duration.

        Args:
            fingerprint: Chromaprint base64 string
            duration: Audio duration in seconds
            file_path: Optional path for reporting and caching

        Returns:
            AudioAvailabilityResult
        """
        if not fingerprint:
            return AudioAvailabilityResult(
                file_path=file_path,
                is_available=False,
                confidence=0.0,
                duration=duration,
                fingerprint=fingerprint,
                fingerprint_length=0,
                status="ERROR",
                error_message="Empty fingerprint supplied"
            )

        # Check cache
        cache_key = hashlib.sha256(f"{int(round(duration))}:{fingerprint}".encode("utf-8")).hexdigest()
        cached_resp = self._get_cache(cache_key)
        if cached_resp is not None:
            return self._parse_api_response(
                response=cached_resp,
                file_path=file_path,
                duration=duration,
                fingerprint=fingerprint,
                latency_ms=0.0,
                cached=True
            )

        # Prepare AcoustID query payload
        payload = {
            "format": "json",
            "client": self.api_key,
            "duration": int(round(duration)),
            "fingerprint": fingerprint,
            "meta": "recordings releases tracks usermeta"
        }

        # Perform network request with exponential backoff retry logic
        last_error = None
        t_start = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            self._rate_limit_wait()
            try:
                resp = self.session.post(
                    ACOUSTID_API_URL,
                    data=payload,
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    resp_data = resp.json()
                    latency_ms = (time.perf_counter() - t_start) * 1000.0
                    parsed = self._parse_api_response(
                        response=resp_data,
                        file_path=file_path,
                        duration=duration,
                        fingerprint=fingerprint,
                        latency_ms=latency_ms,
                        cached=False
                    )
                    # Cache result
                    self._set_cache(
                        cache_key=cache_key,
                        file_path=file_path,
                        duration=duration,
                        fingerprint=fingerprint,
                        response=resp_data,
                        status=parsed.status
                    )
                    return parsed

                elif resp.status_code == 429:
                    # Rate limit exceeded - back off
                    backoff = (2 ** attempt) * 0.5
                    logger.warning("AcoustID HTTP 429 (rate limit). Backing off %.1fs...", backoff)
                    time.sleep(backoff)
                    last_error = f"HTTP 429 Rate Limit (attempt {attempt}/{self.max_retries})"

                elif resp.status_code in (500, 502, 503, 504):
                    backoff = (2 ** attempt) * 0.5
                    logger.warning("AcoustID server error %d. Retrying in %.1fs...", resp.status_code, backoff)
                    time.sleep(backoff)
                    last_error = f"HTTP {resp.status_code} Server Error"

                else:
                    error_text = resp.text[:200]
                    last_error = f"HTTP {resp.status_code}: {error_text}"
                    break

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                backoff = (2 ** attempt) * 0.5
                logger.warning("Network error '%s'. Retrying attempt %d in %.1fs...", e, attempt, backoff)
                time.sleep(backoff)
                last_error = str(e)
            except Exception as e:
                last_error = str(e)
                break

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return AudioAvailabilityResult(
            file_path=file_path,
            is_available=False,
            confidence=0.0,
            duration=duration,
            fingerprint=fingerprint,
            fingerprint_length=len(fingerprint),
            status="ERROR",
            error_message=f"AcoustID lookup failed: {last_error}",
            query_latency_ms=latency_ms
        )

    def _parse_api_response(
        self,
        response: Dict[str, Any],
        file_path: str,
        duration: float,
        fingerprint: str,
        latency_ms: float,
        cached: bool = False
    ) -> AudioAvailabilityResult:
        """Parse raw AcoustID API response into structured AudioAvailabilityResult."""
        status = response.get("status", "")
        if status != "ok":
            error_obj = response.get("error", {})
            msg = error_obj.get("message", "Unknown AcoustID error") if isinstance(error_obj, dict) else str(error_obj)
            return AudioAvailabilityResult(
                file_path=file_path,
                is_available=False,
                confidence=0.0,
                duration=duration,
                fingerprint=fingerprint,
                fingerprint_length=len(fingerprint),
                status="ERROR",
                error_message=msg,
                query_latency_ms=latency_ms,
                cached=cached
            )

        results = response.get("results", [])
        if not results:
            return AudioAvailabilityResult(
                file_path=file_path,
                is_available=False,
                confidence=0.0,
                duration=duration,
                fingerprint=fingerprint,
                fingerprint_length=len(fingerprint),
                status="NOT_FOUND",
                match_count=0,
                query_latency_ms=latency_ms,
                cached=cached
            )

        # Top match
        top_match = results[0]
        top_score = float(top_match.get("score", 0.0))
        acoustid_id = top_match.get("id")
        track_url = f"https://acoustid.org/track/{acoustid_id}" if acoustid_id else None

        # Collect MusicBrainz recordings across results
        all_recordings = []
        for match in results:
            for rec in match.get("recordings", []):
                raw_artists = rec.get("artists", [])
                clean_artists = []
                for a in raw_artists:
                    if isinstance(a, dict):
                        clean_artists.append(a)
                    else:
                        clean_artists.append({"name": str(a)})
                rec_entry = {
                    "mbid": rec.get("id"),
                    "title": rec.get("title"),
                    "artists": clean_artists,
                    "releases": [rel.get("title") if isinstance(rel, dict) else str(rel) for rel in rec.get("releases", [])],
                    "match_score": match.get("score", 0.0)
                }
                all_recordings.append(rec_entry)

        is_available = top_score >= self.min_score_threshold

        return AudioAvailabilityResult(
            file_path=file_path,
            is_available=is_available,
            confidence=top_score,
            acoustid_id=acoustid_id,
            track_url=track_url,
            duration=duration,
            fingerprint=fingerprint,
            fingerprint_length=len(fingerprint),
            recordings=all_recordings,
            match_count=len(results),
            status="AVAILABLE" if is_available else "NOT_FOUND",
            query_latency_ms=latency_ms,
            cached=cached
        )

    def check_audio(self, audio_path: str) -> AudioAvailabilityResult:
        """
        End-to-end check for a single audio file:
        Computes Chromaprint acoustic fingerprint, queries AcoustID,
        and returns whether audio exists on the internet with full metadata.

        Args:
            audio_path: Path to the target audio file

        Returns:
            AudioAvailabilityResult
        """
        t0 = time.perf_counter()
        try:
            duration, fingerprint = self.compute_fingerprint(audio_path)
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return AudioAvailabilityResult(
                file_path=audio_path,
                is_available=False,
                confidence=0.0,
                status="ERROR",
                error_message=str(e),
                query_latency_ms=total_ms
            )

        result = self.lookup_fingerprint(
            fingerprint=fingerprint,
            duration=duration,
            file_path=audio_path
        )
        return result

    def batch_check(
        self,
        audio_paths: List[str],
        show_progress: bool = True
    ) -> List[AudioAvailabilityResult]:
        """
        Check availability for a collection of audio files, respecting rate limits.

        Args:
            audio_paths: List of audio file paths
            show_progress: Whether to log progress to stdout/stderr

        Returns:
            List of AudioAvailabilityResult objects
        """
        results = []
        total = len(audio_paths)
        if total == 0:
            return results

        if show_progress:
            print(f"\n=======================================================", flush=True)
            print(f"  Internet Audio Availability Check (AcoustID API)", flush=True)
            print(f"  Target: {total} files | Min score threshold: {self.min_score_threshold}", flush=True)
            print(f"=======================================================\n", flush=True)

        for idx, file_path in enumerate(audio_paths, 1):
            res = self.check_audio(file_path)
            results.append(res)

            if show_progress:
                base_name = os.path.basename(file_path)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                disp_path = f"{parent_dir}/{base_name}"
                cache_indicator = " (cached)" if res.cached else ""

                if res.is_available:
                    rec_info = ""
                    if res.recordings:
                        r = res.recordings[0]
                        artists = ", ".join([a.get("name", "") for a in r.get("artists", [])])
                        rec_info = f" -> '{r.get('title')}' by {artists}" if artists else f" -> '{r.get('title')}'"
                    elif res.acoustid_id:
                        rec_info = f" -> Track UUID: {res.acoustid_id[:8]}..."

                    print(
                        f"[{idx:3d}/{total:3d}] [AVAILABLE]   Score: {res.confidence:.3f} | {disp_path}{rec_info}{cache_indicator}",
                        flush=True
                    )
                elif res.status == "NOT_FOUND":
                    print(
                        f"[{idx:3d}/{total:3d}] [NOT FOUND]   Score: {res.confidence:.3f} | {disp_path}{cache_indicator}",
                        flush=True
                    )
                else:
                    print(
                        f"[{idx:3d}/{total:3d}] [ERROR]       {disp_path}: {res.error_message}",
                        flush=True
                    )

        return results

    @staticmethod
    def get_summary_stats(results: List[AudioAvailabilityResult]) -> Dict[str, Any]:
        """Compute aggregate summary metrics across a batch of results."""
        total = len(results)
        if total == 0:
            return {}

        available = [r for r in results if r.is_available]
        not_found = [r for r in results if r.status == "NOT_FOUND"]
        errors = [r for r in results if r.status == "ERROR"]
        cached_count = sum(1 for r in results if r.cached)

        confidences = [r.confidence for r in available]
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

        latencies = [r.query_latency_ms for r in results if not r.cached and r.query_latency_ms > 0]
        avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

        with_recordings = sum(1 for r in available if len(r.recordings) > 0)

        return {
            "total_files": total,
            "available_count": len(available),
            "available_pct": (len(available) / total) * 100.0,
            "not_found_count": len(not_found),
            "not_found_pct": (len(not_found) / total) * 100.0,
            "error_count": len(errors),
            "error_pct": (len(errors) / total) * 100.0,
            "cached_lookups": cached_count,
            "with_musicbrainz_metadata": with_recordings,
            "avg_available_confidence": avg_conf,
            "avg_network_latency_ms": avg_lat
        }


# ==============================================================================
# Helper for 3-Model Pipeline Integration
# ==============================================================================

def extract_web_availability_features(
    audio_path: str,
    detector: Optional[InternetAudioDetector] = None
) -> Dict[str, Any]:
    """
    Standard interface function for combining with:
      - Model 1: Duplicate Detection
      - Model 2: AI-Generated Audio Detection
      - Model 3: Web Availability (this component)
    """
    if detector is None:
        detector = InternetAudioDetector()
    result = detector.check_audio(audio_path)
    features = result.to_feature_vector()
    features["file_path"] = audio_path
    features["web_status"] = result.status
    features["web_acoustid_id"] = result.acoustid_id
    features["web_track_url"] = result.track_url
    return features


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify whether audio recordings exist on the internet using Chromaprint + AcoustID."
    )
    parser.add_argument("--file", type=str, help="Single audio file to inspect.")
    parser.add_argument("--dir", type=str, help="Directory of audio files to scan.")
    parser.add_argument("--limit", type=int, default=100, help="Max files to test when scanning directory.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5).")
    parser.add_argument("--output-csv", type=str, default=None, help="Save batch results to CSV file.")
    parser.add_argument("--api-key", type=str, default=None, help="AcoustID API key (optional).")
    args = parser.parse_args()

    detector = InternetAudioDetector(
        api_key=args.api_key,
        min_score_threshold=args.threshold
    )

    if args.file:
        res = detector.check_audio(args.file)
        print("\n" + "=" * 50)
        print("AUDIO AVAILABILITY REPORT")
        print("=" * 50)
        print(f"File:         {res.file_path}")
        print(f"Status:       {res.status}")
        print(f"Available:    {res.is_available}")
        print(f"Confidence:   {res.confidence:.4f}")
        print(f"Duration:     {res.duration:.2f}s")
        if res.acoustid_id:
            print(f"AcoustID ID:  {res.acoustid_id}")
            print(f"Web Link:     {res.track_url}")
        if res.recordings:
            print(f"Recordings:   {len(res.recordings)} MusicBrainz match(es)")
            for i, rec in enumerate(res.recordings[:3], 1):
                artists = ", ".join([a.get("name", "") for a in rec.get("artists", [])])
                print(f"  {i}. {rec.get('title')} - {artists}")
        if res.error_message:
            print(f"Error:        {res.error_message}")
        print(f"Latency:      {res.query_latency_ms:.1f}ms (Cached: {res.cached})")
        print("=" * 50 + "\n")

    elif args.dir:
        audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".au", ".m4a")
        files = []
        for root, _, fnames in os.walk(args.dir):
            for fn in fnames:
                if fn.lower().endswith(audio_exts):
                    files.append(os.path.join(root, fn))
                    if len(files) >= args.limit:
                        break
            if len(files) >= args.limit:
                break

        print(f"Found {len(files)} audio files in '{args.dir}'.")
        results = detector.batch_check(files)
        stats = detector.get_summary_stats(results)

        print("\n" + "=" * 50)
        print("BATCH SUMMARY")
        print("=" * 50)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k:30s}: {v:.2f}")
            else:
                print(f"{k:30s}: {v}")
        print("=" * 50)

        if args.output_csv:
            import csv
            with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "file_path", "is_available", "confidence", "status",
                    "acoustid_id", "track_url", "duration", "match_count",
                    "recording_summary", "query_latency_ms", "cached"
                ])
                writer.writeheader()
                for r in results:
                    d = r.to_dict()
                    filtered = {k: d.get(k) for k in writer.fieldnames}
                    writer.writerow(filtered)
            print(f"Saved CSV results to: {args.output_csv}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
