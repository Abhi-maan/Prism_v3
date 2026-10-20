"""
Samsung PRISM — Universal Link & Cloud Dataset Ingestion Engine
===============================================================
Enables recursive download and verification from arbitrary cloud URLs:
  - Google Drive folders & files (via gdown & direct API)
  - Direct HTTP/HTTPS audio links (.wav, .mp3, .flac, .ogg, .m4a, .aac, .au, .wma, .aiff)
  - Direct ZIP / TAR / GZ / 7Z archives (auto-extracted recursively)
  - Dropbox & OneDrive sharing links (auto-converted to direct stream)
  - Multi-URL batch lists
"""

from __future__ import annotations

import os
import re
import sys
import time
import shutil
import zipfile
import tarfile
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".au", ".wma", ".aiff"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".tbz", ".tar.gz", ".tar.bz2", ".7z"}


class UniversalUrlDownloader:
    """Universal downloader and extractor for cloud drive and dataset URLs."""

    def __init__(self, temp_base_dir: Optional[Path] = None):
        self.temp_base_dir = temp_base_dir or (Path(__file__).parent / "uploads" / "cloud_downloads")
        self.temp_base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_url(raw_url: str) -> str:
        """Transform sharing URLs into direct download links where possible."""
        url = raw_url.strip()

        # Dropbox: convert dl=0 to dl=1
        if "dropbox.com" in url:
            if "dl=0" in url:
                url = url.replace("dl=0", "dl=1")
            elif "?" not in url:
                url = f"{url}?dl=1"
            elif "dl=1" not in url:
                url = f"{url}&dl=1"

        # OneDrive: convert sharing view to direct download
        if "1drv.ms" in url or "onedrive.live.com" in url:
            if "download=1" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}download=1"

        return url

    @staticmethod
    def is_google_drive(url: str) -> bool:
        return "drive.google.com" in url or "docs.google.com" in url

    @staticmethod
    def is_google_drive_folder(url: str) -> bool:
        return "drive.google.com" in url and ("folders" in url or "embeddedfolderview" in url)

    def extract_archive(self, archive_path: Path, extract_to: Path) -> List[Path]:
        """Extract zip or tar archive safely against path traversal and return all discovered audio paths."""
        extracted_audio: List[Path] = []
        name_lower = archive_path.name.lower()
        target_resolved = extract_to.resolve()

        def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
            for member in zf.infolist():
                member_target = (dest / member.filename).resolve()
                if not member_target.is_relative_to(target_resolved):
                    continue  # Skip dangerous path traversal member
                zf.extract(member, dest)

        def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
            for member in tf.getmembers():
                member_target = (dest / member.name).resolve()
                if not member_target.is_relative_to(target_resolved):
                    continue  # Skip dangerous path traversal member
                tf.extract(member, dest)

        try:
            if name_lower.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    _safe_extract_zip(zf, extract_to)
            elif any(name_lower.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz", ".tar.bz2"]):
                with tarfile.open(archive_path, "r:*") as tf:
                    _safe_extract_tar(tf, extract_to)
            else:
                # Try zip first, then tar
                try:
                    with zipfile.ZipFile(archive_path, "r") as zf:
                        _safe_extract_zip(zf, extract_to)
                except Exception:
                    try:
                        with tarfile.open(archive_path, "r:*") as tf:
                            _safe_extract_tar(tf, extract_to)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Universal Downloader] Archive extract warning for {archive_path.name}: {e}")

        # Scan for extracted audio files
        for p in extract_to.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                extracted_audio.append(p)

        # Clean up the raw archive file after extraction to conserve disk space
        try:
            if archive_path.exists():
                archive_path.unlink()
        except Exception:
            pass

        return extracted_audio

    def download_direct_file(
        self,
        url: str,
        dest_dir: Path,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Path]:
        """Download file from direct HTTP/HTTPS URL with streaming and chunking."""
        norm_url = self.normalize_url(url)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SamsungPRISM/1.0"}

        # Attempt HEAD request to extract filename and size
        filename = "downloaded_audio"
        try:
            head_res = requests.head(norm_url, headers=headers, allow_redirects=True, timeout=10)
            cd = head_res.headers.get("content-disposition", "")
            if "filename=" in cd:
                m = re.search(r'filename=["\']?([^"\';]+)["\']?', cd)
                if m:
                    filename = m.group(1).strip()
            if filename == "downloaded_audio":
                path_part = urllib.parse.urlparse(head_res.url).path
                base = Path(path_part).name
                if base and "." in base:
                    filename = base
        except Exception:
            pass

        if "." not in filename:
            # Fallback extension from URL path
            url_path = urllib.parse.urlparse(norm_url).path
            url_name = Path(url_path).name
            if "." in url_name:
                filename = url_name
            else:
                filename = f"{filename}.wav"

        out_path = dest_dir / filename
        counter = 1
        while out_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            out_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        # Stream download
        with requests.get(norm_url, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb and total_size > 0:
                            pct = min(99.0, round((downloaded / total_size) * 100, 1))
                            progress_cb({
                                "status": "downloading",
                                "current_file": filename,
                                "downloaded_bytes": downloaded,
                                "total_bytes": total_size,
                                "percent": pct,
                            })

        # Check if downloaded file is an archive
        suffix_lower = out_path.suffix.lower()
        if suffix_lower in ARCHIVE_EXTS or out_path.name.lower().endswith((".tar.gz", ".tar.bz2")):
            if progress_cb:
                progress_cb({
                    "status": "extracting",
                    "current_file": f"Extracting {filename}...",
                    "percent": 99.0,
                })
            return self.extract_archive(out_path, dest_dir)
        elif suffix_lower in SUPPORTED_AUDIO_EXTS:
            return [out_path]
        else:
            # Could be a zip with arbitrary extension
            try:
                if zipfile.is_zipfile(out_path):
                    return self.extract_archive(out_path, dest_dir)
            except Exception:
                pass
            return [out_path] if suffix_lower in SUPPORTED_AUDIO_EXTS else []

    def download_google_drive(
        self,
        url: str,
        dest_dir: Path,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Path]:
        """Download from Google Drive using gdown library or fallback."""
        try:
            import gdown

            if self.is_google_drive_folder(url):
                if progress_cb:
                    progress_cb({
                        "status": "downloading",
                        "current_file": "Connecting to Google Drive folder...",
                        "percent": 10.0,
                    })
                # Download entire folder recursively
                downloaded_files = gdown.download_folder(
                    url=url,
                    output=str(dest_dir),
                    quiet=True,
                    use_cookies=False,
                )
                if progress_cb:
                    progress_cb({
                        "status": "extracting",
                        "current_file": "Scanning downloaded Google Drive files...",
                        "percent": 90.0,
                    })
            else:
                # Single file or zip on Google Drive
                if progress_cb:
                    progress_cb({
                        "status": "downloading",
                        "current_file": "Downloading from Google Drive...",
                        "percent": 20.0,
                    })
                downloaded_file = gdown.download(
                    url=url,
                    output=str(dest_dir / "gdrive_download"),
                    quiet=True,
                    fuzzy=True,
                )
                if downloaded_file and Path(downloaded_file).exists():
                    p = Path(downloaded_file)
                    if zipfile.is_zipfile(p):
                        return self.extract_archive(p, dest_dir)
                    elif p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                        return [p]
        except Exception as e:
            print(f"[Universal Downloader] gdown error, attempting fallback: {e}")
            return self.download_direct_file(url, dest_dir, progress_cb)

        # Scan dest_dir for all extracted/downloaded audio files
        audio_files: List[Path] = []
        for p in dest_dir.rglob("*"):
            if p.is_file():
                if p.suffix.lower() in ARCHIVE_EXTS:
                    audio_files.extend(self.extract_archive(p, dest_dir))
                elif p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                    audio_files.append(p)

        return audio_files

    def fetch_all_from_urls(
        self,
        urls_input: str | List[str],
        session_id: Optional[str] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point: downloads, extracts, and stages audio from any URL(s).
        Returns metadata and list of discovered audio files ready for checking or DB indexing.
        """
        t0 = time.time()
        sid = session_id or str(int(time.time() * 1000))
        target_dir = self.temp_base_dir / sid
        target_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(urls_input, str):
            # Split by newline, comma, or semicolon
            raw_urls = [u.strip() for u in re.split(r"[\n,;]+", urls_input) if u.strip()]
        else:
            raw_urls = [u.strip() for u in urls_input if u and u.strip()]

        if not raw_urls:
            return {"status": "error", "message": "No valid URLs provided.", "audio_files": []}

        all_audio_files: List[Path] = []
        total_urls = len(raw_urls)

        for idx, url in enumerate(raw_urls):
            if progress_cb:
                progress_cb({
                    "status": "processing_url",
                    "current_file": f"Processing URL {idx+1}/{total_urls}: {url[:60]}...",
                    "percent": round((idx / total_urls) * 100, 1),
                })

            try:
                # Local archive, file path, or file:// URL support
                if url.startswith("file://") or (Path(url).exists() and not url.startswith("http")):
                    clean_path = url.replace("file:///", "").replace("file://", "")
                    clean_path = urllib.parse.unquote(clean_path)
                    local_p = Path(clean_path)
                    if not local_p.exists():
                        local_p = Path(url)
                    found = []
                    if local_p.exists():
                        if local_p.is_file():
                            if local_p.suffix.lower() in ARCHIVE_EXTS:
                                dest_archive = target_dir / local_p.name
                                shutil.copy2(local_p, dest_archive)
                                found = self.extract_archive(dest_archive, target_dir)
                            elif local_p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                                dest_audio = target_dir / local_p.name
                                shutil.copy2(local_p, dest_audio)
                                found = [dest_audio]
                        elif local_p.is_dir():
                            for p in local_p.rglob("*"):
                                if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                                    dest_audio = target_dir / p.name
                                    shutil.copy2(p, dest_audio)
                                    found.append(dest_audio)
                elif self.is_google_drive(url):
                    found = self.download_google_drive(url, target_dir, progress_cb)
                else:
                    found = self.download_direct_file(url, target_dir, progress_cb)
                all_audio_files.extend(found)
            except Exception as e:
                print(f"[Universal Downloader] Error fetching '{url}': {e}")

        # Final recursive scan in target_dir to catch any nested extracts
        final_files: List[Path] = []
        seen_paths = set()
        for p in target_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                real_p = p.resolve()
                if real_p not in seen_paths:
                    seen_paths.add(real_p)
                    final_files.append(p)

        total_bytes = sum(f.stat().st_size for f in final_files if f.exists())
        elapsed = round(time.time() - t0, 2)

        file_metadata = [
            {
                "name": f.name,
                "path": str(f.resolve()),
                "size": f.stat().st_size,
                "ext": f.suffix.upper().lstrip("."),
            }
            for f in final_files
        ]

        if progress_cb:
            progress_cb({
                "status": "complete",
                "current_file": f"Done! Discovered {len(final_files)} audio file(s).",
                "percent": 100.0,
            })

        return {
            "status": "success",
            "session_id": sid,
            "target_dir": str(target_dir.resolve()),
            "count": len(final_files),
            "total_bytes": total_bytes,
            "elapsed_sec": elapsed,
            "audio_files": file_metadata,
            "message": f"Successfully fetched and extracted {len(final_files)} audio file(s) ({round(total_bytes/(1024*1024), 2)} MB) in {elapsed}s.",
        }


# Global singleton instance
universal_downloader = UniversalUrlDownloader()
