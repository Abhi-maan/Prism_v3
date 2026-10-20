#!/usr/bin/env python3
"""
Environment Setup Script for Internet Audio Detector
===================================================
Installs required Python dependencies and ensures Chromaprint (fpcalc)
is installed and ready to execute.
"""

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import tarfile
import shutil

FPCALC_VERSION = "1.6.0"
DOWNLOAD_URLS = {
    "Windows": f"https://github.com/acoustid/chromaprint/releases/download/v{FPCALC_VERSION}/chromaprint-fpcalc-{FPCALC_VERSION}-windows-x86_64.zip",
    "Linux": f"https://github.com/acoustid/chromaprint/releases/download/v{FPCALC_VERSION}/chromaprint-fpcalc-{FPCALC_VERSION}-linux-x86_64.tar.gz",
    "Darwin": f"https://github.com/acoustid/chromaprint/releases/download/v{FPCALC_VERSION}/chromaprint-fpcalc-{FPCALC_VERSION}-macos-x86_64.tar.gz"
}


def install_requirements():
    print("[*] Installing Python packages from requirements.txt...")
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "pyacoustid", "pandas", "numpy"])
    print("[+] Python requirements successfully installed.")


def ensure_fpcalc():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ext = ".exe" if platform.system() == "Windows" else ""
    local_fpcalc = os.path.join(cur_dir, f"fpcalc{ext}")

    if os.path.isfile(local_fpcalc):
        print(f"[+] fpcalc binary found at: {local_fpcalc}")
        return local_fpcalc

    if shutil.which("fpcalc"):
        sys_fpcalc = shutil.which("fpcalc")
        print(f"[+] System fpcalc binary found at: {sys_fpcalc}")
        return sys_fpcalc

    system_name = platform.system()
    url = DOWNLOAD_URLS.get(system_name)
    if not url:
        print(f"[-] Unsupported OS '{system_name}' for automatic binary download.")
        print("    Please install chromaprint / fpcalc manually.")
        return None

    archive_name = "fpcalc_pkg" + (".zip" if system_name == "Windows" else ".tar.gz")
    archive_path = os.path.join(cur_dir, archive_name)

    print(f"[*] Downloading fpcalc v{FPCALC_VERSION} from {url}...")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as e:
        print(f"[-] Download failed: {e}")
        return None

    print("[*] Extracting fpcalc binary...")
    extracted_path = None
    if system_name == "Windows":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("fpcalc.exe"):
                    with open(local_fpcalc, "wb") as f_out:
                        f_out.write(zf.read(member))
                    extracted_path = local_fpcalc
                    break
    else:
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.name.endswith("fpcalc"):
                    f_in = tf.extractfile(member)
                    if f_in:
                        with open(local_fpcalc, "wb") as f_out:
                            f_out.write(f_in.read())
                        os.chmod(local_fpcalc, 0o755)
                        extracted_path = local_fpcalc
                        break

    if os.path.exists(archive_path):
        os.remove(archive_path)

    if extracted_path:
        print(f"[+] fpcalc extracted to: {extracted_path}")
    return extracted_path


if __name__ == "__main__":
    install_requirements()
    fpcalc_path = ensure_fpcalc()
    if fpcalc_path:
        try:
            res = subprocess.run([fpcalc_path, "-v"], capture_output=True, text=True)
            print("[+] Verification successful:")
            print("   ", res.stdout.strip())
        except Exception as e:
            print(f"[-] Verification error: {e}")
