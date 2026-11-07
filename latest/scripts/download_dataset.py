"""
download_dataset.py
===================
Dataset Downloader & Environment Initializer.

Checks for local presence of NF-UNSW-NB15-v3.csv dataset. If missing, displays instructions or fetches the official NetFlow-UNSW-NB15-v3 benchmark file from the public repository.

Run:
    python scripts/download_dataset.py
"""

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, ".."))
DATASET_PATH = os.path.join(ROOT_DIR, "NF-UNSW-NB15-v3.csv")

# Public mirror for NF-UNSW-NB15-v3 benchmark dataset
DATASET_URL = "https://rdm.uq.edu.au/files/f7546561558c07c5_NFV3DATA-A11964_A11964/NF-UNSW-NB15-v3.csv"


def check_and_download():
    print("==================================================")
    print("NIDS DATASET CHECK & DOWNLOAD UTILITY")
    print("==================================================")
    print(f"Target location: {DATASET_PATH}")

    if os.path.exists(DATASET_PATH):
        size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
        print(f"[OK] Dataset already present locally ({size_mb:.2f} MB). No download needed.")
        return True

    print("Dataset not found locally. Initiating download ...")
    try:
        def _progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / max(total_size, 1))
            sys.stdout.write(f"\rDownloading NF-UNSW-NB15-v3.csv: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH, reporthook=_progress)
        print("\n[OK] Download complete.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Automatic download failed: {e}")
        print("\nManual Instructions:")
        print("1. Download NF-UNSW-NB15-v3.csv from UQ Research Data Repository.")
        print(f"2. Place the file at: {DATASET_PATH}")
        return False


if __name__ == "__main__":
    check_and_download()
