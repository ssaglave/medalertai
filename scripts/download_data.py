"""
scripts/download_data.py — Download raw WPRDC CSVs.

Owner: Greeshma (C1)
Phase: 0

Downloads EMS and Fire dispatch data directly from WPRDC URLs
into data/raw/. Run this once after cloning the repo.

Usage:
    python scripts/download_data.py
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# WPRDC Direct Download URLs
DATA_SOURCES = {
    "EMS_Data.csv": "https://tools.wprdc.org/downstream/ff33ca18-2e0c-4cb5-bdcd-60a5dc3c0418",
    "Fire_Data.csv": "https://tools.wprdc.org/downstream/b6340d98-69a0-4965-a9b4-3480cea1182b",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _progress_hook(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        sys.stdout.write(f"\r    {pct:5.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
    else:
        mb_done = downloaded / 1e6
        sys.stdout.write(f"\r    {mb_done:.1f} MB downloaded")
    sys.stdout.flush()


def download_data(force: bool = False):
    """Download raw CSV files from WPRDC.

    Args:
        force: If True, re-download even if files already exist.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DATA_SOURCES.items():
        filepath = RAW_DIR / filename

        if filepath.exists() and not force:
            size_mb = filepath.stat().st_size / 1e6
            print(f"  ✓ {filename} already exists ({size_mb:.1f} MB) — skipping")
            print(f"    (use --force to re-download)")
            continue

        print(f"  ⬇ Downloading {filename}...")
        print(f"    URL: {url}")

        try:
            urlretrieve(url, filepath, reporthook=_progress_hook)
            size_mb = filepath.stat().st_size / 1e6
            print(f"\n  ✅ {filename} saved ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"\n  ❌ Failed to download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download

    print("\nDone. Data files are in:", RAW_DIR)


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("📥 MedAlertAI Data Downloader")
    print(f"   Target directory: {RAW_DIR}")
    if force:
        print("   Mode: Force re-download\n")
    else:
        print("   Mode: Skip existing files\n")
    download_data(force=force)
