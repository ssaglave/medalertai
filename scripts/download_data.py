"""
scripts/download_data.py — Download raw WPRDC CSVs.

Owner: Greeshma (C1)
Phase: 0

Note: If EMS_Data.csv and Fire_Data.csv are already in data/raw/,
this script can be skipped. It exists for reproducibility.
"""

import os
from pathlib import Path

# WPRDC Data URLs (Pittsburgh EMS and Fire dispatch records)
DATA_URLS = {
    "EMS_Data.csv": "https://data.wprdc.org/dataset/ems-fire-dispatch-data",
    "Fire_Data.csv": "https://data.wprdc.org/dataset/ems-fire-dispatch-data",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_data():
    """Download raw CSV files from WPRDC."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DATA_URLS.items():
        filepath = RAW_DIR / filename
        if filepath.exists():
            print(f"  ✓ {filename} already exists ({filepath.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  ⬇ Download {filename} from: {url}")
            # TODO: Implement actual download via requests or WPRDC API
            print(f"    → Manual download required. Place file at: {filepath}")

    print("\nDone.")


if __name__ == "__main__":
    print("📥 MedAlertAI Data Downloader")
    print(f"   Target directory: {RAW_DIR}\n")
    download_data()
