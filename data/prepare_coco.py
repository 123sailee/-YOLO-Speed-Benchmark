"""Prepare COCO dataset for benchmarking (optional)."""

import os
import subprocess
from pathlib import Path


def download_coco_val2017():
    """
    Download COCO validation 2017 dataset.
    
    This is optional for benchmarking. You can also use your own images.
    
    Usage:
        python data/prepare_coco.py
    
    Note:
        - Downloads ~1 GB of data
        - Requires internet connection
        - Extracts to 'data/coco/val2017/'
    """
    coco_dir = Path('data/coco')
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    val2017_url = "http://images.cocodataset.org/zips/val2017.zip"
    
    print("Downloading COCO val2017 dataset...")
    print(f"URL: {val2017_url}")
    print(f"Destination: {coco_dir}")
    
    # Example download command (requires wget or curl)
    # subprocess.run(['wget', val2017_url, '-P', str(coco_dir)])
    
    print("\n📝 To download manually:")
    print(f"1. Visit: {val2017_url}")
    print(f"2. Extract to: {coco_dir}/val2017/")
    print("\nThen run benchmark with:")
    print("  python benchmark.py --sample-dir data/coco/val2017 --samples 100")


if __name__ == '__main__':
    download_coco_val2017()
