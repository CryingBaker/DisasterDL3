#!/usr/bin/env python3
"""
Download Weakly Labeled Data from Sen1Floods11
==============================================
Downloads the "Weakly Labeled" subset (~13 GB) from the public GCS bucket.

Sources:
- S1Weak (Images): gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak/
- LabelWeak (Labels): gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/

Destinations:
- data/sen1floods11/S1Weak
- data/sen1floods11/LabelWeak
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a shell command."""
    print(f"Running: {cmd}")
    # Don't capture output, let it flow to stdout/stderr (avoids buffer deadlock)
    try:
        ret = subprocess.run(cmd, shell=True)
        return ret.returncode
    except Exception as e:
        print(f"Command failed: {e}")
        return 1

def main():
    base_dir = Path("data/sen1floods11")
    s1_weak_dir = base_dir / "S1Weak"
    label_weak_dir = base_dir / "LabelWeak"
    
    # Create directories
    s1_weak_dir.mkdir(parents=True, exist_ok=True)
    label_weak_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Weakly Labeled Data (This may take a while...)")
    print("=" * 60)
    
    # Check for gsutil
    if subprocess.call("which gsutil", shell=True) != 0:
        print("Error: gsutil not found based on 'which gsutil'.")
        print("Please install Google Cloud SDK or ensure gsutil is in your PATH.")
        sys.exit(1)

    # 1. Download S1Weak Images
    print("\n[1/2] Downloading S1Weak images...")
    cmd_s1 = f"gsutil -m cp -r -n gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak/* {s1_weak_dir}/"
    if run_command(cmd_s1) != 0:
        print("Failed to download S1Weak images.")
        # Don't exit, try labels too
    
    # 2. Download Weak Labels
    print("\n[2/2] Downloading S1OtsuLabelWeak labels...")
    cmd_lbl = f"gsutil -m cp -r -n gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/* {label_weak_dir}/"
    if run_command(cmd_lbl) != 0:
        print("Failed to download Weak Labels.")
        
    print("\n" + "=" * 60)
    print("Download process finished.")
    
    # Check counts
    s1_count = len(list(s1_weak_dir.glob("*.tif")))
    lbl_count = len(list(label_weak_dir.glob("*.tif")))
    print(f"S1Weak Files: {s1_count}")
    print(f"LabelWeak Files: {lbl_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
