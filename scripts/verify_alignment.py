
import os
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

def verify_alignment(sen1floods_dir, pre_event_dir):
    """
    Verifies that Pre-Event SAR images match the geospatial bounds and CRS
    of the Post-Event (Sen1Floods11) images.
    """
    s1_dir = Path(sen1floods_dir) / "S1Hand"
    pre_dir = Path(pre_event_dir)
    
    print(f"Scanning Post-Event: {s1_dir}")
    print(f"Scanning Pre-Event: {pre_dir}")
    
    # Get common IDs
    s1_files = {f.name.replace("_S1Hand.tif", ""): f for f in s1_dir.glob("*_S1Hand.tif")}
    pre_files = {f.name.replace("_PreEvent.tif", ""): f for f in pre_dir.glob("*_PreEvent.tif")}
    
    common_ids = set(s1_files.keys()).intersection(set(pre_files.keys()))
    print(f"Found {len(common_ids)} pairs to verify.")
    
    if len(common_ids) == 0:
        print("No paired files found yet. Waiting for download...")
        return

    mismatches = []
    checked = 0
    
    for sample_id in tqdm(common_ids, desc="Verifying Alignment"):
        post_path = s1_files[sample_id]
        pre_path = pre_files[sample_id]
        
        try:
            with rasterio.open(post_path) as src_post, rasterio.open(pre_path) as src_pre:
                # 1. Check CRS
                if src_post.crs != src_pre.crs:
                    mismatches.append(f"{sample_id}: CRS mismatch ({src_post.crs} vs {src_pre.crs})")
                    continue
                
                # 2. Check Bounds (allow tiny float tolerance)
                # Bounds: left, bottom, right, top
                if not np.allclose(src_post.bounds, src_pre.bounds, atol=1e-5):
                    mismatches.append(f"{sample_id}: Bounds mismatch\n  Post: {src_post.bounds}\n  Pre:  {src_pre.bounds}")
                    continue
                
                # 3. Check Dimensions
                if src_post.shape != src_pre.shape:
                    mismatches.append(f"{sample_id}: Shape mismatch ({src_post.shape} vs {src_pre.shape})")
                    continue
                
                checked += 1
                
        except Exception as e:
            mismatches.append(f"{sample_id}: Error reading files: {e}")

    print("\n" + "="*40)
    print("VERIFICATION RESULTS")
    print("="*40)
    print(f"Checked: {checked}/{len(common_ids)}")
    
    if not mismatches:
        print("✅ SUCCESS: All checked Pre-Event images are perfectly aligned with Post-Event data.")
    else:
        print(f"❌ FAILURE: Found {len(mismatches)} mismatches.")
        for m in mismatches[:10]:
            print(m)
        if len(mismatches) > 10:
            print(f"...and {len(mismatches)-10} more.")

if __name__ == "__main__":
    verify_alignment(
        sen1floods_dir="./data/sen1floods11",
        pre_event_dir="./data/pre_event_sar"
    )
