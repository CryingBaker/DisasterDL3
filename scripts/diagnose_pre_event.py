import rasterio
import glob
import numpy as np
from pathlib import Path

def check_files():
    files = glob.glob('data/pre_event_sar/*.tif')
    blank_count = 0
    non_blank = []
    
    print(f"Checking {len(files)} files...")
    
    for f in files:
        with rasterio.open(f) as src:
            data = src.read()
            if data.max() == 0 and data.min() == 0:
                blank_count += 1
            else:
                non_blank.append(f)
                
    print(f"Total files: {len(files)}")
    print(f"Blank files: {blank_count}")
    print(f"Non-blank files: {len(non_blank)}")
    
    if non_blank:
        print("First 5 non-blank files:")
        for f in non_blank[:5]:
            with rasterio.open(f) as src:
                data = src.read()
                print(f"  {f}: min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}")

if __name__ == "__main__":
    check_files()
