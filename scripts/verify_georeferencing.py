import rasterio
from pathlib import Path
import glob

DATA_DIR = Path("data/sen1floods11/S1Hand")
files = list(DATA_DIR.glob("*.tif"))

if not files:
    print("No files found.")
else:
    with rasterio.open(files[0]) as src:
        print(f"File: {files[0].name}")
        print(f"Bounds: {src.bounds}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
