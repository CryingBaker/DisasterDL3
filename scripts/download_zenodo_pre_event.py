import os
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/record/7946979/files/PRE_S1-20230517T191707Z-001.zip"
DEST_DIR = Path("data/pre_event_sar")
ZIP_PATH = DEST_DIR / "zenodo_pre_event.zip"

def download_file(url, params=None):
    local_filename = ZIP_PATH
    with requests.get(url, stream=True, params=params) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_filename.name) as pbar:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
                    pbar.update(len(chunk))
    return local_filename

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download
    if not ZIP_PATH.exists():
        print(f"Downloading from {ZENODO_URL}...")
        try:
            download_file(ZENODO_URL)
        except Exception as e:
            print(f"Download failed: {e}")
            return
    else:
        print("Zip file already exists, skipping download.")

    # 2. Verify Contents
    print("Verifying zip contents against local dataset...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_files = set(zip_ref.namelist())
            
            # Get local S1Hand files to match against
            # Expected naming in zip: "Bolivia_12345_S1Hand.tif" or similar?
            # Actually pre-event likely usually named "Bolivia_12345_PreEvent.tif" or similar.
            # But the STEM (Bolivia_12345) should match.
            
            local_s1 = list(Path("data/sen1floods11/S1Hand").glob("*_S1Hand.tif"))
            local_stems = {f.name.replace("_S1Hand.tif", "") for f in local_s1}
            
            print(f"Local samples: {len(local_stems)}")
            
            # Let's check a few filenames from the zip to guess the pattern
            sample_zip = list(zip_files)[:5]
            print(f"Sample zip files: {sample_zip}")
            
            # Heuristic check
            match_count = 0
            for stem in local_stems:
                # patterns to check
                p1 = f"{stem}_PreEvent.tif"
                p2 = f"pre_event_sar/{stem}.tif" 
                # Check for partial matches in the list
                if any(stem in z for z in zip_files):
                    match_count += 1
            
            print(f"Found matches for {match_count}/{len(local_stems)} samples.")
            
            if match_count < len(local_stems) * 0.5:
                print("WARNING: Low match rate. This might not be the right dataset.")
                user_input = input("Continue extraction? (y/n): ")
                if user_input.lower() != 'y':
                    print("Aborted by validation check.")
                    return
            else:
                print("VALIDATION SUCCESS: Dataset matches local samples.")

            print("Extracting...")
            zip_ref.extractall(DEST_DIR)
            
        print("Extraction complete.")
        
        # 3. Flatten/Organize if needed
        # Often these zips have a subfolder. Let's list and see.
        print("Contents of data/pre_event_sar:")
        for child in DEST_DIR.iterdir():
            print(f" - {child.name}")
            
    except zipfile.BadZipFile:
        print("Error: Bad zip file. Delete it and try again.")
        # os.remove(ZIP_PATH)

if __name__ == "__main__":
    main()
