
import os
import time
import shutil
import subprocess
from pathlib import Path
import sys

def sync_to_drive():
    print("="*60)
    print("🌊 DisasterDL3 Google Drive Auto-Sync")
    print("="*60)
    
    # 1. Get Source Paths
    project_root = Path.cwd()
    logs_dir = project_root / "logs"
    checkpoints_dir = project_root / "checkpoints"
    results_dir = project_root / "results"
    
    print(f"Source: {project_root}")
    
    # 2. Get Destination Path
    if len(sys.argv) > 1:
        dest_root = Path(sys.argv[1])
    else:
        # Suggest common locations on macOS
        cloud_storage = Path.home() / "Library/CloudStorage"
        drive_candidates = list(cloud_storage.glob("GoogleDrive*")) if cloud_storage.exists() else []
        if not drive_candidates:
            # Check /Volumes
            drive_candidates = list(Path("/Volumes").glob("GoogleDrive*"))
            
        default_path = drive_candidates[0] if drive_candidates else None
        
        print("\n" + "="*60)
        print("❓ Google Drive Setup")
        print("="*60)
        
        if default_path:
            print(f"I found Google Drive at: {default_path}")
            print("Press [Enter] to use this path.")
            print("Or paste a different path below.")
        else:
            print("⚠️  I could not automatically find your Google Drive folder.")
            print("Please paste the full path to your Google Drive folder below.")
            print("If you don't know it, just press [Enter] and I will save to your Desktop.")
            print("(You can then drag that folder to Google Drive manually).")
            
        user_input = input(f"\nPath > ").strip()
        
        if user_input:
            dest_root = Path(user_input)
        elif default_path:
            dest_root = default_path
        else:
            dest_root = Path.home() / "Desktop"
            print(f"\n⚠️  Using Desktop fallback: {dest_root}")

    # Verify path (unless it's the Desktop fallback which we know exists/can create)
    if not dest_root.exists() and "Desktop" not in str(dest_root):
        print(f"❌ Error: Path '{dest_root}' does not exist.")
        print("Please ensure Google Drive is running.")
        return

    # Create destination folder structure
    # User asked for "DisasterDL" folder
    dest_folder = dest_root / "DisasterDL"
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Create subfolders for better organization
    (dest_folder / "active_run").mkdir(exist_ok=True)
    
    print(f"✅ Syncing to: {dest_folder}")
    print(f"   (Folder 'DisasterDL' created)")
    
    print("\nStarting Auto-Sync (Ctrl+C to stop)...")
    print("Monitoring: logs/, checkpoints/, results/ -> DisasterDL/")
    
    try:
        while True:
            # Sync to 'active_run' to avoid clutter
            target = dest_folder # Sync directly to DisasterDL root or active_run? 
            # User said "upload it there", implies root of that folder.
            
            # Use rsync for efficient syncing
            subprocess.run(["rsync", "-av", "--exclude", "*.DS_Store", str(logs_dir), str(dest_folder)], check=True)
            subprocess.run(["rsync", "-av", "--exclude", "*.DS_Store", str(checkpoints_dir), str(dest_folder)], check=True)
            if results_dir.exists():
                subprocess.run(["rsync", "-av", "--exclude", "*.DS_Store", str(results_dir), str(dest_folder)], check=True)
            
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Sync complete. Next update in 60s...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n🛑 Sync stopped.")
    except Exception as e:
        print(f"\n❌ Error during sync: {e}")

if __name__ == "__main__":
    # Ensure we are in project root
    if not (Path.cwd() / "src").exists():
        print("Please run this script from the project root directory.")
        sys.exit(1)
    sync_to_drive()
