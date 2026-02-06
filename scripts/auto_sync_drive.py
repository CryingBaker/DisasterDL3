
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
        
        print("\nPlease enter the path to your local Google Drive folder.")
        if default_path:
            print(f"Found potential match: {default_path}")
            user_input = input(f"Press Enter to use this, or type path: ").strip()
            dest_root = Path(user_input) if user_input else default_path
        else:
            dest_input = input("Path (e.g. /Volumes/GoogleDrive/My Drive): ").strip()
            if not dest_input:
                print("❌ No path provided. Exiting.")
                return
            dest_root = Path(dest_input)
        
    if not dest_root.exists():
        print(f"❌ Error: Path '{dest_root}' does not exist.")
        print("Please ensure Google Drive is running and mounted.")
        return

    # Create destination folder
    dest_folder = dest_root / "DisasterDL3_Backups"
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"✅ Syncing to: {dest_folder}")
    
    print("\nStarting Auto-Sync (Ctrl+C to stop)...")
    print("Target: logs/, checkpoints/, results/")
    
    try:
        while True:
            # Use rsync for efficient syncing (only changed files)
            # 1. Sync Logs
            subprocess.run(["rsync", "-av", "--exclude", "*.DS_Store", str(logs_dir), str(dest_folder)], check=True)
            
            # 2. Sync Checkpoints
            subprocess.run(["rsync", "-av", "--exclude", "*.DS_Store", str(checkpoints_dir), str(dest_folder)], check=True)

            # 3. Sync Results (Top 5 Samples)
            if results_dir.exists():
                subprocess.run(["rsync", "-av", "--exclude", "*.DS_Store", str(results_dir), str(dest_folder)], check=True)
            
            print(f"[{time.strftime('%H:%M:%S')}] Sync complete. Waiting 60s...")
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
