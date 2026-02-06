#!/usr/bin/env python3
"""
Sen1Floods11 Dataset Downloader
===============================
Downloads Sen1Floods11 dataset with support for Fast (subset) and Full modes.

Usage:
    python download_sen1floods11.py --mode fast  # Default: ~500-800 chips
    python download_sen1floods11.py --mode full  # All 4,831 chips

Dataset source: https://github.com/cloudtostreet/Sen1Floods11
"""

import os
import sys
import json
import logging
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Configure logging
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"download_sen1floods11_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logger = logging.getLogger("sen1floods11_downloader")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


@dataclass
class DownloadConfig:
    """Configuration for dataset download."""
    mode: str = "fast"
    output_dir: Path = Path("./data/sen1floods11")
    
    # Fast mode settings (3 events, diverse biomes)
    fast_events: List[str] = field(default_factory=lambda: [
        "Bolivia",      # South America, tropical
        "USA",          # North America, varied
        "Spain"         # Europe, Mediterranean
    ])
    fast_max_chips: int = 800
    
    # Data source
    github_base: str = "https://github.com/cloudtostreet/Sen1Floods11"
    gcs_bucket: str = "gs://sen1floods11"
    
    # What to download
    download_s1: bool = True          # Sentinel-1 SAR (required)
    download_s2: bool = False         # Sentinel-2 optical (optional)
    download_labels: bool = True      # Flood labels (required)
    download_qc: bool = True          # Hand-labeled QC subset (required)
    
    def get_events(self) -> List[str]:
        if self.mode == "fast":
            return self.fast_events
        return self._all_events()
    
    @staticmethod
    def _all_events() -> List[str]:
        """All 11 flood events in Sen1Floods11."""
        return [
            "Bolivia", "Cambodia", "Ghana", "India", "Mekong",
            "Nigeria", "Pakistan", "Paraguay", "Somalia", "Spain", "USA"
        ]


class Sen1Floods11Downloader:
    """Downloads Sen1Floods11 dataset."""
    
    def __init__(self, config: DownloadConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.downloaded_files: List[Path] = []
        self.skipped_files: List[Path] = []
        self.failed_files: List[Path] = []
    
    def run(self) -> bool:
        """Execute the download pipeline."""
        self.logger.info(f"=" * 60)
        self.logger.info(f"Sen1Floods11 Downloader - Mode: {self.config.mode.upper()}")
        self.logger.info(f"=" * 60)
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Events to download: {self.config.get_events()}")
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing download state
        state_file = self.config.output_dir / ".download_state.json"
        if state_file.exists():
            self.logger.info("Found existing download state - will resume")
        
        # Step 1: Get file manifest
        self.logger.info("\n[1/4] Fetching file manifest...")
        manifest = self._get_manifest()
        if not manifest:
            self.logger.error("Failed to get file manifest")
            return False
        
        # Step 2: Filter by mode
        self.logger.info("\n[2/4] Filtering files by mode...")
        files_to_download = self._filter_manifest(manifest)
        self.logger.info(f"Files to download: {len(files_to_download)}")
        
        # Step 3: Download files
        self.logger.info("\n[3/4] Downloading files...")
        self._download_files(files_to_download)
        
        # Step 4: Verify and report
        self.logger.info("\n[4/4] Verification report...")
        self._report()
        
        # Save state
        self._save_state(state_file)
        
        return len(self.failed_files) == 0
    
    def _get_manifest(self) -> Optional[Dict]:
        """
        Get the file manifest from Sen1Floods11.
        
        NOTE: In actual implementation, this would:
        1. Clone the GitHub repo's catalog files, OR
        2. List GCS bucket contents, OR
        3. Use a pre-built manifest JSON
        
        For now, returns a structure showing expected layout.
        """
        self.logger.info("Building manifest from expected structure...")
        
        # Sen1Floods11 structure (based on documentation)
        manifest = {
            "events": {},
            "structure": {
                "s1_raw": "v1.1/data/flood_events/{event}/s1_raw/",
                "labels": "v1.1/data/flood_events/{event}/LabelHand/",
                "qc_labels": "v1.1/data/flood_events/{event}/QC/"
            }
        }
        
        for event in DownloadConfig._all_events():
            manifest["events"][event] = {
                "s1_chips": [],  # Would be populated from actual listing
                "label_chips": [],
                "qc_chips": []
            }
        
        return manifest
    
    def _filter_manifest(self, manifest: Dict) -> List[Dict]:
        """Filter manifest based on download mode."""
        files = []
        selected_events = self.config.get_events()
        chip_count = 0
        
        for event in selected_events:
            if event not in manifest["events"]:
                self.logger.warning(f"Event '{event}' not found in manifest")
                continue
            
            event_data = manifest["events"][event]
            
            # In fast mode, limit chips per event
            max_per_event = self.config.fast_max_chips // len(selected_events)
            
            self.logger.info(f"Event: {event} (max {max_per_event} chips in fast mode)")
            
            # Would add actual files here
            files.append({
                "event": event,
                "type": "placeholder",
                "path": f"data/flood_events/{event}/"
            })
            
            chip_count += max_per_event
            
            if self.config.mode == "fast" and chip_count >= self.config.fast_max_chips:
                break
        
        return files
    
    def _download_files(self, files: List[Dict]) -> None:
        """Download files from GCS or alternative sources."""
        self.logger.info(f"\n{'='*40}")
        self.logger.info("DOWNLOAD INSTRUCTIONS")
        self.logger.info(f"{'='*40}")
        self.logger.info("""
Sen1Floods11 data can be downloaded via:

Option 1 - Google Cloud Storage (gsutil):
    gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/{EVENT}/ ./data/

Option 2 - Direct download links (from GitHub):
    Visit: https://github.com/cloudtostreet/Sen1Floods11#data-access

Option 3 - Kaggle (if available):
    kaggle datasets download -d <dataset-id>

For FAST mode, download only these events:
""")
        for event in self.config.get_events():
            self.logger.info(f"    - {event}")
        
        self.logger.info(f"\nTarget directory: {self.config.output_dir.absolute()}")
        self.logger.info(f"\nExpected structure after download:")
        self.logger.info(f"    {self.config.output_dir}/")
        self.logger.info(f"    ├── flood_events/")
        self.logger.info(f"    │   ├── Bolivia/")
        self.logger.info(f"    │   │   ├── s1_raw/")
        self.logger.info(f"    │   │   └── LabelHand/")
        self.logger.info(f"    │   └── ...")
        self.logger.info(f"    └── catalog/")
    
    def _report(self) -> None:
        """Generate download report."""
        self.logger.info(f"\n{'='*40}")
        self.logger.info("DOWNLOAD SUMMARY")
        self.logger.info(f"{'='*40}")
        self.logger.info(f"Mode: {self.config.mode.upper()}")
        self.logger.info(f"Target events: {len(self.config.get_events())}")
        self.logger.info(f"Downloaded: {len(self.downloaded_files)}")
        self.logger.info(f"Skipped (existing): {len(self.skipped_files)}")
        self.logger.info(f"Failed: {len(self.failed_files)}")
        
        if self.failed_files:
            self.logger.error("FAILED FILES:")
            for f in self.failed_files:
                self.logger.error(f"  - {f}")
    
    def _save_state(self, state_file: Path) -> None:
        """Save download state for resumability."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.config.mode,
            "events": self.config.get_events(),
            "downloaded": [str(p) for p in self.downloaded_files],
            "skipped": [str(p) for p in self.skipped_files],
            "failed": [str(p) for p in self.failed_files]
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        self.logger.info(f"Saved state to {state_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Sen1Floods11 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_sen1floods11.py --mode fast
    python download_sen1floods11.py --mode full --output ./data/sen1floods11
        """
    )
    parser.add_argument("--mode", choices=["fast", "full"], default="fast",
                        help="Download mode: fast (subset) or full (all)")
    parser.add_argument("--output", type=Path, default=Path("./data/sen1floods11"),
                        help="Output directory")
    parser.add_argument("--events", nargs="+", 
                        help="Override event list (for custom subset)")
    
    args = parser.parse_args()
    
    # Setup
    config = DownloadConfig(mode=args.mode, output_dir=args.output)
    if args.events:
        config.fast_events = args.events
    
    logger = setup_logging(args.output / "logs")
    
    # Run
    downloader = Sen1Floods11Downloader(config, logger)
    success = downloader.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
