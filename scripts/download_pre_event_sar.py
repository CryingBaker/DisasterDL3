#!/usr/bin/env python3
"""
Pre-Event Sentinel-1 SAR Downloader
====================================
Retrieves pre-event SAR imagery matching Sen1Floods11 post-event data.

Requires: earthengine-api (pip install earthengine-api)

Usage:
    python download_pre_event_sar.py --mode fast --sen1floods11_dir ./data/sen1floods11
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Note: Requires Google Earth Engine authentication
# Run: earthengine authenticate (one-time setup)

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"download_pre_event_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logger = logging.getLogger("pre_event_downloader")
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


@dataclass
class PreEventConfig:
    """Configuration for pre-event SAR retrieval."""
    mode: str = "fast"
    sen1floods11_dir: Path = Path("./data/sen1floods11")
    output_dir: Path = Path("./data/pre_event_sar")
    source_dir: str = "S1Hand"  # Subdirectory to scan for events
    
    # Temporal selection
    min_days_before: int = 14   # Minimum days before flood
    max_days_before: int = 45   # Maximum days before flood
    target_days_before: int = 21  # Ideal target
    
    # Fallback settings
    use_median_composite: bool = True
    max_composite_images: int = 3
    
    # Matching requirements
    require_same_orbit: bool = True
    require_same_pass: bool = True
    polarizations: List[str] = field(default_factory=lambda: ["VV", "VH"])


# FLOOD EVENT METADATA (from Sen1Floods11 documentation)
# FLOOD EVENT METADATA (from Sen1Floods11 documentation)
FLOOD_EVENTS = {
    "Bolivia": {
        "date": "2018-02-15",
        "bbox": [-65.64, -15.96, -64.36, -11.39],
        "orbit_pass": "DESCENDING"
    },
    "Ghana": {
        "date": "2018-09-18",
        "bbox": [-2.30, 6.30, 0.22, 11.93],
        "orbit_pass": "ASCENDING"
    },
    "India": {
        "date": "2016-08-12",
        "bbox": [92.15, 24.85, 94.16, 28.28],
        "orbit_pass": "DESCENDING"
    },
    "Mekong": {  # Mapped from "Cambodia" in metadata
        "date": "2018-08-05", 
        "bbox": [104.07, 10.57, 106.43, 14.27],
        "orbit_pass": "ASCENDING"
    },
    "Nigeria": {
        "date": "2018-09-21",
        "bbox": [4.53, 4.12, 6.95, 10.54],
        "orbit_pass": "ASCENDING"
    },
    "Pakistan": {
        "date": "2017-06-28",
        "bbox": [68.99, 28.04, 72.55, 34.35],
        "orbit_pass": "DESCENDING"
    },
    "Paraguay": {
        "date": "2018-10-31",
        "bbox": [-58.59, -28.19, -54.65, -21.64],
        "orbit_pass": "DESCENDING"
    },
    "Somalia": {
        "date": "2018-05-07",
        "bbox": [44.20, 1.31, 46.58, 6.55],
        "orbit_pass": "ASCENDING"
    },
    "Spain": {
        "date": "2019-09-17",
        "bbox": [-1.11, 37.66, 1.22, 39.41],
        "orbit_pass": "DESCENDING"
    },
    "Sri-Lanka": {
        "date": "2017-05-30",
        "bbox": [80.13, 5.14, 82.09, 9.79],
        "orbit_pass": "DESCENDING"
    },
    "USA": {
        "date": "2019-05-22",
        "bbox": [-95.69, 38.36, -94.36, 41.06],
        "orbit_pass": "ASCENDING"
    },
}


class PreEventSARDownloader:
    """Downloads pre-event SAR imagery from Google Earth Engine."""
    
    def __init__(self, config: PreEventConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.ee_initialized = False
        self.stats = {"success": 0, "failed": 0, "skipped": 0}
    
    def run(self) -> bool:
        """Execute the pre-event retrieval pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Pre-Event SAR Retrieval")
        self.logger.info("=" * 60)
        
        # Initialize GEE
        if not self._init_gee():
            return False
        
        # Get chips to process
        chips = self._get_post_event_chips()
        self.logger.info(f"Found {len(chips)} post-event chips to match")
        
        # Process each chip
        for chip in chips:
            self._process_chip(chip)
        
        # Report
        self._report()
        return self.stats["failed"] == 0
    
    def _init_gee(self) -> bool:
        """Initialize Google Earth Engine."""
        try:
            import ee
        except ImportError:
            self.logger.error("earthengine-api not installed. Run: pip install earthengine-api")
            self._print_manual_instructions()
            return False

        PROJECT_ID = 'rare-ridge-486516-k1'
        try:
            # Explicitly use the project ID in initialization
            ee.Initialize(project=PROJECT_ID)
            self.ee_initialized = True
            self.logger.info(f"GEE initialized with project: {PROJECT_ID}")
            return True
        except Exception as e:
            self.logger.warning(f"Standard initialization failed: {e}")
            self.logger.info("Attempting to use Application Default Credentials (ADC)...")
            try:
                import google.auth
                credentials, project = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
                )
                # Override with user provided project if ADC project is missing or different
                ee.Initialize(credentials=credentials, project=PROJECT_ID)
                self.ee_initialized = True
                self.logger.info(f"GEE initialized with ADC and project: {PROJECT_ID}")
                return True
            except Exception as adc_e:
                self.logger.error(f"ADC initialization failed: {adc_e}")
                self.logger.info("Run: earthengine authenticate")
                self._print_manual_instructions()
                return False
    
    def _get_post_event_chips(self) -> List[Dict]:
        """Get list of post-event chips from Sen1Floods11."""
        chips = []
        
        # USE CONFIGURABLE SOURCE DIRECTORY
        source_dir = self.config.sen1floods11_dir / self.config.source_dir
        if source_dir.exists():
            self.logger.info(f"Scanning source directory: {source_dir}")
            found_chips = False
            # Match any tif ending with suffix or just .tif if standard
            # Heuristic: S1Hand uses _S1Hand.tif, S1Weak uses _S1Weak.tif
            # Let's glob all .tif and handle naming in process_chip
            for chip_file in source_dir.glob("*.tif"):
                # Filter out obvious non-chip files if any
                if chip_file.name.startswith("."): continue
                
                # Parse event from filename: Event_ID_S1Hand.tif
                event_name = chip_file.name.split('_')[0]
                chips.append({
                    "event": event_name,
                    "chip_path": chip_file,
                    "level": "chip"
                })
                found_chips = True
            
            if found_chips:
                self.logger.info(f"Found {len(chips)} chips in {self.config.source_dir}.")
                return chips

        self.logger.info(f"Source dir {source_dir} not found or empty.")
        return []
    
    def _process_chip(self, chip: Dict) -> None:
        """Process a single chip/event to retrieve pre-event SAR."""
        # Skip small/corrupt files (e.g. 3KB XML errors)
        if chip["chip_path"].stat().st_size < 50000:
            self.logger.warning(f"Skipping small file (likely corrupt): {chip['chip_path'].name}")
            self.stats["skipped"] += 1
            return

        event = chip["event"]
        
        if event not in FLOOD_EVENTS:
            self.logger.warning(f"No metadata for event: {event}")
            self.stats["skipped"] += 1
            return
        
        meta = FLOOD_EVENTS[event]
        flood_date = datetime.strptime(meta["date"], "%Y-%m-%d")
        
        # Calculate search window
        start_date = flood_date - timedelta(days=self.config.max_days_before)
        end_date = flood_date - timedelta(days=self.config.min_days_before)
        
        self.logger.info(f"\n{event}: Searching {start_date.date()} to {end_date.date()}")
        
        # Determine strict bounds from the local chip
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            import ee
            
            with rasterio.open(chip["chip_path"]) as src:
                # Get bounds in source CRS
                left, bottom, right, top = src.bounds
                crs = src.crs
                
                # Transform to WGS84 (EPSG:4326) for GEE
                minx, miny, maxx, maxy = transform_bounds(crs, {'init': 'epsg:4326'}, left, bottom, right, top)
                
                # Create GEE Geometry
                chip_geom = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
                
        except Exception as e:
            self.logger.error(f"Failed to read bounds from {chip['chip_path']}: {e}")
            self.stats["failed"] += 1
            return

        # Determine strict bounds from the local chip
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            import ee
            
            with rasterio.open(chip["chip_path"]) as src:
                # Get bounds in source CRS
                left, bottom, right, top = src.bounds
                crs = src.crs
                
                # Transform to WGS84 (EPSG:4326) for GEE
                minx, miny, maxx, maxy = transform_bounds(crs, {'init': 'epsg:4326'}, left, bottom, right, top)
                
                # Create GEE Geometry
                chip_geom = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
                
        except Exception as e:
            self.logger.error(f"Failed to read bounds from {chip['chip_path']}: {e}")
            self.stats["failed"] += 1
            return
            
        # GEE query
        if self.ee_initialized:
            # UPDATED: Pass chip_geom to query for chip-specific coverage
            image = self._query_gee(event, chip_geom, start_date, end_date)
            if image:
                # Construct filename
                # Remove known suffixes to get base ID
                stem = chip["chip_path"].stem
                chip_id = stem.replace("_S1Hand", "").replace("_S1Weak", "")
                
                out_name = f"{chip_id}_PreEvent.tif"
                out_path = self.config.output_dir / out_name
                
                # Force re-download if file is suspiciously small or missing
                # (Blank files are usually complete TIFs but all zeros, so we just force overwrite for now if needed,
                # or rely on the fact that we're re-running it.)
                if not out_path.exists():
                    self._download_image(image, chip_geom, out_path)
                else:
                    self.logger.info(f"  File exists: {out_name} (overwriting to ensure content)")
                    self._download_image(image, chip_geom, out_path)
        else:
            self.logger.info("  [DRY RUN] Would query GEE for pre-event imagery")
            self.stats["success"] += 1
    
    def _query_gee(self, event: str, region: object, start: datetime, end: datetime) -> Optional[object]:
        """Query Google Earth Engine for pre-event SAR. Returns median composite."""
        try:
            import ee
            
            # Query Sentinel-1 collection filtered by the SPECIFIC CHIP REGION
            collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(region)
                .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
            
            # Relax orbit/pass constraints for pre-event if needed, but try to keep pass if possible
            # For now, let's keep it simple and just use a median composite of what's available
            
            count = collection.size().getInfo()
            if count == 0:
                self.logger.warning(f"  No images found in window for {event} chip")
                return None
            
            self.logger.info(f"  Found {count} images for chip. Creating median composite.")
            
            # Use median composite to ensure full spatial coverage across the chip area
            # and reduce noise/transient effects. 
            # UPDATED: Explicitly select VV/VH and cast to float32 to match Sen1Floods11
            image = collection.select(["VV", "VH"]).median().float()
            
            return image
            
        except Exception as e:
            self.logger.error(f"  GEE query failed: {e}")
            self.stats["failed"] += 1
            return None

    def _download_image(self, image, region, filename):
        """Download image from GEE."""
        import requests
        try:
            url = image.getDownloadURL({
                'name': filename.stem,
                'scale': 10,
                'crs': 'EPSG:4326',
                'region': region,
                'format': 'GEO_TIFF'
            })
            self.logger.info(f"  Downloading {filename.name}...")
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                self.stats["success"] += 1
            else:
                self.logger.error(f"  Download failed: {r.status_code}")
                self.stats["failed"] += 1
        except Exception as e:
            self.logger.error(f"  Error getting download URL: {e}")
            self.stats["failed"] += 1
    
    def _try_fallback(self, event: str, meta: Dict) -> None:
        """Try fallback strategies for missing pre-event data."""
        self.logger.info("  Trying fallback strategies...")
        
        # Strategy 1: Extend search window
        self.logger.info("  - Strategy 1: Extend window to 60 days")
        
        # Strategy 2: Dry season baseline
        flood_date = datetime.strptime(meta["date"], "%Y-%m-%d")
        dry_season_start = flood_date.replace(month=1, day=1) - timedelta(days=180)
        self.logger.info(f"  - Strategy 2: Dry season ({dry_season_start.date()})")
        
        # Strategy 3: Accept different orbit
        self.logger.info("  - Strategy 3: Relax orbit matching (last resort)")
        
        self.stats["skipped"] += 1
    
    def _print_manual_instructions(self) -> None:
        """Print manual download instructions."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
        self.logger.info("=" * 60)
        self.logger.info("""
If GEE is unavailable, pre-event SAR can be downloaded from:

1. Copernicus Data Space (https://dataspace.copernicus.eu/)
   - Search for Sentinel-1 GRD
   - Filter by date, orbit, and bounding box
   - Download VV+VH products

2. Alaska Satellite Facility (https://search.asf.alaska.edu/)
   - Sentinel-1 archive
   - Free account required

For each flood event, you need:
   - Date range: 14-45 days BEFORE flood
   - Same orbit pass (ASCENDING or DESCENDING)
   - VV and VH polarizations
""")
        for event, meta in FLOOD_EVENTS.items():
            flood_date = datetime.strptime(meta["date"], "%Y-%m-%d")
            start = flood_date - timedelta(days=45)
            end = flood_date - timedelta(days=14)
            self.logger.info(f"\n{event}:")
            self.logger.info(f"   Search: {start.date()} to {end.date()}")
            self.logger.info(f"   Orbit: {meta['orbit_pass']}")
            self.logger.info(f"   BBox: {meta['bbox']}")
    
    def _report(self) -> None:
        """Print summary report."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PRE-EVENT SAR RETRIEVAL SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Success: {self.stats['success']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")


def main():
    parser = argparse.ArgumentParser(description="Download pre-event Sentinel-1 SAR")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--sen1floods11_dir", type=Path, default=Path("./data/sen1floods11"))
    parser.add_argument("--output", type=Path, default=Path("./data/pre_event_sar"))
    parser.add_argument("--source_dir", type=str, default="S1Hand", help="Subdirectory to scan (S1Hand or S1Weak)")
    
    args = parser.parse_args()
    
    config = PreEventConfig(
        mode=args.mode,
        sen1floods11_dir=args.sen1floods11_dir,
        output_dir=args.output,
        source_dir=args.source_dir
    )
    
    logger = setup_logging(args.output / "logs")
    downloader = PreEventSARDownloader(config, logger)
    
    success = downloader.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
