#!/usr/bin/env python3
"""
OpenStreetMap Infrastructure Downloader
========================================
Downloads and rasterizes road/building data for flood segmentation context.

Usage:
    python download_infrastructure_osm.py --mode fast --sen1floods11_dir ./data/sen1floods11
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"download_osm_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logger = logging.getLogger("osm_downloader")
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
class OSMConfig:
    """Configuration for OSM data download."""
    mode: str = "fast"
    sen1floods11_dir: Path = Path("./data/sen1floods11")
    output_dir: Path = Path("./data/infrastructure")
    
    # Target resolution (match SAR)
    resolution_m: float = 10.0
    chip_size: int = 512
    
    # OSM data types
    download_roads: bool = True
    download_buildings: bool = True
    
    # Overpass API settings
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    timeout: int = 300


# Event bounding boxes (same as pre-event SAR)
# Event bounding boxes (same as pre-event SAR)
EVENT_BBOXES = {
    "Bolivia": [-65.64, -15.96, -64.36, -11.39],
    "Ghana": [-2.30, 6.30, 0.22, 11.93],
    "India": [92.15, 24.85, 94.16, 28.28],
    "Mekong": [104.07, 10.57, 106.43, 14.27],
    "Nigeria": [4.53, 4.12, 6.95, 10.54],
    "Pakistan": [68.99, 28.04, 72.55, 34.35],
    "Paraguay": [-58.59, -28.19, -54.65, -21.64],
    "Somalia": [44.20, 1.31, 46.58, 6.55],
    "Spain": [-1.11, 37.66, 1.22, 39.41],
    "Sri-Lanka": [80.13, 5.14, 82.09, 9.79],
    "USA": [-95.69, 38.36, -94.36, 41.06],
}


class OSMDownloader:
    """Downloads and rasterizes OSM infrastructure data."""
    
    def __init__(self, config: OSMConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.stats = {"roads": 0, "buildings": 0, "failed": 0}
    
    def run(self) -> bool:
        """Execute OSM download pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("OSM Infrastructure Downloader")
        self.logger.info("=" * 60)
        
        # Check dependencies
        if not self._check_dependencies():
            self._print_manual_instructions()
            return False
        
        # Get regions to download
        regions = self._get_regions()
        self.logger.info(f"Regions to process: {len(regions)}")
        
        # Download each region
        for region in regions:
            try:
                self._process_region(region)
            except Exception as e:
                self.logger.error(f"Failed to process region {region['event']}: {str(e)}")
                self.stats["failed"] += 1
        
        self._report()
        return self.stats["failed"] == 0
    
    def _check_dependencies(self) -> bool:
        """Check for required dependencies."""
        try:
            import requests
            import rasterio
            from shapely.geometry import shape
            return True
        except ImportError as e:
            self.logger.error(f"Missing dependency: {e}")
            self.logger.info("Install: pip install requests rasterio shapely")
            return False
    
    def _get_regions(self) -> List[Dict]:
        """Get regions to download."""
        # Always return all events for full mode request
        events = list(EVENT_BBOXES.keys())
        return [{"event": e, "bbox": EVENT_BBOXES[e]} for e in events]
    
    def _process_region(self, region: Dict) -> None:
        """Download and rasterize OSM data for a region."""
        event = region["event"]
        bbox = region["bbox"]
        
        self.logger.info(f"\nProcessing: {event}")
        self.logger.info(f"  BBox: {bbox}")
        
        output_dir = self.config.output_dir / event
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download roads
        if self.config.download_roads:
            roads_file = output_dir / "roads.geojson"
            if roads_file.exists():
                self.logger.info("  Roads: Already exists, skipping")
            else:
                self._download_osm_layer(bbox, "highway", roads_file)
        
        # Download buildings
        if self.config.download_buildings:
            buildings_file = output_dir / "buildings.geojson"
            if buildings_file.exists():
                self.logger.info("  Buildings: Already exists, skipping")
            else:
                self._download_osm_layer(bbox, "building", buildings_file)
        
        # Rasterize
        self.logger.info("  Rasterization: Creating aligned binary masks...")
        self._rasterize_vectors(output_dir, event)

    def _rasterize_vectors(self, output_dir: Path, event: str) -> None:
        """Rasterize vector files to match SAR chip specifications."""
        try:
            import rasterio
            from rasterio import features
            from rasterio.transform import from_bounds
            import numpy as np
            import json
            
            # Find a reference SAR chip to get dimensions/transform
            # We assume S1Hand exists. If not, we can't align perfectly yet.
            # But we can approximate from bbox and resolution.
            # Ideally, we loop through all chips for this event.
            
            # Alternative strategy: 
            # 1. We don't have a single "master" raster for the whole event usually in this dataset structure.
            # 2. But we have downloaded huge GeoJSONs covering the whole event bbox.
            # 3. We actually need to cut these into chips matching the S1Hand/LabelHand chips.
            
            # Let's verify how dataset.py expects them. 
            # dataset.py expects: root/infrastructure/rasterized/{sample_id}_Infra.tif
            # This is chip-level!
            
            # So this function needs to:
            # 1. Load the big GeoJSONs (Roads, Buildings)
            # 2. Iterate over all existing SAR chips for this event
            # 3. For each chip:
            #    a. Read SAR transform/bounds
            #    b. Rasterize vectors into that window
            #    c. Save as {sample_id}_Infra.tif
            
            s1_dir = self.config.sen1floods11_dir / "S1Hand"
            infra_raster_dir = self.config.output_dir / "rasterized"
            infra_raster_dir.mkdir(parents=True, exist_ok=True)
            
            # Load vectors once
            roads = []
            buildings = []
            
            roads_file = output_dir / "roads.geojson"
            if roads_file.exists():
                with open(roads_file) as f:
                    roads = json.load(f).get("features", [])
            
            buildings_file = output_dir / "buildings.geojson"
            if buildings_file.exists():
                with open(buildings_file) as f:
                    buildings = json.load(f).get("features", [])
                    
            if not roads and not buildings:
                self.logger.warning(f"No vectors found for {event}, skipping rasterization")
                return

            # Filter chips for this event
            # Filename format: {Event}_{ID}_S1Hand.tif
            chips = list(s1_dir.glob(f"{event}*_S1Hand.tif"))
            self.logger.info(f"  Found {len(chips)} SAR chips to align with.")
            
            for chip_path in chips:
                sample_id = chip_path.name.replace("_S1Hand.tif", "")
                out_path = infra_raster_dir / f"{sample_id}_Infra.tif"
                
                if out_path.exists():
                    continue
                    
                with rasterio.open(chip_path) as src:
                    shape = src.shape
                    transform = src.transform
                    
                    # Create masks
                    # Channel 0: Roads
                    # Channel 1: Buildings
                    
                    # Optimization: Filter coords roughly? 
                    # rasterize() is relatively fast if we pass all shapes, but slow for huge lists.
                    # For now, pass all.
                    
                    mask = np.zeros((2, *shape), dtype=np.uint8)
                    
                    if roads:
                        # Extract geometries
                        road_geoms = [
                            (s, 1) for s in [f["geometry"] for f in roads] 
                            if s is not None
                        ]
                        if road_geoms:
                             try:
                                road_mask = features.rasterize(
                                    road_geoms,
                                    out_shape=shape,
                                    transform=transform,
                                    fill=0,
                                    default_value=1,
                                    dtype=np.uint8
                                )
                                mask[0] = road_mask
                             except Exception as e:
                                 self.logger.warning(f"Failed to rasterize roads for {sample_id}: {e}")

                    if buildings:
                        building_geoms = [
                            (s, 1) for s in [f["geometry"] for f in buildings]
                            if s is not None
                        ]
                        if building_geoms:
                            try:
                                building_mask = features.rasterize(
                                    building_geoms,
                                    out_shape=shape,
                                    transform=transform,
                                    fill=0,
                                    default_value=1,
                                    dtype=np.uint8
                                )
                                mask[1] = building_mask
                            except Exception as e:
                                self.logger.warning(f"Failed to rasterize buildings for {sample_id}: {e}")

                    # Save
                    with rasterio.open(
                        out_path,
                        'w',
                        driver='GTiff',
                        height=shape[0],
                        width=shape[1],
                        count=2,
                        dtype='uint8',
                        crs=src.crs,
                        transform=transform,
                    ) as dst:
                        dst.write(mask)
                        
            self.logger.info(f"  Rasterized {len(chips)} chips for {event}")

        except Exception as e:
            self.logger.error(f"Rasterization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _download_osm_layer(self, bbox: List[float], tag: str, output_file: Path) -> bool:
        """Download a single OSM layer via Overpass API."""
        self.logger.info(f"  Downloading {tag}...")
        
        # Overpass query
        query = f"""
        [out:json][bbox:{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}];
        (
          way["{tag}"];
          relation["{tag}"];
        );
        out geom;
        """
        
        try:
            import requests
            
            response = requests.post(
                self.config.overpass_url,
                data={"data": query},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            feature_count = len(data.get("elements", []))
            self.logger.info(f"  Downloaded {feature_count} {tag} features")
            
            # Convert to GeoJSON and save
            geojson = self._osm_to_geojson(data)
            with open(output_file, 'w') as f:
                json.dump(geojson, f)
            
            if tag == "highway":
                self.stats["roads"] += feature_count
            else:
                self.stats["buildings"] += feature_count
            
            return True
            
        except Exception as e:
            self.logger.error(f"  Failed to download {tag}: {e}")
            self.stats["failed"] += 1
            return False
    
    def _osm_to_geojson(self, osm_data: Dict) -> Dict:
        """Convert Overpass response to GeoJSON."""
        features = []
        
        for element in osm_data.get("elements", []):
            if element["type"] == "way" and "geometry" in element:
                coords = [[g["lon"], g["lat"]] for g in element["geometry"]]
                feature = {
                    "type": "Feature",
                    "properties": element.get("tags", {}),
                    "geometry": {
                        "type": "LineString" if len(coords) < 3 or coords[0] != coords[-1] else "Polygon",
                        "coordinates": coords if len(coords) < 3 else [coords]
                    }
                }
                features.append(feature)
        
        return {"type": "FeatureCollection", "features": features}
    
    def _print_manual_instructions(self) -> None:
        """Print manual download instructions."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MANUAL OSM DOWNLOAD INSTRUCTIONS")
        self.logger.info("=" * 60)
        self.logger.info("""
If API download fails, use Geofabrik extracts:

1. Visit https://download.geofabrik.de/
2. Download .shp.zip for relevant regions
3. Extract roads and buildings shapefiles
4. Rasterize using GDAL:

   gdal_rasterize -burn 1 -tr 10 10 -ot Byte \\
       roads.shp roads_mask.tif

WARNING: OSM coverage varies by region!
- Europe/USA: Excellent coverage
- Africa/Asia: Variable, may be sparse
- Handle missing data gracefully (zero-fill)
""")
    
    def _report(self) -> None:
        """Print summary report."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("OSM DOWNLOAD SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Road features: {self.stats['roads']}")
        self.logger.info(f"Building features: {self.stats['buildings']}")
        self.logger.info(f"Failed downloads: {self.stats['failed']}")
        
        if self.stats['roads'] == 0 and self.stats['buildings'] == 0:
            self.logger.warning("No data downloaded - check network/API availability")


def main():
    parser = argparse.ArgumentParser(description="Download OSM infrastructure data")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--sen1floods11_dir", type=Path, default=Path("./data/sen1floods11"))
    parser.add_argument("--output", type=Path, default=Path("./data/infrastructure"))
    
    args = parser.parse_args()
    
    config = OSMConfig(
        mode=args.mode,
        sen1floods11_dir=args.sen1floods11_dir,
        output_dir=args.output
    )
    
    logger = setup_logging(args.output / "logs")
    downloader = OSMDownloader(config, logger)
    
    success = downloader.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
