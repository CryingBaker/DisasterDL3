import os
import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union

class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 flood segmentation task.
    
    Inputs:
        - Post-event SAR (VV, VH)
        - Pre-event SAR (VV, VH) [Optional]
        - Infrastructure (Roads, Buildings) [Optional]
    
    Target:
        - Flood mask (0: Non-flood, 1: Flood)
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform = None,
        use_pre_event: bool = True,
        use_infrastructure: bool = True,
        require_complete: bool = False
    ):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_pre_event = use_pre_event
        self.use_infrastructure = use_infrastructure
        self.require_complete = require_complete
        
        # Paths
        self.s1_dir = self.root / "sen1floods11" / "S1Hand"
        self.label_dir = self.root / "sen1floods11" / "LabelHand"
        self.pre_dir = self.root / "pre_event_sar"  # Fixed to match downloader output
        self.infra_dir = self.root / "infrastructure" / "rasterized"
        
        # Load sample list from splits
        self.samples = self._load_split_file()
        
        # Verification
        self._verify_files()
        
    def _load_split_file(self) -> List[str]:
        """Load list of sample IDs from split CSVs."""
        split_file = self.root / "sen1floods11" / "splits" / "flood_handlabeled" / f"flood_{self.split}_data.csv"
        
        # If specific split file doesn't exist (e.g. fast mode might not have full splits),
        # fallback to finding intersection of available files
        if not split_file.exists():
            print(f"Warning: Split file {split_file} not found. Using all available paired files.")
            return self._scan_available_files()
            
        with open(split_file, "r") as f:
            # Skip header if present, assume single column of IDs
            lines = f.read().splitlines()
            # Filter out header if it looks like one "image_name"
            if lines and "image_name" in lines[0]:
                lines = lines[1:]
        return lines

    def _scan_available_files(self) -> List[str]:
        """Fallback: scan directories for matching pairs."""
        s1_files = {f.name.replace("_S1Hand.tif", "") for f in self.s1_dir.glob("*_S1Hand.tif")}
        label_files = {f.name.replace("_LabelHand.tif", "") for f in self.label_dir.glob("*_LabelHand.tif")}
        common = sorted(list(s1_files.intersection(label_files)))
        
        # Simple split for checking
        if self.split == "train":
            return common[:int(0.8*len(common))]
        elif self.split == "valid":
            return common[int(0.8*len(common)):]
        else:
            return common

    def _verify_files(self):
        """Check if required files exist."""
        valid_samples = []
        for sample_id in self.samples:
            s1_path = self.s1_dir / f"{sample_id}_S1Hand.tif"
            label_path = self.label_dir / f"{sample_id}_LabelHand.tif"
            
            # Base requirements
            if not (s1_path.exists() and label_path.exists()):
                continue

            # Strict mode checks
            if self.require_complete:
                pre_path = self.pre_dir / f"{sample_id}_PreEvent.tif"
                infra_path = self.infra_dir / f"{sample_id}_Infra.tif"
                
                if self.use_pre_event and not pre_path.exists():
                    continue
                if self.use_infrastructure and not infra_path.exists():
                    continue
            
            valid_samples.append(sample_id)
        
        print(f"Dataset ({self.split}, strict={self.require_complete}): {len(valid_samples)}/{len(self.samples)} valid samples found.")
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # 1. Load Post-Event SAR
        post_sar = self._load_tif(self.s1_dir / f"{sample_id}_S1Hand.tif") # (2, H, W)
        
        # 2. Load Label
        label = self._load_tif(self.label_dir / f"{sample_id}_LabelHand.tif") # (1, H, W)
        label = (label == 1).astype(np.float32) # Convert to 0/1 float
        
        # 3. Load Pre-Event SAR (Optional)
        if self.use_pre_event:
            # Try to find matching pre-event file
            pre_path = self.pre_dir / f"{sample_id}_PreEvent.tif"
            if pre_path.exists():
                pre_sar = self._load_tif(pre_path)
                # Enforce 2 channels (VV, VH)
                if pre_sar.shape[0] > 2:
                    pre_sar = pre_sar[:2, :, :]
                
                # Robustness Check: Handle potential 1-pixel GEE export differences
                # Post-event is the anchor (512x512). Pre-event might be 513x513.
                if pre_sar.shape != post_sar.shape:
                    # If mismatch is small, just center crop or resize
                    # Given geospatial bounds match, simple resize (interpolation) or crop is valid.
                    # Since these are numpy arrays (C, H, W)
                    tgt_h, tgt_w = post_sar.shape[1], post_sar.shape[2]
                    
                    # Convert to tensor for easy interpolate, or use opencv/skimage?
                    # Let's keep dependencies low. Simple crop if larger, pad if smaller (unlikely).
                    # Actually, usually it's just +1 pixel. Slicing is safest.
                    curr_h, curr_w = pre_sar.shape[1], pre_sar.shape[2]
                    
                    if curr_h >= tgt_h and curr_w >= tgt_w:
                        pre_sar = pre_sar[:, :tgt_h, :tgt_w]
                    else:
                        # Fallback for smaller images: Pad
                        pad_h = max(0, tgt_h - curr_h)
                        pad_w = max(0, tgt_w - curr_w)
                        pre_sar = np.pad(pre_sar, ((0,0), (0, pad_h), (0, pad_w)), mode='edge')
                        # Ensure exact match after padding
                        pre_sar = pre_sar[:, :tgt_h, :tgt_w]
                        
            else:
                # Zero-fill or copy post-event if missing (ablation/fallback)
                pre_sar = np.zeros_like(post_sar)
        else:
            pre_sar = np.zeros((0, *post_sar.shape[1:]), dtype=np.float32) # Empty channel dim
            
        # 4. Load Infrastructure (Optional)
        if self.use_infrastructure:
            infra_path = self.infra_dir / f"{sample_id}_Infra.tif"
            if infra_path.exists():
                infra = self._load_tif(infra_path)
            else:
                infra = np.zeros((2, *post_sar.shape[1:]), dtype=np.float32) # 2 channels (road, building)
        else:
            infra = np.zeros((2, *post_sar.shape[1:]), dtype=np.float32)

        # Normalize SAR (Log scale for intensity)
        post_sar = np.clip(post_sar, -50, 0) # Clip dB range (approx)
        post_sar = (post_sar + 50) / 50.0   # Normalize to [0, 1]
        
        if self.use_pre_event and pre_sar.shape[0] > 0:
             pre_sar = np.clip(pre_sar, -50, 0)
             pre_sar = (pre_sar + 50) / 50.0

        # Create sample dict
        sample = {
            "image": post_sar,       # Main input
            "label": label,          # Target
            "pre_image": pre_sar,    # Aux input 1
            "infra": infra,          # Aux input 2
            "id": sample_id
        }

        if self.transform:
            sample = self.transform(sample)

        # Convert to tensors
        sample["image"] = torch.from_numpy(sample["image"]).float()
        sample["label"] = torch.from_numpy(sample["label"]).float()
        if self.use_pre_event:
            sample["pre_image"] = torch.from_numpy(sample["pre_image"]).float()
        if self.use_infrastructure:
            sample["infra"] = torch.from_numpy(sample["infra"]).float()
            
        return sample

    def _load_tif(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)
