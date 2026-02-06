import os
import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
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
    
    Features:
        - Filters out samples with black pixels (invalid SAR data)
        - Filters out samples with NaN/Inf values
        - Supports oversampling of heavily flooded images
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform = None,
        use_pre_event: bool = True,
        use_infrastructure: bool = True,
        require_complete: bool = False,
        min_flood_pixels: int = 0,
        oversample_flood: bool = False,  # NEW: oversample flooded samples
        max_black_ratio: float = 0.05,   # NEW: max 5% black pixels allowed
        use_weak: bool = False           # NEW: include weakly labeled data
    ):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_pre_event = use_pre_event
        self.use_infrastructure = use_infrastructure
        self.require_complete = require_complete
        self.min_flood_pixels = min_flood_pixels
        self.oversample_flood = oversample_flood
        self.max_black_ratio = max_black_ratio
        self.use_weak = use_weak
        
        # Paths
        self.s1_dir = self.root / "sen1floods11" / "S1Hand"
        self.label_dir = self.root / "sen1floods11" / "LabelHand"
        self.s1_weak_dir = self.root / "sen1floods11" / "S1Weak"
        self.label_weak_dir = self.root / "sen1floods11" / "LabelWeak"
        self.pre_dir = self.root / "pre_event_sar"
        self.infra_dir = self.root / "infrastructure" / "rasterized"
        
        # Load sample list from splits
        self.samples = self._load_split_file()
        
        # Verification and filtering
        self._verify_files()
        
        # Calculate flood percentages for oversampling
        if self.oversample_flood and self.split == "train":
            self._setup_oversampling()
        else:
            self.sample_weights = None

    def _load_split_file(self) -> List[str]:
        """Load train/valid/test split from CSV file."""
        sample_ids = []
        # Fix path to point to the actual split directory in the dataset
        split_file = self.root / "sen1floods11" / "splits" / "splits" / "flood_handlabeled" / f"flood_{self.split}_data.csv"
        
        if split_file.exists():
            with open(split_file, "r") as f:
                lines = f.read().splitlines()
                
            for line in lines:
                # Handle CSV format: "Ghana_103272_S1Hand.tif,Ghana_103272_LabelHand.tif"
                if "," in line:
                    first_col = line.split(",")[0].strip()
                    if first_col.endswith("_S1Hand.tif"):
                        sample_id = first_col.replace("_S1Hand.tif", "")
                        sample_ids.append(sample_id)
                elif line.endswith("_S1Hand.tif"):
                    sample_ids.append(line.replace("_S1Hand.tif", ""))
                elif "image_name" not in line.lower() and line.strip():
                    sample_ids.append(line.strip())
        
        # 3. Add Weakly Labeled (specifically for training) if enabled
        if self.use_weak and self.split == "train":
            s1_weak = {f.name.replace("_S1Weak.tif", "") for f in self.s1_weak_dir.glob("*_S1Weak.tif")}
            sample_ids.extend(sorted(list(s1_weak)))
            print(f"  Added {len(s1_weak)} weak samples to training set.")
                
        return sample_ids

    def _scan_available_files(self) -> List[str]:
        """Fallback: scan directories for matching pairs."""
        # 1. Hand Labeled
        s1_files = {f.name.replace("_S1Hand.tif", "") for f in self.s1_dir.glob("*_S1Hand.tif")}
        label_files = {f.name.replace("_LabelHand.tif", "") for f in self.label_dir.glob("*_LabelHand.tif")}
        common = sorted(list(s1_files.intersection(label_files)))
        
        # 2. Weakly Labeled (if enabled)
        if self.use_weak:
            s1_weak = {f.name.replace("_S1Weak.tif", "") for f in self.s1_weak_dir.glob("*_S1Weak.tif")}
            label_weak = {f.name.replace("_LabelWeak.tif", "") for f in self.label_weak_dir.glob("*_LabelWeak.tif")}
            # Note: label files might be named differently (S1OtsuLabelWeak) - check download script
            # In download script we save to LabelWeak directory.
            # Filenames are typically same but with suffix. 
            # If download script kept original filenames, they might be *LabelWeak.tif.
            # Assuming our download script handles this or files match ID.
            
            # Let's perform a loose match if needed, but for now assuming suffix consistency
            # Scan destination:
            common_weak = sorted(list(s1_weak))
            # Just take all S1Weak that exist, assuming labels exist (we verify later)
            common.extend(common_weak)
            print(f"Added {len(common_weak)} weak samples.")

        if self.split == "train":
            # For weak data, we usually put all in train, or split same way?
            # Standard: Split everything 80/20 or use Weak only for train
            # Here keeping simple: 80/20 on everything
            return common[:int(0.8*len(common))]
        elif self.split == "valid":
            return common[int(0.8*len(common)):]
        else:
            return common

    def _verify_files(self):
        """Check if required files exist and data is valid (no NaN, no black pixels)."""
        valid_samples = []
        nan_count = 0
        black_count = 0
        
        for sample_id in self.samples:
            # Determine paths (Hand vs Weak)
            if (self.s1_dir / f"{sample_id}_S1Hand.tif").exists():
                s1_path = self.s1_dir / f"{sample_id}_S1Hand.tif"
                label_path = self.label_dir / f"{sample_id}_LabelHand.tif"
            elif self.use_weak and (self.s1_weak_dir / f"{sample_id}_S1Weak.tif").exists():
                s1_path = self.s1_weak_dir / f"{sample_id}_S1Weak.tif"
                # Handle label name - could be _LabelWeak or _S1OtsuLabelWeak depending on download
                # Try simple first
                label_path = self.label_weak_dir / f"{sample_id}_LabelWeak.tif"
                if not label_path.exists():
                     # Try original name if copied directly
                     label_path = self.label_weak_dir / f"{sample_id}_S1OtsuLabelWeak.tif" 
            else:
                continue

            # DEBUG TRACE
            if len(valid_samples) % 100 == 0:
                print(f"  Verifying sample {len(valid_samples)}/{len(self.samples)}: {sample_id}...", end='\r')

            # Base requirements
            if not (s1_path.exists() and label_path.exists()):
                continue
                
            # Check S1 file size (skip small/empty S1 files - likely corrupt)
            if s1_path.stat().st_size < 50000:
                continue

            # Verify checking label validity by opening it (size check is unsafe for labels)
            try:
                with rasterio.open(label_path) as src:
                    pass
            except Exception:
                continue

            # Check SAR data quality
            try:
                with rasterio.open(s1_path) as src:
                    sar_data = src.read()
                    total_pixels = sar_data.size
                    
                    # Check for NaN/Inf
                    if np.isnan(sar_data).any() or np.isinf(sar_data).any():
                        nan_count += 1
                        continue
                    
                    # NEW: Check for black pixels (zeros or very low values)
                    # SAR data in dB, black pixels are typically exact 0 or very negative
                    black_pixels = np.sum(sar_data == 0) + np.sum(sar_data < -60)
                    black_ratio = black_pixels / total_pixels
                    
                    if black_ratio > self.max_black_ratio:
                        black_count += 1
                        continue
                        
            except Exception:
                continue

            # Strict mode checks
            if self.require_complete:
                pre_path = self.pre_dir / f"{sample_id}_PreEvent.tif"
                infra_path = self.infra_dir / f"{sample_id}_Infra.tif"
                
                if self.use_pre_event and not pre_path.exists():
                    continue
                if self.use_infrastructure and not infra_path.exists():
                    continue
            
            # Filter by flood pixel count
            if self.min_flood_pixels > 0:
                with rasterio.open(label_path) as src:
                    lbl = src.read(1)
                    if (lbl == 1).sum() < self.min_flood_pixels:
                        continue
            
            valid_samples.append(sample_id)
        
        if nan_count > 0:
            print(f"  Skipped {nan_count} samples with NaN/Inf in SAR data.")
        if black_count > 0:
            print(f"  Skipped {black_count} samples with >{self.max_black_ratio*100:.0f}% black pixels.")
        print(f"Dataset ({self.split}, strict={self.require_complete}, min_flood={self.min_flood_pixels}): {len(valid_samples)}/{len(self.samples)} valid samples found.")
        self.samples = valid_samples

    def _setup_oversampling(self):
        """Calculate weights for oversampling heavily flooded samples."""
        print("  Setting up flood-based oversampling...")
        weights = []
        flood_percentages = []
        
        for sample_id in self.samples:
            # Determine correct label path
            label_path = self.label_dir / f"{sample_id}_LabelHand.tif"
            if not label_path.exists() and self.use_weak:
                label_path = self.label_weak_dir / f"{sample_id}_LabelWeak.tif"
                if not label_path.exists():
                    label_path = self.label_weak_dir / f"{sample_id}_S1OtsuLabelWeak.tif"
            
            try:
                with rasterio.open(label_path) as src:
                    lbl = src.read(1)
                    flood_pct = (lbl == 1).sum() / lbl.size
                    flood_percentages.append(flood_pct)
            except Exception:
                # Should have been verified, but just in case
                flood_percentages.append(0.0)
        
        # Weight samples by flood percentage (more flood = higher weight)
        # Scale: samples with >20% flood get 3x weight, >10% get 2x, others get 1x
        for pct in flood_percentages:
            if pct > 0.20:
                weights.append(4.0)  # Heavy flood: 4x
            elif pct > 0.10:
                weights.append(2.5)  # Moderate flood: 2.5x
            elif pct > 0.05:
                weights.append(1.5)  # Light flood: 1.5x
            else:
                weights.append(1.0)  # Minimal: 1x
        
        self.sample_weights = weights
        avg_pct = np.mean(flood_percentages) * 100
        print(f"  Avg flood coverage: {avg_pct:.1f}%. Max weight: {max(weights):.1f}x")
    
    def get_sampler(self):
        """Get weighted sampler for DataLoader to oversample flooded images."""
        if self.sample_weights is None:
            return None
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.samples) * 2,  # Double epoch size via oversampling
            replacement=True
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # 1. Load Post-Event SAR
        # Determine path (Hand vs Weak)
        s1_path = self.s1_dir / f"{sample_id}_S1Hand.tif"
        if not s1_path.exists() and self.use_weak:
            s1_path = self.s1_weak_dir / f"{sample_id}_S1Weak.tif"
            
        post_sar = self._load_tif(s1_path)
        
        # 2. Load Label
        label_path = self.label_dir / f"{sample_id}_LabelHand.tif"
        if not label_path.exists() and self.use_weak:
            # Try LabelWeak
            label_path = self.label_weak_dir / f"{sample_id}_LabelWeak.tif"
            if not label_path.exists():
                label_path = self.label_weak_dir / f"{sample_id}_S1OtsuLabelWeak.tif"
        
        label = self._load_tif(label_path)
        label = (label == 1).astype(np.float32)
        
        # 3. Load Pre-Event SAR (Optional)
        if self.use_pre_event:
            pre_path = self.pre_dir / f"{sample_id}_PreEvent.tif"
            if pre_path.exists():
                try:
                    pre_sar = self._load_tif(pre_path)
                    if pre_sar.shape[0] > 2:
                        pre_sar = pre_sar[:2, :, :]
                    
                    # Handle size mismatch
                    if pre_sar.shape != post_sar.shape:
                        tgt_h, tgt_w = post_sar.shape[1], post_sar.shape[2]
                        curr_h, curr_w = pre_sar.shape[1], pre_sar.shape[2]
                        
                        if curr_h >= tgt_h and curr_w >= tgt_w:
                            pre_sar = pre_sar[:, :tgt_h, :tgt_w]
                        else:
                            pad_h = max(0, tgt_h - curr_h)
                            pad_w = max(0, tgt_w - curr_w)
                            pre_sar = np.pad(pre_sar, ((0,0), (0, pad_h), (0, pad_w)), mode='edge')
                            pre_sar = pre_sar[:, :tgt_h, :tgt_w]
                except Exception:
                    # Fallback for corrupt files
                    pre_sar = np.zeros_like(post_sar)
            else:
                pre_sar = np.zeros_like(post_sar)
        else:
            pre_sar = np.zeros((0, *post_sar.shape[1:]), dtype=np.float32)
            
        # 4. Load Infrastructure (Optional)
        if self.use_infrastructure:
            infra_path = self.infra_dir / f"{sample_id}_Infra.tif"
            if infra_path.exists():
                infra = self._load_tif(infra_path)
            else:
                infra = np.zeros((2, *post_sar.shape[1:]), dtype=np.float32)
        else:
            infra = np.zeros((2, *post_sar.shape[1:]), dtype=np.float32)

        # Normalize SAR data
        post_sar = np.clip(post_sar, -50, 20) 
        post_sar = (post_sar + 50) / 70.0
        
        if self.use_pre_event and pre_sar.shape[0] > 0:
            pre_sar = np.clip(pre_sar, -50, 20)
            pre_sar = (pre_sar + 50) / 70.0

        # Create sample dict
        sample = {
            "image": post_sar,
            "label": label,
            "pre_image": pre_sar,
            "infra": infra,
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
            data = src.read().astype(np.float32)
            
            # Normalize SAR data (dB range is ~[-50, 10])
            # Labels should NOT be normalized (they are 0 or 1)
            filename = path.name.lower()
            if "label" not in filename:
                # Use a standard clipping range for Sentinel-1 dB
                # -30 is typical floor for SAR, 0 is typical ceiling for non-specular
                data = np.clip(data, -30, 0)
                data = (data + 30) / 30.0
                
            return data
