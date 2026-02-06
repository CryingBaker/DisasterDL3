
import sys
import os
sys.path.append(os.getcwd())
import torch
from src.dataset import Sen1Floods11Dataset
from src.augmentations import get_val_transforms

def verify():
    print("Verifying Sen1Floods11Dataset with use_weak=True...")
    
    # Initialize dataset
    try:
        ds = Sen1Floods11Dataset(
            "./data",
            split="train",
            use_weak=True,
            transform=get_val_transforms(),
            require_complete=False, # Weak mode
            use_infrastructure=False,
            use_pre_event=False # Simplify check
        )
        
        print(f"Dataset size: {len(ds)}")
        
        # Check if size matches expectation (Hand + Weak - Bad)
        # Hand=446 (approx 330 train)
        # Weak=4371 (valid)
        # Total should be around 4700
        
        if len(ds) > 4000:
            print("SUCCESS: Loaded correct number of samples.")
        else:
            print(f"WARNING: Dataset size {len(ds)} seems low.")
            
        # Try loading one sample
        if len(ds) > 0:
            sample = ds[0]
            print(f"Sample 0 ID: {sample['id']}")
            print(f"Image shape: {sample['image'].shape}")
            if torch.isnan(sample['image']).any():
                print("FAILURE: NaN found in loaded sample!")
            else:
                print("SUCCESS: Sample loaded without NaNs.")
                
    except Exception as e:
        print(f"FAILURE: Dataset init failed: {e}")
        raise e

if __name__ == "__main__":
    verify()
