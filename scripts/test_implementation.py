import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.dataset import Sen1Floods11Dataset
from src.model import SiameseResNetUNet

def test_pipeline():
    print("========================================")
    print("      Testing Implementation Pipeline   ")
    print("========================================")
    
    # 1. Test Dataset Loading
    print("\n1. Initializing Dataset...")
    data_dir = "./data"
    
    # We expect some missing files (PreEvent, Infra), dataset should handle it
    dataset = Sen1Floods11Dataset(data_dir, split="train")
    
    if len(dataset) == 0:
        print("ERROR: No samples found in dataset!")
        return
        
    print(f"   found {len(dataset)} samples.")
    
    # 2. Test Sample Fetching
    print("\n2. Fetching first sample...")
    try:
        sample = dataset[0]
        img = sample["image"]
        label = sample["label"]
        pre_img = sample["pre_image"]
        infra = sample["infra"]
        
        print(f"   Image shape: {img.shape} (Post-Event SAR)")
        print(f"   Label shape: {label.shape} (Flood Mask)")
        print(f"   Pre-Event shape: {pre_img.shape}")
        print(f"   Infra shape: {infra.shape}")
        
        # Verify shapes
        assert img.shape[0] == 2, "Image should have 2 channels (VV, VH)"
        assert label.shape[0] == 1, "Label should have 1 channel"
        
        # Check pre-event handling (might be empty/zero if missing)
        if pre_img.shape[0] > 0:
            assert pre_img.shape == img.shape, "Pre-event should match post-event shape"
        else:
            print("   (Pre-Event SAR missing - verified graceful handling)")
            
    except Exception as e:
        print(f"ERROR fetching sample: {e}")
        return

    # 3. Test Model Initialization
    print("\n3. Initializing Model...")
    try:
        model = SiameseResNetUNet(n_channels=2, n_classes=1)
        print("   Model initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing model: {e}")
        return

    # 4. Test Forward Pass
    print("\n4. Running Forward Pass...")
    try:
        # Batch dimension
        b_img = img.unsqueeze(0)
        
        # Handle optional inputs for model
        if pre_img.shape[0] == 0:
            print("   Synthesizing zero-filled pre-event input for model test...")
            b_pre = torch.zeros_like(b_img)
        else:
            b_pre = pre_img.unsqueeze(0)
            
        if infra.shape[0] == 0:
             print("   Synthesizing zero-filled infra input for model test...")
             b_infra = torch.zeros((1, 2, b_img.shape[2], b_img.shape[3]))
        else:
            b_infra = infra.unsqueeze(0)
            
        output = model(b_img, b_pre, b_infra)
        print(f"   Output shape: {output.shape}")
        
        assert output.shape == (1, 1, 512, 512), f"Expected (1, 1, 512, 512), got {output.shape}"
        print("✅ Forward pass successful!")
        
    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
