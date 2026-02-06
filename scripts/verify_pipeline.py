
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import Sen1Floods11Dataset
from src.model import SiameseResNetUNet

def verify_pipeline():
    print("="*60)
    print("🚀 PIPELINE VERIFICATION STARTED")
    print("="*60)

    # 1. Device Check
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"✅ Device: {device}")

    # 2. Dataset Sanity Check
    print("\n[Step 1] Loading Dataset Subset...")
    dataset = Sen1Floods11Dataset("./data", split="train")
    if len(dataset) == 0:
        raise ValueError("❌ Dataset is empty!")
    print(f"✅ Dataset size: {len(dataset)}")

    # Load 1 batch
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    
    img = batch["image"].to(device)
    pre = batch["pre_image"].to(device)
    infra = batch["infra"].to(device)
    label = batch["label"].to(device)

    print(f"\n[Step 2] Input Tensor Shapes:")
    print(f"Compare (Post): {img.shape} (Expected: B, 2, 512, 512)")
    print(f"Compare (Pre):  {pre.shape} (Expected: B, 2, 512, 512)")
    print(f"Compare (Infra):{infra.shape} (Expected: B, 2, 512, 512)")
    print(f"Compare (Label):{label.shape} (Expected: B, 1, 512, 512)")

    # Assertions
    assert img.shape == (2, 2, 512, 512), "❌ Post-SAR shape mismatch"
    assert pre.shape == (2, 2, 512, 512), "❌ Pre-SAR shape mismatch"
    assert infra.shape == (2, 2, 512, 512), "❌ Infra shape mismatch"
    assert label.shape == (2, 1, 512, 512), "❌ Label shape mismatch"
    print("✅ All Input Shapes Correct.")

    # 3. Model & Loss Check
    print("\n[Step 3] Model Forward Pass & Loss Logic...")
    model = SiameseResNetUNet(n_channels=2, n_classes=1).to(device)
    
    # Forward
    output = model(img, pre, infra)
    print(f"Output Shape: {output.shape} (Expected: B, 1, 512, 512)")
    assert output.shape == (2, 1, 512, 512), "❌ Output shape mismatch"
    
    # Value Range Check (for Sigmoid)
    min_val, max_val = output.min().item(), output.max().item()
    print(f"Output Range: [{min_val:.4f}, {max_val:.4f}]")
    if min_val >= 0 and max_val <= 1:
        print("✅ Output is in [0, 1] range (Sigmoid confirmed).")
    else:
        raise ValueError(f"❌ Output out of range [{min_val}, {max_val}]. Missing Sigmoid?")

    # Loss Calculation
    criterion = nn.BCELoss()
    loss = criterion(output, label)
    print(f"✅ Loss Calculated: {loss.item():.4f}")

    # 4. Visualization (Qualitative Check)
    print("\n[Step 4] saving visual proof to 'verification_sample.png'...")
    
    # helper to visualize SAR
    def to_img(tensor):
        # Taking VV channel (index 0)
        img = tensor[0, 0].cpu().numpy()
        # Normalize for viz [-20, 0] roughly to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img

    def to_mask(tensor):
        return tensor[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(to_img(pre), cmap='gray')
    axes[0].set_title("Pre-Event SAR (VV)")
    
    axes[1].imshow(to_img(img), cmap='gray')
    axes[1].set_title("Post-Event SAR (VV)")
    
    axes[2].imshow(to_mask(infra), cmap='jet')
    axes[2].set_title("Infrastructure (Roads)")
    
    axes[3].imshow(to_mask(label), cmap='Blues')
    axes[3].set_title("Ground Truth Label")
    
    plt.savefig("verification_sample.png")
    print("✅ Verification image saved.")

    print("\n" + "="*60)
    print("🎉 ALL SYSTEMS GO. TRAINING IS SAFE TO START.")
    print("="*60)

if __name__ == "__main__":
    verify_pipeline()
