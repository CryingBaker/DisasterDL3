
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
try:
    from src.dataset import Sen1Floods11Dataset
    from src.model import SiameseResNetUNet
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def dry_run():
    print("="*60)
    print("🚀 FINAL PRE-FLIGHT CHECK: DRY RUN")
    print("="*60)
    
    # 1. Load Data (Strict Mode + No Infra)
    print("[1/4] Loading Dataset (Strict=True, Infra=False)...")
    ds = Sen1Floods11Dataset("./data", split="train", require_complete=True, use_infrastructure=False)
    
    if len(ds) == 0:
        print("❌ ERROR: Dataset is empty!")
        return
        
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(f"✅ Batch Loaded: Post:{batch['image'].shape}, Pre:{batch['pre_image'].shape}, Label:{batch['label'].shape}")
    
    # 2. Load Model
    print("\n[2/4] Initializing Model...")
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = SiameseResNetUNet(n_channels=2, n_classes=1).to(DEVICE)
    print(f"✅ Model on {DEVICE}")
    
    # 3. Forward Pass
    print("\n[3/4] Testing Forward Pass...")
    img = batch['image'].to(DEVICE)
    pre = batch['pre_image'].to(DEVICE)
    infra = batch['infra'].to(DEVICE)
    label = batch['label'].to(DEVICE)
    
    output = model(img, pre, infra)
    print(f"✅ Output Shape: {output.shape} (Expected: {label.shape})")
    
    # 4. Backward Pass
    print("\n[4/4] Testing Backward Pass (Learning)...")
    criterion = torch.nn.BCELoss()
    loss = criterion(output, label)
    loss.backward()
    print(f"✅ Loss Computed: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("🎉 SYSTEM READY FOR LIFT-OFF")
    print("="*60)

if __name__ == "__main__":
    dry_run()
