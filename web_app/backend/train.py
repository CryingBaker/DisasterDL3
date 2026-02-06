import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time

from src.dataset import Sen1Floods11Dataset
from src.model import SiameseResNetUNet
from src.loss import BCEDiceLoss

def train_model():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--strict_data", action="store_true", help="Only use samples with ALL data (Pre+Infra)")
    args = parser.parse_args()

    # --- Configuration ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    STRICT_DATA = args.strict_data
    
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {DEVICE}")
    print(f"Strict Data Mode: {STRICT_DATA}")

    # --- Data ---
    print("Initializing DataLoaders (Infrastructure Disabled)...")
    # STRICT_DATA now enforces Pre-Event SAR existence only (since infra is disabled)
    train_ds = Sen1Floods11Dataset("./data", split="train", require_complete=STRICT_DATA, use_infrastructure=False)
    if len(train_ds) == 0:
        print("Dataset empty. Aborting.")
        return
        
    val_ds = Sen1Floods11Dataset("./data", split="valid", require_complete=STRICT_DATA, use_infrastructure=False)
    
    # MPS optimization: num_workers=0 often more stable locally, or 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model ---
    print("Initializing Model...")
    model = SiameseResNetUNet(n_channels=2, n_classes=1).to(DEVICE)
    
    # --- Loss & Optimizer ---
    # UPDATED: Using BCEDiceLoss to prevent background collapse
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- Logging ---
    from torch.utils.tensorboard import SummaryWriter
    log_dir = "logs/experiment_dice_loss"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging enabled in '{log_dir}'")

    # --- Training Loop ---
    print("Starting Training...")
    best_loss = float('inf')
    
    # --- Helper: IoU Metric ---
    def calculate_iou(pred, target):
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        if union == 0:
            return 1.0 # Correctly handled empty masks match
        return (intersection / union).item()

    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            # Move to device
            img = batch["image"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            pre_img = batch["pre_image"].to(DEVICE)
            infra = batch["infra"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(img, pre_img, infra)
            
            # Loss
            loss = criterion(output, label)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            batch_iou = calculate_iou(output, label)
            train_iou += batch_iou
            
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            writer.add_scalar('IoU/train_batch', batch_iou, global_step)
            global_step += 1
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} [Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f} IoU: {batch_iou:.4f}")
                
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                pre_img = batch["pre_image"].to(DEVICE)
                infra = batch["infra"].to(DEVICE)
                
                output = model(img, pre_img, infra)
                loss = criterion(output, label)
                val_loss += loss.item()
                val_iou += calculate_iou(output, label)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_iou = val_iou / len(val_loader) if len(val_loader) > 0 else 0
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('IoU/train_epoch', avg_train_iou, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('IoU/val_epoch', avg_val_iou, epoch)
        
        print(f"Epoch {epoch+1} Complete. Time: {time.time()-start_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val Loss:  {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f}")
        
        # Checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("  Checkpoint saved.")
            
    writer.close()

if __name__ == "__main__":
    train_model()
