import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time

from src.dataset import Sen1Floods11Dataset
from src.model import SiameseResNet18UNet, SiameseResNet50UNet
from src.loss import BCEDiceLoss
from src.augmentations import get_train_transforms, get_val_transforms
import matplotlib.pyplot as plt
import numpy as np

def train_model():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)  # Smaller batch for ResNet50
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--strict_data", action="store_true", help="Only use samples with ALL data (Pre+Infra)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no_oversample", action="store_true", help="Disable flood oversampling")
    parser.add_argument("--use_weak", action="store_true", help="Include weakly labeled data")  # New argument
    parser.add_argument("--model_arch", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Model architecture")
    parser.add_argument("--checkpoint_name", type=str, default="best_model", help="Base name for checkpoint (no extension)")
    args = parser.parse_args()

    # --- Configuration ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    STRICT_DATA = args.strict_data
    USE_OVERSAMPLE = not args.no_oversample
    
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")
    print(f"Strict Data Mode: {STRICT_DATA}, Oversampling: {USE_OVERSAMPLE}")

    # --- Data with Augmentation + Oversampling ---
    print("=" * 60)
    print("Initializing DataLoaders...")
    
    # --- Data Loading ---
    print(f"Loading data (Weak data: {'ENABLED' if args.use_weak else 'DISABLED'})...")
    
    # If using weak data, we must disable strict requirements because weak samples 
    # likely lack aligned Pre-Event/Infrastructure data.
    strict_mode = STRICT_DATA
    if args.use_weak:
        print("Note: Strict mode (require_complete) disabled for weak data support.")
        strict_mode = False

    # Training set with augmentation AND oversampling
    train_ds = Sen1Floods11Dataset(
        "./data", 
        split="train", 
        transform=get_train_transforms(),
        require_complete=strict_mode,
        use_infrastructure=False,  # Weak data won't have infra
        # For weak data, we might not have pre-event either, so handle that:
        # Use pre-event if we have it (dataset handles missing ones with strict_mode=False)
        use_pre_event=True,
        min_flood_pixels=10,  # Lower threshold to include more data
        oversample_flood=USE_OVERSAMPLE, 
        max_black_ratio=0.05,
        use_weak=args.use_weak
    )
    
    if len(train_ds) == 0:
        print("Dataset empty. Aborting.")
        return
    
    # Validation set WITHOUT augmentation or oversampling
    val_ds = Sen1Floods11Dataset(
        "./data", 
        split="valid", 
        transform=get_val_transforms(),
        require_complete=strict_mode, 
        use_infrastructure=False,
        use_pre_event=not args.use_weak,
        max_black_ratio=0.05,
        use_weak=args.use_weak  # Typically valid is only hand-labeled, but keeping logic consistent
    )
    
    # Use weighted sampler for training if oversampling enabled
    sampler = train_ds.get_sampler() if USE_OVERSAMPLE else None
    shuffle = (sampler is None)  # Don't shuffle if using sampler
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,        # Optimized for M2 Pro
        pin_memory=True,      # Faster GPU transfer
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Batches per epoch: {len(train_loader)} (with oversampling: ~{len(train_loader) if sampler else 'N/A'})")
    print("=" * 60)

    # --- Model Selection ---
    print(f"Initializing Model: {args.model_arch}")
    if args.model_arch == "resnet50":
        model = SiameseResNet50UNet(n_channels=2, n_classes=1).to(DEVICE)
    else:
        model = SiameseResNet18UNet(n_channels=2, n_classes=1).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=False)
    
    # --- Loss & Optimizer ---
    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # --- Logging ---
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard import SummaryWriter
    log_dir = f"logs/{args.model_arch}_{args.checkpoint_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging: {log_dir}")

    # --- Training Loop ---
    print("=" * 60)
    print("Starting Training (ResNet50 + Oversampling)...")
    print("=" * 60)
    best_iou = 0.0
    patience = 15  # Early stopping patience
    patience_counter = 0
    
    def calculate_iou(pred, target):
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        has_flood = (target.sum() > 0).item()
        if union == 0:
            return 1.0, has_flood
        return (intersection / union).item(), has_flood

    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        flood_count = 0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            img = batch["image"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            pre_img = batch["pre_image"].to(DEVICE)
            infra = batch["infra"].to(DEVICE)
            
            optimizer.zero_grad()
            output = model(img, pre_img, infra)
            loss = criterion(output, label)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_iou, has_flood = calculate_iou(output, label)
            if has_flood:
                train_iou += batch_iou
                flood_count += 1
            
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} [{i}/{len(train_loader)}] Loss: {loss.item():.4f} IoU: {batch_iou:.4f}")
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_flood_count = 0
        
        # Track top samples
        val_samples = []  # will store (iou, sample_id, prediction_mask, ground_truth, input_image)

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                pre_img = batch["pre_image"].to(DEVICE)
                infra = batch["infra"].to(DEVICE)
                sample_ids = batch["id"]
                
                output = model(img, pre_img, infra)
                loss = criterion(output, label)
                val_loss += loss.item()
                
                # Batch IoU logging
                b_iou, has_flood = calculate_iou(output, label)
                if has_flood:
                    val_iou += b_iou
                    val_flood_count += 1
                
                # Per-sample analysis for "Top 5"
                preds = (output > 0.5).float()
                for j in range(len(img)):
                    # Only calculate/store if there is actual flood to detect (otherwise IoU=1.0 is trivial)
                    tgt = label[j]
                    if tgt.sum() > 0:
                        p = preds[j]
                        intersection = (p * tgt).sum()
                        union = p.sum() + tgt.sum() - intersection
                        siou = (intersection / union).item() if union > 0 else 0.0
                        
                        # Store essential data (CPU) for sorting/saving
                        # Move to CPU immediately to save GPU memory
                        val_samples.append({
                            "iou": siou,
                            "id": sample_ids[j],
                            "pred": p.cpu(),
                            "label": tgt.cpu(),
                            "img": img[j].cpu()
                        })

        # Process Top 5
        val_samples.sort(key=lambda x: x["iou"], reverse=True)
        top_5 = val_samples[:5]
        
        # Save top 5 images
        if len(top_5) > 0:
            save_dir = Path("results/best_samples")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear old files for this epoch
            # (optional, or just overwrite by ID/Epoch)
            
            print(f"Top 5 Validation Samples (Epoch {epoch+1}):")
            for idx, item in enumerate(top_5):
                print(f"  #{idx+1}: {item['id']} - IoU: {item['iou']:.4f}")
                
                # Create plot
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                
                # Input (Channel 0 for VV)
                inp_img = item["img"][0].numpy()
                # Denormalize? It's [0,1] from our previous fix.
                
                ax[0].imshow(inp_img, cmap="gray")
                ax[0].set_title(f"Input (VV) - {item['id']}")
                ax[0].axis('off')
                
                # Label
                ax[1].imshow(item["label"][0].numpy(), cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[1].axis('off')
                
                # Pred
                ax[2].imshow(item["pred"][0].numpy(), cmap="gray")
                ax[2].set_title(f"Prediction (IoU: {item['iou']:.3f})")
                ax[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_dir / f"Best_Epoch{epoch+1}_{idx+1}_{item['id']}.png")
                plt.close(fig)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / flood_count if flood_count > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_iou = val_iou / val_flood_count if val_flood_count > 0 else 0
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('IoU/train_epoch', avg_train_iou, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('IoU/val_epoch', avg_val_iou, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        epoch_time = time.time() - start_time
        print("=" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS} Complete. Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_train_loss:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f}")
        
        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{args.checkpoint_name}.pth")
            print(f"  ✅ New best model! IoU: {best_iou:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")
            print(f"  📁 Checkpoint saved at epoch {epoch+1}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break
            
        print("=" * 60)
            
    writer.close()
    print(f"\n🎉 Training complete! Best Val IoU: {best_iou:.4f}")

if __name__ == "__main__":
    train_model()
