import sys
import uvicorn
import os
import glob
import rasterio
import numpy as np
import torch
import base64
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.dataset import Sen1Floods11Dataset
from src.model import SiameseResNetUNet
from web_app.backend.utils import load_tif_as_array, normalize_sar, prepare_image_response, array_to_base64

app = FastAPI(title="Flood Visualization API")

# Global Model Variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    try:
        # Search for best available checkpoint
        # Priority: ResNet50 (New) > ResNet18 (Old)
        possible_models = list((BASE_DIR / "checkpoints").glob("*resnet_50*.pth")) + \
                          list((BASE_DIR / "checkpoints").glob("*best_model*.pth"))
        
        if possible_models:
            # Sort by modification time (newest first)
            possible_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = possible_models[0]
            print(f"🔄 Selecting newest model: {model_path.name}")
            
            # Decide architecture based on filename
            if "resnet50" in model_path.name.lower() or "resnet_50" in model_path.name.lower():
                from src.model import SiameseResNet50UNet
                model = SiameseResNet50UNet(n_channels=2, n_classes=1)
            else:
                model = SiameseResNetUNet(n_channels=2, n_classes=1)
                
            checkpoint = torch.load(model_path, map_location=device)
            # Handle both state_dict and full checkpoint
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            # Remove 'module.' prefix if saved with DataParallel
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Warning: No checkpoint found in {BASE_DIR}/checkpoints. Running with uninitialized ResNet18.")
            model = SiameseResNetUNet(n_channels=2, n_classes=1)
        
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Dataset Helper (using 'valid' split for visualization usually)
# We can also scan the directory directly for all available pairs.
DATA_DIR = BASE_DIR / "data"

def get_all_samples():
    """Scan for all samples with both S1 and Label."""
    s1_dir = DATA_DIR / "sen1floods11" / "S1Hand"
    label_dir = DATA_DIR / "sen1floods11" / "LabelHand"
    
    if not s1_dir.exists():
        return []
        
    s1_files = {f.name.replace("_S1Hand.tif", "") for f in s1_dir.glob("*_S1Hand.tif")}
    label_files = {f.name.replace("_LabelHand.tif", "") for f in label_dir.glob("*_LabelHand.tif")}
    
    common = sorted(list(s1_files.intersection(label_files)))
    return common

def get_split_info():
    """Load train/valid/test split information from CSV files."""
    splits_dir = DATA_DIR / "splits"
    split_map = {}
    
    # Try to load from CSV split files
    for split_name in ["train", "valid", "test"]:
        csv_path = splits_dir / f"flood_{split_name}_data.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                for line in f.read().splitlines():
                    if "," in line:
                        first_col = line.split(",")[0].strip()
                        if first_col.endswith("_S1Hand.tif"):
                            sample_id = first_col.replace("_S1Hand.tif", "")
                            split_map[sample_id] = split_name
                    elif line.endswith("_S1Hand.tif"):
                        sample_id = line.replace("_S1Hand.tif", "")
                        split_map[sample_id] = split_name
    
    # If no split files found, use 80/20 heuristic
    if not split_map:
        all_samples = get_all_samples()
        split_idx = int(len(all_samples) * 0.8)
        for i, sample_id in enumerate(all_samples):
            split_map[sample_id] = "train" if i < split_idx else "valid"
    
    return split_map

@app.get("/")
async def root():
    return {"message": "Flood Visualization API is running"}

@app.get("/samples")
async def list_samples():
    """Return list of available samples with split information."""
    samples = get_all_samples()
    split_map = get_split_info()
    
    sample_list = []
    for sample_id in samples:
        sample_list.append({
            "id": sample_id,
            "split": split_map.get(sample_id, "unknown")
        })
    
    # Count by split
    train_count = sum(1 for s in sample_list if s["split"] == "train")
    valid_count = sum(1 for s in sample_list if s["split"] == "valid")
    test_count = sum(1 for s in sample_list if s["split"] == "test")
    
    return {
        "samples": sample_list, 
        "count": len(samples),
        "splits": {"train": train_count, "valid": valid_count, "test": test_count}
    }

@app.get("/samples/{sample_id}/images/{image_type}")
async def get_sample_image(sample_id: str, image_type: str):
    """
    Get image for a sample.
    image_type: post, pre, gt, pred, infra
    """
    s1_dir = DATA_DIR / "sen1floods11" / "S1Hand"
    label_dir = DATA_DIR / "sen1floods11" / "LabelHand"
    pre_dir = DATA_DIR / "pre_event_sar"
    infra_dir = DATA_DIR / "infrastructure" / "rasterized"

    data = None
    is_mask = False
    mask_color = "blue"
    
    if image_type == "post":
        path = s1_dir / f"{sample_id}_S1Hand.tif"
        raw = load_tif_as_array(path)
        if raw is not None:
             data = normalize_sar(raw)
             
    elif image_type == "pre":
        path = pre_dir / f"{sample_id}_PreEvent.tif"
        raw = load_tif_as_array(path)
        if raw is not None:
            # Handle potential extra channels or size mismatch handled in utils/dataset
            if raw.shape[0] > 2: raw = raw[:2]
            data = normalize_sar(raw)
            
    elif image_type == "gt":
        path = label_dir / f"{sample_id}_LabelHand.tif"
        raw = load_tif_as_array(path)
        if raw is not None:
            data = raw
            is_mask = True
            mask_color = "blue"  # Ground truth = blue
            
    elif image_type == "pred":
        # Run model inference for prediction mask
        if model is None:
            return Response(status_code=503, content="Model not loaded")
        
        path = s1_dir / f"{sample_id}_S1Hand.tif"
        raw = load_tif_as_array(path)
        if raw is not None:
            try:
                # Preprocessing matching training pipeline
                def process_sar(d):
                    if d.shape[0] > 2: d = d[:2]
                    d = np.nan_to_num(d, nan=0.0)  # Handle NaN
                    # EXACT MATCH to dataset.py normalization
                    d = np.clip(d, -50, 20)
                    return (d + 50) / 70.0
                
                post_t = torch.from_numpy(process_sar(raw)).unsqueeze(0).float().to(device)
                pre_t = torch.zeros_like(post_t)
                infra_t = torch.zeros((1, 2, *post_t.shape[2:])).float().to(device)
                
                model.eval()
                with torch.no_grad():
                    output = model(post_t, pre_t, infra_t)
                    pred = (output > 0.5).float().cpu().numpy()[0]  # (1, H, W)
                
                data = pred
                is_mask = True
                mask_color = "red"  # Prediction = red
            except Exception as e:
                print(f"Prediction error for {sample_id}: {e}")
                return Response(status_code=500)
            
    elif image_type == "infra":
        path = infra_dir / f"{sample_id}_Infra.tif"
        # Infra is 2 channels: Road, Building. Visualize as Red/Green?
        # Or just return raw for now.
        pass 
        
    if data is None:
        return Response(status_code=404)
        
    buf = prepare_image_response(data, colormap=is_mask, color=mask_color)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/samples/{sample_id}/metadata")
async def get_sample_metadata(sample_id: str):
    """Get georeferencing bounds and metrics for the sample."""
    s1_dir = DATA_DIR / "sen1floods11" / "S1Hand"
    label_dir = DATA_DIR / "sen1floods11" / "LabelHand"
    path = s1_dir / f"{sample_id}_S1Hand.tif"
    label_path = label_dir / f"{sample_id}_LabelHand.tif"
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
        
    metrics = {"status": "Flooded"}
    bounds = None
    crs = "Unknown"
    
    try:
        with rasterio.open(path) as src:
            b = src.bounds
            bounds = [[b.bottom, b.left], [b.top, b.right]]
            crs = str(src.crs)
            post_raw = src.read().astype(np.float32)

        if label_path.exists() and model is not None:
            with rasterio.open(label_path) as src:
                label_arr = src.read(1).astype(np.float32)
                
            # Preprocessing - MUST MATCH dataset.py exactly
            def process_sar(data):
                if data.shape[0] > 2: data = data[:2]
                data = np.nan_to_num(data, nan=0.0)
                data = np.clip(data, -50, 20)
                return (data + 50) / 70.0

            post_t = torch.from_numpy(process_sar(post_raw)).unsqueeze(0).to(device)
            pre_t = torch.zeros_like(post_t)
            infra_t = torch.zeros((1, 2, *post_t.shape[2:])).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(post_t, pre_t, infra_t)
                pred = (output > 0.5).float().cpu().numpy()[0, 0]
            
            # Binary metrics
            target = np.array(label_arr == 1, dtype=np.float32)
            intersection = np.logical_and(pred > 0.5, target > 0.5).sum()
            union = np.logical_or(pred > 0.5, target > 0.5).sum()
            iou = float(intersection / union) if union > 0 else 1.0
            accuracy = float(np.mean((pred > 0.5) == (target > 0.5)))
            
            metrics = {
                "iou": float(np.round(iou, 4)),
                "accuracy": float(np.round(accuracy, 4)),
                "status": "Flooded" if target.any() else "No Flood"
            }
    except Exception as e:
        print(f"Error computing metrics for {sample_id}: {e}")
        metrics = {"status": "Error", "error": str(e)}

    return {
        "bounds": bounds,
        "crs": crs,
        "metrics": metrics
    }

@app.post("/predict")
async def predict(post_image: UploadFile = File(...), pre_image: Optional[UploadFile] = File(None)):
    """Run inference on uploaded images with tiling support for large images."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Load images from bytes
        post_bytes = await post_image.read()
        
        with rasterio.MemoryFile(post_bytes) as mem:
            with mem.open() as src:
                post_arr = src.read().astype(np.float32)
                
        if pre_image:
            pre_bytes = await pre_image.read()
            with rasterio.MemoryFile(pre_bytes) as mem:
                with mem.open() as src:
                    pre_arr = src.read().astype(np.float32)
        else:
            pre_arr = np.zeros_like(post_arr)
            
        # Ensure both have same shape
        if pre_arr.shape != post_arr.shape:
            # Crop or pad to match
            c, h, w = post_arr.shape
            if pre_arr.shape[1] >= h and pre_arr.shape[2] >= w:
                pre_arr = pre_arr[:c, :h, :w]
            else:
                pre_arr = np.zeros_like(post_arr)
            
        # Preprocessing - MUST MATCH dataset.py exactly
        def process_sar(data):
            if data.shape[0] > 2: data = data[:2]
            data = np.nan_to_num(data, nan=0.0)
            data = np.clip(data, -50, 20)
            return (data + 50) / 70.0

        post_proc = process_sar(post_arr)
        pre_proc = process_sar(pre_arr)
        
        _, full_h, full_w = post_proc.shape
        
        # --- Tile-based inference for large images ---
        TILE_SIZE = 512
        OVERLAP = 64  # Overlap to avoid edge artifacts
        STEP = TILE_SIZE - OVERLAP
        
        if full_h <= TILE_SIZE and full_w <= TILE_SIZE:
            # Small image: direct inference
            infra = np.zeros((2, full_h, full_w), dtype=np.float32)
            post_t = torch.from_numpy(post_proc).unsqueeze(0).float().to(device)
            pre_t = torch.from_numpy(pre_proc).unsqueeze(0).float().to(device)
            infra_t = torch.from_numpy(infra).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                output = model(post_t, pre_t, infra_t)
                prediction = (output > 0.5).float().cpu().numpy()[0]
        else:
            # Large image: tile-based inference
            prediction = np.zeros((1, full_h, full_w), dtype=np.float32)
            weight_map = np.zeros((1, full_h, full_w), dtype=np.float32)
            
            # Calculate tile positions
            y_starts = list(range(0, full_h - TILE_SIZE + 1, STEP))
            if y_starts[-1] + TILE_SIZE < full_h:
                y_starts.append(full_h - TILE_SIZE)
            
            x_starts = list(range(0, full_w - TILE_SIZE + 1, STEP))
            if x_starts[-1] + TILE_SIZE < full_w:
                x_starts.append(full_w - TILE_SIZE)
            
            # Process each tile
            for y in y_starts:
                for x in x_starts:
                    # Extract tile
                    post_tile = post_proc[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
                    pre_tile = pre_proc[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
                    infra_tile = np.zeros((2, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    
                    # To tensors
                    post_t = torch.from_numpy(post_tile).unsqueeze(0).float().to(device)
                    pre_t = torch.from_numpy(pre_tile).unsqueeze(0).float().to(device)
                    infra_t = torch.from_numpy(infra_tile).unsqueeze(0).float().to(device)
                    
                    # Inference
                    with torch.no_grad():
                        output = model(post_t, pre_t, infra_t)
                        tile_pred = output.cpu().numpy()[0]
                    
                    # Create blending weight (higher in center, lower at edges)
                    weight = np.ones((1, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    # Soft edges for blending
                    blend = min(OVERLAP, 32)
                    for i in range(blend):
                        factor = (i + 1) / blend
                        weight[:, i, :] *= factor
                        weight[:, -i-1, :] *= factor
                        weight[:, :, i] *= factor
                        weight[:, :, -i-1] *= factor
                    
                    # Accumulate
                    prediction[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += tile_pred * weight
                    weight_map[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += weight
            
            # Normalize by weight
            prediction = prediction / np.maximum(weight_map, 1e-6)
            prediction = (prediction > 0.5).astype(np.float32)
            
        # Convert prediction to base64 for frontend display
        pred_buf = prepare_image_response(prediction, colormap=True, color="red")
        img_base64 = base64.b64encode(pred_buf.getvalue()).decode('utf-8')
        
        flood_pixels = np.sum(prediction > 0)
        total_pixels = prediction.size
        flood_pct = (flood_pixels / total_pixels) * 100
        
        return {
            "prediction": f"data:image/png;base64,{img_base64}",
            "is_flooded": bool(flood_pixels > 0),
            "flood_percentage": round(float(flood_pct), 2),
            "image_size": [full_h, full_w],
            "tiles_processed": len(y_starts) * len(x_starts) if full_h > TILE_SIZE or full_w > TILE_SIZE else 1
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

