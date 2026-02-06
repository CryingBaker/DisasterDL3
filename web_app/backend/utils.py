import rasterio
import numpy as np
from PIL import Image
import io
import torch
import base64
from pathlib import Path
from typing import Tuple, Optional

def load_tif_as_array(path: Path) -> np.ndarray:
    """Load a TIF file and return it as a numpy array."""
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read()

def normalize_sar(data: np.ndarray) -> np.ndarray:
    """Normalize SAR data (assumes shape C, H, W)."""
    # Clip and normalize like in dataset.py
    data = np.clip(data, -50, 0)
    data = (data + 50) / 50.0 * 255
    return data.astype(np.uint8)

def prepare_image_response(
    data: np.ndarray, 
    channel_indices: Tuple[int, ...] = (0, 1, 0),
    colormap: bool = False,
    color: str = "blue"
) -> io.BytesIO:
    """
    Convert numpy array to image bytes.
    Args:
        data: (C, H, W) array
        channel_indices: Which channels to use for RGB. Replicate if fewer channels.
        colormap: If True, apply false color map (for masks).
        color: Color for mask overlay - "blue" (ground truth) or "red" (prediction)
    """
    if data is None:
        # Return transparent placeholder
        img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    else:
        if colormap:
            # Assumes data is (1, H, W) mask
            mask = data[0].astype(np.uint8)
            # Create RGBA image: colored for flood (1), Transparent for non-flood (0)
            img = Image.fromarray(mask, mode='L')
            img = img.convert("RGBA")
            datas = img.getdata()
            
            # Color options: blue=ground truth, red=prediction
            if color == "red":
                flood_color = (255, 60, 60, 180)  # Semi-transparent Red
            else:
                flood_color = (0, 100, 255, 180)  # Semi-transparent Blue
            
            new_data = []
            for item in datas:
                if item[0] > 0: # Flood
                    new_data.append(flood_color)
                else:
                    new_data.append((0, 0, 0, 0)) # Transparent
            img.putdata(new_data)
        else:
            # Normal image processing
            # Ensure we have enough channels
            c, h, w = data.shape
            
            # Simple visualization for 2-channel SAR
            # R: VV, G: VH, B: VV (or ratio?)
            
            # If 2 channels (VV, VH)
            if c == 2:
                vv = data[0]
                vh = data[1]
                # RGB = VV, VH, VV/VH ratio or just VV
                img_data = np.stack([vv, vh, vv], axis=2).astype(np.uint8)
            elif c == 1:
                img_data = np.stack([data[0]]*3, axis=2).astype(np.uint8)
            else:
                img_data = np.moveaxis(data[:3], 0, 2).astype(np.uint8)
                
            img = Image.fromarray(img_data)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def array_to_base64(data: np.ndarray) -> str:
    buf = prepare_image_response(data)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
