
import rasterio
from pathlib import Path

def print_proof(sample_id):
    s1_path = Path(f"data/sen1floods11/S1Hand/{sample_id}_S1Hand.tif")
    pre_path = Path(f"data/pre_event_sar/{sample_id}_PreEvent.tif")
    
    if not s1_path.exists() or not pre_path.exists():
        print(f"Files for {sample_id} not found.")
        return

    print(f"Verifying Pair: {sample_id}")
    
    with rasterio.open(s1_path) as src1:
        print(f"\n--- Post-Event (Reference) ---")
        print(f"CRS: {src1.crs}")
        print(f"Bounds (Left, Bottom, Right, Top): {src1.bounds}")
        b1 = src1.bounds
        
    with rasterio.open(pre_path) as src2:
        print(f"\n--- Pre-Event (Downloaded) ---")
        print(f"CRS: {src2.crs}")
        print(f"Bounds (Left, Bottom, Right, Top): {src2.bounds}")
        b2 = src2.bounds
        
    print(f"\n--- Comparison ---")
    print(f"Left Match:   {abs(b1.left - b2.left) < 0.0001}")
    print(f"Bottom Match: {abs(b1.bottom - b2.bottom) < 0.0001}")
    print(f"Right Match:  {abs(b1.right - b2.right) < 0.0001}")
    print(f"Top Match:    {abs(b1.top - b2.top) < 0.0001}")

if __name__ == "__main__":
    # Pick a random downloaded file
    print_proof("USA_933610")
