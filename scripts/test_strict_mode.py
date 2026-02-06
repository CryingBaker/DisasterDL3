
import sys
import os
sys.path.append(os.getcwd())
from src.dataset import Sen1Floods11Dataset

def test_strict():
    print("="*60)
    print("🧪 TESTING STRICT DATA FILTERING")
    print("="*60)
    
    # 1. Normal Mode
    print("\n[Mode: All Available]")
    ds_all = Sen1Floods11Dataset("./data", split="train", require_complete=False)
    len_all = len(ds_all)
    print(f"✅ Dataset size: {len_all} samples")
    
    # 2. Strict Mode (Pre-Event Required, Infra Disabled)
    print("\n[Mode: Strict (Pre-Event Required)]")
    ds_strict = Sen1Floods11Dataset("./data", split="train", require_complete=True, use_infrastructure=False)
    len_strict = len(ds_strict)
    print(f"✅ Dataset size: {len_strict} samples")
    
    # 3. Conclusion
    dropped = len_all - len_strict
    print("\n" + "="*60)
    print(f"📉 Filtered out {dropped} incomplete samples.")
    print(f"📊 Retention Rate: {(len_strict/len_all)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    test_strict()
