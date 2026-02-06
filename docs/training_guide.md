# Training Guide

## Preventing Background Collapse in Flood Segmentation

You may notice that your model initially predicts "no flood" (all zeros) everywhere. This is known as **background collapse**.

### Why BCE Alone Fails
Flood pixels are extremely rare (often <1% of the image). A standard Binary Cross Entropy (BCE) loss treats every pixel equally. The model learns that it can achieve a very low loss (e.g., 0.01) by simply predicting "background" for every single pixel. Mathematically, this is a local minimum that is hard to escape.

### Why Dice Loss Fixes It
Dice Loss is based on the **Dice Coefficient** (Overlap). It doesn't care about the number of background pixels; it only cares about the *intersection* of the predicted flood and the actual flood. 
- If the model predicts all background, the intersection is 0, and the Dice Loss is maximum (1.0).
- This huge penalty forces the model to start predicting *some* flood pixels, breaking out of the "all-zero" trap.

### Expected Behavior After the Fix
When training with `BCEDiceLoss`, you should observe:
1.  **Initial Volatility:** The loss might not decrease as smoothly as before. This is good—it means the model is exploring.
2.  **Low IoU Start:** IoU might stay near 0.0 for the first few epochs while the model learns *where* to look.
3.  **Sharp Rise:** Once the model identifies flood patterns, IoU should increase rapidly (e.g., jumping from 0.05 to 0.40+ quickly).

**Do not panic if IoU is low for epochs 1-5. Give it time to find the signal.**
