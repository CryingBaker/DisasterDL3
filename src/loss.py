import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for flood segmentation.
    
    Why this is needed:
    Flood pixels are extremely rare (class imbalance). BCE alone leads to "background collapse"
    (model predicting 0 everywhere to minimize loss). Dice loss optimizes overlap directly,
    forcing the model to care about the minority class.
    
    Args:
        bce_weight (float): Weight for BCE component. Default: 0.5
        dice_weight (float): Weight for Dice component. Default: 1.0 (emphasize Dice)
        smooth (float): Smoothing factor for Dice numerical stability.
    """
    def __init__(self, bce_weight=0.5, dice_weight=1.0, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCELoss() # Expects Sigmoid probabilities, not logits

    def forward(self, inputs, targets):
        """
        inputs: Sigmoid probabilities (B, 1, H, W)
        targets: Binary Ground Truth (B, 1, H, W)
        """
        # 1. BCE Component
        bce_loss = self.bce(inputs, targets)

        # 2. Dice Component
        # Reshape to (B, -1) to compute per-sample
        B = inputs.shape[0]
        inputs_flat = inputs.view(B, -1)
        targets_flat = targets.view(B, -1)
        
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        
        # Dice = 2 * (intersection) / (union)
        # Compute per image and then average over batch
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        dice_loss = (1 - dice_score).mean()
        
        # 3. Combine
        final_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
        
        return final_loss
