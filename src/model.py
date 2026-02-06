import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net decoder - matches checkpoint structure exactly."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # Checkpoint structure: Conv -> BN (with bias) -> ReLU -> Conv -> BN (with bias) -> ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpOriginal(nn.Module):
    """Upscaling then double conv - matches original checkpoint structure."""
    def __init__(self, in_channels, out_channels, bilinear=True, mid_channels=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution matching checkpoint structure."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# ResNet-18 Model (ORIGINAL - matches trained checkpoint EXACTLY)
# =============================================================================

class SiameseResNet18UNet(nn.Module):
    """
    Siamese U-Net with ResNet-18 Encoder.
    This is the ORIGINAL model structure that matches the trained checkpoint.
    
    Checkpoint structure:
    - change_fusion: 1536 (512*3) → 512
    - infra_enc: 2 → 32 → 64 → 128
    - infra_fusion: 192 (128+64) → 64
    - up1: 768 (512+256) → 384 (mid) → 128
    - up2: 256 (128+128) → 128 (mid) → 64
    - up3: 128 (64+64) → 64 (mid) → 64
    """
    def __init__(self, n_channels=2, n_classes=1, bilinear=True):
        super(SiameseResNet18UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # --- Input Conv (matches 'inc' in checkpoint) ---
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # --- Encoder (Shared Weights) - Using ResNet-18 ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.maxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1  # 64ch
        self.encoder2 = resnet.layer2  # 128ch
        self.encoder3 = resnet.layer3  # 256ch
        self.encoder4 = resnet.layer4  # 512ch
        
        # --- Change Detection Module ---
        # Checkpoint: 1536 (512*3) → 512 with bias on Conv2d
        self.change_fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        # --- Infrastructure Encoder (matches checkpoint: 2→32→64→128) ---
        self.infra_enc = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # --- Infrastructure Fusion ---
        self.infra_fusion = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),  # 128 + 64 = 192
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # --- Decoder (matches checkpoint structure EXACTLY) ---
        # up1: 768 (512+256) → mid=384 → out=128
        self.up1 = UpOriginal(768, 128, bilinear, mid_channels=384)
        # up2: 256 (128+128) → mid=128 → out=64
        self.up2 = UpOriginal(256, 64, bilinear, mid_channels=128)
        # up3: 128 (64+64) → mid=64 → out=64
        self.up3 = UpOriginal(128, 64, bilinear, mid_channels=64)
        
        # --- Output ---
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x1, x2, infra):
        if x2.shape[1] == 0:
            x2 = torch.zeros_like(x1)
            
        x1_0 = self.inc(x1)
        x1_1 = self.encoder1(self.maxpool(x1_0))
        x1_2 = self.encoder2(x1_1)
        x1_3 = self.encoder3(x1_2)
        x1_4 = self.encoder4(x1_3)
        
        x2_0 = self.inc(x2)
        x2_1 = self.encoder1(self.maxpool(x2_0))
        x2_2 = self.encoder2(x2_1)
        x2_3 = self.encoder3(x2_2)
        x2_4 = self.encoder4(x2_3)
        
        diff = torch.abs(x1_4 - x2_4)
        fused = torch.cat([x1_4, x2_4, diff], dim=1)  # 512*3 = 1536
        fused = self.change_fusion(fused)  # → 512
        
        x = self.up1(fused, x1_3)  # 512+256=768 → 128
        x = self.up2(x, x1_2)      # 128+128=256 → 64
        x = self.up3(x, x1_1)      # 64+64=128 → 64
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


# =============================================================================
# ResNet-50 Model (NEW - for improved performance)
# Set SiameseResNetUNet = SiameseResNet50UNet to use this model
# =============================================================================

class SiameseResNet50UNet(nn.Module):
    """
    Siamese U-Net with ResNet-50 Encoder for improved performance.
    Use this for new training runs - larger capacity model.
    """
    def __init__(self, n_channels=2, n_classes=1, bilinear=True):
        super(SiameseResNet50UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1  # 256ch
        self.encoder2 = resnet.layer2  # 512ch
        self.encoder3 = resnet.layer3  # 1024ch
        self.encoder4 = resnet.layer4  # 2048ch
        
        self.change_fusion = nn.Sequential(
            nn.Conv2d(2048 * 3, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.infra_enc = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder with skip connections
        self.up1 = UpOriginal(2048, 512, bilinear)   # 1024 + 1024 = 2048 -> 512
        self.up2 = UpOriginal(1024, 256, bilinear)   # 512 + 512 = 1024 -> 256
        self.up3 = UpOriginal(512, 128, bilinear)    # 256 + 256 = 512 -> 128
        self.up4 = UpOriginal(192, 64, bilinear)     # 128 + 64 = 192 -> 64
        
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def forward(self, x1, x2, infra):
        if x2.shape[1] == 0:
            x2 = torch.zeros_like(x1)
        
        input_size = x1.shape[2:]
            
        x1_stem = self.stem(x1)
        x1_pool = self.maxpool(x1_stem)
        x1_e1 = self.encoder1(x1_pool)
        x1_e2 = self.encoder2(x1_e1)
        x1_e3 = self.encoder3(x1_e2)
        x1_e4 = self.encoder4(x1_e3)
        
        x2_stem = self.stem(x2)
        x2_pool = self.maxpool(x2_stem)
        x2_e1 = self.encoder1(x2_pool)
        x2_e2 = self.encoder2(x2_e1)
        x2_e3 = self.encoder3(x2_e2)
        x2_e4 = self.encoder4(x2_e3)
        
        diff = torch.abs(x1_e4 - x2_e4)
        fused = torch.cat([x1_e4, x2_e4, diff], dim=1)
        fused = self.change_fusion(fused)
        
        x = self.up1(fused, x1_e3)
        x = self.up2(x, x1_e2)
        x = self.up3(x, x1_e1)
        x = self.up4(x, x1_stem)
        x = self.up5(x)
        
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


# =============================================================================
# MODEL SELECTION - Change this to switch between models
# =============================================================================
# CURRENT: Using ResNet18 (has trained checkpoint: best_model_resnet18.pth)
# TO SWITCH TO RESNET50: Change to SiameseResNet50UNet and use best_model.pth
# =============================================================================

SiameseResNetUNet = SiameseResNet18UNet  # <-- CHANGE THIS TO SWITCH MODELS
