import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseResNetUNet(nn.Module):
    """
    Siamese U-Net with ResNet-18 Encoder, Change Detection Module, and Infrastructure Fusion.
    
    Inputs:
        - x1: Post-event SAR (B, 2, H, W)
        - x2: Pre-event SAR (B, 2, H, W)
        - infra: Infrastructure masks (B, 2, H, W) [Roads, Buildings]
    """
    def __init__(self, n_channels=2, n_classes=1, bilinear=True):
        super(SiameseResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # --- Encoder (Shared Weights) ---
        # Load ResNet-18 (we'll modify first layer for 2 input channels)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify input layer: ResNet expects 3 channels, we have 2 (VV, VH)
        # We can reuse weights by averaging or just init new. Let's init new for simplicity or adapt.
        # Adaptation: Sum first 2 channels weights + average of 3rd? 
        # Simpler: Conv2d(2, 64, ...)
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.encoder1 = resnet.layer1 # 64
        self.encoder2 = resnet.layer2 # 128
        self.encoder3 = resnet.layer3 # 256
        self.encoder4 = resnet.layer4 # 512
        
        # --- Change Detection Module ---
        # Concatenate features from Pre and Post encoders at bottleneck
        # (512 from pre + 512 from post + 512 from diff?) -> 1536 -> 512
        self.change_fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # --- Infrastructure Encoder ---
        self.infra_enc = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Downsample
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Downsample
            nn.ReLU()
        )
        # Output is (B, 128, H/4, W/4) roughly matching decoder stage resolution
        
        # --- Decoder ---
        # ResNet18 downsamples: /2 (inc), /4 (layer1 - no stride?), wait.
        # ResNet18:
        # Input: H, W
        # inc (conv7x7/2 + maxpool/2): H/4, W/4. Channels 64.
        # layer1: H/4, W/4, 64
        # layer2: H/8, W/8, 128
        # layer3: H/16, W/16, 256
        # layer4: H/32, W/32, 512
        
        factor = 2 if bilinear else 1
        
        # Up1: Input 512 (bottleneck), Skip 256 (layer3) -> 768. Output 128.
        self.up1 = Up(768, 128, bilinear)
        
        # Up2: Input 128 (up1), Skip 128 (layer2) -> 256. Output 64.
        self.up2 = Up(256, 64, bilinear)
        
        # Up3: Input 64 (up2), Skip 64 (layer1) -> 128. Output 64.
        self.up3 = Up(128, 64, bilinear)
        
        # Infra Fusion Module
        # We fuse at H/4 resolution (after up3)
        self.infra_fusion = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=1), # 64 from main, 128 from infra
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.up4 removed (unused)
        
        self.outc = OutConv(64, n_classes)
        
        # Final Upsample to original resolution (H/2 -> H or H/4 -> H)
        # Since first layer did /4, we need 2 more upsamples or just a big one.
        # Let's check:
        # enc4: /32
        # up1: /16
        # up2: /8
        # up3: /4
        # up4: /2 (using layer1 skip? layer1 is /4). 
        # We need to use 'inc' skip? inc is pre-maxpool /2.
        
    def forward(self, x_post, x_pre, x_infra):
        # --- Siamese Parameter Sharing ---
        f_post = self.forward_encoder(x_post)
        f_pre = self.forward_encoder(x_pre)
        
        # --- Bottleneck Fusion ---
        # f_post[-1] is the bottleneck features (B, 512, H/32, W/32)
        diff = torch.abs(f_post[-1] - f_pre[-1])
        x = torch.cat([f_post[-1], f_pre[-1], diff], dim=1)
        x = self.change_fusion(x)
        
        # --- Decoder ---
        # Skips come from post-event branch (primary)
        # f_post = [x_inc, x1, x2, x3, x4]
        # x4 is input to fusion.
        x = self.up1(x, f_post[3]) # 512 + 256 -> 256
        x = self.up2(x, f_post[2]) # 256 + 128 -> 128
        x = self.up3(x, f_post[1]) # 128 + 64 -> 64 (Resolution H/4)
        
        # --- Infrastructure Fusion ---
        infra_feat = self.infra_enc(x_infra) # (B, 128, H/4, W/4) expected
        # Resize infra if needed to match x
        if infra_feat.shape[-2:] != x.shape[-2:]:
            infra_feat = F.interpolate(infra_feat, size=x.shape[-2:], mode='bilinear', align_corners=True)
            
        x_fused = torch.cat([x, infra_feat], dim=1)
        x = self.infra_fusion(x_fused) + x # Residual connection (bypass)
        
        # --- Final Upsampling ---
        # Use simple upsample or learned?
        # We are at H/4. Need H.
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)

    def forward_encoder(self, x):
        x_inc = self.inc(x) # /4
        x1 = self.encoder1(x_inc) # /4
        x2 = self.encoder2(x1) # /8
        x3 = self.encoder3(x2) # /16
        x4 = self.encoder4(x3) # /32
        return [x_inc, x1, x2, x3, x4]

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
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

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
