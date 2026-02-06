# Model Structure Details

## Siamese ResNet-18 U-Net (2-Channel Input)

```
Input: (B, 2, 512, 512) for both Pre and Post branches.

--------------------------------------------------------------------------------
Layer (type)               Output Shape         Param #    Connected to
--------------------------------------------------------------------------------
Conv2d-1 (Shared)          [-1, 64, 256, 256]   6,272      [Input_Pre, Input_Post]
BatchNorm2d-2              [-1, 64, 256, 256]   128        [Conv2d-1]
ReLU-3                     [-1, 64, 256, 256]   0          [BatchNorm2d-2]
MaxPool2d-4                [-1, 64, 128, 128]   0          [ReLU-3]

Layer1 (ResBlock)          [-1, 64, 128, 128]   147,968    [MaxPool2d-4]
Layer2 (ResBlock)          [-1, 128, 64, 64]    525,568    [Layer1]
Layer3 (ResBlock)          [-1, 256, 32, 32]    2,100,224  [Layer2]
Layer4 (ResBlock)          [-1, 512, 16, 16]    8,394,752  [Layer3]

Difference_Module          [-1, 1536, 16, 16]   0          [Layer4_Pre, Layer4_Post]
Fusion_Conv                [-1, 512, 16, 16]    7,078,400  [Difference_Module]

Up1 (Deconv/Interp)        [-1, 256, 32, 32]    0          [Fusion_Conv]
Skip_Connect_1             [-1, 768, 32, 32]    0          [Up1, Layer3_Post]
Dec_Block_1                [-1, 256, 32, 32]    Target     [Skip_Connect_1]

Up2 (Deconv/Interp)        [-1, 128, 64, 64]    0          [Dec_Block_1]
Skip_Connect_2             [-1, 384, 64, 64]    0          [Up2, Layer2_Post]
Dec_Block_2                [-1, 128, 64, 64]    Target     [Skip_Connect_2]

Up3 (Deconv/Interp)        [-1, 64, 128, 128]   0          [Dec_Block_2]
Skip_Connect_3             [-1, 192, 128, 128]  0          [Up3, Layer1_Post]
Dec_Block_3                [-1, 64, 128, 128]   Target     [Skip_Connect_3]

Infra_Encoder_Out          [-1, 128, 128, 128]  Target     [Infra_Input]
Infra_Fusion               [-1, 64, 128, 128]   Target     [Dec_Block_3, Infra_Encoder_Out]

Final_Upsample             [-1, 64, 512, 512]   0          [Infra_Fusion]
OutConv                    [-1, 1, 512, 512]    65         [Final_Upsample]
--------------------------------------------------------------------------------
Total params: ~22,000,000
Trainable params: ~22,000,000
Non-trainable params: 0
--------------------------------------------------------------------------------
```
