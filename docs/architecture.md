# System Architecture

## Overview
The system uses a **Siamese ResNet-U-Net** architecture to detect floods by comparing pre-event and post-event SAR imagery, fused with OpenStreetMap (OSM) infrastructure data.

```mermaid
graph LR
    subgraph Inputs
        PRE[Pre-Event SAR<br/>(VV, VH)]
        POST[Post-Event SAR<br/>(VV, VH)]
        INFRA[Infrastructure<br/>(Roads, Buildings)]
    end
    
    subgraph Siamese_Encoder
        ENC_PRE[ResNet-18 Encoder]
        ENC_POST[ResNet-18 Encoder]
    end
    
    subgraph Fusion
        DIFF[Difference Module]
        CONCAT[Concatenation]
    end
    
    subgraph Decoder
        DEC[U-Net Decoder]
        SKIP[Skip Connections]
        INFRA_FUSE[Infra Fusion Gate]
    end
    
    PRE --> ENC_PRE
    POST --> ENC_POST
    
    ENC_PRE --> DIFF
    ENC_POST --> DIFF
    ENC_POST --> CONCAT
    DIFF --> CONCAT
    
    CONCAT --> DEC
    ENC_POST -.-> SKIP -.-> DEC
    
    INFRA --> INFRA_FUSE --> DEC
    
    DEC --> OUT[Flood Probability Map]
```

## Component Details

### 1. Siamese Encoder (ResNet-18)
- **Purpose:** Extract features from SAR imagery while being invariant to absolute temporal shifts.
- **Weights:** Shared between Pre and Post branches.
- **Backbone:** ResNet-18 (Pretrained on ImageNet, adapted for 2-channel input).

### 2. Change Detection Module
- **Inputs:** Feature maps from Pre and Post encoders.
- **Operation:** Computes absolute difference `|Pre - Post|` and concatenates with original features.
- **Goal:** Explicitly highlight changes (flood waters appear as dark patches in SAR).

### 3. Infrastructure Fusion (Auxiliary Branch)
- **Input:** Rasterized OSM masks (Roads, Buildings).
- **Mechanism:** Light CNN encoder fused into the decoder via a residual connection.
- **Rationale:** Provides context (e.g., "this dark area is a street, so water is likely") without overriding the SAR signal.

### 4. Loss Function
- **Hybrid Loss:** `0.5 * BCE + 0.5 * Dice`.
- **Reason:** Handles class imbalance (flood pixels are rare).
