#!/bin/bash
# Auto Training Pipeline
# ----------------------
# 1. Waits for Raw Weak Data Download (download_weak.py) to finish.
# 2. Runs Pre-Event Data Generation (download_pre_event_sar.py) for ALL files.
# 3. Trains ResNet50 with Weak Data, saving to 'best_model_resnet50_weak.pth'.

LOG_FILE="scripts/pipeline.log"

{
    echo "Pipeline started at $(date)"
    
    # 1. Wait for Raw Download
    echo "[1/3] Waiting for raw download (download_weak.py)..."
    while pgrep -f "scripts/download_weak.py" > /dev/null; do
        echo "  - Download still running... waiting 60s"
        sleep 60
    done
    echo "  - Raw download complete!"

    # 2. Pre-Event Generation
    echo "[2/3] Generating pre-event imagery (GEE)..."
    ./venv/bin/python scripts/download_pre_event_sar.py --source_dir S1Weak --mode full
    echo "  - Pre-event generation complete!"

    echo "[3/3] Starting ResNet50 Training..."
    echo "  - Model: ResNet50"
    echo "  - Data: Hand + Weak"
    echo "  - Checkpoint: checkpoints/resnet_50_weakly_labelled.pth"
    
    ./venv/bin/python src/train.py \
        --model_arch resnet50 \
        --use_weak \
        --checkpoint_name resnet_50_weakly_labelled \
        --batch_size 16 \
        --epochs 50
        
    echo "Pipeline finished at $(date)"
} > "$LOG_FILE" 2>&1
