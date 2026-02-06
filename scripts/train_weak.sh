#!/bin/bash
# Train with Weakly Labeled Data
# ------------------------------
# 1. Ensures pre-event data download is up to date (syncs any new files)
# 2. Runs training with --use_weak flag

# Sync pre-event data (in case new raw files arrived)
echo "Syncing pre-event data..."
./venv/bin/python scripts/download_pre_event_sar.py --source_dir S1Weak

# Run Training
echo "Starting Training with Weak Data..."
./venv/bin/python src/train.py --use_weak --batch_size 16 --epochs 50

# Note: Checkpoints will be saved to checkpoints/
