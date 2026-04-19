#!/bin/bash
# Two-step training for EfficientNet and ViT
# Step 1: Train with val split to find best epoch
# Step 2: Train on 100% data with that epoch count

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$PWD"
export MLFLOW_TRACKING_URI="file://$PWD/mlruns"
export PYTHONUNBUFFERED=1

for MODEL in efficientnet vit; do
    echo ""
    echo "============================================================"
    echo "=== ${MODEL} Training Started: $(date) ==="
    echo "============================================================"

    # ─── Step 1: Train with val split (30 epochs) ───
    echo ""
    echo "=== STEP 1 [${MODEL}]: Training with validation split (30 epochs) ==="
    echo "Started: $(date)"

    python3 -m src.train --model "$MODEL" --data-dir data --epochs 30 --num-workers 0 2>&1 | tee "/tmp/${MODEL}_step1.log"

    # Extract best epoch from MLflow metrics (more reliable than log parsing)
    echo ""
    echo "=== Step 1 [${MODEL}] Complete ==="
    echo "Finished: $(date)"

    # Find the latest MLflow run for this model and get best epoch from checkpoint
    BEST_EPOCH=$(python3 -c "
import torch, glob, os
ckpt = torch.load('models/${MODEL}_best.pth', map_location='cpu', weights_only=True)
print(ckpt.get('epoch', 30))
" 2>/dev/null || echo "30")

    echo "Best epoch for ${MODEL}: ${BEST_EPOCH}"

    # ─── Step 2: Train on 100% data with best epoch count ───
    echo ""
    echo "=== STEP 2 [${MODEL}]: Training on FULL data (${BEST_EPOCH} epochs, --full-train) ==="
    echo "Started: $(date)"

    python3 -m src.train --model "$MODEL" --data-dir data --epochs "$BEST_EPOCH" --num-workers 0 --full-train 2>&1 | tee "/tmp/${MODEL}_step2.log"

    echo ""
    echo "=== STEP 2 [${MODEL}] Complete ==="
    echo "=== ${MODEL} All Done: $(date) ==="
    echo "Final model saved to: models/${MODEL}_best.pth"
done

echo ""
echo "============================================================"
echo "=== ALL MODELS TRAINING COMPLETE: $(date) ==="
echo "============================================================"
