#!/usr/bin/env bash
# ============================================================================
# train_tuned.sh — Optimized training for improved unhealthy food recall
# ============================================================================
#
# BASELINE (before tuning):
#   Accuracy:  72%
#   Recall (unhealthy): 65%
#   Dataset: ~7500 frames
#   Config: lr=1e-4, epochs=10, dropout=0.3, no class weighting
#
# CHANGES & RATIONALE:
#
# 1. DATASET ENRICHMENT
#    - Added 27 new unhealthy queries (from 23 to 50) covering:
#      fried food, sugary desserts, sugary drinks, processed snacks,
#      street food, buffet/mukbang scenes
#    - Added 7 new healthy queries for balance
#    - Why: 35% miss rate on unhealthy suggests insufficient visual diversity
#
# 2. DATA AUGMENTATION (in dataset.py)
#    - RandomResizedCrop scale: (0.8,1.0) -> (0.6,1.0) — more aggressive crop
#    - Added RandomPerspective(0.2, p=0.3) — simulates camera angles
#    - ColorJitter: 0.3 -> 0.4, hue: 0.05 -> 0.1 — handles lighting variety
#    - Added GaussianBlur(k=3) — simulates low-quality video
#    - RandomErasing: p=0.2 -> 0.3, scale up to 0.2 — occlusion robustness
#    - Why: Unhealthy food appears in diverse settings (restaurants, street,
#      delivery boxes) with varied lighting and camera quality
#
# 3. HYPERPARAMETERS
#    - Learning rate: 1e-4 -> 3e-5 (lower LR for finer convergence)
#    - Epochs: 10 -> 25 (more time to learn subtle features)
#    - Patience: 5 -> 7 (allow plateaus before stopping)
#    - Dropout: 0.3 -> 0.4 (stronger regularization with more epochs)
#    - min-delta: 1e-4 -> 5e-5 (detect smaller improvements)
#    - Class weighting: OFF -> ON (compensate for class imbalance)
#    - LR warmup: 3 epochs linear warmup before cosine decay
#    - Why: Lower LR + more epochs lets the model learn harder examples;
#      class weighting directly addresses the recall gap
#
# EXPECTED OUTCOME:
#   Accuracy: 78-85%
#   Recall (unhealthy): 78-88%
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIT_ROOT="$(dirname "$SCRIPT_DIR")"

# Step 1: Download new videos and extract frames (skip if data exists)
echo "=== Step 1: Dataset enrichment ==="
python "$VIT_ROOT/generatedata.py" \
    --videos-per-keyword 15 \
    --max-frames-per-video 60 \
    --frame-size 224 \
    --min-frames 8

# Step 2: Train with tuned hyperparameters
echo ""
echo "=== Step 2: Training with tuned hyperparameters ==="
python "$VIT_ROOT/train.py" \
    --epochs 25 \
    --lr 3e-5 \
    --dropout 0.4 \
    --patience 7 \
    --min-delta 5e-5 \
    --batch-size 8 \
    --num-frames 8 \
    --img-size 224 \
    --class-weighting \
    --backbone auto \
    --weight-decay 1e-3 \
    --temporal-pool avg

# Step 3: Evaluate
echo ""
echo "=== Step 3: Evaluation ==="
python "$VIT_ROOT/test.py" \
    --output-dir results

echo ""
echo "=== Training complete. Check results/test_results.json for metrics ==="
