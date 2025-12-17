#!/bin/bash
# Run full geometry retrofit comparison experiment
#
# This runs three training methods on the HF baseline model:
# 1. Standard continued training (control)
# 2. Gram regularization only
# 3. Full geometry (GLT + Gram)

set -e

STEPS=3000
MAX_DOCS=40000  # Increased for more diverse training data
BATCH_SIZE=16

echo "=================================="
echo "Geometry Retrofit Comparison"
echo "=================================="
echo ""
echo "Running 3 experiments with ${STEPS} steps each"
echo ""

# Create output directory
mkdir -p artifacts_ts/retrofit_comparison

# 1. Standard training (control)
echo "[1/3] Running STANDARD training..."
uv run python experiments/geometry_retrofit.py \
  --model_path experiments/hf_pytorch_model.bin \
  --model_type hf \
  --method standard \
  --num_steps $STEPS \
  --max_docs $MAX_DOCS \
  --batch_size $BATCH_SIZE \
  --eval_interval 100 \
  --output artifacts_ts/retrofit_comparison/standard.json

echo ""
echo "[1/3] Standard training complete!"
echo ""

# 2. Gram regularization only
echo "[2/3] Running GRAM-ONLY training..."
uv run python experiments/geometry_retrofit.py \
  --model_path experiments/hf_pytorch_model.bin \
  --model_type hf \
  --method gram \
  --num_steps $STEPS \
  --max_docs $MAX_DOCS \
  --batch_size $BATCH_SIZE \
  --gram_weight 0.01 \
  --eval_interval 100 \
  --output artifacts_ts/retrofit_comparison/gram_only.json

echo ""
echo "[2/3] Gram-only training complete!"
echo ""

# 3. Full geometry (GLT + Gram)
echo "[3/3] Running FULL GEOMETRY training..."
uv run python experiments/geometry_retrofit.py \
  --model_path experiments/hf_pytorch_model.bin \
  --model_type hf \
  --method geometry \
  --num_steps $STEPS \
  --max_docs $MAX_DOCS \
  --batch_size $BATCH_SIZE \
  --gram_weight 0.01 \
  --glt_interval 50 \
  --eval_interval 100 \
  --output artifacts_ts/retrofit_comparison/geometry.json

echo ""
echo "[3/3] Full geometry training complete!"
echo ""

# Create comparison summary
echo "=================================="
echo "Creating comparison summary..."
echo "=================================="

uv run python experiments/compare_retrofit_results.py \
  --results_dir artifacts_ts/retrofit_comparison \
  --output artifacts_ts/retrofit_comparison/summary.json

echo ""
echo "All experiments complete!"
echo "Results saved to: artifacts_ts/retrofit_comparison/"
echo ""
