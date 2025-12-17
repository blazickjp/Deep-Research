#!/bin/bash
# Verify the sweep issue by running a quick test with exact sweep parameters

echo "========================================================================"
echo "Testing TinyStories with EXACT sweep parameters"
echo "========================================================================"
echo ""
echo "This will run the base case with the same parameters as your sweep:"
echo "  - max_docs: 10000"
echo "  - epochs: 2"
echo "  - batch: 32"
echo "  - lr: 3e-4"
echo "  - seed: 0 (same as first sweep run)"
echo ""

# Run with exact sweep parameters
uv run python experiments/tinystories_gpt_geom.py \
  --epochs 2 \
  --batch 32 \
  --lr 3e-4 \
  --max_docs 10000 \
  --seed 0 \
  --run_name verify_sweep_params

echo ""
echo "========================================================================"
echo "Now testing with MORE data (max_docs=50000)"
echo "========================================================================"
echo ""

uv run python experiments/tinystories_gpt_geom.py \
  --epochs 2 \
  --batch 32 \
  --lr 3e-4 \
  --max_docs 50000 \
  --seed 0 \
  --run_name verify_more_data

echo ""
echo "========================================================================"
echo "COMPARISON"
echo "========================================================================"
echo "If max_docs=10000 gives PPL ~30-50 (bad)"
echo "But max_docs=50000 gives PPL ~3-5 (good)"
echo "Then the issue is: SWEEP IS USING TOO LITTLE DATA!"
echo ""
echo "Solution: Update your sweep script to use --max_docs 50000 or more"
