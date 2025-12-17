# Geometry Retrofit Experiments

Tests whether geometry-aware training can improve a pre-trained model's conditioning.

## Quick Start

### Run All Three Experiments (Recommended)

```bash
cd /Users/jblazick/Documents/deep-research
./experiments/run_retrofit_comparison.sh
```

This runs:
1. **Standard training** - Control (continued SGD)
2. **Gram only** - Gram regularization without GLT
3. **Full geometry** - GLT + Gram regularization

Each runs for 1000 steps (~2-3 hours per experiment, ~8 hours total).

Results saved to: `artifacts_ts/retrofit_comparison/`

### Run Single Experiment

```bash
# Standard training (control)
uv run python experiments/geometry_retrofit.py \
  --method standard \
  --num_steps 1000

# Gram regularization only
uv run python experiments/geometry_retrofit.py \
  --method gram \
  --num_steps 1000 \
  --gram_weight 0.01

# Full geometry (GLT + Gram)
uv run python experiments/geometry_retrofit.py \
  --method geometry \
  --num_steps 1000 \
  --gram_weight 0.01 \
  --glt_interval 50
```

### Quick Test (200 steps, ~30 min)

```bash
uv run python experiments/geometry_retrofit.py \
  --method geometry \
  --num_steps 200 \
  --eval_interval 50
```

## What Gets Measured

Every 100 steps (configurable with `--eval_interval`):
- **FF Gram conditioning** - Average across all layers
- **Validation perplexity**
- **Validation loss**

## Expected Results

Based on theoretical analysis, we expect:

| Method | Final Conditioning | Final PPL | Improvement |
|--------|-------------------|-----------|-------------|
| HF Baseline | 4,738 | 10.9 | (starting point) |
| + Standard (1000 steps) | ~4,600 | ~10.7 | Minimal (~3%) |
| + Gram Only (1000 steps) | ~3,200 | ~9.8 | Moderate (~15%) |
| + Full Geometry (1000 steps) | ~2,500 | ~9.2 | Good (~30%) |
| Your Model (trained w/ geometry) | 189 | 6.20 | Best (96%) |

**Key insight:** Geometry can improve post-hoc, but starting with geometry is ~5x better.

## Command Line Options

```bash
python experiments/geometry_retrofit.py [OPTIONS]

Required:
  --method {standard,gram,geometry}  Training method

Optional:
  --num_steps INT                    Training steps (default: 1000)
  --lr FLOAT                         Learning rate (default: 1e-4)
  --gram_weight FLOAT                Gram reg weight (default: 0.01)
  --glt_interval INT                 GLT transport interval (default: 50)
  --eval_interval INT                Evaluation interval (default: 100)
  --max_docs INT                     Max training docs (default: 1000)
  --batch_size INT                   Batch size (default: 16)
  --device {auto,cuda,mps,cpu}       Device (default: auto)
  --output PATH                      Output JSON file
```

## Output Files

After running experiments:

```
artifacts_ts/retrofit_comparison/
├── standard.json          # Standard training results
├── gram_only.json         # Gram-only results
├── geometry.json          # Full geometry results
├── summary.json           # Comparison summary
└── trajectories.png       # Visualization
```

Each JSON contains:
```json
{
  "method": "geometry",
  "initial": {
    "cond": 4738,
    "ppl": 10.9,
    "loss": 2.39
  },
  "final": {
    "cond": 2500,
    "ppl": 9.2,
    "loss": 2.22
  },
  "trajectory": {
    "steps": [0, 100, 200, ...],
    "conditioning": [4738, 4200, 3800, ...],
    "perplexity": [10.9, 10.5, 10.1, ...],
    "loss": [2.39, 2.35, 2.30, ...]
  }
}
```

## Analysis

After experiments complete, view results:

```bash
# View comparison table
uv run python experiments/compare_retrofit_results.py \
  --results_dir artifacts_ts/retrofit_comparison

# Or just check the summary
cat artifacts_ts/retrofit_comparison/summary.json | jq
```

## Interpreting Results

### Scenario 1: Geometry Rescues Well (Best Case)

```
Geometry achieves: cond ~1500-2000, PPL ~8-9
Standard achieves: cond ~4600, PPL ~10.7

→ Geometry-aware training can retrofit existing models
→ Huge practical impact (can improve deployed models)
```

### Scenario 2: Partial Rescue (Likely)

```
Geometry achieves: cond ~2500-3500, PPL ~9-10
Standard achieves: cond ~4600, PPL ~10.7

→ Geometry helps but limited by starting point
→ Starting with geometry from scratch is superior
→ Validates your training approach
```

### Scenario 3: Minimal Effect (Unlikely)

```
Geometry achieves: cond ~4200, PPL ~10.5
Standard achieves: cond ~4600, PPL ~10.7

→ Conditioning baked in early training
→ Must have geometry from the start
→ Still validates importance of geometry
```

## Troubleshooting

**CUDA out of memory:**
```bash
# Reduce batch size
--batch_size 8
```

**Takes too long:**
```bash
# Quick test with fewer steps
--num_steps 200 --max_docs 500
```

**Model not found:**
```bash
# Make sure HF model is downloaded
ls experiments/hf_pytorch_model.bin
# If missing, run measure_hf_conditioning_v2.py first
```

## Next Steps

After retrofit experiments:

1. **If geometry rescues well:**
   - Test on fine-tuning tasks
   - Apply to other pre-trained models
   - Publish "geometry retrofit" technique

2. **If partial rescue:**
   - Emphasize importance of training-from-scratch
   - Compare with your well-conditioned model
   - Show geometry must be maintained throughout

3. **Either way:**
   - Write up results for paper
   - Show conditioning trajectories
   - Discuss implications for training practices

## Implementation Notes

### Simplified GLT

The script uses a simplified GLT that normalizes rows/columns:

```python
def gradient_less_transport(model):
    for W in model.parameters():
        # Row normalization
        W = W / (row_norms + eps)
        # Column normalization
        W = W / (col_norms + eps)
        # Rescale to preserve magnitude
```

This is not the full Riemannian transport but captures the key idea: projecting onto balanced manifolds.

### Gram Regularization

Regularizes FFN weight matrices toward identity Gram matrix:

```python
def compute_gram_loss(model):
    for W in ffn_weights:
        G = W @ W.T
        I = eye(G.shape)
        loss += ||G - I||²
```

This encourages orthogonal, well-conditioned features.

## Time Estimates

On M-series Mac (MPS):
- **Quick test (200 steps)**: ~30 minutes
- **Single experiment (1000 steps)**: ~2-3 hours
- **Full comparison (3 × 1000 steps)**: ~8 hours

On CUDA GPU:
- **Quick test**: ~10 minutes
- **Single experiment**: ~45 minutes
- **Full comparison**: ~2.5 hours

## Questions?

Check the detailed analysis in:
- `docs/geometry_rescue_experiment.md` - Full theoretical analysis
- `docs/geometry_fine_tuning.md` - Fine-tuning implications

Or just run it and see what happens! 🚀
