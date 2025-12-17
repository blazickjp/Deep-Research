#!/usr/bin/env python3
"""
Measure FF Gram conditioning for HF baseline model.
Uses the downloaded model architecture and weights directly.
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add current directory to path to import hf_model
sys.path.insert(0, str(Path(__file__).parent))
from hf_model import GPT, GPTConfig

console = Console()

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    console.print("[red]datasets not installed. Install with: pip install datasets[/red]")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    console.print("[red]transformers not installed. Install with: pip install transformers[/red]")
    sys.exit(1)


def compute_gram_condition(features):
    """
    Compute Gram matrix condition number for a batch of features.

    Args:
        features: [B, T, D] tensor

    Returns:
        condition_number: float
    """
    B, T, D = features.shape

    # Reshape to [B*T, D]
    X = features.reshape(-1, D)

    # Compute Gram matrix: G = X^T X / (B*T)
    G = (X.t() @ X) / (B * T)

    # Add small regularization for numerical stability
    G = G + 1e-5 * torch.eye(D, device=G.device)

    # Compute eigenvalues
    try:
        eigenvalues = torch.linalg.eigvalsh(G)
    except:
        # Fallback to CPU
        eigenvalues = torch.linalg.eigvalsh(G.cpu()).to(G.device)

    # Condition number = max eigenvalue / min eigenvalue
    max_eig = eigenvalues.max()
    min_eig = eigenvalues.min()

    condition_number = (max_eig / min_eig).item()

    return condition_number


def prepare_validation_data(max_docs=1000, block_size=256, batch_size=16):
    """Prepare TinyStories validation data."""
    console.print(f"[cyan]Loading TinyStories dataset (max_docs={max_docs})...[/cyan]")
    ds = load_dataset("roneneldan/TinyStories")

    # Use validation split
    val_text = ds["validation"]["text"][:max_docs] if max_docs > 0 else ds["validation"]["text"]

    console.print(f"[cyan]Tokenizing {len(val_text)} validation stories...[/cyan]")

    # Use GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Tokenize and concatenate
    def enc(s): return tokenizer.encode(s, add_special_tokens=False)
    ids = [torch.tensor(enc(s), dtype=torch.long) for s in val_text if len(s) > 0]

    stream = torch.cat([x for x in ids if len(x) > 0])
    console.print(f"[green]Tokenized stream: {len(stream):,} tokens[/green]")

    # Create non-overlapping chunks
    toks = stream.unfold(0, block_size+1, block_size+1)  # [N, block_size+1]

    console.print(f"[green]Created {toks.size(0)} validation examples[/green]")

    # Create simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __len__(self):
            return toks.size(0)
        def __getitem__(self, i):
            seq = toks[i]
            return seq[:-1], seq[1:]

    loader = torch.utils.data.DataLoader(
        SimpleDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    console.print(f"[green]Created validation loader: {len(loader)} batches[/green]")
    return loader


def main():
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Measuring HuggingFace Baseline Model Conditioning[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Load config
    console.print("[cyan]Loading model configuration...[/cyan]")
    config_path = Path(__file__).parent / "hf_config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    # Create GPTConfig from their format
    config = GPTConfig(
        block_size=config_dict.get("block_size", 256),
        vocab_size=config_dict.get("vocab_size", 50257),
        n_layer=config_dict.get("n_layer", 6),
        n_head=config_dict.get("n_head", 6),
        n_embd=config_dict.get("n_embd", 384),
        dropout=config_dict.get("dropout", 0.1),
    )

    console.print(f"[green]Config loaded:[/green]")
    console.print(f"  Layers: {config.n_layer}")
    console.print(f"  Hidden size: {config.n_embd}")
    console.print(f"  Heads: {config.n_head}")
    console.print(f"  Block size: {config.block_size}")

    # Create model
    console.print("\n[cyan]Creating model...[/cyan]")
    model = GPT(config)

    # Load weights
    console.print("[cyan]Loading model weights...[/cyan]")
    weights_path = Path(__file__).parent / "hf_pytorch_model.bin"

    try:
        # Try with weights_only for security
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    except:
        # Fallback for older torch versions
        console.print("[yellow]Using legacy torch.load (weights_only not available)[/yellow]")
        state_dict = torch.load(weights_path, map_location='cpu')

    # Load state dict
    model.load_state_dict(state_dict)
    console.print("[green]Weights loaded successfully![/green]")

    # Move to device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    console.print(f"[green]Model moved to {device}[/green]")

    # Load validation data (use model's block_size from config)
    val_loader = prepare_validation_data(max_docs=500, block_size=config.block_size, batch_size=16)

    # Collect FFN features from all layers
    console.print(f"\n[cyan]Running inference to collect FFN features...[/cyan]")

    layer_conditions = []

    with torch.no_grad():
        for layer_idx in range(config.n_layer):
            console.print(f"[cyan]Processing layer {layer_idx}...[/cyan]")
            all_features = []

            for batch_idx, (xb, yb) in enumerate(val_loader):
                if batch_idx >= 50:  # Limit to first 50 batches for speed
                    break

                xb = xb.to(device)

                # Forward through embeddings
                x = model.transformer.wte(xb) + model.transformer.wpe(torch.arange(xb.size(1), device=device))
                x = model.transformer.drop(x)

                # Process through blocks
                for i, block in enumerate(model.transformer.h):
                    if i < layer_idx:
                        # Run full block for earlier layers
                        x = block(x)
                    elif i == layer_idx:
                        # Extract FFN features from target layer
                        x = x + block.attn(block.ln1(x))
                        x_norm = block.ln2(x)
                        ffn_features = block.mlp.gelu(block.mlp.c_fc(x_norm))  # [B, T, 4*n_embd]
                        all_features.append(ffn_features.cpu())
                        break  # Don't need to process further layers

            # Concatenate all batches for this layer
            layer_features = torch.cat(all_features, dim=0)  # [N, T, 4*n_embd]

            # Compute condition number
            cond = compute_gram_condition(layer_features)
            layer_conditions.append(cond)

            console.print(f"  [green]Layer {layer_idx}: FF Gram cond = {cond:.0f}[/green]")

    # Compute statistics
    avg_cond = np.mean(layer_conditions)
    max_cond = np.max(layer_conditions)
    min_cond = np.min(layer_conditions)

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"  Average FF Gram cond: {avg_cond:.0f}")
    console.print(f"  Max FF Gram cond:     {max_cond:.0f}")
    console.print(f"  Min FF Gram cond:     {min_cond:.0f}")

    # Print comparison table
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Comparison with Your Model[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Avg FF Gram Cond", justify="right")
    table.add_column("Val PPL", justify="right")
    table.add_column("Training Method", style="dim")

    table.add_row(
        "HF Baseline",
        f"{avg_cond:.0f}",
        "10.9",
        "Standard (no geometry)"
    )

    table.add_row(
        "Your Model (Epoch 10)",
        "189",
        "6.20",
        "GLT + Gram regularization"
    )

    console.print(table)

    improvement_cond = avg_cond / 189.0
    improvement_ppl = 10.9 / 6.20

    console.print(f"\n[bold green]Your model has:[/bold green]")
    console.print(f"  {improvement_cond:.1f}x better conditioning")
    console.print(f"  {improvement_ppl:.1f}x better perplexity")

    console.print(f"\n[dim]Note: Lower condition numbers indicate better-conditioned features[/dim]")

    # Save results
    results = {
        "hf_model": {
            "layer_conditions": layer_conditions,
            "average_condition": float(avg_cond),
            "max_condition": float(max_cond),
            "min_condition": float(min_cond),
            "val_ppl": 10.9
        },
        "your_model": {
            "average_condition": 189,
            "val_ppl": 6.20
        }
    }

    results_path = Path(__file__).parent.parent / "artifacts_ts" / "conditioning_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to: {results_path}[/green]")


if __name__ == "__main__":
    main()
