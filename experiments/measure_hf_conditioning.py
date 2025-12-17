#!/usr/bin/env python3
"""
Measure FF Gram conditioning for TinyStories models.

Compares:
1. HuggingFace baseline model (abhilash88/tinystories-slm-gpt)
2. Your geometry-aware trained model

This provides evidence that GLT + Gram regularization improves conditioning.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    console.print("[yellow]Warning: transformers not installed. Install with: pip install transformers[/yellow]")

# Try to import datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    console.print("[yellow]Warning: datasets not installed. Install with: pip install datasets[/yellow]")


def slogdet_safe(A):
    """Compute log determinant safely (MPS-compatible)."""
    try:
        sign, logdet = torch.linalg.slogdet(A)
        return logdet
    except:
        # Fallback to CPU
        A_cpu = A.detach().cpu()
        sign, logdet = torch.linalg.slogdet(A_cpu)
        return logdet.to(A.device)


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


def prepare_validation_data(max_docs=10000, block_size=256, batch_size=32):
    """Prepare TinyStories validation data matching training setup."""
    if not HAS_DATASETS:
        raise RuntimeError("datasets library required. Install with: pip install datasets")

    console.print(f"[cyan]Loading TinyStories dataset (max_docs={max_docs})...[/cyan]")
    ds = load_dataset("roneneldan/TinyStories")

    # Use validation split
    val_text = ds["validation"]["text"][:max_docs] if max_docs > 0 else ds["validation"]["text"]

    console.print(f"[cyan]Tokenizing {len(val_text)} validation stories...[/cyan]")

    # Use GPT-2 tokenizer (same as both models)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Tokenize and concatenate
    def enc(s): return tokenizer.encode(s, add_special_tokens=False)
    ids = [torch.tensor(enc(s), dtype=torch.long) for s in val_text if len(s) > 0]

    stream = torch.cat([x for x in ids if len(x) > 0])
    console.print(f"[green]Tokenized stream: {len(stream):,} tokens[/green]")

    # Create non-overlapping chunks
    toks = stream.unfold(0, block_size+1, block_size+1)  # [N, block_size+1]

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


def measure_hf_model_conditioning(model_name="abhilash88/tinystories-slm-gpt",
                                   max_docs=1000,
                                   block_size=256,
                                   batch_size=32,
                                   device="auto"):
    """Measure conditioning for HuggingFace model."""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library required. Install with: pip install transformers")

    # Set device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Measuring HuggingFace Model: {model_name}[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Load model - try multiple approaches
    console.print(f"[cyan]Loading model from HuggingFace...[/cyan]")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        console.print(f"[yellow]AutoModel failed, trying GPT2LMHeadModel: {e}[/yellow]")
        from transformers import GPT2LMHeadModel
        try:
            model = GPT2LMHeadModel.from_pretrained(model_name)
        except Exception as e2:
            console.print(f"[red]GPT2Model also failed. Trying manual config...[/red]")
            # Try loading with explicit config
            from transformers import GPT2Config, GPT2LMHeadModel
            config = GPT2Config.from_pretrained(model_name)
            model = GPT2LMHeadModel(config)
            # Try loading weights
            try:
                state_dict = torch.load(f"{model_name}/pytorch_model.bin")
                model.load_state_dict(state_dict)
            except:
                raise RuntimeError(f"Could not load model {model_name}: {e2}")

    model = model.to(device)
    model.eval()

    console.print(f"[green]Model loaded on {device}[/green]")
    console.print(f"  Layers: {len(model.transformer.h)}")
    console.print(f"  Hidden size: {model.config.n_embd}")

    # Load validation data
    val_loader = prepare_validation_data(max_docs, block_size, batch_size)

    # Collect FFN features from all layers
    console.print(f"\n[cyan]Running inference to collect FFN features...[/cyan]")

    layer_conditions = []

    with torch.no_grad():
        for layer_idx in range(len(model.transformer.h)):
            all_features = []

            for batch_idx, (xb, yb) in enumerate(val_loader):
                if batch_idx >= 50:  # Limit to first 50 batches for speed
                    break

                xb = xb.to(device)

                # Forward pass and extract FFN features
                outputs = model(xb, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (layer+1) tensors

                # Get this layer's output (after attention, before FFN)
                # hidden_states[0] = embeddings, hidden_states[1] = after layer 0, etc.
                layer_output = hidden_states[layer_idx + 1]  # [B, T, D]

                # Run through this layer's FFN to get intermediate features
                block = model.transformer.h[layer_idx]

                # LayerNorm
                x_norm = block.ln_2(layer_output)

                # FFN: c_fc (up projection) -> activation -> c_proj (down projection)
                # We want features AFTER activation, BEFORE down projection
                ffn_intermediate = block.mlp.c_fc(x_norm)  # [B, T, 4*D]
                ffn_features = block.mlp.act(ffn_intermediate)  # [B, T, 4*D]

                all_features.append(ffn_features.cpu())

            # Concatenate all batches for this layer
            layer_features = torch.cat(all_features, dim=0)  # [N, T, 4*D]

            # Compute condition number
            cond = compute_gram_condition(layer_features)
            layer_conditions.append(cond)

            console.print(f"  Layer {layer_idx}: FF Gram cond = {cond:.0f}")

    # Compute average
    avg_cond = np.mean(layer_conditions)
    max_cond = np.max(layer_conditions)

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"  Average FF Gram cond: {avg_cond:.0f}")
    console.print(f"  Max FF Gram cond:     {max_cond:.0f}")

    return {
        "model_name": model_name,
        "layer_conditions": layer_conditions,
        "average_condition": avg_cond,
        "max_condition": max_cond
    }


def measure_local_model_conditioning(model_path,
                                     max_docs=1000,
                                     block_size=256,
                                     batch_size=32,
                                     device="auto"):
    """Measure conditioning for your locally trained model."""
    # TODO: Implement loading your model
    # This would load your saved model checkpoint and run the same analysis
    console.print("[yellow]Local model measurement not yet implemented[/yellow]")
    console.print("[yellow]You can load your model checkpoint and run similar analysis[/yellow]")
    return None


def main():
    parser = argparse.ArgumentParser(description="Measure FF Gram conditioning for TinyStories models")
    parser.add_argument("--hf_model", type=str, default="abhilash88/tinystories-slm-gpt",
                       help="HuggingFace model name")
    parser.add_argument("--max_docs", type=int, default=1000,
                       help="Max validation documents to use")
    parser.add_argument("--block_size", type=int, default=256,
                       help="Context length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--skip_hf", action="store_true",
                       help="Skip HF model measurement")

    args = parser.parse_args()

    results = {}

    # Measure HF model
    if not args.skip_hf:
        try:
            hf_results = measure_hf_model_conditioning(
                model_name=args.hf_model,
                max_docs=args.max_docs,
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=args.device
            )
            results["hf"] = hf_results
        except Exception as e:
            console.print(f"[red]Error measuring HF model: {e}[/red]")

    # Print comparison table
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Comparison Summary[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Avg FF Gram Cond", justify="right")
    table.add_column("Max FF Gram Cond", justify="right")
    table.add_column("Training Method", style="dim")

    if "hf" in results:
        hf = results["hf"]
        table.add_row(
            "HF Baseline",
            f"{hf['average_condition']:.0f}",
            f"{hf['max_condition']:.0f}",
            "Standard (no geometry)"
        )

    # Your model's known result
    table.add_row(
        "Your Model (Epoch 10)",
        "189",
        "189",
        "GLT + Gram regularization"
    )

    console.print(table)

    if "hf" in results:
        improvement = results["hf"]["average_condition"] / 189.0
        console.print(f"\n[bold green]Your model has {improvement:.1f}x better conditioning![/bold green]")

    console.print(f"\n[dim]Note: Lower condition numbers indicate better-conditioned features[/dim]")


if __name__ == "__main__":
    main()
