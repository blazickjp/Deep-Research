#!/usr/bin/env python3
"""
Analyze eigenvalue spectrum of Gram matrices from retrofit experiments.

This reveals the *shape* of the conditioning problem, not just its magnitude.
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hf_model import GPT, GPTConfig

console = Console()

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_model(checkpoint_path, config_path, device):
    """Load a model from checkpoint."""
    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    config = GPTConfig(
        block_size=config_dict["block_size"],
        vocab_size=config_dict["vocab_size"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        n_embd=config_dict["n_embd"],
        dropout=0.0,
        bias=config_dict["bias"]
    )

    # Create model
    model = GPT(config)

    # Load checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


def prepare_validation_data(max_docs=100, block_size=128, batch_size=16):
    """Prepare validation data."""
    if not HAS_DATASETS:
        console.print("[red]datasets library required[/red]")
        return None

    ds = load_dataset("roneneldan/TinyStories")
    text = ds["validation"]["text"][:max_docs]

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    all_tokens = []
    for story in text:
        tokens = tokenizer.encode(story, add_special_tokens=False)
        all_tokens.extend(tokens)

    # Create blocks
    blocks = []
    for i in range(0, len(all_tokens) - block_size, block_size):
        blocks.append(all_tokens[i:i+block_size])

    # Create batches
    dataset = torch.tensor(blocks[:batch_size*10], dtype=torch.long)  # 10 batches
    return dataset


@torch.no_grad()
def collect_activations(model, data, device, layer_idx):
    """Collect activations from a specific layer."""
    activations = []

    for i in range(0, len(data), 16):  # Process in batches of 16
        batch = data[i:i+16].to(device)
        if batch.size(0) == 0:
            continue

        # Forward pass with hooks
        features_list = []

        def hook_fn(module, input, output):
            # output is (B, T, C) for transformer block
            features_list.append(output.detach().cpu())

        # Register hook on the specific layer
        hook = model.transformer.h[layer_idx].register_forward_hook(hook_fn)

        try:
            _ = model(batch)
        finally:
            hook.remove()

        if features_list:
            activations.append(features_list[0])

    if activations:
        return torch.cat(activations, dim=0)  # (total_samples, T, C)
    return None


def compute_gram_eigenvalues(activations):
    """Compute eigenvalues of Gram matrix from activations."""
    # activations: (B, T, C)
    B, T, C = activations.shape
    X = activations.reshape(-1, C)  # (B*T, C)

    # Compute Gram matrix
    G = (X.t() @ X) / (B * T)
    G = G + 1e-5 * torch.eye(C)  # Small regularization

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(G)
    eigenvalues = eigenvalues.sort(descending=True)[0]  # Sort descending

    return eigenvalues.numpy()


def compute_effective_rank(eigenvalues):
    """Compute effective rank from eigenvalues."""
    # Normalize to probabilities
    eigenvalues_norm = eigenvalues / eigenvalues.sum()

    # Compute entropy
    entropy = -(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10)).sum()

    # Effective rank is exp(entropy)
    return np.exp(entropy)


def compute_participation_ratio(eigenvalues):
    """Compute participation ratio (another measure of effective dimensionality)."""
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()


def analyze_spectrum(eigenvalues, method_name):
    """Analyze eigenvalue spectrum."""
    max_eig = eigenvalues.max()
    min_eig = eigenvalues.min()
    condition = max_eig / min_eig

    effective_rank = compute_effective_rank(eigenvalues)
    participation_ratio = compute_participation_ratio(eigenvalues)

    # Compute cumulative explained variance
    cumsum = np.cumsum(eigenvalues)
    total = cumsum[-1]
    percent_90 = np.searchsorted(cumsum, 0.90 * total) + 1
    percent_95 = np.searchsorted(cumsum, 0.95 * total) + 1
    percent_99 = np.searchsorted(cumsum, 0.99 * total) + 1

    return {
        'max_eigenvalue': float(max_eig),
        'min_eigenvalue': float(min_eig),
        'condition_number': float(condition),
        'effective_rank': float(effective_rank),
        'participation_ratio': float(participation_ratio),
        'dimensions_for_90pct': int(percent_90),
        'dimensions_for_95pct': int(percent_95),
        'dimensions_for_99pct': int(percent_99),
        'total_dimensions': len(eigenvalues),
        'spectrum': eigenvalues.tolist()
    }


def plot_eigenvalue_spectra(all_spectra, output_path):
    """Create comprehensive eigenvalue spectrum visualization."""
    n_methods = len(all_spectra)
    n_layers = len(next(iter(all_spectra.values())))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    method_names = {
        "standard": "Standard SGD",
        "gram_only": "Gram Reg Only",
        "geometry": "Full Geometry"
    }

    colors = {
        "standard": "#ff7f0e",
        "gram_only": "#2ca02c",
        "geometry": "#1f77b4"
    }

    # Plot each layer
    for layer_idx in range(min(6, n_layers)):
        ax = axes[layer_idx]

        for method, layer_spectra in all_spectra.items():
            if layer_idx < len(layer_spectra):
                eigenvalues = layer_spectra[layer_idx]['spectrum']

                # Plot log-scale spectrum
                ax.semilogy(
                    eigenvalues,
                    label=method_names.get(method, method),
                    color=colors.get(method, 'gray'),
                    linewidth=2,
                    alpha=0.8
                )

        ax.set_xlabel('Eigenvalue Index', fontsize=10)
        ax.set_ylabel('Eigenvalue (log scale)', fontsize=10)
        ax.set_title(f'Layer {layer_idx} Spectrum', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]Spectrum plot saved to: {output_path}[/green]")


def plot_effective_dimensions(all_spectra, output_path):
    """Plot effective rank and participation ratio across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    method_names = {
        "standard": "Standard SGD",
        "gram_only": "Gram Reg Only",
        "geometry": "Full Geometry"
    }

    colors = {
        "standard": "#ff7f0e",
        "gram_only": "#2ca02c",
        "geometry": "#1f77b4"
    }

    # Plot effective rank
    for method, layer_spectra in all_spectra.items():
        layers = list(range(len(layer_spectra)))
        eff_ranks = [s['effective_rank'] for s in layer_spectra]

        ax1.plot(
            layers,
            eff_ranks,
            label=method_names.get(method, method),
            color=colors.get(method, 'gray'),
            linewidth=2.5,
            marker='o',
            markersize=8
        )

    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Effective Rank', fontsize=12, fontweight='bold')
    ax1.set_title('Effective Feature Dimensionality', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot dimensions for 90% variance
    for method, layer_spectra in all_spectra.items():
        layers = list(range(len(layer_spectra)))
        dims_90 = [s['dimensions_for_90pct'] for s in layer_spectra]

        ax2.plot(
            layers,
            dims_90,
            label=method_names.get(method, method),
            color=colors.get(method, 'gray'),
            linewidth=2.5,
            marker='s',
            markersize=8
        )

    ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dimensions for 90% Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Space Utilization', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]Effective dimensions plot saved to: {output_path}[/green]")


def create_summary_table(all_spectra):
    """Create table summarizing spectrum analysis."""
    table = Table(show_header=True, header_style="bold magenta", title="Eigenvalue Spectrum Summary")
    table.add_column("Layer", style="cyan", justify="center")
    table.add_column("Method", style="yellow")
    table.add_column("Eff. Rank", justify="right")
    table.add_column("90% Dims", justify="right")
    table.add_column("Condition", justify="right")

    for layer_idx in range(6):
        for method in ["standard", "gram_only", "geometry"]:
            if method in all_spectra and layer_idx < len(all_spectra[method]):
                spectrum = all_spectra[method][layer_idx]

                method_name = {
                    "standard": "Standard",
                    "gram_only": "Gram",
                    "geometry": "Geometry"
                }[method]

                table.add_row(
                    f"{layer_idx}" if method == "standard" else "",
                    method_name,
                    f"{spectrum['effective_rank']:.1f}",
                    f"{spectrum['dimensions_for_90pct']}",
                    f"{spectrum['condition_number']:.0f}"
                )

        if layer_idx < 5:
            table.add_row("", "", "", "", "")  # Separator

    return table


def main():
    parser = argparse.ArgumentParser(description="Analyze eigenvalue spectrum")
    parser.add_argument("--results_dir", type=str, default="artifacts_ts/retrofit_comparison",
                       help="Directory with model checkpoints")
    parser.add_argument("--config", type=str, default="experiments/hf_config.json",
                       help="Model config file")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--max_docs", type=int, default=100,
                       help="Number of validation docs to use")

    args = parser.parse_args()

    device = args.device
    console.print(f"\n[cyan]Using device: {device}[/cyan]")

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Eigenvalue Spectrum Analysis[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Prepare data
    console.print("[cyan]Preparing validation data...[/cyan]")
    data = prepare_validation_data(max_docs=args.max_docs)

    if data is None:
        console.print("[red]Failed to prepare data[/red]")
        return

    console.print(f"[green]Loaded {len(data)} sequences[/green]\n")

    # Analyze each method
    all_spectra = {}
    results_dir = Path(args.results_dir)

    for method in ["standard", "gram_only", "geometry"]:
        # Map method names to checkpoint names
        checkpoint_name = "gram_model.pt" if method == "gram_only" else f"{method}_model.pt"
        checkpoint_path = results_dir / checkpoint_name

        if not checkpoint_path.exists():
            console.print(f"[yellow]Skipping {method} - checkpoint not found[/yellow]")
            continue

        console.print(f"[bold cyan]Analyzing {method.upper()}...[/bold cyan]")

        # Load model
        model, config = load_model(checkpoint_path, args.config, device)

        # Collect spectra for each layer
        layer_spectra = []

        for layer_idx in range(config.n_layer):
            console.print(f"  Layer {layer_idx}...", end=" ")

            # Collect activations
            activations = collect_activations(model, data, device, layer_idx)

            if activations is not None:
                # Compute eigenvalues
                eigenvalues = compute_gram_eigenvalues(activations)

                # Analyze
                analysis = analyze_spectrum(eigenvalues, f"{method}_layer{layer_idx}")

                layer_spectra.append(analysis)
                console.print(f"[green]Eff. Rank: {analysis['effective_rank']:.1f}[/green]")
            else:
                console.print("[red]Failed[/red]")

        all_spectra[method] = layer_spectra

    # Create visualizations
    console.print(f"\n[cyan]Creating visualizations...[/cyan]")

    output_dir = results_dir
    spectrum_plot = output_dir / "eigenvalue_spectra.png"
    plot_eigenvalue_spectra(all_spectra, spectrum_plot)

    dims_plot = output_dir / "effective_dimensions.png"
    plot_effective_dimensions(all_spectra, dims_plot)

    # Create summary table
    console.print("\n")
    table = create_summary_table(all_spectra)
    console.print(table)

    # Save results
    output_path = output_dir / "spectrum_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(all_spectra, f, indent=2)

    console.print(f"\n[green]Analysis saved to: {output_path}[/green]\n")


if __name__ == "__main__":
    main()
