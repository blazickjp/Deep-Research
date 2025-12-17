#!/usr/bin/env python3
"""
Analyze layer-wise conditioning from retrofit experiments.

Creates visualizations and analysis of how conditioning varies across layers.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def load_layerwise_data(results_dir):
    """Load layer-wise conditioning data from all methods."""
    results_dir = Path(results_dir)
    data = {}

    for method in ["standard", "gram_only", "geometry"]:
        result_file = results_dir / f"{method}.json"
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
                if 'layer_conditions' in result:
                    data[method] = {
                        'layer_conditions': result['layer_conditions'],
                        'final_cond': result['final']['cond']
                    }

    return data


def plot_layerwise_conditioning(data, output_path):
    """Create visualization of layer-wise conditioning."""
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

    # Plot 1: Layer-wise conditioning
    for method, method_data in data.items():
        layer_conds = method_data['layer_conditions']
        layers = list(range(len(layer_conds)))

        ax1.plot(
            layers,
            layer_conds,
            label=method_names.get(method, method),
            color=colors.get(method, 'gray'),
            linewidth=2.5,
            marker='o',
            markersize=8
        )

    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FF Gram Conditioning', fontsize=12, fontweight='bold')
    ax1.set_title('Layer-wise Conditioning Across Network Depth', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(6))

    # Plot 2: Conditioning improvement per layer (vs Standard)
    if 'standard' in data:
        std_conds = np.array(data['standard']['layer_conditions'])

        for method in ['gram_only', 'geometry']:
            if method in data:
                method_conds = np.array(data[method]['layer_conditions'])
                improvement = ((std_conds - method_conds) / std_conds) * 100  # % improvement

                ax2.bar(
                    np.arange(len(improvement)) + (0.2 if method == 'gram_only' else -0.2),
                    improvement,
                    width=0.35,
                    label=method_names.get(method, method),
                    color=colors.get(method, 'gray'),
                    alpha=0.8
                )

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Conditioning Improvement vs Standard (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Per-Layer Conditioning Gains', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xticks(range(6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]Plot saved to: {output_path}[/green]")


def create_analysis_table(data):
    """Create table analyzing layer-wise patterns."""
    table = Table(show_header=True, header_style="bold magenta", title="Layer-wise Conditioning Analysis")
    table.add_column("Layer", style="cyan", justify="center")
    table.add_column("Standard SGD", justify="right")
    table.add_column("Gram Only", justify="right")
    table.add_column("Full Geometry", justify="right")
    table.add_column("Best Method", style="green")

    # Get max number of layers
    max_layers = max(len(d['layer_conditions']) for d in data.values())

    for layer_idx in range(max_layers):
        std_cond = data.get('standard', {}).get('layer_conditions', [None]*(layer_idx+1))[layer_idx]
        gram_cond = data.get('gram_only', {}).get('layer_conditions', [None]*(layer_idx+1))[layer_idx]
        geom_cond = data.get('geometry', {}).get('layer_conditions', [None]*(layer_idx+1))[layer_idx]

        # Find best (lowest) conditioning
        values = []
        if std_cond is not None:
            values.append(('Standard', std_cond))
        if gram_cond is not None:
            values.append(('Gram', gram_cond))
        if geom_cond is not None:
            values.append(('Geometry', geom_cond))

        best = min(values, key=lambda x: x[1])[0] if values else "N/A"

        table.add_row(
            f"Layer {layer_idx}",
            f"{std_cond:.0f}" if std_cond is not None else "—",
            f"{gram_cond:.0f}" if gram_cond is not None else "—",
            f"{geom_cond:.0f}" if geom_cond is not None else "—",
            best
        )

    return table


def analyze_patterns(data):
    """Analyze patterns in layer-wise conditioning."""
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print(f"[bold green]Layer-wise Pattern Analysis[/bold green]")
    console.print(f"[bold green]{'='*70}[/bold green]\n")

    for method, method_data in data.items():
        layer_conds = np.array(method_data['layer_conditions'])

        console.print(f"\n[bold cyan]{method.upper()}:[/bold cyan]")

        # Overall stats
        console.print(f"  Min conditioning: {layer_conds.min():.0f} (Layer {layer_conds.argmin()})")
        console.print(f"  Max conditioning: {layer_conds.max():.0f} (Layer {layer_conds.argmax()})")
        console.print(f"  Mean conditioning: {layer_conds.mean():.0f}")
        console.print(f"  Std conditioning: {layer_conds.std():.0f}")

        # Depth pattern
        early_layers = layer_conds[:2].mean()
        late_layers = layer_conds[-2:].mean()
        ratio = early_layers / late_layers

        console.print(f"  Early layers (0-1) avg: {early_layers:.0f}")
        console.print(f"  Late layers (4-5) avg: {late_layers:.0f}")
        console.print(f"  Early/Late ratio: {ratio:.2f}x")

    # Comparative analysis
    if 'standard' in data and 'geometry' in data:
        console.print(f"\n[bold yellow]Geometry vs Standard Comparison:[/bold yellow]")

        std_conds = np.array(data['standard']['layer_conditions'])
        geom_conds = np.array(data['geometry']['layer_conditions'])

        improvements = ((std_conds - geom_conds) / std_conds) * 100

        console.print(f"  Layer 0 (embedding): {improvements[0]:.1f}% improvement")
        console.print(f"  Layer 1: {improvements[1]:.1f}% improvement")
        console.print(f"  Middle layers (2-3): {improvements[2:4].mean():.1f}% improvement")
        console.print(f"  Final layers (4-5): {improvements[4:].mean():.1f}% improvement")

        # Which layers benefit most?
        best_layer = improvements.argmax()
        worst_layer = improvements.argmin()

        console.print(f"\n  Most improved layer: {best_layer} ({improvements[best_layer]:.1f}%)")
        console.print(f"  Least improved layer: {worst_layer} ({improvements[worst_layer]:.1f}%)")


def create_summary(data):
    """Create summary statistics."""
    summary = {}

    for method, method_data in data.items():
        layer_conds = np.array(method_data['layer_conditions'])

        summary[method] = {
            'layer_conditions': method_data['layer_conditions'],
            'min': float(layer_conds.min()),
            'max': float(layer_conds.max()),
            'mean': float(layer_conds.mean()),
            'std': float(layer_conds.std()),
            'early_layers_mean': float(layer_conds[:2].mean()),
            'late_layers_mean': float(layer_conds[-2:].mean()),
            'early_late_ratio': float(layer_conds[:2].mean() / layer_conds[-2:].mean())
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze layer-wise conditioning")
    parser.add_argument("--results_dir", type=str, default="artifacts_ts/retrofit_comparison",
                       help="Directory with result JSON files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for plots and analysis")

    args = parser.parse_args()

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Layer-wise Conditioning Analysis[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Load data
    data = load_layerwise_data(args.results_dir)

    if not data:
        console.print("[red]No layer-wise data found![/red]")
        return

    console.print(f"[cyan]Loaded layer-wise data for {len(data)} methods[/cyan]\n")

    # Create table
    table = create_analysis_table(data)
    console.print(table)

    # Analyze patterns
    analyze_patterns(data)

    # Create plots
    output_dir = Path(args.output) if args.output else Path(args.results_dir)
    plot_path = output_dir / "layerwise_conditioning.png"
    plot_layerwise_conditioning(data, plot_path)

    # Save summary
    summary = create_summary(data)
    summary_path = output_dir / "layerwise_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"[green]Summary saved to: {summary_path}[/green]\n")


if __name__ == "__main__":
    main()
