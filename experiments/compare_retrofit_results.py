#!/usr/bin/env python3
"""
Compare geometry retrofit results across different methods.
"""

import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import numpy as np

console = Console()


def load_results(results_dir):
    """Load all experiment results."""
    results_dir = Path(results_dir)

    results = {}
    for method in ["standard", "gram_only", "geometry"]:
        result_file = results_dir / f"{method}.json"
        if result_file.exists():
            with open(result_file) as f:
                results[method] = json.load(f)

    return results


def create_comparison_table(results):
    """Create rich table comparing methods."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Method", style="cyan")
    table.add_column("Initial Cond", justify="right")
    table.add_column("Final Cond", justify="right")
    table.add_column("Δ Cond", justify="right", style="green")
    table.add_column("Initial Val PPL", justify="right")
    table.add_column("Final Val PPL", justify="right")
    table.add_column("Initial Train PPL", justify="right")
    table.add_column("Final Train PPL", justify="right")
    table.add_column("% Improvement", justify="right", style="bold green")

    method_names = {
        "standard": "Standard SGD",
        "gram_only": "Gram Reg Only",
        "geometry": "Full Geometry"
    }

    for method, data in results.items():
        initial = data['initial']
        final = data['final']

        delta_cond = initial['cond'] - final['cond']
        delta_val_ppl = initial.get('val_ppl', initial.get('ppl', 0)) - final.get('val_ppl', final.get('ppl', 0))
        pct_cond = 100 * delta_cond / initial['cond']
        pct_val_ppl = 100 * delta_val_ppl / initial.get('val_ppl', initial.get('ppl', 1))

        initial_val_ppl = initial.get('val_ppl', initial.get('ppl', 0))
        final_val_ppl = final.get('val_ppl', final.get('ppl', 0))
        initial_train_ppl = initial.get('train_ppl', 0)
        final_train_ppl = final.get('train_ppl', 0)

        table.add_row(
            method_names.get(method, method),
            f"{initial['cond']:.0f}",
            f"{final['cond']:.0f}",
            f"{delta_cond:.0f}",
            f"{initial_val_ppl:.2f}",
            f"{final_val_ppl:.2f}",
            f"{initial_train_ppl:.2f}" if initial_train_ppl > 0 else "—",
            f"{final_train_ppl:.2f}" if final_train_ppl > 0 else "—",
            f"{pct_cond:.1f}% cond, {pct_val_ppl:.1f}% val ppl"
        )

    return table


def plot_trajectories(results, output_path):
    """Plot conditioning and perplexity trajectories."""
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

    # Plot conditioning
    for method, data in results.items():
        traj = data['trajectory']
        ax1.plot(
            traj['steps'],
            traj['conditioning'],
            label=method_names.get(method, method),
            color=colors.get(method, 'gray'),
            linewidth=2,
            marker='o',
            markersize=4
        )

    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('FF Gram Conditioning', fontsize=12)
    ax1.set_title('Conditioning Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot perplexity
    for method, data in results.items():
        traj = data['trajectory']
        ax2.plot(
            traj['steps'],
            traj['val_perplexity'],
            label=method_names.get(method, method),
            color=colors.get(method, 'gray'),
            linewidth=2,
            marker='o',
            markersize=4
        )

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Validation Perplexity', fontsize=12)
    ax2.set_title('Perplexity Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]Plot saved to: {output_path}[/green]")


def create_summary(results):
    """Create summary dict with key findings."""
    summary = {
        "methods": {},
        "comparison": {}
    }

    # Individual method results
    for method, data in results.items():
        initial = data['initial']
        final = data['final']

        summary["methods"][method] = {
            "conditioning": {
                "initial": initial['cond'],
                "final": final['cond'],
                "improvement": initial['cond'] - final['cond'],
                "pct_improvement": 100 * (initial['cond'] - final['cond']) / initial['cond']
            },
            "perplexity": {
                "initial": initial['val_ppl'],
                "final": final['val_ppl'],
                "improvement": initial['val_ppl'] - final['val_ppl'],
                "pct_improvement": 100 * (initial['val_ppl'] - final['val_ppl']) / initial['val_ppl']
            }
        }

    # Relative comparisons
    if "geometry" in results and "standard" in results:
        geom_final_cond = results["geometry"]["final"]["cond"]
        std_final_cond = results["standard"]["final"]["cond"]

        geom_final_ppl = results["geometry"]["final"]["val_ppl"]
        std_final_ppl = results["standard"]["final"]["val_ppl"]

        summary["comparison"]["geometry_vs_standard"] = {
            "conditioning_advantage": std_final_cond - geom_final_cond,
            "conditioning_ratio": std_final_cond / geom_final_cond,
            "perplexity_advantage": std_final_ppl - geom_final_ppl,
            "perplexity_ratio": std_final_ppl / geom_final_ppl
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare retrofit results")
    parser.add_argument("--results_dir", type=str, default="artifacts_ts/retrofit_comparison",
                       help="Directory with result JSON files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output summary JSON")

    args = parser.parse_args()

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Geometry Retrofit Results Comparison[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Load results
    results = load_results(args.results_dir)

    if not results:
        console.print("[red]No results found![/red]")
        return

    console.print(f"[cyan]Loaded {len(results)} experiment results[/cyan]\n")

    # Create comparison table
    table = create_comparison_table(results)
    console.print(table)

    # Plot trajectories
    plot_path = Path(args.results_dir) / "trajectories.png"
    plot_trajectories(results, plot_path)

    # Create summary
    summary = create_summary(results)

    # Print key findings
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print(f"[bold green]Key Findings[/bold green]")
    console.print(f"[bold green]{'='*70}[/bold green]\n")

    if "geometry" in summary["methods"]:
        geom = summary["methods"]["geometry"]
        console.print(f"[cyan]Full Geometry (GLT + Gram):[/cyan]")
        console.print(f"  Conditioning: {geom['conditioning']['improvement']:.0f} improvement ({geom['conditioning']['pct_improvement']:.1f}%)")
        console.print(f"  Perplexity:   {geom['perplexity']['improvement']:.2f} improvement ({geom['perplexity']['pct_improvement']:.1f}%)")

    if "gram_only" in summary["methods"]:
        gram = summary["methods"]["gram_only"]
        console.print(f"\n[cyan]Gram Only:[/cyan]")
        console.print(f"  Conditioning: {gram['conditioning']['improvement']:.0f} improvement ({gram['conditioning']['pct_improvement']:.1f}%)")
        console.print(f"  Perplexity:   {gram['perplexity']['improvement']:.2f} improvement ({gram['perplexity']['pct_improvement']:.1f}%)")

    if "standard" in summary["methods"]:
        std = summary["methods"]["standard"]
        console.print(f"\n[cyan]Standard Training:[/cyan]")
        console.print(f"  Conditioning: {std['conditioning']['improvement']:.0f} improvement ({std['conditioning']['pct_improvement']:.1f}%)")
        console.print(f"  Perplexity:   {std['perplexity']['improvement']:.2f} improvement ({std['perplexity']['pct_improvement']:.1f}%)")

    if "geometry_vs_standard" in summary["comparison"]:
        comp = summary["comparison"]["geometry_vs_standard"]
        console.print(f"\n[bold yellow]Geometry vs Standard:[/bold yellow]")
        console.print(f"  {comp['conditioning_ratio']:.1f}x better final conditioning")
        console.print(f"  {comp['perplexity_advantage']:.2f} better final perplexity")

    # Save summary
    output_path = args.output or Path(args.results_dir) / "summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]Summary saved to: {output_path}[/green]\n")

    # Save samples if available
    samples_output = Path(args.results_dir) / "samples.txt"
    with open(samples_output, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Generated Samples from Retrofit Experiments\n")
        f.write("="*70 + "\n\n")

        for method, data in results.items():
            if 'samples' in data and data['samples']:
                method_names = {
                    "standard": "Standard SGD",
                    "gram_only": "Gram Regularization Only",
                    "geometry": "Full Geometry (GLT + Gram)"
                }

                f.write(f"\n{'='*70}\n")
                f.write(f"{method_names.get(method, method)}\n")
                f.write(f"Final Conditioning: {data['final']['cond']:.0f}\n")
                f.write(f"Final Val PPL: {data['final']['val_ppl']:.2f}\n")
                f.write(f"Final Train PPL: {data['final']['train_ppl']:.2f}\n")
                f.write(f"{'='*70}\n\n")

                for i, sample in enumerate(data['samples'], 1):
                    f.write(f"Sample {i}:\n")
                    f.write(f"Prompt: {sample['prompt']}\n")
                    f.write(f"{sample['text']}\n\n")
                    f.write("-"*70 + "\n\n")

    console.print(f"[green]Samples saved to: {samples_output}[/green]\n")


if __name__ == "__main__":
    main()
