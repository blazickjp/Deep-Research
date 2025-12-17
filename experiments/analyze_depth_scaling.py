"""
Analysis and Visualization for Depth Scaling Experiments

Loads experimental results and generates:
1. Performance comparison tables
2. Visualizations (plots, charts)
3. Statistical analysis
4. Key findings summary

Usage:
    python analyze_depth_scaling.py results/depth_scaling/depth_scaling_TIMESTAMP.json
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Use non-interactive backend for plots
mpl.use('Agg')


def load_results(results_file: str) -> Dict[str, Any]:
    """Load experimental results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_deep_linear_metrics(results: Dict[str, Any]) -> tuple:
    """Extract metrics for deep linear networks at different depths."""
    deep_linear = results['deep_linear']

    depths = sorted([int(d) for d in deep_linear.keys()])

    test_mse = [deep_linear[str(d)]['test_metrics']['mse'] for d in depths]
    test_mae = [deep_linear[str(d)]['test_metrics']['mae'] for d in depths]
    train_time = [deep_linear[str(d)]['train_time_sec'] for d in depths]
    parameters = [deep_linear[str(d)]['parameters'] for d in depths]
    memory_mb = [deep_linear[str(d)]['memory_mb'] for d in depths]
    inf_time_ms = [deep_linear[str(d)]['test_metrics']['avg_inference_time'] * 1000 for d in depths]
    best_val_loss = [deep_linear[str(d)]['best_val_loss'] for d in depths]

    return depths, test_mse, test_mae, train_time, parameters, memory_mb, inf_time_ms, best_val_loss


def extract_baseline_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract metrics for baseline models."""
    baselines = results['baselines']

    baseline_metrics = {}
    for name, data in baselines.items():
        baseline_metrics[name] = {
            'mse': data['test_metrics']['mse'],
            'mae': data['test_metrics']['mae'],
            'parameters': data.get('parameters', 0),
            'train_time': data.get('train_time_sec', 0),
            'inf_time_ms': data['test_metrics'].get('avg_inference_time', 0) * 1000,
        }

    return baseline_metrics


def print_summary_table(results: Dict[str, Any]):
    """Print comprehensive summary table."""
    print("=" * 100)
    print("DEPTH SCALING EXPERIMENT RESULTS")
    print("=" * 100)
    print()

    # Configuration
    config = results['config']
    print("Configuration:")
    print(f"  Depths tested: {config['depths']}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Training samples: {config['n_samples']}")
    print(f"  Epochs: {config['n_epochs']}")
    print(f"  Sequence length: {config['sequence_length']}")
    print(f"  Prediction length: {config['prediction_length']}")
    print()

    # Deep linear results
    print("-" * 100)
    print("DEEP LINEAR NETWORKS")
    print("-" * 100)
    print(f"{'Depth':<10} {'Parameters':<15} {'Memory (MB)':<15} {'Train Time (s)':<18} "
          f"{'Test MSE':<12} {'Test MAE':<12} {'Inf. Time (ms)':<15}")
    print("-" * 100)

    depths, test_mse, test_mae, train_time, parameters, memory_mb, inf_time_ms, best_val_loss = \
        extract_deep_linear_metrics(results)

    for i, depth in enumerate(depths):
        print(f"{depth:<10} {parameters[i]:<15,} {memory_mb[i]:<15.2f} {train_time[i]:<18.1f} "
              f"{test_mse[i]:<12.6f} {test_mae[i]:<12.6f} {inf_time_ms[i]:<15.2f}")

    print()

    # Baseline results
    print("-" * 100)
    print("BASELINE MODELS")
    print("-" * 100)
    print(f"{'Model':<20} {'Parameters':<15} {'Train Time (s)':<18} "
          f"{'Test MSE':<12} {'Test MAE':<12} {'Inf. Time (ms)':<15}")
    print("-" * 100)

    baseline_metrics = extract_baseline_metrics(results)
    for name, metrics in baseline_metrics.items():
        print(f"{name.title():<20} {metrics['parameters']:<15,} {metrics['train_time']:<18.1f} "
              f"{metrics['mse']:<12.6f} {metrics['mae']:<12.6f} {metrics['inf_time_ms']:<15.2f}")

    print()

    # Key findings
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print()

    # Best performing depth
    best_depth_idx = np.argmin(test_mse)
    best_depth = depths[best_depth_idx]
    best_mse = test_mse[best_depth_idx]

    print(f"1. Best Performing Depth: {best_depth} layers")
    print(f"   Test MSE: {best_mse:.6f}")
    print(f"   Test MAE: {test_mae[best_depth_idx]:.6f}")
    print()

    # Comparison with baselines
    print("2. Comparison with Baselines:")
    best_baseline_name = min(baseline_metrics.keys(), key=lambda k: baseline_metrics[k]['mse'])
    best_baseline_mse = baseline_metrics[best_baseline_name]['mse']

    improvement = ((best_baseline_mse - best_mse) / best_baseline_mse) * 100
    print(f"   Best baseline: {best_baseline_name.title()} (MSE: {best_baseline_mse:.6f})")
    print(f"   Best deep linear: Depth {best_depth} (MSE: {best_mse:.6f})")
    print(f"   Improvement: {improvement:+.2f}%")
    print()

    # Efficiency analysis
    print("3. Efficiency Analysis:")
    print(f"   Depth {best_depth}:")
    print(f"     Training time: {train_time[best_depth_idx]:.1f}s")
    print(f"     Inference time: {inf_time_ms[best_depth_idx]:.2f}ms")
    print(f"     Parameters: {parameters[best_depth_idx]:,}")
    print()

    # Depth scaling trend
    print("4. Depth Scaling Trend:")
    if len(depths) >= 3:
        shallow_mse = test_mse[0]
        deep_mse = test_mse[-1]
        trend = "improving" if deep_mse < shallow_mse else "degrading"
        print(f"   Depth {depths[0]} → {depths[-1]}: MSE {shallow_mse:.6f} → {deep_mse:.6f} ({trend})")
    print()

    print("=" * 100)


def create_visualizations(results: Dict[str, Any], output_dir: Path):
    """Generate all visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    depths, test_mse, test_mae, train_time, parameters, memory_mb, inf_time_ms, best_val_loss = \
        extract_deep_linear_metrics(results)

    baseline_metrics = extract_baseline_metrics(results)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))

    # Figure 1: Performance vs Depth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MSE vs Depth
    ax1.plot(depths, test_mse, 'o-', color=colors[0], linewidth=2, markersize=8, label='Deep Linear')

    # Add baseline lines
    for name, metrics in baseline_metrics.items():
        if name != 'persistence':  # Skip persistence baseline
            ax1.axhline(metrics['mse'], linestyle='--', alpha=0.6, label=name.title())

    ax1.set_xlabel('Network Depth (layers)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
    ax1.set_title('Performance vs Depth', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # MAE vs Depth
    ax2.plot(depths, test_mae, 'o-', color=colors[1], linewidth=2, markersize=8, label='Deep Linear')

    # Add baseline lines
    for name, metrics in baseline_metrics.items():
        if name != 'persistence':
            ax2.axhline(metrics['mae'], linestyle='--', alpha=0.6, label=name.title())

    ax2.set_xlabel('Network Depth (layers)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test MAE', fontsize=12, fontweight='bold')
    ax2.set_title('MAE vs Depth', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Computational Resources
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Training time vs Depth
    ax1.plot(depths, train_time, 'o-', color=colors[2], linewidth=2, markersize=8)
    ax1.set_xlabel('Network Depth (layers)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Training Time vs Depth', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Parameters vs Depth
    ax2.plot(depths, parameters, 'o-', color=colors[3], linewidth=2, markersize=8)
    ax2.set_xlabel('Network Depth (layers)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Parameters', fontsize=11, fontweight='bold')
    ax2.set_title('Model Size vs Depth', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Memory vs Depth
    ax3.plot(depths, memory_mb, 'o-', color=colors[4], linewidth=2, markersize=8)
    ax3.set_xlabel('Network Depth (layers)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold')
    ax3.set_title('Memory vs Depth', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    # Inference time vs Depth
    ax4.plot(depths, inf_time_ms, 'o-', color=colors[0], linewidth=2, markersize=8)
    ax4.set_xlabel('Network Depth (layers)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
    ax4.set_title('Inference Speed vs Depth', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'resources_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Collect all models for comparison
    model_names = [f'Depth-{d}' for d in depths[:3]]  # First 3 depths only
    model_names.extend([name.title() for name in baseline_metrics.keys() if name != 'persistence'])

    model_mse = test_mse[:3]
    model_mse.extend([baseline_metrics[name]['mse'] for name in baseline_metrics.keys() if name != 'persistence'])

    model_mae = test_mae[:3]
    model_mae.extend([baseline_metrics[name]['mae'] for name in baseline_metrics.keys() if name != 'persistence'])

    # MSE comparison
    bars1 = ax1.bar(range(len(model_names)), model_mse, color=colors[2], alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
    ax1.set_title('MSE Comparison: Deep Linear vs Baselines', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # MAE comparison
    bars2 = ax2.bar(range(len(model_names)), model_mae, color=colors[1], alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test MAE', fontsize=12, fontweight='bold')
    ax2.set_title('MAE Comparison: Deep Linear vs Baselines', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}/")
    print(f"  - performance_vs_depth.png")
    print(f"  - resources_vs_depth.png")
    print(f"  - model_comparison.png")


def analyze_depth_dependency(results: Dict[str, Any]):
    """Analyze how performance depends on depth."""
    print("\n" + "=" * 100)
    print("DEPTH DEPENDENCY ANALYSIS")
    print("=" * 100)
    print()

    depths, test_mse, test_mae, _, _, _, _, _ = extract_deep_linear_metrics(results)

    # Find optimal depth
    optimal_idx = np.argmin(test_mse)
    optimal_depth = depths[optimal_idx]

    print(f"Optimal depth: {optimal_depth} layers (MSE: {test_mse[optimal_idx]:.6f})")
    print()

    # Analyze trends
    print("Trend analysis:")
    for i in range(1, len(depths)):
        depth_prev, depth_curr = depths[i-1], depths[i]
        mse_prev, mse_curr = test_mse[i-1], test_mse[i]
        change_pct = ((mse_curr - mse_prev) / mse_prev) * 100

        trend = "improved" if mse_curr < mse_prev else "degraded"
        print(f"  {depth_prev} → {depth_curr}: MSE {trend} by {abs(change_pct):.2f}%")

    print()
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Analyze depth scaling experiment results')
    parser.add_argument('results_file', type=str, help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations (default: same as results file)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print()

    # Print summary table
    print_summary_table(results)

    # Depth dependency analysis
    analyze_depth_dependency(results)

    # Generate visualizations
    if not args.no_plots:
        if args.output_dir is None:
            output_dir = Path(args.results_file).parent / 'visualizations'
        else:
            output_dir = Path(args.output_dir)

        print(f"\nGenerating visualizations...")
        create_visualizations(results, output_dir)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
