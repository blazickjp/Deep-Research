"""
Evaluation utilities for deep linear time series models.

Includes:
- Comprehensive metrics (MSE, MAE, MAPE, etc.)
- Baseline comparisons
- Inference time benchmarking
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import time

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, environment variables must be set manually


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of model performance.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        # Check if CPU is forced via environment variable
        force_cpu = os.getenv('FORCE_CPU', '0') == '1'

        if force_cpu:
            device = torch.device('cpu')
        # Prioritize: CUDA > MPS > CPU
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    model = model.to(device)
    model.eval()

    all_outputs = []
    all_targets = []
    inference_times = []

    pbar = tqdm(test_loader, desc='Evaluating')
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = model(inputs)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    mse = torch.mean((all_outputs - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_outputs - all_targets)).item()
    rmse = np.sqrt(mse)

    # Symmetric Mean Absolute Percentage Error (sMAPE)
    # More robust for data near zero than standard MAPE
    # sMAPE = 200 * |y_true - y_pred| / (|y_true| + |y_pred| + epsilon)
    epsilon = 1e-8
    numerator = torch.abs(all_targets - all_outputs)
    denominator = torch.abs(all_targets) + torch.abs(all_outputs) + epsilon
    mape = torch.mean(numerator / denominator).item() * 200  # sMAPE ranges 0-200%

    # R² score
    ss_res = torch.sum((all_targets - all_outputs) ** 2)
    ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot).item()

    # Inference metrics
    avg_inference_time = np.mean(inference_times)
    throughput = len(test_loader.dataset) / sum(inference_times)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'avg_inference_time': avg_inference_time,
        'throughput': throughput,
    }


def benchmark_depth_scaling(
    model_class,
    depths: list,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    **model_kwargs
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark model performance across different depths.

    Args:
        model_class: Model class to instantiate
        depths: List of depths to test
        test_loader: Test data loader
        device: Device to evaluate on
        **model_kwargs: Additional arguments for model initialization

    Returns:
        Dictionary mapping depth to metrics
    """
    results = {}

    for depth in depths:
        print(f"\nEvaluating depth={depth}")

        # Create model
        model = model_class(depth=depth, **model_kwargs)

        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        metrics['n_parameters'] = sum(p.numel() for p in model.parameters())

        results[depth] = metrics

        print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")
        print(f"Parameters: {metrics['n_parameters']:,}")
        print(f"Inference time: {metrics['avg_inference_time']*1000:.2f}ms")

    return results


def compare_with_baselines(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare model with simple baselines.

    Args:
        model: Trained model to compare
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary of results for each method
    """
    results = {}

    # Evaluate main model
    print("Evaluating trained model...")
    results['model'] = evaluate_model(model, test_loader, device)

    # Baseline 1: Last value persistence
    print("Evaluating persistence baseline...")
    results['persistence'] = evaluate_persistence_baseline(test_loader, device)

    # Baseline 2: Linear regression
    print("Evaluating linear regression baseline...")
    results['linear_regression'] = evaluate_linear_regression_baseline(
        test_loader, device
    )

    return results


@torch.no_grad()
def evaluate_persistence_baseline(
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate simple persistence baseline (predict last observed value).

    Args:
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary of metrics
    """
    if device is None:
        # Prioritize: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    all_outputs = []
    all_targets = []

    for inputs, targets in test_loader:
        # Persistence: repeat last input value
        last_values = inputs[:, -1:, :].repeat(1, targets.size(1), 1)

        all_outputs.append(last_values)
        all_targets.append(targets)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    mse = torch.mean((all_outputs - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_outputs - all_targets)).item()
    rmse = np.sqrt(mse)

    return {'mse': mse, 'mae': mae, 'rmse': rmse}


@torch.no_grad()
def evaluate_linear_regression_baseline(
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate simple linear regression baseline.

    Args:
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary of metrics
    """
    if device is None:
        # Prioritize: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # Simple linear model: just one linear layer
    sample_input, sample_target = next(iter(test_loader))
    input_size = sample_input.size(1) * sample_input.size(2)
    output_size = sample_target.size(1) * sample_target.size(2)

    model = nn.Linear(input_size, output_size).to(device)

    all_outputs = []
    all_targets = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Flatten inputs
        batch_size = inputs.size(0)
        inputs_flat = inputs.reshape(batch_size, -1)

        # Forward pass
        outputs_flat = model(inputs_flat)
        outputs = outputs_flat.reshape(targets.shape)

        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    mse = torch.mean((all_outputs - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_outputs - all_targets)).item()
    rmse = np.sqrt(mse)

    return {'mse': mse, 'mae': mae, 'rmse': rmse}


def main():
    """Main evaluation script entry point."""
    from .models import DeepLinearTimeSeries
    from .data import create_synthetic_dataset

    print("Creating synthetic dataset...")
    _, _, test_loader = create_synthetic_dataset(
        n_samples=10000,
        n_features=7,
        sequence_length=96,
        prediction_length=24,
        batch_size=32,
        pattern='sine'
    )

    print("\nInitializing model...")
    model = DeepLinearTimeSeries(
        input_dim=7,
        hidden_dim=64,
        output_dim=7,
        depth=100,
        sequence_length=96,
    )

    print("\nEvaluating model...")
    results = compare_with_baselines(model, test_loader)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()
