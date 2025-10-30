"""
Utility functions for deep linear networks research.

Includes:
- Geometry analysis (singular values, rank, balancedness)
- Gradient flow monitoring
- Visualization tools
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class DynamicsAnalyzer:
    """
    Track geometric quantities during training.

    Based on Menon et al. (2024) theory of deep linear networks.
    Tracks:
    - Singular values and effective rank
    - Balancedness of layers
    - Gradient flow statistics
    """

    def __init__(self):
        self.history = {
            'singular_values': [],
            'effective_ranks': [],
            'balancedness': [],
            'gradient_norms': [],
        }

    def track_singular_values(self, model: nn.Module) -> List[np.ndarray]:
        """
        Compute singular values for all linear layers.

        Args:
            model: Model to analyze

        Returns:
            List of singular value arrays
        """
        singular_values = []

        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                _, s, _ = np.linalg.svd(weight)
                singular_values.append(s)

        self.history['singular_values'].append(singular_values)
        return singular_values

    def compute_effective_rank(self, W: np.ndarray, threshold: float = 0.01) -> float:
        """
        Compute effective rank of a matrix.

        Effective rank = exp(H(σ)) where H is entropy of singular values.

        Args:
            W: Weight matrix
            threshold: Threshold for near-zero singular values

        Returns:
            Effective rank
        """
        _, s, _ = np.linalg.svd(W)

        # Normalize singular values
        s = s / (s.sum() + 1e-10)

        # Filter near-zero values
        s = s[s > threshold]

        # Compute entropy
        entropy = -np.sum(s * np.log(s + 1e-10))

        # Effective rank
        return np.exp(entropy)

    def measure_balancedness(self, model: nn.Module) -> float:
        """
        Measure how balanced the network is.

        A network is balanced if W_i^T W_i ≈ W_j W_j^T for all layers.

        Args:
            model: Model to analyze

        Returns:
            Balancedness score (lower is more balanced)
        """
        gram_norms = []

        for module in model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.detach()

                # Left and right Gram matrices
                left_gram = W @ W.T
                right_gram = W.T @ W

                # Frobenius norms
                left_norm = torch.norm(left_gram, 'fro').item()
                right_norm = torch.norm(right_gram, 'fro').item()

                gram_norms.append((left_norm, right_norm))

        if not gram_norms:
            return 0.0

        # Measure variance in gram norms
        left_norms = np.array([x[0] for x in gram_norms])
        right_norms = np.array([x[1] for x in gram_norms])

        # Perfect balance: all left norms equal, all right norms equal
        balance_score = np.std(left_norms) + np.std(right_norms)

        self.history['balancedness'].append(balance_score)
        return balance_score

    def analyze_gradient_flow(self, model: nn.Module) -> Dict[str, float]:
        """
        Analyze gradient flow through the network.

        Args:
            model: Model to analyze

        Returns:
            Dictionary of gradient statistics
        """
        grad_norms = []

        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                grad_norm = torch.norm(module.weight.grad).item()
                grad_norms.append(grad_norm)

        if not grad_norms:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}

        stats = {
            'mean': np.mean(grad_norms),
            'std': np.std(grad_norms),
            'max': np.max(grad_norms),
            'min': np.min(grad_norms),
        }

        self.history['gradient_norms'].append(grad_norms)
        return stats

    def plot_singular_values(
        self,
        save_path: Optional[Path] = None,
        layer_idx: int = 0
    ):
        """
        Plot evolution of singular values over training.

        Args:
            save_path: Path to save figure
            layer_idx: Which layer to plot
        """
        if not self.history['singular_values']:
            print("No singular values tracked yet")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for epoch_idx, sv_list in enumerate(self.history['singular_values']):
            if layer_idx < len(sv_list):
                sv = sv_list[layer_idx]
                ax.plot(sv, alpha=0.5, label=f'Epoch {epoch_idx}')

        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value')
        ax.set_title(f'Singular Value Evolution (Layer {layer_idx})')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_balancedness(self, save_path: Optional[Path] = None):
        """
        Plot balancedness over training.

        Args:
            save_path: Path to save figure
        """
        if not self.history['balancedness']:
            print("No balancedness tracked yet")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.history['balancedness'], marker='o')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Balancedness Score')
        ax.set_title('Network Balancedness Over Training')
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }


def compute_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory usage of model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with memory estimates (in MB)
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
        'total_mb': (param_size + buffer_size) / 1024 / 1024,
    }


def get_model_summary(model: nn.Module) -> str:
    """
    Get a comprehensive summary of the model.

    Args:
        model: Model to summarize

    Returns:
        Summary string
    """
    params = count_parameters(model)
    memory = compute_memory_usage(model)

    summary = []
    summary.append("="*60)
    summary.append("MODEL SUMMARY")
    summary.append("="*60)
    summary.append(f"Total Parameters: {params['total']:,}")
    summary.append(f"Trainable Parameters: {params['trainable']:,}")
    summary.append(f"Non-trainable Parameters: {params['non_trainable']:,}")
    summary.append(f"Memory Usage: {memory['total_mb']:.2f} MB")
    summary.append("="*60)

    return "\n".join(summary)


def visualize_predictions(
    model: nn.Module,
    test_loader,
    n_samples: int = 3,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
):
    """
    Visualize model predictions on test data.

    Args:
        model: Trained model
        test_loader: Test data loader
        n_samples: Number of samples to visualize
        device: Device to run on
        save_path: Path to save figure
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # Get a batch
    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)[:n_samples]
    targets = targets.to(device)[:n_samples]

    with torch.no_grad():
        predictions = model(inputs)

    # Convert to numpy
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Plot
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Time axis
        input_len = inputs.shape[1]
        pred_len = predictions.shape[1]

        input_time = np.arange(input_len)
        pred_time = np.arange(input_len, input_len + pred_len)

        # Plot first feature only
        ax.plot(input_time, inputs[i, :, 0], 'b-', label='Input', linewidth=2)
        ax.plot(pred_time, targets[i, :, 0], 'g-', label='Target', linewidth=2)
        ax.plot(pred_time, predictions[i, :, 0], 'r--', label='Prediction', linewidth=2)

        ax.axvline(input_len, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
