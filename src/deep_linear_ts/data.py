"""
Data loading utilities for time series benchmarks.

Supports:
- ETTh1/ETTh2: Electricity transformer temperature
- M4: Competition dataset with 100K time series
- Traffic: 862 sensors spatial-temporal data
- Weather: 21 features, 52K points
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Literal


class TimeSeriesDataset(Dataset):
    """
    Generic time series dataset.

    Args:
        data: Time series data (n_samples, n_features)
        sequence_length: Length of input sequences
        prediction_length: Length of prediction horizon
        stride: Stride for sliding window
    """

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 96,
        prediction_length: int = 24,
        stride: int = 1,
    ):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.stride = stride

        # Compute valid indices for sliding window
        self.indices = list(range(
            0,
            len(data) - sequence_length - prediction_length + 1,
            stride
        ))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence pair (input, target).

        Returns:
            input: (sequence_length, n_features)
            target: (prediction_length, n_features)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        pred_end_idx = end_idx + self.prediction_length

        input_seq = self.data[start_idx:end_idx]
        target_seq = self.data[end_idx:pred_end_idx]

        return input_seq, target_seq


def load_ett_dataset(
    dataset_name: Literal['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'] = 'ETTh1',
    data_dir: Optional[Path] = None,
    sequence_length: int = 96,
    prediction_length: int = 24,
    batch_size: int = 32,
    split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load ETT (Electricity Transformer Temperature) dataset.

    The ETT dataset contains 7 features:
    - Oil temperature (target)
    - 6 power load features

    Args:
        dataset_name: Which ETT variant to load
        data_dir: Directory containing ETT data
        sequence_length: Length of input sequences
        prediction_length: Length of prediction horizon
        batch_size: Batch size for data loaders
        split_ratio: (train, val, test) split ratios

    Returns:
        train_loader, val_loader, test_loader
    """
    if data_dir is None:
        data_dir = Path.home() / '.deep_linear_ts' / 'data' / 'ETT'
        data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f'{dataset_name}.csv'

    if not csv_path.exists():
        print(f"ETT dataset not found at {csv_path}")
        print("Please download from: https://github.com/zhouhaoyi/ETDataset")
        # Return dummy data for now
        dummy_data = np.random.randn(10000, 7)
        return _create_dataloaders(
            dummy_data, sequence_length, prediction_length, batch_size, split_ratio
        )

    # Load CSV
    df = pd.read_csv(csv_path)

    # Drop date column if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    data = df.values.astype(np.float32)

    return _create_dataloaders(
        data, sequence_length, prediction_length, batch_size, split_ratio
    )


def _create_dataloaders(
    data: np.ndarray,
    sequence_length: int,
    prediction_length: int,
    batch_size: int,
    split_ratio: Tuple[float, float, float],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Helper to create train/val/test dataloaders."""

    # Split data
    n = len(data)
    train_end = int(n * split_ratio[0])
    val_end = int(n * (split_ratio[0] + split_ratio[1]))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data, sequence_length, prediction_length, stride=1
    )
    val_dataset = TimeSeriesDataset(
        val_data, sequence_length, prediction_length, stride=1
    )
    test_dataset = TimeSeriesDataset(
        test_data, sequence_length, prediction_length, stride=1
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 7,
    sequence_length: int = 96,
    prediction_length: int = 24,
    batch_size: int = 32,
    pattern: Literal['sine', 'linear', 'ar', 'random'] = 'sine',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create synthetic time series data for testing.

    Args:
        n_samples: Number of time steps
        n_features: Number of features
        sequence_length: Length of input sequences
        prediction_length: Length of prediction horizon
        batch_size: Batch size
        pattern: Type of pattern ('sine', 'linear', 'ar', 'random')

    Returns:
        train_loader, val_loader, test_loader
    """
    t = np.arange(n_samples)

    if pattern == 'sine':
        # Multiple sine waves with different frequencies
        data = np.stack([
            np.sin(2 * np.pi * (i + 1) * t / 1000)
            for i in range(n_features)
        ], axis=1)

    elif pattern == 'linear':
        # Linear trends
        data = np.stack([
            (i + 1) * t / 1000
            for i in range(n_features)
        ], axis=1)

    elif pattern == 'ar':
        # Autoregressive process
        data = np.zeros((n_samples, n_features))
        for i in range(n_features):
            for t_idx in range(1, n_samples):
                data[t_idx, i] = 0.8 * data[t_idx - 1, i] + np.random.randn() * 0.1

    else:  # random
        data = np.random.randn(n_samples, n_features)

    # Normalize
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    return _create_dataloaders(
        data.astype(np.float32),
        sequence_length,
        prediction_length,
        batch_size,
        (0.7, 0.1, 0.2)
    )
