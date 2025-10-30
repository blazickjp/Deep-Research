"""
Training utilities for deep linear time series models.

Includes:
- Training loop with geometry tracking
- Custom optimizer considerations
- Gradient flow monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm
import numpy as np


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping if specified
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': total_loss / n_batches})

    return {
        'loss': total_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    pbar = tqdm(val_loader, desc='Validation')
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute metrics
        loss = criterion(outputs, targets)
        mae = torch.abs(outputs - targets).mean()

        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

        pbar.set_postfix({
            'loss': total_loss / n_batches,
            'mae': total_mae / n_batches
        })

    return {
        'loss': total_loss / n_batches,
        'mae': total_mae / n_batches,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    early_stopping_patience: int = 10,
) -> Dict[str, List[float]]:
    """
    Complete training loop with validation and early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        checkpoint_path: Path to save best model
        early_stopping_patience: Patience for early stopping

    Returns:
        Dictionary containing training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Loss function (MSE for time series)
    criterion = nn.MSELoss()

    # Optimizer (Adam works well for deep linear networks)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])

        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])

        # Print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}")

        # Early stopping and checkpointing
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return history


def main():
    """Main training script entry point."""
    from .models import DeepLinearTimeSeries
    from .data import create_synthetic_dataset

    print("Creating synthetic dataset...")
    train_loader, val_loader, test_loader = create_synthetic_dataset(
        n_samples=10000,
        n_features=7,
        sequence_length=96,
        prediction_length=24,
        batch_size=32,
        pattern='sine'
    )

    print("Initializing model...")
    model = DeepLinearTimeSeries(
        input_dim=7,
        hidden_dim=64,
        output_dim=7,
        depth=100,
        sequence_length=96,
        temporal_mixing='toeplitz',
        use_residual=True,
        residual_weight=0.1,
    )

    print("Starting training...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=50,
        learning_rate=1e-3,
        checkpoint_path='checkpoints/best_model.pt',
        early_stopping_patience=10,
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")


if __name__ == '__main__':
    main()
