"""
Training utilities for deep linear time series models.

Includes:
- Training loop with geometry tracking
- Custom optimizer considerations
- Gradient flow monitoring
- Geometry regularization (head conditioning, feature Gram)
- GLT (Gradient-Less Transport) integration
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console

# Import geometry regularizers and GLT utilities
from .geometry import RunningSecondMoment, head_logdet_whitened, feature_gram_logdet
from .glt import glt_over_core, glt_encoder_decoder

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, environment variables must be set manually

console = Console()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
    show_progress: bool = True,
    # Geometry regularization parameters
    head_ema: Optional[RunningSecondMoment] = None,
    head_lambda: float = 0.0,
    feat_gram_lambda: float = 0.0,
    # GLT parameters
    glt_freq: Optional[int] = None,
    glt_burnin: int = 0,
    global_step: int = 0,
) -> Dict[str, float]:
    """
    Train for one epoch with optional geometry regularization and GLT.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        show_progress: Whether to show tqdm progress bar
        head_ema: RunningSecondMoment tracker for head conditioning (optional)
        head_lambda: Head conditioning regularization strength
        feat_gram_lambda: Feature Gram regularization strength
        glt_freq: Apply GLT every N steps (None to disable)
        glt_burnin: Don't apply GLT before this many steps
        global_step: Current global training step

    Returns:
        Dictionary of training metrics including gradient norms and regularization losses
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_head_reg = 0.0
    total_feat_reg = 0.0
    n_batches = 0
    n_glt_applied = 0

    # Determine if we need features for regularization
    use_regularization = (head_lambda > 0 or feat_gram_lambda > 0)

    iterator = tqdm(train_loader, desc='Training', disable=not show_progress)
    for _, (inputs, targets) in enumerate(iterator):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()

        if use_regularization:
            outputs, feats = model(inputs, return_feats=True)
            pre_head = feats['pre_head']  # [B, hidden_dim] or [B, D_core]
        else:
            outputs = model(inputs)

        # Compute base loss
        loss = criterion(outputs, targets)

        # Add geometry regularization
        reg_loss = 0.0
        head_reg_val = 0.0
        feat_reg_val = 0.0

        if use_regularization:
            # Head conditioning regularization
            if head_lambda > 0 and head_ema is not None:
                # Update EMA with current batch features
                head_ema.update(pre_head.detach())

                # Get the last linear layer (head)
                if hasattr(model, 'decoder') and len(model.decoder) > 0:
                    head_layer = model.decoder[-1]
                    if isinstance(head_layer, nn.Linear):
                        S = head_ema.get()
                        head_reg = head_logdet_whitened(head_layer.weight, S, head_lambda)
                        reg_loss = reg_loss + head_reg
                        head_reg_val = head_reg.item()
                elif hasattr(model, 'output_proj') and model.output_proj is not None:
                    S = head_ema.get()
                    head_reg = head_logdet_whitened(model.output_proj.weight, S, head_lambda)
                    reg_loss = reg_loss + head_reg
                    head_reg_val = head_reg.item()

            # Feature Gram regularization
            if feat_gram_lambda > 0:
                # Apply to pre-head features
                feat_reg = feature_gram_logdet(pre_head, feat_gram_lambda)
                reg_loss = reg_loss + feat_reg
                feat_reg_val = feat_reg.item()

        # Total loss
        total_loss_with_reg = loss + reg_loss

        # Backward pass
        total_loss_with_reg.backward()

        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float('inf')).item()
        total_grad_norm += grad_norm

        # Gradient clipping if specified
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Apply GLT if enabled and conditions met
        if glt_freq is not None and global_step >= glt_burnin:
            if global_step % glt_freq == 0:
                if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
                    # Time series mode: apply to encoder and decoder
                    glt_encoder_decoder(model.encoder, model.decoder, optimizer)
                    n_glt_applied += 1
                elif hasattr(model, 'core'):
                    # Strict DLN mode: apply to core
                    glt_over_core(model.core, optimizer)
                    n_glt_applied += 1

        # Track metrics
        total_loss += loss.item()
        total_head_reg += head_reg_val
        total_feat_reg += feat_reg_val
        n_batches += 1
        global_step += 1

        # Update progress bar
        if show_progress:
            postfix = {
                'loss': total_loss / n_batches,
                'grad': total_grad_norm / n_batches
            }
            if head_lambda > 0:
                postfix['head_reg'] = total_head_reg / n_batches
            if feat_gram_lambda > 0:
                postfix['feat_reg'] = total_feat_reg / n_batches
            iterator.set_postfix(postfix)

    result = {
        'loss': total_loss / n_batches,
        'grad_norm': total_grad_norm / n_batches,
        'global_step': global_step,
    }

    if head_lambda > 0:
        result['head_reg'] = total_head_reg / n_batches
    if feat_gram_lambda > 0:
        result['feat_reg'] = total_feat_reg / n_batches
    if n_glt_applied > 0:
        result['glt_applied'] = n_glt_applied

    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        show_progress: Whether to show tqdm progress bar

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    iterator = tqdm(val_loader, desc='Validation', disable=not show_progress)
    for inputs, targets in iterator:
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

        if show_progress:
            iterator.set_postfix({
                'loss': total_loss / n_batches,
                'mae': total_mae / n_batches
            })

    if n_batches == 0:
        return {'loss': 0.0, 'mae': 0.0}

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
    verbose: bool = True,
    max_grad_norm: Optional[float] = 1.0,
    # Geometry regularization parameters
    head_lambda: float = 0.0,
    head_ema_momentum: float = 0.99,
    feat_gram_lambda: float = 0.0,
    # GLT parameters
    glt_freq: Optional[int] = None,
    glt_burnin: int = 0,
) -> Dict[str, List[float]]:
    """
    Complete training loop with validation, early stopping, and geometry regularization.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        checkpoint_path: Path to save best model
        early_stopping_patience: Patience for early stopping
        verbose: Whether to show progress bars and print statements
        max_grad_norm: Maximum gradient norm for clipping (default 1.0, None to disable)
        head_lambda: Head conditioning regularization strength (0 to disable)
        head_ema_momentum: EMA momentum for head conditioning (default 0.99)
        feat_gram_lambda: Feature Gram regularization strength (0 to disable)
        glt_freq: Apply GLT every N steps (None to disable)
        glt_burnin: Don't apply GLT before this many steps

    Returns:
        Dictionary containing training history with geometry metrics
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

    # Log device selection if enabled
    if os.getenv('DEEP_LINEAR_LOG_DEVICE', '0') == '1':
        print(f"Training on device: {device}")
        if device.type == 'mps':
            watermark = os.getenv(
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'default (~0.9)')
            print(f"MPS cache ratio: {watermark}")

    model = model.to(device)

    # Loss function (MSE for time series)
    criterion = nn.MSELoss()

    # Optimizer (Adam works well for deep linear networks)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )

    # Initialize RunningSecondMoment for head conditioning if enabled
    head_ema = None
    if head_lambda > 0:
        # Determine feature dimension for head EMA
        if hasattr(model, 'hidden_dim'):
            feat_dim = model.hidden_dim
        elif hasattr(model, 'D_core'):
            feat_dim = model.D_core
        else:
            raise ValueError("Cannot determine feature dimension for head conditioning")

        head_ema = RunningSecondMoment(feat_dim, momentum=head_ema_momentum, device=device)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'grad_norm': [],
        'learning_rate': [],
    }

    # Add geometry metrics to history if enabled
    if head_lambda > 0:
        history['head_reg'] = []
    if feat_gram_lambda > 0:
        history['feat_reg'] = []
    if glt_freq is not None:
        history['glt_applied'] = []

    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    if verbose:
        console.print(f"\n[bold cyan]Training on {device}[/bold cyan]")
        console.print(
            f"[cyan]Model has {sum(p.numel() for p in model.parameters()):,} parameters[/cyan]")

        # Show geometry config if enabled
        geom_config = []
        if head_lambda > 0:
            geom_config.append(f"head_λ={head_lambda:.0e}")
        if feat_gram_lambda > 0:
            geom_config.append(f"feat_λ={feat_gram_lambda:.0e}")
        if glt_freq is not None:
            geom_config.append(f"GLT every {glt_freq} steps (burnin={glt_burnin})")
        if geom_config:
            console.print(f"[cyan]Geometry: {', '.join(geom_config)}[/cyan]\n")
        else:
            console.print()

    # Create rich progress bar for epochs
    if verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[cyan]{task.fields[status]}"),
            console=console,
        ) as progress:
            epoch_task = progress.add_task(
                "[green]Training Progress",
                total=n_epochs,
                status=""
            )

            for epoch in range(n_epochs):
                # Train
                train_metrics = train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    max_grad_norm=max_grad_norm, show_progress=False,  # Disable tqdm, use rich instead
                    head_ema=head_ema, head_lambda=head_lambda, feat_gram_lambda=feat_gram_lambda,
                    glt_freq=glt_freq, glt_burnin=glt_burnin, global_step=global_step
                )

                # Update global step counter
                global_step = train_metrics['global_step']

                # Validate
                val_metrics = validate(
                    model, val_loader, criterion, device, show_progress=False)

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_mae'].append(val_metrics['mae'])
                history['grad_norm'].append(train_metrics['grad_norm'])
                history['learning_rate'].append(current_lr)

                # Track geometry metrics
                if head_lambda > 0:
                    history['head_reg'].append(train_metrics.get('head_reg', 0.0))
                if feat_gram_lambda > 0:
                    history['feat_reg'].append(train_metrics.get('feat_reg', 0.0))
                if glt_freq is not None:
                    history['glt_applied'].append(train_metrics.get('glt_applied', 0))

                # Learning rate scheduling
                scheduler.step(val_metrics['loss'])

                # Build status string with aggregate stats
                status = (
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train: {train_metrics['loss']:.4f} | "
                    f"Val: {val_metrics['loss']:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"Grad: {train_metrics['grad_norm']:.3f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Best: {best_val_loss:.4f} | "
                    f"Pat: {patience_counter}/{early_stopping_patience}"
                )

                # Add geometry metrics to status
                if head_lambda > 0:
                    status += f" | HeadReg: {train_metrics.get('head_reg', 0):.2e}"
                if feat_gram_lambda > 0:
                    status += f" | FeatReg: {train_metrics.get('feat_reg', 0):.2e}"
                if glt_freq is not None and train_metrics.get('glt_applied', 0) > 0:
                    status += f" | GLT: {train_metrics.get('glt_applied', 0)}"

                # Early stopping and checkpointing
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    status += " [bold green]✓ NEW BEST[/bold green]"

                    if checkpoint_path:
                        # Create checkpoint directory if it doesn't exist
                        from pathlib import Path
                        Path(checkpoint_path).parent.mkdir(
                            parents=True, exist_ok=True)

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                        }, checkpoint_path)
                        status += " [dim](saved)[/dim]"
                else:
                    patience_counter += 1

                # Update progress bar
                progress.update(epoch_task, advance=1, status=status)

                if patience_counter >= early_stopping_patience:
                    progress.update(
                        epoch_task,
                        status=f"[yellow]Early stopping at epoch {epoch + 1}[/yellow]"
                    )
                    break

    else:
        # Non-verbose mode: no progress bars
        for epoch in range(n_epochs):
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device,
                max_grad_norm=max_grad_norm, show_progress=False,
                head_ema=head_ema, head_lambda=head_lambda, feat_gram_lambda=feat_gram_lambda,
                glt_freq=glt_freq, glt_burnin=glt_burnin, global_step=global_step
            )

            # Update global step counter
            global_step = train_metrics['global_step']

            # Validate
            val_metrics = validate(
                model, val_loader, criterion, device, show_progress=False)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['grad_norm'].append(train_metrics['grad_norm'])
            history['learning_rate'].append(current_lr)

            # Track geometry metrics
            if head_lambda > 0:
                history['head_reg'].append(train_metrics.get('head_reg', 0.0))
            if feat_gram_lambda > 0:
                history['feat_reg'].append(train_metrics.get('feat_reg', 0.0))
            if glt_freq is not None:
                history['glt_applied'].append(train_metrics.get('glt_applied', 0))

            # Learning rate scheduling
            scheduler.step(val_metrics['loss'])

            # Early stopping and checkpointing
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                if checkpoint_path:
                    from pathlib import Path
                    Path(checkpoint_path).parent.mkdir(
                        parents=True, exist_ok=True)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                    }, checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
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
