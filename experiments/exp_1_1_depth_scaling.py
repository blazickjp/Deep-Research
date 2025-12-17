"""
Experiment 1.1: Depth Scaling Study

Tests deep linear networks at various depths (10-1000 layers) to investigate:
1. Training stability as depth increases
2. Performance vs depth trade-offs
3. Gradient flow and balancedness maintenance
4. Comparison with standard baselines (Transformer, LSTM, GRU)

Results saved to experiments/results/depth_scaling/
"""

from contextlib import nullcontext
import math
from baselines import TransformerBaseline, LSTMBaseline, GRUBaseline
from deep_linear_ts.utils import count_parameters, compute_memory_usage, DynamicsAnalyzer
from deep_linear_ts.evaluate import evaluate_model, evaluate_persistence_baseline, evaluate_linear_regression_baseline
from deep_linear_ts.train import train_model
from deep_linear_ts.data import create_synthetic_dataset
from deep_linear_ts import DeepLinearTimeSeries
import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Rich logging imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich import box

# Setup rich console
console = Console()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------- GEOMETRY HELPERS (DLN) ----------


def _psd_power(A: torch.Tensor, alpha: float, eps: float = 1e-10) -> torch.Tensor:
    # A: symmetric PSD [m x m]
    # MPS doesn't support eigh - move to CPU if needed
    device = A.device
    if device.type == 'mps':
        A = A.cpu()

    evals, U = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    D = torch.diag(evals**alpha)
    result = U @ D @ U.T

    # Move back to original device
    if device.type == 'mps':
        result = result.to(device)

    return result


def A_NW_apply(W: torch.Tensor, Z: torch.Tensor, N: int) -> torch.Tensor:
    """Apply 𝒜_{N,W}(Z) = Σ_{k=1..N} (WWᵀ)^{(N-k)/N} Z (WᵀW)^{(k-1)/N}."""
    WWt = W @ W.T
    WtW = W.T @ W
    out = torch.zeros_like(W)
    for k in range(1, N + 1):
        L = _psd_power(WWt, (N - k) / N)
        R = _psd_power(WtW, (k - 1) / N)
        out = out + L @ Z @ R
    return out


@torch.no_grad()
def end_to_end_matrix_from_model(model, input_dim, sequence_length, output_dim, prediction_length, device):
    """
    Build the full linear operator W: R^{D_in} -> R^{D_out} by probing basis vectors.
    Cost: D_in forward passes; OK for moderate D_in (e.g., 7*96=672).
    """
    model.eval()
    D_in = input_dim * sequence_length
    D_out = output_dim * prediction_length
    Wcols = []
    I = torch.eye(D_in, device=device)
    for j in range(D_in):
        xj = I[:, j].reshape(1, sequence_length, input_dim)  # [1, T, C_in]
        yj = model(xj).reshape(-1)                            # [D_out]
        Wcols.append(yj)
    W = torch.stack(Wcols, dim=1)  # [D_out, D_in]
    return W


def batch_to_mats(xb: torch.Tensor, yb: torch.Tensor):
    """
    xb: [B, T, C_in], yb: [B, T_pred, C_out]
    Returns X: [D_in, B], Y: [D_out, B]
    """
    B = xb.size(0)
    X = xb.reshape(B, -1).T.contiguous()
    Y = yb.reshape(B, -1).T.contiguous()
    return X, Y


@torch.no_grad()
def Eprime_from_batch(W: torch.Tensor, xb: torch.Tensor, yb: torch.Tensor):
    """
    E(W) = 1/(2B) Σ ||W x_i - y_i||^2  ->  E'(W) = 1/B (W X - Y) Xᵀ
    """
    X, Y = batch_to_mats(xb, yb)        # [D_in, B], [D_out, B]
    # Ensure X, Y are on same device as W
    X = X.to(W.device)
    Y = Y.to(W.device)
    R = W @ X - Y                       # [D_out, B]
    Eprime = (R @ X.T) / xb.size(0)     # [D_out, D_in]
    Eb = 0.5 * (R**2).sum().item() / xb.size(0)
    return Eprime, Eb


@torch.no_grad()
def gN_grad_norm_sq(Eprime: torch.Tensor, W: torch.Tensor, N: int) -> float:
    """‖grad_{g_N}E‖^2_{g_N} = Tr(E'ᵀ 𝒜_{N,W}(E'))."""
    ANE = A_NW_apply(W, Eprime, N)
    return float(torch.einsum('ij,ij->', Eprime, ANE))


@torch.no_grad()
def balancedness_stats_from_model(model):
    """
    Collect 2D weights in depth order (nn.Linear only, safest for DLN stacks).
    Compute G_p = W_{p+1}ᵀ W_{p+1} - W_p W_pᵀ.
    """
    Ws = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.dim() == 2:
            Ws.append(m.weight)  # [out, in]
    G_norms = []
    for p in range(len(Ws) - 1):
        Wp = Ws[p]
        Wp1 = Ws[p+1]
        # inner boundary dim must match: rows(Wp)=cols(Wp1)=hidden
        try:
            Gp = Wp1.T @ Wp1 - Wp @ Wp.T
            G_norms.append(Gp.norm().item())
        except RuntimeError:
            # shapes don’t match (nonlinear/conv/Toeplitz layer?) – skip
            continue
    if not G_norms:
        return {'max_G_norm': None, 'mean_G_norm': None, 'num_pairs': 0}
    return {
        'max_G_norm': max(G_norms),
        'mean_G_norm': float(np.mean(G_norms)),
        'num_pairs': len(G_norms)
    }


@torch.no_grad()
def svd_spectrum(W: torch.Tensor, top_k: int = 16):
    # MPS doesn't support svdvals - move to CPU if needed
    device = W.device
    if device.type == 'mps':
        W = W.cpu()

    s = torch.linalg.svdvals(W)
    s_sorted, _ = torch.sort(s, descending=True)
    k = min(top_k, s_sorted.numel())
    return s_sorted[:k].tolist()


@torch.no_grad()
def entropy_S_vol_orbit(W: torch.Tensor, N: int, eps: float = 1e-12):
    """
    S(W) up to an additive constant, from Theorem 10:
    vol(O_W) ∝ sqrt( van(Σ^2) / van(Σ^{2/N}) ), so
    S(W) = 0.5 [ log van(Σ^2) - log van(Σ^{2/N}) ] + const.
    """
    # MPS doesn't support svdvals - move to CPU if needed
    device = W.device
    if device.type == 'mps':
        W = W.cpu()

    s = torch.linalg.svdvals(W)
    s = torch.clamp(s, min=eps)
    # build pairwise Vandermonde terms in σ^2 and σ^{2/N}
    log_van_sig2 = 0.0
    log_van_lam2 = 0.0
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            log_van_sig2 += math.log(abs(s[i]**2 - s[j]**2) + eps)
            lam_i = s[i]**(1.0/N)
            lam_j = s[j]**(1.0/N)
            log_van_lam2 += math.log(abs(lam_i**2 - lam_j**2) + eps)
    return 0.5 * (log_van_sig2 - log_van_lam2)


def depth_scaled_lr(base_lr: float, depth: int, schedule: str = "sqrt"):
    schedule = schedule.lower()
    if schedule == "1_over_n":
        return base_lr / max(depth, 1)
    elif schedule == "sqrt":
        return base_lr / math.sqrt(max(depth / 10.0, 1e-8))
    else:
        return base_lr


def tiny_one_step_dissipation_probe(model, batch, input_dim, sequence_length,
                                    output_dim, prediction_length, depth, device,
                                    probe_lr=5e-5):
    """
    Measure ΔE vs -η * Tr(E'^T 𝒜_{N,W}(E')) on a single tiny SGD step.
    """
    model_copy = type(model)(
        **model.__dict__['_modules']) if False else None  # placeholder
    # Simpler: clone parameters manually
    model_copy = deepcopy_model(model).to(device)
    model_copy.eval()  # we drive it deterministically
    xb, yb = batch
    xb = xb.to(device)
    yb = yb.to(device)
    # E before and E'(W) on the batch
    W0 = end_to_end_matrix_from_model(model_copy, input_dim, sequence_length,
                                      output_dim, prediction_length, device)
    Eprime0, Eb0 = Eprime_from_batch(W0, xb, yb)
    gn_norm_sq = gN_grad_norm_sq(Eprime0, W0, depth)
    # Do one tiny step with true loss/backprop
    loss_fn = torch.nn.MSELoss(reduction='mean')
    opt = torch.optim.SGD(model_copy.parameters(), lr=probe_lr)
    opt.zero_grad(set_to_none=True)
    yhat = model_copy(xb)
    loss = loss_fn(yhat, yb)
    loss.backward()
    opt.step()
    # E after on the same batch, recompute with W1
    W1 = end_to_end_matrix_from_model(model_copy, input_dim, sequence_length,
                                      output_dim, prediction_length, device)
    _, Eb1 = Eprime_from_batch(W1, xb, yb)
    deltaE = Eb1 - Eb0
    predicted = -probe_lr * gn_norm_sq
    ratio = float(deltaE / (predicted + 1e-12))
    return {
        'E_before': Eb0, 'E_after': Eb1, 'deltaE': deltaE,
        'predicted_deltaE': predicted, 'deltaE_over_prediction': ratio,
        'gN_grad_norm_sq': gn_norm_sq, 'probe_lr': probe_lr
    }


def deepcopy_model(model: torch.nn.Module) -> torch.nn.Module:
    import copy
    m2 = copy.deepcopy(model)
    for p1, p2 in zip(model.parameters(), m2.parameters()):
        p2.data.copy_(p1.data)
    return m2
# ---------- END GEOMETRY HELPERS ----------


def run_depth_experiment(
    depths=[10, 50, 100, 500, 1000],
    hidden_dim=64,
    n_samples=10000,
    n_epochs=20,
    quick_mode=False,
    device=None,
    # ---------- NEW ----------
    lr_base=1e-3,
    lr_schedule="sqrt",            # "sqrt" (old default) or "1_over_N"
    use_residual=None,             # None->default False for DLN; True for ablation
    geometry_probe_every=1,        # epochs; set 0 to disable
    do_one_step_probe=True,        # run the tiny-step dissipation check
    beta=None                      # if set, also log S(W) and F_beta
):
    """
    Run depth scaling experiment.

    Args:
        depths: List of depths to test
        hidden_dim: Hidden dimension for all models
        n_samples: Number of training samples
        n_epochs: Training epochs per model
        quick_mode: If True, use fewer samples and epochs for testing
        device: Device to use (auto-detected if None)
    """
    # Quick mode adjustments
    if quick_mode:
        n_samples = 2000
        n_epochs = 5

    # Setup
    results_dir = Path(__file__).parent / "results" / "depth_scaling"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"depth_scaling_{timestamp}.json"
    log_file = results_dir / f"depth_scaling_{timestamp}.log"

    # Setup logging with Rich handler for console, plain handler for file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'))

    rich_handler = RichHandler(
        console=console, rich_tracebacks=True, markup=True)
    rich_handler.setFormatter(logging.Formatter('%(message)s'))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, rich_handler]
    )
    logger = logging.getLogger(__name__)

    # Display experiment configuration in rich panel
    console.print("\n")
    config_table = Table.grid(padding=(0, 2))
    config_table.add_column(style="cyan", justify="right")
    config_table.add_column(style="magenta")

    config_table.add_row("Experiment", "Depth Scaling Study (1.1)")
    config_table.add_row(
        "Start Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    config_table.add_row("Depths", str(depths))
    config_table.add_row("Hidden Dimension", str(hidden_dim))
    config_table.add_row("Training Samples", f"{n_samples:,}")
    config_table.add_row("Epochs per Model", str(n_epochs))
    config_table.add_row("Quick Mode", str(quick_mode))
    config_table.add_row("Device", str(device) if device else "auto-detect")
    config_table.add_row("Results File", str(results_file.name))
    config_table.add_row("Log File", str(log_file.name))
    # ---------- log config additions ----------
    config_table.add_row("LR schedule", lr_schedule)
    config_table.add_row("LR base", f"{lr_base:.2e}")
    config_table.add_row("Geom. probe every", str(geometry_probe_every))
    config_table.add_row("One-step probe", str(do_one_step_probe))
    if beta is not None:
        config_table.add_row("Beta (entropy)", str(beta))

    console.print(Panel(
        config_table,
        title="[bold blue]Experiment 1.1: Depth Scaling Study[/bold blue]",
        border_style="blue",
        box=box.ROUNDED
    ))

    # Dataset configuration
    input_dim = 7
    output_dim = 7
    sequence_length = 96
    prediction_length = 24
    batch_size = 32

    # Create dataset
    logger.info("Creating synthetic dataset (sine pattern)...")

    train_loader, val_loader, test_loader = create_synthetic_dataset(
        n_samples=n_samples,
        n_features=input_dim,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        pattern='sine',
    )

    # Display dataset info in panel
    dataset_table = Table.grid(padding=(0, 2))
    dataset_table.add_column(style="cyan", justify="right")
    dataset_table.add_column(style="green")

    dataset_table.add_row("Pattern", "Sine Wave")
    dataset_table.add_row("Total Samples", f"{n_samples:,}")
    dataset_table.add_row("Input Features", str(input_dim))
    dataset_table.add_row("Output Features", str(output_dim))
    dataset_table.add_row("Sequence Length", str(sequence_length))
    dataset_table.add_row("Prediction Length", str(prediction_length))
    dataset_table.add_row("Batch Size", str(batch_size))
    dataset_table.add_row("Train Batches", str(len(train_loader)))
    dataset_table.add_row("Val Batches", str(len(val_loader)))
    dataset_table.add_row("Test Batches", str(len(test_loader)))

    console.print(Panel(
        dataset_table,
        title="[bold green]Dataset Configuration[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))
    console.print()

    # Storage for all results
    all_results = {
        'config': {
            'depths': depths,
            'hidden_dim': hidden_dim,
            'n_samples': n_samples,
            'n_epochs': n_epochs,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'sequence_length': sequence_length,
            'prediction_length': prediction_length,
            'batch_size': batch_size,
            'quick_mode': quick_mode,
            'timestamp': timestamp,
            'lr_schedule': lr_schedule,
            'lr_base': lr_base,
            'geometry_probe_every': geometry_probe_every,
            'do_one_step_probe': do_one_step_probe,
            'beta': beta
        },
        'deep_linear': {},
        'baselines': {},
    }

    # Test deep linear networks at each depth
    console.print(
        "\n[bold yellow]Starting Deep Linear Network Experiments[/bold yellow]\n")

    # Progress tracker for depths
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as depth_progress:
        depth_task = depth_progress.add_task(
            "[cyan]Training depths", total=len(depths))

        for depth in depths:
            try:
                depth_progress.update(
                    depth_task, description=f"[cyan]Training depth {depth}")
                logger.info(
                    f"[bold yellow]Starting experiment for DEPTH = {depth}[/bold yellow]")

                # Clear MPS cache before creating new model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    logger.debug("Cleared MPS cache")

                if use_residual is None:
                    # DLN default per paper (pure product)
                    residual_flag = False
                else:
                    residual_flag = bool(use_residual)

                # Create model
                logger.info(
                    f"Creating DeepLinearTimeSeries model with depth={depth}, hidden_dim={hidden_dim}")

                model = DeepLinearTimeSeries(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    depth=depth,                               # Use loop variable
                    sequence_length=sequence_length,
                    prediction_length=prediction_length,       # Use dataset config
                    temporal_mixing='identity',                # no interleaving; pure product
                    use_residual=residual_flag,                # Use computed flag
                    mode='strict_dln',
                )
                # optional: ensure exact balanced factorization of current core product
                model.project_core_to_balanced()

                geometry_log = {}
                if do_one_step_probe:
                    try:
                        first_batch = next(iter(train_loader))
                        probe = tiny_one_step_dissipation_probe(
                            model, first_batch, input_dim, sequence_length,
                            output_dim, prediction_length, depth,
                            device if device else torch.device('cpu'),
                            probe_lr=5e-5
                        )
                        geometry_log['one_step_probe'] = probe
                    except Exception as e:
                        logger.warning(
                            f"[yellow]One-step dissipation probe failed: {e}[/yellow]")

                # ---------- Model stats ----------
                params = count_parameters(model)
                memory = compute_memory_usage(model)
                logger.info(
                    f"Model created - Parameters: {params['total']:,}, Memory: {memory['total_mb']:.2f} MB")

                # ---------- LR scaled by depth ----------
                learning_rate = depth_scaled_lr(lr_base, depth, lr_schedule)
                logger.info(
                    f"Starting training for {n_epochs} epochs (lr={learning_rate:.2e}, early_stopping_patience=5, grad_clip=1.0)")

                # ---------- GEOMETRY: per-epoch probes via train_model callback (if supported) ----------
                # If your train_model doesn't support callbacks, we do post-hoc per-epoch probes below.

                start_time = time.time()

                history = train_model(
                    model,
                    train_loader,
                    val_loader,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    checkpoint_path=str(
                        results_dir / f"depth_{depth}_model.pt"),
                    early_stopping_patience=5,
                    device=device,
                    verbose=False,
                    max_grad_norm=1.0,
                )

                train_time = time.time() - start_time
                logger.info(f"Training completed in {train_time:.1f}s")

                # ---------- GEOMETRY: probes at selected epochs/end ----------
                per_epoch_geom = []
                if geometry_probe_every and geometry_probe_every > 0:
                    # Use validation loader's first batch for stable probes
                    try:
                        vbatch = next(iter(val_loader))
                    except StopIteration:
                        vbatch = next(iter(train_loader))
                    xb, yb = vbatch
                    if device:
                        xb = xb.to(device)
                        yb = yb.to(device)
                    # Probe each epoch index in history (coarse: end of training only, or every k epochs)
                    epoch_indices = list(range(n_epochs))
                    epoch_indices = [e for e in epoch_indices if (
                        e+1) % geometry_probe_every == 0 or (e+1) == n_epochs]
                    # We only have final model; log final geometry. If you want epoch-by-epoch, add a callback in train_model.
                    W_end = model.end_to_end_matrix()
                    Eprime, batch_E = Eprime_from_batch(W_end, xb, yb)
                    gn_sq = gN_grad_norm_sq(Eprime, W_end, depth)
                    bal = model.balancedness_stats()
                    spec_top = svd_spectrum(W_end, top_k=16)
                    s = torch.linalg.svdvals(W_end).cpu().numpy().tolist()
                    geometry_log.update({
                        'final_epoch_gN_grad_norm_sq': gn_sq,
                        'final_epoch_batch_E': batch_E,
                        'final_epoch_balancedness': bal,
                        'final_epoch_top_singular_values': spec_top,
                        'final_epoch_spec_downstairs': s
                    })
                    if beta is not None:
                        S_val = entropy_S_vol_orbit(W_end, depth)
                        F_val = float(batch_E - (1.0/float(beta))*S_val)
                        geometry_log.update(
                            {'S_entropy': float(S_val), 'F_beta': F_val})

                # ---------- Evaluate on test set ----------
                logger.info("Evaluating on test set...")
                metrics = evaluate_model(model, test_loader, device=device)

                # ---------- Store results with geometry ----------
                all_results['deep_linear'][str(depth)] = {
                    'parameters': params['total'],
                    'memory_mb': memory['total_mb'],
                    'train_time_sec': train_time,
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'best_val_loss': min(history['val_loss']),
                    'test_metrics': metrics,
                    'training_history': history,
                    'geometry': geometry_log
                }
                # Display results table for this depth
                result_table = Table(
                    title=f"Depth {depth} Results", box=box.SIMPLE, show_header=True, header_style="bold cyan")
                result_table.add_column("Metric", style="cyan", justify="left")
                result_table.add_column(
                    "Value", style="yellow", justify="right")

                result_table.add_row("Parameters", f"{params['total']:,}")
                result_table.add_row(
                    "Memory (MB)", f"{memory['total_mb']:.2f}")
                result_table.add_row("Train Time (s)", f"{train_time:.1f}")
                result_table.add_row("Final Train Loss",
                                     f"{history['train_loss'][-1]:.6f}")
                result_table.add_row(
                    "Best Val Loss", f"{min(history['val_loss']):.6f}")
                result_table.add_row("Test MSE", f"{metrics['mse']:.6f}")
                result_table.add_row("Test MAE", f"{metrics['mae']:.6f}")
                result_table.add_row("Test RMSE", f"{metrics['rmse']:.6f}")
                result_table.add_row(
                    "Test MAPE (%)", f"{metrics.get('mape', 0)*100:.2f}")
                result_table.add_row("Test R²", f"{metrics.get('r2', 0):.4f}")
                result_table.add_row(
                    "Inference Time (ms)", f"{metrics['avg_inference_time']*1000:.2f}")
                result_table.add_row(
                    "Throughput (samples/s)", f"{metrics.get('throughput', 0):.1f}")

                if 'final_epoch_balancedness' in geometry_log:
                    bal = geometry_log['final_epoch_balancedness']
                    result_table.add_row(
                        "||G||_F (max)", f"{bal['max_G_norm'] if bal['max_G_norm'] is not None else 'n/a'}")
                if 'final_epoch_gN_grad_norm_sq' in geometry_log:
                    result_table.add_row(
                        "‖grad_gN E‖²", f"{geometry_log['final_epoch_gN_grad_norm_sq']:.6e}")
                if 'one_step_probe' in geometry_log:
                    ratio = geometry_log['one_step_probe'].get(
                        'deltaE_over_prediction', None)
                    if ratio is not None:
                        result_table.add_row("ΔE / prediction", f"{ratio:.3f}")

                console.print(result_table)
                console.print()

                # Clean up
                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    logger.debug("Cleaned up model and cleared MPS cache")

                # Update progress
                depth_progress.advance(depth_task)

            except Exception as e:
                logger.error(
                    f"[red]✗[/red] ERROR training depth={depth}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                depth_progress.advance(depth_task)
                # Continue with next depth
                continue

    # Test baselines
    console.print(
        "\n[bold yellow]Starting Baseline Model Experiments[/bold yellow]\n")

    # Persistence baseline
    logger.info(
        "[bold cyan]Testing PERSISTENCE BASELINE[/bold cyan] (last value repeat)")
    persistence_metrics = evaluate_persistence_baseline(test_loader)
    all_results['baselines']['persistence'] = {
        'parameters': 0,
        'test_metrics': persistence_metrics,
    }
    logger.info(
        f"[green]✓[/green] PERSISTENCE BASELINE COMPLETED - Test MSE: {persistence_metrics['mse']:.4f}, Test MAE: {persistence_metrics['mae']:.4f}")

    # Linear regression baseline
    logger.info(
        "[bold cyan]Testing LINEAR REGRESSION BASELINE[/bold cyan] (single linear layer)")
    # Linear layer flattens: (seq_len * input_dim) -> (pred_len * output_dim)
    lr_params = (sequence_length * input_dim) * (prediction_length *
                                                 output_dim) + (prediction_length * output_dim)
    lr_metrics = evaluate_linear_regression_baseline(
        test_loader, device=device)
    all_results['baselines']['linear_regression'] = {
        'parameters': lr_params,
        'test_metrics': lr_metrics,
    }
    logger.info(
        f"[green]✓[/green] LINEAR REGRESSION BASELINE COMPLETED - Parameters: {lr_params}, Test MSE: {lr_metrics['mse']:.4f}, Test MAE: {lr_metrics['mae']:.4f}")

    # Transformer baseline
    logger.info("[bold cyan]Testing TRANSFORMER BASELINE[/bold cyan]")
    transformer = TransformerBaseline(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=hidden_dim,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
    )
    params_tf = count_parameters(transformer)
    logger.info(f"Transformer created - Parameters: {params_tf['total']:,}")

    logger.info(f"Starting Transformer training for {n_epochs} epochs")
    start_time = time.time()
    _ = train_model(transformer, train_loader, val_loader, n_epochs=n_epochs,
                    learning_rate=1e-3, device=device, verbose=False, max_grad_norm=1.0)
    train_time_tf = time.time() - start_time
    logger.info(f"Transformer training completed in {train_time_tf:.1f}s")

    logger.info("Evaluating Transformer on test set...")
    metrics_tf = evaluate_model(transformer, test_loader, device=device)
    all_results['baselines']['transformer'] = {
        'parameters': params_tf['total'],
        'train_time_sec': train_time_tf,
        'test_metrics': metrics_tf,
    }
    logger.info(
        f"[green]✓[/green] TRANSFORMER BASELINE COMPLETED - Train time: {train_time_tf:.1f}s, Test MSE: {metrics_tf['mse']:.4f}, Test MAE: {metrics_tf['mae']:.4f}")

    # LSTM baseline
    logger.info("[bold cyan]Testing LSTM BASELINE[/bold cyan]")
    lstm = LSTMBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
    )
    params_lstm = count_parameters(lstm)
    logger.info(f"LSTM created - Parameters: {params_lstm['total']:,}")

    # MPS doesn't support LSTM gradients - force CPU for LSTM/GRU
    # Detect actual device if not specified
    actual_device = device
    if actual_device is None:
        if torch.cuda.is_available():
            actual_device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            actual_device = torch.device('mps')
        else:
            actual_device = torch.device('cpu')

    # Force CPU for LSTM if MPS is detected
    lstm_device = actual_device
    if actual_device.type == 'mps':
        lstm_device = torch.device('cpu')
        logger.warning(
            "[yellow]⚠ MPS doesn't support LSTM gradients - using CPU for LSTM training[/yellow]")

    logger.info(
        f"Starting LSTM training for {n_epochs} epochs (device: {lstm_device})")
    start_time = time.time()
    _ = train_model(lstm, train_loader, val_loader, n_epochs=n_epochs,
                    learning_rate=1e-3, device=lstm_device, verbose=False, max_grad_norm=1.0)
    train_time_lstm = time.time() - start_time
    logger.info(f"LSTM training completed in {train_time_lstm:.1f}s")

    logger.info("Evaluating LSTM on test set...")
    metrics_lstm = evaluate_model(lstm, test_loader, device=lstm_device)
    all_results['baselines']['lstm'] = {
        'parameters': params_lstm['total'],
        'train_time_sec': train_time_lstm,
        'test_metrics': metrics_lstm,
    }
    logger.info(
        f"[green]✓[/green] LSTM BASELINE COMPLETED - Train time: {train_time_lstm:.1f}s, Test MSE: {metrics_lstm['mse']:.4f}, Test MAE: {metrics_lstm['mae']:.4f}")

    # GRU baseline
    logger.info("[bold cyan]Testing GRU BASELINE[/bold cyan]")
    gru = GRUBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
    )
    params_gru = count_parameters(gru)
    logger.info(f"GRU created - Parameters: {params_gru['total']:,}")

    # MPS doesn't support GRU gradients - use same device as LSTM
    gru_device = lstm_device
    if actual_device.type == 'mps':
        logger.warning(
            "[yellow]⚠ MPS doesn't support GRU gradients - using CPU for GRU training[/yellow]")

    logger.info(
        f"Starting GRU training for {n_epochs} epochs (device: {gru_device})")
    start_time = time.time()
    _ = train_model(gru, train_loader, val_loader, n_epochs=n_epochs,
                    learning_rate=1e-3, device=gru_device, verbose=False, max_grad_norm=1.0)
    train_time_gru = time.time() - start_time
    logger.info(f"GRU training completed in {train_time_gru:.1f}s")

    logger.info("Evaluating GRU on test set...")
    metrics_gru = evaluate_model(gru, test_loader, device=gru_device)
    all_results['baselines']['gru'] = {
        'parameters': params_gru['total'],
        'train_time_sec': train_time_gru,
        'test_metrics': metrics_gru,
    }
    logger.info(
        f"[green]✓[/green] GRU BASELINE COMPLETED - Train time: {train_time_gru:.1f}s, Test MSE: {metrics_gru['mse']:.4f}, Test MAE: {metrics_gru['mae']:.4f}")

    # Save results
    logger.info("Saving results...")

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(
        f"[green]✓[/green] Results saved successfully to {results_file}")

    # Generate summary comparison table
    console.print("\n[bold magenta]Final Results Summary[/bold magenta]\n")

    summary_table = Table(
        title="Model Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    summary_table.add_column("Model", style="cyan", justify="left")
    summary_table.add_column("Parameters", style="yellow", justify="right")
    summary_table.add_column("Test MSE", style="green", justify="right")
    summary_table.add_column("Test MAE", style="green", justify="right")
    summary_table.add_column("Inf. Time (ms)", style="blue", justify="right")

    # Deep linear models
    for depth in depths:
        if str(depth) in all_results['deep_linear']:
            results = all_results['deep_linear'][str(depth)]
            inf_time_ms = results['test_metrics']['avg_inference_time'] * 1000
            summary_table.add_row(
                f"DeepLinear-{depth}",
                f"{results['parameters']:,}",
                f"{results['test_metrics']['mse']:.6f}",
                f"{results['test_metrics']['mae']:.6f}",
                f"{inf_time_ms:.2f}"
            )

    # Separator
    summary_table.add_section()

    # Baselines
    for name in ['persistence', 'linear_regression', 'transformer', 'lstm', 'gru']:
        results = all_results['baselines'][name]
        params = results.get('parameters', 0)
        mse = results['test_metrics']['mse']
        mae = results['test_metrics']['mae']
        inf_time = results['test_metrics'].get('avg_inference_time', 0) * 1000
        summary_table.add_row(
            name.title().replace('_', ' '),
            f"{params:,}",
            f"{mse:.6f}",
            f"{mae:.6f}",
            f"{inf_time:.2f}"
        )

    console.print(summary_table)
    console.print()

    # Final summary panel
    completion_table = Table.grid(padding=(0, 2))
    completion_table.add_column(style="cyan", justify="right")
    completion_table.add_column(style="green")

    completion_table.add_row(
        "Completion Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    completion_table.add_row("Results File", str(results_file))
    completion_table.add_row("Log File", str(log_file))
    completion_table.add_row("Depths Tested", str(len(
        [d for d in depths if str(d) in all_results['deep_linear']])) + f"/{len(depths)}")
    completion_table.add_row(
        "Baselines Tested", str(len(all_results['baselines'])))

    console.print(Panel(
        completion_table,
        title="[bold green]Experiment Complete[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Depth scaling experiment for deep linear networks')
    parser.add_argument('--depths', nargs='+', type=int, default=[10, 50, 100, 500, 1000],
                        help='Depths to test')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--n-epochs', type=int, default=20,
                        help='Training epochs per model')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer samples and epochs for testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu, auto-detected if not specified)')
    parser.add_argument('--lr-base', type=float, default=1e-3,
                        help='Base learning rate before depth scaling')
    parser.add_argument('--lr-schedule', type=str, default='sqrt', choices=['sqrt', '1_over_N'],
                        help='Depth scaling: sqrt (old default) or 1_over_N (paper-friendly)')
    parser.add_argument('--residual', action='store_true',
                        help='Use residual connections (off by default for DLN)')
    parser.add_argument('--geometry-probe-every', type=int, default=1,
                        help='Epoch spacing for geometry probes; 0 disables')
    parser.add_argument('--no-one-step-probe', action='store_true',
                        help='Disable the tiny one-step dissipation probe')
    parser.add_argument('--beta', type=float, default=None,
                        help='If set, log entropy S(W) and free energy F_beta')

    args = parser.parse_args()

    # Run experiment
    results = run_depth_experiment(
        depths=args.depths,
        hidden_dim=args.hidden_dim,
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        quick_mode=args.quick,
        device=args.device,
        lr_base=args.lr_base,
        lr_schedule=args.lr_schedule,
        use_residual=args.residual,
        geometry_probe_every=args.geometry_probe_every,
        do_one_step_probe=(not args.no_one_step_probe),
        beta=args.beta,
    )
