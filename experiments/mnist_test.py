import argparse
import math
import os
import time
import random
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Rich logging imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich import box
import logging

# Setup rich console and logger
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("mnist_experiment")

# ----------------------------
# Model
# ----------------------------


class SmallConv(nn.Module):
    def __init__(self, c1=32, c2=64, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pool = nn.MaxPool2d(2)            # <— NEW
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(c2)
        self.head = nn.Linear(c2, num_classes)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)                        # <— NEW
        x = F.relu(self.bn2(self.conv2(x)))
        return x

    def forward(self, x):
        f = self.forward_features(x)
        g = torch.mean(f, dim=(2, 3))
        logits = self.head(g)
        return logits, f

# ----------------------------
# Aligned GLT: channel permutation with optimizer-state transport
# ----------------------------


@torch.no_grad()
def _permute_param_and_state_(param: torch.Tensor, dim: int, index: torch.Tensor, opt: optim.Optimizer):
    new = param.data.index_select(dim, index)
    param.data.copy_(new)
    if opt is not None and param in opt.state:
        st = opt.state[param]
        for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            if k in st and isinstance(st[k], torch.Tensor) and st[k].shape == param.data.shape:
                st[k].data.copy_(st[k].data.index_select(dim, index))


@torch.no_grad()
def aligned_glt_permute(model: SmallConv, opt: optim.Optimizer, global_step: int = None):
    device = next(model.parameters()).device
    C1 = model.conv1.out_channels
    perm = torch.randperm(C1, device=device)
    invperm = torch.argsort(perm)
    _permute_param_and_state_(model.conv1.weight, 0, perm, opt)
    if model.conv1.bias is not None:
        _permute_param_and_state_(model.conv1.bias, 0, perm, opt)
    if model.bn1.affine:
        _permute_param_and_state_(model.bn1.weight, 0, perm, opt)
        _permute_param_and_state_(model.bn1.bias,   0, perm, opt)
    model.bn1.running_mean.data.copy_(
        model.bn1.running_mean.data.index_select(0, perm))
    model.bn1.running_var.data.copy_(
        model.bn1.running_var.data.index_select(0, perm))
    _permute_param_and_state_(model.conv2.weight, 1, invperm, opt)
    if global_step is not None:
        logger.debug(
            f"[cyan]Applied GLT permutation at step {global_step}[/cyan] (C1={C1} channels)", extra={"markup": True})

# ----------------------------
# Regularizers (MPS-safe slogdet)
# ----------------------------


def _slogdet_safe(mat: torch.Tensor, eps_eye: torch.Tensor):
    dev = mat.device
    if dev.type == 'mps':
        sign, logdet = torch.slogdet(mat.detach().cpu())
        if (sign <= 0).any():
            logdet = torch.logdet(mat.detach().cpu() + eps_eye.cpu())
        return logdet.to(dev)
    else:
        sign, logdet = torch.slogdet(mat)
        if (sign <= 0).any():
            logdet = torch.logdet(mat + eps_eye)
        return logdet


def feature_gram_logdet(features: torch.Tensor, lam: float, eps: float = 1e-4, normalize: bool = True):
    if lam <= 0.0:
        return features.new_zeros(())
    C = features.shape[1]
    Fm = features.permute(1, 0, 2, 3).reshape(C, -1)
    I = torch.eye(C, device=features.device, dtype=features.dtype)
    G = (Fm @ Fm.t()) / float(Fm.shape[1]) + eps * I
    logdet = _slogdet_safe(G, eps*I)
    if normalize:
        tr = torch.trace(G)
        return - lam * (logdet - C * torch.log(tr + 1e-12))
    else:
        return - lam * logdet


def head_jac_logdet(model: SmallConv, lamJ: float, eps: float = 1e-4, normalize: bool = False):
    if lamJ <= 0.0:
        return model.head.weight.new_zeros(())
    V = model.head.weight
    K = V.shape[0]
    A = V @ V.t() + eps * torch.eye(K, device=V.device, dtype=V.dtype)

    # MPS-safe slogdet (reuse your _slogdet_safe if you have it)
    I = eps * torch.eye(K, device=V.device, dtype=V.dtype)
    logdet = _slogdet_safe(
        A, I) if '_slogdet_safe' in globals() else torch.slogdet(A)[1]

    if normalize:
        tr = torch.trace(A)
        return - lamJ * (logdet - K * torch.log(tr + 1e-12))
    else:
        return - lamJ * logdet


def head_logdet_whitened(model: SmallConv, S: torch.Tensor, lamJ: float, eps: float = 1e-4, normalize: bool = True):
    """
    Scale-invariant head equalizer in the feature metric:
        A = V S V^T + eps I_K,  V in R^{K x C2}, S ~ E[g g^T].
    Penalize  -lamJ * [logdet(A) - K*log(tr(A))]  if normalize=True, else -lamJ*logdet(A).
    """
    if lamJ <= 0.0:
        return model.head.weight.new_zeros(())
    V = model.head.weight
    K = V.shape[0]
    S_dev = S.to(V.device, dtype=V.dtype)
    A = V @ (S_dev + eps *
             torch.eye(S_dev.shape[0], device=S_dev.device, dtype=S_dev.dtype)) @ V.t()

    I = eps * torch.eye(K, device=A.device, dtype=A.dtype)
    logdet = _slogdet_safe(
        A, I) if '_slogdet_safe' in globals() else torch.slogdet(A)[1]
    if normalize:
        tr = torch.trace(A)
        return - lamJ * (logdet - K * torch.log(tr + 1e-12))
    else:
        return - lamJ * logdet


# ----------------------------
# Initialization utilities
# ----------------------------


@torch.no_grad()
def init_kaiming(model: SmallConv):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)


@torch.no_grad()
def init_orthogonal(model: SmallConv):
    """Orthogonal initialization with MPS compatibility (uses CPU for QR decomposition)."""
    gain = math.sqrt(2.0)  # ReLU
    device = next(model.parameters()).device

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            W = m.weight.data
            flat = W.view(W.size(0), -1)
            # Move to CPU for orthogonal init (QR not supported on MPS)
            if device.type == 'mps':
                flat_cpu = flat.cpu()
                nn.init.orthogonal_(flat_cpu, gain=gain)
                flat.copy_(flat_cpu.to(device))
            else:
                nn.init.orthogonal_(flat, gain=gain)
            m.weight.data.copy_(flat.view_as(W))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Move to CPU for orthogonal init (QR not supported on MPS)
            if device.type == 'mps':
                weight_cpu = m.weight.data.cpu()
                nn.init.orthogonal_(weight_cpu, gain=1.0)
                m.weight.data.copy_(weight_cpu.to(device))
            else:
                nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@torch.no_grad()
def init_head_orthonormal(model: SmallConv, scale: float = 1.0):
    """Head orthonormal initialization with MPS compatibility (uses CPU for QR decomposition)."""
    # rows orthonormal if K<=C2
    device = model.head.weight.device
    if device.type == 'mps':
        weight_cpu = model.head.weight.data.cpu()
        nn.init.orthogonal_(weight_cpu, gain=scale)
        model.head.weight.data.copy_(weight_cpu.to(device))
    else:
        nn.init.orthogonal_(model.head.weight, gain=scale)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)


@torch.no_grad()
def balance_interface_channels(model: SmallConv, eps: float = 1e-8):
    """
    Channel-wise 'balanced factorization' between conv1 (out-channels) and conv2 (in-channels).
    For each interface channel j: pick s_j so that ||W1[j]||_F and ||W2[:,j,:]||_F become equal after rescaling:
        W1[j] *= s_j,  W2[:, j, :] /= s_j,  where s_j = sqrt( ||W2[:,j,:]|| / (||W1[j]|| + eps) )
    With BN between convs this does not preserve the exact function, but it equalizes scales (DLN-style).
    """
    W1 = model.conv1.weight.data
    W2 = model.conv2.weight.data
    C1 = W1.size(0)
    for j in range(C1):
        n1 = torch.norm(W1[j]).item()
        n2 = torch.norm(W2[:, j]).item()
        s = math.sqrt((n2 + eps)/(n1 + eps))
        W1[j] *= s
        W2[:, j] /= s


@torch.no_grad()
def set_bn_affine(model: SmallConv, gamma: float = 1.0, beta: float = 0.0):
    for bn in [model.bn1, model.bn2]:
        if bn.affine:
            bn.weight.data.fill_(gamma)
            bn.bias.data.fill_(beta)
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)

# ----------------------------
# Metrics (CPU-safe)
# ----------------------------


def accuracy(logits, y): return (
    logits.argmax(dim=1) == y).float().mean().item()


def compute_gram_cond(feats: torch.Tensor) -> float:
    C = feats.shape[1]
    Fm = feats.permute(1, 0, 2, 3).reshape(C, -1).float()
    G = (Fm @ Fm.t())/float(Fm.shape[1]) + 1e-5*torch.eye(C, dtype=Fm.dtype)
    ev = torch.linalg.eigvalsh(G)
    return (ev.max()/ev.min()).item()


def compute_head_cond(model: SmallConv, S: torch.Tensor = None) -> float:
    V = model.head.weight.detach().float().cpu()
    if S is None:
        M = V @ V.t() + 1e-6*torch.eye(V.shape[0])
    else:
        S_cpu = S.detach().float().cpu()
        M = V @ (S_cpu + 1e-6*torch.eye(S_cpu.shape[0])) @ V.t()
    ev = torch.linalg.eigvalsh(M)
    return (ev.max()/ev.min()).item()

# ----------------------------
# Utilities: device, data, schedulers
# ----------------------------

# ---- Running second moment for pooled features g (B x C2) ----


class RunningSecondMoment:
    def __init__(self, dim, momentum=0.99, device="cpu"):
        self.momentum = float(momentum)
        self.S = torch.eye(dim, device=device)
        self._initialized = False

    @torch.no_grad()
    def update(self, g: torch.Tensor):
        # g: [B, C2] on device
        B = g.shape[0]
        Sb = (g.t() @ g) / float(B)  # E[g g^T] on the batch
        if not self._initialized:
            self.S.copy_(Sb)
            self._initialized = True
        else:
            self.S.mul_(self.momentum).add_(Sb, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def get(self):
        return self.S


def pick_device(device_arg: str) -> torch.device:
    if device_arg.lower() != "auto":
        return torch.device(device_arg)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(normalize: bool, augment: bool):
    ops = [transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize((0.1307,), (0.3081,)))
    if augment:
        # Gentle MNIST augments
        ops = [transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))] + ops
    return transforms.Compose(ops)


def make_dataloaders(args, device):
    transform = build_transforms(args.normalize, args.augment)
    train_ds = datasets.MNIST(args.data, train=True,
                              download=True, transform=transform)
    test_ds = datasets.MNIST(args.data, train=False, download=True,
                             transform=build_transforms(args.normalize, False))
    workers = args.workers if args.workers is not None else (
        0 if device.type == "mps" else 2)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=workers, pin_memory=pin)
    test_loader = DataLoader(test_ds,  batch_size=512,
                             shuffle=False, num_workers=workers, pin_memory=pin)
    return train_ds, test_ds, train_loader, test_loader


def make_lr_lambda(total_steps, warmup_steps, kind="cosine", step_size=0, gamma=0.1):
    warm = max(0, int(warmup_steps))

    def lr_mult(step):
        if total_steps <= 1:
            return 1.0
        if step < warm:
            return (step + 1) / max(1, warm)
        t = step - warm
        T = max(1, total_steps - warm)
        if kind == "none":
            return 1.0
        elif kind == "cosine":
            return 0.5 * (1 + math.cos(math.pi * t / T))
        elif kind == "step":
            if step_size <= 0:
                return 1.0
            k = t // step_size
            return (gamma ** int(k))
        else:
            return 1.0
    return lr_mult


def make_val_scheduler(init_val, final_val, total_steps, warmup_steps, kind="const"):
    init_val = float(init_val)
    final_val = float(final_val)
    warm = max(0, int(warmup_steps))

    def val_at(step):
        if total_steps <= 1:
            return final_val
        if kind == "const" or abs(final_val - init_val) < 1e-12:
            return init_val
        if step < warm:
            # warm from 0 -> init_val
            return init_val * (step + 1) / max(1, warm)
        t = step - warm
        T = max(1, total_steps - warm)
        if kind == "linear":
            return init_val + (final_val - init_val) * (t / T)
        elif kind == "cosine":
            # cosine from init_val to final_val
            c = 0.5 * (1 - math.cos(math.pi * t / T))
            return init_val + (final_val - init_val) * c
        else:
            return init_val
    return val_at

# ----------------------------
# Train/Eval helpers
# ----------------------------


def one_batch_overfit(model, opt, args, device, train_loader, feat_sched, head_sched, clip_grad_norm=0.0):
    model.train()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    logger.info(f"[overfit] batch size = {xb.size(0)}")
    for step in range(args.one_batch_overfit):
        opt.zero_grad(set_to_none=True)
        logits, feats = model(xb)
        ce = F.cross_entropy(logits, yb)
        lam_f = feat_sched(step)
        lam_h = head_sched(step)
        reg_feat = feature_gram_logdet(
            feats, lam_f, eps=1e-4, normalize=args.feat_norm)
        reg_head = head_jac_logdet(
            model, lam_h, eps=1e-4, normalize=args.head_norm)
        loss = ce + reg_feat + reg_head
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad_norm)
        opt.step()
        if args.grad_log and (step % 10 == 0 or step == args.one_batch_overfit - 1):
            with torch.no_grad():
                total_norm = torch.norm(torch.stack(
                    [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]))
                acc = accuracy(logits, yb)
            logger.info(f"[overfit {step+1}/{args.one_batch_overfit}] loss={loss.item():.4f} acc={acc:.4f} "
                        f"CE={ce.item():.4f} Feat={reg_feat.item():.4f} Head={reg_head.item():.4f} "
                        f"||grad||={total_norm.item():.3e} λ_f={lam_f:.2e} λ_h={lam_h:.2e}")
    logger.info("[overfit] final train acc: %.4f" % accuracy(logits, yb))

# ----------------------------
# Main
# ----------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head_whiten", action="store_true",
                   help="use whitened head objective & log whitened head cond (uses EMA of pooled features)")
    p.add_argument("--head_whiten_momentum", type=float, default=0.99)

    # training / data
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--opt", type=str, default="adamw",
                   choices=["sgd", "adamw"])
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--augment", action="store_true")

    # GLT
    p.add_argument("--glt_freq", type=int, default=20)
    p.add_argument("--glt_prob", type=float, default=0.0)
    p.add_argument("--glt_burnin_steps", type=int, default=0)

    # regularizers init λ
    p.add_argument("--feat_lambda", type=float, default=0.0)
    p.add_argument("--feat_norm", action="store_true")
    p.add_argument("--head_jac_lambda", type=float, default=0.0)

    # regularizer schedules (same as before)
    p.add_argument("--feat_lambda_final", type=float, default=None)
    p.add_argument("--feat_lambda_sched", type=str,
                   default="const", choices=["const", "linear", "cosine"])
    p.add_argument("--feat_warmup_steps", type=int, default=0)
    p.add_argument("--head_jac_lambda_final", type=float, default=None)
    p.add_argument("--head_jac_lambda_sched", type=str,
                   default="const", choices=["const", "linear", "cosine"])
    p.add_argument("--head_warmup_steps", type=int, default=0)

    # LR schedule
    p.add_argument("--lr_sched", type=str, default="cosine",
                   choices=["cosine", "step", "none"])
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--step_size", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--clip_grad_norm", type=float, default=0.0)
    p.add_argument("--deterministic", action="store_true")

    # sanity
    p.add_argument("--one_batch_overfit", type=int, default=0)
    p.add_argument("--grad_log", action="store_true")

    # *** NEW: initialization controls ***
    p.add_argument("--init_mode", type=str, default="kaiming",
                   choices=["kaiming", "orthogonal"])
    p.add_argument("--init_head_ortho", action="store_true")
    p.add_argument("--head_scale", type=float, default=1.0)
    p.add_argument("--init_balance_channels", action="store_true")
    p.add_argument("--bn_gamma", type=float, default=1.0)
    p.add_argument("--bn_beta",  type=float, default=0.0)
    p.add_argument("--init_log", action="store_true")
    p.add_argument("--head_norm", action="store_true",
                   help="use scale-invariant head log-det (equalizes VV^T eigenvalues)")

    args = p.parse_args()

    # Display configuration
    config_table = Table.grid(padding=(0, 2))
    config_table.add_column(style="cyan", justify="right")
    config_table.add_column(style="magenta")
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Batch Size", str(args.batch))
    config_table.add_row("Learning Rate", f"{args.lr:.2e}")
    config_table.add_row("Weight Decay", str(args.wd))
    config_table.add_row("Optimizer", args.opt.upper())
    config_table.add_row("Device", args.device)
    config_table.add_row("Seed", str(args.seed))
    c1 = getattr(args, 'c1', 32)
    c2 = getattr(args, 'c2', 64)
    config_table.add_row("Model", f"SmallConv(c1={c1}, c2={c2})")
    config_table.add_row("Initialization", args.init_mode +
                         (" + head_ortho" if args.init_head_ortho else ""))
    if args.init_head_ortho:
        config_table.add_row("Head Scale", f"{args.head_scale:.2f}")
    if args.init_balance_channels:
        config_table.add_row("Init Balance Channels", "True")
    config_table.add_row("BN Params", f"γ={args.bn_gamma}, β={args.bn_beta}")
    config_table.add_row(
        "Feature λ (init)", f"{args.feat_lambda:.2e}" + (" [normalized]" if args.feat_norm else ""))
    config_table.add_row("Feature λ (final/sched)",
                         f"{(args.feat_lambda_final if args.feat_lambda_final is not None else args.feat_lambda):.2e} / {args.feat_lambda_sched} (+warmup {args.feat_warmup_steps})")
    config_table.add_row("Head Jac λ (init)", f"{args.head_jac_lambda:.2e}")
    config_table.add_row("Head Jac λ (final/sched)",
                         f"{(args.head_jac_lambda_final if args.head_jac_lambda_final is not None else args.head_jac_lambda):.2e} / {args.head_jac_lambda_sched} (+warmup {args.head_warmup_steps})")
    config_table.add_row(
        "GLT", f"freq={args.glt_freq}, prob={args.glt_prob}, burn-in={args.glt_burnin_steps}")
    config_table.add_row(
        "LR sched", f"{args.lr_sched} (warmup {args.warmup_steps}, step_size={args.step_size}, gamma={args.gamma})")
    config_table.add_row("Augment/Normalize",
                         f"{args.augment}/{args.normalize}")
    console.print(Panel(
        config_table, title="[bold blue]MNIST Experiment Configuration[/bold blue]", border_style="blue", box=box.ROUNDED))

    # Seeds / determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {args.seed}")

    # Device & data
    device = pick_device(args.device)
    train_ds, test_ds, train_loader, test_loader = make_dataloaders(
        args, device)
    logger.info(
        f"Train samples: {len(train_ds):,} | Test samples: {len(test_ds):,} | Batches/epoch: {len(train_loader)}")
    if device.type == 'mps':
        logger.warning(
            "Using MPS - eigens/logdet/QR on CPU (PyTorch limitation)")

    model = SmallConv(c1=getattr(args, 'c1', 32),
                      c2=getattr(args, 'c2', 64)).to(device)

    c2 = model.head.in_features
    head_S_ema = RunningSecondMoment(
        c2, momentum=args.head_whiten_momentum, device=device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} (trainable: {n_trainable:,})")

    if args.init_mode == "orthogonal":
        init_orthogonal(model)
    else:
        init_kaiming(model)
    set_bn_affine(model, gamma=args.bn_gamma, beta=args.bn_beta)
    if args.init_head_ortho:
        init_head_orthonormal(model, scale=args.head_scale)
    if args.init_balance_channels:
        balance_interface_channels(model)

    # Optional: log initial conditioning
    if args.init_log:
        model.eval()
        with torch.no_grad():
            xb, _ = next(iter(train_loader))
            xb = xb[:128].to(device)
            logits, feats = model(xb)
            gram0 = compute_gram_cond(feats.detach().cpu())
            head0 = compute_head_cond(model.to("cpu"))
            model.to(device)
        logger.info(f"[init] Gram cond: {gram0:.1f} | Head cond: {head0:.2f}")

    # Optimizer & schedulers (same as before)
    if args.opt == "adamw":
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=args.wd)

    total_steps = max(1, args.epochs * len(train_loader))
    # make_lr_lambda / make_val_scheduler defined earlier in your script
    lr_lambda = make_lr_lambda(total_steps, args.warmup_steps,
                               kind=args.lr_sched, step_size=args.step_size, gamma=args.gamma)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    feat_final = args.feat_lambda if args.feat_lambda_final is None else args.feat_lambda_final
    head_final = args.head_jac_lambda if args.head_jac_lambda_final is None else args.head_jac_lambda_final
    feat_sched = make_val_scheduler(
        args.feat_lambda, feat_final, total_steps, args.feat_warmup_steps, args.feat_lambda_sched)
    head_sched = make_val_scheduler(args.head_jac_lambda, head_final,
                                    total_steps, args.head_warmup_steps, args.head_jac_lambda_sched)

    # One-batch overfit (unchanged logic, call with lam from schedulers)
    if args.one_batch_overfit > 0:
        one_batch_overfit(model, opt, args, device, train_loader,
                          feat_sched, head_sched, clip_grad_norm=args.clip_grad_norm)
        return

    # Results table
    results_table = Table(title="Training Results", box=box.ROUNDED,
                          show_header=True, header_style="bold magenta")
    for col, jst, style in [
        ("Epoch", "center", "cyan"), ("Train Loss", "right",
                                      "yellow"), ("Train Acc", "right", "green"),
        ("Test Acc", "right", "green bold"), ("Gram Cond",
                                              "right", "blue"), ("Head Cond", "right", "blue"),
        ("LR", "right", "magenta"), ("Time", "right", "magenta")
    ]:
        results_table.add_column(col, justify=jst, style=style)

    # CSV + JSON artifacts
    os.makedirs('artifacts', exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = f'artifacts/metrics_{tag}.csv'
    json_path = f'artifacts/summary_{tag}.json'
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_acc,test_acc,gram_cond,head_cond,lr,ce,reg_feat,reg_head,feat_lambda_mean,head_lambda_mean,epoch_time\n")

    console.print("\n[bold green]Starting Training...[/bold green]\n")
    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        total_loss = total_acc = total_ce = total_reg_feat = total_reg_head = 0.0
        lam_f_sum = lam_h_sum = 0.0
        n_batches = 0

        with Progress(
            SpinnerColumn(), TextColumn(
                "[progress.description]{task.description}"), BarColumn(),
            TextColumn(
                "[progress.percentage]{task.percentage:>3.0f}%"), TextColumn("•"),
            TextColumn(
                "[cyan]{task.completed}/{task.total}"), TextColumn("•"), TimeElapsedColumn(),
            console=console, transient=True
        ) as progress:
            task = progress.add_task(
                f"[yellow]Epoch {epoch}/{args.epochs}", total=len(train_loader))
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)

                logits, feats = model(xb)
                ce = F.cross_entropy(logits, yb)

                lam_f = feat_sched(global_step)
                lam_h = head_sched(global_step)
                reg_feat = feature_gram_logdet(
                    feats, lam_f, eps=1e-4, normalize=args.feat_norm)

                if args.head_whiten:
                    reg_head = head_logdet_whitened(
                        model, head_S_ema.get(), lam_h, eps=1e-4, normalize=args.head_norm)
                else:
                    reg_head = head_jac_logdet(
                        model, lam_h, eps=1e-4, normalize=args.head_norm)

                loss = ce + reg_feat + reg_head
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.clip_grad_norm)
                opt.step()
                sched.step()

                # GLT: freq OR stochastic; respect burn-in
                if global_step >= args.glt_burnin_steps:
                    fired = False
                    if args.glt_freq > 0 and (global_step % args.glt_freq) == 0:
                        aligned_glt_permute(model, opt, global_step)
                        fired = True
                    if (not fired) and args.glt_prob > 0.0 and random.random() < args.glt_prob:
                        aligned_glt_permute(model, opt, global_step)

                total_loss += loss.item()
                total_ce += ce.item()
                total_reg_feat += reg_feat.item()
                total_reg_head += reg_head.item()
                total_acc += accuracy(logits.detach(), yb)
                lam_f_sum += lam_f
                lam_h_sum += lam_h
                n_batches += 1
                global_step += 1

                lr_now = opt.param_groups[0]['lr']
                progress.update(task, advance=1,
                                description=f"[yellow]Epoch {epoch}/{args.epochs}[/yellow] - "
                                            f"Loss {total_loss/n_batches:.4f} Acc {total_acc/n_batches:.4f} LR {lr_now:.2e}")

        # evaluation
        logger.info(f"Evaluating epoch {epoch}...")
        model.eval()
        with torch.no_grad():
            acc_list = []
            gram_cond = None
            for i, (xb, yb) in enumerate(test_loader):
                xb, yb = xb.to(device), yb.to(device)
                logits, feats = model(xb)
                acc_list.append(accuracy(logits, yb))

                with torch.no_grad():
                    g_batch = feats.mean(dim=(2, 3))  # [B, C2] pooled features
                    head_S_ema.update(g_batch)

                if i == 0:
                    # Gram cond on CPU (MPS safe)
                    feats_cpu = feats.detach().float().cpu()
                    gram_cond = compute_gram_cond(feats_cpu)
            test_acc = float(np.mean(acc_list))
            # compute on CPU for consistency
            head_cond = compute_head_cond(model.to("cpu")).__float__()
            # evaluation (inside with torch.no_grad())
            S_now = head_S_ema.get()  # on device
            raw_head_cond = compute_head_cond(
                model.to("cpu"), None).__float__()
            white_head_cond = compute_head_cond(
                model.to("cpu"), S_now).__float__()

            model.to(device)

        dt = time.time() - t0
        lr_now = opt.param_groups[0]['lr']
        results_table.add_row(f"{epoch:02d}",
                              f"{total_loss/n_batches:.4f}",
                              f"{total_acc/n_batches:.4f}",
                              f"{test_acc:.4f}",
                              f"{gram_cond:.1f}",
                              f"{head_cond:.2f}",
                              f"{raw_head_cond:.2f}",
                              f"{white_head_cond:.2f}",
                              f"{lr_now:.2e}",
                              f"{dt:.1f}s")

        logger.info(f"Epoch {epoch:02d} complete | "
                    f"CE: {total_ce/n_batches:.4f} | Reg(feat): {total_reg_feat/n_batches:.4f} | "
                    f"Reg(head): {total_reg_head/n_batches:.4f} | "
                    f"λ_f(mean): {lam_f_sum/max(1,n_batches):.2e} | λ_h(mean): {lam_h_sum/max(1,n_batches):.2e}")

        # CSV row
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{total_loss/n_batches:.6f},{total_acc/n_batches:.6f},{test_acc:.6f},"
                    f"{gram_cond:.6e},{head_cond:.6e},{lr_now:.6e},{total_ce/n_batches:.6f},"
                    f"{total_reg_feat/n_batches:.6f},{total_reg_head/n_batches:.6f},"
                    f"{lam_f_sum/max(1,n_batches):.6e},{lam_h_sum/max(1,n_batches):.6e},{dt:.3f}\n")

    # Display final results table
    console.print("\n")
    console.print(results_table)

    # Save artifacts
    console.print("\n[bold cyan]Saving artifacts...[/bold cyan]")
    os.makedirs('artifacts', exist_ok=True)
    model_path = f'artifacts/mnist_conv_glt_feat_{tag}.pt'
    args_txt = f'artifacts/args_{tag}.txt'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model checkpoint saved to {model_path}")
    with open(args_txt, 'w') as f:
        f.write(str(vars(args)))
        logger.info(f"Arguments saved to {args_txt}")

    # JSON summary
    summary = {
        "timestamp": tag,
        "params_total": int(n_params),
        "device": str(device),
        "last_lr": float(opt.param_groups[0]['lr']),
        "csv": csv_path,
        "checkpoint": model_path,
        "args": vars(args)
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON saved to {json_path}")

    # Final panel
    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="cyan", justify="right")
    summary_table.add_column(style="green")
    summary_table.add_row("Model checkpoint", model_path)
    summary_table.add_row("Metrics CSV", csv_path)
    summary_table.add_row("Summary JSON", json_path)
    summary_table.add_row("Total parameters", f"{n_params:,}")
    console.print(Panel(
        summary_table, title="[bold green]Experiment Complete[/bold green]", border_style="green", box=box.ROUNDED))


if __name__ == "__main__":
    main()
