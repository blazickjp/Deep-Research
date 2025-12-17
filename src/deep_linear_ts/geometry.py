"""
Geometry regularizers for deep linear networks.

Includes:
- RunningSecondMoment: EMA of feature covariance for head conditioning
- head_logdet_whitened: Whitened head regularization
- feature_gram_logdet: Scale-invariant channel diversity penalty
"""

import torch
import torch.nn as nn
from typing import Optional


class RunningSecondMoment(nn.Module):
    """
    EMA of g g^T for whitened head regularizer.

    Maintains a running estimate of E[gg^T] where g are pre-head features.
    Used to compute whitened head conditioning regularization.

    Args:
        dim: Feature dimension (C)
        momentum: EMA momentum (default: 0.99)
        device: Device for buffer allocation
    """

    def __init__(self, dim: int, momentum: float = 0.99, device=None):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('S', torch.eye(dim, device=device))

    @torch.no_grad()
    def update(self, g: torch.Tensor):
        """
        Update running second moment with new batch of features.

        Args:
            g: Features tensor [B, C]
        """
        # g: [B, C]
        cov = g.T @ g / g.size(0)  # [C, C]
        self.S.mul_(self.momentum).add_(cov, alpha=1 - self.momentum)

    def get(self) -> torch.Tensor:
        """Return current second moment estimate."""
        return self.S


def head_logdet_whitened(
    V: torch.Tensor,
    S: torch.Tensor,
    lam: float,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Whitened head log-det regularization.

    Penalizes: -λ(log det(VSV^T + εI) - K log tr(VSV^T + εI))

    This encourages the head matrix V to have well-conditioned columns
    in the feature metric S, promoting balanced utilization of all features.

    Args:
        V: Head weight matrix [K, C]
        S: Running second moment of features [C, C]
        lam: Regularization strength
        eps: Numerical stability constant

    Returns:
        Scalar regularization loss
    """
    K, C = V.shape

    # Compute V S V^T + εI
    M = V @ S @ V.T + eps * torch.eye(K, device=V.device)

    # Scale-invariant penalty: log det - K log tr
    logdet = torch.logdet(M)
    logtrace = torch.log(M.trace() + eps)
    penalty = -(logdet - K * logtrace)

    return lam * penalty


def feature_gram_logdet(
    x: torch.Tensor,
    lam: float,
    eps: float = 1e-4,
    normalize: bool = True
) -> torch.Tensor:
    """
    Scale-invariant log-det penalty on channel Gram matrix.

    Encourages diverse channel activations by penalizing the gap between
    log det and normalized trace of the feature Gram matrix.

    Args:
        x: Feature tensor [B, C] (batch, channels)
        lam: Regularization strength
        eps: Numerical stability constant
        normalize: Whether to normalize channels before computing Gram

    Returns:
        Scalar regularization loss: λ(log det G̃ - C log tr G̃)
        where G̃ = G + εI and G = x^T x / B
    """
    B, C = x.shape

    # Normalize channels to make scale-invariant
    if normalize:
        x = x / (x.norm(dim=0, keepdim=True) + 1e-8)

    # Compute Gram matrix: G = x^T x / B
    G = (x.T @ x) / B  # [C, C]
    G_reg = G + eps * torch.eye(C, device=x.device)

    # Scale-invariant penalty: log det - C log tr
    logdet = torch.logdet(G_reg)
    logtrace = torch.log(G_reg.trace() + eps)
    penalty = logdet - C * logtrace

    return lam * penalty
