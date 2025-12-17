"""
Gradient-Less Transport (GLT) for Deep Linear Networks.

Implements channel permutations with optimizer state transport to maintain
preconditioner alignment along symmetry orbits. Based on theoretical work
showing that permutations are isometries of the product metric on the space
of factorizations.

Key concepts:
- Channel permutations leave end-to-end map unchanged
- Transporting Adam state maintains local preconditioners
- Empirically improves stability and test performance
"""

import torch
import torch.nn as nn
from typing import Optional


@torch.no_grad()
def _permute_param_and_state_linear_(
    param: torch.Tensor,
    dim: int,
    idx: torch.Tensor,
    opt: Optional[torch.optim.Optimizer]
):
    """
    Permute a parameter tensor and its optimizer state.

    Args:
        param: Parameter tensor to permute
        dim: Dimension along which to permute
        idx: Permutation indices
        opt: Optimizer (if None, only permutes parameter)
    """
    # Permute the parameter
    new = param.data.index_select(dim, idx)
    param.data.copy_(new)

    # Permute optimizer state if it exists
    if opt is not None and param in opt.state:
        st = opt.state[param]
        for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            if k in st and isinstance(st[k], torch.Tensor) and st[k].shape == param.data.shape:
                st[k].data.copy_(st[k].data.index_select(dim, idx))


@torch.no_grad()
def glt_perm_linear_pair(
    L1: nn.Linear,
    L2: nn.Linear,
    opt: Optional[torch.optim.Optimizer] = None
):
    """
    Permute hidden channels at the L1->L2 interface while transporting optimizer state.

    This leaves the end-to-end map unchanged: (L2 · L1) = (L2 · P^-1) · (P · L1)
    where P is a permutation matrix.

    Args:
        L1: First linear layer [m x h]
        L2: Second linear layer [h x n]
        opt: Optimizer (for state transport)

    The permutation is applied to:
    - Rows of L1.weight (output channels of L1)
    - L1.bias if it exists
    - Columns of L2.weight (input channels of L2)
    """
    device = L1.weight.device
    h = L1.weight.shape[0]  # Hidden dimension

    # Generate random permutation
    perm = torch.randperm(h, device=device)
    invp = torch.argsort(perm)

    # Permute L1 outputs (rows of weight, and bias)
    _permute_param_and_state_linear_(L1.weight, 0, perm, opt)
    if L1.bias is not None:
        _permute_param_and_state_linear_(L1.bias, 0, perm, opt)

    # Permute L2 inputs (columns of weight, using inverse permutation)
    _permute_param_and_state_linear_(L2.weight, 1, invp, opt)


@torch.no_grad()
def glt_over_core(
    core: nn.ModuleList,
    opt: Optional[torch.optim.Optimizer] = None
):
    """
    Apply GLT permutations over adjacent linear layers in a core chain.

    For a deep linear network with layers [L1, L2, ..., LN], this applies
    permutations at each interface: (L1, L2), (L2, L3), ..., (L(N-1), LN).

    Args:
        core: ModuleList of linear layers
        opt: Optimizer (for state transport)

    Returns:
        Number of permutations applied
    """
    n_perms = 0
    for i in range(len(core) - 1):
        if isinstance(core[i], nn.Linear) and isinstance(core[i+1], nn.Linear):
            glt_perm_linear_pair(core[i], core[i+1], opt)
            n_perms += 1
    return n_perms


@torch.no_grad()
def glt_encoder_decoder(
    encoder: nn.ModuleList,
    decoder: nn.ModuleList,
    opt: Optional[torch.optim.Optimizer] = None
):
    """
    Apply GLT permutations over encoder and decoder chains.

    Applies permutations to adjacent linear layers in both encoder and decoder.
    Does NOT permute at the encoder-decoder interface (would require temporal
    layer to be permutation-equivariant).

    Args:
        encoder: ModuleList of encoder layers
        decoder: ModuleList of decoder layers
        opt: Optimizer (for state transport)

    Returns:
        Tuple of (encoder_perms, decoder_perms)
    """
    enc_perms = glt_over_core(encoder, opt)
    dec_perms = glt_over_core(decoder, opt)
    return enc_perms, dec_perms
