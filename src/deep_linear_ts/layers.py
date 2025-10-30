"""
Specialized linear layers for deep time series networks.

Includes:
- MonarchLinearLayer: O(d) parameters using structured decomposition
- LinearAttentionLayer: Linear attention without softmax
- ToeplitzLayer: Convolution-like temporal mixing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from einops import rearrange


class MonarchLinearLayer(nn.Module):
    """
    Monarch matrix decomposition: W = (P₁L₁)(P₂L₂)...(PₖLₖ)

    Uses butterfly structure to reduce parameters from O(d²) to O(d).
    Critical for scaling to 1000-5000+ layers.

    Args:
        dim: Input/output dimension (must be power of 2)
        n_blocks: Number of block diagonal matrices (default: log2(dim))
    """

    def __init__(self, dim: int, n_blocks: Optional[int] = None):
        super().__init__()

        assert (dim & (dim - 1)) == 0, "dim must be power of 2"

        self.dim = dim
        self.n_blocks = n_blocks or int(np.log2(dim))

        # Block diagonal matrices (each is O(d) parameters)
        self.blocks = nn.ParameterList([
            nn.Parameter(torch.randn(2, dim // 2) * 0.01)
            for _ in range(self.n_blocks)
        ])

        # Permutations (fixed, no learnable parameters)
        self.register_buffer('permutations', self._compute_butterfly_permutations())

    def _compute_butterfly_permutations(self) -> torch.Tensor:
        """Compute fixed butterfly permutation pattern."""
        indices = torch.arange(self.dim)
        perms = []

        for i in range(self.n_blocks):
            stride = 2 ** (i + 1)
            perm = indices.clone()

            # Butterfly shuffle pattern
            for j in range(0, self.dim, stride):
                perm[j:j+stride] = torch.cat([
                    indices[j:j+stride//2],
                    indices[j+stride//2:j+stride]
                ])

            perms.append(perm)

        return torch.stack(perms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Monarch structured linear transformation.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Output tensor [..., dim]
        """
        for i, block in enumerate(self.blocks):
            # Reshape to apply block diagonal structure
            x_reshaped = x.reshape(*x.shape[:-1], 2, self.dim // 2)

            # Apply block diagonal matrix
            x = torch.einsum('...ij,ji->...ij', x_reshaped, block)
            x = x.reshape(*x.shape[:-2], self.dim)

            # Apply butterfly permutation
            if i < self.n_blocks - 1:
                x = x[..., self.permutations[i]]

        return x


class LinearAttentionLayer(nn.Module):
    """
    Linear attention without softmax (truly linear!).

    Uses kernel feature maps to compute attention in O(n d²) instead of O(n² d).
    Key insight: No softmax means no nonlinearity!

    Args:
        dim: Feature dimension
        sequence_length: Maximum sequence length
        feature_dim: Dimension of kernel feature map (default: dim)
    """

    def __init__(
        self,
        dim: int,
        sequence_length: int,
        feature_dim: Optional[int] = None
    ):
        super().__init__()

        self.dim = dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim or dim

        # Linear projections (no nonlinearity!)
        self.query = nn.Linear(dim, self.feature_dim, bias=False)
        self.key = nn.Linear(dim, self.feature_dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear attention computation.

        Args:
            x: Input tensor (batch, sequence, dim)

        Returns:
            Output tensor (batch, sequence, dim)
        """
        # Project to queries, keys, values
        Q = self.query(x)  # (batch, seq, feature_dim)
        K = self.key(x)    # (batch, seq, feature_dim)
        V = self.value(x)  # (batch, seq, dim)

        # Linear attention: (K^T V) / (K^T 1), then multiply by Q
        # This is O(n d²) instead of O(n² d)
        KV = torch.einsum('bsi,bsj->bij', K, V)  # (batch, feature_dim, dim)
        K_sum = K.sum(dim=1, keepdim=True)       # (batch, 1, feature_dim)

        # Normalize and apply to queries
        output = torch.einsum('bsi,bij->bsj', Q, KV)  # (batch, seq, dim)
        normalizer = torch.einsum('bsi,bji->bsj', Q, K_sum.transpose(-1, -2))

        # Avoid division by zero
        output = output / (normalizer + 1e-6)

        return output


class ToeplitzLayer(nn.Module):
    """
    Toeplitz matrix layer for temporal convolution.

    A Toeplitz matrix has constant diagonals, making it equivalent to
    convolution. This is a linear operation on the temporal dimension.

    Args:
        sequence_length: Length of input sequences
        kernel_size: Size of the Toeplitz kernel (default: sequence_length)
    """

    def __init__(self, sequence_length: int, kernel_size: Optional[int] = None):
        super().__init__()

        self.sequence_length = sequence_length
        self.kernel_size = kernel_size or sequence_length

        # Learnable kernel (defines the Toeplitz structure)
        self.kernel = nn.Parameter(torch.randn(self.kernel_size) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Toeplitz transformation along sequence dimension.

        Args:
            x: Input tensor (batch, sequence, features)

        Returns:
            Output tensor (batch, sequence, features)
        """
        # Apply convolution along sequence dimension (Toeplitz structure)
        # Rearrange to (batch, features, sequence) for conv1d
        x = rearrange(x, 'b s f -> b f s')

        # Pad to maintain sequence length
        padding = self.kernel_size // 2
        x = F.conv1d(x, self.kernel.view(1, 1, -1), padding=padding)

        # Crop to original length if needed
        if x.size(-1) > self.sequence_length:
            x = x[..., :self.sequence_length]

        # Rearrange back to (batch, sequence, features)
        x = rearrange(x, 'b f s -> b s f')

        return x


class FourierLayer(nn.Module):
    """
    Fourier domain linear layer.

    Applies linear transformation in frequency domain, which is still
    a linear operation but can capture global temporal patterns efficiently.

    Args:
        dim: Feature dimension
        modes: Number of Fourier modes to use (default: dim // 2)
    """

    def __init__(self, dim: int, modes: Optional[int] = None):
        super().__init__()

        self.dim = dim
        self.modes = modes or dim // 2

        # Complex weights for Fourier modes (stored as real + imaginary)
        self.weight_real = nn.Parameter(torch.randn(self.modes, dim, dim) * 0.01)
        self.weight_imag = nn.Parameter(torch.randn(self.modes, dim, dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier domain linear transformation.

        Args:
            x: Input tensor (batch, sequence, features)

        Returns:
            Output tensor (batch, sequence, features)
        """
        # FFT along sequence dimension
        x_ft = torch.fft.rfft(x, dim=1)

        # Apply linear transformation to first 'modes' frequencies
        out_ft = torch.zeros_like(x_ft)

        for i in range(min(self.modes, x_ft.size(1))):
            # Complex multiplication: (a + bi)(W_r + W_i i)
            real = x_ft[:, i].real
            imag = x_ft[:, i].imag

            out_real = torch.einsum('bf,fk->bk', real, self.weight_real[i]) - \
                       torch.einsum('bf,fk->bk', imag, self.weight_imag[i])

            out_imag = torch.einsum('bf,fk->bk', real, self.weight_imag[i]) + \
                       torch.einsum('bf,fk->bk', imag, self.weight_real[i])

            out_ft[:, i] = torch.complex(out_real, out_imag)

        # IFFT back to time domain
        x = torch.fft.irfft(out_ft, n=x.size(1), dim=1)

        return x
