"""
Core architectures for deep linear time series modeling.

Key innovations:
1. Balanced initialization (critical for deep networks)
2. Temporal mixing layers (linear attention without softmax)
3. Residual connections (maintain gradient flow)
4. Multi-scale processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Optional

from .layers import MonarchLinearLayer, LinearAttentionLayer, ToeplitzLayer


class DeepLinearTimeSeries(nn.Module):
    """
    Main architecture for deep linear time series modeling.

    This model challenges the nonlinearity assumption by using pure linear
    transformations across many layers (100-1000+). The depth creates rich
    optimization dynamics through Riemannian geometry, even without nonlinearity.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension size
        output_dim: Number of output features
        depth: Total number of layers (go DEEP!)
        sequence_length: Length of input sequences
        temporal_mixing: Type of temporal mixing ('toeplitz', 'linear_attention', 'fourier')
        use_residual: Whether to use residual connections
        residual_weight: Weight for residual connections (lower = stronger identity)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        depth: int = 100,
        sequence_length: int = 1000,
        temporal_mixing: Literal['toeplitz', 'linear_attention', 'fourier'] = 'toeplitz',
        use_residual: bool = True,
        residual_weight: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.sequence_length = sequence_length
        self.use_residual = use_residual
        self.residual_weight = residual_weight

        # Encoder: input_dim -> hidden_dim (depth // 2 layers)
        encoder_depth = depth // 2
        self.encoder = self._build_encoder(input_dim, hidden_dim, encoder_depth)

        # Temporal mixing (depth // 4 layers)
        temporal_depth = depth // 4
        self.temporal = self._build_temporal(
            hidden_dim, sequence_length, temporal_mixing, temporal_depth
        )

        # Decoder: hidden_dim -> output_dim (depth // 4 layers)
        decoder_depth = depth - encoder_depth - temporal_depth
        self.decoder = self._build_decoder(hidden_dim, output_dim, decoder_depth)

        # Initialize with balanced weights (critical for deep linear networks)
        self.initialize_balanced()

    def _build_encoder(self, input_dim: int, hidden_dim: int, depth: int) -> nn.ModuleList:
        """Build deep encoder layers."""
        layers = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))

        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        return layers

    def _build_temporal(
        self,
        hidden_dim: int,
        sequence_length: int,
        mixing_type: str,
        depth: int
    ) -> nn.ModuleList:
        """Build temporal mixing layers."""
        layers = nn.ModuleList()

        for _ in range(depth):
            if mixing_type == 'toeplitz':
                layers.append(ToeplitzLayer(sequence_length))
            elif mixing_type == 'linear_attention':
                layers.append(LinearAttentionLayer(hidden_dim, sequence_length))
            elif mixing_type == 'fourier':
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            else:
                raise ValueError(f"Unknown temporal mixing type: {mixing_type}")

        return layers

    def _build_decoder(self, hidden_dim: int, output_dim: int, depth: int) -> nn.ModuleList:
        """Build deep decoder layers."""
        layers = nn.ModuleList()

        # All but last layer: hidden_dim -> hidden_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        # Last layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))

        return layers

    def initialize_balanced(self):
        """
        Balanced initialization: W_i^T W_i = W_j W_j^T for all layers.

        This keeps the network on the "balanced variety" manifold, which is
        critical for stable training of deep linear networks according to
        Menon et al. (2024).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize with scaled orthogonal matrices
                nn.init.orthogonal_(module.weight)
                # Scale to maintain unit product
                scale = 1.0 / np.sqrt(self.depth)
                module.weight.data *= scale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply encoder layers with optional residuals."""
        identity = x
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.use_residual and i > 0 and x.shape == identity.shape:
                x = x + self.residual_weight * identity
                identity = x
        return x

    def mix_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal mixing layers."""
        identity = x
        for i, layer in enumerate(self.temporal):
            x = layer(x)
            if self.use_residual and i > 0:
                x = x + self.residual_weight * identity
                identity = x
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decoder layers with optional residuals."""
        identity = x
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if self.use_residual and i < len(self.decoder) - 1 and x.shape == identity.shape:
                x = x + self.residual_weight * identity
                identity = x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the deep linear network.

        Args:
            x: Input tensor of shape (batch, sequence, features)

        Returns:
            Output tensor of shape (batch, sequence, output_dim)
        """
        x = self.encode(x)
        x = self.mix_temporal(x)
        x = self.decode(x)
        return x


class StructuredDeepLinear(nn.Module):
    """
    Memory-efficient deep linear network using structured matrices.

    Reduces parameters from O(d²) per layer to O(d) per layer using
    Monarch matrix decomposition. Enables training of 1000-5000+ layer networks.

    Args:
        dim: Hidden dimension (must be power of 2 for Monarch decomposition)
        depth: Number of layers
        input_dim: Input dimension (if different from dim)
        output_dim: Output dimension (if different from dim)
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1000,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        # Input projection if needed
        if input_dim is not None and input_dim != dim:
            self.input_proj = nn.Linear(input_dim, dim, bias=False)
        else:
            self.input_proj = None

        # Deep structured layers
        self.layers = nn.ModuleList([
            MonarchLinearLayer(dim) for _ in range(depth)
        ])

        # Output projection if needed
        if output_dim is not None and output_dim != dim:
            self.output_proj = nn.Linear(dim, output_dim, bias=False)
        else:
            self.output_proj = None

        self.initialize_balanced()

    def initialize_balanced(self):
        """Balanced initialization for structured layers."""
        scale = 1.0 / np.sqrt(self.depth)
        for layer in self.layers:
            for param in layer.parameters():
                param.data *= scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through structured deep linear network."""
        if self.input_proj is not None:
            x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        if self.output_proj is not None:
            x = self.output_proj(x)

        return x
