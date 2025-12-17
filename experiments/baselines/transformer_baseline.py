"""
Transformer baseline for time series prediction.

Standard transformer encoder-decoder architecture for comparison with
deep linear networks.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, d_model)

        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerBaseline(nn.Module):
    """
    Transformer encoder-decoder for time series forecasting.

    Architecture:
    - Input projection: input_dim -> d_model
    - Positional encoding
    - Transformer encoder (multiple layers)
    - Transformer decoder (multiple layers)
    - Output projection: d_model -> output_dim

    Matches DeepLinearTimeSeries interface for fair comparison.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        sequence_length: int = 96,
        prediction_length: int = 24,
    ):
        super().__init__()

        self.d_model = d_model
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        # Input/output projections
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, output_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length + prediction_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Initialize decoder input (learned)
        self.decoder_input = nn.Parameter(torch.randn(prediction_length, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, sequence_length, input_dim)

        Returns:
            predictions: (batch, prediction_length, output_dim)
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)  # (batch, seq, d_model)

        # Reshape to (seq, batch, d_model) for transformer
        x = x.transpose(0, 1)  # (seq, batch, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode
        memory = self.transformer_encoder(x)  # (seq, batch, d_model)

        # Prepare decoder input (expand learned parameter for batch)
        tgt = self.decoder_input.expand(-1, batch_size, -1)  # (pred_len, batch, d_model)

        # Decode
        output = self.transformer_decoder(tgt, memory)  # (pred_len, batch, d_model)

        # Reshape back to (batch, pred_len, d_model)
        output = output.transpose(0, 1)

        # Project to output dimension
        output = self.output_projection(output)  # (batch, pred_len, output_dim)

        return output
