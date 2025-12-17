"""
GRU baseline for time series prediction.

Standard GRU encoder-decoder architecture for comparison with
deep linear networks.
"""

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    """
    GRU encoder-decoder for time series forecasting.

    Architecture:
    - Encoder GRU: Processes input sequence
    - Decoder GRU: Generates predictions
    - Output projection layer

    Similar to LSTM baseline but uses GRU cells (simpler, often faster).
    Matches DeepLinearTimeSeries interface for fair comparison.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        sequence_length: int = 96,
        prediction_length: int = 24,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.prediction_length = prediction_length

        # Encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # Decoder is always unidirectional
        )

        # Output projection
        # If encoder is bidirectional, we need to project concatenated hidden states
        encoder_output_dim = hidden_dim * self.num_directions
        self.encoder_projection = nn.Linear(encoder_output_dim, hidden_dim) if bidirectional else None

        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Learned initial decoder input
        self.decoder_start = nn.Parameter(torch.zeros(1, 1, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, sequence_length, input_dim)

        Returns:
            predictions: (batch, prediction_length, output_dim)
        """
        batch_size = x.size(0)

        # Encode
        encoder_outputs, hidden = self.encoder_gru(x)
        # encoder_outputs: (batch, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_dim)

        # If bidirectional, combine forward and backward states
        if self.num_directions == 2:
            # Reshape hidden: (num_layers, 2, batch, hidden_dim)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)

            # Concatenate forward and backward
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # (num_layers, batch, hidden_dim*2)

            # Project to decoder hidden size
            hidden = self.encoder_projection(hidden.transpose(0, 1)).transpose(0, 1)  # (num_layers, batch, hidden_dim)

        # Decode autoregressively
        decoder_input = self.decoder_start.expand(batch_size, 1, -1)  # (batch, 1, output_dim)
        outputs = []

        for _ in range(self.prediction_length):
            # Single step prediction
            decoder_output, hidden = self.decoder_gru(decoder_input, hidden)
            # decoder_output: (batch, 1, hidden_dim)

            # Project to output dimension
            output = self.output_projection(decoder_output)  # (batch, 1, output_dim)
            outputs.append(output)

            # Use output as next input (autoregressive)
            decoder_input = output

        # Concatenate all predictions
        predictions = torch.cat(outputs, dim=1)  # (batch, prediction_length, output_dim)

        return predictions
