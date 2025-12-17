"""
LSTM baseline for time series prediction.

Standard LSTM encoder-decoder architecture for comparison with
deep linear networks.
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    LSTM encoder-decoder for time series forecasting.

    Architecture:
    - Encoder LSTM: Processes input sequence
    - Decoder LSTM: Generates predictions
    - Output projection layer

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

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Decoder LSTM
        # Input to decoder is the output from previous step
        self.decoder_lstm = nn.LSTM(
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
        encoder_outputs, (hidden, cell) = self.encoder_lstm(x)
        # encoder_outputs: (batch, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_dim)
        # cell: (num_layers * num_directions, batch, hidden_dim)

        # If bidirectional, we need to combine forward and backward states
        if self.num_directions == 2:
            # Reshape hidden: (num_layers, 2, batch, hidden_dim)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)

            # Concatenate forward and backward, then project
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # (num_layers, batch, hidden_dim*2)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

            # Project to decoder hidden size
            hidden = self.encoder_projection(hidden.transpose(0, 1)).transpose(0, 1)  # (num_layers, batch, hidden_dim)
            cell = self.encoder_projection(cell.transpose(0, 1)).transpose(0, 1)

        # Decode autoregressively
        decoder_input = self.decoder_start.expand(batch_size, 1, -1)  # (batch, 1, output_dim)
        outputs = []

        for _ in range(self.prediction_length):
            # Single step prediction
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            # decoder_output: (batch, 1, hidden_dim)

            # Project to output dimension
            output = self.output_projection(decoder_output)  # (batch, 1, output_dim)
            outputs.append(output)

            # Use output as next input (autoregressive)
            decoder_input = output

        # Concatenate all predictions
        predictions = torch.cat(outputs, dim=1)  # (batch, prediction_length, output_dim)

        return predictions
