"""
Baseline models for comparison with deep linear networks.

All baselines match the DeepLinearTimeSeries interface:
- Input: (batch, sequence_length, input_dim)
- Output: (batch, prediction_length, output_dim)
"""

from .transformer_baseline import TransformerBaseline
from .lstm_baseline import LSTMBaseline
from .gru_baseline import GRUBaseline

__all__ = ['TransformerBaseline', 'LSTMBaseline', 'GRUBaseline']
