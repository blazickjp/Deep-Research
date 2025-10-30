"""
Deep Linear Networks for Time Series

A research framework for exploring deep linear networks (100-1000+ layers)
for time series modeling, challenging the nonlinearity assumption in deep learning.

Core components:
- DeepLinearTimeSeries: Main architecture with balanced initialization
- StructuredDeepLinear: Memory-efficient version using structured matrices
- Data utilities for time series benchmarks (ETT, M4, Traffic, Weather)
- Training and evaluation tools with geometry tracking
"""

__version__ = "0.1.0"

from .models import DeepLinearTimeSeries, StructuredDeepLinear
from .layers import MonarchLinearLayer, LinearAttentionLayer, ToeplitzLayer

__all__ = [
    "DeepLinearTimeSeries",
    "StructuredDeepLinear",
    "MonarchLinearLayer",
    "LinearAttentionLayer",
    "ToeplitzLayer",
]
