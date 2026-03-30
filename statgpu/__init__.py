"""
statgpu: GPU-accelerated statistical methods

A sklearn-compatible library for statistical computing with GPU support.
"""

__version__ = "0.1.0"

from ._config import get_device, set_device, Device
from ._base import BaseEstimator

__all__ = [
    "get_device",
    "set_device", 
    "Device",
    "BaseEstimator",
]
