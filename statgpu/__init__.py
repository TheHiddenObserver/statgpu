"""
statgpu: GPU-accelerated statistical methods

A sklearn-compatible library for statistical computing with GPU support.
"""

__version__ = "0.1.0"

from ._config import get_device, set_device, Device
from ._base import BaseEstimator
from .linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from .survival import CoxPH
from .backends import get_backend, NumpyBackend, CuPyBackend, TorchBackend
from .evaluation import evaluate_binary_classification
from .inference import adjust_pvalues, combine_pvalues, multipletests
from .inference import bootstrap_statistic, permutation_test

__all__ = [
    "get_device",
    "set_device",
    "Device",
    "BaseEstimator",
    "LinearRegression",
    "LogisticRegression",
    "Ridge",
    "Lasso",
    "CoxPH",
    "get_backend",
    "NumpyBackend",
    "CuPyBackend",
    "TorchBackend",
    "evaluate_binary_classification",
    "adjust_pvalues",
    "combine_pvalues",
    "multipletests",
    "bootstrap_statistic",
    "permutation_test",
]
