"""Generic cross-validation framework.

Provides CVEstimatorBase, kfold_indices, hash_cv_data, batch_mse,
and other CV utilities. Used by linear_model, survival, and other modules.
"""

from ._base import (
    CVEstimatorBase,
    CVCache,
    kfold_indices,
    folds_are_complete,
    hash_cv_data,
    validate_cv_sample_weight,
    batch_mse,
    detect_gpu_input,
    INTERCEPT_CLIP_BOUND,
)
from ._engine import run_cv

__all__ = [
    "CVEstimatorBase",
    "CVCache",
    "kfold_indices",
    "folds_are_complete",
    "hash_cv_data",
    "validate_cv_sample_weight",
    "batch_mse",
    "detect_gpu_input",
    "INTERCEPT_CLIP_BOUND",
    "run_cv",
]
