"""Generic cross-validation engine for penalized models.

``run_cv`` is a readable reference engine. Production wrappers use optimized
warm-started loops, but this implementation still enforces the same validation
and fold-completeness contracts.
"""

from __future__ import annotations

__all__ = ["run_cv"]

import logging
from numbers import Integral
from typing import Callable, Optional, Tuple

import numpy as np

from statgpu.backends import _to_float_scalar
from statgpu.cross_validation._base import (
    CVCache,
    kfold_indices,
    validate_cv_sample_weight,
)

logger = logging.getLogger(__name__)


def _backend_indices(array, indices):
    """Move small fold-index metadata to the array's backend when required."""
    module = type(array).__module__
    if module.startswith("torch"):
        import torch

        return torch.as_tensor(indices, dtype=torch.long, device=array.device)
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asarray(indices, dtype=cp.int64)
    return indices


def run_cv(
    X,
    y,
    alpha_grid: np.ndarray,
    evaluate_fold_fn: Callable,
    n_folds: int = 5,
    random_state: Optional[int] = None,
    minimize: bool = True,
    cache: Optional[CVCache] = None,
    cache_key_fn: Optional[Callable] = None,
    sample_weight=None,
    raise_on_error: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Execute K-fold cross-validation.

    An alpha is eligible for selection only when every fold returns a finite
    score. This prevents an alpha evaluated on a subset of folds from being
    compared with fully evaluated candidates.
    """
    if not callable(evaluate_fold_fn):
        raise TypeError("evaluate_fold_fn must be callable")
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError(f"X must be 2D, got shape {getattr(X, 'shape', None)}")
    if not hasattr(y, "shape"):
        y = np.asarray(y)
    if len(y.shape) not in (1, 2):
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")

    n_samples = int(X.shape[0])
    if int(y.shape[0]) != n_samples:
        raise ValueError(
            f"X and y have different number of samples: {n_samples} vs {y.shape[0]}"
        )
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral):
        raise TypeError("n_folds must be an integer")
    n_folds = int(n_folds)

    alpha_grid = np.asarray(alpha_grid, dtype=float)
    if alpha_grid.ndim != 1 or alpha_grid.size == 0:
        raise ValueError("alpha_grid must be a non-empty 1D array")
    if not np.all(np.isfinite(alpha_grid)):
        raise ValueError("alpha_grid must contain only finite values")
    if np.any(alpha_grid < 0):
        raise ValueError("alpha_grid must be non-negative")

    sample_weight = validate_cv_sample_weight(sample_weight, n_samples)
    folds = kfold_indices(n_samples, n_folds, random_state)

    cache_key = None
    if cache is not None and cache_key_fn is not None:
        cache_key = cache_key_fn(X, y, alpha_grid, folds)
        cached = cache.get(cache_key)
        if cached is not None:
            best_alpha, mean_scores, all_scores = cached
            return best_alpha, mean_scores.copy(), all_scores.copy()

    n_alphas = int(alpha_grid.size)
    all_scores = np.full((n_folds, n_alphas), np.nan, dtype=float)

    for fold_idx, (train_idx_cpu, val_idx_cpu) in enumerate(folds):
        train_idx_x = _backend_indices(X, train_idx_cpu)
        val_idx_x = _backend_indices(X, val_idx_cpu)
        train_idx_y = _backend_indices(y, train_idx_cpu)
        val_idx_y = _backend_indices(y, val_idx_cpu)

        X_train = X[train_idx_x]
        y_train = y[train_idx_y]
        X_val = X[val_idx_x]
        y_val = y[val_idx_y]

        if sample_weight is not None:
            train_idx_w = _backend_indices(sample_weight, train_idx_cpu)
            val_idx_w = _backend_indices(sample_weight, val_idx_cpu)
            sw_train = sample_weight[train_idx_w]
            sw_val = sample_weight[val_idx_w]
        else:
            sw_train = sw_val = None

        for alpha_idx, alpha in enumerate(alpha_grid):
            try:
                score = evaluate_fold_fn(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    float(alpha),
                    sample_weight_train=sw_train,
                    sample_weight_val=sw_val,
                )
                score_value = _to_float_scalar(score)
                if not np.isfinite(score_value):
                    raise FloatingPointError("fold score is not finite")
                all_scores[fold_idx, alpha_idx] = score_value
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                if raise_on_error:
                    raise
                logger.warning(
                    "CV fold %d, alpha_idx %d failed: %s",
                    fold_idx,
                    alpha_idx,
                    exc,
                )

    complete = np.all(np.isfinite(all_scores), axis=0)
    mean_scores = np.full(n_alphas, np.nan, dtype=float)
    mean_scores[complete] = np.mean(all_scores[:, complete], axis=0)

    if not np.any(complete):
        raise ValueError(
            "No alpha completed every CV fold. Check the data, parameter grid, "
            "or estimator convergence settings."
        )
    if not np.all(complete):
        failed = np.flatnonzero(~complete).tolist()
        logger.warning(
            "Excluded alpha indices with incomplete fold results: %s", failed
        )

    eligible = np.flatnonzero(complete)
    eligible_scores = mean_scores[eligible]
    local_best = (
        int(np.argmin(eligible_scores))
        if minimize
        else int(np.argmax(eligible_scores))
    )
    best_idx = int(eligible[local_best])
    best_alpha = float(alpha_grid[best_idx])

    if cache is not None and cache_key_fn is not None:
        cache.put(cache_key, (best_alpha, mean_scores.copy(), all_scores.copy()))

    return best_alpha, mean_scores, all_scores
