"""
Generic cross-validation engine for penalized GLM models.

Provides a reusable CV loop that can be parameterized by:
- Any loss function (squared_error, logistic, poisson, etc.)
- Any penalty type (l1, l2, elasticnet, scad, mcp, etc.)
- Any backend (numpy, cupy, torch)

.. note::

    **Reference Implementation**: ``run_cv`` is a simple, readable reference
    implementation intended for:
    - Custom estimators that need a basic CV loop
    - Testing and prototyping new CV strategies
    - Documentation of the CV algorithm

    The production CV paths (PenalizedGLM_CV, LassoCV, RidgeCV, etc.) use
    their own optimized loops with warm-starting, fold batching, and
    backend-specific optimizations.  For production use, prefer those
    estimators directly.
"""

from __future__ import annotations

__all__ = ["CVEngine"]

import logging
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from statgpu.linear_model._cv_base import (
    CVCache,
    kfold_indices,
)

logger = logging.getLogger(__name__)


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

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Feature matrix.
    y : array, shape (n_samples,)
        Target vector.
    alpha_grid : array, shape (n_alphas,)
        Regularization parameter grid.
    evaluate_fold_fn : callable
        Function ``(X_train, y_train, X_val, y_val, alpha,
        sample_weight_train=None, sample_weight_val=None) -> score``
        that trains on the training fold and returns a scalar score on
        the validation fold.
    n_folds : int
        Number of CV folds.
    random_state : int or None
        Random seed for fold generation.
    minimize : bool
        If True, lower score is better. If False, higher score is better.
    cache : CVCache or None
        Optional LRU cache for CV results.
    cache_key_fn : callable or None
        Function ``(X, y, alpha_grid, folds) -> str`` for cache key.
    sample_weight : array or None
        Optional sample weights (passed through to evaluate_fold_fn).
    raise_on_error : bool, default False
        If True, re-raise exceptions from evaluate_fold_fn instead of
        logging a warning and setting the score to NaN.

    Returns
    -------
    best_alpha : float
        Alpha value that optimizes the CV score.
    mean_scores : array, shape (n_alphas,)
        Mean CV score for each alpha.
    all_scores : array, shape (n_folds, n_alphas,)
        Per-fold CV scores.
    """
    # 0. Validate inputs
    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError(f"X and y have different number of samples: {n_samples} vs {y.shape[0]}")
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid must not be empty")
    if sample_weight is not None and len(sample_weight) != n_samples:
        raise ValueError(
            f"sample_weight length {len(sample_weight)} != n_samples {n_samples}"
        )

    # 1. Generate folds
    folds = kfold_indices(n_samples, n_folds, random_state)

    # 2. Check cache
    cache_key = None
    if cache is not None and cache_key_fn is not None:
        cache_key = cache_key_fn(X, y, alpha_grid, folds)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # 3. Evaluate each (fold, alpha) pair
    n_alphas = len(alpha_grid)
    all_scores = np.full((n_folds, n_alphas), np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        sw_val = sample_weight[val_idx] if sample_weight is not None else None

        for alpha_idx, alpha in enumerate(alpha_grid):
            try:
                score = evaluate_fold_fn(
                    X_train, y_train, X_val, y_val, alpha,
                    sample_weight_train=sw_train,
                    sample_weight_val=sw_val,
                )
                all_scores[fold_idx, alpha_idx] = score
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                if raise_on_error:
                    raise
                all_scores[fold_idx, alpha_idx] = np.nan
                logger.warning(
                    "CV fold %d, alpha_idx %d failed: %s",
                    fold_idx, alpha_idx, exc,
                )

    # 4. Aggregate across folds
    mean_scores = np.nanmean(all_scores, axis=0)

    # Guard against all-NaN slices (all folds failed for every alpha)
    finite_mask = np.isfinite(mean_scores)
    if not np.any(finite_mask):
        raise ValueError(
            "All CV scores are NaN — every fold failed for every alpha. "
            "Check for data issues or increase max_iter."
        )

    if minimize:
        best_idx = int(np.nanargmin(mean_scores))
    else:
        best_idx = int(np.nanargmax(mean_scores))

    best_alpha = float(alpha_grid[best_idx])

    # 5. Cache results (copy arrays to prevent mutation corruption)
    if cache is not None and cache_key_fn is not None:
        cache.put(cache_key, (best_alpha, mean_scores.copy(), all_scores.copy()))

    return best_alpha, mean_scores, all_scores
