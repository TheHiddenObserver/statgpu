"""
Generic cross-validation engine for penalized GLM models.

Provides a reusable CV loop that can be parameterized by:
- Any loss function (squared_error, logistic, poisson, etc.)
- Any penalty type (l1, l2, elasticnet, scad, mcp, etc.)
- Any backend (numpy, cupy, torch)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model._cv_base import (
    CVCache,
    batch_mse,
    folds_are_complements,
    kfold_indices,
)


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
        Function ``(X_train, y_train, X_val, y_val, alpha) -> score``
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

    Returns
    -------
    best_alpha : float
        Alpha value that optimizes the CV score.
    mean_scores : array, shape (n_alphas,)
        Mean CV score for each alpha.
    all_scores : array, shape (n_folds, n_alphas,)
        Per-fold CV scores.
    """
    n_samples = X.shape[0]

    # 1. Generate folds
    folds = kfold_indices(n_samples, n_folds, random_state)

    # 2. Check cache
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
            except Exception:
                all_scores[fold_idx, alpha_idx] = np.nan

    # 4. Aggregate across folds
    mean_scores = np.nanmean(all_scores, axis=0)

    if minimize:
        best_idx = int(np.nanargmin(mean_scores))
    else:
        best_idx = int(np.nanargmax(mean_scores))

    best_alpha = float(alpha_grid[best_idx])

    # 5. Cache results
    if cache is not None and cache_key_fn is not None:
        cache.put(cache_key, (best_alpha, mean_scores, all_scores))

    return best_alpha, mean_scores, all_scores
