"""
Shared base class and utilities for cross-validated estimators.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy


# ---------------------------------------------------------------------------
# K-fold splitting
# ---------------------------------------------------------------------------

def kfold_indices(
    n_samples: int,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate K-fold train/validation index pairs.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_splits : int
        Number of folds.
    random_state : int or None
        Random seed for reproducibility.
    shuffle : bool
        Whether to shuffle indices before splitting.

    Returns
    -------
    folds : list of (train_idx, val_idx) tuples
    """
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current : current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size:]])
        folds.append((train_idx, val_idx))
        current += size

    return folds


def folds_are_complements(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_samples: int,
) -> bool:
    """Check that train and val indices are complementary (no overlap, no gap)."""
    combined = np.concatenate([train_idx, val_idx])
    return len(combined) == n_samples and len(np.unique(combined)) == n_samples


# ---------------------------------------------------------------------------
# LRU cache for CV results
# ---------------------------------------------------------------------------

class CVCache:
    """Simple LRU cache for cross-validation results.

    Parameters
    ----------
    maxsize : int
        Maximum number of cached entries.
    """

    def __init__(self, maxsize: int = 64):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str):
        """Retrieve cached result, or None if not found."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value):
        """Store a result in the cache."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    @staticmethod
    def make_key(*args) -> str:
        """Generate a blake2b hash key from arbitrary arguments."""
        h = hashlib.blake2b(digest_size=32)
        for arg in args:
            h.update(str(arg).encode())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# GPU input detection
# ---------------------------------------------------------------------------

def detect_gpu_input(X, y) -> Tuple[str, Any, Any]:
    """Detect whether inputs are CuPy or Torch arrays.

    Returns
    -------
    backend : str
        One of 'numpy', 'cupy', 'torch'.
    X, y : arrays
        Original arrays (unchanged).
    """
    try:
        import cupy as cp
        if isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray):
            return 'cupy', X, y
    except ImportError:
        pass

    try:
        import torch
        if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
            return 'torch', X, y
    except ImportError:
        pass

    return 'numpy', X, y


# ---------------------------------------------------------------------------
# Batch MSE computation
# ---------------------------------------------------------------------------

def batch_mse(
    X_val,
    y_val,
    coefs: np.ndarray,
    intercepts: Optional[np.ndarray] = None,
    sample_weight=None,
) -> np.ndarray:
    """Compute MSE for multiple coefficient vectors on a validation set.

    Parameters
    ----------
    X_val : array, shape (n_val, n_features)
    y_val : array, shape (n_val,)
    coefs : array, shape (n_models, n_features)
    intercepts : array, shape (n_models,) or None
    sample_weight : array, shape (n_val,) or None

    Returns
    -------
    mse : array, shape (n_models,)
    """
    X_val = _to_numpy(X_val)
    y_val = _to_numpy(y_val).ravel()
    coefs = _to_numpy(coefs)
    if intercepts is not None:
        intercepts = _to_numpy(intercepts)

    # y_pred shape: (n_models, n_val)
    y_pred = coefs @ X_val.T
    if intercepts is not None:
        y_pred = y_pred + intercepts[:, None]

    residuals = y_val[None, :] - y_pred  # (n_models, n_val)

    if sample_weight is not None:
        sw = _to_numpy(sample_weight).ravel()
        mse = np.mean(residuals ** 2 * sw[None, :], axis=1)
    else:
        mse = np.mean(residuals ** 2, axis=1)

    return mse


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CVEstimatorBase(BaseEstimator):
    """
    Common scaffolding for model-specific CV estimators.

    This is intentionally lightweight: each model keeps its own CV search
    routine and fitted attributes, while shared plumbing lives here.
    """

    def __init__(
        self,
        *,
        cv: int = 5,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cv = int(cv)
        self.random_state = random_state

        # Common fitted attributes for CV estimators.
        self.best_score_ = None
        self.cv_results_ = None
        self.estimator_ = None

    def predict(self, X):
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted base estimator is available.")
        return self.estimator_.predict(X)

    def score(self, X, y):
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted base estimator is available.")
        return self.estimator_.score(X, y)

    def summary(self):
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted base estimator is available.")
        if not hasattr(self.estimator_, "summary"):
            raise RuntimeError(
                f"{self.estimator_.__class__.__name__} does not implement summary()."
            )
        return self.estimator_.summary()
