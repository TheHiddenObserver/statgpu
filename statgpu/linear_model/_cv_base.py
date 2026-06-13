"""
Shared base class and utilities for cross-validated estimators.
"""

from __future__ import annotations

__all__ = ["CVEstimatorBase", "folds_are_complete", "INTERCEPT_CLIP_BOUND"]

import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from statgpu._base import BaseEstimator

# Shared constant: intercept clipping bound for CV proximal operators
INTERCEPT_CLIP_BOUND = 15.0
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
    if n_splits < 2:
        raise ValueError(f"n_splits={n_splits} must be at least 2")
    if n_splits > n_samples:
        raise ValueError(
            f"n_splits={n_splits} cannot be greater than n_samples={n_samples}"
        )

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
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


def folds_are_complete(folds, n_samples: int) -> bool:
    """Check that all folds together cover every sample exactly once."""
    val_indices = np.concatenate([f[1] for f in folds])
    if len(val_indices) != n_samples:
        return False
    return np.array_equal(np.sort(val_indices), np.arange(n_samples))


def hash_cv_data(X, y, sample_weight=None) -> bytes:
    """Compute a compact hash of X, y, and optionally sample_weight.

    For small datasets (n * p <= 10,000,000), hashes full content for zero
    collision risk.  For very large datasets, samples evenly spaced rows plus
    first/last rows, row indices, and aggregate statistics to keep hashing fast
    while minimizing collision probability.
    """
    h = hashlib.blake2b(digest_size=16)
    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    y_np = np.asarray(_to_numpy(y), dtype=np.float64).ravel()
    n, p = X_np.shape
    h.update(np.asarray([n, p], dtype=np.int64).tobytes())

    _FULL_HASH_THRESHOLD = 10_000_000  # n * p threshold for full hashing
    if n * p <= _FULL_HASH_THRESHOLD:
        # Small dataset: hash full content (zero collision risk)
        h.update(X_np.tobytes())
        h.update(y_np.tobytes())
        if sample_weight is not None:
            sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
            h.update(sw_np.tobytes())
    else:
        # Very large dataset: sample rows + indices + aggregate statistics
        # Include first and last rows (boundary) plus evenly spaced interior
        step = max(1, n // 100)
        idx = np.arange(0, n, step)[:100]
        # Ensure first and last rows are always included
        if idx[0] != 0:
            idx = np.concatenate([[0], idx])
        if idx[-1] != n - 1:
            idx = np.concatenate([idx, [n - 1]])
        # Hash row indices to prevent collision from reordered data
        h.update(idx.astype(np.int64).tobytes())
        h.update(X_np[idx].tobytes())
        h.update(y_np[idx].tobytes())
        h.update(np.asarray([X_np.mean(), X_np.std()], dtype=np.float64).tobytes())
        h.update(np.asarray([y_np.mean(), y_np.std()], dtype=np.float64).tobytes())
        if sample_weight is not None:
            sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
            h.update(sw_np[idx].tobytes())
            h.update(np.asarray([sw_np.mean(), sw_np.std()], dtype=np.float64).tobytes())
    return h.digest()


def validate_cv_sample_weight(sample_weight, n_samples: int):
    """Validate sample_weight for CV: must be non-negative and finite.

    Returns None if sample_weight is None, otherwise returns validated array.
    Raises ValueError for invalid weights.  Preserves the original backend
    (CuPy/Torch/numpy) — does not force conversion to numpy.
    """
    if sample_weight is None:
        return None
    # Validate on numpy (single D2H sync) but return original array
    sw_np = _to_numpy(sample_weight).ravel().astype(np.float64)
    if sw_np.shape[0] != n_samples:
        raise ValueError(
            f"sample_weight length {sw_np.shape[0]} != n_samples {n_samples}"
        )
    if np.any(sw_np < 0):
        raise ValueError("sample_weight must be non-negative")
    if not np.all(np.isfinite(sw_np)):
        raise ValueError("sample_weight must be finite")
    # Return the original array (preserves CuPy/Torch backend)
    return sample_weight


# ---------------------------------------------------------------------------
# LRU cache for CV results
# ---------------------------------------------------------------------------

class CVCache:
    """Simple LRU cache for cross-validation results.

    Thread-safe: all mutations are protected by a lock.

    Parameters
    ----------
    maxsize : int
        Maximum number of cached entries.
    """

    def __init__(self, maxsize: int = 64):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = __import__('threading').Lock()

    def get(self, key: str):
        """Retrieve cached result, or None if not found."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value):
        """Store a result in the cache."""
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    @staticmethod
    def make_key(*args) -> str:
        """Generate a blake2b hash key from arbitrary arguments.

        Uses content-based hashing for arrays (tobytes) to avoid collisions
        from str() truncation on large arrays.
        """
        h = hashlib.blake2b(digest_size=32)
        for arg in args:
            if hasattr(arg, 'tobytes') and hasattr(arg, 'shape'):
                # Array-like: hash shape + content bytes
                h.update(str(arg.shape).encode())
                h.update(np.ascontiguousarray(_to_numpy(arg)).tobytes())
            else:
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
    import warnings as _warnings

    x_type = None
    y_type = None

    try:
        import cupy as cp
        if isinstance(X, cp.ndarray):
            x_type = 'cupy'
        if isinstance(y, cp.ndarray):
            y_type = 'cupy'
    except ImportError:
        pass

    try:
        import torch
        if isinstance(X, torch.Tensor):
            x_type = 'torch'
        if isinstance(y, torch.Tensor):
            y_type = 'torch'
    except ImportError:
        pass

    if x_type is not None and y_type is not None and x_type != y_type:
        _warnings.warn(
            f"Mixed backend detected: X is {x_type} but y is {y_type}. "
            f"Both arrays should use the same backend. Falling back to numpy.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Convert both arrays to numpy for consistent backend
        X_np = _to_numpy(X)
        y_np = _to_numpy(y)
        return 'numpy', X_np, y_np

    if x_type == 'cupy' and y_type == 'cupy':
        return 'cupy', X, y
    if x_type == 'torch' and y_type == 'torch':
        return 'torch', X, y

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
    chunk_size: int = 256,
) -> np.ndarray:
    """Compute MSE for multiple coefficient vectors on a validation set.

    Processes models in chunks to limit peak memory to
    O(chunk_size * n_val) instead of O(n_models * n_val).

    Parameters
    ----------
    X_val : array, shape (n_val, n_features)
    y_val : array, shape (n_val,)
    coefs : array, shape (n_models, n_features)
    intercepts : array, shape (n_models,) or None
    sample_weight : array, shape (n_val,) or None
    chunk_size : int
        Number of models to process at once (default 256).

    Returns
    -------
    mse : array, shape (n_models,)
    """
    X_val = _to_numpy(X_val)
    y_val = _to_numpy(y_val).ravel()
    coefs = _to_numpy(coefs)

    # Validate dimensions
    if coefs.ndim != 2:
        raise ValueError(f"coefs must be 2D (n_models, n_features), got shape {coefs.shape}")
    if X_val.ndim != 2:
        raise ValueError(f"X_val must be 2D (n_samples, n_features), got shape {X_val.shape}")
    if coefs.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: coefs has {coefs.shape[1]} features, "
            f"X_val has {X_val.shape[1]} features"
        )
    if y_val.shape[0] != X_val.shape[0]:
        raise ValueError(
            f"Sample count mismatch: y has {y_val.shape[0]} samples, "
            f"X_val has {X_val.shape[0]} samples"
        )
    n_models = coefs.shape[0]

    if intercepts is not None:
        intercepts = _to_numpy(intercepts)

    if sample_weight is not None:
        sw = _to_numpy(sample_weight).ravel()
        sw_sum = float(np.sum(sw))
    else:
        sw = None
        sw_sum = 0.0

    mse = np.empty(n_models, dtype=np.float64)

    # Process in chunks to limit peak memory
    for start in range(0, n_models, chunk_size):
        end = min(start + chunk_size, n_models)
        coefs_chunk = coefs[start:end]

        # y_pred shape: (chunk_size, n_val)
        y_pred = coefs_chunk @ X_val.T
        if intercepts is not None:
            y_pred = y_pred + intercepts[start:end, None]

        residuals = y_val[None, :] - y_pred  # (chunk_size, n_val)

        if sw is not None:
            if sw_sum > 0:
                mse[start:end] = np.sum(residuals ** 2 * sw[None, :], axis=1) / sw_sum
            else:
                mse[start:end] = np.nan
        else:
            mse[start:end] = np.mean(residuals ** 2, axis=1)

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
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2, got {self.cv}")
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
