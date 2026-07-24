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
from statgpu.backends import (
    _get_xp,
    _resolve_backend,
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
)

# Small inputs retain exact full-content hashes. Large inputs use a bounded
# row sample plus backend-native reductions to avoid an eager full GPU-to-host
# transfer solely for cache identity construction.
_FULL_HASH_THRESHOLD = 10_000_000
_LARGE_HASH_SAMPLE_ROWS = 100


def _torch_cuda_available():
    """Check if torch CUDA is available (shared utility)."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


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
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be a positive integer")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise TypeError("n_splits must be an integer")
    n_samples = int(n_samples)
    n_splits = int(n_splits)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
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
    """Check that validation folds cover every sample exactly once."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        return False
    n_samples = int(n_samples)
    if n_samples < 0 or not folds:
        return False
    try:
        val_indices = np.concatenate([np.asarray(fold[1], dtype=int) for fold in folds])
    except (TypeError, ValueError, IndexError):
        return False
    if len(val_indices) != n_samples:
        return False
    return np.array_equal(np.sort(val_indices), np.arange(n_samples))


def _shape_without_host_transfer(value) -> Tuple[int, ...]:
    """Read shape metadata without materializing a GPU array on the host."""
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    return tuple(int(dimension) for dimension in shape)


def _hash_array_metadata(h, label: bytes, value) -> None:
    """Hash shape, dtype, backend, and device metadata."""
    backend_name = _resolve_backend("auto", value)
    shape = _shape_without_host_transfer(value)
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    if backend_name == "torch":
        device = str(getattr(value, "device", "cpu"))
    elif backend_name == "cupy":
        array_device = getattr(value, "device", None)
        device = f"cuda:{getattr(array_device, 'id', array_device)}"
    else:
        device = "cpu"
    metadata = repr((shape, str(dtype), backend_name, device)).encode("utf-8")
    h.update(len(label).to_bytes(2, "big"))
    h.update(label)
    h.update(len(metadata).to_bytes(8, "big"))
    h.update(metadata)


def _as_backend_array(value):
    """Return a native backend-name, array module, and array tuple."""
    backend_name = _resolve_backend("auto", value)
    xp = _get_xp(backend_name)
    ref = value if backend_name == "torch" else None
    return backend_name, xp, xp_asarray(value, xp=xp, ref_arr=ref)


def _sample_and_summarize(value, indices, *, flatten: bool):
    """Copy bounded rows and a two-scalar native summary to the host."""
    backend_name, xp, array = _as_backend_array(value)
    if flatten:
        array = array.reshape(-1)
    index_array = xp_asarray(
        indices,
        dtype=xp.int64,
        xp=xp,
        ref_arr=array if backend_name == "torch" else None,
    )
    sample = np.ascontiguousarray(_to_numpy(array[index_array]))

    if backend_name == "torch":
        stats_array = array
        if not (stats_array.is_floating_point() or stats_array.is_complex()):
            stats_array = stats_array.to(dtype=xp.float64)
        mean = xp.mean(stats_array)
        # NumPy/CuPy and the historical implementation use correction=0.
        std = xp.std(stats_array, correction=0)
        summary = xp.stack((mean, std)).to(dtype=xp.float64)
    else:
        mean = xp.mean(array, dtype=xp.float64)
        std = xp.std(array, dtype=xp.float64)
        summary = xp.stack((mean, std)).astype(xp.float64, copy=False)

    summary_np = np.asarray(_to_numpy(summary), dtype=np.float64)
    return sample, np.ascontiguousarray(summary_np)


def hash_cv_data(X, y, sample_weight=None, *, cache_key=None) -> bytes:
    """Compute a compact cache hash for CV inputs.

    When cache_key is supplied, it is treated as a caller-controlled identity
    and array contents are not inspected. Otherwise, inputs with at most
    10,000,000 X elements hash their full contents. Larger inputs are
    classified from X.shape before any host transfer and hash at most 100
    evenly spaced rows plus backend-native mean/std summaries.

    Large-input hashes are probabilistic cache identities, not proofs that two
    complete datasets are equal. Callers with an authoritative data identity
    should pass it as cache_key to skip content hashing entirely.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"statgpu-cv-data-v2")

    if cache_key is not None:
        # Reuse the framed serializer used by the in-process CV cache instead
        # of repr().  repr(set(...)) and repr() of arbitrary objects can vary
        # with PYTHONHASHSEED or include a process-local address.
        key_payload = CVCache.make_key(cache_key).encode('ascii')
        h.update(b'explicit')
        h.update(len(key_payload).to_bytes(8, 'big'))
        h.update(key_payload)
        return h.digest()

    x_shape = _shape_without_host_transfer(X)
    if len(x_shape) != 2:
        raise ValueError(f"X must be 2D, got shape {x_shape}")
    n, p = x_shape

    _hash_array_metadata(h, b"X", X)
    _hash_array_metadata(h, b"y", y)
    if sample_weight is not None:
        _hash_array_metadata(h, b"sample_weight", sample_weight)
    h.update(np.asarray([n, p], dtype=np.int64).tobytes())

    if n * p <= _FULL_HASH_THRESHOLD:
        # Retain historical float64 normalization while hashing every value.
        # Metadata above distinguishes source dtype, backend, and device.
        X_np = np.ascontiguousarray(
            np.asarray(_to_numpy(X), dtype=np.float64)
        )
        y_np = np.ascontiguousarray(
            np.asarray(_to_numpy(y), dtype=np.float64).ravel()
        )
        h.update(X_np.tobytes())
        h.update(y_np.tobytes())
        if sample_weight is not None:
            sw_np = np.ascontiguousarray(
                np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
            )
            h.update(sw_np.tobytes())
        return h.digest()

    sample_count = min(n, _LARGE_HASH_SAMPLE_ROWS)
    indices = np.unique(
        np.linspace(0, n - 1, num=sample_count, dtype=np.int64)
    )
    h.update(indices.tobytes())

    X_sample, X_summary = _sample_and_summarize(X, indices, flatten=False)
    y_sample, y_summary = _sample_and_summarize(y, indices, flatten=True)
    h.update(X_sample.tobytes())
    h.update(X_summary.tobytes())
    h.update(y_sample.tobytes())
    h.update(y_summary.tobytes())
    if sample_weight is not None:
        sw_sample, sw_summary = _sample_and_summarize(
            sample_weight, indices, flatten=True
        )
        h.update(sw_sample.tobytes())
        h.update(sw_summary.tobytes())
    return h.digest()


def validate_cv_sample_weight(sample_weight, n_samples: int):
    """Validate CV sample weights without transferring the full vector to CPU.

    The returned array uses the same NumPy/CuPy/Torch backend as the input.
    Only scalar validation results are synchronized.
    """
    if sample_weight is None:
        return None
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be a positive integer")
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    resolved = _resolve_backend("auto", sample_weight)
    xp = _get_xp(resolved)
    ref = sample_weight if resolved in ("cupy", "torch") else None
    weights = xp_asarray(sample_weight, dtype=xp.float64, xp=xp, ref_arr=ref).ravel()
    if int(weights.shape[0]) != n_samples:
        raise ValueError(
            f"sample_weight length {weights.shape[0]} != n_samples {n_samples}"
        )
    if not bool(_to_float_scalar(xp.all(xp.isfinite(weights)))):
        raise ValueError("sample_weight must be finite")
    if bool(_to_float_scalar(xp.any(weights < 0))):
        raise ValueError("sample_weight must be non-negative")
    if _to_float_scalar(xp.sum(weights)) <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
    return weights


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
        if not isinstance(maxsize, (int, np.integer)) or int(maxsize) < 0:
            raise ValueError("maxsize must be a non-negative integer")
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = int(maxsize)
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
        """Generate a framed content hash for nested CV arguments.

        Type tags and payload lengths prevent concatenation collisions such as
        ``("ab", "c")`` versus ``("a", "bc")``. Arrays additionally include
        dtype and shape metadata before their contiguous content bytes.
        """
        h = hashlib.blake2b(digest_size=32)

        def frame(tag: bytes, payload: bytes = b"") -> None:
            h.update(len(tag).to_bytes(4, "big"))
            h.update(tag)
            h.update(len(payload).to_bytes(8, "big"))
            h.update(payload)

        def update(value) -> None:
            if value is None:
                frame(b"none")
            elif isinstance(value, (bool, np.bool_)):
                frame(b"bool", b"1" if bool(value) else b"0")
            elif isinstance(value, (int, np.integer)):
                frame(b"int", str(int(value)).encode("ascii"))
            elif isinstance(value, (float, np.floating)):
                frame(b"float", np.float64(value).tobytes())
            elif isinstance(value, str):
                frame(b"str", value.encode("utf-8"))
            elif isinstance(value, (bytes, bytearray, memoryview)):
                frame(b"bytes", bytes(value))
            elif isinstance(value, (list, tuple)):
                frame(b"list" if isinstance(value, list) else b"tuple", str(len(value)).encode())
                for item in value:
                    update(item)
            elif isinstance(value, (set, frozenset)):
                frame(b'set' if isinstance(value, set) else b'frozenset', str(len(value)).encode())
                for item in sorted(value, key=CVCache.make_key):
                    update(item)
            elif isinstance(value, dict):
                frame(b"dict", str(len(value)).encode())
                for key in sorted(value, key=CVCache.make_key):
                    update(key)
                    update(value[key])
            elif hasattr(value, "shape"):
                array = np.ascontiguousarray(_to_numpy(value))
                metadata = (array.dtype.str + "|" + repr(tuple(array.shape))).encode("utf-8")
                frame(b"array-meta", metadata)
                frame(b"array-data", array.tobytes())
            else:
                typename = f"{type(value).__module__}.{type(value).__qualname__}"
                raise TypeError(
                    "CV cache keys must contain deterministic primitive, array, "
                    f"sequence, mapping, or set values; got {typename}"
                )

        for argument in args:
            update(argument)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# GPU input detection
# ---------------------------------------------------------------------------


def detect_gpu_input(X, y) -> Tuple[str, Any, Any]:
    """Detect a common input backend, converting mixed inputs safely.

    Matching CuPy or Torch inputs are preserved. Any mixture of NumPy and
    GPU arrays, or CuPy and Torch arrays, is converted to NumPy so callers
    never receive ``backend='numpy'`` alongside an unconverted GPU object.
    """
    import warnings as _warnings

    def array_type(value):
        try:
            import cupy as cp
            if isinstance(value, cp.ndarray):
                return "cupy"
        except ImportError:
            pass
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return "torch"
        except ImportError:
            pass
        return "numpy"

    x_type = array_type(X)
    y_type = array_type(y)
    if x_type == y_type:
        return x_type, X, y

    _warnings.warn(
        f"Mixed backend detected: X is {x_type} but y is {y_type}. "
        "Converting both arrays to NumPy.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "numpy", _to_numpy(X), _to_numpy(y)


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
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)

    if intercepts is not None:
        intercepts = _to_numpy(intercepts).ravel()
        if intercepts.shape[0] != n_models:
            raise ValueError(
                f"intercepts length {intercepts.shape[0]} != n_models {n_models}"
            )
        if not np.all(np.isfinite(intercepts)):
            raise ValueError("intercepts must be finite")

    if sample_weight is not None:
        sw = _to_numpy(sample_weight).ravel().astype(np.float64, copy=False)
        if sw.shape[0] != X_val.shape[0]:
            raise ValueError(
                f"sample_weight length {sw.shape[0]} != n_samples {X_val.shape[0]}"
            )
        if not np.all(np.isfinite(sw)):
            raise ValueError("sample_weight must be finite")
        if np.any(sw < 0):
            raise ValueError("sample_weight must be non-negative")
        sw_sum = float(np.sum(sw))
        if sw_sum <= 0.0:
            raise ValueError("sample_weight must have a positive sum")
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
            mse[start:end] = np.sum(residuals ** 2 * sw[None, :], axis=1) / sw_sum
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
