"""
RidgeCV: Cross-validated Ridge regression with GPU support.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
from collections import OrderedDict
import hashlib
import numpy as np

from statgpu._config import Device
from statgpu.linear_model._cv_base import CVEstimatorBase
from statgpu.backends import get_backend, _torch_dev
from statgpu.backends._factory import _cupy_backend, _torch_backend
from ._ridge import Ridge


# =============================================================================
# CV Cache for Ridge
# =============================================================================

_RIDGE_CV_ALPHA_CACHE_MAXSIZE = int(64)
_RIDGE_CV_ALPHA_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()


def _ridge_cv_cache_get(key):
    """Get cached Ridge CV results."""
    if key is None:
        return None
    val = _RIDGE_CV_ALPHA_CACHE.get(key)
    if val is not None:
        _RIDGE_CV_ALPHA_CACHE.move_to_end(key)
    return val


def _ridge_cv_cache_put(key, value):
    """Put cached Ridge CV results."""
    if key is None:
        return
    _RIDGE_CV_ALPHA_CACHE[key] = value
    _RIDGE_CV_ALPHA_CACHE.move_to_end(key)
    while len(_RIDGE_CV_ALPHA_CACHE) > _RIDGE_CV_ALPHA_CACHE_MAXSIZE:
        _RIDGE_CV_ALPHA_CACHE.popitem(last=False)


def _make_ridge_cv_auto_cache_key(X, y, alphas, folds, fit_intercept, use_gpu, sample_weight=None):
    """Generate automatic cache key for Ridge CV.

    Includes a digest of X and y content (first/last rows + summary stats)
    to avoid cross-dataset cache collisions.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    h.update(str(X.dtype).encode("utf-8"))
    h.update(np.asarray(alphas, dtype=np.float64).tobytes())
    h.update(str(fit_intercept).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    # Hash data content: first row, last row, mean, std
    from statgpu.backends import _to_numpy
    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    y_np = np.asarray(_to_numpy(y), dtype=np.float64).ravel()
    n = X_np.shape[0]
    h.update(np.asarray(X_np.shape, dtype=np.int64).tobytes())
    # Sample up to 100 evenly spaced rows for robust hashing
    step = max(1, n // 100)
    idx = np.arange(0, n, step)[:100]
    h.update(X_np[idx].tobytes())
    h.update(y_np[idx].tobytes())
    h.update(np.asarray([X_np.mean(), X_np.std()], dtype=np.float64).tobytes())
    h.update(np.asarray([y_np.mean(), y_np.std()], dtype=np.float64).tobytes())
    # Hash fold indices (all elements to avoid collisions)
    for train_idx, val_idx in folds:
        h.update(train_idx.tobytes())
        h.update(val_idx.tobytes())
    if sample_weight is not None:
        sw = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
        h.update(sw[idx].tobytes())
        h.update(np.asarray([sw.mean()], dtype=np.float64).tobytes())
    return h.hexdigest()


# =============================================================================
# K-fold helper
# =============================================================================

from statgpu.linear_model._cv_base import kfold_indices as _kfold_indices, folds_are_complete as _folds_are_complete


# =============================================================================
# Alpha grid generation
# =============================================================================

def _default_ridge_alpha_grid(X, y, n_alphas: int = 100, alpha_min_ratio: float = 1e-3):
    """
    Generate default alpha grid for Ridge CV.

    Mirrors sklearn's approach: alpha values are log-spaced between
    alpha_min and alpha_max based on the data.

    Parameters
    ----------
    X : ndarray
        Design matrix (n_samples, n_features).
    y : ndarray
        Response vector.
    n_alphas : int
        Number of alpha values to generate.
    alpha_min_ratio : float
        Minimum alpha as a ratio of max alpha.

    Returns
    -------
    alphas : ndarray
        Log-spaced alpha values.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

    # Handle intercept by centering
    X_mean = np.mean(X_arr, axis=0)
    y_mean = np.mean(y_arr)
    X_centered = X_arr - X_mean
    y_centered = y_arr - y_mean

    # Compute XtX and Xty for alpha_max estimation
    XtX = X_centered.T @ X_centered
    Xty = X_centered.T @ y_centered

    # alpha_max: smallest alpha where all coefficients become zero
    # For Ridge: alpha_max = max(|Xty|) * 2 / n (approximately)
    n_samples = X_arr.shape[0]
    alpha_max = np.max(np.abs(Xty)) * 2.0 / n_samples

    if alpha_max == 0:
        alpha_max = 1.0

    alpha_min = alpha_max * alpha_min_ratio

    # Log-spaced grid
    if n_alphas <= 1:
        return np.array([alpha_max])

    alphas = np.logspace(
        np.log10(alpha_min),
        np.log10(alpha_max),
        num=n_alphas,
        dtype=np.float64,
    )
    return alphas


# =============================================================================
# Batch MSE computation
# =============================================================================

def _batch_mse_numpy(X_val, y_val, coefs_desc, intercepts_desc, sample_weight=None):
    """
    Compute MSE for multiple coefficient vectors efficiently.

    Parameters
    ----------
    X_val : ndarray
        Validation design matrix.
    y_val : ndarray
        Validation response.
    coefs_desc : ndarray
        Coefficient matrix (n_alphas, n_features).
    intercepts_desc : ndarray
        Intercept vector (n_alphas,).
    sample_weight : ndarray or None
        Sample weights.

    Returns
    -------
    mse : ndarray
        MSE for each alpha (n_alphas,).
    """
    n_alphas = coefs_desc.shape[0]
    n_samples = X_val.shape[0]

    # Ensure intercepts_desc is 1D array of shape (n_alphas,)
    intercepts_desc = np.atleast_1d(np.asarray(intercepts_desc, dtype=np.float64))

    # Predictions: (n_alphas, n_samples)
    # coefs_desc: (n_alphas, n_features), X_val: (n_samples, n_features)
    # coefs_desc @ X_val.T = (n_alphas, n_samples)
    y_pred = coefs_desc @ X_val.T + intercepts_desc[:, np.newaxis]

    # Residuals: (n_alphas, n_samples)
    residuals = y_pred - y_val.reshape(1, -1)

    if sample_weight is not None:
        sw = np.asarray(sample_weight).reshape(1, -1)
        mse = np.sum(sw * residuals ** 2, axis=1) / np.sum(sw)
    else:
        mse = np.mean(residuals ** 2, axis=1)

    return mse


# =============================================================================
# GPU batch solver for Ridge
# =============================================================================

def _solve_ridge_path_gpu_from_gram_eig(XtX_batch, Xty_batch, alphas, backend, fit_intercept=True, n_samples_vec=None):
    """
    Solve Ridge path for multiple folds using eigendecomposition (vectorized over alphas).

    This function uses eigendecomposition to solve the Ridge regression problem for all
    alphas simultaneously, avoiding repeated Cholesky decompositions.

    Mathematical formulation:
    - Given XtX = Q @ Lambda @ Q.T (eigendecomposition)
    - Ridge solution: coef(alpha) = (XtX + alpha*I)^-1 @ Xty
    - Using eigenbasis: coef(alpha) = Q @ diag(1/(lambda_i + alpha)) @ Q.T @ Xty

    Parameters
    ----------
    XtX_batch : array-like
        Batch of Gram matrices (n_folds, n_features, n_features).
    Xty_batch : array-like
        Batch of cross products (n_folds, n_features).
    alphas : ndarray
        Alpha values (n_alphas,).
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    fit_intercept : bool
        Whether intercept is fitted (not used here, kept for API compatibility).

    Returns
    -------
    coefs_desc : ndarray
        Coefficients for each alpha and fold (n_alphas, n_folds, n_features).
    """
    xp = backend.xp

    n_folds = XtX_batch.shape[0]
    n_features = XtX_batch.shape[1]
    n_alphas = alphas.shape[0]

    # Step 1: Eigendecomposition (done once per fold)
    # eigvals: (n_folds, n_features), Q: (n_folds, n_features, n_features)
    eigvals, Q = xp.linalg.eigh(XtX_batch)
    # Clamp eigenvalues to avoid division by zero for rank-deficient X'X
    eigvals = xp.maximum(eigvals, 1e-15)

    # Step 2: Project Xty into eigenbasis
    # QTXty = Q.T @ Xty_batch  -> (n_folds, n_features)
    Q_T = backend.transpose(Q, (0, 2, 1))
    QTXty = xp.matmul(Q_T, Xty_batch[:, :, None])[:, :, 0]

    # Step 3: Convert alphas to backend array and compute inverse diagonal
    # inv_diag: (n_folds, n_features, n_alphas)
    # Scale alpha by n_samples to match Ridge.fit() convention.
    alphas_arr = backend.asarray(alphas, dtype=eigvals.dtype)
    if n_samples_vec is not None:
        n_arr = backend.asarray(n_samples_vec, dtype=eigvals.dtype).reshape(-1, 1, 1)
        inv_diag = 1.0 / (eigvals[:, :, None] + alphas_arr[None, None, :] * n_arr)
    else:
        inv_diag = 1.0 / (eigvals[:, :, None] + alphas_arr[None, None, :])

    # Step 4: Scale projected Xty by inverse diagonal
    # scaled: (n_folds, n_features, n_alphas)
    scaled = QTXty[:, :, None] * inv_diag

    # Step 5: Transform back to original basis
    # coefs: (n_folds, n_features, n_alphas)
    coefs = xp.matmul(Q, scaled)

    # Step 6: Reshape to (n_alphas, n_folds, n_features)
    # Current shape: (n_folds, n_features, n_alphas)
    # Need to transpose to: (n_alphas, n_folds, n_features)
    coefs = backend.transpose(coefs, (2, 0, 1))

    # Keep on GPU for further processing (avoid unnecessary H2D transfer)
    return coefs


def _solve_ridge_path_gpu_from_gram(XtX_batch, Xty_batch, n_samples_vec, alphas, backend, fit_intercept=True):
    """
    Solve Ridge path for multiple folds using eigendecomposition (optimized).

    This function uses eigendecomposition to solve the Ridge regression problem for all
    alphas simultaneously, avoiding repeated Cholesky decompositions.

    Parameters
    ----------
    XtX_batch : array-like
        Batch of Gram matrices (n_folds, n_features, n_features).
    Xty_batch : array-like
        Batch of cross products (n_folds, n_features).
    n_samples_vec : np.ndarray
        Number of samples for each fold (not used, kept for API compatibility).
    alphas : ndarray
        Alpha values (n_alphas,).
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    fit_intercept : bool
        Whether intercept is fitted.

    Returns
    -------
    coefs_desc : ndarray
        Coefficients for each alpha and fold (n_alphas, n_folds, n_features).
    """
    # Use eigendecomposition-based solver (vectorized over alphas)
    return _solve_ridge_path_gpu_from_gram_eig(XtX_batch, Xty_batch, alphas, backend, fit_intercept, n_samples_vec=n_samples_vec)


# =============================================================================
# Main CV selection function
# =============================================================================

def _select_ridge_alpha_cv(
    X,
    y,
    *,
    alphas=None,
    n_alphas: int = 100,
    alpha_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state: Optional[int] = None,
    sample_weight=None,
    fit_intercept: bool = True,
    device: Union[str, Device] = Device.CPU,
    return_details: bool = False,
    cache_key: Optional[Tuple[Any, ...]] = None,
    gpu_cv_mixed_precision: bool = True,
):
    """
    Select alpha for Ridge regression via K-fold cross-validation.

    Parameters
    ----------
    X : array-like
        Design matrix (n_samples, n_features).
    y : array-like
        Response vector.
    alphas : array-like or None
        Alpha values to try. If None, generates n_alphas values.
    n_alphas : int
        Number of alpha values (if alphas is None).
    alpha_min_ratio : float
        Minimum alpha ratio.
    cv_folds : int
        Number of CV folds.
    cv_splits : list or None
        Pre-computed CV splits. If None, uses K-fold.
    random_state : int or None
        Random seed for CV splits.
    sample_weight : array-like or None
        Sample weights.
    fit_intercept : bool
        Whether to fit intercept.
    device : str or Device
        Device to use ('cpu' or 'cuda').
    return_details : bool
        Whether to return full CV details.
    cache_key : tuple or None
        Cache key for CV results.
    gpu_cv_mixed_precision : bool
        Whether to use mixed precision on GPU.

    Returns
    -------
    alpha : float
        Best alpha value.
    details : dict (if return_details=True)
        Full CV results including alpha grid, MSE path, etc.
    """
    device_name = str(device).lower()
    use_gpu = device_name in (Device.CUDA.value, Device.TORCH.value, "torch")
    gpu_requested = use_gpu

    gpu_input_cupy = False
    gpu_input_torch = False
    if use_gpu:
        # Check if inputs are already on GPU (CuPy or Torch)
        try:
            import cupy as cp
            gpu_input_cupy = isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
            if sample_weight is not None and not isinstance(sample_weight, cp.ndarray):
                gpu_input_cupy = False
        except Exception:
            pass

        # Also check for torch tensors
        if not gpu_input_cupy:
            try:
                import torch
                gpu_input_torch = isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
                if sample_weight is not None and not isinstance(sample_weight, torch.Tensor):
                    gpu_input_torch = False
            except Exception:
                pass

    X_np = None
    y_np = None
    sample_weight_np = None

    if gpu_input_cupy or gpu_input_torch:
        # GPU inputs - get backend for validation
        # Use torch backend for torch tensors, cupy for cupy arrays
        if gpu_input_torch:
            backend = get_backend(backend='torch', device='cuda')
        else:
            backend = get_backend(backend='cupy', device='cuda')
        if len(tuple(X.shape)) != 2:
            raise ValueError("X must be a 2D array")
        n_samples = int(X.shape[0])
        y_check = backend.asarray(y).reshape(-1)
        if int(y_check.shape[0]) != n_samples:
            raise ValueError("y must have the same number of rows as X")
        if sample_weight is not None:
            sw_check = backend.asarray(sample_weight).reshape(-1)
            if int(sw_check.shape[0]) != n_samples:
                raise ValueError("sample_weight must have the same number of rows as X")
    else:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)
        if sample_weight is not None:
            sample_weight_np = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_np.shape[0] != X_np.shape[0]:
            raise ValueError("y must have the same number of rows as X")
        if sample_weight_np is not None and sample_weight_np.shape[0] != X_np.shape[0]:
            raise ValueError("sample_weight must have the same number of rows as X")
        n_samples = int(X_np.shape[0])

    # Generate alpha grid
    if alphas is None:
        if gpu_input_cupy or (use_gpu and hasattr(X, 'device') and str(X.device) != 'cpu'):
            # GPU path for alpha grid generation
            if gpu_input_torch:
                backend = get_backend(backend='torch', device='cuda')
            else:
                backend = get_backend(backend='cupy', device='cuda')
            X_temp = backend.asarray(X)
            y_temp = backend.asarray(y)
            X_mean = backend.mean(X_temp, axis=0)
            y_mean = backend.mean(y_temp)
            X_centered = X_temp - X_mean
            y_centered = y_temp - y_mean
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            n = int(X.shape[0])
            alpha_max = float(backend.max(backend.abs(Xty)) * 2.0 / n)
            if alpha_max == 0:
                alpha_max = 1.0
            alpha_min = alpha_max * alpha_min_ratio
            alpha_grid = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=n_alphas)
            del X_temp, y_temp, X_mean, y_mean, X_centered, y_centered, XtX, Xty
        else:
            alpha_grid = _default_ridge_alpha_grid(X_np, y_np, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio)
    else:
        alpha_grid = np.asarray(alphas, dtype=np.float64)
        alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
        alpha_grid = alpha_grid[alpha_grid > 0.0]
        if alpha_grid.size == 0:
            if gpu_input_cupy or (use_gpu and hasattr(X, 'device') and str(X.device) != 'cpu'):
                # GPU path for alpha grid generation
                backend = get_backend(Device.CUDA)
                X_temp = backend.asarray(X)
                y_temp = backend.asarray(y)
                X_mean = backend.mean(X_temp, axis=0)
                y_mean = backend.mean(y_temp)
                X_centered = X_temp - X_mean
                y_centered = y_temp - y_mean
                XtX = X_centered.T @ X_centered
                Xty = X_centered.T @ y_centered
                n = int(X.shape[0])
                alpha_max = float(backend.max(backend.abs(Xty)) * 2.0 / n)
                if alpha_max == 0:
                    alpha_max = 1.0
                alpha_min = alpha_max * alpha_min_ratio
                alpha_grid = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=n_alphas)
            else:
                alpha_grid = _default_ridge_alpha_grid(X_np, y_np, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio)

    # Handle degenerate cases
    if int(n_samples) < 4 or int(alpha_grid.size) == 1 or int(cv_folds) < 2:
        alpha0 = float(alpha_grid[0])
        if not return_details:
            return alpha0
        return {
            "alpha": alpha0,
            "alphas": alpha_grid.astype(np.float64, copy=False),
            "mse_path": np.full((int(alpha_grid.size), 1), np.nan, dtype=np.float64),
            "mean_mse": np.full(int(alpha_grid.size), np.nan, dtype=np.float64),
        }

    # Generate CV folds
    if cv_splits is not None:
        folds = cv_splits
    else:
        folds = _kfold_indices(n_samples=int(n_samples), n_splits=int(cv_folds), random_state=random_state)

    folds_are_complete = _folds_are_complete(folds, n_samples=int(n_samples))

    alpha_grid = alpha_grid.astype(np.float64, copy=False)
    n_alpha = int(alpha_grid.size)
    n_folds = int(len(folds))

    # Cache handling
    # Auto-cache disabled by default to prevent stale results across datasets.
    # Only use explicit cache_key if provided by the caller.
    cache_key_eff = cache_key

    cached_details = _ridge_cv_cache_get(cache_key_eff)
    if cached_details is not None:
        if return_details:
            return cached_details
        return float(cached_details["alpha"])

    # Initialize MSE path
    mse_path = np.full((n_alpha, n_folds), np.nan, dtype=np.float64)

    # GPU path
    if use_gpu:
        try:
            # Get backend based on input data type to avoid cross-backend conversion
            # Torch input -> TorchBackend, CuPy input -> CuPyBackend
            import torch
            try:
                import cupy as cp
                cupy_available = True
            except ImportError:
                cupy_available = False

            # Detect input type and select appropriate backend
            if hasattr(X, '__module__') and 'torch' in str(type(X).__module__):
                backend = _torch_backend
            elif cupy_available and hasattr(X, '__cuda_array_interface__'):
                backend = _cupy_backend
            else:
                # Default to auto-selection for numpy input
                backend = get_backend(backend='auto', device='cuda')

            xp = backend.xp

            cv_dtype = backend.float32 if bool(gpu_cv_mixed_precision) else backend.float64

            # Convert inputs to backend arrays
            if gpu_input_cupy or (hasattr(X, 'device') and str(X.device) != 'cpu'):
                # Already on GPU (CuPy or Torch)
                X_full = backend.asarray(X, dtype=cv_dtype)
                y_full = backend.asarray(y, dtype=cv_dtype).reshape(-1)
                if sample_weight is not None:
                    sw_full = backend.asarray(sample_weight, dtype=cv_dtype).reshape(-1)
                else:
                    sw_full = None
            else:
                # Convert from numpy
                X_full = backend.asarray(X_np, dtype=cv_dtype)
                y_full = backend.asarray(y_np, dtype=cv_dtype)
                if sample_weight_np is not None:
                    sw_full = backend.asarray(sample_weight_np, dtype=cv_dtype)
                else:
                    sw_full = None

            # Precompute for fast fold statistics
            XtX_folds = []
            Xty_folds = []
            n_train_folds = []
            X_mean_folds = []
            y_mean_folds = []

            # For batched MSE evaluation (Phase 2 optimization)
            X_val_folds = []
            y_val_folds = []
            sw_val_folds = []
            n_val_folds = []

            fast_fold_stats = (sw_full is None) and bool(folds_are_complete)
            sw_train = None  # initialized per-fold in slow path; None for fast path
            if fast_fold_stats:
                n_total = int(X_full.shape[0])
                XtX_full = X_full.T @ X_full
                Xty_full = X_full.T @ y_full
                if bool(fit_intercept):
                    X_sum_full = backend.sum(X_full, axis=0)
                    y_sum_full = backend.sum(y_full)
                else:
                    X_sum_full = None
                    y_sum_full = None

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                train_idx_gpu = backend.asarray(train_idx)
                val_idx_gpu = backend.asarray(val_idx)

                X_val = X_full[val_idx_gpu]
                y_val = y_full[val_idx_gpu]
                sw_val = None if sw_full is None else sw_full[val_idx_gpu]

                # Store validation data for batched MSE
                X_val_folds.append(X_val)
                y_val_folds.append(y_val)
                sw_val_folds.append(sw_val)
                n_val_folds.append(int(val_idx_gpu.shape[0]))

                if fast_fold_stats:
                    n_val = int(val_idx_gpu.shape[0])
                    n_train = int(n_total - n_val)

                    XtX_val = X_val.T @ X_val
                    Xty_val = X_val.T @ y_val
                    XtX_raw = XtX_full - XtX_val
                    Xty_raw = Xty_full - Xty_val

                    if bool(fit_intercept):
                        X_sum_val = backend.sum(X_val, axis=0)
                        y_sum_val = backend.sum(y_val)
                        X_sum_train = X_sum_full - X_sum_val
                        y_sum_train = y_sum_full - y_sum_val

                        inv_n = backend.asarray(1.0 / float(max(1, n_train)), dtype=X_full.dtype)
                        X_mean = X_sum_train * inv_n
                        y_mean = y_sum_train * inv_n
                        XtX = XtX_raw - backend.outer(X_sum_train, X_sum_train) * inv_n
                        Xty = Xty_raw - X_sum_train * y_mean
                    else:
                        X_mean = backend.zeros((X_full.shape[1],), dtype=X_full.dtype)
                        y_mean = backend.array(0.0, dtype=X_full.dtype)
                        XtX = XtX_raw
                        Xty = Xty_raw
                else:
                    X_train = X_full[train_idx_gpu]
                    y_train = y_full[train_idx_gpu]
                    sw_train = None if sw_full is None else sw_full[train_idx_gpu]

                    if sw_train is not None:
                        # Weighted Ridge: use X'WX, X'Wy directly
                        sw_col = sw_train[:, None]
                        if bool(fit_intercept):
                            w_sum = float(backend.sum(sw_train))
                            X_wmean = backend.sum(X_train * sw_col, axis=0) / w_sum
                            y_wmean = float(backend.sum(y_train * sw_train)) / w_sum
                            XtX = (X_train * sw_col).T @ X_train - w_sum * backend.outer(X_wmean, X_wmean)
                            Xty = (X_train * sw_col).T @ y_train - w_sum * X_wmean * y_wmean
                            X_mean = X_wmean
                            y_mean = y_wmean
                        else:
                            XtX = (X_train * sw_col).T @ X_train
                            Xty = (X_train * sw_col).T @ y_train
                            X_mean = backend.zeros((X_train.shape[1],), dtype=X_train.dtype)
                            y_mean = backend.array(0.0, dtype=X_train.dtype)
                        n_train = float(sw_train.sum()) if bool(fit_intercept) else int(X_train.shape[0])
                    else:
                        if bool(fit_intercept):
                            X_mean = backend.mean(X_train, axis=0)
                            y_mean = backend.mean(y_train)
                            X_centered = X_train - X_mean
                            y_centered = y_train - y_mean
                        else:
                            X_mean = backend.zeros((X_train.shape[1],), dtype=X_train.dtype)
                            y_mean = backend.array(0.0, dtype=X_train.dtype)
                            X_centered = X_train
                            y_centered = y_train

                        XtX = X_centered.T @ X_centered
                        Xty = X_centered.T @ y_centered
                        n_train = int(X_train.shape[0])

                XtX_folds.append(XtX)
                Xty_folds.append(Xty)
                # For weighted Ridge, n_train is sum(sw) (float); for unweighted, it's the count (int)
                n_train_folds.append(float(n_train) if sw_train is not None else int(n_train))
                X_mean_folds.append(X_mean)
                y_mean_folds.append(y_mean)

            # Batch solve for all alphas (Phase 1 optimization)
            XtX_batch = backend.stack(XtX_folds, axis=0)
            Xty_batch = backend.stack(Xty_folds, axis=0)
            # Use float64 to preserve fractional sum(sw) for weighted Ridge
            n_samples_vec = np.asarray(n_train_folds, dtype=np.float64)

            coefs_batch = _solve_ridge_path_gpu_from_gram(
                XtX_batch, Xty_batch, n_samples_vec, alpha_grid, backend, fit_intercept=bool(fit_intercept)
            )

            # Batch compute intercepts (Phase 2 optimization)
            X_mean_batch = backend.stack(X_mean_folds, axis=0)  # (n_folds, n_features)
            y_mean_batch = backend.stack(y_mean_folds, axis=0)  # (n_folds,)

            intercepts_batch = _compute_intercepts_batch(
                coefs_batch, X_mean_batch, y_mean_batch, backend, fit_intercept=bool(fit_intercept)
            )  # (n_alphas, n_folds)

            # Batch compute MSE for all folds (Phase 2 optimization)
            # Pad validation sets to same size
            n_val_max = max(n_val_folds)
            n_features = int(X_full.shape[1])

            # Pre-allocate padded batches (Phase 3 optimization - memory pre-allocation)
            X_val_batch = backend.zeros((n_folds, n_val_max, n_features), dtype=cv_dtype)
            y_val_batch = backend.zeros((n_folds, n_val_max), dtype=cv_dtype)

            if sw_full is not None:
                sw_val_batch = backend.zeros((n_folds, n_val_max), dtype=cv_dtype)
            else:
                sw_val_batch = None

            # Fill padded batches
            for fold_idx in range(n_folds):
                n_val = n_val_folds[fold_idx]
                X_val_batch[fold_idx, :n_val, :] = X_val_folds[fold_idx]
                y_val_batch[fold_idx, :n_val] = y_val_folds[fold_idx]
                if sw_val_batch is not None:
                    sw_val_batch[fold_idx, :n_val] = sw_val_folds[fold_idx]

            # Batched MSE computation (fully vectorized)
            mse_path_gpu = _batch_mse_all_folds(
                X_val_batch, y_val_batch, coefs_batch, intercepts_batch, backend, sw_val_batch,
                n_val_folds=n_val_folds,
            )

            # Convert to numpy
            mse_path = backend.to_numpy(mse_path_gpu)

        except Exception as exc:
            raise RuntimeError(
                "GPU path failed in _select_ridge_alpha_cv with device='cuda'; "
                "CPU fallback is disabled for strict CUDA execution."
            ) from exc

    # CPU path
    if not use_gpu:
        if gpu_requested:
            raise RuntimeError(
                "device='cuda' requested but GPU path was not executed; "
                "CPU fallback is disabled for strict CUDA execution."
            )

        fast_fold_stats = (sample_weight_np is None) and bool(folds_are_complete)
        if fast_fold_stats:
            n_total = int(X_np.shape[0])
            XtX_full = X_np.T @ X_np
            Xty_full = X_np.T @ y_np
            if bool(fit_intercept):
                X_sum_full = np.sum(X_np, axis=0)
                y_sum_full = float(np.sum(y_np))
            else:
                X_sum_full = None
                y_sum_full = None

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_val = X_np[val_idx]
            y_val = y_np[val_idx]
            sw_val = None if sample_weight_np is None else sample_weight_np[val_idx]

            if fast_fold_stats:
                n_val = int(np.asarray(val_idx, dtype=np.int64).reshape(-1).size)
                n_train = int(n_total - n_val)

                XtX_val = X_val.T @ X_val
                Xty_val = X_val.T @ y_val
                XtX_raw = XtX_full - XtX_val
                Xty_raw = Xty_full - Xty_val

                if bool(fit_intercept):
                    X_sum_val = np.sum(X_val, axis=0)
                    y_sum_val = float(np.sum(y_val))
                    X_sum_train = X_sum_full - X_sum_val
                    y_sum_train = y_sum_full - y_sum_val

                    inv_n = 1.0 / float(max(1, n_train))
                    X_mean = X_sum_train * inv_n
                    y_mean = y_sum_train * inv_n
                    XtX = XtX_raw - np.outer(X_sum_train, X_sum_train) * inv_n
                    Xty = Xty_raw - X_sum_train * y_mean
                else:
                    X_mean = np.zeros((X_np.shape[1],), dtype=np.float64)
                    y_mean = 0.0
                    XtX = XtX_raw
                    Xty = Xty_raw
            else:
                X_train = X_np[train_idx]
                y_train = y_np[train_idx]
                sw_train = None if sample_weight_np is None else sample_weight_np[train_idx]

                if sw_train is not None:
                    # Weighted Ridge: use X'WX, X'Wy directly (matches GPU path)
                    sw_col = sw_train[:, np.newaxis]
                    if bool(fit_intercept):
                        w_sum = float(np.sum(sw_train))
                        X_wmean = np.sum(X_train * sw_col, axis=0) / w_sum
                        y_wmean = float(np.sum(y_train * sw_train)) / w_sum
                        XtX = (X_train * sw_col).T @ X_train - w_sum * np.outer(X_wmean, X_wmean)
                        Xty = (X_train * sw_col).T @ y_train - w_sum * X_wmean * y_wmean
                        X_mean = X_wmean
                        y_mean = y_wmean
                    else:
                        XtX = (X_train * sw_col).T @ X_train
                        Xty = (X_train * sw_col).T @ y_train
                        X_mean = np.zeros((X_train.shape[1],), dtype=np.float64)
                        y_mean = 0.0
                    n_train = float(np.sum(sw_train))
                else:
                    if bool(fit_intercept):
                        X_mean = np.mean(X_train, axis=0)
                        y_mean = float(np.mean(y_train))
                        X_centered = X_train - X_mean
                        y_centered = y_train - y_mean
                    else:
                        X_mean = np.zeros((X_train.shape[1],), dtype=np.float64)
                        y_mean = 0.0
                        X_centered = X_train
                        y_centered = y_train

                    XtX = X_centered.T @ X_centered
                    Xty = X_centered.T @ y_centered
                    n_train = int(X_train.shape[0])

            # Solve for all alphas: (XtX + n_eff*alpha*I)^-1 @ Xty
            # n_eff scaling matches Ridge.fit() and PGLM exact ridge.
            I = np.eye(XtX.shape[0])
            coefs_desc = []
            for alpha in alpha_grid:
                XtX_reg = XtX + alpha * float(n_train) * I
                try:
                    coef = np.linalg.solve(XtX_reg, Xty)
                except np.linalg.LinAlgError:
                    coef = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]
                coefs_desc.append(coef.flatten())
            coefs_desc = np.stack(coefs_desc, axis=0)

            # Compute intercepts
            if bool(fit_intercept):
                # X_mean: (p,), coefs_desc: (n_alphas, p)
                # X_mean @ coefs_desc.T = coefs_desc @ X_mean = (n_alphas,)
                intercepts_desc = y_mean - coefs_desc @ X_mean
            else:
                intercepts_desc = np.zeros((coefs_desc.shape[0],))

            # Compute MSE
            mse_desc = _batch_mse_numpy(X_val, y_val, coefs_desc, intercepts_desc, sw_val)
            mse_path[:, fold_idx] = mse_desc

    # Compute mean MSE across folds
    mean_mse = np.nanmean(mse_path, axis=1)

    # Find best alpha (minimum MSE)
    best_idx = int(np.nanargmin(mean_mse))
    best_alpha = float(alpha_grid[best_idx])

    details = {
        "alpha": best_alpha,
        "alphas": alpha_grid,
        "mse_path": mse_path,
        "mean_mse": mean_mse,
    }

    _ridge_cv_cache_put(cache_key_eff, details)

    if return_details:
        return details
    return best_alpha


# =============================================================================
# GPU MSE helper
# =============================================================================

def _batch_mse(X_val, y_val, coefs_desc, intercepts_desc, backend, sample_weight=None):
    """
    Compute MSE for multiple coefficient vectors.

    Parameters
    ----------
    X_val : array-like
        Validation design matrix (n_samples, n_features).
    y_val : array-like
        Validation response (n_samples,).
    coefs_desc : array-like
        Coefficient matrix (n_alphas, n_features).
    intercepts_desc : array-like
        Intercept vector (n_alphas,).
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    sample_weight : array-like or None
        Sample weights.

    Returns
    -------
    mse : array-like
        MSE for each alpha (n_alphas,).
    """
    # Ensure intercepts_desc is 1D array of shape (n_alphas,)
    intercepts_desc = backend.atleast_1d(backend.asarray(intercepts_desc, dtype=backend.float64))

    # Predictions: (n_alphas, n_samples)
    # coefs_desc: (n_alphas, n_features), X_val: (n_samples, n_features)
    y_pred = coefs_desc @ X_val.T + intercepts_desc[:, None]

    # Residuals: (n_alphas, n_samples)
    residuals = y_pred - y_val.reshape(1, -1)

    if sample_weight is not None:
        sw = sample_weight.reshape(1, -1)
        mse = backend.sum(sw * residuals ** 2, axis=1) / backend.sum(sw)
    else:
        mse = backend.mean(residuals ** 2, axis=1)

    return mse


def _batch_mse_all_folds(X_val_batch, y_val_batch, coefs_batch, intercepts_batch, backend, sample_weights_batch=None, n_val_folds=None):
    """
    Compute MSE for all folds and all alphas simultaneously (fully vectorized).

    This function batches the MSE computation across all CV folds and all alphas,
    maximizing GPU parallelism and minimizing kernel launch overhead.

    Parameters
    ----------
    X_val_batch : array-like
        Batched validation matrices (n_folds, n_val_max, n_features).
        Padded with zeros if fold sizes differ.
    y_val_batch : array-like
        Batched validation responses (n_folds, n_val_max).
        Padded with zeros if fold sizes differ.
    coefs_batch : array-like
        Coefficient matrix (n_alphas, n_folds, n_features). Same device as X_val_batch.
    intercepts_batch : array-like
        Intercept vector (n_alphas, n_folds). Same device as X_val_batch.
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    sample_weights_batch : array-like or None
        Batched sample weights (n_folds, n_val_max), or None.

    Returns
    -------
    mse : array-like
        MSE for each alpha and fold (n_alphas, n_folds). Same device as input.
    """
    xp = backend.xp
    n_folds = X_val_batch.shape[0]

    # coefs_batch and intercepts_batch are already on GPU (no conversion needed)
    # Compute predictions: (n_folds, n_val_max, n_alphas)
    # X_val_batch: (n_folds, n_val_max, n_features)
    # coefs_batch: (n_alphas, n_folds, n_features) -> transpose to (n_folds, n_features, n_alphas)
    coefs_T = backend.transpose(coefs_batch, (1, 2, 0))  # (n_folds, n_features, n_alphas)
    y_pred = xp.matmul(X_val_batch, coefs_T)  # (n_folds, n_val_max, n_alphas)

    # Add intercepts: (n_alphas, n_folds) -> (n_folds, 1, n_alphas) broadcasts
    # intercepts_batch.T: (n_folds, n_alphas) -> expand_dims to (1, n_folds, n_alphas)
    _is_torch = _torch_dev(coefs_batch) is not None
    _expand = lambda a, dim: a.unsqueeze(dim) if _is_torch else xp.expand_dims(a, axis=dim)

    intercepts_expanded = _expand(intercepts_batch.T, 1)  # (1, n_folds, n_alphas)
    y_pred = y_pred + intercepts_expanded  # broadcasts to (n_folds, n_val_max, n_alphas)

    # Residuals: (n_folds, n_val_max, n_alphas)
    y_val_expanded = _expand(y_val_batch, 2)  # (n_folds, n_val_max, 1)
    residuals = y_pred - y_val_expanded

    # Compute MSE — use per-fold n_val to exclude padded zeros
    if sample_weights_batch is not None:
        sw = _expand(sample_weights_batch, 2)  # (n_folds, n_val_max, 1)
        ssr = xp.sum(sw * residuals ** 2, axis=1)  # (n_folds, n_alphas)
        sw_sum = xp.sum(sw, axis=1)  # (n_folds,)
        # Guard against zero weight sum (avoid division by zero)
        sw_sum_safe = xp.where(sw_sum > 0, sw_sum, xp.ones_like(sw_sum))
        mse = (ssr / sw_sum_safe[:, None]).T  # (n_alphas, n_folds)
    else:
        ssr = xp.sum(residuals ** 2, axis=1)  # (n_folds, n_alphas)
        if n_val_folds is not None:
            n_val_vec = backend.asarray(n_val_folds, dtype=ssr.dtype).reshape(-1, 1)
            mse = (ssr / n_val_vec).T  # (n_alphas, n_folds)
        else:
            mse = xp.mean(residuals ** 2, axis=1).T

    return mse


def _compute_intercepts_batch(coefs_batch, X_mean_batch, y_mean_batch, backend, fit_intercept=True):
    """
    Compute intercepts for all alphas and all folds simultaneously.

    Parameters
    ----------
    coefs_batch : array-like
        Coefficient matrix (n_alphas, n_folds, n_features). Can be GPU or CPU.
    X_mean_batch : array-like
        Training set means (n_folds, n_features).
    y_mean_batch : array-like
        Training set response means (n_folds,).
    backend : BackendBase
        Backend instance.
    fit_intercept : bool
        Whether to compute intercepts.

    Returns
    -------
    intercepts : array-like
        Intercept matrix (n_alphas, n_folds). Same device as input.
    """
    xp = backend.xp

    if not fit_intercept:
        return backend.zeros((coefs_batch.shape[0], coefs_batch.shape[1]), dtype=backend.float64)

    n_alphas = coefs_batch.shape[0]
    n_folds = coefs_batch.shape[1]
    n_features = coefs_batch.shape[2]

    # Compute coefs @ X_mean for each fold
    # Reshape coefs to (n_alphas * n_folds, n_features)
    coefs_reshaped = coefs_batch.reshape((n_alphas * n_folds, n_features))

    # Tile X_mean for each alpha
    X_mean_tiled = xp.tile(X_mean_batch, (n_alphas, 1))

    # Batched dot product: sum over features
    coefs_dot_sum = xp.sum(coefs_reshaped * X_mean_tiled, axis=1)  # (n_alphas * n_folds,)
    coefs_dot_sum = coefs_dot_sum.reshape((n_alphas, n_folds))     # (n_alphas, n_folds)

    # y_mean_batch: (n_folds,) -> (1, n_folds) broadcasts to (n_alphas, n_folds)
    if _torch_dev(coefs_batch) is not None:
        y_mean_expanded = y_mean_batch.unsqueeze(0)
    else:
        y_mean_expanded = xp.expand_dims(y_mean_batch, axis=0)
    intercepts = y_mean_expanded - coefs_dot_sum

    return intercepts


# =============================================================================
# RidgeCV Class
# =============================================================================

class RidgeCV(CVEstimatorBase):
    """
    Cross-validated Ridge regression with GPU support.

    This class implements K-fold cross-validation to select the optimal
    regularization parameter alpha for Ridge regression.

    Parameters
    ----------
    alphas : array-like or None
        Alpha values to try. If None, generates n_alphas values.
    n_alphas : int
        Number of alpha values (if alphas is None). Default is 100.
    alpha_min_ratio : float
        Minimum alpha as a ratio of max alpha.
    cv : int
        Number of CV folds. Default is 5.
    fit_intercept : bool
        Whether to fit intercept. Default is True.
    device : str or Device
        Computation device: 'cpu', 'cuda', or 'auto'.
    n_jobs : int or None
        Number of parallel jobs (not yet implemented).
    compute_inference : bool
        Whether to compute standard errors, t-stats, p-values and CI.
    cov_type : str
        Covariance estimator for inference. One of:
        'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'.
    gpu_memory_cleanup : bool
        Whether to free CuPy memory pool after fitting.
    random_state : int or None
        Random seed for CV splits.
    gpu_cv_mixed_precision : bool
        Whether to use mixed precision on GPU.

    Attributes
    ----------
    alpha_ : float
        Selected alpha value.
    alphas_ : ndarray
        All alpha values tested.
    cv_results_ : dict
        CV results including mse_path and mean_mse.
    best_score_ : float
        Best (minimum) MSE across CV folds.
    coef_ : ndarray
        Coefficients of the final model.
    intercept_ : float
        Intercept of the final model.
    estimator_ : Ridge
        The fitted Ridge estimator with selected alpha.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.linear_model import RidgeCV
    >>> X = np.random.randn(1000, 20)
    >>> y = X @ np.random.randn(20) + 0.1 * np.random.randn(1000)
    >>> model = RidgeCV(cv=5, device='cuda')
    >>> model.fit(X, y)
    >>> print(f"Selected alpha: {model.alpha_:.4f}")
    >>> print(f"Best CV score: {model.best_score_:.4f}")
    """

    def __init__(
        self,
        alphas=None,
        n_alphas: int = 100,
        alpha_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
        random_state: Optional[int] = None,
        gpu_cv_mixed_precision: bool = True,
    ):
        super().__init__(
            cv=cv,
            random_state=random_state,
            device=device,
            n_jobs=n_jobs,
        )
        self.alphas = alphas
        self.n_alphas = int(n_alphas)
        self.alpha_min_ratio = float(alpha_min_ratio)
        self.cv = int(cv)
        self.cv_splits = cv_splits
        self.fit_intercept = bool(fit_intercept)
        self.compute_inference = bool(compute_inference)
        self.cov_type = str(cov_type)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.gpu_cv_mixed_precision = bool(gpu_cv_mixed_precision)

        self.alpha_ = None
        self.alphas_ = None
        self.cv_results_ = None
        self.mean_mse_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit Ridge regression with cross-validation to select alpha.

        Parameters
        ----------
        X : array-like
            Training data (n_samples, n_features).
        y : array-like
            Target values.
        sample_weight : array-like or None
            Sample weights.

        Returns
        -------
        self : RidgeCV
            Fitted estimator.
        """
        from statgpu.linear_model._cv_base import validate_cv_sample_weight
        n_samples = int(X.shape[0]) if hasattr(X, 'shape') else len(X)
        sample_weight = validate_cv_sample_weight(sample_weight, n_samples)

        device_name = self._get_compute_device().value

        # Run CV to select alpha
        details = _select_ridge_alpha_cv(
            X,
            y,
            alphas=self.alphas,
            n_alphas=self.n_alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            cv_folds=self.cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            sample_weight=sample_weight,
            fit_intercept=self.fit_intercept,
            device=device_name,
            gpu_cv_mixed_precision=self.gpu_cv_mixed_precision,
            return_details=True,
        )

        # Store CV results
        self.alpha_ = float(details["alpha"])
        self.alphas_ = np.asarray(details["alphas"], dtype=np.float64)
        mse_path = np.asarray(details["mse_path"], dtype=np.float64)
        mean_mse = np.asarray(details["mean_mse"], dtype=np.float64)

        self.cv_results_ = {"mse_path": mse_path}
        self.mean_mse_ = mean_mse

        if np.any(np.isfinite(mean_mse)):
            # sklearn convention: best_score_ is negative MSE (higher is better)
            self.best_score_ = -float(np.nanmin(mean_mse))
        else:
            self.best_score_ = np.nan

        # Fit final model with selected alpha.
        # Exact solve uses n*alpha on unnormalized X'X, matching the
        # per-sample convention (loss/n + alpha*||w||^2) used by all paths.
        # alpha_ stores the CV-selected value; pass it directly to Ridge.
        estimator = Ridge(
            alpha=self.alpha_,
            fit_intercept=self.fit_intercept,
            device=self.device,
            n_jobs=self.n_jobs,
            compute_inference=self.compute_inference,
            cov_type=self.cov_type,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
        )

        estimator.fit(X, y, sample_weight=sample_weight)

        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)

        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted Ridge model."""
        self._check_is_fitted()
        return self.estimator_.predict(X)

    def score(self, X, y):
        """Return R² score."""
        self._check_is_fitted()
        return self.estimator_.score(X, y)

    def summary(self):
        """Return summary of the fitted model."""
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted estimator available.")
        if not hasattr(self.estimator_, "summary"):
            raise RuntimeError(f"{self.estimator_.__class__.__name__} does not implement summary().")
        return self.estimator_.summary()
