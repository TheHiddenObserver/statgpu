"""
LogisticRegressionCV: Cross-validated Logistic regression with GPU support.
"""

from typing import Any, Dict, Optional, Tuple, Union
from collections import OrderedDict
import hashlib
import numpy as np

from statgpu._config import Device
from statgpu.linear_model._cv_base import CVEstimatorBase
from statgpu.backends import get_backend, _torch_dev
from ._logistic import LogisticRegression


# =============================================================================
# CV Cache for LogisticRegression
# =============================================================================

import threading

_LOGISTIC_CV_C_CACHE_MAXSIZE = int(64)
_LOGISTIC_CV_C_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_LOGISTIC_CV_CACHE_LOCK = threading.Lock()


def _logistic_cv_cache_get(key):
    """Get cached LogisticRegression CV results."""
    if key is None:
        return None
    with _LOGISTIC_CV_CACHE_LOCK:
        val = _LOGISTIC_CV_C_CACHE.get(key)
        if val is not None:
            _LOGISTIC_CV_C_CACHE.move_to_end(key)
        return val


def _logistic_cv_cache_put(key, value):
    """Put cached LogisticRegression CV results."""
    if key is None:
        return
    with _LOGISTIC_CV_CACHE_LOCK:
        _LOGISTIC_CV_C_CACHE[key] = value
        _LOGISTIC_CV_C_CACHE.move_to_end(key)
        while len(_LOGISTIC_CV_C_CACHE) > _LOGISTIC_CV_C_CACHE_MAXSIZE:
            _LOGISTIC_CV_C_CACHE.popitem(last=False)


from statgpu.linear_model._cv_base import hash_cv_data as _hash_logistic_data


def _make_logistic_cv_auto_cache_key(X, y, Cs, folds, fit_intercept, max_iter, tol, use_gpu, sample_weight=None):
    """Generate automatic cache key for LogisticRegression CV."""
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    h.update(str(X.dtype).encode("utf-8"))
    h.update(np.asarray(Cs, dtype=np.float64).tobytes())
    h.update(str(fit_intercept).encode("utf-8"))
    h.update(str(max_iter).encode("utf-8"))
    h.update(str(tol).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    # Hash data content to avoid cross-dataset collisions
    h.update(_hash_logistic_data(X, y, sample_weight))
    # Hash fold indices (sample evenly to keep hash fast for large folds)
    for train_idx, val_idx in folds:
        train_arr = np.asarray(train_idx, dtype=np.int64)
        val_arr = np.asarray(val_idx, dtype=np.int64)
        # Hash a representative sample: first 5, last 5, and length
        n_sample = min(5, len(train_arr))
        h.update(train_arr[:n_sample].tobytes())
        h.update(train_arr[-n_sample:].tobytes())
        h.update(np.int64(len(train_arr)).tobytes())
        n_sample_v = min(5, len(val_arr))
        h.update(val_arr[:n_sample_v].tobytes())
        h.update(val_arr[-n_sample_v:].tobytes())
        h.update(np.int64(len(val_arr)).tobytes())
    return h.hexdigest()


# =============================================================================
# K-fold helper (reuse from RidgeCV)
# =============================================================================

from statgpu.linear_model._cv_base import kfold_indices as _kfold_indices, folds_are_complete as _folds_are_complete


# =============================================================================
# C grid generation (C = 1/alpha, so we use similar approach)
# =============================================================================

def _default_logistic_c_grid(X, y, n_Cs: int = 100, C_min_ratio: float = 1e-3):
    """
    Generate default C grid for LogisticRegressionCV.

    C values are log-spaced. Larger C = weaker regularization.

    Parameters
    ----------
    X : ndarray
        Design matrix (n_samples, n_features).
    y : ndarray
        Response vector.
    n_Cs : int
        Number of C values to generate.
    C_min_ratio : float
        Minimum C as a ratio of max C.

    Returns
    -------
    Cs : ndarray
        Log-spaced C values.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

    # Estimate C_max based on data
    # For logistic regression, C_max is where coefficients become very large
    # We use a heuristic based on the gradient at zero coefficients.
    # Gradient of logistic loss at beta=0: X'(y - sigmoid(0)) = X'(y - 0.5)
    grad = X_arr.T @ (y_arr - 0.5)
    C_max = np.max(np.abs(grad)) * 2.0 / len(y_arr)

    if C_max == 0:
        C_max = 1.0

    C_min = C_max * C_min_ratio

    # Log-spaced grid
    if n_Cs <= 1:
        return np.array([C_max])

    Cs = np.logspace(
        np.log10(C_min),
        np.log10(C_max),
        num=n_Cs,
        dtype=np.float64,
    )
    return Cs


# =============================================================================
# Batch log-loss computation
# =============================================================================

def _batch_log_loss(y_val, probs_desc, sample_weight=None):
    """
    Compute log-loss for multiple probability vectors efficiently.

    Parameters
    ----------
    y_val : ndarray
        Validation labels (n_samples,).
    probs_desc : ndarray
        Predicted probabilities (n_Cs, n_samples).
    sample_weight : ndarray or None
        Sample weights.

    Returns
    -------
    log_loss : ndarray
        Log-loss for each C (n_Cs,).
    """
    n_Cs = probs_desc.shape[0]
    eps = 1e-15

    # Clip probabilities
    probs_clipped = np.clip(probs_desc, eps, 1 - eps)

    # Log-loss: -mean(y * log(p) + (1-y) * log(1-p))
    ll = -(y_val.reshape(1, -1) * np.log(probs_clipped) +
           (1 - y_val.reshape(1, -1)) * np.log(1 - probs_clipped))

    if sample_weight is not None:
        sw = np.asarray(sample_weight).reshape(1, -1)
        log_loss = np.sum(sw * ll, axis=1) / np.sum(sw)
    else:
        log_loss = np.mean(ll, axis=1)

    return log_loss


def _batch_log_loss_backend(y_val, probs_desc, backend, sample_weight=None):
    """Compute log-loss for multiple probability vectors (backend-aware).

    Delegates to numpy version when backend is numpy, otherwise uses
    backend methods for GPU arrays.
    """
    xp = getattr(backend, 'xp', np)
    eps = 1e-15
    probs_clipped = xp.clip(probs_desc, eps, 1 - eps) if hasattr(xp, 'clip') else np.clip(probs_desc, eps, 1 - eps)

    ll = -(y_val.reshape(1, -1) * xp.log(probs_clipped) +
           (1 - y_val.reshape(1, -1)) * xp.log(1 - probs_clipped))

    if sample_weight is not None:
        sw = sample_weight.reshape(1, -1)
        log_loss = xp.sum(sw * ll, axis=1) / xp.sum(sw)
    else:
        log_loss = xp.mean(ll, axis=1)

    return log_loss


# =============================================================================
# GPU batch solver for Logistic (IRLS)
# =============================================================================

def _solve_logistic_path_gpu_from_batch(X_batch, y_batch, n_train_vec, Cs, backend, fit_intercept=True, max_iter=100, tol=1e-4):
    """
    Solve logistic regression path for multiple folds using batched IRLS.

    Parameters
    ----------
    X_batch : array-like
        Batch of design matrices (n_folds, n_train_max, n_features).
    y_batch : array-like
        Batch of labels (n_folds, n_train_max).
    n_train_vec : np.ndarray
        Number of training samples for each fold.
    Cs : ndarray
        C values.
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    fit_intercept : bool
        Whether to fit intercept.
    max_iter : int
        Maximum iterations for IRLS.
    tol : float
        Convergence tolerance.

    Returns
    -------
    coefs_desc : ndarray
        Coefficients for each C and fold (n_Cs, n_folds, n_features).
    intercepts_desc : ndarray
        Intercepts for each C and fold (n_Cs, n_folds).
    """
    xp = backend.xp

    n_folds = X_batch.shape[0]
    n_Cs = len(Cs)

    # Allocate outputs
    all_coefs = []
    all_intercepts = []

    for fold_idx in range(n_folds):
        X_fold = X_batch[fold_idx][:n_train_vec[fold_idx]]
        y_fold = y_batch[fold_idx][:n_train_vec[fold_idx]]
        n_train = n_train_vec[fold_idx]

        fold_coefs = []
        fold_intercepts = []

        for C in Cs:
            # Initialize
            if fit_intercept:
                ones_col = backend.ones(n_train, dtype=X_fold.dtype)
                if _torch_dev(X_fold) is not None:
                    if ones_col.ndim == 1:
                        ones_col = ones_col.unsqueeze(1)
                    X_design = xp.cat([ones_col, X_fold], dim=1)
                else:
                    X_design = xp.column_stack([ones_col, X_fold])
                params = backend.zeros(X_design.shape[1])
            else:
                X_design = X_fold
                params = backend.zeros(X_fold.shape[1])

            # sklearn convention: reg term = 1/(2C) * ||w||^2, Hessian contribution = 1/C * I
            alpha = 1.0 / C if C > 0 else 0.0

            # IRLS
            for iteration in range(max_iter):
                params_old = backend.copy(params)

                eta = X_design @ params
                p = 1 / (1 + backend.exp(-backend.clip(eta, -500, 500)))

                W = p * (1 - p)
                W = backend.clip(W, 1e-8, 1 - 1e-8)

                z = eta + (y_fold - p) / W

                XtWX = X_design.T @ (X_design * W[:, None])

                if alpha > 0:
                    reg_diag = backend.full(XtWX.shape[0], alpha)
                    if fit_intercept:
                        reg_diag = backend.asarray(reg_diag)
                        reg_diag[0] = 0.0
                    XtWX += backend.diag(reg_diag)

                Xtz = X_design.T @ (W * z)

                try:
                    params = backend.solve(XtWX, Xtz)
                except Exception:
                    lstsq_result = backend.lstsq(XtWX, Xtz)
                    params = lstsq_result[0]

                if backend.sqrt(backend.sum((params - params_old) ** 2)) < tol:
                    break

            if fit_intercept:
                fold_coefs.append(backend.to_numpy(params[1:]))
                fold_intercepts.append(float(backend.to_numpy(params[0])))
            else:
                fold_coefs.append(backend.to_numpy(params))
                fold_intercepts.append(0.0)

        all_coefs.append(np.stack(fold_coefs, axis=0))
        all_intercepts.append(np.array(fold_intercepts))

    coefs_desc = np.stack(all_coefs, axis=1)  # (n_Cs, n_folds, n_features)
    intercepts_desc = np.stack(all_intercepts, axis=1)  # (n_Cs, n_folds)

    return coefs_desc, intercepts_desc


# =============================================================================
# Main CV selection function
# =============================================================================

def _select_logistic_c_cv(
    X,
    y,
    *,
    Cs=None,
    n_Cs: int = 100,
    C_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state: Optional[int] = None,
    sample_weight=None,
    fit_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-4,
    device: Union[str, Device] = Device.CPU,
    return_details: bool = False,
    cache_key: Optional[Tuple[Any, ...]] = None,
    gpu_cv_mixed_precision: bool = True,
):
    """
    Select C for Logistic regression via K-fold cross-validation.

    Parameters
    ----------
    X : array-like
        Design matrix (n_samples, n_features).
    y : array-like
        Binary response vector.
    Cs : array-like or None
        C values to try. If None, generates n_Cs values.
    n_Cs : int
        Number of C values (if Cs is None).
    C_min_ratio : float
        Minimum C ratio.
    cv_folds : int
        Number of CV folds.
    cv_splits : list or None
        Pre-computed CV splits.
    random_state : int or None
        Random seed for CV splits.
    sample_weight : array-like or None
        Sample weights.
    fit_intercept : bool
        Whether to fit intercept.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.
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
    C : float
        Best C value.
    details : dict (if return_details=True)
        Full CV results including C grid, loss path, etc.
    """
    device_name = str(device).lower()
    use_gpu = device_name in (Device.CUDA.value, Device.TORCH.value)
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
        backend = get_backend(backend='auto', device='cuda')
        if len(tuple(X.shape)) != 2:
            raise ValueError("X must be a 2D array")
        n_samples = int(X.shape[0])
    else:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)
        if sample_weight is not None:
            sample_weight_np = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_np.shape[0] != X_np.shape[0]:
            raise ValueError("y must have the same number of rows as X")
        n_samples = int(X_np.shape[0])

    # Generate C grid
    if Cs is None:
        if gpu_input_cupy or gpu_input_torch:
            # GPU path for C grid generation
            # Gradient of logistic loss at beta=0: X'(y - sigmoid(0)) = X'(y - 0.5)
            # Do NOT center X/y — centering is incorrect for logistic regression
            backend = get_backend(backend='auto', device='cuda')
            X_temp = backend.asarray(X)
            y_temp = backend.asarray(y)
            grad = X_temp.T @ (y_temp - 0.5)
            C_max = float(backend.max(backend.abs(grad)) * 2.0 / len(y_temp))
            if C_max == 0:
                C_max = 1.0
            C_min = C_max * C_min_ratio
            C_grid = np.logspace(np.log10(C_min), np.log10(C_max), num=n_Cs)
        else:
            C_grid = _default_logistic_c_grid(X_np, y_np, n_Cs=n_Cs, C_min_ratio=C_min_ratio)
    else:
        C_grid = np.asarray(Cs, dtype=np.float64)
        C_grid = C_grid[np.isfinite(C_grid)]
        C_grid = C_grid[C_grid > 0.0]
        if C_grid.size == 0:
            if gpu_input_cupy or gpu_input_torch:
                # GPU path for C grid generation
                backend = get_backend(backend='auto', device='cuda')
                X_temp = backend.asarray(X)
                y_temp = backend.asarray(y)
                grad = X_temp.T @ (y_temp - 0.5)
                C_max = float(backend.max(backend.abs(grad)) * 2.0 / len(y_temp))
                if C_max == 0:
                    C_max = 1.0
                C_min = C_max * C_min_ratio
                C_grid = np.logspace(np.log10(C_min), np.log10(C_max), num=n_Cs)
            else:
                C_grid = _default_logistic_c_grid(X_np, y_np, n_Cs=n_Cs, C_min_ratio=C_min_ratio)

    # Handle degenerate cases
    if int(n_samples) < 4 or int(C_grid.size) == 1 or int(cv_folds) < 2:
        C0 = float(C_grid[0])
        if not return_details:
            return C0
        return {
            "C": C0,
            "Cs": C_grid.astype(np.float64, copy=False),
            "loss_path": np.full((int(C_grid.size), 1), np.nan, dtype=np.float64),
            "mean_loss": np.full(int(C_grid.size), np.nan, dtype=np.float64),
        }

    # Generate CV folds
    if cv_splits is not None:
        from statgpu.linear_model._lasso import _normalize_cv_splits
        folds = _normalize_cv_splits(cv_splits, n_samples=int(n_samples))
    else:
        folds = _kfold_indices(n_samples=int(n_samples), n_splits=int(cv_folds), random_state=random_state)


    C_grid = C_grid.astype(np.float64, copy=False)
    n_C = int(C_grid.size)
    n_folds = int(len(folds))

    # Cache handling
    # Auto-cache disabled by default to prevent stale results across datasets.
    cache_key_eff = cache_key
    if cache_key_eff is None and False and _LOGISTIC_CV_C_CACHE_MAXSIZE > 0:
        cache_key_eff = _make_logistic_cv_auto_cache_key(
            X=X, y=y, Cs=C_grid, folds=folds,
            fit_intercept=bool(fit_intercept), max_iter=max_iter, tol=tol,
            use_gpu=bool(use_gpu), sample_weight=sample_weight,
        )

    cached_details = _logistic_cv_cache_get(cache_key_eff)
    if cached_details is not None:
        if return_details:
            return cached_details
        return float(cached_details["C"])

    # Initialize loss path
    loss_path = np.full((n_C, n_folds), np.nan, dtype=np.float64)

    # GPU path
    if use_gpu:
        try:
            # Get backend - supports both CuPy and Torch
            backend = get_backend(backend='auto', device='cuda')
            xp = backend.xp

            cv_dtype = backend.float32 if bool(gpu_cv_mixed_precision) else backend.float64

            # Convert inputs to backend arrays
            if gpu_input_cupy or gpu_input_torch:
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

            # Prepare batch data
            X_batch_list = []
            y_batch_list = []
            n_train_folds = []
            fold_eval_payload = []

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                train_idx_gpu = backend.asarray(train_idx)
                val_idx_gpu = backend.asarray(val_idx)

                X_train = X_full[train_idx_gpu]
                y_train = y_full[train_idx_gpu]
                X_val = X_full[val_idx_gpu]
                y_val = y_full[val_idx_gpu]
                sw_val = None if sw_full is None else sw_full[val_idx_gpu]

                X_batch_list.append(X_train)
                y_batch_list.append(y_train)
                n_train_folds.append(int(X_train.shape[0]))
                fold_eval_payload.append((X_val, y_val, sw_val))

            # Pad batch to same size
            n_train_max = max(n_train_folds)
            n_features = X_full.shape[1]

            X_batch = backend.zeros((n_folds, n_train_max, n_features), dtype=cv_dtype)
            y_batch = backend.zeros((n_folds, n_train_max), dtype=cv_dtype)

            for fold_idx in range(n_folds):
                n_train = n_train_folds[fold_idx]
                X_batch[fold_idx, :n_train] = X_batch_list[fold_idx]
                y_batch[fold_idx, :n_train] = y_batch_list[fold_idx]

            n_train_vec = np.asarray(n_train_folds, dtype=np.int32)

            # Solve for all Cs
            coefs_batch, intercepts_batch = _solve_logistic_path_gpu_from_batch(
                X_batch, y_batch, n_train_vec, C_grid, backend,
                fit_intercept=bool(fit_intercept), max_iter=max_iter, tol=tol
            )

            # Evaluate log-loss for each fold and C (vectorized across C)
            for fold_idx in range(n_folds):
                X_val, y_val, sw_val = fold_eval_payload[fold_idx]
                n_val = int(X_val.shape[0])

                # Batched matmul: X_val @ coefs_all.T for all C at once
                # coefs_batch shape: (n_C, n_folds, n_features)
                coefs_all = backend.asarray(coefs_batch[:, fold_idx, :])  # (n_C, n_features)
                intercepts_all = backend.asarray(intercepts_batch[:, fold_idx])  # (n_C,)

                # eta_all shape: (n_val, n_C)
                eta_all = X_val @ coefs_all.T + intercepts_all.reshape(1, -1)
                # probs_all shape: (n_C, n_val)
                probs_all = (1 / (1 + backend.exp(-backend.clip(eta_all, -500, 500)))).T

                loss_desc = _batch_log_loss_backend(y_val, probs_all, backend, sw_val)
                loss_path[:, fold_idx] = backend.to_numpy(loss_desc)

        except Exception as exc:
            raise RuntimeError(
                "GPU path failed in _select_logistic_c_cv with device='cuda'; "
                "CPU fallback is disabled for strict CUDA execution."
            ) from exc

    # CPU path
    if not use_gpu:
        if gpu_requested:
            raise RuntimeError(
                "device='cuda' requested but GPU path was not executed; "
                "CPU fallback is disabled for strict CUDA execution."
            )

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train = X_np[train_idx]
            y_train = y_np[train_idx]
            X_val = X_np[val_idx]
            y_val = y_np[val_idx]
            sw_val = None if sample_weight_np is None else sample_weight_np[val_idx]

            # Fit logistic regression for each C
            fold_losses = []
            for C in C_grid:
                model = LogisticRegression(
                    C=C,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    tol=tol,
                    device='cpu',
                    compute_inference=False,
                )
                model.fit(X_train, y_train, sample_weight=sample_weight_np[train_idx] if sample_weight_np is not None else None)

                # Predict probabilities on validation set
                probs = model.predict_proba(X_val)[:, 1]

                # Compute log-loss
                eps = 1e-15
                probs_clipped = np.clip(probs, eps, 1 - eps)
                ll = -(y_val * np.log(probs_clipped) + (1 - y_val) * np.log(1 - probs_clipped))

                if sw_val is not None:
                    fold_losses.append(np.sum(sw_val * ll) / np.sum(sw_val))
                else:
                    fold_losses.append(np.mean(ll))

            loss_path[:, fold_idx] = fold_losses

    # Compute mean loss across folds
    mean_loss = np.nanmean(loss_path, axis=1)

    # Find best C (minimum loss)
    best_idx = int(np.nanargmin(mean_loss))
    best_C = float(C_grid[best_idx])

    details = {
        "C": best_C,
        "Cs": C_grid,
        "loss_path": loss_path,
        "mean_loss": mean_loss,
    }

    _logistic_cv_cache_put(cache_key_eff, details)

    if return_details:
        return details
    return best_C


# =============================================================================
# LogisticRegressionCV Class
# =============================================================================

class LogisticRegressionCV(CVEstimatorBase):
    """
    Cross-validated Logistic regression with GPU support.

    This class implements K-fold cross-validation to select the optimal
    regularization parameter C for Logistic regression.

    Parameters
    ----------
    Cs : array-like or None
        C values to try. If None, generates n_Cs values.
    n_Cs : int
        Number of C values (if Cs is None). Default is 100.
    C_min_ratio : float
        Minimum C as a ratio of max C.
    cv : int
        Number of CV folds. Default is 5.
    fit_intercept : bool
        Whether to fit intercepts. Default is True.
    max_iter : int
        Maximum number of IRLS iterations. Default is 100.
    tol : float
        Convergence tolerance. Default is 1e-4.
    device : str or Device
        Computation device: 'cpu', 'cuda', or 'auto'.
    compute_inference : bool
        Whether to compute standard errors, z-stats, p-values and CI.
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
    C_ : float
        Selected C value.
    Cs_ : ndarray
        All C values tested.
    cv_results_ : dict
        CV results including loss_path and mean_loss.
    best_score_ : float
        Best (minimum) log-loss across CV folds.
    coef_ : ndarray
        Coefficients of the final model.
    intercept_ : float
        Intercept of the final model.
    estimator_ : LogisticRegression
        The fitted LogisticRegression with selected C.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.linear_model import LogisticRegressionCV
    >>> X = np.random.randn(1000, 20)
    >>> y = (X @ np.random.randn(20) > 0).astype(int)
    >>> model = LogisticRegressionCV(cv=5, device='cuda')
    >>> model.fit(X, y)
    >>> print(f"Selected C: {model.C_:.4f}")
    >>> print(f"Best CV score: {model.best_score_:.4f}")
    """

    def __init__(
        self,
        Cs=None,
        n_Cs: int = 100,
        C_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
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
        self.Cs = Cs
        self.n_Cs = int(n_Cs)
        self.C_min_ratio = float(C_min_ratio)
        self.cv = int(cv)
        self.cv_splits = cv_splits
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.compute_inference = bool(compute_inference)
        self.cov_type = str(cov_type)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.gpu_cv_mixed_precision = bool(gpu_cv_mixed_precision)

        self.C_ = None
        self.Cs_ = None
        self.cv_results_ = None
        self.mean_loss_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit Logistic regression with cross-validation to select C.

        Parameters
        ----------
        X : array-like
            Training data (n_samples, n_features).
        y : array-like
            Target values (binary: 0 or 1).
        sample_weight : array-like or None
            Sample weights.

        Returns
        -------
        self : LogisticRegressionCV
            Fitted estimator.
        """
        device_name = self._get_compute_device().value

        # Run CV to select C
        details = _select_logistic_c_cv(
            X,
            y,
            Cs=self.Cs,
            n_Cs=self.n_Cs,
            C_min_ratio=self.C_min_ratio,
            cv_folds=self.cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            sample_weight=sample_weight,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            device=device_name,
            gpu_cv_mixed_precision=self.gpu_cv_mixed_precision,
            return_details=True,
        )

        # Store CV results
        self.C_ = float(details["C"])
        self.Cs_ = np.asarray(details["Cs"], dtype=np.float64)
        loss_path = np.asarray(details["loss_path"], dtype=np.float64)
        mean_loss = np.asarray(details["mean_loss"], dtype=np.float64)

        self.cv_results_ = {"loss_path": loss_path}
        self.mean_loss_ = mean_loss

        if np.any(np.isfinite(mean_loss)):
            # sklearn convention: best_score_ is negative loss (higher is better)
            self.best_score_ = -float(np.nanmin(mean_loss))
        else:
            self.best_score_ = np.nan

        # Fit final model with selected C
        estimator = LogisticRegression(
            C=self.C_,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
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
        """Predict class labels using the fitted Logistic model."""
        self._check_is_fitted()
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        self._check_is_fitted()
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        """Return accuracy score."""
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
