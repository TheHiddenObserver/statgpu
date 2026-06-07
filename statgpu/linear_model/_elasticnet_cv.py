"""
ElasticNetCV: Cross-validated Elastic Net regression with GPU support.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import hashlib
import numpy as np

from statgpu._config import Device, cuda_available
from statgpu.linear_model._cv_base import CVEstimatorBase
from statgpu.backends import get_backend
from ._elasticnet import ElasticNet


# =============================================================================
# CV Cache
# =============================================================================

_ELASTICNET_CV_CACHE_MAXSIZE = int(64)
_ELASTICNET_CV_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()


def _elasticnet_cv_cache_get(cache_key: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    """Get cached ElasticNet CV results."""
    if cache_key is None:
        return None
    val = _ELASTICNET_CV_CACHE.get(cache_key)
    if val is not None:
        _ELASTICNET_CV_CACHE.move_to_end(cache_key)
    return val


def _elasticnet_cv_cache_put(cache_key: Optional[Tuple[Any, ...]], value: Dict[str, Any]) -> None:
    """Put cached ElasticNet CV results."""
    if cache_key is None:
        return
    _ELASTICNET_CV_CACHE[cache_key] = value
    _ELASTICNET_CV_CACHE.move_to_end(cache_key)
    while len(_ELASTICNET_CV_CACHE) > _ELASTICNET_CV_CACHE_MAXSIZE:
        _ELASTICNET_CV_CACHE.popitem(last=False)


def _make_elasticnet_cv_auto_cache_key(
    X_shape: Tuple[int, ...],
    y_shape: Tuple[int, ...],
    l1_ratios: Tuple[float, ...],
    alphas: Optional[np.ndarray],
    n_alphas: int,
    alpha_min_ratio: float,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    fit_intercept: bool,
    use_gpu: bool,
    max_iter: int,
    tol: float,
    sample_weight_shape: Optional[Tuple[int, ...]] = None,
    data_digest: Optional[bytes] = None,
) -> Tuple[Any, ...]:
    """Generate automatic cache key for ElasticNet CV."""
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(X_shape, dtype=np.int64).tobytes())
    h.update(np.asarray(y_shape, dtype=np.int64).tobytes())
    if data_digest is not None:
        h.update(data_digest)
    h.update(np.asarray(l1_ratios, dtype=np.float64).tobytes())
    if alphas is not None:
        h.update(np.asarray(alphas, dtype=np.float64).tobytes())
    h.update(str(n_alphas).encode("utf-8"))
    h.update(str(alpha_min_ratio).encode("utf-8"))
    h.update(str(fit_intercept).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    h.update(str(max_iter).encode("utf-8"))
    h.update(str(tol).encode("utf-8"))
    h.update(str(len(folds)).encode("utf-8"))
    for train_idx, test_idx in folds:
        train_idx_arr = (
            train_idx
            if isinstance(train_idx, np.ndarray) and train_idx.dtype == np.int64
            else np.asarray(train_idx, dtype=np.int64)
        )
        test_idx_arr = (
            test_idx
            if isinstance(test_idx, np.ndarray) and test_idx.dtype == np.int64
            else np.asarray(test_idx, dtype=np.int64)
        )
        h.update(train_idx_arr.tobytes())
        h.update(test_idx_arr.tobytes())
    if sample_weight_shape is not None:
        h.update(np.asarray(sample_weight_shape, dtype=np.int64).tobytes())
    return h.hexdigest()


def _hash_data(X, y) -> bytes:
    """Compute a compact hash of X and y data content.

    Samples evenly spaced rows to avoid collisions from different middle rows.
    """
    from statgpu.backends import _to_numpy
    h = hashlib.blake2b(digest_size=16)
    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    y_np = np.asarray(_to_numpy(y), dtype=np.float64).ravel()
    n = X_np.shape[0]
    h.update(np.asarray(X_np.shape, dtype=np.int64).tobytes())
    step = max(1, n // 100)
    idx = np.arange(0, n, step)[:100]
    h.update(X_np[idx].tobytes())
    h.update(y_np[idx].tobytes())
    h.update(np.asarray([X_np.mean(), X_np.std()], dtype=np.float64).tobytes())
    h.update(np.asarray([y_np.mean(), y_np.std()], dtype=np.float64).tobytes())
    return h.digest()


# =============================================================================
# K-fold helpers
# =============================================================================

from statgpu.linear_model._cv_base import kfold_indices as _kfold_indices, folds_are_complete as _folds_are_complements


# =============================================================================
# Alpha grid generation for ElasticNet
# =============================================================================

def _default_elasticnet_alpha_grid(
    X,
    y,
    l1_ratio: float = 0.5,
    n_alphas: int = 100,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    """
    Generate default alpha grid for ElasticNet.

    Parameters
    ----------
    X : array-like
        Design matrix (n_samples, n_features).
    y : array-like
        Response vector.
    l1_ratio : float
        L1 ratio (0.0 = Ridge, 1.0 = Lasso).
    n_alphas : int
        Number of alpha values.
    alpha_min_ratio : float
        Minimum alpha as a ratio of max alpha.

    Returns
    -------
    alphas : ndarray
        Log-spaced alpha values.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

    n_samples, n_features = X_arr.shape

    # Handle intercept by centering
    X_mean = np.mean(X_arr, axis=0)
    y_mean = np.mean(y_arr)
    X_centered = X_arr - X_mean
    y_centered = y_arr - y_mean

    # Compute correlation for alpha_max
    Xty = X_centered.T @ y_centered
    n = n_samples

    # For ElasticNet, alpha_max depends on l1_ratio
    # When l1_ratio > 0, use Lasso-like alpha_max
    # When l1_ratio = 0, use Ridge-like alpha_max
    if l1_ratio > 0:
        # Lasso component: alpha_max = max(|Xty|) * 2 / n
        alpha_max_lasso = np.max(np.abs(Xty)) * 2.0 / n
    else:
        alpha_max_lasso = 0.0

    # Ridge component (for stability)
    alpha_max_ridge = np.max(np.abs(Xty)) * 2.0 / n

    # Combined: use the Lasso component as the primary driver
    alpha_max = max(alpha_max_lasso, alpha_max_ridge, 1e-6)

    if alpha_max <= 0:
        alpha_max = 1.0

    # Log-spaced grid
    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _default_elasticnet_alpha_grid_backend(
    X,
    y,
    backend,
    l1_ratio: float = 0.5,
    n_alphas: int = 100,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    """
    Generate default alpha grid for ElasticNet using backend abstraction.

    Parameters
    ----------
    X : array-like
        Design matrix.
    y : array-like
        Response vector.
    backend : BackendBase
        Backend instance.
    l1_ratio : float
        L1 ratio.
    n_alphas : int
        Number of alpha values.
    alpha_min_ratio : float
        Minimum alpha ratio.

    Returns
    -------
    alphas : ndarray
        Log-spaced alpha values.
    """
    X_arr = backend.asarray(X, dtype=backend.float64)
    y_arr = backend.asarray(y, dtype=backend.float64).reshape(-1)

    n_samples = int(X_arr.shape[0])

    # Center data
    X_mean = backend.mean(X_arr, axis=0)
    y_mean = backend.mean(y_arr)
    X_centered = X_arr - X_mean
    y_centered = y_arr - y_mean

    # Compute Xty
    Xty = X_centered.T @ y_centered

    # Alpha max
    alpha_max = float(backend.max(backend.abs(Xty))) * 2.0 / n_samples

    if alpha_max <= 0:
        alpha_max = 1.0

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


# =============================================================================
# Batch MSE helper
# =============================================================================

def _batch_mse_elasticnet(
    X_val,
    y_val,
    coefs_path,
    intercepts_path,
    backend,
    sample_weight_val=None,
) -> np.ndarray:
    """
    Compute MSE for multiple coefficient vectors.

    Parameters
    ----------
    X_val : array-like
        Validation features.
    y_val : array-like
        Validation targets.
    coefs_path : array-like
        Coefficient paths (n_alphas, n_features).
    intercepts_path : array-like
        Intercept values (n_alphas,).
    backend : BackendBase
        Backend instance.
    sample_weight_val : array-like, optional
        Sample weights for validation set.

    Returns
    -------
    mse : ndarray
        MSE values (n_alphas,).
    """
    # Ensure coefs_path is backend array
    if not hasattr(coefs_path, 'reshape'):
        coefs_path = backend.asarray(coefs_path)

    # Ensure intercepts_path is backend array and reshape correctly
    if not hasattr(intercepts_path, 'reshape'):
        intercepts_path = backend.asarray(intercepts_path)
    intercepts_reshaped = intercepts_path.reshape(1, -1)

    # Compute predictions and squared errors
    preds = X_val @ coefs_path.T + intercepts_reshaped
    sq_err = (y_val.reshape(-1, 1) - preds) ** 2

    if sample_weight_val is None:
        mse = backend.mean(sq_err, axis=0)
    else:
        denom = backend.sum(sample_weight_val)
        if float(backend.to_numpy(denom)) <= 0.0:
            mse = backend.mean(sq_err, axis=0)
        else:
            mse = backend.sum(sample_weight_val.reshape(-1, 1) * sq_err, axis=0) / denom

    return backend.to_numpy(mse)


# =============================================================================
# CV main function
# =============================================================================

def _select_elasticnet_params_cv(
    X,
    y,
    *,
    l1_ratios=None,
    alphas=None,
    n_alphas: int = 100,
    alpha_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state: Optional[int] = None,
    sample_weight=None,
    fit_intercept: bool = True,
    device: Union[str, Device] = Device.CPU,
    max_iter: int = 1000,
    tol: float = 1e-4,
    return_details: bool = False,
    cache_key: Optional[Tuple[Any, ...]] = None,
):
    """
    Select alpha and l1_ratio for Elastic Net via K-fold cross-validation.

    Parameters
    ----------
    X : array-like
        Design matrix (n_samples, n_features).
    y : array-like
        Response vector.
    l1_ratios : array-like or None
        L1 ratios to try. If None, uses [0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99].
    alphas : array-like or None
        Alpha values to try. If None, generates n_alphas values.
    n_alphas : int
        Number of alpha values (if alphas is None).
    alpha_min_ratio : float
        Minimum alpha ratio.
    cv_folds : int
        Number of CV folds.
    cv_splits : list or None
        Pre-computed CV splits.
    random_state : int or None
        Random seed.
    sample_weight : array-like or None
        Sample weights.
    fit_intercept : bool
        Whether to fit intercept.
    device : str or Device
        Device to use.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    return_details : bool
        Whether to return full CV details.
    cache_key : tuple or None
        Cache key.

    Returns
    -------
    best_alpha : float
    best_l1_ratio : float
    details : dict (if return_details=True)
    """
    if isinstance(device, Device):
        device_name = device.value
    else:
        device_name = str(device).lower()
        if device_name.startswith("device."):
            enum_name = device_name.split(".", 1)[1].upper()
            if enum_name not in Device.__members__:
                valid = ", ".join(sorted(d.value for d in Device))
                raise ValueError(f"Invalid device '{device}'. Expected one of: {valid}")
            device_name = Device[enum_name].value
    if device_name == Device.AUTO.value:
        use_gpu = bool(cuda_available())
    elif device_name in (Device.CUDA.value, Device.TORCH.value):
        use_gpu = True
    else:
        use_gpu = False
    gpu_requested = use_gpu

    # Detect GPU input
    gpu_input_cupy = False
    gpu_input_torch = False
    if use_gpu:
        try:
            import cupy as cp
            gpu_input_cupy = isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
            if sample_weight is not None and not isinstance(sample_weight, cp.ndarray):
                gpu_input_cupy = False
        except Exception:
            pass
        if not gpu_input_cupy:
            try:
                import torch
                gpu_input_torch = isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
                if sample_weight is not None and not isinstance(sample_weight, torch.Tensor):
                    gpu_input_torch = False
            except Exception:
                pass

    # Validate inputs
    X_np = None
    y_np = None
    sample_weight_np = None

    if gpu_input_cupy or gpu_input_torch:
        if len(tuple(X.shape)) != 2:
            raise ValueError("X must be a 2D array")
        n_samples = int(X.shape[0])
        backend = get_backend(backend='auto', device='cuda')
        y_check = backend.asarray(y).reshape(-1)
        if int(y_check.shape[0]) != n_samples:
            raise ValueError("y must have the same number of rows as X")
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

    # Default l1_ratios
    if l1_ratios is None:
        l1_ratios_arr = np.asarray([0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99], dtype=np.float64)
    else:
        l1_ratios_arr = np.asarray(l1_ratios, dtype=np.float64)
        l1_ratios_arr = l1_ratios_arr[(l1_ratios_arr >= 0.0) & (l1_ratios_arr <= 1.0)]

    if l1_ratios_arr.size == 0:
        l1_ratios_arr = np.asarray([0.5], dtype=np.float64)

    n_l1_ratios = int(l1_ratios_arr.size)

    # Generate alpha grids for each l1_ratio
    alpha_grids = {}
    for l1r in l1_ratios_arr:
        if alphas is None:
            if gpu_input_cupy or gpu_input_torch:
                backend = get_backend(backend='torch' if gpu_input_torch else 'cupy', device='cuda')
                alpha_grids[l1r] = _default_elasticnet_alpha_grid_backend(
                    X, y, backend, l1_ratio=l1r, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio
                )
            else:
                alpha_grids[l1r] = _default_elasticnet_alpha_grid(
                    X_np, y_np, l1_ratio=l1r, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio
                )
        else:
            alpha_grid = np.asarray(alphas, dtype=np.float64)
            alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
            alpha_grid = alpha_grid[alpha_grid > 0.0]
            if alpha_grid.size == 0:
                if gpu_input_cupy or gpu_input_torch:
                    backend = get_backend(backend='torch' if gpu_input_torch else 'cupy', device='cuda')
                    alpha_grids[l1r] = _default_elasticnet_alpha_grid_backend(
                        X, y, backend, l1_ratio=l1r, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio
                    )
                else:
                    alpha_grids[l1r] = _default_elasticnet_alpha_grid(
                        X_np, y_np, l1_ratio=l1r, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio
                    )
            else:
                alpha_grids[l1r] = alpha_grid

    # Handle degenerate cases
    if int(n_samples) < 4 or int(cv_folds) < 2:
        # Use first l1_ratio and first alpha
        l1r0 = float(l1_ratios_arr[0])
        alpha0 = float(alpha_grids[l1r0][0])
        if not return_details:
            return alpha0, l1r0
        return {
            "alpha": alpha0,
            "l1_ratio": l1r0,
            "alphas": alpha_grids[l1r0].astype(np.float64),
            "l1_ratios": l1_ratios_arr.astype(np.float64),
            "mse_path": np.full((int(n_l1_ratios), int(alpha_grids[l1r0].size), 1), np.nan, dtype=np.float64),
            "mean_mse": np.full((int(n_l1_ratios), int(alpha_grids[l1r0].size)), np.nan, dtype=np.float64),
        }

    # Generate CV folds
    if cv_splits is not None:
        folds = cv_splits
    else:
        folds = _kfold_indices(n_samples=int(n_samples), n_splits=int(cv_folds), random_state=random_state)

    folds_are_complements_flag = _folds_are_complements(folds, n_samples=int(n_samples))

    n_folds = int(len(folds))

    # Cache handling
    cache_key_eff = cache_key
    if cache_key_eff is None and _ELASTICNET_CV_CACHE_MAXSIZE > 0:
        cache_key_eff = _make_elasticnet_cv_auto_cache_key(
            X_shape=X_np.shape if X_np is not None else tuple(X.shape),
            y_shape=y_np.shape if y_np is not None else tuple(y.shape),
            l1_ratios=tuple(l1_ratios_arr.tolist()),
            alphas=alphas,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            folds=folds,
            fit_intercept=bool(fit_intercept),
            use_gpu=use_gpu,
            max_iter=max_iter,
            tol=tol,
            sample_weight_shape=sample_weight_np.shape if sample_weight_np is not None else None,
            data_digest=_hash_data(X_np if X_np is not None else X, y_np if y_np is not None else y),
        )

    cached_result = _elasticnet_cv_cache_get(cache_key_eff)
    if cached_result is not None:
        if return_details:
            return cached_result["alpha"], cached_result["l1_ratio"], cached_result
        return cached_result["alpha"], cached_result["l1_ratio"]

    # Initialize MSE storage
    # mse_path: (n_l1_ratios, n_alphas, n_folds)
    max_n_alphas = max(len(ag) for ag in alpha_grids.values())
    mse_path = np.full((n_l1_ratios, max_n_alphas, n_folds), np.nan, dtype=np.float64)

    # Get backend
    if gpu_input_torch:
        backend = get_backend(backend='torch', device='cuda')
    elif gpu_input_cupy:
        backend = get_backend(backend='cupy', device='cuda')
    else:
        backend = get_backend(backend='auto', device='cuda' if use_gpu else 'cpu')

    xp = backend.xp

    # Check if we should use warm-start path optimization
    # Warm-start works when: CPU backend, no sample_weight, fit_intercept handled by centering
    use_warm_start = (
        backend.name == 'numpy'
        and not use_gpu
        and sample_weight_np is None
    )

    # CV loop
    for l1_idx, l1_ratio in enumerate(l1_ratios_arr):
        alpha_grid = alpha_grids[l1_ratio]
        n_alphas_this = len(alpha_grid)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_idx_arr = backend.asarray(train_idx)
            val_idx_arr = backend.asarray(val_idx)

            # Split data
            if gpu_input_cupy or gpu_input_torch:
                X_train_raw = X[train_idx_arr]
                y_train_raw = y[train_idx_arr]
                X_val = X[val_idx_arr]
                y_val = y[val_idx_arr]
                if sample_weight is not None:
                    sw_train = sample_weight[train_idx_arr]
                    sw_val = sample_weight[val_idx_arr]
                else:
                    sw_train = None
                    sw_val = None
                X_train = X_train_raw
                y_train = y_train_raw
            else:
                X_train_np = X_np[train_idx]
                y_train_np = y_np[train_idx]
                X_val = backend.asarray(X_np[val_idx])
                y_val = backend.asarray(y_np[val_idx])
                if sample_weight_np is not None:
                    sw_train = backend.asarray(sample_weight_np[train_idx])
                    sw_val = backend.asarray(sample_weight_np[val_idx])
                else:
                    sw_train = None
                    sw_val = None
                X_train = X_train_np
                y_train = y_train_np

            # For CPU warm-start path: precompute per-fold data to avoid redundant work
            if use_warm_start:
                # The alpha grid should be sorted descending for warm-start to work well
                # (largest alpha first -> sparsest solution -> warm-start to smaller alpha)
                alpha_grid_sorted = np.sort(alpha_grid)[::-1]
                sort_indices = np.argsort(alpha_grid)[::-1]
                inv_sort = np.argsort(sort_indices)

                # Sort alpha_grid for warm-start path
                alpha_grid_ws = alpha_grid_sorted

                # Center data for this fold (only when fit_intercept=True)
                if fit_intercept:
                    X_mean_fold = np.mean(X_train_np, axis=0)
                    y_mean_fold = np.mean(y_train_np)
                    Xc = X_train_np - X_mean_fold
                    yc = y_train_np - y_mean_fold
                else:
                    Xc = X_train_np
                    yc = y_train_np
                    X_mean_fold = np.zeros(X_train_np.shape[1])
                    y_mean_fold = 0.0

                # Precompute XtX, Xty for this fold
                XtX_fold = Xc.T @ Xc
                Xty_fold = Xc.T @ yc

                # Precompute Lipschitz constant
                eig_max = np.linalg.eigvalsh(XtX_fold)[-1]
                L_fold = float(eig_max / len(train_idx))

                # Fit alphas with warm-start (descending order)
                prev_coef = None
                for alpha_idx_ws, alpha in enumerate(alpha_grid_ws):
                    orig_idx = inv_sort[alpha_idx_ws]

                    # Create model with known L to avoid recomputation
                    model = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        tol=tol,
                        fit_intercept=fit_intercept,
                        device='cpu',
                        lipschitz_L=L_fold,
                    )

                    model.fit(X_train_np, y_train_np, initial_coef=prev_coef)

                    # Store result
                    mse_val = _batch_mse_elasticnet(
                        X_val, y_val,
                        model.coef_.reshape(1, -1),
                        np.array([model.intercept_]),
                        backend,
                        None,
                    )
                    mse_path[l1_idx, orig_idx, fold_idx] = float(mse_val[0])
                    prev_coef = model.coef_.copy()
            else:
                # Original approach for GPU or with sample weights
                for alpha_idx, alpha in enumerate(alpha_grid):
                    # Convert backend to device string that ElasticNet understands
                    if backend.name == 'numpy':
                        enet_device = 'cpu'
                    elif backend.name == 'cupy':
                        enet_device = 'cuda'
                    elif backend.name == 'torch':
                        enet_device = 'torch'
                    else:
                        enet_device = 'cpu'

                    model = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        tol=tol,
                        fit_intercept=fit_intercept,
                        device=enet_device,
                    )
                    model.fit(X_train, y_train, sample_weight=sw_train)

                    # Compute validation MSE
                    mse_val = _batch_mse_elasticnet(
                        X_val, y_val,
                        model.coef_.reshape(1, -1),
                        np.array([model.intercept_]),
                        backend,
                        sw_val,
                    )

                    mse_path[l1_idx, alpha_idx, fold_idx] = float(mse_val[0])

    # Compute mean and std MSE across folds
    mean_mse = np.nanmean(mse_path, axis=2)  # (n_l1_ratios, n_alphas)
    std_mse = np.nanstd(mse_path, axis=2)

    # Find best (l1_ratio, alpha) combination
    best_flat_idx = np.nanargmin(mean_mse)
    best_l1_idx = best_flat_idx // max_n_alphas
    best_alpha_idx = best_flat_idx % max_n_alphas

    best_l1_ratio = float(l1_ratios_arr[best_l1_idx])
    best_alpha_grid = alpha_grids[l1_ratios_arr[best_l1_idx]]
    best_alpha = float(best_alpha_grid[best_alpha_idx])
    best_mse = float(mean_mse[best_l1_idx, best_alpha_idx])

    # Prepare details
    details = {
        "alpha": best_alpha,
        "l1_ratio": best_l1_ratio,
        "alphas": alpha_grids,
        "l1_ratios": l1_ratios_arr.astype(np.float64),
        "mse_path": mse_path.astype(np.float64),
        "mean_mse": mean_mse.astype(np.float64),
        "std_mse": std_mse.astype(np.float64),
        "best_mse": best_mse,
        "n_folds": n_folds,
    }

    # Cache result
    if _ELASTICNET_CV_CACHE_MAXSIZE > 0:
        _elasticnet_cv_cache_put(cache_key_eff, details)

    if return_details:
        return best_alpha, best_l1_ratio, details

    return best_alpha, best_l1_ratio


# =============================================================================
# ElasticNetCV Class
# =============================================================================

class ElasticNetCV(CVEstimatorBase):
    """
    Cross-validated Elastic Net regression with GPU support.

    Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization:

        minimize (1/(2n)) * ||y - Xw||²₂ + α * l1_ratio * ||w||₁ + 0.5 * α * (1 - l1_ratio) * ||w||²₂

    This class uses K-fold cross-validation to select the optimal alpha and l1_ratio.

    Parameters
    ----------
    l1_ratio : float or array-like, default=0.5
        L1 regularization ratio. 0.0 = Ridge, 1.0 = Lasso.
        If array-like, CV is performed over all values.
    alphas : array-like or None
        Alpha values to try. If None, generates n_alphas values.
    n_alphas : int, default=100
        Number of alpha values (if alphas is None).
    alpha_min_ratio : float, default=1e-3
        Minimum alpha as a ratio of max alpha.
    cv : int, default=5
        Number of CV folds.
    fit_intercept : bool, default=True
        Whether to fit intercept.
    max_iter : int, default=1000
        Maximum iterations for solver.
    tol : float, default=1e-4
        Convergence tolerance.
    device : str or Device, default=Device.AUTO
        Computation device: 'cpu', 'cuda', or 'auto'.
    compute_inference : bool, default=False
        Whether to compute inference statistics.
    random_state : int or None
        Random seed for CV splits.
    n_jobs : int or None
        Number of parallel jobs (not yet implemented).

    Attributes
    ----------
    alpha_ : float
        Selected alpha value.
    l1_ratio_ : float
        Selected l1_ratio value.
    coef_ : ndarray
        Coefficients of the final model.
    intercept_ : float
        Intercept of the final model.
    cv_results_ : dict
        CV results including mse_path and mean_mse.
    best_score_ : float
        Best (minimum) MSE across CV folds.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.linear_model import ElasticNetCV
    >>> X = np.random.randn(1000, 50)
    >>> y = X @ np.random.randn(50) + 0.1 * np.random.randn(1000)
    >>> model = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, device='cuda')
    >>> model.fit(X, y)
    >>> print(f"Selected alpha: {model.alpha_:.4f}")
    >>> print(f"Selected l1_ratio: {model.l1_ratio_:.4f}")
    """

    def __init__(
        self,
        l1_ratio=0.5,
        *,
        alphas=None,
        n_alphas: int = 100,
        alpha_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            cv=cv,
            random_state=random_state,
            device=device,
            n_jobs=n_jobs,
        )
        self.l1_ratio = l1_ratio
        self.alphas = alphas
        self.n_alphas = int(n_alphas)
        self.alpha_min_ratio = float(alpha_min_ratio)
        self.cv = int(cv)
        self.cv_splits = cv_splits
        self.fit_intercept = bool(fit_intercept)
        self.compute_inference = bool(compute_inference)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Output attributes
        self.alpha_ = None
        self.l1_ratio_ = None
        self.coef_ = None
        self.intercept_ = None
        self.cv_results_ = None
        self.best_score_ = None
        self.n_iter_ = None
        self.estimator_ = None

    def _fit_cv(self, X, y, sample_weight=None):
        """
        Fit Elastic Net with K-fold cross-validation.

        Parameters
        ----------
        X : array-like
            Design matrix.
        y : array-like
            Response vector.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self
        """
        compute_device = self._get_compute_device()

        # Normalize l1_ratio to list
        if isinstance(self.l1_ratio, (list, tuple, np.ndarray)):
            l1_ratios = np.asarray(self.l1_ratio, dtype=np.float64)
        else:
            l1_ratios = np.asarray([self.l1_ratio], dtype=np.float64)

        # Perform CV
        best_alpha, best_l1_ratio, details = _select_elasticnet_params_cv(
            X, y,
            l1_ratios=l1_ratios,
            alphas=self.alphas,
            n_alphas=self.n_alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            cv_folds=self.cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            sample_weight=sample_weight,
            fit_intercept=self.fit_intercept,
            device=compute_device,
            max_iter=self.max_iter,
            tol=self.tol,
            return_details=True,
        )

        # Store CV results
        self.alpha_ = best_alpha
        self.l1_ratio_ = best_l1_ratio
        self.cv_results_ = {
            "mse_path": details["mse_path"],
            "mean_mse": details["mean_mse"],
            "std_mse": details["std_mse"],
            "alphas": details["alphas"],
            "l1_ratios": details["l1_ratios"],
            "best_alpha": self.alpha_,
            "best_l1_ratio": self.l1_ratio_,
        }
        self.best_score_ = details["best_mse"]

        # Fit final model on full data with best parameters
        final_model = ElasticNet(
            alpha=self.alpha_,
            l1_ratio=self.l1_ratio_,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            device=self.device,
        )
        final_model.fit(X, y, sample_weight=sample_weight)

        self.coef_ = final_model.coef_.copy()
        self.intercept_ = final_model.intercept_
        self.n_iter_ = final_model.n_iter_
        self.estimator_ = final_model

        return self

    def fit(self, X, y, sample_weight=None):
        """
        Fit Elastic Net model with cross-validation.

        Parameters
        ----------
        X : array-like
            Design matrix (n_samples, n_features).
        y : array-like
            Response vector (n_samples,).
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self
        """
        return self._fit_cv(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """
        Predict using Elastic Net model.

        Parameters
        ----------
        X : array-like
            Test features.

        Returns
        -------
        y_pred : ndarray
            Predicted values.
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Return R² score.

        Parameters
        ----------
        X : array-like
            Test features.
        y : array-like
            True values.

        Returns
        -------
        score : float
            R² score.
        """
        y_pred = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

        return 1.0 - ss_res / ss_tot
