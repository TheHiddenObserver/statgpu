"""
LassoCV: Cross-validated Lasso regression with GPU support.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import hashlib
import numpy as np

from statgpu._config import Device
from statgpu._cv_base import CVEstimatorBase
from statgpu.backends import get_backend
from ._lasso import Lasso


# =============================================================================
# CV Cache
# =============================================================================

_LASSO_CV_ALPHA_CACHE_MAXSIZE = int(64)
_LASSO_CV_ALPHA_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()


def _lasso_cv_cache_get(cache_key: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    """Get cached Lasso CV results."""
    if cache_key is None:
        return None
    val = _LASSO_CV_ALPHA_CACHE.get(cache_key)
    if val is not None:
        _LASSO_CV_ALPHA_CACHE.move_to_end(cache_key)
    return val


def _lasso_cv_cache_put(cache_key: Optional[Tuple[Any, ...]], value: Dict[str, Any]) -> None:
    """Put cached Lasso CV results."""
    if cache_key is None:
        return
    _LASSO_CV_ALPHA_CACHE[cache_key] = value
    _LASSO_CV_ALPHA_CACHE.move_to_end(cache_key)
    while len(_LASSO_CV_ALPHA_CACHE) > _LASSO_CV_ALPHA_CACHE_MAXSIZE:
        _LASSO_CV_ALPHA_CACHE.popitem(last=False)


def _make_lasso_cv_auto_cache_key(
    X_shape: Tuple[int, ...],
    y_shape: Tuple[int, ...],
    alphas: Optional[np.ndarray],
    n_alphas: int,
    alpha_min_ratio: float,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    fit_intercept: bool,
    use_gpu: bool,
    max_iter: int,
    tol: float,
    sample_weight_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[Any, ...]:
    """Generate automatic cache key for Lasso CV."""
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(X_shape, dtype=np.int64).tobytes())
    h.update(np.asarray(y_shape, dtype=np.int64).tobytes())
    if alphas is not None:
        h.update(np.asarray(alphas, dtype=np.float64).tobytes())
    h.update(str(n_alphas).encode("utf-8"))
    h.update(str(alpha_min_ratio).encode("utf-8"))
    h.update(str(fit_intercept).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    h.update(str(max_iter).encode("utf-8"))
    h.update(str(tol).encode("utf-8"))
    if sample_weight_shape is not None:
        h.update(np.asarray(sample_weight_shape, dtype=np.int64).tobytes())
    return h.hexdigest()


# =============================================================================
# K-fold helpers
# =============================================================================

def _kfold_indices(n_samples: int, n_splits: int, random_state: Optional[int] = None):
    """Generate K-fold train/test indices."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int64)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        current = stop
    return folds


def _folds_are_complements(folds, n_samples: int) -> bool:
    """Check if folds are complementary."""
    test_indices = np.concatenate([f[1] for f in folds])
    if len(test_indices) != n_samples:
        return False
    return np.array_equal(np.sort(test_indices), np.arange(n_samples))


# =============================================================================
# Alpha grid generation
# =============================================================================

def _default_lasso_alpha_grid_backend(X, y, backend, n_alphas: int = 12, alpha_min_ratio: float = 1e-3) -> np.ndarray:
    """Generate default alpha grid for Lasso using backend abstraction."""
    X_arr = backend.asarray(X, dtype=backend.float64)
    y_arr = backend.asarray(y, dtype=backend.float64).reshape(-1)

    n_samples = int(X_arr.shape[0])
    corr = backend.abs(X_arr.T @ y_arr) / float(max(1, n_samples))
    # Use shape to check size - works for both numpy and torch
    corr_size = int(corr.shape[0]) if hasattr(corr, 'shape') else len(corr)
    alpha_max = float(backend.to_numpy(backend.max(corr))) if corr_size > 0 else 1.0

    if n_samples > 1:
        y_std = backend.sqrt(backend.mean((y_arr - backend.mean(y_arr)) ** 2))
        sigma_hat = float(backend.to_numpy(y_std))
    else:
        sigma_hat = 0.0

    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(X_arr.shape[1]))) / max(1, n_samples))
    alpha_max = max(alpha_max, float(sigma_hat * penalty_scale), 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _default_lasso_alpha_grid(X, y, n_alphas: int = 12, alpha_min_ratio: float = 1e-3) -> np.ndarray:
    """Generate default alpha grid for Lasso (CPU)."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

    n_samples = int(X_arr.shape[0])
    corr = np.abs(X_arr.T @ y_arr) / float(max(1, n_samples))
    alpha_max = float(np.max(corr)) if int(corr.size) > 0 else 1.0

    if n_samples > 1:
        sigma_hat = float(np.std(y_arr, ddof=1))
    else:
        sigma_hat = 0.0

    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(X_arr.shape[1]))) / max(1, n_samples))
    alpha_max = max(alpha_max, float(sigma_hat * penalty_scale), 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


# =============================================================================
# Batch MSE helper
# =============================================================================

def _batch_mse(X_val, y_val, coefs_path, intercepts_path, backend, sample_weight_val, return_backend_array=False):
    """Compute MSE for multiple coefficient vectors.

    Parameters
    ----------
    return_backend_array : bool
        If True, return result as backend array (for Torch GPU backend).
        If False, return as numpy array (default behavior).
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

    if return_backend_array:
        return mse
    return backend.to_numpy(mse)


# =============================================================================
# LassoCV Class
# =============================================================================

class LassoCV(CVEstimatorBase):
    """
    Cross-validated Lasso regression with GPU support.

    This class implements K-fold cross-validation to select the optimal
    regularization parameter alpha for Lasso regression.

    Parameters
    ----------
    alphas : array-like or None
        Alpha values to try. If None, generates n_alphas values.
    n_alphas : int
        Number of alpha values (if alphas is None). Default is 12.
    alpha_min_ratio : float
        Minimum alpha as a ratio of max alpha.
    cv : int
        Number of CV folds. Default is 5.
    fit_intercept : bool
        Whether to fit intercept. Default is False.
    device : str or Device
        Computation device: 'cpu', 'cuda', or 'auto'.
    max_iter : int
        Maximum iterations for Lasso solver. Default is 3000.
    tol : float
        Convergence tolerance. Default is 1e-4.
    compute_inference : bool
        Whether to compute standard errors, t-stats, p-values and CI.
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
    estimator_ : Lasso
        The fitted Lasso estimator with selected alpha.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.linear_model import LassoCV
    >>> X = np.random.randn(1000, 20)
    >>> y = X @ np.random.randn(20) + 0.1 * np.random.randn(1000)
    >>> model = LassoCV(cv=5, device='cuda')
    >>> model.fit(X, y)
    >>> print(f"Selected alpha: {model.alpha_:.4f}")
    >>> print(f"Best CV score: {model.best_score_:.4f}")
    """

    def __init__(
        self,
        alphas=None,
        n_alphas: int = 12,
        alpha_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        fit_intercept: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        max_iter: int = 3000,
        tol: float = 1e-4,
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
        self.max_iter = int(max_iter)
        self.tol = float(tol)
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

    def _fit_cv(self, X, y, sample_weight=None):
        """
        Fit Lasso regression with K-fold cross-validation.

        This is the internal method that performs the actual CV fitting.
        """
        device_name = self._get_compute_device().value
        use_gpu = device_name == Device.CUDA.value

        # Detect input type
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

        # Convert to numpy for validation if needed
        X_np = None
        y_np = None
        sample_weight_np = None
        if gpu_input_cupy or gpu_input_torch:
            n_samples = int(X.shape[0])
        else:
            X_np = np.asarray(X, dtype=np.float64)
            y_np = np.asarray(y, dtype=np.float64).reshape(-1)
            if sample_weight is not None:
                sample_weight_np = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            n_samples = int(X_np.shape[0])

        # Generate alpha grid
        if self.alphas is None:
            if gpu_input_cupy or gpu_input_torch:
                # Get backend based on input type
                if gpu_input_torch:
                    backend = get_backend(backend='torch', device='cuda')
                else:
                    backend = get_backend(backend='cupy', device='cuda')
                alpha_grid = _default_lasso_alpha_grid_backend(
                    X, y, backend, n_alphas=self.n_alphas, alpha_min_ratio=self.alpha_min_ratio
                )
            else:
                alpha_grid = _default_lasso_alpha_grid(
                    X_np, y_np, n_alphas=self.n_alphas, alpha_min_ratio=self.alpha_min_ratio
                )
        else:
            alpha_grid = np.asarray(self.alphas, dtype=np.float64)
            alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
            alpha_grid = alpha_grid[alpha_grid > 0.0]
            if alpha_grid.size == 0:
                if gpu_input_cupy or gpu_input_torch:
                    # Get backend based on input type
                    if gpu_input_torch:
                        backend = get_backend(backend='torch', device='cuda')
                    else:
                        backend = get_backend(backend='cupy', device='cuda')
                    alpha_grid = _default_lasso_alpha_grid_backend(
                        X, y, backend, n_alphas=self.n_alphas, alpha_min_ratio=self.alpha_min_ratio
                    )
                else:
                    alpha_grid = _default_lasso_alpha_grid(
                        X_np, y_np, n_alphas=self.n_alphas, alpha_min_ratio=self.alpha_min_ratio
                    )

        # Handle degenerate cases
        if int(n_samples) < 4 or int(alpha_grid.size) == 1 or int(self.cv) < 2:
            alpha0 = float(alpha_grid[0])
            return {
                "alpha": alpha0,
                "alphas": alpha_grid.astype(np.float64, copy=False),
                "mse_path": np.full((int(alpha_grid.size), 1), np.nan, dtype=np.float64),
                "mean_mse": np.full(int(alpha_grid.size), np.nan, dtype=np.float64),
            }

        # Generate CV folds
        if self.cv_splits is not None:
            folds = self.cv_splits
        else:
            folds = _kfold_indices(n_samples=int(n_samples), n_splits=int(self.cv), random_state=self.random_state)

        folds_are_complements = _folds_are_complements(folds, n_samples=int(n_samples))
        alpha_grid = alpha_grid.astype(np.float64, copy=False)
        n_alpha = int(alpha_grid.size)
        n_folds = int(len(folds))

        # Cache handling
        cache_key = _make_lasso_cv_auto_cache_key(
            X_shape=(n_samples, X.shape[1] if len(X.shape) > 1 else 1),
            y_shape=(n_samples,),
            alphas=self.alphas,
            n_alphas=self.n_alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            folds=folds,
            fit_intercept=self.fit_intercept,
            use_gpu=use_gpu,
            max_iter=self.max_iter,
            tol=self.tol,
            sample_weight_shape=sample_weight.shape if sample_weight is not None else None,
        )

        cached_details = _lasso_cv_cache_get(cache_key)
        if cached_details is not None:
            return cached_details

        # Evaluate alpha path
        alpha_order_desc = np.argsort(-alpha_grid)
        alpha_desc = alpha_grid[alpha_order_desc]
        mse_path = np.full((n_alpha, n_folds), np.nan, dtype=np.float64)

        # GPU path
        if use_gpu:
            try:
                # Get backend based on input type - prefer Torch backend for Torch tensors
                if gpu_input_torch:
                    backend = get_backend(backend='torch', device='cuda')
                elif gpu_input_cupy:
                    backend = get_backend(backend='cupy', device='cuda')
                else:
                    backend = get_backend(backend='auto', device='cuda')
                xp = backend.xp
                cv_dtype = backend.float32 if bool(self.gpu_cv_mixed_precision) else backend.float64

                # Convert inputs
                if gpu_input_cupy or gpu_input_torch:
                    X_full = backend.asarray(X, dtype=cv_dtype)
                    y_full = backend.asarray(y, dtype=cv_dtype).reshape(-1)
                    sw_full = None if sample_weight is None else backend.asarray(sample_weight, dtype=cv_dtype).reshape(-1)
                else:
                    X_full = backend.asarray(X_np, dtype=cv_dtype)
                    y_full = backend.asarray(y_np, dtype=cv_dtype)
                    sw_full = None if sample_weight_np is None else backend.asarray(sample_weight_np, dtype=cv_dtype)

                # Precompute fold statistics
                XtX_folds = []
                Xty_folds = []
                n_train_folds = []
                X_mean_folds = []
                y_mean_folds = []
                fold_eval_payload = []

                fast_fold_stats = (sw_full is None) and bool(folds_are_complements)
                if fast_fold_stats:
                    n_total = int(X_full.shape[0])
                    XtX_full = X_full.T @ X_full
                    Xty_full = X_full.T @ y_full
                    if bool(self.fit_intercept):
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

                    if fast_fold_stats:
                        n_val = int(val_idx_gpu.shape[0])
                        n_train = int(n_total - n_val)
                        XtX_val = X_val.T @ X_val
                        Xty_val = X_val.T @ y_val
                        XtX_raw = XtX_full - XtX_val
                        Xty_raw = Xty_full - Xty_val

                        if bool(self.fit_intercept):
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
                            sqrt_sw = backend.sqrt(sw_train)
                            X_train = X_train * sqrt_sw[:, backend.newaxis]
                            y_train = y_train * sqrt_sw

                        if bool(self.fit_intercept):
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
                    n_train_folds.append(int(n_train))
                    X_mean_folds.append(X_mean)
                    y_mean_folds.append(y_mean)
                    fold_eval_payload.append((X_val, y_val, sw_val))

                # Solve using Lasso's GPU solver
                # Import both CuPy and Torch solvers
                from ._lasso import (
                    _solve_lasso_path_gpu_fista_multi_fold_from_gram,
                    _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch,
                )

                XtX_batch = backend.stack(XtX_folds, axis=0)
                Xty_batch = backend.stack(Xty_folds, axis=0)

                # Handle Torch backend by using native Torch solver
                if hasattr(xp, '__name__') and 'torch' in xp.__name__.lower():
                    import torch
                    n_samples_vec_torch = torch.tensor(np.asarray(n_train_folds, dtype=np.int32), device=XtX_batch.device, dtype=XtX_batch.dtype)

                    coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
                        XtX_batch, Xty_batch, n_samples_vec=n_samples_vec_torch,
                        alphas_desc=alpha_desc, max_iter=int(self.max_iter), tol=float(self.tol),
                        stopping="coef_delta", lipschitz_L=None, check_every=8,
                    )

                    # Keep everything on GPU for Torch backend
                    mse_path_gpu = torch.zeros((n_alpha, n_folds), dtype=XtX_batch.dtype, device=XtX_batch.device)

                    for fold_idx in range(n_folds):
                        coefs_gpu = coefs_batch_desc[fold_idx]  # Already torch tensor
                        if bool(self.fit_intercept):
                            y_mean_gpu = y_mean_folds[fold_idx]  # Already torch tensor
                            X_mean_gpu = X_mean_folds[fold_idx]  # Already torch tensor
                            intercepts_gpu = y_mean_gpu - X_mean_gpu @ coefs_gpu.T
                        else:
                            intercepts_gpu = torch.zeros((coefs_gpu.shape[0],), dtype=XtX_batch.dtype, device=XtX_batch.device)

                        X_val, y_val, sw_val = fold_eval_payload[fold_idx]
                        # Use return_backend_array=True to keep result on GPU
                        mse_gpu = _batch_mse(X_val, y_val, coefs_gpu, intercepts_gpu, backend, sw_val, return_backend_array=True)
                        mse_path_gpu[:, fold_idx] = mse_gpu

                    # Convert to numpy only at the end for cache storage
                    mse_path[alpha_order_desc, :] = mse_path_gpu.cpu().numpy()
                else:
                    # CuPy backend - use solver directly
                    import cupy as cp
                    n_samples_vec_cp = cp.asarray(np.asarray(n_train_folds, dtype=np.int32))

                    coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram(
                        XtX_batch, Xty_batch, n_samples_vec=n_samples_vec_cp,
                        alphas_desc=alpha_desc, max_iter=int(self.max_iter), tol=float(self.tol),
                        stopping="coef_delta", lipschitz_L=None, check_every=8,
                    )

                    for fold_idx in range(n_folds):
                        coefs_desc = coefs_batch_desc[fold_idx]
                        if bool(self.fit_intercept):
                            intercepts_desc = y_mean_folds[fold_idx] - X_mean_folds[fold_idx] @ coefs_desc.T
                        else:
                            intercepts_desc = backend.zeros((coefs_desc.shape[0],), dtype=coefs_desc.dtype)

                        X_val, y_val, sw_val = fold_eval_payload[fold_idx]
                        mse_desc = _batch_mse(X_val, y_val, coefs_desc, intercepts_desc, backend, sw_val)
                        mse_path[alpha_order_desc, fold_idx] = mse_desc

            except Exception as exc:
                raise RuntimeError(
                    f"GPU path failed in LassoCV._fit_cv: {exc}"
                ) from exc

        # CPU path
        else:
            from ._lasso import _fit_lasso_single_alpha_fast

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train = X_np[train_idx]
                y_train = y_np[train_idx]
                X_val = X_np[val_idx]
                y_val = y_np[val_idx]
                sw_train = None if sample_weight_np is None else sample_weight_np[train_idx]
                sw_val = None if sample_weight_np is None else sample_weight_np[val_idx]

                fold_mse = []
                for alpha in alpha_desc:
                    result = _fit_lasso_single_alpha_fast(
                        X_train, y_train, alpha=alpha, fit_intercept=self.fit_intercept,
                        max_iter=self.max_iter, tol=self.tol, stopping="coef_delta",
                        device='cpu', cpu_solver="coordinate_descent",
                        cd_kkt_check_every=1, sample_weight=sw_train,
                    )
                    coef = result['coef']
                    intercept = result['intercept'] if self.fit_intercept else 0.0

                    preds = X_val @ coef + intercept
                    if sw_val is None:
                        mse = np.mean((y_val - preds) ** 2)
                    else:
                        mse = np.sum(sw_val * (y_val - preds) ** 2) / np.sum(sw_val)
                    fold_mse.append(mse)

                mse_path[alpha_order_desc, fold_idx] = fold_mse

        # Compute mean MSE and find best alpha
        mean_mse = np.nanmean(mse_path, axis=1)
        best_idx = int(np.nanargmin(mean_mse))
        best_alpha = float(alpha_grid[best_idx])

        details = {
            "alpha": best_alpha,
            "alphas": alpha_grid,
            "mse_path": mse_path,
            "mean_mse": mean_mse,
        }

        _lasso_cv_cache_put(cache_key, details)
        return details

    def fit(self, X, y, sample_weight=None):
        """
        Fit Lasso regression with cross-validation to select alpha.

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
        self : LassoCV
            Fitted estimator.
        """
        details = self._fit_cv(X, y, sample_weight=sample_weight)

        # Store CV results
        self.alpha_ = float(details["alpha"])
        self.alphas_ = np.asarray(details["alphas"], dtype=np.float64)
        mse_path = np.asarray(details["mse_path"], dtype=np.float64)
        mean_mse = np.asarray(details["mean_mse"], dtype=np.float64)

        self.cv_results_ = {"mse_path": mse_path}
        self.mean_mse_ = mean_mse
        self.best_score_ = float(np.nanmin(mean_mse)) if np.any(np.isfinite(mean_mse)) else np.nan

        # Fit final model with selected alpha
        estimator = Lasso(
            alpha=self.alpha_,
            fit_intercept=self.fit_intercept,
            device=self.device,
            n_jobs=self.n_jobs,
            compute_inference=self.compute_inference,
            inference_method="debiased" if self.compute_inference else "cpu_ols_inference",
        )
        estimator.fit(X, y, sample_weight=sample_weight)

        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)

        # Copy inference attributes if available
        _bse_exists = hasattr(estimator, '_bse') and estimator._bse is not None
        _pvalues_exists = hasattr(estimator, '_pvalues') and estimator._pvalues is not None
        if _bse_exists:
            self.coef_std_ = np.asarray(estimator._bse)
        if _pvalues_exists:
            self.coef_pvalues_ = np.asarray(estimator._pvalues)
        if hasattr(estimator, '_tvalues') and estimator._tvalues is not None:
            self.coef_zscores_ = np.asarray(estimator._tvalues)
        if hasattr(estimator, '_conf_int') and estimator._conf_int is not None:
            self.coef_conf_int_ = np.asarray(estimator._conf_int)

        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted Lasso model."""
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
