"""
Lasso regression with full statistical inference and GPU support.
"""

from collections import OrderedDict
import hashlib
import threading
from typing import Any, Dict, Optional, Tuple, Union
import os
import warnings
import numpy as np
from scipy import stats
from scipy.stats import norm as _norm_dist

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.linear_model._cv_base import CVEstimatorBase, kfold_indices as _kfold_indices
from statgpu.backends import get_backend
from statgpu.inference._distributions_backend import (
    norm,
    t,
)


_NUMBA_CD_DISABLED = str(os.getenv("STATGPU_DISABLE_NUMBA_CD", "0")).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_LASSO_CV_ALPHA_CACHE_MAXSIZE = int(os.getenv("STATGPU_LASSO_CV_CACHE_SIZE", "64"))
_LASSO_CV_ALPHA_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_LASSO_DEBIASED_M_CACHE_MAXSIZE = int(os.getenv("STATGPU_LASSO_DEBIASED_M_CACHE_SIZE", "16"))
_LASSO_DEBIASED_M_CACHE: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
_LASSO_DEBIASED_M_GPU_HASH_ROW_CHUNK = 1024
_cache_lock = threading.Lock()


# ============================================================================
# CuPy Fused Kernels for Lasso - Now implemented as Lasso class methods
# See Lasso._get_cupy_fused_kernels() for details.
# ============================================================================


def _debiased_m_cache_get(key):
    with _cache_lock:
        val = _LASSO_DEBIASED_M_CACHE.get(key)
        if val is not None:
            _LASSO_DEBIASED_M_CACHE.move_to_end(key)
        return val


def _debiased_m_cache_put(key, value):
    with _cache_lock:
        _LASSO_DEBIASED_M_CACHE[key] = value
        _LASSO_DEBIASED_M_CACHE.move_to_end(key)
        while len(_LASSO_DEBIASED_M_CACHE) > _LASSO_DEBIASED_M_CACHE_MAXSIZE:
            _LASSO_DEBIASED_M_CACHE.popitem(last=False)


def _debiased_m_key_from_numpy_design(
    X: np.ndarray,
    *,
    n: int,
    p: int,
    lam_nw: float,
    tol: float,
):
    X_cache = np.asarray(X)
    if not X_cache.flags["C_CONTIGUOUS"]:
        X_cache = np.ascontiguousarray(X_cache)
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray([int(n), int(p)], dtype=np.int64).tobytes())
    h.update(str(X_cache.dtype).encode("utf-8"))
    h.update(np.asarray([float(lam_nw), float(tol)], dtype=np.float64).tobytes())
    h.update(X_cache.view(np.uint8).tobytes())
    return h.hexdigest()


def _debiased_m_key_from_sample(
    *,
    n: int,
    p: int,
    dtype_name: str,
    sample_block: np.ndarray,
    lam_nw: float,
    tol: float,
):
    """Generate cache key for debiased M matrix from a sample block of X.

    This is used for Torch backend where we don't want to hash the entire matrix.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray([int(n), int(p)], dtype=np.int64).tobytes())
    h.update(dtype_name.encode("utf-8"))
    h.update(np.asarray([float(lam_nw), float(tol)], dtype=np.float64).tobytes())
    if not sample_block.flags["C_CONTIGUOUS"]:
        sample_block = np.ascontiguousarray(sample_block)
    h.update(sample_block.view(np.uint8).tobytes())
    return h.hexdigest()



def _lasso_alpha_heuristic(y_centered: np.ndarray, n_features: int) -> float:
    n_samples = int(y_centered.shape[0])
    if n_samples > 1:
        sigma_hat = float(np.std(y_centered, ddof=1))
    else:
        sigma_hat = float(np.std(y_centered))
    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(n_features))) / max(1, n_samples))
    return float(sigma_hat * penalty_scale)


def _default_lasso_alpha_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    n_samples = int(X.shape[0])
    corr = np.abs(X.T @ y) / float(max(1, n_samples))
    alpha_max = float(np.max(corr)) if corr.size else 1.0
    alpha_max = max(alpha_max, _lasso_alpha_heuristic(y, n_features=int(X.shape[1])))
    alpha_max = max(alpha_max, 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _default_lasso_alpha_grid_backend(
    X,
    y,
    backend,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    """Generate default alpha grid for Lasso using backend abstraction."""
    X_arr = backend.asarray(X, dtype=backend.float64)
    y_arr = backend.asarray(y, dtype=backend.float64).reshape(-1)

    n_samples = int(X_arr.shape[0])
    corr = backend.abs(X_arr.T @ y_arr) / float(max(1, n_samples))
    # Use shape to check size - works for both numpy and torch
    corr_size = int(corr.shape[0]) if hasattr(corr, 'shape') else len(corr)
    alpha_max = float(backend.to_numpy(backend.max(corr))) if corr_size > 0 else 1.0

    if n_samples > 1:
        # Use ddof=1 (sample std) to match numpy _lasso_alpha_heuristic
        y_var = backend.sum((y_arr - backend.mean(y_arr)) ** 2) / (n_samples - 1)
        sigma_hat = float(backend.to_numpy(backend.sqrt(y_var)))
    else:
        sigma_hat = 0.0

    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(X_arr.shape[1]))) / max(1, n_samples))
    alpha_max = max(alpha_max, float(sigma_hat * penalty_scale), 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _default_lasso_alpha_grid_cupy(
    X,
    y,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    import cupy as cp

    X_cp = cp.asarray(X, dtype=cp.float64)
    y_cp = cp.asarray(y, dtype=cp.float64).reshape(-1)

    n_samples = int(X_cp.shape[0])
    corr = cp.abs(X_cp.T @ y_cp) / float(max(1, n_samples))
    alpha_max = float(cp.max(corr).item()) if int(corr.size) > 0 else 1.0

    if n_samples > 1:
        sigma_hat = float(cp.std(y_cp, ddof=1).item())
    else:
        sigma_hat = float(cp.std(y_cp).item())

    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(X_cp.shape[1]))) / max(1, n_samples))
    alpha_max = max(alpha_max, float(sigma_hat * penalty_scale), 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _normalize_cv_splits(cv_splits, n_samples: int):
    if cv_splits is None:
        return None

    n = int(n_samples)
    folds = []

    for split in cv_splits:
        if not isinstance(split, (tuple, list)) or len(split) != 2:
            raise ValueError("Each cv_splits entry must be a (train_idx, val_idx) pair")

        train_idx = np.asarray(split[0], dtype=np.int64).reshape(-1)
        val_idx = np.asarray(split[1], dtype=np.int64).reshape(-1)

        if train_idx.size == 0 or val_idx.size == 0:
            continue

        if (
            bool(np.any(train_idx < 0))
            or bool(np.any(train_idx >= n))
            or bool(np.any(val_idx < 0))
            or bool(np.any(val_idx >= n))
        ):
            raise ValueError("cv_splits indices are out of range")

        folds.append((train_idx, val_idx))

    if len(folds) == 0:
        raise ValueError("cv_splits must contain at least one non-empty split")

    return folds


def _folds_are_complements(folds, n_samples: int) -> bool:
    """Return True when each fold uses train as the exact complement of validation."""
    n = int(n_samples)
    for train_idx, val_idx in folds:
        train_arr = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        val_arr = np.asarray(val_idx, dtype=np.int64).reshape(-1)

        if int(train_arr.size + val_arr.size) != n:
            return False

        mask = np.zeros((n,), dtype=np.int8)
        mask[train_arr] = 1
        if bool(np.any(mask[val_arr] != 0)):
            return False
        mask[val_arr] = 1
        if bool(np.any(mask == 0)):
            return False

    return True


def _array_identity_token(x: Any) -> Tuple[Any, ...]:
    """Content-based hash token for array cache keys.

    Uses sampled rows (via blake2b digest) to keep hashing fast for large
    arrays while avoiding false cache hits from memory pointer reuse.
    """
    if x is None:
        return ("none",)

    import hashlib

    def _hash_bytes(data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    def _sample_and_hash(arr_np, n_sample=100):
        """Hash a representative sample of rows for large arrays."""
        n = arr_np.shape[0]
        if n <= n_sample:
            sample = arr_np
        else:
            idx = np.linspace(0, n - 1, n_sample, dtype=int)
            sample = arr_np[idx]
        return _hash_bytes(np.ascontiguousarray(sample).tobytes())

    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            # Sample on GPU first, then transfer only sampled rows
            n = x.shape[0]
            if n <= 100:
                arr_np = cp.asnumpy(x).astype(np.float64)
            else:
                idx = cp.linspace(0, n - 1, 100, dtype=cp.int64)
                arr_np = cp.asnumpy(x[idx]).astype(np.float64)
            h = _hash_bytes(np.ascontiguousarray(arr_np).tobytes())
            return ("cupy", h, tuple(int(v) for v in x.shape), str(x.dtype))
    except Exception:
        pass

    # Check for Torch tensors
    try:
        import torch

        if isinstance(x, torch.Tensor):
            # Sample on GPU first, then transfer only sampled rows
            n = x.shape[0]
            if n <= 100:
                arr_np = x.detach().cpu().numpy().astype(np.float64)
            else:
                idx = torch.linspace(0, n - 1, 100, dtype=torch.long, device=x.device)
                arr_np = x[idx].detach().cpu().numpy().astype(np.float64)
            h = _hash_bytes(np.ascontiguousarray(arr_np).tobytes())
            return ("torch", h, tuple(int(v) for v in x.shape), str(x.dtype))
    except Exception:
        pass

    arr = np.asarray(x, dtype=np.float64)
    h = _sample_and_hash(arr)
    return ("numpy", h, tuple(int(v) for v in arr.shape), str(arr.dtype))


def _alphas_signature(alphas: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(alphas, dtype=np.float64).reshape(-1))
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _folds_signature(folds) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for train_idx, val_idx in folds:
        train_arr = np.ascontiguousarray(np.asarray(train_idx, dtype=np.int64).reshape(-1))
        val_arr = np.ascontiguousarray(np.asarray(val_idx, dtype=np.int64).reshape(-1))
        hasher.update(train_arr.tobytes())
        hasher.update(b"|")
        hasher.update(val_arr.tobytes())
        hasher.update(b";")
    return hasher.hexdigest()


def _make_lasso_cv_auto_cache_key(
    *,
    X,
    y,
    sample_weight,
    alpha_grid: np.ndarray,
    folds,
    fit_intercept: bool,
    use_gpu: bool,
    max_iter: int,
    tol: float,
    cpu_solver: str,
    cv_method: str,
    cd_kkt_check_every: Optional[int],
    gpu_cv_mixed_precision: bool,
) -> Tuple[Any, ...]:
    return (
        "lasso_cv_auto_v1",
        _array_identity_token(X),
        _array_identity_token(y),
        _array_identity_token(sample_weight),
        _alphas_signature(alpha_grid),
        _folds_signature(folds),
        bool(fit_intercept),
        bool(use_gpu),
        int(max_iter),
        float(tol),
        str(cpu_solver).lower(),
        str(cv_method).lower(),
        None if cd_kkt_check_every is None else int(cd_kkt_check_every),
        bool(gpu_cv_mixed_precision),
    )


def _clone_lasso_cv_cache_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alpha": float(payload["alpha"]),
        "alphas": np.asarray(payload["alphas"], dtype=np.float64).copy(),
        "mse_path": np.asarray(payload["mse_path"], dtype=np.float64).copy(),
        "mean_mse": np.asarray(payload["mean_mse"], dtype=np.float64).copy(),
    }


def _lasso_cv_cache_get(cache_key: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    if cache_key is None or _LASSO_CV_ALPHA_CACHE_MAXSIZE <= 0:
        return None

    with _cache_lock:
        cached = _LASSO_CV_ALPHA_CACHE.get(cache_key)
        if cached is None:
            return None
        _LASSO_CV_ALPHA_CACHE.move_to_end(cache_key)
        return _clone_lasso_cv_cache_payload(cached)


def _lasso_cv_cache_put(cache_key: Optional[Tuple[Any, ...]], payload: Dict[str, Any]) -> None:
    if cache_key is None or _LASSO_CV_ALPHA_CACHE_MAXSIZE <= 0:
        return

    with _cache_lock:
        _LASSO_CV_ALPHA_CACHE[cache_key] = _clone_lasso_cv_cache_payload(payload)
        _LASSO_CV_ALPHA_CACHE.move_to_end(cache_key)
        while len(_LASSO_CV_ALPHA_CACHE) > int(_LASSO_CV_ALPHA_CACHE_MAXSIZE):
            _LASSO_CV_ALPHA_CACHE.popitem(last=False)


def _adaptive_gpu_check_every(
    *,
    base_check_every: int,
    iteration: int,
    max_iter: int,
    active_ratio: float,
) -> int:
    """Adaptive cadence for expensive global convergence checks on GPU."""
    base = max(1, int(base_check_every))
    ratio = float(max(0.0, min(1.0, active_ratio)))

    if ratio >= 0.75:
        interval = max(base, 16)
    elif ratio >= 0.40:
        interval = max(base, 12)
    elif ratio >= 0.15:
        interval = max(4, base)
    else:
        interval = max(2, base // 2)

    progress = float(iteration + 1) / float(max(1, int(max_iter)))
    if progress >= 0.90:
        interval = min(interval, 2)
    elif progress >= 0.75:
        interval = min(interval, 4)

    return max(1, int(interval))


def _soft_threshold_numpy(x: np.ndarray, gamma: float) -> np.ndarray:
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    return np.sign(x) * np.maximum(np.abs(x) - gamma_arr, 0.0)


def _soft_threshold_scalar(x: float, gamma: float) -> float:
    ax = abs(float(x))
    g = float(gamma)
    if ax <= g:
        return 0.0
    return float(np.sign(x) * (ax - g))


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _soft_threshold_scalar_numba(x: float, gamma: float) -> float:
        ax = abs(x)
        if ax <= gamma:
            return 0.0
        if x >= 0.0:
            return ax - gamma
        return -(ax - gamma)


    @njit(cache=True)
    def _solve_lasso_path_cpu_cd_numba_impl(
        XtX: np.ndarray,
        Xty: np.ndarray,
        n_samples: int,
        alphas_desc: np.ndarray,
        max_iter: int,
        tol: float,
        stopping_is_kkt: bool,
        cd_kkt_check_every: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_features = XtX.shape[0]
        n_alphas = alphas_desc.shape[0]

        coefs_path = np.zeros((n_alphas, n_features), dtype=np.float64)
        n_iters = np.zeros((n_alphas,), dtype=np.int32)

        coef = np.zeros((n_features,), dtype=np.float64)
        grad = -Xty.copy()

        X_sq_norms = np.empty((n_features,), dtype=np.float64)
        for j in range(n_features):
            X_sq_norms[j] = XtX[j, j]

        n_samp = float(max(1, n_samples))
        alpha_scaled_desc = np.empty((n_alphas,), dtype=np.float64)
        for idx in range(n_alphas):
            alpha_scaled_desc[idx] = alphas_desc[idx] * n_samp

        active_mask = np.zeros((n_features,), dtype=np.bool_)
        check_every = max(1, int(cd_kkt_check_every))

        for alpha_idx in range(n_alphas):
            alpha = float(alphas_desc[alpha_idx])
            alpha_scaled = float(alpha_scaled_desc[alpha_idx])
            if alpha_idx > 0:
                prev_alpha_scaled = float(alpha_scaled_desc[alpha_idx - 1])
            else:
                prev_alpha_scaled = alpha_scaled

            strong_thresh = 2.0 * alpha_scaled - prev_alpha_scaled
            if strong_thresh < 0.0:
                strong_thresh = 0.0

            any_active = False
            max_abs_xty = -1.0
            max_abs_xty_idx = 0
            for j in range(n_features):
                abs_xty = abs(Xty[j])
                if abs_xty >= strong_thresh:
                    active_mask[j] = True
                    any_active = True
                if abs_xty > max_abs_xty:
                    max_abs_xty = abs_xty
                    max_abs_xty_idx = j

            if not any_active:
                active_mask[max_abs_xty_idx] = True

            converged = False

            for iteration in range(int(max_iter)):
                coef_delta_l1 = 0.0

                for j in range(n_features):
                    if not active_mask[j]:
                        continue

                    denom = float(X_sq_norms[j])
                    old_val = float(coef[j])

                    if denom > 1e-10:
                        rho_j = -float(grad[j]) + denom * old_val
                        new_val = _soft_threshold_scalar_numba(rho_j, alpha_scaled) / denom
                    else:
                        new_val = 0.0

                    delta = new_val - old_val
                    if delta != 0.0:
                        coef[j] = new_val
                        coef_delta_l1 += abs(delta)
                        for row_idx in range(n_features):
                            grad[row_idx] += XtX[row_idx, j] * delta

                should_kkt_scan = (
                    ((iteration + 1) % check_every == 0)
                    or (coef_delta_l1 < float(tol))
                    or (iteration + 1 == int(max_iter))
                )

                violation = 0.0
                has_inactive_violation = False

                if should_kkt_scan:
                    for j in range(n_features):
                        v = abs(grad[j] / n_samp) - alpha
                        if v < 0.0:
                            v = 0.0
                        if v > violation:
                            violation = v
                        if v > float(tol) and (not active_mask[j]):
                            active_mask[j] = True
                            has_inactive_violation = True

                if stopping_is_kkt:
                    if should_kkt_scan and violation < float(tol):
                        n_iters[alpha_idx] = int(iteration) + 1
                        converged = True
                        break
                else:
                    if coef_delta_l1 < float(tol) and (not has_inactive_violation):
                        n_iters[alpha_idx] = int(iteration) + 1
                        converged = True
                        break

            if not converged:
                n_iters[alpha_idx] = int(max_iter)

            for j in range(n_features):
                coefs_path[alpha_idx, j] = coef[j]
                if abs(coef[j]) > 0.0:
                    active_mask[j] = True

        return coefs_path, n_iters


def _solve_lasso_path_cpu_cd_numba(
    XtX: np.ndarray,
    Xty: np.ndarray,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    cd_kkt_check_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    XtX_c = np.ascontiguousarray(XtX, dtype=np.float64)
    Xty_c = np.ascontiguousarray(Xty, dtype=np.float64)
    alphas_c = np.ascontiguousarray(np.asarray(alphas_desc, dtype=np.float64))
    stopping_is_kkt = str(stopping).lower() == "kkt"
    return _solve_lasso_path_cpu_cd_numba_impl(
        XtX_c,
        Xty_c,
        int(n_samples),
        alphas_c,
        int(max_iter),
        float(tol),
        bool(stopping_is_kkt),
        int(cd_kkt_check_every),
    )


def _normalize_lassocv_method(method: str) -> str:
    """Normalize CV optimization profile name."""
    key = str(method).strip().lower()
    alias_map = {
        "default": "standard",
        "classic": "standard",
        "glmnet_cv": "glmnet",
        "glmnet.cv": "glmnet",
    }
    key = alias_map.get(key, key)
    if key not in ("standard", "glmnet"):
        raise ValueError("method must be one of: 'standard', 'glmnet'")
    return key


def _normalize_cd_kkt_check_every(cd_kkt_check_every: Optional[int]) -> Optional[int]:
    """Validate optional coordinate-descent global KKT scan cadence."""
    if cd_kkt_check_every is None:
        return None
    value = int(cd_kkt_check_every)
    if value <= 0:
        raise ValueError("cd_kkt_check_every must be a positive integer or None")
    return value


def _solve_lasso_path_cpu_fista_batched_from_gram(
    XtX: np.ndarray,
    Xty: np.ndarray,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve descending-alpha Lasso path with a batched CPU FISTA update."""
    n_features = int(XtX.shape[0])
    n_alphas = int(alphas_desc.shape[0])

    coefs = np.zeros((n_features, n_alphas), dtype=np.float64)
    yk = coefs.copy()
    tk = np.ones((n_alphas,), dtype=np.float64)
    n_iters = np.zeros((n_alphas,), dtype=np.int32)

    if lipschitz_L is not None:
        L = float(lipschitz_L)
    else:
        try:
            eigvals = np.linalg.eigvalsh(XtX)
            L = float(eigvals[-1] / float(max(1, n_samples)))
        except Exception:
            row_sum_bound = float(np.max(np.sum(np.abs(XtX), axis=1)) / float(max(1, n_samples)))
            L = max(row_sum_bound, 1e-12)

    if L <= 0.0:
        return coefs.T, n_iters

    n_samp = float(max(1, n_samples))
    step = 1.0 / L
    alphas_desc = np.asarray(alphas_desc, dtype=np.float64)
    thresholds = alphas_desc * step
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active = np.arange(n_alphas, dtype=np.int64)

    for iteration in range(int(max_iter)):
        if active.size == 0:
            break

        y_active = yk[:, active]
        coef_old = coefs[:, active]

        grad = (XtX @ y_active - Xty.reshape(-1, 1)) / n_samp
        thresh = thresholds[active].reshape(1, -1)
        coef_new = _soft_threshold_numpy(y_active - step * grad, thresh)

        t_old = tk[active]
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_new = coef_new + beta.reshape(1, -1) * (coef_new - coef_old)

        coefs[:, active] = coef_new
        yk[:, active] = y_new
        tk[active] = t_new

        should_check = ((iteration + 1) % check_every == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (XtX @ coef_new - Xty.reshape(-1, 1)) / n_samp
            viol = np.max(
                np.maximum(
                    np.abs(grad_sse) - alphas_desc[active].reshape(1, -1),
                    0.0,
                ),
                axis=0,
            )
            converged_local = viol < float(tol)
        else:
            delta = np.sum(np.abs(coef_new - coef_old), axis=0)
            converged_local = delta < float(tol)

        if not np.any(converged_local):
            continue

        done = active[converged_local]
        n_iters[done] = int(iteration) + 1
        yk[:, done] = coefs[:, done]
        active = active[~converged_local]

    if active.size > 0:
        n_iters[active] = int(max_iter)

    return coefs.T, n_iters


def _solve_lasso_path_gpu_fista_batched_from_gram(
    XtX,
    Xty,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve descending-alpha Lasso path with a batched GPU FISTA update."""
    import cupy as cp

    n_features = int(XtX.shape[0])
    n_alphas = int(alphas_desc.shape[0])

    coefs = cp.zeros((n_features, n_alphas), dtype=XtX.dtype)
    yk = coefs.copy()
    tk = cp.ones((n_alphas,), dtype=XtX.dtype)
    n_iters_gpu = cp.zeros((n_alphas,), dtype=cp.int32)

    if lipschitz_L is not None:
        L = cp.array(float(lipschitz_L), dtype=XtX.dtype)
    else:
        try:
            eigvals = cp.linalg.eigvalsh(XtX)
            L = eigvals[-1] / float(max(1, n_samples))
        except Exception:
            row_sum_bound = cp.max(cp.sum(cp.abs(XtX), axis=1)) / float(max(1, n_samples))
            L = cp.maximum(row_sum_bound, cp.asarray(1e-12, dtype=XtX.dtype))

    L_scalar = float(cp.asnumpy(L))
    if L_scalar <= 0.0:
        return coefs.T, np.zeros((n_alphas,), dtype=np.int32)

    n_samp = float(max(1, n_samples))
    step = 1.0 / L
    alphas_desc = np.asarray(alphas_desc, dtype=np.float64)
    alpha_gpu = cp.asarray(alphas_desc, dtype=XtX.dtype)
    thresholds = alpha_gpu * step
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active_gpu = cp.arange(n_alphas, dtype=cp.int32)

    for iteration in range(int(max_iter)):
        if int(active_gpu.size) == 0:
            break

        y_active = yk[:, active_gpu]
        coef_old = coefs[:, active_gpu]

        grad = (XtX @ y_active - Xty.reshape(-1, 1)) / n_samp
        thresh = thresholds[active_gpu].reshape(1, -1)
        coef_new = cp.sign(y_active - step * grad) * cp.maximum(cp.abs(y_active - step * grad) - thresh, 0.0)

        t_old = tk[active_gpu]
        t_new = (1.0 + cp.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_new = coef_new + beta.reshape(1, -1) * (coef_new - coef_old)

        coefs[:, active_gpu] = coef_new
        yk[:, active_gpu] = y_new
        tk[active_gpu] = t_new

        active_ratio = float(int(active_gpu.size)) / float(max(1, n_alphas))
        check_every_eff = _adaptive_gpu_check_every(
            base_check_every=check_every,
            iteration=iteration,
            max_iter=int(max_iter),
            active_ratio=active_ratio,
        )
        should_check = ((iteration + 1) % check_every_eff == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (XtX @ coef_new - Xty.reshape(-1, 1)) / n_samp
            viol = cp.max(
                cp.maximum(
                    cp.abs(grad_sse) - alpha_gpu[active_gpu].reshape(1, -1),
                    0.0,
                ),
                axis=0,
            )
            converged_local_gpu = viol < float(tol)
        else:
            delta = cp.sum(cp.abs(coef_new - coef_old), axis=0)
            converged_local_gpu = delta < float(tol)

        done_gpu = active_gpu[converged_local_gpu]
        if int(done_gpu.size) == 0:
            continue

        n_iters_gpu[done_gpu] = int(iteration) + 1
        yk[:, done_gpu] = coefs[:, done_gpu]
        active_gpu = active_gpu[~converged_local_gpu]

    if int(active_gpu.size) > 0:
        n_iters_gpu[active_gpu] = int(max_iter)

    return coefs.T, cp.asnumpy(n_iters_gpu)


def _solve_lasso_path_gpu_fista_multi_fold_from_gram(
    XtX_batch,
    Xty_batch,
    *,
    n_samples_vec,
    alphas_desc,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve descending-alpha Lasso paths for all folds together on GPU.

    Note: Fused kernel optimization is disabled for multi-fold solver due to
    dtype complexity. The single-fold Lasso solver uses fused kernels.
    """
    import cupy as cp

    n_folds = int(XtX_batch.shape[0])
    n_features = int(XtX_batch.shape[1])
    n_alphas = int(alphas_desc.shape[0])

    coefs = cp.zeros((n_folds, n_features, n_alphas), dtype=XtX_batch.dtype)
    yk = coefs.copy()
    tk = cp.ones((n_folds, n_alphas), dtype=XtX_batch.dtype)
    n_iters_gpu = cp.zeros((n_folds, n_alphas), dtype=cp.int32)

    # Convert n_samples_vec to numpy using .get() if it's a CuPy array
    if hasattr(n_samples_vec, 'get'):
        n_vec_cpu = n_samples_vec.get().astype(np.float64).reshape(-1)
    else:
        n_vec_cpu = np.asarray(n_samples_vec, dtype=np.float64).reshape(-1)
    if n_vec_cpu.size != n_folds:
        raise ValueError("n_samples_vec must have one entry per fold")
    n_vec = cp.asarray(n_vec_cpu, dtype=XtX_batch.dtype)

    if lipschitz_L is not None:
        L = cp.full((n_folds,), float(lipschitz_L), dtype=XtX_batch.dtype)
    else:
        try:
            eigvals = cp.linalg.eigvalsh(XtX_batch)
            L = eigvals[:, -1] / n_vec
        except Exception:
            row_sum_bound = cp.max(cp.sum(cp.abs(XtX_batch), axis=2), axis=1) / n_vec
            L = cp.maximum(row_sum_bound, cp.asarray(1e-12, dtype=XtX_batch.dtype))

    step = 1.0 / L.reshape(n_folds, 1, 1)
    # Convert alphas_desc to numpy using .get() if it's a CuPy array
    if hasattr(alphas_desc, 'get'):
        alphas_cpu = alphas_desc.get().astype(np.float64)
    else:
        alphas_cpu = np.asarray(alphas_desc, dtype=np.float64)
    alpha_gpu = cp.asarray(alphas_cpu, dtype=XtX_batch.dtype).reshape(1, 1, n_alphas)
    thresholds = alpha_gpu * step

    Xty_expanded = Xty_batch.reshape(n_folds, n_features, 1)
    n_vec_expanded = n_vec.reshape(n_folds, 1, 1)
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active_gpu = cp.ones((n_folds, n_alphas), dtype=cp.bool_)
    active_count = int(n_folds * n_alphas)

    # Note: Fused kernels disabled for multi-fold solver due to dtype complexity
    # The single-fold Lasso._fit_gpu uses fused kernels
    use_fused = False
    fused = None

    for iteration in range(int(max_iter)):
        if active_count == 0:
            break

        active_expanded = active_gpu[:, cp.newaxis, :]

        coef_old = coefs.copy()
        grad = (cp.matmul(XtX_batch, yk) - Xty_expanded) / n_vec_expanded

        # Proximal step: soft thresholding
        yk_step = yk - step * grad
        coef_candidate = cp.sign(yk_step) * cp.maximum(cp.abs(yk_step) - thresholds, 0.0)
        coefs = cp.where(active_expanded, coef_candidate, coefs)

        t_old = tk
        t_new = (1.0 + cp.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_candidate = coefs + beta[:, cp.newaxis, :] * (coefs - coef_old)
        yk = cp.where(active_expanded, y_candidate, yk)
        tk = cp.where(active_gpu, t_new, tk)

        active_ratio = float(active_count) / float(max(1, n_folds * n_alphas))
        check_every_eff = _adaptive_gpu_check_every(
            base_check_every=check_every,
            iteration=iteration,
            max_iter=int(max_iter),
            active_ratio=active_ratio,
        )
        should_check = ((iteration + 1) % check_every_eff == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (cp.matmul(XtX_batch, coefs) - Xty_expanded) / n_vec_expanded
            violation = cp.max(cp.maximum(cp.abs(grad_sse) - alpha_gpu, 0.0), axis=1)
            converged_local_gpu = violation < float(tol)
        else:
            delta = cp.sum(cp.abs(coefs - coef_old), axis=1)
            converged_local_gpu = delta < float(tol)

        newly_done_gpu = active_gpu & converged_local_gpu
        done_count = int(cp.count_nonzero(newly_done_gpu).item())
        if done_count == 0:
            continue

        n_iters_gpu[newly_done_gpu] = int(iteration) + 1
        yk = cp.where(newly_done_gpu[:, cp.newaxis, :], coefs, yk)
        active_gpu = active_gpu & (~converged_local_gpu)
        active_count -= done_count

    n_iters_gpu[active_gpu] = int(max_iter)

    return cp.transpose(coefs, (0, 2, 1)), cp.asnumpy(n_iters_gpu)


def _solve_lasso_path_cpu_from_gram(
    XtX: np.ndarray,
    Xty: np.ndarray,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    cpu_solver: str,
    lipschitz_L: Optional[float] = None,
    cd_kkt_check_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a descending-alpha Lasso path on CPU using one precomputed Gram matrix."""
    n_features = int(XtX.shape[0])
    n_alphas = int(alphas_desc.shape[0])

    coefs_path = np.zeros((n_alphas, n_features), dtype=np.float64)
    n_iters = np.zeros(n_alphas, dtype=np.int32)

    coef = np.zeros(n_features, dtype=np.float64)
    stopping_name = str(stopping).lower()
    solver_name = str(cpu_solver).lower()

    if solver_name == "fista":
        return _solve_lasso_path_cpu_fista_batched_from_gram(
            XtX,
            Xty,
            n_samples=n_samples,
            alphas_desc=alphas_desc,
            max_iter=max_iter,
            tol=tol,
            stopping=stopping,
            lipschitz_L=lipschitz_L,
            check_every=2,
        )

    global _NUMBA_CD_DISABLED
    use_numba_cd = (
        _NUMBA_AVAILABLE
        and (not _NUMBA_CD_DISABLED)
        and solver_name == "coordinate_descent"
    )

    if use_numba_cd:
        try:
            return _solve_lasso_path_cpu_cd_numba(
                XtX,
                Xty,
                n_samples=n_samples,
                alphas_desc=alphas_desc,
                max_iter=max_iter,
                tol=tol,
                stopping=stopping,
                cd_kkt_check_every=cd_kkt_check_every,
            )
        except Exception:
            _NUMBA_CD_DISABLED = True

    # Coordinate descent with incremental gradient updates.
    X_sq_norms = np.diag(XtX).astype(np.float64, copy=False)
    grad = XtX @ coef - Xty
    alpha_scaled_desc = np.asarray(alphas_desc, dtype=np.float64) * float(max(1, n_samples))
    active_mask = np.zeros((n_features,), dtype=bool)
    cd_kkt_check_every = max(1, int(cd_kkt_check_every))

    for alpha_idx, alpha in enumerate(alphas_desc):
        alpha_scaled = float(alpha_scaled_desc[alpha_idx])
        prev_alpha_scaled = float(alpha_scaled_desc[alpha_idx - 1]) if alpha_idx > 0 else alpha_scaled

        # Strong rule screening: expand active set before cyclic updates.
        strong_thresh = max(0.0, 2.0 * alpha_scaled - prev_alpha_scaled)
        active_mask |= np.abs(Xty) >= strong_thresh
        if not bool(np.any(active_mask)):
            active_mask[int(np.argmax(np.abs(Xty)))] = True

        converged = False

        for iteration in range(int(max_iter)):
            coef_delta_l1 = 0.0

            active_idx = np.flatnonzero(active_mask)
            for j in active_idx:
                denom = float(X_sq_norms[j])
                old_val = float(coef[j])

                if denom > 1e-10:
                    rho_j = -float(grad[j]) + denom * old_val
                    new_val = _soft_threshold_scalar(rho_j, alpha_scaled) / denom
                else:
                    new_val = 0.0

                delta = new_val - old_val
                if abs(delta) > 0.0:
                    coef[j] = new_val
                    grad += XtX[:, j] * delta
                    coef_delta_l1 += abs(delta)

            # glmnet-style optimization can skip full inactive KKT scans on every pass,
            # then force a check when updates become small.
            should_kkt_scan = (
                ((iteration + 1) % cd_kkt_check_every == 0)
                or (coef_delta_l1 < float(tol))
                or (iteration + 1 == int(max_iter))
            )
            violation = float("inf")
            inactive_violation_idx = np.empty((0,), dtype=np.int64)

            if should_kkt_scan:
                violation_vec = np.maximum(
                    np.abs(grad / float(max(1, n_samples))) - float(alpha),
                    0.0,
                )
                inactive_violation_idx = np.where((violation_vec > float(tol)) & (~active_mask))[0]
                if inactive_violation_idx.size > 0:
                    active_mask[inactive_violation_idx] = True
                violation = float(np.max(violation_vec))

            if stopping_name == "kkt":
                if should_kkt_scan and violation < float(tol):
                    n_iters[alpha_idx] = iteration + 1
                    converged = True
                    break
            else:
                if coef_delta_l1 < float(tol) and inactive_violation_idx.size == 0:
                    n_iters[alpha_idx] = iteration + 1
                    converged = True
                    break

        if not converged:
            n_iters[alpha_idx] = int(max_iter)

        coefs_path[alpha_idx, :] = coef
        active_mask |= np.abs(coef) > 0.0

    return coefs_path, n_iters


def _solve_lasso_path_gpu_from_gram(
    XtX,
    Xty,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve a descending-alpha Lasso path on GPU using one precomputed Gram matrix."""
    return _solve_lasso_path_gpu_fista_batched_from_gram(
        XtX,
        Xty,
        n_samples=n_samples,
        alphas_desc=alphas_desc,
        max_iter=max_iter,
        tol=tol,
        stopping=stopping,
        lipschitz_L=lipschitz_L,
        check_every=check_every,
    )


def _batch_mse_numpy(
    X_val: np.ndarray,
    y_val: np.ndarray,
    coefs_path: np.ndarray,
    intercepts_path: np.ndarray,
    sample_weight_val: Optional[np.ndarray],
) -> np.ndarray:
    preds = X_val @ coefs_path.T + intercepts_path.reshape(1, -1)
    sq_err = (y_val.reshape(-1, 1) - preds) ** 2

    if sample_weight_val is None:
        return np.mean(sq_err, axis=0)

    denom = float(np.sum(sample_weight_val))
    if denom <= 0.0:
        return np.mean(sq_err, axis=0)

    return np.sum(sample_weight_val.reshape(-1, 1) * sq_err, axis=0) / denom


def _batch_mse(
    X_val,
    y_val,
    coefs_path,
    intercepts_path,
    backend,
    sample_weight_val,
) -> np.ndarray:
    """
    Compute MSE for multiple coefficient vectors.

    Parameters
    ----------
    X_val : array-like
        Validation design matrix.
    y_val : array-like
        Validation response.
    coefs_path : array-like
        Coefficient matrix (n_alphas, n_features).
    intercepts_path : array-like
        Intercept vector (n_alphas,).
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    sample_weight_val : array-like or None
        Sample weights.

    Returns
    -------
    mse : ndarray
        MSE for each alpha.
    """
    preds = X_val @ coefs_path.T + intercepts_path.reshape(1, -1)
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


def _soft_threshold_torch(x, gamma):
    """Soft thresholding operator for Torch tensors."""
    import torch
    return torch.sign(x) * torch.maximum(torch.abs(x) - gamma, torch.tensor(0.0, dtype=x.dtype, device=x.device))


def _fit_lasso_single_alpha_fast(
    X,
    y,
    *,
    alpha: float,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
    stopping: str,
    device: str,
    cpu_solver: str,
    cd_kkt_check_every: int = 1,
    sample_weight=None,
) -> Dict[str, object]:
    """Fast single-alpha Lasso fit using optimized Gram-based path solvers."""
    device_name = str(device).lower()
    alpha_vec = np.asarray([float(alpha)], dtype=np.float64)

    # Check if inputs are torch tensors on GPU
    is_torch_gpu = False
    try:
        import torch
        is_torch_gpu = device_name == Device.CUDA.value and isinstance(X, torch.Tensor)
    except Exception:
        pass

    if device_name == Device.CUDA.value and not is_torch_gpu:
        # CuPy GPU path
        import cupy as cp

        X_arr = cp.asarray(X)
        y_arr = cp.asarray(y).reshape(-1)

        if sample_weight is not None:
            sw = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sw)
            X_arr = X_arr * sqrt_sw[:, cp.newaxis]
            y_arr = y_arr * sqrt_sw

        if bool(fit_intercept):
            if sw is not None:
                # Weighted mean on original (pre-sqrt) data
                X_orig = X_arr / sqrt_sw[:, cp.newaxis]
                y_orig = y_arr / sqrt_sw
                w_sum = float(cp.sum(sw))
                X_mean = cp.sum(X_orig * sw[:, cp.newaxis], axis=0) / w_sum
                y_mean = float(cp.sum(y_orig * sw)) / w_sum
                X_centered = X_arr - sqrt_sw[:, cp.newaxis] * X_mean
                y_centered = y_arr - sqrt_sw * y_mean
            else:
                X_mean = cp.mean(X_arr, axis=0)
                y_mean = cp.mean(y_arr)
                X_centered = X_arr - X_mean
                y_centered = y_arr - y_mean
        else:
            X_mean = cp.zeros((X_arr.shape[1],), dtype=X_arr.dtype)
            y_mean = cp.array(0.0, dtype=X_arr.dtype)
            X_centered = X_arr
            y_centered = y_arr

        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        coefs_desc, n_iters = _solve_lasso_path_gpu_from_gram(
            XtX,
            Xty,
            n_samples=int(X_arr.shape[0]),
            alphas_desc=alpha_vec,
            max_iter=int(max_iter),
            tol=float(tol),
            stopping=str(stopping),
            lipschitz_L=None,
            check_every=8,
        )

        coef_gpu = coefs_desc[0]
        if bool(fit_intercept):
            intercept_gpu = y_mean - X_mean @ coef_gpu
            intercept = float(cp.asnumpy(intercept_gpu))
        else:
            intercept = 0.0

        coef = np.asarray(cp.asnumpy(coef_gpu), dtype=np.float64)
        return {
            "coef": coef,
            "intercept": float(intercept),
            "n_iter": int(n_iters[0]),
            "n_samples": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
        }

    elif is_torch_gpu:
        # Torch GPU path - use FISTA solver directly on GPU tensors
        import torch

        X_arr = X
        y_arr = y.reshape(-1) if isinstance(y, torch.Tensor) else torch.as_tensor(
            y, dtype=X_arr.dtype, device=X_arr.device
        ).reshape(-1)

        if sample_weight is not None:
            sw = sample_weight if isinstance(sample_weight, torch.Tensor) else torch.as_tensor(
                sample_weight, dtype=X_arr.dtype, device=X_arr.device
            )
            sqrt_sw = torch.sqrt(sw)
            X_arr = X_arr * sqrt_sw[:, None]
            y_arr = y_arr * sqrt_sw

        if bool(fit_intercept):
            if sw is not None:
                # Weighted mean: sum(w*X)/sum(w) on original (pre-sqrt) data
                # But X_arr is already sqrt(w)*X, so mean of sqrt(w)*X is not
                # the weighted mean. Use the original data for centering.
                X_orig = X_arr / sqrt_sw[:, None]
                y_orig = y_arr / sqrt_sw
                w_sum = float(sw.sum())
                X_mean = torch.sum(X_orig * sw[:, None], dim=0) / w_sum
                y_mean = float(torch.sum(y_orig * sw)) / w_sum
                # Re-center the sqrt-weighted data using the weighted mean
                X_centered = X_arr - sqrt_sw[:, None] * X_mean
                y_centered = y_arr - sqrt_sw * y_mean
            else:
                X_mean = torch.mean(X_arr, dim=0)
                y_mean = torch.mean(y_arr)
                X_centered = X_arr - X_mean
                y_centered = y_arr - y_mean
        else:
            X_mean = torch.zeros((X_arr.shape[1],), dtype=X_arr.dtype, device=X_arr.device)
            y_mean = torch.tensor(0.0, dtype=X_arr.dtype, device=X_arr.device)
            X_centered = X_arr
            y_centered = y_arr

        n_samples = int(X_arr.shape[0])
        n_features = int(X_arr.shape[1])

        # Precompute Gram matrix and X'y for FISTA gradient
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Compute Lipschitz constant L = max eigenvalue of XtX / n
        try:
            eigvals = torch.linalg.eigvalsh(XtX)
            L = eigvals[-1] / n_samples
        except Exception:
            L = torch.sum(X_centered ** 2) / n_samples
        L = torch.clamp(L, min=1e-10)

        step = 1.0 / L
        thresh = float(alpha) * step

        # FISTA initialization
        coef = torch.zeros(n_features, dtype=X_arr.dtype, device=X_arr.device)
        z = coef.clone()
        t = torch.tensor(1.0, dtype=X_arr.dtype, device=X_arr.device)

        # FISTA iterations
        for iteration in range(int(max_iter)):
            coef_old = coef.clone()

            # Gradient step at z
            grad = (XtX @ z - Xty) / n_samples
            coef = _soft_threshold_torch(z - step * grad, thresh)

            # Momentum update
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            z = coef + ((t - 1.0) / t_new) * (coef - coef_old)
            t = t_new

            # Convergence check
            if str(stopping).lower() == "kkt":
                grad_sse = (XtX @ coef - Xty) / n_samples
                violation = torch.max(torch.maximum(torch.abs(grad_sse) - float(alpha), torch.tensor(0.0, dtype=X_arr.dtype, device=X_arr.device)))
                if violation < float(tol):
                    break
            else:
                if torch.sum(torch.abs(coef - coef_old)) < float(tol):
                    break

        # Build coefficients
        if bool(fit_intercept):
            intercept_torch = y_mean - X_mean @ coef
            intercept = float(intercept_torch.item())
        else:
            intercept = 0.0

        coef_np = np.asarray(coef.detach().cpu().numpy(), dtype=np.float64)
        return {
            "coef": coef_np,
            "intercept": float(intercept),
            "n_iter": int(iteration + 1),
            "n_samples": n_samples,
            "n_features": n_features,
        }

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).reshape(-1)

    if sample_weight is not None:
        sw = np.asarray(sample_weight)
        sqrt_sw = np.sqrt(sw)
        X_arr = X_arr * sqrt_sw[:, np.newaxis]
        y_arr = y_arr * sqrt_sw

    if bool(fit_intercept):
        X_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        X_centered = X_arr - X_mean
        y_centered = y_arr - y_mean
    else:
        X_mean = np.zeros((X_arr.shape[1],), dtype=np.float64)
        y_mean = 0.0
        X_centered = X_arr
        y_centered = y_arr

    XtX = X_centered.T @ X_centered
    Xty = X_centered.T @ y_centered

    coefs_desc, n_iters = _solve_lasso_path_cpu_from_gram(
        XtX,
        Xty,
        n_samples=int(X_arr.shape[0]),
        alphas_desc=alpha_vec,
        max_iter=int(max_iter),
        tol=float(tol),
        stopping=str(stopping),
        cpu_solver=str(cpu_solver),
        lipschitz_L=None,
        cd_kkt_check_every=int(cd_kkt_check_every),
    )

    coef = np.asarray(coefs_desc[0], dtype=np.float64)
    if bool(fit_intercept):
        intercept = float(y_mean - X_mean @ coef)
    else:
        intercept = 0.0

    return {
        "coef": coef,
        "intercept": float(intercept),
        "n_iter": int(n_iters[0]),
        "n_samples": int(X_arr.shape[0]),
        "n_features": int(X_arr.shape[1]),
    }


def _select_lasso_alpha_cv(
    X,
    y,
    *,
    alphas=None,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state: Optional[int] = None,
    sample_weight=None,
    fit_intercept: bool = False,
    device: Union[str, Device] = Device.CPU,
    max_iter: int = 3000,
    tol: float = 1e-4,
    cpu_solver: str = "coordinate_descent",
    method: str = "standard",
    cd_kkt_check_every: Optional[int] = None,
    gpu_cv_mixed_precision: bool = True,
    return_details: bool = False,
    cache_key: Optional[Tuple[Any, ...]] = None,
):
    """
    Select alpha via K-fold CV using statgpu's own Lasso implementation.

    Notes
    -----
    - Does not depend on sklearn.
    - Supports GPU path by setting ``device='cuda'``.
    """
    device_name = str(device).lower()
    use_gpu = device_name == Device.CUDA.value
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

    cv_method = _normalize_lassocv_method(method)
    requested_cd_kkt_check_every = _normalize_cd_kkt_check_every(cd_kkt_check_every)

    if alphas is None:
        if gpu_input_cupy or gpu_input_torch:
            # Get backend based on input type
            if gpu_input_torch:
                backend = get_backend(backend='torch', device='cuda')
            else:
                backend = get_backend(backend='cupy', device='cuda')
            alpha_grid = _default_lasso_alpha_grid_backend(
                X,
                y,
                backend,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
            )
        else:
            alpha_grid = _default_lasso_alpha_grid(
                X_np,
                y_np,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
            )
    else:
        alpha_grid = np.asarray(alphas, dtype=np.float64).reshape(-1)
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
                    X,
                    y,
                    backend,
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                )
            else:
                alpha_grid = _default_lasso_alpha_grid(
                    X_np,
                    y_np,
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                )

    user_folds = _normalize_cv_splits(cv_splits, n_samples=n_samples)
    effective_n_folds = int(len(user_folds)) if user_folds is not None else int(cv_folds)

    if int(n_samples) < 4 or int(alpha_grid.size) == 1 or int(effective_n_folds) < 2:
        alpha0 = float(alpha_grid[0])
        if not return_details:
            return alpha0
        return {
            "alpha": alpha0,
            "alphas": alpha_grid.astype(np.float64, copy=False),
            "mse_path": np.full((int(alpha_grid.size), 1), np.nan, dtype=np.float64),
            "mean_mse": np.full(int(alpha_grid.size), np.nan, dtype=np.float64),
        }

    if user_folds is not None:
        folds = user_folds
    else:
        folds = _kfold_indices(
            n_samples=int(n_samples),
            n_splits=int(cv_folds),
            random_state=random_state,
        )

    folds_are_complements = _folds_are_complements(folds, n_samples=int(n_samples))

    alpha_grid = alpha_grid.astype(np.float64, copy=False)
    n_alpha = int(alpha_grid.size)
    n_folds = int(len(folds))

    cache_key_eff = cache_key
    if cache_key_eff is None and _LASSO_CV_ALPHA_CACHE_MAXSIZE > 0:
        cache_key_eff = _make_lasso_cv_auto_cache_key(
            X=X,
            y=y,
            sample_weight=sample_weight,
            alpha_grid=alpha_grid,
            folds=folds,
            fit_intercept=bool(fit_intercept),
            use_gpu=bool(use_gpu),
            max_iter=int(max_iter),
            tol=float(tol),
            cpu_solver=str(cpu_solver),
            cv_method=str(cv_method),
            cd_kkt_check_every=requested_cd_kkt_check_every,
            gpu_cv_mixed_precision=bool(gpu_cv_mixed_precision),
        )

    cached_details = _lasso_cv_cache_get(cache_key_eff)
    if cached_details is not None:
        if return_details:
            return cached_details
        return float(cached_details["alpha"])

    # Evaluate alpha path in descending order for warm-start efficiency.
    alpha_order_desc = np.argsort(-alpha_grid)
    alpha_desc = alpha_grid[alpha_order_desc]

    mse_path = np.full((n_alpha, n_folds), np.nan, dtype=np.float64)

    best_alpha = float(alpha_grid[0])
    best_mse = float("inf")

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
                sw_train = None  # initialized per-fold in slow path; None for fast path

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
                        sqrt_sw = backend.sqrt(sw_train)
                        X_train = X_train * sqrt_sw[:, None]
                        y_train = y_train * sqrt_sw

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
                    # For weighted case, effective sample size is sum(weights)
                    if sw_train is not None:
                        n_train = float(backend.sum(sw_train))
                    else:
                        n_train = int(X_train.shape[0])

                XtX_folds.append(XtX)
                Xty_folds.append(Xty)
                n_train_folds.append(float(n_train) if sw_train is not None else int(n_train))
                X_mean_folds.append(X_mean)
                y_mean_folds.append(y_mean)
                fold_eval_payload.append((X_val, y_val, sw_val))

            XtX_batch = backend.stack(XtX_folds, axis=0)
            Xty_batch = backend.stack(Xty_folds, axis=0)

            # Use native Torch FISTA solver for Torch backend
            if hasattr(xp, '__name__') and 'torch' in xp.__name__.lower():
                import torch
                n_samples_vec_torch = torch.tensor(np.asarray(n_train_folds, dtype=np.int32), device=XtX_batch.device, dtype=XtX_batch.dtype)

                coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
                    XtX_batch,
                    Xty_batch,
                    n_samples_vec=n_samples_vec_torch,
                    alphas_desc=alpha_desc,
                    max_iter=int(max_iter),
                    tol=float(tol),
                    stopping="coef_delta",
                    lipschitz_L=None,
                    check_every=8,
                )

                # Convert results back to numpy for evaluation
                for fold_idx in range(int(len(folds))):
                    coefs_desc_np = coefs_batch_desc[fold_idx]  # already numpy from the solver

                    if bool(fit_intercept):
                        y_mean_val = float(y_mean_folds[fold_idx])
                        X_mean_val = X_mean_folds[fold_idx]
                        intercepts_desc = y_mean_val - X_mean_val @ coefs_desc_np.T
                        intercepts_desc_gpu = backend.asarray(intercepts_desc)
                        coefs_desc_gpu = backend.asarray(coefs_desc_np)
                    else:
                        intercepts_desc_gpu = backend.zeros((coefs_desc_np.shape[0],), dtype=coefs_desc_np.dtype)
                        coefs_desc_gpu = backend.asarray(coefs_desc_np)

                    X_val, y_val, sw_val = fold_eval_payload[fold_idx]
                    mse_desc = _batch_mse(X_val, y_val, coefs_desc_gpu, intercepts_desc_gpu, backend, sw_val)

                    mse_path[alpha_order_desc, fold_idx] = mse_desc
            else:
                # CuPy backend - use existing solver directly
                import cupy as cp
                n_samples_vec_cp = cp.asarray(np.asarray(n_train_folds, dtype=np.int32))

                coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram(
                    XtX_batch,
                    Xty_batch,
                    n_samples_vec=n_samples_vec_cp,
                    alphas_desc=alpha_desc,
                    max_iter=int(max_iter),
                    tol=float(tol),
                    stopping="coef_delta",
                    lipschitz_L=None,
                    check_every=8,
                )

                for fold_idx in range(int(len(folds))):
                    coefs_desc = coefs_batch_desc[fold_idx]

                    if bool(fit_intercept):
                        intercepts_desc = y_mean_folds[fold_idx] - X_mean_folds[fold_idx] @ coefs_desc.T
                    else:
                        intercepts_desc = backend.zeros((coefs_desc.shape[0],), dtype=coefs_desc.dtype)

                    X_val, y_val, sw_val = fold_eval_payload[fold_idx]
                    mse_desc = _batch_mse(X_val, y_val, coefs_desc, intercepts_desc, backend, sw_val)

                    mse_path[alpha_order_desc, fold_idx] = mse_desc

        except Exception as exc:
            raise RuntimeError(
                "GPU path failed in _select_lasso_alpha_cv with device='cuda'; "
                "CPU fallback is disabled for strict CUDA execution."
            ) from exc

    if not use_gpu:
        if gpu_requested:
            raise RuntimeError(
                "device='cuda' requested but GPU path was not executed; "
                "CPU fallback is disabled for strict CUDA execution."
            )
        cpu_solver_name = str(cpu_solver).lower()

        if cv_method == "glmnet":
            # glmnet-like CV profile: coordinate-descent path with periodic full KKT scans.
            cpu_solver_name = "coordinate_descent"

        if requested_cd_kkt_check_every is None:
            cd_kkt_check_every_effective = 4 if cv_method == "glmnet" else 1
        else:
            cd_kkt_check_every_effective = int(requested_cd_kkt_check_every)

        fast_fold_stats = (sample_weight_np is None) and bool(folds_are_complements)
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
                    sqrt_sw = np.sqrt(sw_train)
                    X_train = X_train * sqrt_sw[:, np.newaxis]
                    y_train = y_train * sqrt_sw

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

            coefs_desc, _ = _solve_lasso_path_cpu_from_gram(
                XtX,
                Xty,
                n_samples=int(n_train),
                alphas_desc=alpha_desc,
                max_iter=int(max_iter),
                tol=float(tol),
                stopping="coef_delta",
                cpu_solver=cpu_solver_name,
                lipschitz_L=None,
                cd_kkt_check_every=cd_kkt_check_every_effective,
            )

            if bool(fit_intercept):
                intercepts_desc = y_mean - X_mean @ coefs_desc.T
            else:
                intercepts_desc = np.zeros((coefs_desc.shape[0],), dtype=np.float64)

            mse_desc = _batch_mse_numpy(
                X_val,
                y_val,
                coefs_desc,
                intercepts_desc,
                sw_val,
            )

            mse_path[alpha_order_desc, fold_idx] = np.asarray(mse_desc, dtype=np.float64)

    for alpha_idx, alpha in enumerate(alpha_grid):
        alpha_f = float(alpha)
        valid = np.isfinite(mse_path[alpha_idx])
        if not bool(np.any(valid)):
            continue

        mean_mse = float(np.mean(mse_path[alpha_idx, valid]))
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = alpha_f

    mean_mse_vec = np.full(int(alpha_grid.size), np.nan, dtype=np.float64)
    for alpha_idx in range(int(alpha_grid.size)):
        valid = np.isfinite(mse_path[alpha_idx])
        if bool(np.any(valid)):
            mean_mse_vec[alpha_idx] = float(np.mean(mse_path[alpha_idx, valid]))

    details = {
        "alpha": float(best_alpha),
        "alphas": alpha_grid.astype(np.float64, copy=False),
        "mse_path": mse_path,
        "mean_mse": mean_mse_vec,
    }

    _lasso_cv_cache_put(cache_key_eff, details)

    if return_details:
        return details

    return float(details["alpha"])


from ._penalized import PenalizedLinearRegression as _PenalizedLinearRegression


class Lasso(_PenalizedLinearRegression):
    """Thin sklearn-style wrapper over ``PenalizedLinearRegression`` with L1 penalty."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        inference_method: str = "debiased",
        n_bootstrap: int = 200,
        bootstrap_random_state: Optional[int] = None,
        enable_simultaneous_inference: bool = False,
        simultaneous_method: str = "maxz_bootstrap",
        simultaneous_alpha: float = 0.05,
        simultaneous_n_bootstrap: int = 1000,
        simultaneous_random_state: Optional[int] = None,
        simultaneous_include_intercept: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        solver: str = "fista",
        cpu_solver: str = "coordinate_descent",
        lipschitz_L: Optional[float] = None,
        admm_rho: float = 1.0,
        gpu_memory_cleanup: bool = False,
    ):
        self.stopping = str(stopping).lower()
        self.inference_method = str(inference_method).lower()
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_random_state = bootstrap_random_state
        self.enable_simultaneous_inference = bool(enable_simultaneous_inference)
        self.simultaneous_method = str(simultaneous_method).lower()
        self.simultaneous_alpha = float(simultaneous_alpha)
        self.simultaneous_n_bootstrap = int(simultaneous_n_bootstrap)
        self.simultaneous_random_state = simultaneous_random_state
        self.simultaneous_include_intercept = bool(simultaneous_include_intercept)
        self.admm_rho = float(admm_rho)
        super().__init__(
            penalty="l1",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            stopping=stopping,
        )