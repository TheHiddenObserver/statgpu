"""Utility functions for knockoff feature selection."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

from statgpu.backends import (
    _get_torch_device_str,
    _get_xp,
    _resolve_backend,
    _to_float_scalar,
    _to_numpy,
)

# Re-export for backward compatibility with modules that import from here
__all__ = [
    "_get_xp",
    "_resolve_backend",
    "_to_numpy",
    "_to_float_scalar",
]


_LASSO_DIFF_CACHE_MAXSIZE = int(os.getenv("STATGPU_KNOCKOFF_LASSO_CACHE_SIZE", "32"))
_LASSO_DIFF_CACHE: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()


def _array_identity_token(x: Any) -> Tuple[Any, ...]:
    if x is None:
        return ("none",)

    # Try CuPy array
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return ("cupy", int(x.data.ptr), tuple(int(v) for v in x.shape), str(x.dtype))
    except Exception:
        pass

    # Try Torch tensor
    try:
        import torch

        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                return ("torch_cuda", int(x.data_ptr()), tuple(int(v) for v in x.shape), str(x.dtype))
            else:
                return ("torch_cpu", int(x.data_ptr()), tuple(int(v) for v in x.shape), str(x.dtype))
    except Exception:
        pass

    # Default to NumPy
    arr = np.asarray(x)
    ptr = int(arr.__array_interface__["data"][0]) if int(arr.size) > 0 else 0
    return ("numpy", ptr, tuple(int(v) for v in arr.shape), str(arr.dtype))


def _int_array_signature(x: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.int64).reshape(-1))
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _lasso_diff_cache_get(cache_key: Optional[Tuple[Any, ...]]) -> Optional[np.ndarray]:
    if cache_key is None or _LASSO_DIFF_CACHE_MAXSIZE <= 0:
        return None

    cached = _LASSO_DIFF_CACHE.get(cache_key)
    if cached is None:
        return None

    _LASSO_DIFF_CACHE.move_to_end(cache_key)
    return np.asarray(cached, dtype=np.float64).copy()


def _lasso_diff_cache_put(cache_key: Optional[Tuple[Any, ...]], value: np.ndarray) -> None:
    if cache_key is None or _LASSO_DIFF_CACHE_MAXSIZE <= 0:
        return

    _LASSO_DIFF_CACHE[cache_key] = np.asarray(value, dtype=np.float64).copy()
    _LASSO_DIFF_CACHE.move_to_end(cache_key)

    while len(_LASSO_DIFF_CACHE) > int(_LASSO_DIFF_CACHE_MAXSIZE):
        _LASSO_DIFF_CACHE.popitem(last=False)


def _make_lasso_coef_diff_cache_key(
    *,
    X_std,
    X_knock,
    y,
    random_state: Optional[int],
    backend_name: str,
    max_iter_eff: int,
    tol_eff: float,
    cv_folds_eff: int,
    n_alphas_eff: int,
    lasso_cv_impl: str,
    fast_profile_eff: str,
    knockpy_style: bool,
) -> Optional[Tuple[Any, ...]]:
    # random_state=None implies a fresh random permutation every call; disable reuse.
    if random_state is None:
        return None

    return (
        "knockoff_lasso_diff_v1",
        _array_identity_token(X_std),
        _array_identity_token(X_knock),
        _array_identity_token(y),
        int(random_state),
        str(backend_name).lower(),
        int(max_iter_eff),
        float(tol_eff),
        int(cv_folds_eff),
        int(n_alphas_eff),
        str(lasso_cv_impl).lower(),
        str(fast_profile_eff).lower(),
        bool(knockpy_style),
    )


def _normalize_compat_mode(compat_mode: str) -> str:
    key = str(compat_mode).strip().lower()
    if key in ("statgpu", "default"):
        return "statgpu"
    if key in ("knockpy", "compat", "knockpy_compat"):
        return "knockpy"
    raise ValueError("compat_mode must be one of: 'statgpu', 'knockpy'")


def _normalize_lasso_cv_impl(lasso_cv_impl: str) -> str:
    key = str(lasso_cv_impl).strip().lower()
    if key in ("auto", "default"):
        return "auto"
    if key in ("statgpu", "internal"):
        return "statgpu"
    if key in ("sklearn", "knockpy", "knockpy_sklearn"):
        return "sklearn"
    raise ValueError("lasso_cv_impl must be one of: 'auto', 'statgpu', 'sklearn'")


def _normalize_lasso_fast_profile(lasso_fast_profile: str) -> str:
    key = str(lasso_fast_profile).strip().lower()
    if key in ("off", "none", "default"):
        return "off"
    if key in ("auto",):
        return "auto"
    if key in ("moderate", "balanced"):
        return "moderate"
    if key in ("aggressive", "fast"):
        return "aggressive"
    raise ValueError(
        "lasso_fast_profile must be one of: 'off', 'auto', 'moderate', 'aggressive'"
    )


def _resolve_lasso_fast_profile_for_problem(lasso_fast_profile: str, problem_size: int) -> str:
    profile = _normalize_lasso_fast_profile(lasso_fast_profile)
    if profile != "auto":
        return profile

    if int(problem_size) >= 2_000_000:
        return "moderate"
    return "off"


@contextmanager
def _temporary_numpy_seed(seed: Optional[int]):
    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _calc_mineig_np(M: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
    return float(np.min(eigvals))


def _shift_until_psd_np(M: np.ndarray, tol: float) -> np.ndarray:
    mineig = _calc_mineig_np(M)
    if mineig < float(tol):
        M = M + (float(tol) - mineig) * np.eye(M.shape[0], dtype=np.float64)
    return 0.5 * (M + M.T)


def _scale_until_psd_np(
    Sigma: np.ndarray,
    S: np.ndarray,
    tol: float = 1e-5,
    num_iter: int = 25,
):
    S_shifted = _shift_until_psd_np(S, tol)

    lower = 0.0
    upper = 1.0
    for _ in range(int(num_iter)):
        gamma = 0.5 * (lower + upper)
        V = 2.0 * Sigma - gamma * S_shifted
        try:
            np.linalg.cholesky(V - float(tol) * np.eye(V.shape[0], dtype=np.float64))
            lower = gamma
        except np.linalg.LinAlgError:
            upper = gamma

    gamma = float(lower)
    return gamma * S_shifted, gamma


def _estimate_covariance_knockpy_style(
    X: np.ndarray,
    *,
    shrinkage: str = "ledoitwolf",
    tol: float = 1e-4,
):
    X_np = np.asarray(X, dtype=np.float64)

    shrink_key = str(shrinkage).strip().lower()
    if shrink_key in ("none", "mle"):
        shrink_key = "none"

    Sigma = None
    inv_sigma = None
    estimator_name = shrink_key

    if shrink_key == "none":
        Sigma = np.cov(X_np.T)
        if _calc_mineig_np(Sigma) < float(tol):
            shrink_key = "ledoitwolf"
            estimator_name = "ledoitwolf_auto"

    if shrink_key != "none":
        try:
            from sklearn import covariance as sk_cov
        except Exception:
            # Fallback keeps compatibility even when sklearn is unavailable.
            Sigma = np.cov(X_np.T)
            estimator_name = "mle_fallback_no_sklearn"
        else:
            if shrink_key == "ledoitwolf":
                estimator = sk_cov.LedoitWolf()
            elif shrink_key in ("graphicallasso", "glasso"):
                estimator = sk_cov.GraphicalLasso(alpha=0.1)
            else:
                raise ValueError(
                    "modelx_shrinkage must be one of: 'ledoitwolf', 'none', 'mle', 'graphicallasso'"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_np)
            Sigma = np.asarray(estimator.covariance_, dtype=np.float64)
            inv_sigma = np.asarray(estimator.precision_, dtype=np.float64)
            estimator_name = shrink_key

    Sigma = 0.5 * (np.asarray(Sigma, dtype=np.float64) + np.asarray(Sigma, dtype=np.float64).T)
    if inv_sigma is None:
        try:
            inv_sigma = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            ridge = max(1e-8, -_calc_mineig_np(Sigma) + 1e-8)
            Sigma = Sigma + ridge * np.eye(Sigma.shape[0], dtype=np.float64)
            Sigma = 0.5 * (Sigma + Sigma.T)
            inv_sigma = np.linalg.inv(Sigma)

    return Sigma, np.asarray(inv_sigma, dtype=np.float64), estimator_name


def _compute_smatrix_knockpy_style(
    Sigma: np.ndarray,
    *,
    method: str = "mvr",
    tol: float = 1e-5,
):
    Sigma_np = np.asarray(Sigma, dtype=np.float64)
    p = int(Sigma_np.shape[0])
    groups = np.arange(1, p + 1, dtype=np.int64)

    source = "equicorrelated_fallback"
    S = None
    try:
        from knockpy import smatrix as kp_smatrix

        S = kp_smatrix.compute_smatrix(
            Sigma=Sigma_np,
            groups=groups,
            method=str(method).strip().lower(),
        )
        source = "knockpy"
    except Exception:
        # Robust fallback if knockpy is not installed.
        min_eig = _calc_mineig_np(Sigma_np)
        s_val = min(2.0 * min_eig, 1.0)
        if s_val <= 1e-12:
            raise ValueError("Failed to construct model-X knockoff S-matrix")
        S = s_val * np.eye(p, dtype=np.float64)

    S = _shift_until_psd_np(np.asarray(S, dtype=np.float64), tol=float(tol))
    S, gamma = _scale_until_psd_np(Sigma_np, S, tol=float(tol), num_iter=25)
    return S, source, float(gamma)


def _random_permutation_inds(length: int, random_state: Optional[int]):
    rng = np.random.default_rng(random_state)
    inds = rng.permutation(int(length)).astype(np.int64, copy=False)
    rev_inds = np.empty(int(length), dtype=np.int64)
    rev_inds[inds] = np.arange(int(length), dtype=np.int64)
    return inds, rev_inds


def _validate_q(q: float) -> float:
    q_f = float(q)
    if q_f <= 0.0 or q_f >= 1.0:
        raise ValueError("q must be in (0, 1)")
    return q_f


def _normalize_fdr_control(fdr_control: str) -> int:
    key = str(fdr_control).strip().lower()
    if key in ("knockoff_plus", "plus", "knockoff+"):
        return 1
    if key in ("knockoff", "standard"):
        return 0
    raise ValueError("fdr_control must be one of: 'knockoff_plus', 'knockoff'")


def _normalize_knockoff_type(knockoff_type: str) -> str:
    key = str(knockoff_type).strip().lower()
    if key in ("fixed_x", "fixed-x", "fixedx"):
        return "fixed_x"
    if key in ("model_x", "model-x", "modelx"):
        return "model_x"
    raise ValueError("knockoff_type must be one of: 'fixed_x', 'model_x'")


def _standardize_design(X, xp):
    """Standardize design matrix to unit norm (L2 norm = 1 per column).

    This centers each column to zero mean and scales to unit L2 norm,
    which is the standard normalization for Fixed-X knockoff construction.

    Note: This differs from R glmnet's internal standardization (unit variance),
    but is the conventional scaling for knockoff methods as it ensures the
    knockoff construction is invariant to feature scaling.
    """
    X = xp.asarray(X, dtype=xp.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    X_centered = X - xp.mean(X, axis=0, keepdims=True)
    scale = xp.sqrt(xp.sum(X_centered * X_centered, axis=0))
    if bool(xp.any(scale <= 1e-12)):
        raise ValueError("X contains near-constant columns; knockoff construction is unstable")

    return X_centered / scale


def _standardize_features_unit_variance(X, xp):
    X_arr = xp.asarray(X, dtype=xp.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")

    n = int(X_arr.shape[0])
    if n < 2:
        raise ValueError("model-X knockoff requires at least 2 samples")

    X_centered = X_arr - xp.mean(X_arr, axis=0, keepdims=True)
    scale = xp.std(X_centered, axis=0, ddof=1)
    if bool(xp.any(scale <= 1e-12)):
        raise ValueError("X contains near-constant columns; model-X knockoff is unstable")

    return X_centered / scale


def _build_fixed_x_knockoffs(X_std, random_state: Optional[int], xp):
    n, p = int(X_std.shape[0]), int(X_std.shape[1])
    if n < 2 * p:
        raise ValueError("fixed-X knockoff requires n_samples >= 2 * n_features")

    Sigma = X_std.T @ X_std
    Sigma = 0.5 * (Sigma + Sigma.T)

    eigvals = xp.linalg.eigvalsh(Sigma)
    min_eig = _to_float_scalar(xp.min(eigvals))
    if min_eig <= 1e-10:
        raise ValueError("X'X is near-singular; fixed-X knockoff requires full-rank design")

    s_val = min(2.0 * min_eig, 1.0)
    if s_val <= 1e-12:
        raise ValueError("Failed to construct a valid knockoff S-matrix")

    # Create identity matrix on the same device as X_std (important for torch)
    # Handle numpy (no device), cupy (device attribute but different API), and torch
    if xp is np:
        S = s_val * xp.eye(p, dtype=xp.float64)
    elif getattr(xp, '__name__', '') == 'cupy':
        # CuPy: create eye on current device context (same as X_std)
        S = s_val * xp.eye(p, dtype=xp.float64)
    else:
        # Torch: use device keyword
        device = getattr(X_std, 'device', None)
        S = s_val * xp.eye(p, dtype=xp.float64, device=device)

    # For torch, use torch.linalg.solve which preserves device better
    if xp is np:
        Sigma_inv_S = xp.linalg.solve(Sigma, S)
    elif getattr(xp, '__name__', '') == 'cupy':
        # CuPy
        Sigma_inv_S = xp.linalg.solve(Sigma, S)
    else:
        # Torch: use explicit torch.linalg.solve to ensure device consistency
        import torch
        # Ensure both inputs are on the same device
        torch_device = getattr(X_std, 'device', None)
        Sigma_on_device = Sigma.to(torch_device) if hasattr(Sigma, 'to') else Sigma
        S_on_device = S.to(torch_device) if hasattr(S, 'to') else S
        Sigma_inv_S = torch.linalg.solve(Sigma_on_device, S_on_device)
        # Ensure result is on the correct device
        if torch_device is not None and hasattr(Sigma_inv_S, 'to'):
            Sigma_inv_S = Sigma_inv_S.to(torch_device)

    c_arg = 2.0 * S - S @ Sigma_inv_S
    c_arg = 0.5 * (c_arg + c_arg.T)

    c_eigvals, c_eigvecs = xp.linalg.eigh(c_arg)
    c_eigvals = xp.clip(c_eigvals, 0.0, None)
    C = c_eigvecs @ xp.diag(xp.sqrt(c_eigvals)) @ c_eigvecs.T

    # Generate random matrix A with appropriate backend
    if xp is np:
        rng = np.random.default_rng(random_state)
        A = rng.standard_normal(size=(n, p))
    else:
        # CuPy or Torch
        seed = 0 if random_state is None else int(random_state)
        try:
            # Try CuPy API
            rng = xp.random.RandomState(seed)
            A = rng.standard_normal(size=(n, p), dtype=xp.float64)
        except (AttributeError, TypeError):
            # Torch API: use manual_seed and randn
            import torch
            if xp is torch:
                if hasattr(X_std, "device"):
                    torch_device = X_std.device
                else:
                    torch_device = torch.device(_get_torch_device_str())
                gen = torch.Generator(device=torch_device)
                gen.manual_seed(seed)
                A = torch.randn(
                    n,
                    p,
                    dtype=torch.float64,
                    device=torch_device,
                    generator=gen,
                )
            else:
                # Fallback
                rng = xp.random.Generator(xp.random.PCG64(seed))
                A = rng.standard_normal(size=(n, p), dtype=xp.float64)

    # Q[:, :p] spans col(X), Q[:, p:2p] spans an orthonormal complement basis.
    Q, _ = xp.linalg.qr(xp.concatenate([X_std, A], axis=1), mode="reduced")
    U = Q[:, p : 2 * p]

    # Create identity matrix on the same device as X_std (important for torch)
    if xp is np:
        eye_matrix = xp.eye(p, dtype=xp.float64)
    elif getattr(xp, '__name__', '') == 'cupy':
        # CuPy: create eye on current device context (same as X_std)
        eye_matrix = xp.eye(p, dtype=xp.float64)
    else:
        # Torch: use device keyword
        device = getattr(X_std, 'device', None)
        eye_matrix = xp.eye(p, dtype=xp.float64, device=device)

    X_knock = X_std @ (eye_matrix - Sigma_inv_S) + U @ C
    return X_knock


def _build_model_x_knockoffs(
    X_std,
    random_state: Optional[int],
    xp,
    covariance_shrinkage: float = 0.20,
    s_scale: float = 0.999,
):
    n, p = int(X_std.shape[0]), int(X_std.shape[1])

    Sigma = (X_std.T @ X_std) / float(max(1, n - 1))
    Sigma = 0.5 * (Sigma + Sigma.T)

    shrinkage = float(min(1.0, max(0.0, covariance_shrinkage)))
    if shrinkage > 0.0:
        trace_mean = xp.trace(Sigma) / float(max(1, p))
        # Create identity matrix - numpy/cupy don't need device, torch does
        if xp is np:
            eye_matrix = xp.eye(p, dtype=xp.float64)
        elif getattr(xp, '__name__', '') == 'cupy':
            # CuPy: create eye on current device context (same as X_std)
            eye_matrix = xp.eye(p, dtype=xp.float64)
        else:
            # Torch: use device keyword
            device = getattr(X_std, 'device', None)
            eye_matrix = xp.eye(p, dtype=xp.float64, device=device)
        Sigma = (1.0 - shrinkage) * Sigma + shrinkage * trace_mean * eye_matrix
        Sigma = 0.5 * (Sigma + Sigma.T)

    eigvals = xp.linalg.eigvalsh(Sigma)
    min_eig = _to_float_scalar(xp.min(eigvals))

    ridge = 0.0
    if min_eig < 1e-6:
        ridge = float((1e-6 - min_eig) + 1e-8)
        # Create identity matrix - numpy/cupy don't need device, torch does
        if xp is np:
            eye_matrix = xp.eye(p, dtype=xp.float64)
        elif getattr(xp, '__name__', '') == 'cupy':
            # CuPy: create eye on current device context (same as X_std)
            eye_matrix = xp.eye(p, dtype=xp.float64)
        else:
            # Torch: use device keyword
            device = getattr(X_std, 'device', None)
            eye_matrix = xp.eye(p, dtype=xp.float64, device=device)
        Sigma = Sigma + ridge * eye_matrix
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigvals = xp.linalg.eigvalsh(Sigma)
        min_eig = _to_float_scalar(xp.min(eigvals))

    if min_eig <= 1e-12:
        raise ValueError("Estimated covariance is near-singular; model-X knockoff failed")

    s_val = min(2.0 * min_eig * float(s_scale), 1.0)
    if s_val <= 1e-12:
        raise ValueError("Failed to construct a valid model-X knockoff S-matrix")

    # Create identity matrix - numpy/cupy don't need device, torch does
    if xp is np:
        S = s_val * xp.eye(p, dtype=xp.float64)
    elif getattr(xp, '__name__', '') == 'cupy':
        # CuPy: create eye on current device context (same as X_std)
        S = s_val * xp.eye(p, dtype=xp.float64)
    else:
        # Torch: use device keyword
        device = getattr(X_std, 'device', None)
        S = s_val * xp.eye(p, dtype=xp.float64, device=device)

    # For torch, use explicit torch.linalg.solve to ensure device consistency
    if xp is np:
        Sigma_inv_S = xp.linalg.solve(Sigma, S)
    elif getattr(xp, '__name__', '') == 'cupy':
        # CuPy
        Sigma_inv_S = xp.linalg.solve(Sigma, S)
    else:
        # Torch: use explicit torch.linalg.solve to ensure device consistency
        import torch
        torch_device = getattr(X_std, 'device', None)
        Sigma_on_device = Sigma.to(torch_device) if hasattr(Sigma, 'to') else Sigma
        S_on_device = S.to(torch_device) if hasattr(S, 'to') else S
        Sigma_inv_S = torch.linalg.solve(Sigma_on_device, S_on_device)
        if torch_device is not None and hasattr(Sigma_inv_S, 'to'):
            Sigma_inv_S = Sigma_inv_S.to(torch_device)

    c_arg = 2.0 * S - S @ Sigma_inv_S
    c_arg = 0.5 * (c_arg + c_arg.T)
    c_eigvals, c_eigvecs = xp.linalg.eigh(c_arg)
    c_eigvals = xp.clip(c_eigvals, 0.0, None)
    C = c_eigvecs @ xp.diag(xp.sqrt(c_eigvals)) @ c_eigvecs.T

    # Generate random matrix Z with appropriate backend
    if xp is np:
        rng = np.random.default_rng(random_state)
        Z = rng.standard_normal(size=(n, p))
    else:
        # CuPy or Torch
        seed = 0 if random_state is None else int(random_state)
        try:
            # Try CuPy API
            rng = xp.random.RandomState(seed)
            Z = rng.standard_normal(size=(n, p), dtype=xp.float64)
        except (AttributeError, TypeError):
            # Torch API: use manual_seed and randn
            import torch
            if isinstance(xp, type(torch)):
                gen = torch.Generator(device=_get_torch_device_str())
                gen.manual_seed(seed)
                Z = torch.randn(n, p, dtype=torch.float64, device=_get_torch_device_str())
            else:
                # Fallback
                rng = xp.random.Generator(xp.random.PCG64(seed))
                Z = rng.standard_normal(size=(n, p), dtype=xp.float64)

    X_knock = X_std - X_std @ Sigma_inv_S + Z @ C
    return X_knock, {
        "s_value": float(s_val),
        "ridge": float(ridge),
        "min_eigenvalue": float(min_eig),
        "covariance_shrinkage": float(shrinkage),
        "s_scale": float(s_scale),
    }


def _build_model_x_knockoffs_knockpy_compat(
    X,
    random_state: Optional[int],
    *,
    modelx_shrinkage: str = "ledoitwolf",
    modelx_smatrix_method: str = "mvr",
    sample_tol: float = 1e-5,
):
    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError("X must be a 2D array")

    n, p = int(X_np.shape[0]), int(X_np.shape[1])
    if n < 2:
        raise ValueError("model-X knockoff requires at least 2 samples")

    mu = np.mean(X_np, axis=0)
    Sigma, inv_sigma, cov_estimator = _estimate_covariance_knockpy_style(
        X_np,
        shrinkage=modelx_shrinkage,
        tol=1e-4,
    )
    S, smatrix_source, smatrix_gamma = _compute_smatrix_knockpy_style(
        Sigma,
        method=modelx_smatrix_method,
        tol=float(sample_tol),
    )

    inv_sigma_S = inv_sigma @ S
    mu_k = X_np - (X_np - mu.reshape(1, -1)) @ inv_sigma_S
    Vk = 2.0 * S - S @ inv_sigma_S
    Vk = _shift_until_psd_np(Vk, tol=float(sample_tol))

    Lk = np.linalg.cholesky(Vk)
    with _temporary_numpy_seed(random_state):
        Z = np.random.randn(n, p)
    X_knock = Z @ Lk.T + mu_k

    return np.asarray(X_knock, dtype=np.float64), {
        "s_value": float(np.mean(np.diag(S))),
        "ridge": 0.0,
        "min_eigenvalue": float(_calc_mineig_np(Sigma)),
        "covariance_shrinkage": None,
        "s_scale": float(smatrix_gamma),
        "modelx_shrinkage": str(modelx_shrinkage),
        "modelx_smatrix_method": str(modelx_smatrix_method),
        "modelx_covariance_estimator": str(cov_estimator),
        "modelx_smatrix_source": str(smatrix_source),
    }


def _model_x_draw_seed(random_state: Optional[int], draw_index: int) -> Optional[int]:
    if random_state is None:
        return None
    return int(random_state) + 104729 * int(draw_index)


def _corr_diff_statistics(X_std, X_knock, y, xp):
    y_arr = xp.asarray(y, dtype=xp.float64).reshape(-1)
    if y_arr.shape[0] != X_std.shape[0]:
        raise ValueError("y must have the same number of rows as X")

    y_centered = y_arr - xp.mean(y_arr)
    score_orig = xp.abs(X_std.T @ y_centered)
    score_knock = xp.abs(X_knock.T @ y_centered)
    return score_orig - score_knock


def _ols_coef_diff_statistics(X_std, X_knock, y, xp, ridge: float = 1e-8):
    y_arr = xp.asarray(y, dtype=xp.float64).reshape(-1)
    if y_arr.shape[0] != X_std.shape[0]:
        raise ValueError("y must have the same number of rows as X")

    y_centered = y_arr - xp.mean(y_arr)
    p = int(X_std.shape[1])

    Z = xp.concatenate([X_std, X_knock], axis=1)
    ridge_f = float(max(0.0, ridge))

    if ridge_f > 0.0:
        # Create identity matrix - numpy/cupy don't need device, torch does
        if xp is np:
            eye_matrix = xp.eye(2 * p, dtype=xp.float64)
        elif getattr(xp, '__name__', '') == 'cupy':
            # CuPy: create eye on current device context (same as Z)
            eye_matrix = xp.eye(2 * p, dtype=xp.float64)
        else:
            # Torch: use device keyword
            device = getattr(Z, 'device', None)
            eye_matrix = xp.eye(2 * p, dtype=xp.float64, device=device)
        gram = Z.T @ Z + ridge_f * eye_matrix
        rhs = Z.T @ y_centered
        try:
            coef = xp.linalg.solve(gram, rhs)
        except Exception:
            coef = xp.linalg.lstsq(Z, y_centered, rcond=None)[0]
    else:
        coef = xp.linalg.lstsq(Z, y_centered, rcond=None)[0]

    coef_orig = coef[:p]
    coef_knock = coef[p:]
    return xp.abs(coef_orig) - xp.abs(coef_knock)


def _lasso_coef_diff_statistics(
    X_std,
    X_knock,
    y,
    xp,
    random_state: Optional[int] = None,
    backend_name: str = "numpy",
    max_iter: int = 3000,
    tol: float = 1e-4,
    cv_folds: int = 5,
    n_alphas: int = 12,
    lasso_cv_impl: str = "statgpu",
    lasso_fast_profile: str = "off",
    knockpy_style: bool = False,
):
    y_arr = xp.asarray(y, dtype=xp.float64).reshape(-1)
    if y_arr.shape[0] != X_std.shape[0]:
        raise ValueError("y must have the same number of rows as X")

    if bool(knockpy_style):
        y_model = y_arr
    else:
        y_model = y_arr - xp.mean(y_arr)
    p = int(X_std.shape[1])
    problem_size_full = int(X_std.shape[0]) * int(2 * p)
    fast_profile_eff = _resolve_lasso_fast_profile_for_problem(
        lasso_fast_profile,
        problem_size_full,
    )

    cv_folds_eff = max(2, int(cv_folds))
    n_alphas_eff = max(2, int(n_alphas))
    max_iter_eff = max(500, int(max_iter))
    tol_base = float(tol)

    if fast_profile_eff == "moderate":
        if problem_size_full >= 1_000_000:
            cv_folds_eff = min(cv_folds_eff, 4)
            n_alphas_eff = min(n_alphas_eff, 14 if bool(knockpy_style) else 12)
            max_iter_eff = min(max_iter_eff, 2800)
    elif fast_profile_eff == "aggressive":
        if problem_size_full >= 2_000_000:
            cv_folds_eff = min(cv_folds_eff, 2)
            n_alphas_eff = min(n_alphas_eff, 6 if bool(knockpy_style) else 5)
            max_iter_eff = min(max_iter_eff, 1600)
        else:
            cv_folds_eff = min(cv_folds_eff, 3)
            n_alphas_eff = min(n_alphas_eff, 8 if bool(knockpy_style) else 7)
            max_iter_eff = min(max_iter_eff, 2200)

    tol_eff = max(1e-3, tol_base) if bool(knockpy_style) else tol_base
    if fast_profile_eff == "aggressive":
        tol_eff = max(tol_eff, 4e-3 if problem_size_full >= 2_000_000 else 2e-3)

    lasso_diff_cache_key = _make_lasso_coef_diff_cache_key(
        X_std=X_std,
        X_knock=X_knock,
        y=y_arr,
        random_state=random_state,
        backend_name=backend_name,
        max_iter_eff=int(max_iter_eff),
        tol_eff=float(tol_eff),
        cv_folds_eff=int(cv_folds_eff),
        n_alphas_eff=int(n_alphas_eff),
        lasso_cv_impl=lasso_cv_impl,
        fast_profile_eff=fast_profile_eff,
        knockpy_style=bool(knockpy_style),
    )
    cached_w = _lasso_diff_cache_get(lasso_diff_cache_key)
    if cached_w is not None:
        return xp.asarray(cached_w, dtype=xp.float64)

    Z = xp.concatenate([X_std, X_knock], axis=1)

    # Knockpy-style symmetry preservation: permute [X, Xk] jointly, then undo at the end.
    inds, rev_inds = _random_permutation_inds(2 * p, random_state=random_state)
    alphas = np.logspace(-4.0, 4.0, base=10.0, num=int(n_alphas_eff))

    cv_impl = _normalize_lasso_cv_impl(lasso_cv_impl)

    # Force statgpu for torch backend since sklearn doesn't support torch tensors
    backend_is_torch = str(backend_name).lower() == "torch"
    if backend_is_torch and cv_impl == "sklearn":
        cv_impl = "statgpu"

    if cv_impl == "sklearn":
        try:
            from sklearn import linear_model
        except Exception:
            cv_impl = "statgpu"

    if cv_impl == "sklearn":
        Z_np = _to_numpy(Z).astype(np.float64, copy=False)
        y_np = _to_numpy(y_model).astype(np.float64, copy=False).reshape(-1)
        Z_perm = Z_np[:, inds]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = linear_model.LassoCV(
                alphas=alphas,
                cv=int(cv_folds_eff),
                verbose=False,
                max_iter=int(max_iter_eff),
                tol=float(tol_eff),
            ).fit(Z_perm, y_np)
        coef_perm = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    else:
        from ..linear_model._lasso import _fit_lasso_single_alpha_fast, _select_lasso_alpha_cv

        use_cupy_native = str(backend_name).lower() == "cupy" and _is_cupy_array(Z)
        use_torch_native = str(backend_name).lower() == "torch" and hasattr(Z, 'shape')
        if use_cupy_native:
            import cupy as cp

            inds_device = cp.asarray(inds, dtype=cp.int64)
            Z_perm = xp.asarray(Z, dtype=xp.float64)[:, inds_device]
            y_fit = xp.asarray(y_model, dtype=xp.float64).reshape(-1)
        elif use_torch_native:
            import torch
            inds_tensor = torch.tensor(inds, dtype=torch.int64, device=Z.device)
            Z_perm = Z[:, inds_tensor]
            y_fit = y_model.reshape(-1)
        else:
            Z_np = _to_numpy(Z).astype(np.float64, copy=False)
            Z_perm = Z_np[:, inds]
            y_fit = _to_numpy(y_model).astype(np.float64, copy=False).reshape(-1)

        problem_size = int(Z_perm.shape[0]) * int(Z_perm.shape[1])

        fit_intercept_eff = bool(knockpy_style)
        if random_state is None:
            alpha_cache_key = None
        else:
            alpha_cache_key = (
                "knockoff_lasso_cv_v1",
                _array_identity_token(X_std),
                _array_identity_token(X_knock),
                _array_identity_token(y_arr),
                int(random_state),
                str(backend_name).lower(),
                bool(knockpy_style),
                str(fast_profile_eff).lower(),
                int(cv_folds_eff),
                int(n_alphas_eff),
                int(max_iter_eff),
                float(tol_eff),
                _int_array_signature(inds),
            )
        alpha_select_kwargs = {
            "cv_folds": int(cv_folds_eff),
            "random_state": random_state,
            "fit_intercept": fit_intercept_eff,
            "device": "cuda" if str(backend_name).lower() in ("cupy", "torch") else "cpu",
            "max_iter": int(max_iter_eff),
            "tol": tol_eff,
            "cpu_solver": "coordinate_descent",
            "cache_key": alpha_cache_key,
        }
        if bool(knockpy_style):
            # Match knockpy-oriented branch settings used by the sklearn path as closely as possible.
            alpha_select_kwargs["alphas"] = alphas
            alpha_select_kwargs["method"] = "glmnet"
            # For large designs, reduce full KKT scan frequency to lower CV overhead.
            cd_kkt_check_every_eff = 4 if problem_size >= 1_000_000 else 2
            if fast_profile_eff == "moderate":
                cd_kkt_check_every_eff = max(cd_kkt_check_every_eff, 6)
            elif fast_profile_eff == "aggressive":
                cd_kkt_check_every_eff = max(
                    cd_kkt_check_every_eff,
                    12 if problem_size >= 2_000_000 else 8,
                )
            alpha_select_kwargs["cd_kkt_check_every"] = cd_kkt_check_every_eff
        else:
            alpha_select_kwargs["n_alphas"] = int(n_alphas_eff)

        alpha = _select_lasso_alpha_cv(
            Z_perm,
            y_fit,
            **alpha_select_kwargs,
        )

        fit_out = _fit_lasso_single_alpha_fast(
            Z_perm,
            y_fit,
            alpha=float(alpha),
            fit_intercept=fit_intercept_eff,
            max_iter=int(max_iter_eff),
            tol=tol_eff,
            device="cuda" if str(backend_name).lower() in ("cupy", "torch") else "cpu",
            stopping="coef_delta",
            cpu_solver="coordinate_descent",
            cd_kkt_check_every=int(alpha_select_kwargs.get("cd_kkt_check_every", 1)),
        )

        coef_perm = np.asarray(fit_out["coef"], dtype=np.float64).reshape(-1)
    if coef_perm.shape[0] != 2 * p:
        raise RuntimeError("lasso_coef_diff produced unexpected coefficient shape")

    coef = coef_perm[rev_inds]

    W_np = np.abs(coef[:p]) - np.abs(coef[p:])
    _lasso_diff_cache_put(lasso_diff_cache_key, W_np)
    return xp.asarray(W_np, dtype=xp.float64)


def _compute_w_statistics(
    X_std,
    X_knock,
    y,
    method: str,
    xp,
    random_state: Optional[int] = None,
    backend_name: str = "numpy",
    lasso_cv_impl: str = "statgpu",
    lasso_fast_profile: str = "off",
    lasso_knockpy_style: bool = False,
):
    key = str(method).strip().lower()
    if key == "corr_diff":
        return _corr_diff_statistics(X_std, X_knock, y, xp), "corr_diff"
    if key in ("ols_coef_diff", "ols", "coef_diff"):
        return _ols_coef_diff_statistics(X_std, X_knock, y, xp), "ols_coef_diff"
    if key in ("lasso_coef_diff", "lasso", "lasso_diff"):
        return (
            _lasso_coef_diff_statistics(
                X_std,
                X_knock,
                y,
                xp,
                random_state=random_state,
                backend_name=backend_name,
                lasso_cv_impl=lasso_cv_impl,
                lasso_fast_profile=lasso_fast_profile,
                knockpy_style=lasso_knockpy_style,
                n_alphas=20 if bool(lasso_knockpy_style) else 12,
            ),
            "lasso_coef_diff",
        )
    raise ValueError("method must be one of: 'corr_diff', 'ols_coef_diff', 'lasso_coef_diff'")


def _knockoff_threshold_and_path(W, q: float, offset: int):
    W_np = np.asarray(_to_numpy(W), dtype=np.float64).reshape(-1)
    if W_np.size == 0:
        return float(np.inf), 0.0, []

    abs_w = np.abs(W_np)
    if not np.any(abs_w > 0):
        return float(np.inf), 0.0, []

    inds = np.argsort(-abs_w, kind="stable")
    negatives = np.cumsum(W_np[inds] <= 0)
    positives = np.cumsum(W_np[inds] > 0)
    positives[positives == 0] = 1
    hat_fdrs = (negatives + int(offset)) / positives

    trajectory: List[Dict[str, float]] = []
    for rank, idx in enumerate(inds):
        trajectory.append(
            {
                "rank": int(rank + 1),
                "threshold": float(abs_w[idx]),
                "fdr_hat": float(min(1.0, hat_fdrs[rank])),
                "n_selected": int(positives[rank]),
            }
        )

    if np.any(hat_fdrs <= float(q)):
        valid = np.where(hat_fdrs <= float(q))[0]
        chosen_rank = int(valid.max())
        chosen_threshold = float(abs_w[inds[chosen_rank]])
        if chosen_threshold == 0.0:
            positive_w = W_np[W_np > 0.0]
            if positive_w.size > 0:
                chosen_threshold = float(np.min(positive_w))
            else:
                chosen_threshold = float(np.inf)
        chosen_fdr = float(min(1.0, hat_fdrs[chosen_rank]))
        return chosen_threshold, chosen_fdr, trajectory

    return float(np.inf), 0.0, trajectory
