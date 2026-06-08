"""
Unified cross-validated penalized GLM estimator.

Supports all GLM loss functions (squared_error, logistic, poisson, gamma,
inverse_gaussian, negative_binomial, tweedie) with all penalty types
(l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso).

Optimizations:
- Warm-start across alpha values (descending order)
- Batch eigendecomposition for squared_error + l2 (CPU/CuPy/Torch)
- Precomputed loss function and cached validation data per fold
- Minimal D2H transfers
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_numpy
from statgpu.linear_model._cv_base import CVEstimatorBase, kfold_indices


# ---------------------------------------------------------------------------
# Numerical constants for GLM loss computation (shared across CV paths)
# ---------------------------------------------------------------------------

# Eta clipping bounds (prevents overflow in exp/link functions)
_ETA_CLIP_STANDARD = 30.0       # Poisson, Gamma, NB, InvGauss
_ETA_CLIP_TWEEDIE = 50.0        # Tweedie (wider range for mu^p stability)
_ETA_CLIP_LOGISTIC = 500.0      # Logistic (sigmoid saturates, safe range)

# Mu clipping bounds (prevents division by zero / log(0))
_MU_LO = 1e-10                  # Standard lower bound for mu
_MU_LO_TWEEDIE = 1e-3           # Tweedie lower bound
_MU_HI_TWEEDIE = 1e4            # Tweedie upper bound
_MU_LO_INVGAUSS = 5e-2          # Inverse Gaussian lower bound
_MU_HI_INVGAUSS = 1e3           # Inverse Gaussian upper bound
_MU_LO_NB = 1e-300              # Negative binomial lower bound

# Default loss parameters (must match loss object defaults)
_NB_ALPHA_DEFAULT = 1.0         # NegativeBinomialLoss default alpha
_TWEEDIE_POWER_DEFAULT = 1.5    # TweedieLoss default power


class ApproximateCVWarning(UserWarning):
    """Warning emitted when approximate two-stage CV screening is enabled."""


def _device_to_name(device):
    if isinstance(device, Device):
        return device.value
    return str(device).lower()


def _slice_rows(arr, idx):
    """Slice rows with backend-native indices when arr lives on GPU."""
    mod = type(arr).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return arr[cp.asarray(idx)]
    if mod.startswith("torch"):
        import torch
        return arr[torch.as_tensor(idx, dtype=torch.long, device=arr.device)]
    try:
        return arr[idx]
    except TypeError:
        return np.asarray(arr)[idx]


def _nanargmin_prefer_larger_alpha(scores, alpha_grid, rel_tol=1e-10, abs_tol=1e-12):
    """Select min score with deterministic tie-break toward stronger regularization."""
    scores = np.asarray(scores, dtype=np.float64)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return int(np.nanargmin(scores))
    best = float(np.nanmin(scores))
    tol = max(float(abs_tol), abs(best) * float(rel_tol))
    candidates = np.flatnonzero(finite & (scores <= best + tol))
    return int(candidates[np.argmax(alpha_grid[candidates])])


def _two_stage_candidate_mask(scores, refine_top_k=3):
    """Return alpha candidates to strictly refine after approximate screening."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_scores = scores.size
    mask = np.zeros(n_scores, dtype=bool)
    finite = np.isfinite(scores)
    if n_scores == 0:
        return mask
    if not np.any(finite):
        mask[:] = True
        return mask

    # Endpoint alphas are common optima on flat or monotone CV curves. Always
    # refine them so approximate screening cannot drop boundary solutions.
    mask[0] = True
    mask[-1] = True

    k = min(max(1, int(refine_top_k)), int(np.count_nonzero(finite)))
    ranked = np.argsort(np.where(finite, scores, np.inf))[:k]
    for idx in ranked:
        lo = max(0, int(idx) - 1)
        hi = min(n_scores, int(idx) + 2)
        mask[lo:hi] = True

    best = float(np.nanmin(scores))
    near_tol = max(abs(best) * 0.005, 1e-6)
    mask |= finite & (scores <= best + near_tol)
    return mask


# ---------------------------------------------------------------------------
# Per-sample loss/residual functions (numpy-only, for _evaluate_loss_numpy)
# ---------------------------------------------------------------------------
# These functions define the canonical per-sample loss formulas.
# The inline code in _glm_sparse_cv_folds uses the same mathematical formulas
# with torch/cupy operations for performance. When modifying a formula,
# update BOTH this section AND the inline code in _glm_sparse_cv_folds.

def _ps_logistic(eta, y, **_):
    log1pexp = np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0.0)
    return -y * eta + log1pexp

def _ps_squared_error(eta, y, X_design=None, coef_with_intercept=None, **_):
    return (y - X_design @ coef_with_intercept) ** 2

def _ps_poisson(eta, y, **_):
    # loss = mu - y*log(mu), where mu = exp(eta)
    mu = np.exp(np.clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    return mu - y * np.log(np.clip(mu, _MU_LO, None))

def _ps_gamma(eta, y, **_):
    # loss = y/mu + log(mu), where mu = exp(eta)
    mu = np.exp(np.clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    return y / np.clip(mu, _MU_LO, None) + np.log(np.clip(mu, _MU_LO, None))

def _ps_inverse_gaussian(eta, y, **_):
    # loss = y/(2*mu^2) - 1/mu, where mu = exp(eta)
    mu = np.exp(np.clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    return y / (2.0 * np.clip(mu * mu, _MU_LO, None)) - 1.0 / np.clip(mu, _MU_LO, None)

def _ps_negative_binomial(eta, y, alpha=_NB_ALPHA_DEFAULT, **_):
    # loss = -y*log(mu/(1+alpha*mu)) + (1/alpha)*log(1+alpha*mu)
    mu = np.exp(np.clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = np.clip(mu, _MU_LO, None)
    one_plus = 1.0 + alpha * mu_c
    return -y * np.log(mu_c / one_plus) + (1.0 / alpha) * np.log(one_plus)

def _ps_tweedie(eta, y, power=_TWEEDIE_POWER_DEFAULT, **_):
    # loss = -y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p)
    mu = np.exp(np.clip(eta, -_ETA_CLIP_TWEEDIE, _ETA_CLIP_TWEEDIE))
    mu_c = np.clip(mu, _MU_LO_TWEEDIE, _MU_HI_TWEEDIE)
    return -y * np.exp((1 - power) * np.log(mu_c)) / (1 - power) + np.exp((2 - power) * np.log(mu_c)) / (2 - power)


# loss_name -> (per_sample_fn, uses_design)
# uses_design=True: fn needs X_design and coef_with_intercept (squared_error)
# uses_design=False: fn uses eta directly (all GLM losses)
_LOSS_EVAL_DISPATCH = {
    "logistic": (_ps_logistic, False),
    "squared_error": (_ps_squared_error, True),
    "poisson": (_ps_poisson, False),
    "gamma": (_ps_gamma, False),
    "inverse_gaussian": (_ps_inverse_gaussian, False),
    "negative_binomial": (_ps_negative_binomial, False),
    "tweedie": (_ps_tweedie, False),
}


def _weighted_mean(per_sample, sw):
    """Compute weighted or unweighted mean of per-sample values."""
    if sw is not None:
        return float(np.sum(sw * per_sample) / np.sum(sw))
    return float(np.mean(per_sample))


def _evaluate_loss_numpy(loss_name, loss_fn, X_val_np, y_val_np, coef_np, intercept, fit_intercept, sample_weight=None):
    """Backend-independent validation loss in float64 numpy.

    When sample_weight is provided, returns weighted mean loss.
    """
    coef_np = np.asarray(coef_np, dtype=np.float64).ravel()
    sw = np.asarray(sample_weight, dtype=np.float64).ravel() if sample_weight is not None else None

    entry = _LOSS_EVAL_DISPATCH.get(loss_name)
    if entry is not None:
        # Resolve loss-specific parameters from loss_fn object
        _loss_params = {}
        if loss_name == "negative_binomial":
            _loss_params["alpha"] = float(getattr(loss_fn, "alpha", _NB_ALPHA_DEFAULT))
        elif loss_name == "tweedie":
            _loss_params["power"] = float(getattr(loss_fn, "power", _TWEEDIE_POWER_DEFAULT))

        per_sample_fn, uses_design = entry
        if uses_design:
            n_val = X_val_np.shape[0]
            if fit_intercept:
                X_design = np.column_stack([np.ones(n_val), X_val_np])
                coef_with_intercept = np.concatenate([[float(intercept)], coef_np])
            else:
                X_design = X_val_np
                coef_with_intercept = coef_np
            eta = X_val_np @ coef_np + (float(intercept) if fit_intercept else 0.0)
            per_sample = per_sample_fn(eta, y_val_np, X_design=X_design, coef_with_intercept=coef_with_intercept, **_loss_params)
        else:
            eta = X_val_np @ coef_np + (float(intercept) if fit_intercept else 0.0)
            per_sample = per_sample_fn(eta, y_val_np, **_loss_params)
        return _weighted_mean(per_sample, sw)

    # Fallback for unknown loss types
    n_val = X_val_np.shape[0]
    if fit_intercept:
        X_design = np.column_stack([np.ones(n_val), X_val_np])
        coef_with_intercept = np.concatenate([[float(intercept)], coef_np])
    else:
        X_design = X_val_np
        coef_with_intercept = coef_np
    # Fallback: unweighted loss. Weighted mean cannot be derived from
    # unweighted mean, so weights are ignored for unknown loss types.
    if sw is not None:
        import warnings
        warnings.warn(
            f"_evaluate_loss_numpy: loss '{loss_name}' not in dispatch table, "
            f"falling back to unweighted loss_fn.value(). Sample weights ignored.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(loss_fn.value(X_design, y_val_np, coef_with_intercept))


def _ridge_eig_batch(X_train_np, y_train_np, X_val_np, y_val_np, alphas_np):
    """Batch Ridge solve via eigendecomposition on numpy.

    Returns (mse_array, coefs_matrix, intercepts_array).
    All computation in float64 numpy for maximum precision.
    """
    n, p = X_train_np.shape
    n_alphas = len(alphas_np)

    X_mean = np.mean(X_train_np, axis=0)
    y_mean = np.mean(y_train_np)
    Xc = X_train_np - X_mean
    yc = y_train_np - y_mean

    XtX = Xc.T @ Xc
    eigvals, Q = np.linalg.eigh(XtX)
    eigvals = np.maximum(eigvals, 1e-15)

    QtXty = Q.T @ (Xc.T @ yc)
    n_alpha = n * alphas_np
    inv_diag = 1.0 / (eigvals[:, None] + n_alpha[None, :])
    coefs = Q @ (inv_diag * QtXty[:, None])
    intercepts = y_mean - X_mean @ coefs

    # Predict: X_val @ coef + intercept (intercept already includes -X_mean @ coef)
    y_pred = X_val_np @ coefs + intercepts[None, :]
    mse = np.mean((y_val_np[:, None] - y_pred) ** 2, axis=0)

    return mse, coefs, intercepts


def _ridge_eig_single(X_train_np, y_train_np, alpha, sample_weight=None):
    """Single Ridge solve via eigendecomposition. Returns (coef, intercept).

    When sample_weight is provided, uses weighted centering and weighted
    normal equations: X'WX coef = X'Wy, solved via eigendecomposition of
    X'WX. Same O(p³) cost as unweighted path.
    """
    n, p = X_train_np.shape
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        w_sum = w.sum()
        X_mean = np.average(X_train_np, axis=0, weights=w)
        y_mean = float(np.average(y_train_np, weights=w))
        Xc = X_train_np - X_mean
        yc = y_train_np - y_mean
        # Weighted normal equations: Xc' diag(w) Xc
        W_sqrt_Xc = Xc * np.sqrt(w)[:, None]
        XtWX = W_sqrt_Xc.T @ W_sqrt_Xc
        XtWy = (Xc * w[:, None]).T @ yc
        eigvals, Q = np.linalg.eigh(XtWX)
        eigvals = np.maximum(eigvals, 1e-15)
        QtXtWy = Q.T @ XtWy
        inv_diag = 1.0 / (eigvals + w_sum * alpha)
        coef = Q @ (inv_diag * QtXtWy)
        intercept = float(y_mean - X_mean @ coef)
        return coef, intercept
    X_mean = np.mean(X_train_np, axis=0)
    y_mean = np.mean(y_train_np)
    Xc = X_train_np - X_mean
    yc = y_train_np - y_mean

    XtX = Xc.T @ Xc
    eigvals, Q = np.linalg.eigh(XtX)
    eigvals = np.maximum(eigvals, 1e-15)

    QtXty = Q.T @ (Xc.T @ yc)
    inv_diag = 1.0 / (eigvals + n * alpha)
    coef = Q @ (inv_diag * QtXty)
    intercept = float(y_mean - X_mean @ coef)
    return coef, intercept


def _backend_name_for_cv_device(device):
    name = _device_to_name(device)
    if name == "cuda":
        return "cupy"
    if name == "torch":
        return "torch"
    return "numpy"


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _logistic_sparse_effective_max_iter(max_iter, device, penalty_name, refit=False):
    backend = _backend_name_for_cv_device(device)
    penalty_name = str(penalty_name).lower()
    if backend in ("cupy", "torch") and not refit:
        if penalty_name == "l1":
            return min(int(max_iter), 400)
        if penalty_name in ("elasticnet", "en"):
            return min(int(max_iter), 600)
    return int(max_iter)


def _glm_cv_effective_max_iter(max_iter, loss_name, penalty_name, device, refit=False):
    """CV-only iteration caps for GPU paths whose alpha ranking stabilizes early."""
    backend = _backend_name_for_cv_device(device)
    loss_name = str(loss_name).lower()
    penalty_name = str(penalty_name).lower()
    if backend in ("cupy", "torch") and not refit:
        if loss_name == "tweedie" and penalty_name in ("l1", "elasticnet", "en"):
            return min(int(max_iter), 200)
    if backend == "cupy" and not refit:
        if loss_name == "negative_binomial" and penalty_name == "l2":
            return min(int(max_iter), 30)
    return int(max_iter)


def _to_backend_float64(arr, backend):
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(arr, dtype=cp.float64)
    if backend == "torch":
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=torch.float64, device="cuda")
        return torch.as_tensor(np.asarray(arr, dtype=np.float64), dtype=torch.float64, device="cuda")
    return np.asarray(arr, dtype=np.float64)


def _stable_sigmoid(x, backend):
    if backend == "torch":
        import torch
        return torch.sigmoid(torch.clamp(x, -500.0, 500.0))
    if backend == "cupy":
        import cupy as cp
        return 1.0 / (1.0 + cp.exp(-cp.clip(x, -500.0, 500.0)))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _softplus(x, backend):
    if backend == "torch":
        import torch
        return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp(x, min=0.0)
    if backend == "cupy":
        import cupy as cp
        return cp.log1p(cp.exp(-cp.abs(x))) + cp.maximum(x, 0.0)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Unified fold-batched CV framework
# ---------------------------------------------------------------------------

def _fb_ones(shape, dtype, is_torch, device=None):
    """Create ones tensor on the appropriate backend."""
    if is_torch:
        import torch
        return torch.ones(shape, dtype=dtype, device=device)
    import cupy as cp
    return cp.ones(shape, dtype=dtype)


def _fb_zeros(shape, dtype, is_torch, device=None):
    """Create zeros tensor on the appropriate backend."""
    if is_torch:
        import torch
        return torch.zeros(shape, dtype=dtype, device=device)
    import cupy as cp
    return cp.zeros(shape, dtype=dtype)


def _fb_as_tensor(arr, is_torch, device=None):
    """Convert numpy array to int64 backend tensor (for index arrays)."""
    arr_i64 = np.asarray(arr, dtype=np.int64)
    if is_torch:
        import torch
        return torch.as_tensor(arr_i64, dtype=torch.long, device=device)
    import cupy as cp
    return cp.asarray(arr_i64)


def _fb_copy(x, is_torch):
    """Copy a backend tensor."""
    return x.clone() if is_torch else x.copy()


def _fb_cat(tensors, is_torch, dim=1):
    """Concatenate tensors along dim."""
    if is_torch:
        import torch
        return torch.cat(tensors, dim=dim)
    import cupy as cp
    return cp.concatenate(tensors, axis=dim)


def _fb_sum(x, is_torch, axis=0, keepdims=False):
    """Sum along axis."""
    if is_torch:
        return x.sum(dim=axis, keepdim=keepdims)
    return x.sum(axis=axis, keepdims=keepdims)


def _fb_stack(arrays, is_torch, dim=1):
    """Stack arrays along dim."""
    if is_torch:
        import torch
        return torch.stack(arrays, dim=dim)
    import cupy as cp
    return cp.stack(arrays, axis=dim)


def _fold_batch_lipschitz_logistic(X_aug, y_train, n_train, is_torch):
    if is_torch:
        import torch
        eig_max = float(torch.linalg.eigvalsh(X_aug.T @ X_aug).max().item())
    else:
        import cupy as cp
        eig_max = float(cp.linalg.eigvalsh(X_aug.T @ X_aug).max())
    return max(eig_max / (4.0 * max(n_train, 1)), 1e-12)


def _fold_batch_lipschitz_exp_link(X_aug, y_train, n_train, is_torch):
    """Lipschitz for log-link GLMs (Poisson, Gamma, NB, InvGauss, Tweedie).
    Uses y-scaling: max(1, y_mean, sqrt(y_mean * y_max))."""
    if is_torch:
        import torch
        eig_max = float(torch.linalg.eigvalsh(X_aug.T @ X_aug).max().item())
        y_mean = float(y_train.mean().item())
        y_max = float(y_train.max().item())
    else:
        import cupy as cp
        eig_max = float(cp.linalg.eigvalsh(X_aug.T @ X_aug).max())
        y_mean = float(y_train.mean())
        y_max = float(y_train.max())
    y_scale = max(1.0, y_mean, np.sqrt(y_mean * max(y_max, 1e-10)))
    return max(eig_max / max(n_train, 1), 1e-12) * y_scale


def _fold_batch_lipschitz_gamma(X_aug, y_train, n_train, is_torch):
    """Lipschitz for Gamma log-link: eig_max(X'X)/n * max(y/y_mean).

    Differs from _fold_batch_lipschitz_exp_link because Gamma's Hessian
    weights are y/mu (not mu), so scaling uses y-ratio instead of y-moment.
    """
    if is_torch:
        import torch
        eig_max = float(torch.linalg.eigvalsh(X_aug.T @ X_aug).max().item())
        y_mean = float(y_train.mean().item())
        y_ratio_max = float((y_train / y_mean).max().item()) if y_mean > 0 else 1.0
    else:
        import cupy as cp
        eig_max = float(cp.linalg.eigvalsh(X_aug.T @ X_aug).max())
        y_mean = float(y_train.mean())
        y_ratio_max = float((y_train / y_mean).max()) if y_mean > 0 else 1.0
    return max(eig_max / max(n_train, 1), 1e-12) * max(1.0, y_ratio_max)


# Loss-specific configs: lipschitz_fn and intercept_fn only.
# ---------------------------------------------------------------------------
# Loss formula registry — single source of truth for residual and val_loss
# ---------------------------------------------------------------------------
# Each loss registers (residual_fn, val_loss_fn) that work with any backend
# (numpy/torch/cupy) via elementwise ops. The FISTA hot loop calls these
# instead of inline if/elif chains, eliminating formula duplication.
#
# Signature: fn(eta, y, **params) -> per_sample_loss_or_residual
# `eta` and `y` are backend arrays; `params` carries loss-specific scalars.

_LOSS_RESIDUAL_FNS = {}
_LOSS_VALLOSS_FNS = {}

def _register_loss_fns(loss_name, residual_fn, val_loss_fn, params=None):
    _LOSS_RESIDUAL_FNS[loss_name] = residual_fn
    _LOSS_VALLOSS_FNS[loss_name] = val_loss_fn

# --- Logistic ---
def _res_logistic(eta, y, **_):
    # Gradient of logistic loss: sigmoid(eta) - y
    xp = _get_xp(eta)
    sig = 1.0 / (1.0 + xp.exp(-_safe_clip(eta, -_ETA_CLIP_LOGISTIC, _ETA_CLIP_LOGISTIC)))
    return sig - y

def _val_logistic(eta, y, **_):
    # Logistic loss: -y*eta + softplus(eta)
    xp = _get_xp(eta)
    log1pexp = xp.log1p(xp.exp(-xp.abs(eta))) + xp.maximum(eta, 0.0)
    return -y * eta + log1pexp

# --- Poisson ---
def _res_poisson(eta, y, **_):
    # Gradient of Poisson loss: d/deta [mu - y*log(mu)] = mu - y
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = _safe_clip(mu, _MU_LO)
    return mu_c - y

def _val_poisson(eta, y, **_):
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = _safe_clip(mu, _MU_LO)
    return mu_c - y * xp.log(mu_c)

# --- Gamma ---
def _res_gamma(eta, y, **_):
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = _safe_clip(mu, _MU_LO)
    return 1.0 - y / mu_c

def _val_gamma(eta, y, **_):
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = _safe_clip(mu, _MU_LO)
    return y / mu_c + xp.log(mu_c)

# --- Inverse Gaussian ---
def _res_invgauss(eta, y, **_):
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = _safe_clip(mu, _MU_LO)
    return (mu_c - y) / (mu_c * mu_c)

def _val_invgauss(eta, y, **_):
    # Inverse Gaussian loss: y/(2*mu^2) - 1/mu
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    # Clip mu^2 (not mu) to match _ps_inverse_gaussian: denom >= 2e-10
    mu_sq_c = _safe_clip(mu * mu, _MU_LO)
    mu_c = _safe_clip(mu, _MU_LO)
    return y / (2.0 * mu_sq_c) - 1.0 / mu_c

# --- Negative Binomial ---
def _res_nb(eta, y, alpha=_NB_ALPHA_DEFAULT, **_):
    # Gradient of NB loss: d/deta L = (mu - y) / (1 + alpha*mu)
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = _safe_clip(mu, _MU_LO)
    return (mu_c - y) / (1.0 + alpha * mu_c)

def _val_nb(eta, y, alpha=_NB_ALPHA_DEFAULT, **_):
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_STANDARD, _ETA_CLIP_STANDARD))
    mu_c = _safe_clip(mu, _MU_LO)
    one_plus = 1.0 + alpha * mu_c
    return -y * xp.log(mu_c / one_plus) + (1.0 / alpha) * xp.log(one_plus)

# --- Tweedie ---
def _res_tweedie(eta, y, power=_TWEEDIE_POWER_DEFAULT, **_):
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_TWEEDIE, _ETA_CLIP_TWEEDIE))
    mu_c = _safe_clip(mu, _MU_LO_TWEEDIE, _MU_HI_TWEEDIE)
    return xp.exp((1 - power) * xp.log(mu_c)) * (mu_c - y)

def _val_tweedie(eta, y, power=_TWEEDIE_POWER_DEFAULT, **_):
    # Tweedie loss: -y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p)
    # Handle boundary: power=1 (Poisson) and power=2 (Gamma) use log form.
    xp = _get_xp(eta)
    mu = xp.exp(_safe_clip(eta, -_ETA_CLIP_TWEEDIE, _ETA_CLIP_TWEEDIE))
    mu_c = _safe_clip(mu, _MU_LO_TWEEDIE, _MU_HI_TWEEDIE)
    log_mu = xp.log(mu_c)
    d1 = 1.0 - power
    d2 = 2.0 - power
    term1 = -y * xp.exp(d1 * log_mu) / d1 if abs(d1) > 1e-10 else -y * log_mu
    term2 = xp.exp(d2 * log_mu) / d2 if abs(d2) > 1e-10 else log_mu
    return term1 + term2


# Use backend-agnostic utilities from statgpu.backends._array_ops
from statgpu.backends._array_ops import _clip as _safe_clip, _xp as _get_xp


_register_loss_fns("logistic", _res_logistic, _val_logistic)
_register_loss_fns("poisson", _res_poisson, _val_poisson)
_register_loss_fns("gamma", _res_gamma, _val_gamma)
_register_loss_fns("inverse_gaussian", _res_invgauss, _val_invgauss)
_register_loss_fns("negative_binomial", _res_nb, _val_nb)
_register_loss_fns("tweedie", _res_tweedie, _val_tweedie)


# Legacy fold-batch config (Lipschitz + intercept only; residual/val_loss now use registry)
_FOLD_BATCH_CONFIGS = {}


def _logistic_intercept(y_mean, is_torch):
    if is_torch:
        import torch
        y_prob = torch.clamp(y_mean, min=1e-3, max=0.999)
        return torch.log(y_prob) - torch.log(1.0 - y_prob)
    else:
        import cupy as cp
        y_prob = cp.clip(y_mean, 1e-3, 0.999)
        return cp.log(y_prob) - cp.log(1.0 - y_prob)


def _exp_link_intercept(y_mean, is_torch):
    """Intercept for log-link GLMs: log(clamp(y_mean, 1e-3, 100))."""
    if is_torch:
        import torch
        return torch.log(torch.clamp(y_mean, min=1e-3, max=100.0))
    else:
        import cupy as cp
        return cp.log(cp.clip(y_mean, 1e-3, 100.0))


def _register_fold_batch(loss_name, lipschitz_fn, intercept_fn):
    _FOLD_BATCH_CONFIGS[loss_name] = {
        "lipschitz_fn": lipschitz_fn,
        "intercept_fn": intercept_fn,
    }


_register_fold_batch("logistic", _fold_batch_lipschitz_logistic, _logistic_intercept)
_register_fold_batch("poisson", _fold_batch_lipschitz_exp_link, _exp_link_intercept)
_register_fold_batch("gamma", _fold_batch_lipschitz_gamma, _exp_link_intercept)
_register_fold_batch("inverse_gaussian", _fold_batch_lipschitz_exp_link, _exp_link_intercept)
_register_fold_batch("negative_binomial", _fold_batch_lipschitz_exp_link, _exp_link_intercept)
_register_fold_batch("tweedie", _fold_batch_lipschitz_exp_link, _exp_link_intercept)


def _glm_sparse_cv_folds(
    X,
    y,
    folds,
    alpha_sorted,
    penalty_name,
    l1_ratio,
    max_iter,
    tol,
    loss_name,
    device_backend,
    sample_weight=None,
):
    """Unified fold-batched sparse GLM CV path for all losses and backends.

    Uses direct torch/cupy API calls (no abstraction layer) for performance
    in the FISTA hot loop.
    """
    cfg = _FOLD_BATCH_CONFIGS.get(loss_name)
    if cfg is None:
        return None

    # Resolve loss-specific parameters from the loss object defaults.
    # This ensures inline code stays in sync with loss class defaults.
    from statgpu.linear_model._penalized import _resolve_loss_name
    _loss_obj = _resolve_loss_name(loss_name)
    _nb_alpha = float(getattr(_loss_obj, 'alpha', _NB_ALPHA_DEFAULT))
    _tw_power = float(getattr(_loss_obj, 'power', _TWEEDIE_POWER_DEFAULT))

    is_torch = (device_backend == "torch")
    if is_torch:
        if _backend_name_for_cv_device("torch") != "torch":
            return None
        try:
            import torch
            if not torch.cuda.is_available():
                return None
        except (ImportError, RuntimeError, OSError):
            return None
    else:
        if _backend_name_for_cv_device("cuda") != "cupy":
            return None
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() <= 0:
                return None
        except (ImportError, RuntimeError, OSError):
            return None

    Xb = _to_backend_float64(X, device_backend)
    yb = _to_backend_float64(y, device_backend).reshape(-1)
    alphas = np.asarray(alpha_sorted, dtype=np.float64).ravel()
    penalty_name = str(penalty_name).lower()
    is_enet = penalty_name in ("elasticnet", "en")
    n_samples, n_features = Xb.shape
    n_folds = len(folds)
    if n_folds < 2 or alphas.size == 0:
        return None

    lipschitz_fn = cfg["lipschitz_fn"]
    intercept_fn = cfg["intercept_fn"]

    # --- Build masks and compute per-fold Lipschitz ---
    dev = Xb.device if is_torch else None
    train_mask = _fb_ones((n_samples, n_folds), Xb.dtype, is_torch, dev)
    val_mask = _fb_zeros((n_samples, n_folds), Xb.dtype, is_torch, dev)

    # Sample weight mask: per-fold weights (n_samples, n_folds)
    has_weights = sample_weight is not None
    if has_weights:
        sw_all = _to_backend_float64(sample_weight, device_backend).reshape(-1)
        sw_mask = sw_all.reshape(-1, 1) * train_mask  # (n_samples, n_folds)
    else:
        sw_mask = train_mask  # effectively all 1s for train, 0s for val

    step_values = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_idx_dev = _fb_as_tensor(train_idx, is_torch, dev)
        val_idx_dev = _fb_as_tensor(val_idx, is_torch, dev)
        train_mask[val_idx_dev, fold_idx] = 0.0
        val_mask[val_idx_dev, fold_idx] = 1.0

        X_train = Xb[train_idx_dev]
        y_train = yb[train_idx_dev]
        ones = _fb_ones((X_train.shape[0], 1), Xb.dtype, is_torch, dev)
        X_aug = _fb_cat([X_train, ones], is_torch)
        n_train = int(X_train.shape[0])
        # For weighted Lipschitz, pass sum(w) as effective n for normalization
        if has_weights:
            sw_fold = sw_all[train_idx_dev]
            sw_col_fold = sw_fold.reshape(-1, 1)
            Xw = X_aug * sw_col_fold.sqrt() if is_torch else X_aug * cp.sqrt(sw_col_fold)
            w_sum_fold = float(sw_fold.sum().item()) if is_torch else float(sw_fold.sum())
            L_loss = lipschitz_fn(Xw, y_train, max(w_sum_fold, 1.0), is_torch)
        else:
            L_loss = lipschitz_fn(X_aug, y_train, n_train, is_torch)
        step_values.append(1.0 / L_loss)

    # Update sw_mask after train_mask has been modified
    if has_weights:
        sw_mask = sw_all.reshape(-1, 1) * train_mask

    # --- Initialize parameters ---
    sw_train_vec = _fb_sum(sw_mask, is_torch, axis=0, keepdims=True).reshape(1, n_folds)
    n_val_vec = _fb_sum(val_mask, is_torch, axis=0, keepdims=True).reshape(1, n_folds)
    y_col = yb.reshape(-1, 1)
    # Weighted mean of y per fold
    y_mean = _fb_sum(y_col * sw_mask, is_torch, axis=0, keepdims=True) / sw_train_vec
    intercept = intercept_fn(y_mean, is_torch).reshape(1, n_folds)
    coef = _fb_zeros((n_features, n_folds), Xb.dtype, is_torch, dev)
    if is_torch:
        import torch
        step = torch.as_tensor(step_values, dtype=Xb.dtype, device=dev).reshape(1, n_folds)
    else:
        import cupy as cp
        step = cp.asarray(step_values, dtype=Xb.dtype).reshape(1, n_folds)

    tol_float = float(tol)
    scores_path = []
    iters_path = []

    # --- FISTA loop over alphas ---
    # y_coef / y_intercept are the extrapolated iterates (standard FISTA notation).
    for alpha in alphas:
        y_coef = _fb_copy(coef, is_torch)
        y_intercept = _fb_copy(intercept, is_torch)
        t_k = 1.0
        if is_torch:
            active = torch.ones((1, n_folds), dtype=torch.bool, device=Xb.device)
            last_iter = torch.zeros((n_folds,), dtype=torch.int64, device=Xb.device)
        else:
            active = cp.ones((1, n_folds), dtype=bool)
            last_iter = cp.zeros((n_folds,), dtype=cp.int64)

        for iteration in range(int(max_iter)):
            coef_old = _fb_copy(coef, is_torch)
            intercept_old = _fb_copy(intercept, is_torch)

            eta = Xb @ y_coef + y_intercept
            # Compute per-sample residual via loss registry.
            # Each loss defines a backend-agnostic residual function.
            resid = _LOSS_RESIDUAL_FNS[loss_name](eta, y_col, alpha=_nb_alpha, power=_tw_power) * train_mask
            # Weighted gradient: multiply residual by sw_mask (includes train_mask)
            # and divide by sum of weights per fold
            grad_coef = (Xb.T @ (resid * sw_mask)) / sw_train_vec
            grad_intercept = _fb_sum(resid * sw_mask, is_torch, axis=0, keepdims=True) / sw_train_vec

            w = y_coef - step * grad_coef
            if is_enet:
                thresh = float(alpha) * float(l1_ratio) * step
                denom = 1.0 + float(alpha) * (1.0 - float(l1_ratio)) * step
            else:
                thresh = float(alpha) * step
                denom = 1.0
            if is_torch:
                coef_new = torch.sign(w) * torch.clamp(torch.abs(w) - thresh, min=0.0) / denom
            else:
                coef_new = cp.sign(w) * cp.maximum(cp.abs(w) - thresh, 0.0) / denom
            intercept_new = y_intercept - step * grad_intercept

            coef = torch.where(active, coef_new, coef) if is_torch else cp.where(active, coef_new, coef)
            intercept = torch.where(active, intercept_new, intercept) if is_torch else cp.where(active, intercept_new, intercept)

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = min((t_k - 1.0) / t_new, 0.5)
            y_coef_new = coef + beta * (coef - coef_old)
            y_intercept_new = intercept + beta * (intercept - intercept_old)
            y_coef = torch.where(active, y_coef_new, coef) if is_torch else cp.where(active, y_coef_new, coef)
            y_intercept = torch.where(active, y_intercept_new, intercept) if is_torch else cp.where(active, y_intercept_new, intercept)
            t_k = t_new
            if is_torch:
                last_iter = torch.where(active.reshape(-1), torch.full_like(last_iter, iteration + 1), last_iter)
            else:
                last_iter = cp.where(active.reshape(-1), cp.full_like(last_iter, iteration + 1), last_iter)

            # Check convergence: every iteration for first 20, then every 50
            if iteration < 20 or iteration % 50 == 0:
                if is_torch:
                    delta = torch.sum(torch.abs(coef - coef_old), dim=0, keepdim=True) + torch.abs(intercept - intercept_old)
                else:
                    delta = cp.sum(cp.abs(coef - coef_old), axis=0, keepdims=True) + cp.abs(intercept - intercept_old)
                active = active & (delta >= tol_float)
                if is_torch:
                    if not bool(torch.any(active).item()):
                        break
                else:
                    if not bool(cp.any(active)):
                        break

        # Validation loss via loss registry (single call, backend-agnostic)
        eta_val = Xb @ coef + intercept
        val_loss = _LOSS_VALLOSS_FNS[loss_name](eta_val, y_col, alpha=_nb_alpha, power=_tw_power) * val_mask
        # Weighted validation scoring: use sw_val_mask when weights present
        if has_weights:
            sw_val_mask = sw_all.reshape(-1, 1) * val_mask
            sw_val_vec = _fb_sum(sw_val_mask, is_torch, axis=0, keepdims=True).reshape(1, n_folds)
            scores_path.append(_fb_sum(val_loss * sw_val_mask, is_torch, axis=0, keepdims=True).reshape(-1) / sw_val_vec.reshape(-1))
        else:
            scores_path.append(_fb_sum(val_loss, is_torch, axis=0, keepdims=True).reshape(-1) / n_val_vec.reshape(-1))
        iters_path.append(last_iter)

    scores = _fb_stack(scores_path, is_torch)
    n_iter = _fb_stack(iters_path, is_torch)
    return {
        "scores": np.asarray(_to_numpy(scores), dtype=np.float64),
        "n_iter": np.asarray(_to_numpy(n_iter), dtype=np.int64),
    }


def _scalar_to_float(x):
    return float(_to_numpy(x))


def _logistic_sparse_cv_path(
    X_train,
    y_train,
    alpha_sorted,
    penalty_name,
    l1_ratio,
    max_iter,
    tol,
    device,
    X_val=None,
    y_val=None,
    sample_weight=None,
    val_sample_weight=None,
    return_path=True,
):
    """Fit a logistic sparse alpha path and optionally score validation loss.

    This CV-only path uses a fixed global Lipschitz bound and sparse proximal
    updates, avoiding per-iteration Armijo/objective synchronizations.

    Parameters
    ----------
    val_sample_weight : array-like, optional
        Per-sample weights for validation scoring. When provided, validation
        loss is computed as weighted mean.
    """
    if sample_weight is not None:
        sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
        if sw_np.size and not np.allclose(sw_np, sw_np[0]):
            return None

    backend = _backend_name_for_cv_device(device)
    Xb = _to_backend_float64(X_train, backend)
    yb = _to_backend_float64(y_train, backend).reshape(-1)
    alphas = np.asarray(alpha_sorted, dtype=np.float64).ravel()
    n_samples, n_features = Xb.shape

    if backend == "torch":
        import torch
        xp = torch
        ones = torch.ones((n_samples, 1), dtype=Xb.dtype, device=Xb.device)
        X_aug = torch.cat([Xb, ones], dim=1)
        eig_max = float(torch.linalg.eigvalsh(X_aug.T @ X_aug).max().item())
        y_mean = float(torch.mean(yb).item())
        coef = torch.zeros(n_features, dtype=Xb.dtype, device=Xb.device)
        intercept = torch.tensor(
            np.log(np.clip(y_mean, 1e-3, 1.0 - 1e-3) / (1.0 - np.clip(y_mean, 1e-3, 1.0 - 1e-3))),
            dtype=Xb.dtype,
            device=Xb.device,
        )
    elif backend == "cupy":
        import cupy as cp
        xp = cp
        ones = cp.ones((n_samples, 1), dtype=Xb.dtype)
        X_aug = cp.concatenate([Xb, ones], axis=1)
        eig_max = float(cp.linalg.eigvalsh(X_aug.T @ X_aug).max())
        y_mean = float(cp.mean(yb))
        coef = cp.zeros(n_features, dtype=Xb.dtype)
        intercept = cp.asarray(
            np.log(np.clip(y_mean, 1e-3, 1.0 - 1e-3) / (1.0 - np.clip(y_mean, 1e-3, 1.0 - 1e-3))),
            dtype=Xb.dtype,
        )
    else:
        xp = np
        ones = np.ones((n_samples, 1), dtype=Xb.dtype)
        X_aug = np.concatenate([Xb, ones], axis=1)
        eig_max = float(np.linalg.eigvalsh(X_aug.T @ X_aug).max())
        y_mean = float(np.mean(yb))
        coef = np.zeros(n_features, dtype=np.float64)
        intercept = float(np.log(np.clip(y_mean, 1e-3, 1.0 - 1e-3) / (1.0 - np.clip(y_mean, 1e-3, 1.0 - 1e-3))))

    L_loss = max(eig_max / (4.0 * max(int(n_samples), 1)), 1e-12)
    step = 1.0 / L_loss
    conv_interval = 10 if backend == "numpy" else 50
    penalty_name = str(penalty_name).lower()
    is_enet = penalty_name in ("elasticnet", "en")

    if X_val is not None and y_val is not None:
        Xv = _to_backend_float64(X_val, backend)
        yv = _to_backend_float64(y_val, backend).reshape(-1)
        swv = _to_backend_float64(val_sample_weight, backend).reshape(-1) if val_sample_weight is not None else None
    else:
        Xv = yv = swv = None

    scores = []
    score_coef_path = []
    score_intercept_path = []
    coef_path = []
    intercept_path = []
    iters = []

    for alpha in alphas:
        y_coef = coef.copy() if backend != "torch" else coef.clone()
        y_intercept = intercept.clone() if backend == "torch" else intercept.copy() if backend == "cupy" else float(intercept)
        t_k = 1.0
        last_iter = 0
        for iteration in range(int(max_iter)):
            coef_old = coef.copy() if backend != "torch" else coef.clone()
            intercept_old = intercept.clone() if backend == "torch" else intercept.copy() if backend == "cupy" else float(intercept)

            eta = Xb @ y_coef + y_intercept
            prob = _stable_sigmoid(eta, backend)
            resid = prob - yb
            grad_coef = Xb.T @ resid / n_samples
            grad_intercept = xp.mean(resid)

            w = y_coef - step * grad_coef
            if is_enet:
                thresh = float(alpha) * float(l1_ratio) * step
                denom = 1.0 + float(alpha) * (1.0 - float(l1_ratio)) * step
            else:
                thresh = float(alpha) * step
                denom = 1.0

            if backend == "torch":
                coef = torch.sign(w) * torch.clamp(torch.abs(w) - thresh, min=0.0) / denom
                intercept = y_intercept - step * grad_intercept
            elif backend == "cupy":
                coef = xp.sign(w) * xp.maximum(xp.abs(w) - thresh, 0.0) / denom
                intercept = y_intercept - step * grad_intercept
            else:
                coef = np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0) / denom
                intercept = y_intercept - step * grad_intercept

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = min((t_k - 1.0) / t_new, 0.5)
            y_coef = coef + beta * (coef - coef_old)
            y_intercept = intercept + beta * (intercept - intercept_old)
            t_k = t_new
            last_iter = iteration + 1

            if iteration < 20 or iteration % conv_interval == 0:
                delta = xp.sum(xp.abs(coef - coef_old)) + xp.abs(intercept - intercept_old)
                if backend == "torch":
                    converged = bool((delta < tol).item())
                elif backend == "cupy":
                    converged = bool(delta < tol)
                else:
                    converged = float(delta) < tol
                if converged:
                    break

        if Xv is not None:
            if backend == "torch":
                score_coef_path.append(coef.clone())
                score_intercept_path.append(intercept.clone())
            else:
                eta_v = Xv @ coef + intercept
                per_sample = -yv * eta_v + _softplus(eta_v, backend)
                if swv is not None:
                    val_loss = xp.sum(swv * per_sample) / xp.sum(swv)
                else:
                    val_loss = xp.mean(per_sample)
                score_coef_path.append(val_loss)
        if return_path:
            coef_path.append(np.asarray(_to_numpy(coef), dtype=np.float64).copy())
            intercept_path.append(_scalar_to_float(intercept))
        iters.append(last_iter)

    # Torch benefits from one alpha-path GEMM for validation.  For NumPy/CuPy
    # at these small alpha-grid widths, per-alpha GEMV is consistently steadier.
    if score_coef_path:
        if backend == "torch":
            import torch
            coef_mat = torch.stack(score_coef_path, dim=1)
            intercept_vec = torch.stack(score_intercept_path).reshape(1, -1)
            eta_v = Xv @ coef_mat + intercept_vec
            per_sample = -yv.reshape(-1, 1) * eta_v + _softplus(eta_v, backend)
            if swv is not None:
                sw_sum = swv.sum()
                scores_tensor = (swv.reshape(-1, 1) * per_sample).sum(dim=0) / sw_sum
            else:
                scores_tensor = per_sample.mean(dim=0)
            scores = _to_numpy(scores_tensor).tolist()
        elif backend == "cupy":
            import cupy as cp
            scores_arr = cp.stack(score_coef_path)
            scores = _to_numpy(scores_arr).tolist()
        else:
            scores = [float(s) for s in score_coef_path]

    out = {
        "scores": np.asarray(scores, dtype=np.float64) if scores else None,
        "n_iter": np.asarray(iters, dtype=np.int64),
    }
    if return_path:
        out["coef"] = np.vstack(coef_path).astype(np.float64, copy=False)
        out["intercept"] = np.asarray(intercept_path, dtype=np.float64)
    return out


# (Old per-loss fold-batched functions removed — replaced by _glm_sparse_cv_folds)


def _squared_error_sparse_cv_path(
    X_train,
    y_train,
    alpha_sorted,
    penalty_name,
    l1_ratio,
    max_iter,
    tol,
    device,
    X_val=None,
    y_val=None,
    sample_weight=None,
    val_sample_weight=None,
    return_path=True,
):
    """Fit a squared-error sparse alpha path with centered data.

    This is used by CV for l1/elasticnet penalties. It solves all alphas in one
    fold using a single Gram matrix and warm-started FISTA path.
    """
    if sample_weight is not None:
        sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
        if sw_np.size and not np.allclose(sw_np, sw_np[0]):
            return None

    backend = _backend_name_for_cv_device(device)
    Xb = _to_backend_float64(X_train, backend)
    yb = _to_backend_float64(y_train, backend).reshape(-1)
    alphas = np.asarray(alpha_sorted, dtype=np.float64).ravel()
    n_samples, n_features = Xb.shape
    penalty_name = str(penalty_name).lower()
    is_enet = penalty_name in ("elasticnet", "en")

    if backend == "torch":
        import torch
        xp = torch
        X_mean = torch.mean(Xb, dim=0)
        y_mean = torch.mean(yb)
        Xc = Xb - X_mean
        yc = yb - y_mean
        XtX = Xc.T @ Xc
        Xty = Xc.T @ yc
        eig_max = float(torch.linalg.eigvalsh(XtX).max().item())
        coef = torch.zeros(n_features, dtype=Xb.dtype, device=Xb.device)
    elif backend == "cupy":
        import cupy as cp
        xp = cp
        X_mean = cp.mean(Xb, axis=0)
        y_mean = cp.mean(yb)
        Xc = Xb - X_mean
        yc = yb - y_mean
        XtX = Xc.T @ Xc
        Xty = Xc.T @ yc
        eig_max = float(cp.linalg.eigvalsh(XtX).max())
        coef = cp.zeros(n_features, dtype=Xb.dtype)
    else:
        xp = np
        X_mean = np.mean(Xb, axis=0)
        y_mean = np.mean(yb)
        Xc = Xb - X_mean
        yc = yb - y_mean
        XtX = Xc.T @ Xc
        Xty = Xc.T @ yc
        eig_max = float(np.linalg.eigvalsh(XtX).max())
        coef = np.zeros(n_features, dtype=np.float64)

    L = max(eig_max / max(int(n_samples), 1), 1e-12)
    step = 1.0 / L
    conv_interval = 10 if backend == "numpy" else 25

    if X_val is not None and y_val is not None:
        Xv = _to_backend_float64(X_val, backend)
        yv = _to_backend_float64(y_val, backend).reshape(-1)
        Xv_centered = Xv - X_mean
        swv = _to_backend_float64(val_sample_weight, backend).reshape(-1) if val_sample_weight is not None else None
    else:
        Xv = yv = Xv_centered = swv = None

    if backend in ("torch", "cupy") and not return_path and Xv_centered is not None:
        n_alpha = int(alphas.size)
        if backend == "torch":
            import torch
            alpha_vec = torch.as_tensor(
                alphas, dtype=Xb.dtype, device=Xb.device
            ).reshape(1, -1)
            coef_mat = torch.zeros(
                (n_features, n_alpha), dtype=Xb.dtype, device=Xb.device
            )
            y_mat = coef_mat.clone()
        else:
            import cupy as cp
            alpha_vec = cp.asarray(alphas, dtype=Xb.dtype).reshape(1, -1)
            coef_mat = cp.zeros((n_features, n_alpha), dtype=Xb.dtype)
            y_mat = coef_mat.copy()

        t_k = 1.0
        last_iter = 0
        x_ty = Xty.reshape(-1, 1)
        for iteration in range(int(max_iter)):
            coef_old = coef_mat.clone() if backend == "torch" else coef_mat.copy()
            grad = (XtX @ y_mat - x_ty) / n_samples
            w = y_mat - step * grad
            if is_enet:
                thresh = alpha_vec * float(l1_ratio) * step
                denom = 1.0 + alpha_vec * (1.0 - float(l1_ratio)) * step
            else:
                thresh = alpha_vec * step
                denom = 1.0

            if backend == "torch":
                coef_mat = (
                    torch.sign(w)
                    * torch.clamp(torch.abs(w) - thresh, min=0.0)
                    / denom
                )
            else:
                coef_mat = xp.sign(w) * xp.maximum(xp.abs(w) - thresh, 0.0) / denom

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = (t_k - 1.0) / t_new
            y_mat = coef_mat + beta * (coef_mat - coef_old)
            t_k = t_new
            last_iter = iteration + 1

            if iteration < 20 or iteration % conv_interval == 0:
                if backend == "torch":
                    delta = torch.sum(torch.abs(coef_mat - coef_old), dim=0)
                    converged = bool(torch.all(delta < tol).item())
                else:
                    delta = cp.sum(cp.abs(coef_mat - coef_old), axis=0)
                    converged = bool(cp.all(delta < tol))
                if converged:
                    break

        pred = Xv_centered @ coef_mat + y_mean
        sq_err = (yv.reshape(-1, 1) - pred) ** 2
        if swv is not None:
            sw_col = swv.reshape(-1, 1)
            if backend == "torch":
                scores_dev = (sw_col * sq_err).sum(dim=0) / swv.sum()
            else:
                scores_dev = (sw_col * sq_err).sum(axis=0) / swv.sum()
        else:
            if backend == "torch":
                scores_dev = sq_err.mean(dim=0)
            else:
                scores_dev = sq_err.mean(axis=0)
        return {
            "scores": np.asarray(_to_numpy(scores_dev), dtype=np.float64),
            "n_iter": np.full(n_alpha, int(last_iter), dtype=np.int64),
        }

    scores = []
    scores_dev = []  # accumulate on device, sync once at end
    coef_path = []
    intercept_path = []
    iters = []

    for alpha in alphas:
        y_k = coef.copy() if backend != "torch" else coef.clone()
        t_k = 1.0
        last_iter = 0
        for iteration in range(int(max_iter)):
            coef_old = coef.copy() if backend != "torch" else coef.clone()
            grad = (XtX @ y_k - Xty) / n_samples
            w = y_k - step * grad
            if is_enet:
                thresh = float(alpha) * float(l1_ratio) * step
                denom = 1.0 + float(alpha) * (1.0 - float(l1_ratio)) * step
            else:
                thresh = float(alpha) * step
                denom = 1.0

            if backend == "torch":
                coef = torch.sign(w) * torch.clamp(torch.abs(w) - thresh, min=0.0) / denom
            else:
                coef = xp.sign(w) * xp.maximum(xp.abs(w) - thresh, 0.0) / denom

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = (t_k - 1.0) / t_new
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new
            last_iter = iteration + 1

            if backend == "numpy" or int(n_features) <= 128:
                check_convergence = iteration < 20 or iteration % conv_interval == 0
            else:
                check_convergence = iteration % conv_interval == 0
            if check_convergence:
                delta = xp.sum(xp.abs(coef - coef_old))
                if backend == "torch":
                    converged = bool((delta < tol).item())
                elif backend == "cupy":
                    converged = bool(delta < tol)
                else:
                    converged = float(delta) < tol
                if converged:
                    break

        intercept = y_mean - X_mean @ coef
        if Xv_centered is not None:
            pred = Xv_centered @ coef + y_mean
            sq_err = (yv - pred) ** 2
            if swv is not None:
                mse = xp.sum(swv * sq_err) / xp.sum(swv)
            else:
                mse = xp.mean(sq_err)
            scores_dev.append(mse)  # keep on device
        if return_path:
            coef_path.append(np.asarray(_to_numpy(coef), dtype=np.float64).copy())
            intercept_path.append(_scalar_to_float(intercept))
        iters.append(last_iter)

    # Batch sync validation scores from device.
    if scores_dev:
        if backend == "torch":
            import torch
            scores_tensor = torch.stack(scores_dev)
            scores = _to_numpy(scores_tensor).tolist()
        elif backend == "cupy":
            import cupy as cp
            scores_arr = cp.stack(scores_dev)
            scores = _to_numpy(scores_arr).tolist()
        else:
            scores = [float(s) for s in scores_dev]

    out = {
        "scores": np.asarray(scores, dtype=np.float64) if scores else None,
        "n_iter": np.asarray(iters, dtype=np.int64),
    }
    if return_path:
        out["coef"] = np.vstack(coef_path).astype(np.float64, copy=False)
        out["intercept"] = np.asarray(intercept_path, dtype=np.float64)
    return out


class _FeatureOnlySparsePenalty:
    """Wrap a sparse penalty so the final intercept coefficient is unpenalized."""

    def __init__(self, base_penalty, n_features, backend):
        self.base_penalty = base_penalty
        self.n_features = int(n_features)
        self.backend = backend

    @property
    def name(self):
        return getattr(self.base_penalty, "name", "")

    @property
    def alpha(self):
        return float(getattr(self.base_penalty, "alpha", 0.0))

    @property
    def l1_ratio(self):
        return float(getattr(self.base_penalty, "l1_ratio", 1.0))

    def value(self, coef):
        return self.base_penalty.value(coef[: self.n_features])

    def proximal(self, w, step, backend=None):
        backend = backend or self.backend
        w_feat = w[: self.n_features]
        result_feat = self.base_penalty.proximal(w_feat, step, backend=backend)
        if backend == "cupy":
            import cupy as cp
            result = cp.empty(w.shape[0], dtype=w.dtype)
            result[: self.n_features] = result_feat
            result[self.n_features] = cp.clip(w[self.n_features], -15.0, 15.0)
            return result
        if backend == "torch":
            import torch
            result = torch.empty(w.shape[0], dtype=w.dtype, device=w.device)
            result[: self.n_features] = result_feat
            result[self.n_features] = torch.clamp(w[self.n_features], -15.0, 15.0)
            return result
        result = np.empty(w.shape[0], dtype=w.dtype)
        result[: self.n_features] = result_feat
        result[self.n_features] = np.clip(w[self.n_features], -15.0, 15.0)
        return result


def _glm_sparse_cv_path(
    loss_name,
    X_train,
    y_train,
    alpha_sorted,
    penalty_name,
    l1_ratio,
    max_iter,
    tol,
    device,
    X_val=None,
    y_val=None,
    sample_weight=None,
    val_sample_weight=None,
    return_path=False,
    solver_name="fista",
    cv_mode=True,
):
    """Warm-started sparse GLM alpha path for CV.

    The helper is intentionally private: it reuses the production loss,
    penalty, and FISTA solver while avoiding estimator reconstruction and
    repeated host/device conversions inside a fold.

    When ``val_sample_weight`` is provided, validation loss is computed as
    a weighted mean instead of a simple mean.
    """
    loss_name = str(loss_name).lower()
    penalty_name = str(penalty_name).lower()
    # Allow any loss registered in the formula registry
    if loss_name not in _LOSS_RESIDUAL_FNS:
        return None
    if penalty_name not in ("l1", "elasticnet", "en"):
        return None
    if sample_weight is not None:
        sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
        if sw_np.size and not np.allclose(sw_np, sw_np[0]):
            return None

    from statgpu.glm_core._solver import fista_solver, fista_bb_solver
    from statgpu.linear_model._penalized import _resolve_loss_name
    from statgpu.penalties import get_penalty

    backend = _backend_name_for_cv_device(device)
    Xb = _to_backend_float64(X_train, backend)
    yb = _to_backend_float64(y_train, backend).reshape(-1)
    alphas = np.asarray(alpha_sorted, dtype=np.float64).ravel()
    n_samples, n_features = Xb.shape

    if backend == "torch":
        import torch
        ones = torch.ones((n_samples, 1), dtype=Xb.dtype, device=Xb.device)
        X_work = torch.cat([Xb, ones], dim=1)
    elif backend == "cupy":
        import cupy as cp
        ones = cp.ones((n_samples, 1), dtype=Xb.dtype)
        X_work = cp.concatenate([Xb, ones], axis=1)
    else:
        ones = np.ones((n_samples, 1), dtype=Xb.dtype)
        X_work = np.concatenate([Xb, ones], axis=1)

    if X_val is not None and y_val is not None:
        Xv = _to_backend_float64(X_val, backend)
        yv = _to_backend_float64(y_val, backend).reshape(-1)
        n_val = Xv.shape[0]
        if backend == "torch":
            import torch
            ones_v = torch.ones((n_val, 1), dtype=Xv.dtype, device=Xv.device)
            X_val_work = torch.cat([Xv, ones_v], dim=1)
        elif backend == "cupy":
            import cupy as cp
            ones_v = cp.ones((n_val, 1), dtype=Xv.dtype)
            X_val_work = cp.concatenate([Xv, ones_v], axis=1)
        else:
            ones_v = np.ones((n_val, 1), dtype=Xv.dtype)
            X_val_work = np.concatenate([Xv, ones_v], axis=1)
    else:
        X_val_work = yv = swv = None

    if X_val is not None and y_val is not None and val_sample_weight is not None:
        swv = _to_backend_float64(val_sample_weight, backend).reshape(-1)
    else:
        swv = None

    sw_fit = (
        _to_backend_float64(sample_weight, backend)
        if sample_weight is not None
        else None
    )
    loss_fn = _resolve_loss_name(loss_name)
    if penalty_name in ("elasticnet", "en"):
        base_penalty = get_penalty("elasticnet", alpha=float(alphas[0]), l1_ratio=float(l1_ratio))
    else:
        base_penalty = get_penalty("l1", alpha=float(alphas[0]))
    penalty = _FeatureOnlySparsePenalty(base_penalty, n_features, backend)

    lipschitz_L = None
    if not getattr(loss_fn, "_lipschitz_at_init", False):
        try:
            if backend == "torch":
                import torch
                zero_lip = torch.zeros(
                    n_features + 1,
                    dtype=X_work.dtype,
                    device=X_work.device,
                )
            elif backend == "cupy":
                import cupy as cp
                zero_lip = cp.zeros(n_features + 1, dtype=X_work.dtype)
            else:
                zero_lip = np.zeros(n_features + 1, dtype=np.float64)
            lipschitz_L = float(_to_numpy(loss_fn.lipschitz(X_work, zero_lip, y=yb)))
            if not np.isfinite(lipschitz_L) or lipschitz_L <= 0.0:
                lipschitz_L = None
        except Exception:
            lipschitz_L = None

    scores = []
    score_params_path = []
    coef_path = []
    intercept_path = []
    iters = []
    if backend == "torch":
        import torch
        y_mean = max(float(torch.mean(yb).item()), 1e-3)
    elif backend == "cupy":
        import cupy as cp
        y_mean = max(float(cp.mean(yb)), 1e-3)
    else:
        y_mean = max(float(np.mean(yb)), 1e-3)
    init_intercept = np.log(y_mean)
    if backend == "torch":
        import torch
        init = torch.zeros(n_features + 1, dtype=X_work.dtype, device=X_work.device)
        init[-1] = init_intercept
    elif backend == "cupy":
        import cupy as cp
        init = cp.zeros(n_features + 1, dtype=X_work.dtype)
        init[-1] = init_intercept
    else:
        init = np.zeros(n_features + 1, dtype=np.float64)
        init[-1] = init_intercept
    solver_name = str(solver_name).lower()
    solver_fn = fista_bb_solver if solver_name == "fista_bb" else fista_solver
    for alpha in alphas:
        base_penalty.alpha = float(alpha)
        solver_kwargs = {
            "max_iter": int(max_iter),
            "tol": tol,
            "init_coef": init,
            "sample_weight": sw_fit,
        }
        if lipschitz_L is not None:
            solver_kwargs["lipschitz_L"] = lipschitz_L
        if solver_fn is fista_solver or solver_name == "fista_bb":
            solver_kwargs["cv_mode"] = bool(cv_mode)
        params, n_iter = solver_fn(
            loss_fn,
            penalty,
            X_work,
            yb,
            **solver_kwargs,
        )
        init = params
        if X_val_work is not None:
            if backend == "torch":
                score_params_path.append(params.clone())
            elif backend == "cupy":
                score_params_path.append(params.copy())
            else:
                # NumPy path: use loss_fn.value() for consistency with GPU paths.
                # loss_fn handles all loss-specific parameters (NB alpha, Tweedie power, etc.)
                val = float(loss_fn.value(X_val_work, yv, params))
                if swv is not None:
                    # Compute per-sample weighted loss.
                    # Use Xv (without intercept column) since _evaluate_loss_numpy
                    # adds its own intercept column when fit_intercept=True.
                    yv_np = np.asarray(_to_numpy(yv), dtype=np.float64).ravel()
                    sw_np = np.asarray(_to_numpy(swv), dtype=np.float64).ravel()
                    Xv_np = np.asarray(_to_numpy(Xv), dtype=np.float64) if Xv is not None else None
                    params_np = np.asarray(_to_numpy(params), dtype=np.float64).ravel()
                    val = _evaluate_loss_numpy(loss_name, loss_fn,
                                               Xv_np, yv_np,
                                               params_np[:n_features],
                                               float(params_np[n_features]),
                                               True, sample_weight=sw_np)
                score_params_path.append(val)
        if return_path:
            params_np = np.asarray(_to_numpy(params), dtype=np.float64).ravel()
            coef_path.append(params_np[:n_features].copy())
            intercept_path.append(float(params_np[n_features]))
        iters.append(int(n_iter))

    # GPU backends: compute per-sample validation loss via registry,
    # then aggregate across samples (weighted or unweighted).
    # The registry functions work with any shape (1D or 2D batched eta).
    if score_params_path:
        _loss_params = {}
        if loss_name == "negative_binomial":
            _loss_params["alpha"] = float(getattr(loss_fn, "alpha", _NB_ALPHA_DEFAULT))
        elif loss_name == "tweedie":
            _loss_params["power"] = float(getattr(loss_fn, "power", _TWEEDIE_POWER_DEFAULT))

        if backend == "torch":
            import torch
            params_mat = torch.stack(score_params_path, dim=1)
            eta = X_val_work @ params_mat  # (n_val, n_alphas)
            yy = yv.reshape(-1, 1)
            per_sample = _LOSS_VALLOSS_FNS[loss_name](eta, yy, **_loss_params)
            if swv is not None:
                sw_col = swv.reshape(-1, 1)
                scores_tensor = (sw_col * per_sample).sum(dim=0) / swv.sum()
            else:
                scores_tensor = per_sample.mean(dim=0)
            scores = _to_numpy(scores_tensor).tolist()
        elif backend == "cupy":
            import cupy as cp
            params_mat = cp.stack(score_params_path, axis=1)
            eta = X_val_work @ params_mat  # (n_val, n_alphas)
            yy = yv.reshape(-1, 1)
            per_sample = _LOSS_VALLOSS_FNS[loss_name](eta, yy, **_loss_params)
            if swv is not None:
                sw_col = swv.reshape(-1, 1)
                scores_arr = (sw_col * per_sample).sum(axis=0) / swv.sum()
            else:
                scores_arr = per_sample.mean(axis=0)
            scores = _to_numpy(scores_arr).tolist()
        else:
            scores = [_scalar_to_float(s) for s in score_params_path]

    out = {
        "scores": np.asarray(scores, dtype=np.float64) if scores else None,
        "n_iter": np.asarray(iters, dtype=np.int64),
    }
    if return_path:
        out["coef"] = np.vstack(coef_path).astype(np.float64, copy=False)
        out["intercept"] = np.asarray(intercept_path, dtype=np.float64)
    return out


def _scad_mcp_cv_path(
    loss_name,
    X_train,
    y_train,
    alpha_sorted,
    penalty_name,
    l1_ratio,
    max_iter,
    tol,
    device,
    X_val=None,
    y_val=None,
    sample_weight=None,
    return_path=False,
    max_lla_per_step=3,
    lla_tol=1e-4,
):
    """Warm-started SCAD/MCP alpha path for CV.

    For each alpha: compute LLA weights from current coef, run FISTA with
    AdaptiveL1Penalty(weights=lla_w), warm-start from previous alpha.
    Avoids per-alpha model.fit() overhead.
    """
    loss_name = str(loss_name).lower()
    penalty_name = str(penalty_name).lower()
    if penalty_name not in ("scad", "mcp"):
        return None
    if sample_weight is not None:
        sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
        if sw_np.size and not np.allclose(sw_np, sw_np[0]):
            return None

    from statgpu.glm_core._solver import fista_solver
    from statgpu.linear_model._penalized import _resolve_loss_name
    from statgpu.penalties import get_penalty, SCADPenalty, MCPPenalty
    from statgpu.penalties._adaptive_l1 import AdaptiveL1Penalty

    backend = _backend_name_for_cv_device(device)
    Xb = _to_backend_float64(X_train, backend)
    yb = _to_backend_float64(y_train, backend).reshape(-1)
    alphas = np.asarray(alpha_sorted, dtype=np.float64).ravel()
    n_samples, n_features = Xb.shape

    # Augment X with intercept column
    if backend == "torch":
        import torch
        ones = torch.ones((n_samples, 1), dtype=Xb.dtype, device=Xb.device)
        X_work = torch.cat([Xb, ones], dim=1)
    elif backend == "cupy":
        import cupy as cp
        ones = cp.ones((n_samples, 1), dtype=Xb.dtype)
        X_work = cp.concatenate([Xb, ones], axis=1)
    else:
        ones = np.ones((n_samples, 1), dtype=Xb.dtype)
        X_work = np.concatenate([Xb, ones], axis=1)

    # Validation data
    if X_val is not None and y_val is not None:
        Xv = _to_backend_float64(X_val, backend)
        yv = _to_backend_float64(y_val, backend).reshape(-1)
        n_val = Xv.shape[0]
        if backend == "torch":
            ones_v = torch.ones((n_val, 1), dtype=Xv.dtype, device=Xv.device)
            X_val_work = torch.cat([Xv, ones_v], dim=1)
        elif backend == "cupy":
            ones_v = cp.ones((n_val, 1), dtype=Xv.dtype)
            X_val_work = cp.concatenate([Xv, ones_v], axis=1)
        else:
            ones_v = np.ones((n_val, 1), dtype=Xv.dtype)
            X_val_work = np.concatenate([Xv, ones_v], axis=1)
    else:
        X_val_work = yv = None

    loss_fn = _resolve_loss_name(loss_name)

    # Create SCAD/MCP penalty object
    if penalty_name == "scad":
        scad_penalty = SCADPenalty(alpha=float(alphas[0]))
    else:
        scad_penalty = MCPPenalty(alpha=float(alphas[0]))

    # Precompute XtX and Lipschitz for squared_error
    _is_quadratic = (loss_name == "squared_error")
    if _is_quadratic:
        if backend == "torch":
            X_mean = torch.mean(X_work[:, :n_features], dim=0)
            y_mean = torch.mean(yb)
            Xc = X_work[:, :n_features] - X_mean
            yc = yb - y_mean
            XtX = Xc.T @ Xc / n_samples
            Xty = Xc.T @ yc / n_samples
            eig_max = float(torch.linalg.eigvalsh(XtX).max().item())
        elif backend == "cupy":
            X_mean = cp.mean(X_work[:, :n_features], axis=0)
            y_mean = cp.mean(yb)
            Xc = X_work[:, :n_features] - X_mean
            yc = yb - y_mean
            XtX = Xc.T @ Xc / n_samples
            Xty = Xc.T @ yc / n_samples
            eig_max = float(cp.linalg.eigvalsh(XtX).max())
        else:
            X_mean = np.mean(X_work[:, :n_features], axis=0)
            y_mean = np.mean(yb)
            Xc = X_work[:, :n_features] - X_mean
            yc = yb - y_mean
            XtX = Xc.T @ Xc / n_samples
            Xty = Xc.T @ yc / n_samples
            eig_max = float(np.linalg.eigvalsh(XtX).max())
        L_base = max(eig_max * 2.0, 1.0)  # safety factor
    else:
        # For GLM losses, compute Lipschitz from loss
        if backend == "torch":
            import torch
            _zero = torch.zeros(n_features + 1, dtype=Xb.dtype, device=Xb.device)
        elif backend == "cupy":
            import cupy as cp
            _zero = cp.zeros(n_features + 1, dtype=Xb.dtype)
        else:
            _zero = np.zeros(n_features + 1)
        L_base = float(_to_numpy(loss_fn.lipschitz(X_work, _zero, y=yb)))
        _safety = getattr(loss_fn, '_lipschitz_safety', 1.0)
        if _safety > 1.0:
            L_base *= _safety

    scores = []
    scores_dev = []
    coef_path = []
    intercept_path = []
    iters = []
    L_glm = None  # Lipschitz constant for GLM losses (computed once)

    # Initialize coef (warm-start from zeros or previous fold)
    if backend == "torch":
        coef = torch.zeros(n_features + 1, dtype=Xb.dtype, device=Xb.device)
    elif backend == "cupy":
        coef = cp.zeros(n_features + 1, dtype=Xb.dtype)
    else:
        coef = np.zeros(n_features + 1)

    # Pre-create inner penalty object (reuse across LLA iterations)
    inner_pen = AdaptiveL1Penalty(alpha=1.0)

    for alpha in alphas:
        scad_penalty.alpha = float(alpha)

        # LLA outer loop
        for lla_iter in range(max_lla_per_step):
            # Compute LLA weights from current coef (features only, intercept gets 0)
            lla_w_feat = scad_penalty.lla_weights(coef[:n_features])
            if backend == "torch":
                lla_w = torch.cat([lla_w_feat, torch.zeros(1, device=coef.device, dtype=coef.dtype)])
            elif backend == "cupy":
                lla_w = cp.concatenate([lla_w_feat, cp.zeros(1, dtype=coef.dtype)])
            else:
                lla_w = np.append(lla_w_feat, 0.0)

            # Update weights in-place (avoid object creation overhead)
            inner_pen._weights = lla_w

            coef_before_lla = coef.copy() if backend != "torch" else coef.clone()

            # FISTA inner solve with warm-start
            if _is_quadratic:
                # Squared error: use precomputed XtX
                step = 1.0 / L_base
                y_k = coef.copy() if backend != "torch" else coef.clone()
                t_k = 1.0
                for iteration in range(max_iter):
                    coef_old = coef.copy() if backend != "torch" else coef.clone()
                    if backend == "torch":
                        grad = XtX @ y_k[:n_features] - Xty
                        # Extend grad to include intercept dimension
                        grad_full = torch.cat([grad, torch.zeros(1, device=grad.device, dtype=grad.dtype)])
                    elif backend == "cupy":
                        grad = XtX @ y_k[:n_features] - Xty
                        grad_full = cp.concatenate([grad, cp.zeros(1, dtype=grad.dtype)])
                    else:
                        grad = XtX @ y_k[:n_features] - Xty
                        grad_full = np.concatenate([grad, [0.0]])

                    w = y_k - step * grad_full
                    coef = inner_pen.proximal(w, step, backend=backend)

                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                    beta_mom = (t_k - 1.0) / t_new
                    t_k = t_new
                    y_k = coef + beta_mom * (coef - coef_old)

                    # Convergence check (device-side, every 10 iters for CV)
                    if iteration % 10 == 0 and iteration > 0:
                        delta = _abs_sum(coef - coef_old, backend)
                        if _scalar_lt(delta, tol, backend):
                            break
            else:
                # GLM loss: direct FISTA loop with device-side convergence.
                # Precompute Lipschitz constant once (reuse across alphas).
                if L_glm is None:
                    if backend == "torch":
                        _zero = torch.zeros(n_features + 1, dtype=Xb.dtype, device=Xb.device)
                    elif backend == "cupy":
                        _zero = cp.zeros(n_features + 1, dtype=Xb.dtype)
                    else:
                        _zero = np.zeros(n_features + 1)
                    L_glm = float(_to_numpy(loss_fn.lipschitz(X_work, _zero, y=yb)))
                    _safety = getattr(loss_fn, '_lipschitz_safety', 1.0)
                    if _safety > 1.0:
                        L_glm *= _safety
                    # Y-scaling for exp-link families
                    _loss_name_inner = getattr(loss_fn, 'name', '')
                    _skip_ys = getattr(loss_fn, '_lipschitz_uses_y', False)
                    if _loss_name_inner not in ('squared_error',) and not _skip_ys:
                        _y_abs = np.abs(_to_numpy(yb))
                        _y_mean = float(np.mean(_y_abs))
                        _y_max = float(np.max(_y_abs))
                        _y_scale = min(10.0, max(1.0, np.sqrt(_y_mean * _y_max)))
                        if _y_scale > 1.0:
                            L_glm *= _y_scale
                    L_glm = max(L_glm, 1.0)

                step = 1.0 / L_glm
                y_k = coef.copy() if backend != "torch" else coef.clone()
                t_k = 1.0
                for iteration in range(max_iter):
                    coef_old = coef.copy() if backend != "torch" else coef.clone()

                    # Gradient: loss.gradient(X, y, coef)
                    grad = loss_fn.gradient(X_work, yb, y_k)

                    # Proximal step
                    w = y_k - step * grad
                    coef = inner_pen.proximal(w, step, backend=backend)

                    # Momentum
                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                    beta_mom = (t_k - 1.0) / t_new
                    t_k = t_new
                    y_k = coef + beta_mom * (coef - coef_old)

                    # Convergence check (device-side, every 10 iters for CV)
                    if iteration % 10 == 0 and iteration > 0:
                        delta = _abs_sum(coef - coef_old, backend)
                        if _scalar_lt(delta, tol, backend):
                            break

            # LLA convergence check
            delta = _abs_sum(coef - coef_before_lla, backend)
            if _scalar_lt(delta, lla_tol, backend):
                break

        # Extract coef and compute intercept from centered-data fit.
        # For squared_error, the FISTA loop works on centered X/y, so
        # coef[n_features] stays at zero. Compute the correct intercept.
        if backend == "torch":
            coef_feat = coef[:n_features]
            coef_np = coef_feat.detach().cpu().numpy()
        elif backend == "cupy":
            coef_feat = coef[:n_features]
            coef_np = coef_feat.get()
        else:
            coef_feat = coef[:n_features]
            coef_np = coef_feat.copy()

        if _is_quadratic and X_mean is not None:
            # intercept = y_mean - X_mean @ coef_features (from centering)
            if backend == "torch":
                intercept = float((y_mean - (torch.as_tensor(X_mean, device=coef.device, dtype=coef.dtype) @ coef_feat).item()))
            elif backend == "cupy":
                intercept = float(y_mean - float((cp.asarray(X_mean) @ coef_feat).item()))
            else:
                intercept = float(y_mean - X_mean @ coef_feat)
            # Update coef[n_features] so validation uses correct intercept
            coef[n_features] = intercept
        else:
            if backend == "torch":
                intercept = float(coef[n_features].item())
            elif backend == "cupy":
                intercept = float(coef[n_features].get())
            else:
                intercept = float(coef[n_features])

        # Validation loss on device
        if X_val_work is not None:
            val_loss = loss_fn.value(X_val_work, yv, coef)
            scores_dev.append(val_loss)

        if return_path:
            coef_path.append(coef_np)
            intercept_path.append(intercept)
        iters.append(1)  # placeholder

    # Batch sync validation scores
    if scores_dev:
        if backend == "torch":
            import torch
            scores_tensor = torch.stack(scores_dev)
            scores = _to_numpy(scores_tensor).tolist()
        elif backend == "cupy":
            import cupy as cp
            scores_arr = cp.stack(scores_dev)
            scores = _to_numpy(scores_arr).tolist()
        else:
            scores = [float(s) for s in scores_dev]

    out = {
        "scores": np.asarray(scores, dtype=np.float64) if scores else None,
        "n_iter": np.asarray(iters, dtype=np.int64),
    }
    if return_path:
        out["coef"] = np.vstack(coef_path).astype(np.float64, copy=False)
        out["intercept"] = np.asarray(intercept_path, dtype=np.float64)
    return out


def _abs_sum(arr, backend):
    """Device-side absolute sum."""
    if backend == "torch":
        import torch
        return torch.sum(torch.abs(arr))
    elif backend == "cupy":
        import cupy as cp
        return cp.sum(cp.abs(arr))
    return np.sum(np.abs(arr))


def _scalar_lt(a, b, backend):
    """Device-side scalar comparison, returns Python bool."""
    if backend == "torch":
        return bool((a < b).item())
    elif backend == "cupy":
        return bool(a < b)
    return float(a) < float(b)


class PenalizedGLM_CV(CVEstimatorBase):
    """Cross-validated penalized GLM supporting all loss + penalty combinations."""

    def __init__(
        self,
        loss: str = 'squared_error',
        penalty: str = 'l2',
        alpha_grid=None,
        n_alphas: int = 100,
        l1_ratio: float = 0.5,
        cv: int = 5,
        cv_splits=None,
        random_state: Optional[int] = 0,
        device: Union[str, Device] = Device.AUTO,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'auto',
        cv_strategy: str = "strict",
        acknowledge_approx: bool = False,
        refine_top_k: int = 3,
    ):
        super().__init__(cv=cv, random_state=random_state, device=device)
        self.cv_splits = cv_splits
        cv_strategy = str(cv_strategy).lower()
        if cv_strategy not in ("strict", "two_stage"):
            raise ValueError(
                "cv_strategy must be either 'strict' or 'two_stage', "
                f"got {cv_strategy!r}."
            )
        if int(refine_top_k) < 1:
            raise ValueError("refine_top_k must be a positive integer")
        self.loss = loss
        self.penalty = penalty
        self._alpha_grid_input = alpha_grid
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.cv_strategy = cv_strategy
        self.acknowledge_approx = bool(acknowledge_approx)
        self.refine_top_k = int(refine_top_k)

        self.alpha_ = None
        self.alpha_grid_ = None
        self.cv_strategy_ = None
        self.cv_selected_device_ = None
        self._cv_selected_device_ = None
        self._cv_auto_reason_ = None

    def _solver_for_cv(self, cv_device=None, X=None):
        """Return the strict internal solver used by the CV loop."""
        solver = str(self.solver).lower()
        if solver != "auto":
            return solver
        from statgpu.linear_model._penalized import _preferred_penalized_glm_solver

        return _preferred_penalized_glm_solver(
            self.loss,
            self.penalty,
            backend_name=_backend_name_for_cv_device(
                self.device if cv_device is None else cv_device
            ),
            l1_ratio=self.l1_ratio,
            cv_mode=True,
            problem_size=None if X is None else int(X.shape[0]) * int(X.shape[1]),
        )

    def _effective_cv_device(self, X, penalty_name, n_alphas):
        """Resolve device for CV-level work; explicit devices are untouched."""
        self._cv_selected_device_ = self.device
        self._cv_auto_reason_ = None
        if _device_to_name(self.device) != "auto":
            return self.device

        n_samples, n_features = X.shape
        penalty_name = str(penalty_name).lower()
        loss_name = str(self.loss).lower()
        continuation_factor = 20 if loss_name != "squared_error" and penalty_name in ("scad", "mcp") else 1
        effective_work = int(n_samples) * int(n_features) * int(self.cv) * int(n_alphas) * continuation_factor

        if int(n_samples) * int(n_features) < 200_000:
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "small CV problem is faster on CPU"
            return "cpu"
        if loss_name == "squared_error" and penalty_name in ("l1", "elasticnet", "en"):
            if (
                int(n_features) >= 256
                and int(n_samples) * int(n_features) >= 1_000_000
                and _torch_cuda_available()
            ):
                self._cv_selected_device_ = "torch"
                self._cv_auto_reason_ = "medium squared-error sparse CV benefits from batched torch alpha path"
                return "torch"
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "squared-error sparse CV is faster on CPU below the benchmarked torch break-even"
            return "cpu"
        if loss_name != "squared_error" and penalty_name in ("scad", "mcp"):
            # GLM SCAD/MCP: allow GPU at large scale where LLA overhead is amortized
            if int(n_samples) * int(n_features) >= 1_000_000 and _torch_cuda_available():
                self._cv_selected_device_ = "torch"
                self._cv_auto_reason_ = "large GLM SCAD/MCP CV benefits from torch async FISTA"
                return "torch"
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "GLM SCAD/MCP CV continuation is faster on CPU for current benchmarked sizes"
            return "cpu"
        if loss_name == "logistic" and penalty_name in ("l1", "elasticnet", "en"):
            # Logistic sparse: fold-batched Torch CV amortizes high-dimensional
            # and large-n alpha paths, but low-dimensional small-n rows remain
            # faster on CPU.
            if (
                (
                    int(n_samples) >= 5_000
                    and int(n_samples) * int(n_features) >= 500_000
                )
                or (
                    int(n_features) >= 500
                    and int(n_samples) * int(n_features) >= 1_000_000
                )
            ) and _torch_cuda_available():
                self._cv_selected_device_ = "torch"
                self._cv_auto_reason_ = "medium logistic sparse CV benefits from fold-batched torch path"
                return "torch"
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "logistic sparse CV is faster on CPU for current benchmarked sizes"
            return "cpu"
        if loss_name == "poisson" and penalty_name in ("l1", "elasticnet", "en"):
            if int(n_features) >= 500 and int(n_samples) * int(n_features) >= 1_000_000 and _torch_cuda_available():
                self._cv_selected_device_ = "torch"
                self._cv_auto_reason_ = "high-dimensional poisson sparse CV benefits from torch"
                return "torch"
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "poisson sparse CV is faster on CPU below the benchmarked torch break-even"
            return "cpu"
        if loss_name == "gamma" and penalty_name in ("l1", "elasticnet", "en"):
            if int(n_features) >= 500 and int(n_samples) * int(n_features) >= 2_000_000 and _torch_cuda_available():
                self._cv_selected_device_ = "torch"
                self._cv_auto_reason_ = "large high-dimensional gamma sparse CV benefits from torch"
                return "torch"
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "gamma sparse CV is faster on CPU below the benchmarked torch break-even"
            return "cpu"
        if loss_name == "inverse_gaussian" and penalty_name in ("l1", "elasticnet", "en"):
            if int(n_features) >= 500 and int(n_samples) * int(n_features) >= 2_000_000 and _torch_cuda_available():
                self._cv_selected_device_ = "torch"
                self._cv_auto_reason_ = "large high-dimensional inverse-gaussian sparse CV benefits from torch"
                return "torch"
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "inverse-gaussian sparse CV is faster on CPU below the benchmarked torch break-even"
            return "cpu"
        if loss_name == "negative_binomial" and penalty_name in ("l2", "l1", "elasticnet", "en"):
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "negative-binomial CV is faster on CPU for current benchmarked sizes"
            return "cpu"
        if loss_name == "tweedie" and penalty_name in ("l1", "elasticnet", "en"):
            if int(n_samples) * int(n_features) >= 300_000 and _torch_cuda_available():
                self._cv_selected_device_ = "torch"
                self._cv_auto_reason_ = f"medium tweedie {penalty_name} CV is faster on torch"
                return "torch"
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = f"small tweedie {penalty_name} CV is faster on CPU"
            return "cpu"
        if effective_work < 100_000_000:
            self._cv_selected_device_ = "cpu"
            self._cv_auto_reason_ = "CV effective work is below GPU break-even"
            return "cpu"

        return self.device

    def _generate_alpha_grid(self, X, y):
        """Auto-generate alpha grid based on loss and penalty type."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        X_np = _to_numpy(X).astype(np.float64)
        y_np = _to_numpy(y).astype(np.float64).ravel()
        n = X_np.shape[0]

        if self.loss == 'squared_error':
            alpha_max = float(np.max(np.abs(X_np.T @ y_np))) / n
        elif self.loss == 'logistic':
            y_centered = y_np - 0.5
            alpha_max = float(np.max(np.abs(X_np.T @ y_centered))) / n
        else:
            try:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty='l2', alpha=0.0,
                    device='cpu', compute_inference=False, max_iter=5,
                )
                model.fit(X_np, y_np)
                grad = X_np.T @ (y_np - _to_numpy(model.predict(X_np))) / n
                alpha_max = float(np.max(np.abs(grad)))
            except Exception as e:
                warnings.warn(
                    f"Alpha grid estimation failed ({e}), using alpha_max=1.0",
                    RuntimeWarning,
                    stacklevel=2,
                )
                alpha_max = 1.0

        if alpha_max <= 0:
            warnings.warn(
                f"Alpha grid estimation returned {alpha_max}, using alpha_max=1.0",
                RuntimeWarning,
                stacklevel=2,
            )
            alpha_max = 1.0

        if self.penalty in ('l1', 'elasticnet', 'scad', 'mcp', 'adaptive_l1'):
            grid = np.geomspace(alpha_max, alpha_max * 1e-4, self.n_alphas)
        else:
            grid = np.logspace(
                np.log10(max(alpha_max * 1e-4, 1e-12)),
                np.log10(alpha_max),
                self.n_alphas,
            )

        return grid

    def _solve_ridge_fold_batch(self, X_train, y_train, X_val, y_val, alphas):
        """Batch solve Ridge CV for all alphas using eigendecomposition."""
        X_train_np = _to_numpy(X_train).astype(np.float64)
        y_train_np = _to_numpy(y_train).astype(np.float64).ravel()
        X_val_np = _to_numpy(X_val).astype(np.float64)
        y_val_np = _to_numpy(y_val).astype(np.float64).ravel()
        alphas_np = _to_numpy(alphas).astype(np.float64).ravel()
        return _ridge_eig_batch(X_train_np, y_train_np, X_val_np, y_val_np, alphas_np)

    def _evaluate_single(self, model, X_val, y_val, loss_fn=None, X_val_np=None, y_val_np=None, sample_weight=None):
        """Evaluate a fitted model on validation data, return validation loss.

        Parameters
        ----------
        loss_fn : optional, pre-resolved loss function (avoids repeated import)
        X_val_np, y_val_np : optional, pre-cached numpy validation data (avoids D2H)
        sample_weight : optional, per-sample weights for weighted validation loss
        """
        from statgpu.linear_model._penalized import _resolve_loss_name

        if loss_fn is None:
            loss_fn = _resolve_loss_name(self.loss)
        if X_val_np is None:
            X_val_np = _to_numpy(X_val).astype(np.float64)
        if y_val_np is None:
            y_val_np = _to_numpy(y_val).astype(np.float64).ravel()
        n_val = X_val_np.shape[0]

        try:
            val_loss = _evaluate_loss_numpy(
                self.loss,
                loss_fn,
                X_val_np,
                y_val_np,
                _to_numpy(model.coef_).ravel(),
                float(model.intercept_),
                model.fit_intercept,
                sample_weight=sample_weight,
            )
        except Exception:
            # Fallback: use loss_fn.value() for correct loss, not raw MSE
            try:
                if model.fit_intercept:
                    X_design = np.column_stack([np.ones(n_val), X_val_np])
                    coef_full = np.concatenate([[float(model.intercept_)], _to_numpy(model.coef_).ravel()])
                else:
                    X_design = X_val_np
                    coef_full = _to_numpy(model.coef_).ravel()
                val_loss = float(loss_fn.value(X_design, y_val_np, coef_full))
            except Exception:
                y_pred_np = _to_numpy(model.predict(X_val_np)).ravel()
                val_loss = float(np.mean((y_val_np - y_pred_np) ** 2))

        return val_loss

    @staticmethod
    def _populate_refit_model(model, coef, intercept, X, device, n_iter=None):
        """Set standard attributes on a refit model from path results."""
        model.coef_ = np.asarray(coef, dtype=np.float64)
        model.intercept_ = float(intercept)
        if n_iter is not None:
            model.n_iter_ = int(n_iter)
        model._params = np.concatenate([[float(intercept)], np.asarray(coef, dtype=np.float64)])
        model._nobs = int(X.shape[0])
        model._df_resid = int(X.shape[0] - X.shape[1] - 1)
        model._selected_backend_name = _backend_name_for_cv_device(device)
        model._fitted = True
        return model

    def _refit_best(self, X, y, best_alpha, sample_weight=None):
        """Refit on full data with best alpha.

        For squared_error + l2, uses eigendecomposition to match the CV path
        exactly, avoiding precision mismatch between CV and refit solvers.
        """
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        # Resolve refit device (used by Ridge and general paths)
        refit_device = self.device
        if _device_to_name(self.device) == "auto":
            refit_device = getattr(self, "_cv_selected_device_", self.device) or self.device

        # For Ridge: use eigendecomposition to match CV path exactly.
        # Supports weighted Ridge via weighted eigensolve (same O(p³) cost).
        if self.loss == 'squared_error' and self.penalty == 'l2':
            X_np = _to_numpy(X).astype(np.float64)
            y_np = _to_numpy(y).astype(np.float64).ravel()
            sw_np = _to_numpy(sample_weight).astype(np.float64).ravel() if sample_weight is not None else None
            coef, intercept = _ridge_eig_single(X_np, y_np, best_alpha, sample_weight=sw_np)
            model = PenalizedGeneralizedLinearModel(
                loss='squared_error', penalty='l2', alpha=best_alpha,
                device=refit_device, compute_inference=False,
                max_iter=self.max_iter, tol=self.tol,
            )
            return self._populate_refit_model(model, coef, intercept, X, refit_device)

        can_infer = (self.loss == 'squared_error' and self.penalty == 'l2')
        penalty_name = str(self.penalty).lower()

        if self.loss == "logistic" and penalty_name in ("l1", "elasticnet", "en"):
            path = _logistic_sparse_cv_path(
                X,
                y,
                np.asarray([best_alpha], dtype=np.float64),
                penalty_name,
                self.l1_ratio,
                _logistic_sparse_effective_max_iter(self.max_iter, refit_device, penalty_name, refit=True),
                self.tol,
                refit_device,
                sample_weight=sample_weight,
            )
            if path is not None:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty=self.penalty, alpha=best_alpha,
                    l1_ratio=self.l1_ratio, device=refit_device,
                    compute_inference=False, max_iter=self.max_iter,
                    tol=self.tol, solver=self._solver_for_cv(refit_device, X=X),
                )
                return self._populate_refit_model(
                    model, path["coef"][-1], path["intercept"][-1],
                    X, refit_device, n_iter=path["n_iter"][-1],
                )

        if self.loss == "squared_error" and penalty_name in ("l1", "elasticnet", "en"):
            path = _squared_error_sparse_cv_path(
                X, y, np.asarray([best_alpha], dtype=np.float64),
                penalty_name, self.l1_ratio, self.max_iter, self.tol,
                refit_device, sample_weight=sample_weight,
            )
            if path is not None:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty=self.penalty, alpha=best_alpha,
                    l1_ratio=self.l1_ratio, device=refit_device,
                    compute_inference=False, max_iter=self.max_iter,
                    tol=self.tol, solver=self._solver_for_cv(refit_device, X=X),
                )
                return self._populate_refit_model(
                    model, path["coef"][-1], path["intercept"][-1],
                    X, refit_device, n_iter=path["n_iter"][-1],
                )

        cv_solver = self._solver_for_cv(refit_device, X=X)
        if self._uses_glm_sparse_path(penalty_name, cv_solver):
            path = _glm_sparse_cv_path(
                self.loss,
                X,
                y,
                np.asarray([best_alpha], dtype=np.float64),
                penalty_name,
                self.l1_ratio,
                self.max_iter,
                self.tol,
                refit_device,
                return_path=True,
                solver_name=cv_solver,
                cv_mode=False,
                sample_weight=sample_weight,
            )
            if path is not None:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty=self.penalty, alpha=best_alpha,
                    l1_ratio=self.l1_ratio, device=refit_device,
                    compute_inference=False, max_iter=self.max_iter,
                    tol=self.tol, solver=cv_solver,
                )
                return self._populate_refit_model(
                    model, path["coef"][-1], path["intercept"][-1],
                    X, refit_device, n_iter=path["n_iter"][-1],
                )

        model = PenalizedGeneralizedLinearModel(
            loss=self.loss,
            penalty=self.penalty,
            alpha=best_alpha,
            l1_ratio=self.l1_ratio,
            device=refit_device,
            compute_inference=can_infer,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self._solver_for_cv(refit_device, X=X),
        )
        model.fit(X, y, sample_weight=sample_weight)
        return model

    def _uses_glm_sparse_path(self, penalty_name, cv_solver):
        penalty_name = str(penalty_name).lower()
        cv_solver = str(cv_solver).lower()
        return (
            (
                (self.loss == "poisson" and penalty_name in ("l1", "elasticnet", "en"))
                or self.loss in ("gamma", "inverse_gaussian", "tweedie")
                or (self.loss == "negative_binomial" and cv_solver == "fista_bb")
            )
            and penalty_name in ("l1", "elasticnet", "en")
            and cv_solver in ("auto", "fista", "fista_bb")
        )

    def _best_index_from_scores(self, mean_scores, alpha_grid, cv_solver):
        penalty_name = str(self.penalty).lower()
        loss_name = str(self.loss).lower()
        if loss_name == "poisson" and penalty_name in ("l1", "elasticnet", "en"):
            # Poisson sparse CV curves can be nearly flat at the low-alpha end.
            # CPU/CuPy/Torch validation scores may differ at ~1e-7 from
            # backend-level summation order, so treat those as ties and keep
            # selection deterministic toward stronger regularization.
            return _nanargmin_prefer_larger_alpha(
                mean_scores,
                alpha_grid,
                rel_tol=5e-7,
                abs_tol=1e-6,
            )
        if self._uses_glm_sparse_path(penalty_name, cv_solver):
            return _nanargmin_prefer_larger_alpha(
                mean_scores,
                alpha_grid,
                rel_tol=5e-6,
                abs_tol=1e-7,
            )
        return _nanargmin_prefer_larger_alpha(mean_scores, alpha_grid)

    def _compute_cv_scores(
        self,
        X,
        y,
        alpha_grid,
        cv_device,
        folds,
        sample_weight=None,
        max_iter=None,
        tol=None,
        strict=True,
    ):
        """Compute CV scores for exactly the supplied alpha grid."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        alpha_grid = np.asarray(alpha_grid, dtype=np.float64).ravel()
        n_alphas = len(alpha_grid)
        penalty_name = str(self.penalty).lower()
        loss_name = str(self.loss).lower()
        device_name = _device_to_name(cv_device)
        max_iter = int(self.max_iter if max_iter is None else max_iter)

        # Validate sample_weight compatibility with penalty.
        # Only L2/IRLS supports non-uniform weights. All other penalties
        # (L1, ElasticNet, SCAD, MCP, etc.) use FISTA which rejects
        # non-uniform sample_weight.
        _non_l2_penalties = ("l1", "elasticnet", "en", "scad", "mcp",
                             "adaptive_l1", "adaptive_lasso",
                             "group_lasso", "gl", "group_mcp", "gmcp",
                             "group_scad", "gscad")
        if sample_weight is not None and penalty_name in _non_l2_penalties:
            sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
            if not np.allclose(sw_np, sw_np[0]):
                raise ValueError(
                    f"Non-uniform sample_weight is not supported for "
                    f"penalty='{penalty_name}' with any current solver. "
                    f"Use penalty='l2' with solver='irls' for weighted "
                    f"GLM fits, or use uniform weights."
                )
        tol = self.tol if tol is None else tol

        # ── Fast path: Ridge eigendecomposition (CPU only, unweighted) ──
        _is_explicit_gpu = device_name in ("cuda", "torch")
        if loss_name == "squared_error" and penalty_name == "l2" and sample_weight is None and not _is_explicit_gpu:
            all_scores = np.full((self.cv, n_alphas), np.nan)
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train = _slice_rows(X, train_idx)
                y_train = _slice_rows(y, train_idx)
                X_val = _slice_rows(X, val_idx)
                y_val = _slice_rows(y, val_idx)
                try:
                    mse, _, _ = self._solve_ridge_fold_batch(
                        X_train, y_train, X_val, y_val, alpha_grid,
                    )
                    all_scores[fold_idx, :] = mse
                except Exception as e:
                    warnings.warn(
                        f"Ridge eig batch failed for fold {fold_idx}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return all_scores

        sort_idx = np.argsort(-alpha_grid)
        alpha_sorted = alpha_grid[sort_idx]
        all_scores = np.full((self.cv, n_alphas), np.nan)
        cv_solver = self._solver_for_cv(cv_device, X=X)

        # ── Fast path: fold-batched CV (all folds at once, GPU only) ──
        use_fold_batch = (
            not strict
            and loss_name in _FOLD_BATCH_CONFIGS
            and penalty_name in ("l1", "elasticnet", "en")
            and device_name in ("torch", "cuda")
        )
        if use_fold_batch:
            try:
                path = _glm_sparse_cv_folds(
                    X, y, folds, alpha_sorted, penalty_name, self.l1_ratio,
                    max_iter, tol, loss_name, device_name,
                    sample_weight=sample_weight,
                )
                if path is not None and path["scores"] is not None:
                    all_scores[:, sort_idx] = path["scores"]
                    return all_scores
            except Exception as e:
                warnings.warn(
                    f"Fold-batched {loss_name} sparse CV failed on {device_name}; "
                    f"falling back to per-fold path: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # ── Per-fold dispatch table ──
        # Each entry: (condition_fn, path_fn)
        # condition_fn(loss_name, penalty_name, cv_solver, strict) -> bool
        # path_fn(X_train, y_train, alpha_sorted, ..., X_val, y_val, sw_train, sw_val) -> dict or None
        _per_fold_paths = []

        def _cond_scad_mcp(loss_name, penalty_name, cv_solver, strict):
            return penalty_name in ("scad", "mcp") and (loss_name == "squared_error" or not strict)

        def _path_scad_mcp(X_train, y_train, alpha_sorted, penalty_name, l1_ratio,
                           max_iter, tol, cv_device, X_val, y_val, sw_train, sw_val):
            return _scad_mcp_cv_path(
                loss_name, X_train, y_train, alpha_sorted, penalty_name,
                l1_ratio, max_iter, tol, cv_device,
                X_val=X_val, y_val=y_val, sample_weight=sw_train,
            )

        def _cond_logistic(loss_name, penalty_name, cv_solver, strict):
            return loss_name == "logistic" and penalty_name in ("l1", "elasticnet", "en")

        def _path_logistic(X_train, y_train, alpha_sorted, penalty_name, l1_ratio,
                           max_iter, tol, cv_device, X_val, y_val, sw_train, sw_val):
            return _logistic_sparse_cv_path(
                X_train, y_train, alpha_sorted, penalty_name, l1_ratio,
                max_iter, tol, cv_device,
                X_val=X_val, y_val=y_val, sample_weight=sw_train,
                val_sample_weight=sw_val, return_path=False,
            )

        def _cond_squared(loss_name, penalty_name, cv_solver, strict):
            return loss_name == "squared_error" and penalty_name in ("l1", "elasticnet", "en")

        def _path_squared(X_train, y_train, alpha_sorted, penalty_name, l1_ratio,
                          max_iter, tol, cv_device, X_val, y_val, sw_train, sw_val, **kw):
            return _squared_error_sparse_cv_path(
                X_train, y_train, alpha_sorted, penalty_name, l1_ratio,
                max_iter, tol, cv_device,
                X_val=X_val, y_val=y_val, sample_weight=sw_train,
                val_sample_weight=sw_val, return_path=False,
            )

        def _cond_glm_sparse(loss_name, penalty_name, cv_solver, strict):
            return self._uses_glm_sparse_path(penalty_name, cv_solver)

        def _path_glm_sparse(X_train, y_train, alpha_sorted, penalty_name, l1_ratio,
                             max_iter, tol, cv_device, X_val, y_val, sw_train, sw_val,
                             cv_solver=cv_solver, strict=strict):
            return _glm_sparse_cv_path(
                loss_name, X_train, y_train, alpha_sorted, penalty_name,
                l1_ratio, max_iter, tol, cv_device,
                X_val=X_val, y_val=y_val, sample_weight=sw_train,
                val_sample_weight=sw_val, return_path=False,
                solver_name=cv_solver, cv_mode=not strict,
            )

        _per_fold_paths = [
            (_cond_scad_mcp,    _path_scad_mcp),
            (_cond_logistic,    _path_logistic),
            (_cond_squared,     _path_squared),
            (_cond_glm_sparse,  _path_glm_sparse),
        ]

        # Pre-check which paths are active for this loss/penalty combo
        active_paths = [(cond, path_fn) for cond, path_fn in _per_fold_paths
                        if cond(loss_name, penalty_name, cv_solver, strict)]

        # ── Per-fold loop ──
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train = _slice_rows(X, train_idx)
            y_train = _slice_rows(y, train_idx)
            X_val = _slice_rows(X, val_idx)
            y_val = _slice_rows(y, val_idx)
            sw_train = _slice_rows(sample_weight, train_idx) if sample_weight is not None else None
            sw_val = _slice_rows(sample_weight, val_idx) if sample_weight is not None else None

            # Try each specialized path in order
            fold_handled = False
            for cond_fn, path_fn in active_paths:
                try:
                    path = path_fn(
                        X_train, y_train, alpha_sorted, penalty_name,
                        self.l1_ratio, max_iter, tol, cv_device,
                        X_val=X_val, y_val=y_val,
                        sw_train=sw_train, sw_val=sw_val,
                    )
                    if path is not None and path["scores"] is not None:
                        all_scores[fold_idx, sort_idx] = path["scores"]
                        fold_handled = True
                        break
                except Exception as e:
                    warnings.warn(
                        f"{path_fn.__name__} failed for {loss_name}+{penalty_name} "
                        f"fold {fold_idx}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

            if fold_handled:
                continue

            # ── General fallback: model.fit() per alpha ──
            self._cv_fold_general(
                all_scores, fold_idx, sort_idx, alpha_sorted,
                loss_name, cv_device, cv_solver, strict,
                X_train, y_train, X_val, y_val,
                sw_train, sw_val, max_iter, tol,
            )

        return all_scores

    def _cv_fold_general(
        self, all_scores, fold_idx, sort_idx, alpha_sorted,
        loss_name, cv_device, cv_solver, strict,
        X_train, y_train, X_val, y_val,
        sw_train, sw_val, max_iter, tol,
    ):
        """General per-fold CV path: model.fit() per alpha with warm-start."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel, _resolve_loss_name

        penalty_name = str(self.penalty).lower()
        device_name = _device_to_name(cv_device)

        X_val_np = _to_numpy(X_val).astype(np.float64)
        y_val_np = _to_numpy(y_val).astype(np.float64).ravel()
        loss_fn = _resolve_loss_name(loss_name)

        # Disable warm-start for SCAD/MCP on non-squared-error losses
        _is_scad_mcp_non_se = penalty_name in ("scad", "mcp") and loss_name != "squared_error"
        use_warm_start = not _is_scad_mcp_non_se
        use_lla_path_cv = (
            not strict and loss_name != "squared_error" and penalty_name in ("scad", "mcp")
        )

        # Transfer to GPU if needed
        if device_name in ("cuda", "torch"):
            fold_backend = _backend_name_for_cv_device(cv_device)
            X_train_fit = _to_backend_float64(X_train, fold_backend)
            y_train_fit = _to_backend_float64(y_train, fold_backend)
            sw_train_fit = _to_backend_float64(sw_train, fold_backend) if sw_train is not None else None
        else:
            X_train_fit = X_train
            y_train_fit = y_train
            sw_train_fit = sw_train

        # Precompute XtX/Xty for squared-error GPU cache
        cv_cache, L_np = self._build_cv_cache(
            loss_name, device_name, X_train, y_train, sw_train
        )

        model = PenalizedGeneralizedLinearModel(
            loss=loss_name, penalty=self.penalty, alpha=alpha_sorted[0],
            l1_ratio=self.l1_ratio, device=cv_device, compute_inference=False,
            max_iter=max_iter, tol=tol, solver=cv_solver,
        )
        if cv_cache is not None:
            model._cv_cache = cv_cache
            model._preserve_cv_cache = True
        if L_np is not None and L_np > 0:
            model.lipschitz_L = L_np

        # LLA path for SCAD/MCP
        if use_lla_path_cv:
            try:
                model.alpha = float(alpha_sorted[-1])
                if hasattr(model, "_penalty") and model._penalty is not None:
                    model._penalty.alpha = float(alpha_sorted[-1])
                model._cv_alpha_path = np.asarray(alpha_sorted, dtype=np.float64)
                model.fit(X_train_fit, y_train_fit, sample_weight=sw_train_fit)
                path = getattr(model, "_cv_path_results", None)
                if path is not None:
                    path_alphas = np.asarray(path["alpha"], dtype=np.float64)
                    path_coefs = np.asarray(path["coef"], dtype=np.float64)
                    path_intercepts = np.asarray(path["intercept"], dtype=np.float64)
                    for alpha_idx_sorted, alpha in enumerate(alpha_sorted):
                        matches = np.flatnonzero(np.isclose(path_alphas, float(alpha), rtol=1e-10, atol=1e-14))
                        if matches.size == 0:
                            continue
                        path_idx = int(matches[-1])
                        val_loss = _evaluate_loss_numpy(
                            loss_name, loss_fn, X_val_np, y_val_np,
                            path_coefs[path_idx], float(path_intercepts[path_idx]),
                            True, sample_weight=sw_val,
                        )
                        all_scores[fold_idx, sort_idx[alpha_idx_sorted]] = val_loss
                    for attr in ("_cv_alpha_path", "_cv_path_results", "_cv_cache", "_preserve_cv_cache"):
                        if hasattr(model, attr): delattr(model, attr)
                    return
                else:
                    # path is None — cleanup LLA state only; _cv_cache and
                    # _preserve_cv_cache are still needed for the warm-start fallback.
                    for attr in ("_cv_alpha_path", "_cv_path_results"):
                        if hasattr(model, attr): delattr(model, attr)
            except Exception:
                # Same as path-is-None: keep _cv_cache for warm-start fallback.
                for attr in ("_cv_alpha_path", "_cv_path_results"):
                    if hasattr(model, attr): delattr(model, attr)

        # Warm-started alpha loop
        prev_coef = None
        prev_intercept = None
        for alpha_idx_sorted, alpha in enumerate(alpha_sorted):
            try:
                if cv_cache is not None:
                    model._cv_cache = cv_cache
                model.alpha = alpha
                if hasattr(model, "_penalty") and model._penalty is not None:
                    model._penalty.alpha = alpha
                if use_warm_start and prev_coef is not None:
                    model._init_coef = np.asarray(prev_coef, dtype=np.float64)
                    model._init_intercept = prev_intercept
                else:
                    model._init_coef = None
                    model._init_intercept = None
                model.fit(X_train_fit, y_train_fit, sample_weight=sw_train_fit)

                coef_np = _to_numpy(model.coef_).ravel()
                intercept = float(model.intercept_)
                val_loss = _evaluate_loss_numpy(
                    loss_name, loss_fn, X_val_np, y_val_np,
                    coef_np, intercept, model.fit_intercept, sample_weight=sw_val,
                )
                orig_idx = sort_idx[alpha_idx_sorted]
                all_scores[fold_idx, orig_idx] = val_loss
                prev_coef = coef_np.copy()
                prev_intercept = intercept
            except Exception:
                orig_idx = sort_idx[alpha_idx_sorted]
                all_scores[fold_idx, orig_idx] = np.nan

        if hasattr(model, "_cv_cache"): del model._cv_cache
        if hasattr(model, "_preserve_cv_cache"): del model._preserve_cv_cache

    def _build_cv_cache(self, loss_name, device_name, X_train, y_train, sw_train):
        """Precompute XtX/Xty for squared-error GPU cache. Returns (cache_dict, L_np)."""
        if loss_name != "squared_error" or device_name not in ("cuda", "torch"):
            return None, None
        X_train_np = _to_numpy(X_train).astype(np.float64)
        y_train_np = _to_numpy(y_train).astype(np.float64).ravel()
        n_tr, _ = X_train_np.shape
        sw_np = _to_numpy(sw_train).astype(np.float64).ravel() if sw_train is not None else None
        if sw_np is not None:
            w_sum = float(sw_np.sum())
            X_mean_np = np.average(X_train_np, axis=0, weights=sw_np)
            y_mean_np = float(np.average(y_train_np, weights=sw_np))
            Xc_np = X_train_np - X_mean_np
            yc_np = y_train_np - y_mean_np
            sqrt_w = np.sqrt(sw_np)
            W_Xc = Xc_np * sqrt_w[:, None]
            XtX_np = W_Xc.T @ W_Xc
            Xty_np = (Xc_np * sw_np[:, None]).T @ yc_np
            L_np = float(np.max(np.linalg.eigvalsh(XtX_np))) / max(w_sum, 1.0)
            n_effective = w_sum
        else:
            X_mean_np = np.mean(X_train_np, axis=0)
            y_mean_np = np.mean(y_train_np)
            Xc_np = X_train_np - X_mean_np
            yc_np = y_train_np - y_mean_np
            XtX_np = Xc_np.T @ Xc_np
            Xty_np = Xc_np.T @ yc_np
            L_np = float(np.max(np.linalg.eigvalsh(XtX_np))) / n_tr
            n_effective = float(n_tr)
        if device_name == "cuda":
            import cupy as cp
            cache = {"XtX": cp.asarray(XtX_np), "Xty": cp.asarray(Xty_np), "n_effective": n_effective}
        else:
            import torch
            cache = {"XtX": torch.as_tensor(XtX_np, device="cuda", dtype=torch.float64),
                     "Xty": torch.as_tensor(Xty_np, device="cuda", dtype=torch.float64),
                     "n_effective": n_effective}
        return cache, L_np

    def fit(self, X, y, sample_weight=None):
        """Fit the CV model with optimized strict or explicit two-stage CV."""
        # Normalize array-like inputs (lists, tuples, etc.) to arrays
        if not hasattr(X, 'shape'):
            X = np.asarray(X, dtype=np.float64)
        if not hasattr(y, 'shape'):
            y = np.asarray(y, dtype=np.float64)

        if self._alpha_grid_input is not None:
            alpha_grid = np.asarray(self._alpha_grid_input, dtype=np.float64)
        else:
            alpha_grid = self._generate_alpha_grid(X, y)
        alpha_grid = np.asarray(alpha_grid, dtype=np.float64).ravel()

        self.alpha_grid_ = alpha_grid
        n_samples = X.shape[0]
        n_alphas = len(alpha_grid)
        penalty_name = str(self.penalty).lower()
        cv_device = self._effective_cv_device(X, penalty_name, n_alphas)
        cv_solver = self._solver_for_cv(cv_device, X=X)
        self.cv_strategy_ = self.cv_strategy
        self.cv_selected_device_ = _device_to_name(cv_device)

        if self.cv_splits is not None:
            # Normalize to list (generators would exhaust on first pass)
            folds = list(self.cv_splits) if not isinstance(self.cv_splits, list) else self.cv_splits
        else:
            folds = kfold_indices(n_samples, self.cv, self.random_state)
        all_scores_stage1 = None
        mean_scores_stage1 = None
        refined_mask = np.ones(n_alphas, dtype=bool)

        if self.cv_strategy == "two_stage":
            if not self.acknowledge_approx:
                warnings.warn(
                    "PenalizedGLM_CV(cv_strategy='two_stage') uses relaxed CV "
                    "solves to screen the alpha grid before strict refinement. "
                    "The final refit still uses the original max_iter and tol. "
                    "Pass acknowledge_approx=True to silence this warning.",
                    ApproximateCVWarning,
                    stacklevel=2,
                )
            stage1_max_iter = min(int(self.max_iter), max(50, int(self.max_iter) // 4))
            stage1_tol = max(float(self.tol) * 10.0, 1e-4)
            all_scores_stage1 = self._compute_cv_scores(
                X,
                y,
                alpha_grid,
                cv_device,
                folds,
                sample_weight=sample_weight,
                max_iter=stage1_max_iter,
                tol=stage1_tol,
                strict=False,
            )
            mean_scores_stage1 = np.nanmean(all_scores_stage1, axis=0)
            refined_mask = _two_stage_candidate_mask(
                mean_scores_stage1,
                refine_top_k=self.refine_top_k,
            )
            if self.loss == "squared_error" and penalty_name in ("scad", "mcp"):
                refined_mask[:] = True
            if not np.any(refined_mask):
                refined_mask[:] = True

            refined_alpha_grid = alpha_grid[refined_mask]
            refined_scores = self._compute_cv_scores(
                X,
                y,
                refined_alpha_grid,
                cv_device,
                folds,
                sample_weight=sample_weight,
                max_iter=self.max_iter,
                tol=self.tol,
                strict=True,
            )
            all_scores = np.array(all_scores_stage1, copy=True)
            all_scores[:, refined_mask] = refined_scores
            mean_scores = np.nanmean(all_scores, axis=0)
            refined_mean = np.nanmean(refined_scores, axis=0)
            refined_best = self._best_index_from_scores(
                refined_mean,
                refined_alpha_grid,
                cv_solver,
            )
            best_idx = int(np.flatnonzero(refined_mask)[refined_best])
        else:
            all_scores = self._compute_cv_scores(
                X,
                y,
                alpha_grid,
                cv_device,
                folds,
                sample_weight=sample_weight,
                max_iter=self.max_iter,
                tol=self.tol,
                strict=True,
            )
            mean_scores = np.nanmean(all_scores, axis=0)
            best_idx = self._best_index_from_scores(mean_scores, alpha_grid, cv_solver)

        best_alpha = float(alpha_grid[best_idx])
        self.alpha_ = best_alpha
        self.best_score_ = float(mean_scores[best_idx])
        self.cv_results_ = {
            "alpha": alpha_grid,
            "mean_score": mean_scores,
            "all_scores": all_scores,
            "cv_strategy_": self.cv_strategy_,
            "cv_selected_device_": self.cv_selected_device_,
            "mean_score_stage1": mean_scores_stage1,
            "all_scores_stage1": all_scores_stage1,
            "refined_mask": refined_mask,
        }

        self.estimator_ = self._refit_best(X, y, best_alpha, sample_weight=sample_weight)
        self.coef_ = self.estimator_.coef_
        self.intercept_ = self.estimator_.intercept_

        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the refit estimator with the best alpha."""
        if not getattr(self, '_fitted', False):
            raise RuntimeError("PenalizedGLM_CV is not fitted yet. Call fit() first.")
        return self.estimator_.predict(X)

    def score(self, X, y, sample_weight=None):
        """Return the mean log-likelihood score on the given data."""
        if not getattr(self, '_fitted', False):
            raise RuntimeError("PenalizedGLM_CV is not fitted yet. Call fit() first.")
        return self.estimator_.score(X, y, sample_weight=sample_weight)
