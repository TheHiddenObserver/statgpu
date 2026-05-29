"""
Unified solvers for GLMLoss + Penalty optimization.

minimize: loss(X, y, w) + penalty(w)

Supports numpy / cupy / torch backends via auto-detection.
"""

import warnings
import numpy as np

from statgpu.backends import _to_numpy


class ConvergenceWarning(UserWarning):
    """Solver did not converge within the iteration limit."""
    pass


# =============================================================================
# torch.compile for FISTA/Newton elementwise ops
# Falls back to eager mode on GPUs with CUDA capability < 7.0
# =============================================================================

_FISTA_STEP_COMPILED = None
_NEWTON_STEP_COMPILED = None


def _torch_compile_supported():
    """Check if torch.compile is safe to use (CUDA Capability >= 7.0)."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 7
    except Exception:
        pass
    return True  # Assume supported if we can't check


def _get_fista_step_compiled():
    """Lazily create a torch.compile'd FISTA step function."""
    global _FISTA_STEP_COMPILED
    if _FISTA_STEP_COMPILED is not None:
        return _FISTA_STEP_COMPILED
    import torch
    def _fista_step(y_k, grad, step, coef_old, coef, beta_t):
        w_tilde = y_k - step * grad
        y_k_new = coef + beta_t * (coef - coef_old)
        return w_tilde, y_k_new
    if _torch_compile_supported():
        try:
            _FISTA_STEP_COMPILED = torch.compile(_fista_step, dynamic=True, fullgraph=False)
        except Exception:
            _FISTA_STEP_COMPILED = _fista_step
    else:
        _FISTA_STEP_COMPILED = _fista_step
    return _FISTA_STEP_COMPILED


def _fista_step_call(compiled_fn, *args):
    """Call compiled FISTA step, falling back to eager on GPU arch mismatch."""
    try:
        return compiled_fn(*args)
    except Exception:
        # Runtime fallback (shouldn't happen if pre-check passed)
        def _fista_eager(y_k, grad, step, coef_old, coef, beta_t):
            w_tilde = y_k - step * grad
            y_k_new = coef + beta_t * (coef - coef_old)
            return w_tilde, y_k_new
        return _fista_eager(*args)


def _get_newton_step_compiled():
    """Lazily create a torch.compile'd Newton step function."""
    global _NEWTON_STEP_COMPILED
    if _NEWTON_STEP_COMPILED is not None:
        return _NEWTON_STEP_COMPILED
    import torch
    def _newton_step(params, direction, params_old):
        params_new = params - direction
        diff_norm = torch.linalg.norm(params_new - params_old)
        return params_new, diff_norm
    if _torch_compile_supported():
        try:
            _NEWTON_STEP_COMPILED = torch.compile(_newton_step, dynamic=True, fullgraph=False)
        except Exception:
            _NEWTON_STEP_COMPILED = _newton_step
    else:
        _NEWTON_STEP_COMPILED = _newton_step
    return _NEWTON_STEP_COMPILED


def _newton_step_call(compiled_fn, *args):
    """Call compiled Newton step, falling back to eager on GPU arch mismatch."""
    try:
        return compiled_fn(*args)
    except Exception:
        def _newton_eager(params, direction, params_old):
            params_new = params - direction
            diff_norm = torch.linalg.norm(params_new - params_old)
            return params_new, diff_norm
        return _newton_eager(*args)


def _infer_backend(X):
    """Detect backend from array type."""
    mod = type(X).__module__
    if mod.startswith("cupy"):
        return "cupy"
    if mod.startswith("torch"):
        return "torch"
    return "numpy"


def _zeros_like(arr):
    """Create zeros array with same shape/type as arr."""
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    mod = type(arr).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return cp.zeros_like(arr)
    if mod.startswith("torch"):
        import torch
        return torch.zeros_like(arr)
    return np.zeros_like(arr)


def _zeros(n, backend, ref_tensor=None):
    """Create a 1-D zeros vector of length n on the correct device/dtype."""
    if backend == "numpy":
        return np.zeros(n)
    if backend == "cupy":
        import cupy as cp
        dtype = getattr(ref_tensor, 'dtype', cp.float64)
        return cp.zeros(n, dtype=dtype)
    import torch
    device = getattr(ref_tensor, 'device', 'cpu') if ref_tensor is not None else 'cpu'
    dtype = getattr(ref_tensor, 'dtype', torch.float64) if ref_tensor is not None else torch.float64
    return torch.zeros(n, device=device, dtype=dtype)


def _sync_scalars(*dev_vals, backend):
    """Batch multiple device scalars into a single GPU→CPU sync.

    Returns a tuple of Python floats.  For numpy the values are already on CPU;
    for torch/cupy a single stack+transfer replaces N individual transfers.
    """
    if backend == "numpy":
        return tuple(float(v) for v in dev_vals)
    if backend == "torch":
        import torch
        stacked = torch.stack(list(dev_vals))
        return tuple(stacked[i].item() for i in range(len(dev_vals)))
    import cupy as cp
    stacked = cp.array(list(dev_vals))
    return tuple(float(stacked[i]) for i in range(len(dev_vals)))


def _abs_sum(x):
    """Sum of absolute values."""
    if isinstance(x, np.ndarray):
        return np.sum(np.abs(x))
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return float(cp.sum(cp.abs(x)))
    if mod.startswith("torch"):
        import torch
        return float(torch.sum(torch.abs(x)).item())
    return np.sum(np.abs(x))


def _abs_max(x):
    """Max of absolute values (L-infinity norm)."""
    if isinstance(x, np.ndarray):
        return float(np.max(np.abs(x)))
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return float(cp.max(cp.abs(x)))
    if mod.startswith("torch"):
        import torch
        return float(torch.max(torch.abs(x)).item())
    return float(np.max(np.abs(x)))


def _norm2(x):
    """L2 norm."""
    if isinstance(x, np.ndarray):
        return float(np.linalg.norm(x))
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return float(cp.linalg.norm(x))
    if mod.startswith("torch"):
        import torch
        return float(torch.linalg.norm(x).item())
    return float(np.linalg.norm(x))


def _dot(a, b):
    """Dot product (returns Python float — forces GPU→CPU sync)."""
    if isinstance(a, np.ndarray):
        return float(a.dot(b))
    mod = type(a).__module__
    if mod.startswith("cupy"):
        return float(a.dot(b))
    if mod.startswith("torch"):
        return float(a.dot(b))
    return float(a.dot(b))


def _dot_dev(a, b):
    """Dot product staying on device (no GPU→CPU sync)."""
    if isinstance(a, np.ndarray):
        return float(a.dot(b))
    return a.dot(b)


def _sum_sq(x):
    """Sum of squares (returns Python float — forces GPU→CPU sync)."""
    if isinstance(x, np.ndarray):
        return float(np.sum(x ** 2))
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return float(cp.sum(x ** 2))
    if mod.startswith("torch"):
        import torch
        return float(torch.sum(x ** 2))
    return float(np.sum(x ** 2))


def _sum_sq_dev(x):
    """Sum of squares staying on device (no GPU→CPU sync)."""
    if isinstance(x, np.ndarray):
        return float(np.sum(x ** 2))
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return cp.sum(x ** 2)
    if mod.startswith("torch"):
        import torch
        return torch.sum(x ** 2)
    return float(np.sum(x ** 2))


def _norm2_dev(x):
    """L2 norm staying on device (no GPU→CPU sync)."""
    if isinstance(x, np.ndarray):
        return float(np.linalg.norm(x))
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return cp.linalg.norm(x)
    if mod.startswith("torch"):
        import torch
        return torch.linalg.norm(x)
    return float(np.linalg.norm(x))


def _abs_sum_dev(x):
    """Sum of absolute values staying on device (no GPU→CPU sync)."""
    if isinstance(x, np.ndarray):
        return float(np.sum(np.abs(x)))
    mod = type(x).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return cp.sum(cp.abs(x))
    if mod.startswith("torch"):
        import torch
        return torch.sum(torch.abs(x))
    return float(np.sum(np.abs(x)))


def _clip_grad_on_device(grad, coef_old, backend):
    """Clip gradient entirely on device — no GPU→CPU sync.

    Used ONLY in the async GPU loop (non-smooth penalties) where
    per-iteration sync is avoided.  For smooth penalties (backtracking),
    use sync-based clipping in the main loop instead.

    Clipping threshold: max(||coef||_1 * 10 + 1e3, 1e4).
    If ||grad||_2 exceeds threshold, scale grad down.
    """
    if backend == "numpy":
        gn = float(np.linalg.norm(grad))
        ca = float(np.sum(np.abs(coef_old)))
        gmax = max(ca * 10.0 + 1e3, 1e4)
        if gn > gmax:
            return grad * (gmax / gn)
        return grad
    if backend == "torch":
        import torch
        gn_sq = torch.sum(grad ** 2)
        coef_abs = torch.sum(torch.abs(coef_old))
        gmax = coef_abs * 10.0 + 1e3
        gmax = torch.clamp(gmax, min=1e4)  # match CPU: max(..., 1e4)
        # Use where to avoid branching on scalar
        scale = torch.where(
            gn_sq > gmax * gmax,
            gmax / torch.sqrt(gn_sq + 1e-30),
            torch.ones(1, device=grad.device, dtype=grad.dtype))
        return grad * scale
    # cupy
    import cupy as cp
    gn_sq = cp.sum(grad ** 2)
    coef_abs = cp.sum(cp.abs(coef_old))
    gmax = coef_abs * 10.0 + 1e3
    gmax = cp.maximum(gmax, 1e4)  # match CPU: max(..., 1e4)
    scale = cp.where(
        gn_sq > gmax * gmax,
        gmax / cp.sqrt(gn_sq + 1e-30),
        cp.ones(1, dtype=grad.dtype))
    return grad * scale


def _copy_arr(arr):
    """Copy array: .clone() for torch, .copy() for numpy/cupy."""
    if hasattr(arr, 'clone'):
        return arr.clone()
    return arr.copy()


def _eye_like(n, ref):
    """Create an identity matrix on the same backend/device as ref."""
    backend = _infer_backend(ref)
    if backend == "cupy":
        import cupy as cp
        return cp.eye(n, dtype=ref.dtype)
    if backend == "torch":
        import torch
        return torch.eye(n, dtype=ref.dtype, device=ref.device)
    return np.eye(n, dtype=getattr(ref, "dtype", np.float64))


def _penalty_name(penalty):
    return str(getattr(penalty, "name", "none")).lower()


def _smooth_penalty_value(penalty, coef):
    """Return smooth penalty value without moving the full vector to CPU."""
    if penalty is None:
        return 0.0
    if hasattr(penalty, "smooth_value"):
        return float(_to_numpy(penalty.smooth_value(coef)))
    if _penalty_name(penalty) in ("none", "null"):
        return 0.0
    if _penalty_name(penalty) == "l2":
        return 0.5 * float(getattr(penalty, "alpha", 0.0)) * _sum_sq(coef)
    if _penalty_name(penalty) == "elasticnet":
        alpha = float(getattr(penalty, "alpha", 0.0))
        l1_ratio = float(getattr(penalty, "l1_ratio", 1.0))
        return 0.5 * alpha * (1.0 - l1_ratio) * _sum_sq(coef)
    raise ValueError(
        f"solver requires a smooth penalty, got penalty='{_penalty_name(penalty)}'."
    )


def _smooth_penalty_gradient(penalty, coef):
    """Return smooth penalty gradient on the same backend as coef."""
    if penalty is None or _penalty_name(penalty) in ("none", "null"):
        return _zeros_like(coef)
    if hasattr(penalty, "smooth_gradient"):
        return penalty.smooth_gradient(coef)
    if _penalty_name(penalty) == "l2":
        return float(getattr(penalty, "alpha", 0.0)) * coef
    if _penalty_name(penalty) == "elasticnet":
        alpha = float(getattr(penalty, "alpha", 0.0))
        l1_ratio = float(getattr(penalty, "l1_ratio", 1.0))
        return alpha * (1.0 - l1_ratio) * coef
    raise ValueError(
        f"solver requires a smooth penalty, got penalty='{_penalty_name(penalty)}'."
    )


def _smooth_penalty_hessian(penalty, coef):
    """Return smooth penalty Hessian on the same backend as coef."""
    n = coef.shape[0]
    if penalty is None or _penalty_name(penalty) in ("none", "null"):
        return 0.0
    if hasattr(penalty, "smooth_hessian"):
        return penalty.smooth_hessian(coef)
    if _penalty_name(penalty) == "l2":
        return float(getattr(penalty, "alpha", 0.0)) * _eye_like(n, coef)
    raise ValueError(
        f"solver requires a smooth penalty, got penalty='{_penalty_name(penalty)}'."
    )


def _objective_value(loss, penalty, X, y, coef):
    return float(_to_numpy(loss.value(X, y, coef))) + _smooth_penalty_value(
        penalty, coef
    )


def _objective_gradient(loss, penalty, X, y, coef):
    return loss.gradient(X, y, coef) + _smooth_penalty_gradient(penalty, coef)


def _smooth_penalty_lipschitz(penalty):
    """Return the Lipschitz constant of the smooth penalty gradient.

    For l2: gradient = alpha * coef → Lipschitz = alpha.
    For ElasticNet: smooth part gradient = alpha*(1-l1_ratio)*coef → Lipschitz = alpha*(1-l1_ratio).
    For pure L1: no smooth part → Lipschitz = 0.
    """
    if penalty is None:
        return 0.0
    _pname = _penalty_name(penalty)
    if _pname in ("none", "null", "l1", "scad", "mcp", "adaptive_l1", "adaptive_lasso",
                  "group_lasso", "group_mcp", "group_scad", "gl", "gmcp", "gscad"):
        return 0.0
    alpha = float(getattr(penalty, 'alpha', 0.0))
    l1_ratio = float(getattr(penalty, 'l1_ratio', 0.0))
    return alpha * (1.0 - l1_ratio)


# =============================================================================
# Device-side helpers (no GPU→CPU sync)
# =============================================================================

def _smooth_penalty_value_dev(penalty, coef):
    """Smooth penalty value staying on device (no GPU→CPU sync)."""
    if penalty is None:
        return 0.0
    pname = _penalty_name(penalty)
    if pname in ("none", "null"):
        return 0.0
    if pname == "l2":
        return 0.5 * float(getattr(penalty, "alpha", 0.0)) * _sum_sq_dev(coef)
    if pname == "elasticnet":
        alpha = float(getattr(penalty, "alpha", 0.0))
        l1_ratio = float(getattr(penalty, "l1_ratio", 1.0))
        return 0.5 * alpha * (1.0 - l1_ratio) * _sum_sq_dev(coef)
    raise ValueError(
        f"smooth_penalty_value_dev requires a smooth penalty, got '{pname}'."
    )


def _objective_value_dev(loss, penalty, X, y, coef):
    """Objective value staying on device (no GPU→CPU sync)."""
    val = loss.value(X, y, coef)
    pen_val = _smooth_penalty_value_dev(penalty, coef)
    if isinstance(pen_val, (int, float)) and pen_val == 0.0:
        return val
    return val + pen_val


def _device_leq(a, b):
    """Device-side a <= b comparison (returns Python bool, single sync)."""
    mod = type(a).__module__
    if mod.startswith('torch'):
        return bool((a <= b).item())
    if mod.startswith('cupy'):
        return bool(a <= b)
    return a <= b


def _device_gt(a, b):
    """Device-side a > b comparison (returns Python bool, single sync)."""
    mod = type(a).__module__
    if mod.startswith('torch'):
        return bool((a > b).item())
    if mod.startswith('cupy'):
        return bool(a > b)
    return a > b


def _fused_glm_value_and_gradient(loss, X, y, coef):
    """Compute GLM loss value and gradient in one pass, avoiding redundant X @ coef.

    For GLM losses, the value() and gradient() both need eta = X @ coef.
    This function computes eta once and reuses it for both.
    Returns (value, gradient) both on the same backend as coef.
    """
    loss_name = getattr(loss, 'name', '')
    n = X.shape[0]

    if loss_name == 'logistic':
        import numpy as np
        eta = X @ coef
        # Sigmoid: p = 1 / (1 + exp(-eta))
        mod = type(eta).__module__
        if mod.startswith('torch'):
            import torch
            p = torch.sigmoid(eta)
            val = torch.sum(-y * eta + torch.log1p(torch.exp(-torch.abs(eta))) + torch.clamp(eta, min=0)) / n
            grad = X.T @ (p - y) / n
        elif mod.startswith('cupy'):
            import cupy as cp
            p = 1.0 / (1.0 + cp.exp(-cp.clip(eta, -500, 500)))
            log1pexp = cp.log1p(cp.exp(-cp.abs(eta))) + cp.maximum(eta, 0)
            val = cp.sum(-y * eta + log1pexp) / n
            grad = X.T @ (p - y) / n
        else:
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
            log1pexp = np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0)
            val = np.sum(-y * eta + log1pexp) / n
            grad = X.T @ (p - y) / n
        return val, grad

    elif loss_name == 'poisson':
        import numpy as np
        eta = X @ coef
        mod = type(eta).__module__
        if mod.startswith('torch'):
            import torch
            mu = torch.exp(torch.clamp(eta, -30, 30))
            val = torch.sum(mu - y * torch.log(mu + 1e-10)) / n
            grad = X.T @ (mu - y) / n
        elif mod.startswith('cupy'):
            import cupy as cp
            mu = cp.exp(cp.clip(eta, -30, 30))
            val = cp.sum(mu - y * cp.log(mu + 1e-10)) / n
            grad = X.T @ (mu - y) / n
        else:
            mu = np.exp(np.clip(eta, -30, 30))
            val = np.sum(mu - y * np.log(mu + 1e-10)) / n
            grad = X.T @ (mu - y) / n
        return val, grad

    elif loss_name == 'gamma':
        import numpy as np
        eta = X @ coef
        mod = type(eta).__module__
        if mod.startswith('torch'):
            import torch
            mu = torch.exp(torch.clamp(eta, -30, 30))
            mu_c = torch.clamp(mu, min=1e-3, max=1e4)
            val = torch.sum(y / mu_c + torch.log(mu_c)) / n
            grad = X.T @ (1.0 - y / mu_c) / n
        elif mod.startswith('cupy'):
            import cupy as cp
            mu = cp.exp(cp.clip(eta, -30, 30))
            mu_c = cp.clip(mu, 1e-3, 1e4)
            val = cp.sum(y / mu_c + cp.log(mu_c)) / n
            grad = X.T @ (1.0 - y / mu_c) / n
        else:
            mu = np.exp(np.clip(eta, -30, 30))
            mu_c = np.clip(mu, 1e-3, 1e4)
            val = np.sum(y / mu_c + np.log(mu_c)) / n
            grad = X.T @ (1.0 - y / mu_c) / n
        return val, grad

    elif loss_name == 'negative_binomial':
        import numpy as np
        eta = X @ coef
        mod = type(eta).__module__
        a = 1.0  # dispersion parameter
        if mod.startswith('torch'):
            import torch
            mu = torch.exp(torch.clamp(eta, -30, 30))
            mu_c = torch.clamp(mu, min=1e-300)
            a_plus_mu = a + mu_c
            val = torch.sum(-y * torch.log(mu_c / a_plus_mu) - (1.0 / a) * torch.log(a / a_plus_mu)) / n
            grad = X.T @ ((mu_c - y) / (1.0 + a * mu_c)) / n
        elif mod.startswith('cupy'):
            import cupy as cp
            mu = cp.exp(cp.clip(eta, -30, 30))
            mu_c = cp.clip(mu, 1e-300, None)
            a_plus_mu = a + mu_c
            val = cp.sum(-y * cp.log(mu_c / a_plus_mu) - (1.0 / a) * cp.log(a / a_plus_mu)) / n
            grad = X.T @ ((mu_c - y) / (1.0 + a * mu_c)) / n
        else:
            mu = np.exp(np.clip(eta, -30, 30))
            mu_c = np.clip(mu, 1e-300, None)
            a_plus_mu = a + mu_c
            val = np.sum(-y * np.log(mu_c / a_plus_mu) - (1.0 / a) * np.log(a / a_plus_mu)) / n
            grad = X.T @ ((mu_c - y) / (1.0 + a * mu_c)) / n
        return val, grad

    elif loss_name == 'tweedie':
        import numpy as np
        eta = X @ coef
        mod = type(eta).__module__
        p = 1.5  # Tweedie variance power
        if mod.startswith('torch'):
            import torch
            mu = torch.exp(torch.clamp(eta, -50, 50))
            mu_c = torch.clamp(mu, min=1e-3, max=1e4)
            val = torch.sum(-y * mu_c.pow(1 - p) / (1 - p) + mu_c.pow(2 - p) / (2 - p)) / n
            grad = X.T @ (mu_c.pow(1 - p) * (mu_c - y)) / n
        elif mod.startswith('cupy'):
            import cupy as cp
            mu = cp.exp(cp.clip(eta, -50, 50))
            mu_c = cp.clip(mu, 1e-3, 1e4)
            val = cp.sum(-y * mu_c ** (1 - p) / (1 - p) + mu_c ** (2 - p) / (2 - p)) / n
            grad = X.T @ (mu_c ** (1 - p) * (mu_c - y)) / n
        else:
            mu = np.exp(np.clip(eta, -50, 50))
            mu_c = np.clip(mu, 1e-3, 1e4)
            val = np.sum(-y * mu_c ** (1 - p) / (1 - p) + mu_c ** (2 - p) / (2 - p)) / n
            grad = X.T @ (mu_c ** (1 - p) * (mu_c - y)) / n
        return val, grad

    elif loss_name == 'inverse_gaussian':
        import numpy as np
        eta = X @ coef
        mod = type(eta).__module__
        if mod.startswith('torch'):
            import torch
            mu = torch.exp(torch.clamp(eta, -30, 30))
            mu_c = torch.clamp(mu, min=5e-2, max=1e3)
            val = torch.sum(y / (2 * mu_c.pow(2)) - 1.0 / mu_c) / n
            grad = X.T @ ((mu_c - y) / (mu_c * mu_c)) / n
        elif mod.startswith('cupy'):
            import cupy as cp
            mu = cp.exp(cp.clip(eta, -30, 30))
            mu_c = cp.clip(mu, 5e-2, 1e3)
            val = cp.sum(y / (2 * mu_c ** 2) - 1.0 / mu_c) / n
            grad = X.T @ ((mu_c - y) / (mu_c * mu_c)) / n
        else:
            mu = np.exp(np.clip(eta, -30, 30))
            mu_c = np.clip(mu, 5e-2, 1e3)
            val = np.sum(y / (2 * mu_c ** 2) - 1.0 / mu_c) / n
            grad = X.T @ ((mu_c - y) / (mu_c * mu_c)) / n
        return val, grad

    else:
        # Fallback: call value() and gradient() separately
        return loss.value(X, y, coef), loss.gradient(X, y, coef)


def fista_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=1000,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
    lipschitz_L=None,
):
    """General FISTA solver with backtracking line search.

    Supports numpy / cupy / torch backends via auto-detection of X.

    Parameters
    ----------
    loss : GLMLoss
        GLM loss function with gradient(), lipschitz(), preprocess(), value().
    penalty : Penalty
        Penalty with proximal().
    X : array
        Design matrix (numpy/cupy/torch).
    y : array
        Target (numpy/cupy/torch).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    init_coef : array, optional
        Initial coefficient vector.
    sample_weight : array, optional
        Sample weights (added to loss gradient as multiplier).

    Returns
    -------
    coef : array
        Fitted coefficients (same backend as X).
    n_iter : int
        Number of iterations.
    """
    backend = _infer_backend(X)
    X_proc, y_proc = loss.preprocess(X, y)
    _loss_name = getattr(loss, 'name', '')
    _is_quadratic = (_loss_name == "squared_error")
    # Exp-link families where Nesterov extrapolation can cause mu = exp(X@w)
    # to explode (Poisson: grad ~ mu) or vanish into extreme oscillation
    # (Inverse Gaussian: grad ~ 1/mu^3).  Gamma and Tweedie have self-
    # stabilizing 1/mu-type gradient scaling and are safe with momentum.
    # Disable momentum entirely for inverse_gaussian (1/mu^3 scaling).
    # Use conservative momentum for Poisson and negative_binomial
    # (exp-link families where Nesterov can cause mu explosion).
    _no_momentum = ()
    # Conservative momentum (cap beta at 0.5) for exp-link families and
    # for logistic/gamma with non-smooth penalties.  Logistic/gamma with
    # smooth penalties (none, l2) benefit from full Nesterov acceleration.
    _non_smooth_pen = getattr(penalty, 'name', '') in (
        "l1", "elasticnet", "en", "scad", "mcp", "adaptive_l1", "adaptive_lasso",
        "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
    )
    _conservative_momentum = (
        _loss_name in ("poisson", "negative_binomial", "tweedie", "inverse_gaussian")
        or (_loss_name in ("logistic", "gamma") and _non_smooth_pen)
    )

    n_features = X_proc.shape[1]
    if init_coef is not None:
        coef = _copy_arr(init_coef) if hasattr(init_coef, 'copy') or hasattr(init_coef, 'clone') else np.array(init_coef).copy()
    else:
        if backend == "numpy":
            coef = np.zeros(n_features)
        elif backend == "cupy":
            import cupy as cp
            coef = cp.zeros(n_features, dtype=X.dtype if hasattr(X, 'dtype') else cp.float64)
        else:
            import torch
            coef = torch.zeros(n_features, device=X.device if hasattr(X, 'device') else 'cpu', dtype=X.dtype if hasattr(X, 'dtype') else torch.float64)

    y_k = _copy_arr(coef)
    t_k = 1.0

    # Divergence detection: track best objective for recovery
    _obj_best_fista = float('inf')
    _coef_best_fista = None
    # Objective-based restart for Nesterov momentum
    _prev_obj_fista = None

    # Initial Lipschitz at zero (safe for all losses).  Computing L at
    # init_coef can produce enormous values for exp-link families (mu =
    # exp(X@coef) explodes for warm-start coefs from OLS), causing step
    # = 1/L to be zero and the solver to exit immediately.
    if lipschitz_L is not None and lipschitz_L > 0:
        L = lipschitz_L
    else:
        _zero_coef = _copy_arr(coef) * 0.0
        L = loss.lipschitz(X_proc, _zero_coef, y=y_proc)
    if L <= 0:
        L = 1.0
    # Add smooth penalty Lipschitz contribution (e.g. l2 penalty gradient
    # alpha*coef has Lipschitz constant alpha).  Without this, the step
    # size 1/L is too large, causing oscillation near the optimum.
    _smooth_lip = _smooth_penalty_lipschitz(penalty)
    if _smooth_lip > 0:
        L = L + _smooth_lip
    # For GLM losses with exp link (Poisson, etc.), mu at coef=0
    # is ~1, but mu near the optimum ≈ y.  Scale Lipschitz up by a
    # geometric-mean factor to avoid oversized first steps that cause
    # divergence on non-smooth penalties (scad, mcp, etc.).
    # Logistic now uses iterate-dependent Lipschitz, so y-scaling applies.
    # Gamma's expected Fisher Hessian X'X/n underestimates
    # true curvature by ~mean(y), so y-scaling IS needed.
    _loss_global_lip = False
    _skip_y_scaling = getattr(loss, '_lipschitz_uses_y', False)
    if not _is_quadratic and not _loss_global_lip and not _skip_y_scaling:
        _y_arr = _to_numpy(y_proc)
        _y_abs = np.abs(_y_arr)
        _y_mean = float(np.mean(_y_abs))
        _y_max = float(np.max(_y_abs))
        _y_scale = max(1.0, _y_mean, np.sqrt(_y_mean * _y_max))
        if _y_scale > 1.0:
            L = L * _y_scale

    # Inverse Gaussian: gradient scales as 1/mu^3, causing extreme
    # sensitivity to step size.  Use a much more conservative Lipschitz.
    if _loss_name == "inverse_gaussian":
        L = L * 3.0
    # Tweedie (p=1.5): loss landscape steepens exponentially — 5x safety.
    if _loss_name == "tweedie":
        L = L * 5.0
    # Gamma: loss = y/mu + log(mu), gradient = (1 - y/mu)/mu. When mu is small
    # (large eta), the Hessian ~y/mu^3 can be very large, causing divergence.
    # 3x safety to handle non-smooth (L1) proximal jumps.
    if _loss_name == "gamma":
        L = L * 3.0
    # Async GPU loop: skip backtracking, deferred checks.
    # For non-smooth penalties (l1, elasticnet, scad, mcp, adaptive, group):
    #   - Quadratic losses (squared_error): Lipschitz is exact, fixed step is optimal
    #   - GLM losses: use 3x safety factor on Lipschitz, no backtracking
    # Smooth penalties (l2, none) need backtracking for GLM losses.
    _pen_name_lower = _penalty_name(penalty)
    _non_smooth = _pen_name_lower not in ("none", "null", "l2", "")
    _use_gpu_loop = backend in ("torch", "cupy") and _non_smooth
    _is_gpu = backend in ("torch", "cupy")
    _conv_interval = 3  # check convergence every N iterations (GPU path)
    _div_interval = 5   # check divergence every N iterations (GPU path)

    for iteration in range(max_iter):
        coef_old = _copy_arr(coef)

        # Compute gradient (fused value+gradient for GLM losses)
        if _loss_name in ('logistic', 'poisson', 'gamma', 'negative_binomial', 'tweedie', 'inverse_gaussian'):
            q_yk_dev, grad = _fused_glm_value_and_gradient(loss, X_proc, y_proc, y_k)
        else:
            q_yk_dev = loss.value(X_proc, y_proc, y_k)
            grad = loss.gradient(X_proc, y_proc, y_k)

        if sample_weight is not None:
            if backend == "cupy":
                import cupy as cp
                sw = cp.asarray(sample_weight)
            elif backend == "torch":
                import torch
                sw = torch.tensor(sample_weight, device=grad.device, dtype=grad.dtype)
            else:
                sw = np.asarray(sample_weight)
            if grad.ndim > 1:
                grad = grad * sw[:, np.newaxis] if backend == "numpy" else grad * sw[:, None]
            else:
                grad = grad * sw

        if _use_gpu_loop:
            # ── GPU async path: all ops stay on device ──
            grad = _clip_grad_on_device(grad, coef_old, backend)

            step = 1.0 / L

            # Single proximal step — no backtracking (L is conservative enough)
            w_tilde = y_k - step * grad
            coef = penalty.proximal(w_tilde, step, backend=backend)

            # ALL safety checks deferred — no per-iteration GPU→CPU sync.
            # Finiteness + divergence + objective tracking batched together.
            if iteration > 0 and (iteration < 20 or iteration % _div_interval == 0):
                _obj_dev = loss.value(X_proc, y_proc, coef)
                if backend == "torch":
                    import torch
                    _all_finite = bool(torch.isfinite(_obj_dev).item())
                else:
                    import cupy as cp
                    _all_finite = bool(cp.isfinite(_obj_dev).item())
                if not _all_finite:
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                    else:
                        coef = _zeros(n_features, backend, ref_tensor=X)
                    y_k = _copy_arr(coef)
                    t_k = 1.0
                    L = L * 2.0
                    continue
                # Track best objective
                _obj_val_f = float(_to_numpy(_obj_dev))
                if _smooth_lip > 0:
                    _obj_val_f += _smooth_penalty_value(penalty, coef)
                if _obj_val_f < _obj_best_fista:
                    _obj_best_fista = _obj_val_f
                    _coef_best_fista = _copy_arr(coef)
                # Periodic Lipschitz recomputation (piggyback on same sync)
                # Skip for quadratic losses — Lipschitz is constant (spectral norm of X^T X).
                # Interval matches CPU path (line 929) for trajectory consistency.
                if not _is_quadratic and iteration % 5 == 0:
                    L_new = loss.lipschitz(X_proc, coef, y=y_proc)
                    if L_new > 0:
                        if _loss_name == "tweedie":
                            L_new *= 5.0
                        elif _loss_name == "gamma":
                            L_new *= 3.0
                        elif _loss_name == "inverse_gaussian":
                            L_new *= 3.0
                        if _smooth_lip > 0:
                            L_new = L_new + _smooth_lip
                        if L_new > L:
                            L = L_new
                        else:
                            L = max(L * 0.8, L_new)


        else:
            # ── CPU/GPU path with backtracking (smooth penalties) ──
            # Use identical sync-based clipping for both CPU and GPU.
            # (Backtracking already syncs every iteration for slack check,
            #  so on-device clipping has no performance benefit here.)
            _gn_f, _coef_abs_f = _sync_scalars(
                _norm2_dev(grad), _abs_sum_dev(coef_old), backend=backend)
            _gmax = max(_coef_abs_f * 10.0 + 1e3, 1e4)
            if _gn_f > _gmax:
                grad = grad * (_gmax / _gn_f)

            step = 1.0 / L
            _q_new_dev_last = None
            for _bt in range(20):
                w_tilde = y_k - step * grad
                coef_new = penalty.proximal(w_tilde, step, backend=backend)

                diff = coef_new - y_k
                q_new_dev = loss.value(X_proc, y_proc, coef_new)
                _q_new_dev_last = q_new_dev
                bound_dev = q_yk_dev + _dot_dev(grad, diff) + 0.5 * L * _sum_sq_dev(diff)
                slack = float(_to_numpy(bound_dev + 1e-14 - q_new_dev))
                if slack >= 0:
                    break
                L *= 1.5
                step = 1.0 / L

            coef = coef_new

            # Finiteness check
            if not _is_quadratic:
                _coef_norm_dev = _norm2_dev(coef)
                _finite_ok = np.isfinite(float(_coef_norm_dev))
                if not _finite_ok:
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                        y_k = _copy_arr(coef)
                        t_k = 1.0
                        L = L * 2.0
                        continue

            # Divergence detection
            if not _is_quadratic and iteration > 0:
                _need_norm_check = (iteration > 10)
                if _q_new_dev_last is not None:
                    _obj_dev = _q_new_dev_last
                    _q_new_dev_last = None
                else:
                    _obj_dev = loss.value(X_proc, y_proc, coef)
                _obj_val_f = float(_to_numpy(_obj_dev))
                if _smooth_lip > 0:
                    _obj_val_f += _smooth_penalty_value(penalty, coef)
                _diverged_f = False
                if not np.isfinite(_obj_val_f):
                    _diverged_f = True
                elif _obj_best_fista > 1e-8:
                    _diverged_f = _obj_val_f > _obj_best_fista * 10.0 + 1e-8
                else:
                    _diverged_f = _obj_val_f > _obj_best_fista + max(abs(_obj_best_fista) * 10.0, 1.0)
                if not _diverged_f and _need_norm_check:
                    if float(_to_numpy(_norm2_dev(coef))) > 100.0:
                        _diverged_f = True
                if _diverged_f:
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                    else:
                        coef = _zeros(n_features, backend, ref_tensor=X)
                    y_k = _copy_arr(coef)
                    t_k = 1.0
                    L = L * 2.0
                    continue
                elif _obj_val_f < _obj_best_fista:
                    _obj_best_fista = _obj_val_f
                    _coef_best_fista = _copy_arr(coef)
                _prev_obj_fista = _obj_val_f

            # Periodic Lipschitz recomputation
            if not _is_quadratic and iteration > 0 and iteration % 5 == 0:
                L_new = loss.lipschitz(X_proc, coef, y=y_proc)
                if L_new > 0:
                    if _loss_name == "tweedie":
                        L_new *= 5.0
                    elif _loss_name == "gamma":
                        L_new *= 3.0
                    elif _loss_name == "inverse_gaussian":
                        L_new *= 3.0
                    if _smooth_lip > 0:
                        L_new = L_new + _smooth_lip
                    if L_new > L:
                        L = L_new
                    else:
                        L = max(L * 0.8, L_new)

        # Momentum update — all backends
        if _no_momentum:
            t_k = 1.0
            y_k = _copy_arr(coef)
        elif _conservative_momentum:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = min((t_k - 1.0) / t_new, 0.5)
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new
        else:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = (t_k - 1.0) / t_new
            if backend == "numpy":
                y_k = coef + beta * (coef - coef_old)
            elif backend == "cupy":
                y_k = coef + beta * (coef - coef_old)
            else:
                _fista_step = _get_fista_step_compiled() if backend == "torch" else None
                if _fista_step is not None:
                    _, y_k = _fista_step_call(_fista_step, coef, coef, 0.0, coef_old, coef, beta)
                else:
                    y_k = coef + beta * (coef - coef_old)
            t_k = t_new

        # Convergence check — deferred for GPU, every iteration for CPU
        if _is_gpu:
            if iteration < 20 or iteration % _conv_interval == 0:
                _conv_dev = _abs_sum_dev(coef - coef_old)
                if _device_leq(_conv_dev, tol):
                    break
        else:
            _conv_dev = _abs_sum_dev(coef - coef_old)
            if float(_conv_dev) < tol:
                break

    # Return best iterate if available
    if _coef_best_fista is not None:
        coef = _coef_best_fista

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"fista_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}). "
            f"Consider increasing max_iter or using a different solver (newton, lbfgs, irls).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return coef, n_iter


def fista_lla_path(
    loss,
    scad_penalty,
    X,
    y,
    alpha_path,
    max_lla_per_step=6,
    lla_tol=1e-6,
    max_iter=1000,
    tol=1e-4,
    fit_intercept=True,
    sample_weight=None,
    lla_penalty_factory=None,
):
    """Fused LLA+FISTA solver for SCAD/MCP over a continuation path.

    Runs the entire continuation → LLA → FISTA loop in one tight function,
    eliminating per-call overhead (backend detect, preprocess, Lipschitz
    recompute, array allocation) that accumulates over 300+ fista_solver calls.

    Parameters
    ----------
    loss : GLMLoss
    scad_penalty : SCADPenalty or MCPPenalty
        Penalty object; its .alpha will be set along the path.
    X, y : array (pre-centered if fit_intercept=True)
    alpha_path : array of alpha values (descending, geomspace)
    max_lla_per_step : int
    lla_tol : float
    max_iter : int or list[int]
        FISTA iteration limit. If a list, one value per continuation step.
    tol : float
    fit_intercept : bool
    sample_weight : array or None

    Returns
    -------
    coef : array (p,)
    intercept : float
    total_iter : int
    """
    from statgpu.penalties._adaptive_l1 import AdaptiveL1Penalty

    backend = _infer_backend(X)
    if backend == "torch":
        import torch as xp
    elif backend == "cupy":
        import cupy as xp
    else:
        xp = np
    X_proc, y_proc = loss.preprocess(X, y)
    _loss_name = getattr(loss, 'name', '')
    _is_quadratic = (_loss_name == "squared_error")
    _no_momentum = _loss_name in ("poisson",)
    _non_smooth_pen_lla = getattr(scad_penalty, 'name', '') in (
        "l1", "elasticnet", "en", "scad", "mcp", "adaptive_l1", "adaptive_lasso",
        "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
    )
    _conservative_momentum_lla = (
        _loss_name in ("negative_binomial", "tweedie", "inverse_gaussian")
        or (_loss_name in ("logistic", "gamma") and _non_smooth_pen_lla)
    )

    n_samples, n_features = X_proc.shape

    # --- Intercept handling ---
    # For squared_error (identity link): centering X, y is exact.
    # For GLM losses (log/logit link): centering is WRONG — it changes
    # the objective.  Instead, augment X with a ones column so the
    # intercept is part of the coefficient vector.
    _augment_intercept = fit_intercept and not _is_quadratic
    if _augment_intercept:
        # Augment X with a column of ones
        if backend == "torch":
            import torch
            ones_col = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
            X_c = torch.cat([X, ones_col], dim=1)
        elif backend == "cupy":
            import cupy as cp
            ones_col = cp.ones((X.shape[0], 1), dtype=X.dtype)
            X_c = cp.concatenate([X, ones_col], axis=1)
        else:
            ones_col = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_c = np.concatenate([X, ones_col], axis=1)
        y_c = y
        n_aug = n_features + 1
    elif fit_intercept:
        # squared_error: centering is exact for identity link
        if backend == "torch":
            X_mean = X.mean(dim=0)
            y_mean = y.mean()
        elif backend == "cupy":
            import cupy as cp
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
        else:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
        X_c = X - X_mean
        y_c = y - y_mean
        n_aug = n_features
    else:
        X_c = X
        y_c = y
        n_aug = n_features

    # Precompute Lipschitz using loss-specific method.
    # Pass zero coef (global bound) — not all losses handle coef=None.
    if backend == "torch":
        import torch
        _zero_coef_lla = torch.zeros(n_aug, device=X_c.device, dtype=X_c.dtype)
    elif backend == "cupy":
        import cupy as cp
        _zero_coef_lla = cp.zeros(n_aug, dtype=X_c.dtype)
    else:
        _zero_coef_lla = np.zeros(n_aug)
    L_base = loss.lipschitz(X_c, _zero_coef_lla, y=y_c)
    # Precompute XtX for squared_error fast path
    XtX = X_c.T @ X_c
    if L_base <= 0:
        L_base = 1.0

    # Apply loss-specific Lipschitz safety factor (e.g. NB=2x, gamma=3x)
    _lipschitz_safety = getattr(loss, '_lipschitz_safety', 1.0)
    if _lipschitz_safety > 1.0:
        L_base = L_base * _lipschitz_safety

    # Y-scaling for exp-link families (Poisson, Gamma, etc.).
    # At coef=0, mu≈1, but near the optimum mu≈y.  The Hessian scales
    # with mu, so L_base underestimates by up to max(y).
    # Cap at 10x — periodic Lipschitz recomputation corrects any remaining
    # underestimate during the FISTA inner loop.
    _skip_y_scaling = getattr(loss, '_lipschitz_uses_y', False)
    if not _is_quadratic and not _skip_y_scaling:
        _y_arr = _to_numpy(y_c)
        _y_abs = np.abs(_y_arr)
        _y_mean = float(np.mean(_y_abs))
        _y_max = float(np.max(_y_abs))
        _y_scale = min(10.0, max(1.0, np.sqrt(_y_mean * _y_max)))
        if _y_scale > 1.0:
            L_base = L_base * _y_scale

    # Init coef
    if backend == "torch":
        import torch
        coef = torch.zeros(n_aug, device=X_c.device, dtype=X_c.dtype)
    elif backend == "cupy":
        import cupy as cp
        coef = cp.zeros(n_aug, dtype=X_c.dtype)
    else:
        coef = np.zeros(n_aug)

    total_iter = 0
    inner_pen = AdaptiveL1Penalty(alpha=1.0)

    # For squared_error + GPU: fully inlined fused loop
    # Keeps coef on GPU throughout entire continuation+LLA loop
    if _is_quadratic and backend in ("torch", "cupy"):
        Xty = X_c.T @ y_c
        yty = float(_to_numpy(y_c @ y_c)) if backend == "cupy" else float((y_c * y_c).sum().item())

        # Get fused proximal kernel
        if backend == "torch":
            _fused = _get_sqerr_proximal_torch()
            coef_old = coef.clone()
            y_k = coef.clone()
        else:
            _fused = _get_sqerr_proximal_cupy()
            coef_old = coef.copy()
            y_k = coef.copy()

        step = 1.0 / L_base
        t_k = 1.0

        for _cont_i, cont_alpha in enumerate(alpha_path):
            scad_penalty.alpha = float(cont_alpha)
            _mi = max_iter[_cont_i] if isinstance(max_iter, (list, tuple)) else max_iter
            for _lla_i in range(max_lla_per_step):
                # lla_weights() is now backend-aware — stays on device
                lla_w = scad_penalty.lla_weights(coef)
                thresh = lla_w * step  # stays on device

                # Save coef for LLA convergence check (on device)
                coef_before_lla = _copy_arr(coef)

                # Reset momentum for new LLA step
                t_k = 1.0
                coef_old = _copy_arr(coef)
                y_k = _copy_arr(coef)

                # FISTA inner solve (inlined, fused proximal+momentum)
                _conv_interval = 5  # check convergence every N iters
                for iteration in range(_mi):
                    coef_old = _copy_arr(coef)

                    # Gradient: grad = (XtX @ y_k - Xty) / n
                    grad = (XtX @ y_k - Xty) / n_samples

                    # Clip gradients
                    if iteration % 10 == 0:
                        gn_sq = _sum_sq_dev(grad)
                        gmax = max(float(_to_numpy(_abs_sum_dev(coef_old))) * 10.0 + 1e3, 1e4)
                        if backend == "torch":
                            scale = torch.where(gn_sq > gmax * gmax, gmax / torch.sqrt(gn_sq + 1e-30), torch.ones_like(gn_sq))
                        else:
                            scale = cp.where(gn_sq > gmax * gmax, gmax / cp.sqrt(gn_sq + 1e-30), cp.ones_like(gn_sq))
                        grad = grad * scale

                    # Compute momentum beta BEFORE proximal so fused kernel does both
                    if _no_momentum:
                        beta_mom = 0.0
                    else:
                        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                        beta_mom = (t_k - 1.0) / t_new
                        t_k = t_new

                    # Fused proximal + momentum in one kernel call
                    coef, y_k = _fused(coef, grad, step, thresh, coef_old, beta_mom)

                    # Convergence check (deferred sync)
                    if iteration < 20 or iteration % _conv_interval == 0:
                        if float(_to_numpy(_abs_sum_dev(coef - coef_old))) < tol:
                            break

                total_iter += iteration + 1

                # LLA convergence check (on device, single sync for scalar)
                delta = float(_to_numpy(_abs_sum_dev(coef - coef_before_lla)))
                if delta < lla_tol:
                    break
    else:
        # Pre-compute XtX and Xty for squared_error (avoids redundant matmuls)
        _use_xtx = _is_quadratic and backend == "numpy"
        if _use_xtx:
            Xty = X_c.T @ y_c

        for _cont_i, cont_alpha in enumerate(alpha_path):
            scad_penalty.alpha = float(cont_alpha)
            _mi = max_iter[_cont_i] if isinstance(max_iter, (list, tuple)) else max_iter

            for _lla_i in range(max_lla_per_step):
                # lla_weights() is now backend-aware — stays on device
                if _augment_intercept:
                    lla_w_feat = scad_penalty.lla_weights(coef[:n_features])
                    # Append 0.0 for intercept on device
                    if backend == "torch":
                        import torch
                        lla_w = torch.cat([lla_w_feat, torch.zeros(1, device=coef.device, dtype=coef.dtype)])
                    elif backend == "cupy":
                        import cupy as cp
                        lla_w = cp.concatenate([lla_w_feat, cp.zeros(1, dtype=coef.dtype)])
                    else:
                        lla_w = np.append(lla_w_feat, 0.0)
                else:
                    lla_w = scad_penalty.lla_weights(coef)
                if lla_penalty_factory is not None:
                    # lla_penalty_factory expects numpy; convert only if needed
                    lla_w_np = _to_numpy(lla_w) if type(lla_w).__module__ != "numpy" else lla_w
                    inner_pen = lla_penalty_factory(lla_w_np)
                else:
                    inner_pen._weights = lla_w

                # Save coef for LLA convergence check (on device)
                coef_before_lla = _copy_arr(coef)

                # --- FISTA inner solve (inlined, no function call overhead) ---
                _obj_best_lla_inner = None
                _coef_best_lla_inner = None
                y_k = _copy_arr(coef)
                t_k = 1.0
                # Use L_base which includes y-scaling for exp-link families.
                # loss.lipschitz(X, zeros) returns the local Hessian at mu=1,
                # but y-scaling approximates the Hessian at mu≈y — much tighter.
                # Backtracking will adapt L within the inner loop.
                L = L_base
                step = 1.0 / L
                _L_recompute_interval = 20

                _fista_step = _get_fista_step_compiled() if backend == "torch" else None

                # Pre-compute device-side tolerance for convergence check
                if backend != "numpy":
                    _tol_dev = xp.asarray(tol)

                for iteration in range(_mi):
                    coef_old = _copy_arr(coef)

                    if _use_xtx:
                        # Fast path: pre-computed XtX for squared_error
                        q_yk_dev = float(_sum_sq_dev(y_c - X_c @ y_k)) * 0.5 / n_samples
                        grad = (XtX @ y_k - Xty) / n_samples
                    elif _loss_name in ('logistic', 'poisson', 'gamma', 'negative_binomial', 'tweedie', 'inverse_gaussian'):
                        q_yk_dev, grad = _fused_glm_value_and_gradient(loss, X_c, y_c, y_k)
                    else:
                        q_yk_dev = loss.value(X_c, y_c, y_k)
                        grad = loss.gradient(X_c, y_c, y_k)

                    if sample_weight is not None:
                        if backend == "cupy":
                            import cupy as cp
                            sw = cp.asarray(sample_weight)
                        elif backend == "torch":
                            import torch
                            sw = torch.tensor(sample_weight, device=grad.device, dtype=grad.dtype)
                        else:
                            sw = np.asarray(sample_weight)
                        if grad.ndim > 1:
                            grad = grad * sw[:, None]
                        else:
                            grad = grad * sw

                    # Clip gradients (device-side, every 10 iterations)
                    if backend == "numpy" or iteration % 10 == 0:
                        _gn_dev = _norm2_dev(grad)
                        _gsum = _abs_sum_dev(coef_old) * 10.0 + 1e3
                        if backend == "torch":
                            _gmax_dev = xp.clamp(_gsum, min=1e4)
                        else:
                            _gmax_dev = xp.maximum(_gsum, 1e4)
                        if backend != "numpy":
                            _clip_needed = bool(_to_numpy(_gn_dev > _gmax_dev))
                        else:
                            _clip_needed = float(_to_numpy(_gn_dev)) > float(_to_numpy(_gmax_dev))
                        if _clip_needed:
                            grad = grad * (_gmax_dev / _gn_dev)

                    # Backtracking (device-side Armijo check)
                    # Optimization: compute quadratic bound first; skip expensive
                    # loss.value() when bound is clearly below current best (NB/matmul cost).
                    for _bt in range(20):
                        if _fista_step is not None:
                            w_tilde, _ = _fista_step_call(_fista_step, y_k, grad, step, coef_old, coef, 0.0)
                        else:
                            w_tilde = y_k - step * grad
                        coef_new = inner_pen.proximal(w_tilde, step, backend=backend)

                        diff = coef_new - y_k
                        bound_dev = q_yk_dev + _dot_dev(grad, diff) + 0.5 * L * _sum_sq_dev(diff)

                        # Skip loss.value() when bound is clearly too low
                        if _obj_best_lla_inner is not None:
                            _bound_f = float(_to_numpy(bound_dev))
                            if _bound_f < _obj_best_lla_inner * 0.9:
                                L *= 1.5
                                step = 1.0 / L
                                continue

                        q_new_dev = loss.value(X_c, y_c, coef_new)
                        slack_dev = bound_dev + 1e-14 - q_new_dev
                        if backend != "numpy":
                            _armijo_ok = bool(_to_numpy(slack_dev >= 0))
                        else:
                            _armijo_ok = float(_to_numpy(slack_dev)) >= 0
                        if _armijo_ok:
                            break
                        L *= 1.5
                        step = 1.0 / L

                    coef = coef_new

                    # Batched safety checks: coef norm capping + finiteness + divergence
                    # All comparisons done on device, single D2H transfer for booleans
                    if not _is_quadratic:
                        # Compute coef norm once (shared by cap + finiteness + divergence)
                        if _augment_intercept:
                            _cn_dev = _norm2_dev(coef[:n_features])
                        else:
                            _cn_dev = _norm2_dev(coef)

                        # On-device checks
                        if backend == "torch":
                            _finite_dev = xp.isfinite(_cn_dev)
                            _cap_needed_dev = _cn_dev > 5.0
                            _diverge_norm_dev = _cn_dev > 100.0 if iteration > 10 else xp.tensor(False, device=coef.device)
                            _obj_finite_dev = xp.isfinite(q_new_dev) if iteration > 0 else xp.tensor(True, device=coef.device)
                        elif backend == "cupy":
                            _finite_dev = xp.isfinite(_cn_dev)
                            _cap_needed_dev = _cn_dev > 5.0
                            _diverge_norm_dev = _cn_dev > 100.0 if iteration > 10 else xp.asarray(False)
                            _obj_finite_dev = xp.isfinite(q_new_dev) if iteration > 0 else xp.asarray(True)
                        else:
                            _finite_dev = np.isfinite(float(_to_numpy(_cn_dev)))
                            _cap_needed_dev = float(_to_numpy(_cn_dev)) > 5.0
                            _diverge_norm_dev = float(_to_numpy(_cn_dev)) > 100.0 if iteration > 10 else False
                            _obj_finite_dev = np.isfinite(float(_to_numpy(q_new_dev))) if iteration > 0 else True

                        # Single D2H transfer: pack booleans
                        if backend == "numpy":
                            _finite = _finite_dev
                            _cap_needed = _cap_needed_dev
                            _obj_finite = _obj_finite_dev
                            _diverge_norm = _diverge_norm_dev
                        else:
                            _checks = _to_numpy(xp.stack([
                                _finite_dev,
                                _cap_needed_dev,
                                _obj_finite_dev,
                                _diverge_norm_dev,
                            ]))
                            _finite = bool(_checks[0])
                            _cap_needed = bool(_checks[1])
                            _obj_finite = bool(_checks[2])
                            _diverge_norm = bool(_checks[3])

                        # Finiteness reset
                        if not _finite:
                            if _coef_best_lla_inner is not None:
                                coef = _copy_arr(_coef_best_lla_inner)
                            else:
                                coef = _copy_arr(coef_old)
                            y_k = _copy_arr(coef)
                            t_k = 1.0
                            L = L * 2.0
                            step = 1.0 / L
                            continue

                        # Coef norm capping
                        if _cap_needed:
                            _cn_f = float(_to_numpy(_cn_dev))
                            _scale = 5.0 / _cn_f
                            if _augment_intercept:
                                coef = _copy_arr(coef)
                                coef[:n_features] = coef[:n_features] * _scale
                            else:
                                coef = coef * _scale
                            y_k = _copy_arr(coef)
                            t_k = 1.0

                        # Divergence detection
                        if iteration > 0:
                            _obj_val_f = float(_to_numpy(q_new_dev))
                            _diverged_f = not _obj_finite
                            if not _diverged_f and _obj_best_lla_inner is not None:
                                if _obj_best_lla_inner > 1e-8:
                                    _diverged_f = _obj_val_f > _obj_best_lla_inner * 10.0 + 1e-8
                                else:
                                    _diverged_f = _obj_val_f > _obj_best_lla_inner + max(abs(_obj_best_lla_inner) * 10.0, 1.0)
                            if not _diverged_f and _diverge_norm:
                                _diverged_f = True
                            if _diverged_f:
                                if _coef_best_lla_inner is not None:
                                    coef = _copy_arr(_coef_best_lla_inner)
                                else:
                                    coef = _copy_arr(coef_old)
                                y_k = _copy_arr(coef)
                                t_k = 1.0
                                L = L * 2.0
                                step = 1.0 / L
                                continue
                            if _obj_best_lla_inner is None or _obj_val_f < _obj_best_lla_inner:
                                _obj_best_lla_inner = _obj_val_f
                                _coef_best_lla_inner = _copy_arr(coef)

                    # Momentum
                    if _no_momentum:
                        t_k = 1.0
                        y_k = _copy_arr(coef)
                    elif _conservative_momentum_lla:
                        # Conservative Nesterov for Tweedie/NB: cap beta to avoid explosion
                        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                        beta_raw = (t_k - 1.0) / t_new
                        if _loss_name == "negative_binomial":
                            beta = min(beta_raw, 0.75)  # NB: moderate cap
                        else:
                            beta = min(beta_raw, 0.5)   # tweedie/inv_gauss: keep tight cap
                        y_k = coef + beta * (coef - coef_old)
                        t_k = t_new
                    else:
                        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                        beta = (t_k - 1.0) / t_new
                        if _fista_step is not None:
                            _, y_k = _fista_step_call(_fista_step, coef, coef, 0.0, coef_old, coef, beta)
                        else:
                            y_k = coef + beta * (coef - coef_old)
                        t_k = t_new

                    # Convergence (device-side comparison, only D2H 1 bool)
                    if backend == "numpy" or iteration < 20 or iteration % 5 == 0:
                        _conv_dev = _abs_sum_dev(coef - coef_old)
                        if backend != "numpy":
                            if bool(_to_numpy(_conv_dev < _tol_dev)):
                                break
                        else:
                            if float(_to_numpy(_conv_dev)) < tol:
                                break

                    # Periodic Lipschitz recomputation — corrects stale L
                    # as coef moves away from zero. Piggybacked on existing
                    # sync (convergence check already syncs every 5 iters).
                    if not _is_quadratic and iteration > 0 and iteration % _L_recompute_interval == 0:
                        L_new = loss.lipschitz(X_c, coef, y=y_c)
                        if not _skip_y_scaling:
                            _y_arr_cur = _to_numpy(y_c)
                            _y_abs_cur = np.abs(_y_arr_cur)
                            _y_mean_cur = float(np.mean(_y_abs_cur))
                            _y_max_cur = float(np.max(_y_abs_cur))
                            _y_scale_cur = min(10.0, max(1.0, np.sqrt(_y_mean_cur * _y_max_cur)))
                            if _y_scale_cur > 1.0:
                                L_new = L_new * _y_scale_cur
                        if L_new > L * 1.5 or L_new < L / 1.5:
                            L = max(L_new, L_base * 0.1)
                            step = 1.0 / L

                    total_iter += 1
                # --- end FISTA ---

                # LLA convergence (on device, single sync for scalar)
                delta = float(_to_numpy(_abs_sum_dev(coef - coef_before_lla)))
                if delta < lla_tol:
                    break

    # Extract coef and intercept
    coef_all_np = _to_numpy(coef)
    if _augment_intercept:
        # Intercept was part of the augmented coef vector (last element)
        coef_np = coef_all_np[:n_features]
        intercept = float(coef_all_np[n_features])
    elif fit_intercept:
        # squared_error: recover intercept from centering
        X_mean_np = _to_numpy(X_mean)
        y_mean_np = _to_numpy(y_mean)
        coef_np = coef_all_np
        intercept = float(y_mean_np - X_mean_np @ coef_np)
    else:
        coef_np = coef_all_np
        intercept = 0.0

    return coef_np, intercept, total_iter


# ---------------------------------------------------------------------------
# Fused FISTA for squared_error + AdaptiveL1 (SCAD/MCP via LLA)
# ---------------------------------------------------------------------------
# Pre-computes XtX, Xty to avoid redundant matmul; fuses element-wise ops;
# defers GPU→CPU syncs for convergence.

_SQERR_PROXIMAL_TORCH = None
_SQERR_PROXIMAL_CUPY = None


def _get_sqerr_proximal_torch():
    global _SQERR_PROXIMAL_TORCH
    if _SQERR_PROXIMAL_TORCH is None:
        import torch
        # torch.compile requires CUDA capability >= 7.0 (Triton).
        # Fall back to JIT script for older GPUs (P100 = 6.0).
        _cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        if _cap >= 7:
            try:
                @torch.compile(mode='reduce-overhead', backend='inductor')
                def _fused_update(coef, grad, step, thresh, coef_old, beta):
                    w = coef - step * grad
                    abs_w = w.abs()
                    sign_w = w.sign()
                    coef_new = sign_w * (abs_w - thresh).clamp(min=0.0)
                    y_k = coef_new + beta * (coef_new - coef_old)
                    return coef_new, y_k
                _SQERR_PROXIMAL_TORCH = _fused_update
            except Exception:
                pass
        if _SQERR_PROXIMAL_TORCH is None:
            def _fused_update_eager(coef, grad, step, thresh, coef_old, beta):
                w = coef - step * grad
                abs_w = w.abs()
                sign_w = w.sign()
                coef_new = sign_w * (abs_w - thresh).clamp(min=0.0)
                y_k = coef_new + beta * (coef_new - coef_old)
                return coef_new, y_k
            _SQERR_PROXIMAL_TORCH = _fused_update_eager
    return _SQERR_PROXIMAL_TORCH


def _get_sqerr_proximal_cupy():
    global _SQERR_PROXIMAL_CUPY
    if _SQERR_PROXIMAL_CUPY is None:
        import cupy as cp
        _SQERR_PROXIMAL_CUPY = cp.ElementwiseKernel(
            'T coef, T grad, T step, T thresh, T coef_old, T beta',
            'T coef_new, T y_k',
            '''
            T w = coef - step * grad;
            T abs_w = abs(w);
            T sign_w = (w > 0) ? 1 : ((w < 0) ? -1 : 0);
            coef_new = (abs_w > thresh) ? sign_w * (abs_w - thresh) : 0;
            y_k = coef_new + beta * (coef_new - coef_old);
            ''',
            'sqerr_proximal_fused',
        )
    return _SQERR_PROXIMAL_CUPY


def fista_sqerr_adaptive_l1_fused(
    X, y, penalty_weights, alpha,
    XtX, Xty, yty, n_samples,
    L_init, max_iter, tol,
    backend, no_momentum=False,
):
    """Fused FISTA for squared_error + AdaptiveL1 with pre-computed XtX/Xty.

    Eliminates:
    - Redundant X@coef matmul (uses XtX instead)
    - GPU→CPU syncs (convergence check deferred)
    - Element-wise kernel overhead (fused update+proximal+momentum)

    Parameters
    ----------
    X, y : array (centered)
    penalty_weights : array (p,) — LLA weights
    alpha : float — penalty alpha
    XtX, Xty, yty : pre-computed
    n_samples : int
    L_init : float — initial Lipschitz
    max_iter, tol : FISTA params
    backend : 'torch' or 'cupy'
    no_momentum : bool

    Returns
    -------
    coef : array (p,)
    n_iter : int
    """
    p = XtX.shape[0]
    step = 1.0 / L_init
    L = L_init

    if backend == "torch":
        import torch
        thresh = torch.tensor(
            alpha * penalty_weights * step,
            device=XtX.device, dtype=XtX.dtype,
        )
        coef = torch.zeros(p, device=XtX.device, dtype=XtX.dtype)
        coef_old = coef.clone()
        y_k = coef.clone()
        _fused = _get_sqerr_proximal_torch()
        # Pre-allocate for momentum-free case
        _zero_beta = 0.0
    else:
        import cupy as cp
        thresh = cp.asarray(alpha * penalty_weights * step, dtype=cp.float64)
        coef = cp.zeros(p, dtype=cp.float64)
        coef_old = coef.copy()
        y_k = coef.copy()
        _fused = _get_sqerr_proximal_cupy()
        _zero_beta = 0.0

    t_k = 1.0
    _sync_interval = 10  # Only check convergence every N iterations

    for iteration in range(max_iter):
        # Gradient: grad = (XtX @ y_k - Xty) / n
        grad = (XtX @ y_k - Xty) / n_samples

        # Clip gradients (avoid sync — do it on GPU)
        if iteration % 10 == 0:
            gn_sq = _sum_sq_dev(grad)
            gmax = max(float(_to_numpy(_abs_sum_dev(coef_old))) * 10.0 + 1e3, 1e4)
            # Scale if ||grad||^2 > gmax^2
            if backend == "torch":
                import torch
                scale = torch.where(gn_sq > gmax * gmax, gmax / torch.sqrt(gn_sq + 1e-30), torch.ones_like(gn_sq))
                grad = grad * scale
            else:
                import cupy as cp
                scale = cp.where(gn_sq > gmax * gmax, gmax / cp.sqrt(gn_sq + 1e-30), cp.ones_like(gn_sq))
                grad = grad * scale

        # Proximal gradient step (no backtracking — Lipschitz is exact for squared_error)
        coef_new, y_k_new = _fused(coef, grad, step, thresh, coef_old, 0.0)
        coef = coef_new

        # Momentum update
        if no_momentum:
            t_k = 1.0
            y_k = coef
        else:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta_mom = (t_k - 1.0) / t_new
            y_k = coef + beta_mom * (coef - coef_old)
            t_k = t_new

        # Convergence check (deferred sync)
        if iteration < 20 or iteration % _sync_interval == 0:
            if float(_to_numpy(_abs_sum_dev(coef - coef_old))) < tol:
                break

        coef_old = coef.clone() if backend == "torch" else coef.copy()

    return _to_numpy(coef), iteration + 1


def newton_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=100,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
):
    """Newton-Raphson solver with Armijo backtracking line search.

    Supports numpy / cupy / torch backends via auto-detection of X.

    For losses with constant Hessian (Gamma, Tweedie with power≈2), the
    Hessian doesn't change across iterations, so the Newton direction is
    always valid and line search is skipped — this avoids the O(n*p*20)
    overhead of repeated objective evaluations in backtracking.

    Requires: loss has hessian() and penalty is smooth.
    """
    backend = _infer_backend(X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    if init_coef is not None:
        params = _copy_arr(init_coef) if hasattr(init_coef, 'copy') or hasattr(init_coef, 'clone') else np.array(init_coef).copy()
    else:
        if backend == "numpy":
            params = np.zeros(n_features)
        elif backend == "cupy":
            import cupy as cp
            params = cp.zeros(n_features, dtype=X_proc.dtype if hasattr(X_proc, 'dtype') else cp.float64)
        else:
            import torch
            params = torch.zeros(n_features, device=X_proc.device if hasattr(X_proc, 'device') else 'cpu', dtype=X_proc.dtype if hasattr(X_proc, 'dtype') else torch.float64)

    # Detect constant-Hessian losses (Gamma: H=X'X/n, Tweedie power≈2).
    # For these, the Newton step is always valid — skip line search.
    _loss_name = getattr(loss, 'name', '')
    _const_hessian = (_loss_name == "gamma")
    if not _const_hessian and _loss_name == "tweedie":
        pw = getattr(loss, 'power', 1.5)
        if abs(pw - 2.0) < 0.01:
            _const_hessian = True

    # Precompute constant Hessian if applicable (saves O(p^2) per iter)
    _fixed_hess = None
    if _const_hessian:
        _fixed_hess = loss.hessian(X_proc, y_proc, params) + _smooth_penalty_hessian(penalty, params)

    _newton_step = _get_newton_step_compiled() if backend == "torch" else None
    _use_fused = _loss_name in ('logistic', 'poisson', 'gamma',
                                'negative_binomial', 'tweedie', 'inverse_gaussian')

    for iteration in range(max_iter):
        params_old = _copy_arr(params)
        grad = _objective_gradient(loss, penalty, X_proc, y_proc, params)
        hess = _fixed_hess if _fixed_hess is not None else (
            loss.hessian(X_proc, y_proc, params) + _smooth_penalty_hessian(penalty, params)
        )

        try:
            if backend == "numpy":
                direction = np.linalg.solve(hess, grad)
            elif backend == "cupy":
                import cupy as cp
                direction = cp.linalg.solve(hess, grad)
            else:
                import torch
                direction = torch.linalg.solve(hess, grad.unsqueeze(1))
                direction = direction.squeeze(1)
        except Exception:
            if backend == "numpy":
                direction = np.linalg.lstsq(hess, grad, rcond=None)[0]
            elif backend == "cupy":
                import cupy as cp
                direction = cp.linalg.lstsq(hess, grad)[0]
            else:
                import torch
                direction = torch.linalg.lstsq(hess, grad.unsqueeze(1)).solution
                direction = direction.squeeze(1)

        # Armijo backtracking line search — device-side.
        if _use_fused:
            obj_old_dev, _ = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params_old)
            obj_old_dev = obj_old_dev + _smooth_penalty_value_dev(penalty, params_old)
        else:
            obj_old_dev = _objective_value_dev(loss, penalty, X_proc, y_proc, params_old)
        gdd_dev = _dot_dev(grad, direction)
        obj_old, gdd = _sync_scalars(obj_old_dev, gdd_dev, backend=backend)

        step = 1.0
        for _bt in range(20):
            params_try = params_old - step * direction
            try:
                if _use_fused:
                    obj_try_dev, _ = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params_try)
                    obj_try_dev = obj_try_dev + _smooth_penalty_value_dev(penalty, params_try)
                else:
                    obj_try_dev = _objective_value_dev(loss, penalty, X_proc, y_proc, params_try)
                if _device_leq(obj_try_dev, obj_old_dev + 1e-4 * step * gdd):
                    params = params_try
                    break
            except (ValueError, RuntimeError, FloatingPointError):
                pass
            step *= 0.5
        else:
            params = params_old - step * direction

        norm_diff_dev = _norm2_dev(params - params_old)
        nd, = _sync_scalars(norm_diff_dev, backend=backend)
        if nd < tol:
            break

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"newton_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return params, n_iter


def fista_bb_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=1000,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
    use_restart=True,
    step_max_factor=1e3,
    step_min_factor=1e-3,
    bb_burn_in=20,
):
    """FISTA with Barzilai-Borwein step sizes and adaptive restart.

    Uses alternating BB1/BB2 steps (Barzilai & Borwein 1988) that adapt to
    local curvature, eliminating the backtracking line search while preserving
    sparsity.  BB1 = <dw,dw>/<dw,dg> (long step), BB2 = <dw,dg>/<dg,dg>
    (short step).  Adaptive restart (O'Donoghue & Candes 2015) resets
    momentum when it opposes the descent direction.

    Supports numpy / cupy / torch backends via auto-detection of X.
    """
    backend = _infer_backend(X)
    _is_gpu = backend in ("torch", "cupy")
    if backend == "torch":
        import torch
    elif backend == "cupy":
        import cupy as cp
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    # --- Initialize coefficients ---
    if init_coef is not None:
        coef = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        if backend == "numpy":
            coef = np.zeros(n_features)
        elif backend == "cupy":
            import cupy as cp
            coef = cp.zeros(n_features)
        else:
            import torch
            coef = torch.zeros(
                n_features, device=X.device, dtype=X.dtype
            )

    y_k = _copy_arr(coef)
    t_k = 1.0

    # Divergence detection: track best objective for recovery
    _obj_best = float('inf')
    _coef_best = None
    _diverge_count = 0

    _bb_use_long = True     # alternate BB1 / BB2
    _loss_name = getattr(loss, 'name', '')
    # For quadratic losses (squared_error) the gradient is linear in coef,
    # so dg = H @ dw and BB1 = BB2 = 1 / Rayleigh_quotient(H, dw).  The BB
    # step gives zero adaptation and the algorithm degenerates to ISTA
    # (O(1/k) convergence), too slow to reach the true sparse solution
    # within max_iter.  Use standard FISTA (fixed Lipschitz step + Nesterov
    # momentum, O(1/k^2)) instead.
    _is_quadratic = (_loss_name == "squared_error")

    # BB steps estimate local curvature from smooth-gradient differences.
    # For non-smooth penalties the proximal operator introduces a
    # discontinuity that makes the gradient differences noisy.
    #
    # On quadratic losses (squared_error) BB adds nothing — BB1 = BB2 =
    # 1/R_H and the method degenerates to ISTA (O(1/k)).  _is_quadratic
    # already disables BB above.
    #
    # For GLM losses with convex non-smooth penalties (L1, elasticnet,
    # adaptive_l1) the subgradient is bounded and BB differences are valid
    # after a burn-in that lets the iterates stabilise.  This gives 2-3×
    # faster convergence for logistic+L1, poisson+L1, etc.
    #
    # For non-convex non-smooth penalties (SCAD, MCP, group_*) the
    # subgradient can change abruptly (reweighting, folding points),
    # amplifying noise through the non-linear link and causing catastrophic
    # divergence.  Disable BB entirely for these.
    _pen_name = getattr(penalty, "name", "")
    _bb_disabled = {
        "scad", "mcp",
        "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
    }
    if _pen_name in _bb_disabled:
        bb_burn_in = max_iter + 1  # never switch to BB
    elif _pen_name in {"l1", "elasticnet", "en", "adaptive_l1", "adaptive_lasso"}:
        bb_burn_in = max(bb_burn_in, 50)  # longer burn-in for non-smooth

    # Initial Lipschitz at zero (safe for all losses).  Computing L at
    # init_coef can produce enormous values for exp-link families (mu =
    # exp(X@coef) explodes for warm-start coefs from OLS).
    _zero_coef_bb = _copy_arr(coef) * 0.0
    L = loss.lipschitz(X_proc, _zero_coef_bb, y=y_proc)
    if L <= 0:
        L = 1.0
    # For GLM losses with exp link (Poisson, etc.), mu at coef=0
    # is ~1, but mu near the optimum ≈ y.  The Hessian X'@diag(mu)@X
    # scales linearly with mu, so Lipschitz at init can underestimate the
    # true curvature by orders of magnitude (e.g. max(y)=2865 vs init mu=1).
    # Use geometric-mean heuristic: robust against extreme outliers while
    # still scaling up enough to avoid oversized first steps.
    # Logistic: BB step handles adaptation, y-scaling causes divergence.
    # Gamma's expected Fisher Hessian X'X/n underestimates
    # true curvature by ~mean(y), so y-scaling IS needed.
    _loss_global_lip_bb = _loss_name in ("logistic",)
    _skip_y_scaling_bb = getattr(loss, '_lipschitz_uses_y', False)
    if not _is_quadratic and not _loss_global_lip_bb and not _skip_y_scaling_bb:
        _y_arr = _to_numpy(y_proc)
        _y_abs = np.abs(_y_arr)
        _y_mean = float(np.mean(_y_abs))
        _y_max = float(np.max(_y_abs))
        _y_scale = max(1.0, _y_mean, np.sqrt(_y_mean * _y_max))
        if _y_scale > 1.0:
            L = L * _y_scale
    # Inverse Gaussian: gradient scales as 1/mu^3, causing extreme
    # sensitivity to step size.  Use a much more conservative Lipschitz
    # to prevent catastrophic divergence.
    _invgauss_like = _loss_name in ("inverse_gaussian",)
    _tweedie_like = _loss_name == "tweedie"
    if _invgauss_like:
        L = L * 3.0
    # Tweedie (p=1.5): loss landscape steepens exponentially — 5x safety.
    if _loss_name == "tweedie":
        L = L * 5.0
    # Gamma: loss = y/mu + log(mu), gradient = (1 - y/mu)/mu. When mu is small
    # (large eta), the Hessian ~y/mu^3 can be very large, causing divergence.
    # 3x safety (up from 2x) to handle L1 proximal jumps better.
    if _loss_name == "gamma":
        L = L * 3.0
    # Add smooth penalty Lipschitz contribution (e.g. l2 gradient alpha*coef
    # has Lipschitz alpha).  Without this the step 1/L is too large.
    _smooth_lip_bb = _smooth_penalty_lipschitz(penalty)
    if _smooth_lip_bb > 0:
        L = L + _smooth_lip_bb
    step_L = 1.0 / L
    step_k = step_L
    step_max = step_L * step_max_factor
    step_min = step_L * step_min_factor

    # Gradient at initial point for first BB difference
    grad_old = loss.gradient(X_proc, y_proc, coef)
    if sample_weight is not None:
        if backend == "cupy":
            sw = cp.asarray(sample_weight, dtype=grad_old.dtype)
        elif backend == "torch":
            sw = torch.tensor(sample_weight, device=grad_old.device, dtype=grad_old.dtype)
        else:
            sw = np.asarray(sample_weight, dtype=np.float64)
        if grad_old.ndim > 1:
            grad_old = grad_old * sw[:, None]
        else:
            grad_old = grad_old * sw

    for iteration in range(max_iter):
        coef_old = _copy_arr(coef)

        # Gradient at extrapolated point
        grad = loss.gradient(X_proc, y_proc, y_k)
        if sample_weight is not None:
            if backend == "cupy":
                sw = cp.asarray(sample_weight, dtype=grad.dtype)
            elif backend == "torch":
                sw = torch.tensor(sample_weight, device=grad.device, dtype=grad.dtype)
            else:
                sw = np.asarray(sample_weight, dtype=np.float64)
            if grad.ndim > 1:
                grad = grad * sw[:, None]
            else:
                grad = grad * sw

        # Clip extreme gradients — every iteration, all backends.
        # Skip for inverse_gaussian: 1/mu^3 gradient scaling produces large but
        # valid gradients; clipping prevents convergence to the true optimum.
        # Use identical sync-based clipping for both CPU and GPU to ensure
        # consistent trajectories (backtracking already syncs for non-quadratic).
        if not _invgauss_like:
            _gn_f, _coef_abs_f = _sync_scalars(
                _norm2_dev(grad), _abs_sum_dev(coef_old), backend=backend)
            _gmax = max(_coef_abs_f * 10.0 + 1e3, 1e4)
            if _gn_f > _gmax:
                grad = grad * (_gmax / _gn_f)

        # --- Divergence detection ---
        # Full objective check every 5 iterations (GPU optimization: reduces
        # expensive loss.value() calls). Coefficient norm check every iteration
        # (cheap) catches catastrophic explosion early.
        # Batch obj + coef-norm into a single sync when both are needed.
        _do_full_div_check = (iteration % 5 == 0 or iteration <= 5)
        # GPU: defer ALL divergence checks to every 5 iterations (no per-iter sync)
        _do_div_check = (not _is_quadratic and iteration > 0 and
                         (not _is_gpu or _do_full_div_check))
        if _do_div_check:
            _diverged = False
            # Cheap norm check (CPU every iter after 10; GPU piggybacks on 5-iter check)
            if iteration > 10 and not _is_gpu:
                _coef_norm_dev = _norm2_dev(coef)
                if float(_coef_norm_dev) > 100.0:
                    _diverged = True
            # Coef norm check (GPU: piggyback on 5-iter sync)
            if not _diverged and iteration > 10 and _is_gpu:
                _coef_norm_dev = _norm2_dev(coef)
                if backend == "torch":
                    import torch
                    if bool((_coef_norm_dev > 100.0).item()):
                        _diverged = True
                else:
                    import cupy as cp
                    if bool((_coef_norm_dev > 100.0).item()):
                        _diverged = True
            # Full objective check every 5 iterations
            if not _diverged:
                _obj_val = float(_to_numpy(loss.value(X_proc, y_proc, coef)))
                try:
                    _pen_val = float(penalty.value(coef))
                except AttributeError:
                    _pen_val = 0.0
                _obj_total = _obj_val + _pen_val
                if not np.isfinite(_obj_total):
                    _diverged = True
                elif _obj_best > 1e-8:
                    _diverge_threshold = _obj_best * 10.0 + 1e-8
                    if _invgauss_like or _tweedie_like:
                        _diverge_threshold = _obj_best * 100.0 + 10.0
                    _diverged = _obj_total > _diverge_threshold
                else:
                    _diverge_threshold = _obj_best + max(abs(_obj_best) * 10.0, 1.0)
                    if _invgauss_like or _tweedie_like:
                        _diverge_threshold = _obj_best + max(abs(_obj_best) * 100.0, 10.0)
                    _diverged = _obj_total > _diverge_threshold
            if _diverged:
                # Diverged: reset to best known iterate (or zeros) and halve step
                _diverge_count += 1
                if _coef_best is not None:
                    coef = _copy_arr(_coef_best)
                else:
                    # No valid iterate yet — reset to zeros
                    coef = _zeros(n_features, backend, ref_tensor=X_proc)
                y_k = _copy_arr(coef)
                t_k = 1.0
                grad_old = loss.gradient(X_proc, y_proc, coef)
                if sample_weight is not None:
                    if grad_old.ndim > 1:
                        grad_old = grad_old * sw[:, None]
                    else:
                        grad_old = grad_old * sw
                # Halve step size bounds
                step_L = step_L * 0.5
                step_k = step_L
                step_max = step_max * 0.5
                step_min = step_min * 0.5
                L = L * 2.0
                # Reset BB state
                dot_dw_dg = 0.0
                dot_dw_dw = 1.0
                continue
            elif _obj_total < _obj_best:
                _obj_best = _obj_total
                _coef_best = _copy_arr(coef)

        # --- Step size selection ---
        if _is_quadratic or iteration < bb_burn_in:
            # Quadratic loss or burn-in phase: use fixed Lipschitz step.
            # During burn-in for GLM losses, BB steps are delayed because
            # early gradient differences (dw, dg) are dominated by the
            # coef trajectory from zero toward the optimum rather than by
            # local curvature; using BB too early amplifies oscillations.
            step_k = step_L
            # Recompute Lipschitz periodically during burn-in since mu
            # (and therefore the Hessian scale) changes rapidly.
            if not _is_quadratic and iteration > 0 and iteration % 5 == 0:
                # Use global Lipschitz (coef=zero) during burn-in to prevent
                # iterate-dependent Lipschitz from shrinking too fast.
                # BB steps handle adaptation after burn-in.
                # Pass zero coef — not all losses handle coef=None.
                L_new = loss.lipschitz(X_proc, _zero_coef_bb, y=y_proc)
                if L_new > 0:
                    # Re-apply safety factor for steep losses
                    if _loss_name in ("inverse_gaussian",):
                        L_new = L_new * 3.0
                    elif _loss_name == "tweedie":
                        L_new = L_new * 5.0
                    elif _loss_name == "gamma":
                        L_new = L_new * 5.0
                    # Allow L to move toward L_new: full increase, gradual decrease
                    if L_new > L:
                        L = L_new
                    else:
                        L = max(L * 0.8, L_new)
                    step_L = 1.0 / L
                    step_k = step_L
                    step_max = step_L * step_max_factor
                    step_min = step_L * step_min_factor
        else:
            # Nonlinear GLM loss, post-burn-in: use BB step when valid,
            # fall back to Lipschitz step otherwise.
            if dot_dw_dg > 1e-14:
                if _bb_use_long:
                    step_k = dot_dw_dw / dot_dw_dg       # BB1: long
                else:
                    dot_dg_dg = float(_to_numpy(_dot_dev(dg, dg)))
                    step_k = dot_dw_dg / max(dot_dg_dg, 1e-14)  # BB2: short
                _bb_use_long = not _bb_use_long
                # Tweedie: cap BB step more aggressively to prevent overshoot
                if _tweedie_like:
                    step_k = min(step_k, step_L * 2.0)
                step_k = min(max(step_k, step_min), step_max)
            # else: keep previous step_k (step_L or last valid BB step)

        # Gradient step + proximal
        w_tilde = y_k - step_k * grad
        coef_new = penalty.proximal(w_tilde, step_k, backend=backend)
        coef = coef_new

        # Safeguarded backtracking for all GLM losses:
        # After proximal, verify the objective didn't explode.  If it did,
        # halve step and recompute.  This catches cases where the BB step
        # or Lipschitz estimate was too optimistic for the new coef region.
        # Particularly important for torch backend where BB steps can differ
        # from CPU due to floating-point precision in dot products.
        if not _is_quadratic:
            _steep_loss = _loss_name in ("tweedie", "negative_binomial")
            for _bt in range(15):
                # Batch obj + coef-norm into a single sync.
                _new_obj, _new_norm = _sync_scalars(
                    loss.value(X_proc, y_proc, coef), _norm2_dev(coef), backend=backend)
                try:
                    _new_pen = float(penalty.value(coef))
                except AttributeError:
                    _new_pen = 0.0
                _new_total = _new_obj + _new_pen
                # Accept if: finite, reasonable norm, and objective not exploded
                if _steep_loss:
                    _obj_acceptable = (np.isfinite(_new_total) and _new_norm < 100.0 and
                                       _new_total < 1e6)
                else:
                    # For logistic/gamma/poisson: accept if finite, reasonable
                    # norm, and objective not significantly worse than best known.
                    _obj_acceptable = (np.isfinite(_new_total) and _new_norm < 100.0 and
                                       _new_total < max(_obj_best * 1.5 + 1.0, 1e3))
                if _obj_acceptable:
                    break
                # Step too large — halve and retry
                step_k = step_k * 0.5
                L = L * 2.0
                w_tilde = y_k - step_k * grad
                coef = penalty.proximal(w_tilde, step_k, backend=backend)

        # Finiteness check: if coef is non-finite after proximal, reset.
        # Do isfinite on GPU, only sync bool.
        if not _is_quadratic:
            _coef_norm_dev2 = _norm2_dev(coef)
            if backend == "torch":
                import torch
                _finite_ok2 = bool(torch.isfinite(_coef_norm_dev2).item())
            elif backend == "cupy":
                import cupy as cp
                _finite_ok2 = bool(cp.isfinite(_coef_norm_dev2).item())
            else:
                _finite_ok2 = np.isfinite(float(_coef_norm_dev2))
            if not _finite_ok2:
                _diverge_count += 1
                if _coef_best is not None:
                    coef = _copy_arr(_coef_best)
                    y_k = _copy_arr(coef)
                    t_k = 1.0
                    grad_old = loss.gradient(X_proc, y_proc, coef)
                    if sample_weight is not None:
                        if grad_old.ndim > 1:
                            grad_old = grad_old * sw[:, None]
                        else:
                            grad_old = grad_old * sw
                    step_L = step_L * 0.5
                    step_k = step_L
                    step_max = step_max * 0.5
                    step_min = step_min * 0.5
                    L = L * 2.0
                    dot_dw_dg = 0.0
                    dot_dw_dw = 1.0
                    continue

        # --- Store BB step info for next iteration (non-quadratic only) ---
        if not _is_quadratic:
            grad_new = loss.gradient(X_proc, y_proc, coef_new)
            if sample_weight is not None:
                if grad_new.ndim > 1:
                    grad_new = grad_new * sw[:, None]
                else:
                    grad_new = grad_new * sw

            dw = coef_new - coef_old
            dg = grad_new - grad_old
            # Batch two dot products into a single GPU→CPU sync.
            dot_dw_dw, dot_dw_dg = _sync_scalars(
                _dot_dev(dw, dw), _dot_dev(dw, dg), backend=backend)
            grad_old = grad_new

        # --- Nesterov momentum with adaptive restart ---
        # Exp-link losses (Poisson) can cause mu = exp(X@w) to explode when
        # the extrapolated point overshoots. Disable momentum entirely for them.
        # Other GLM losses (logistic) have bounded predictions and benefit from
        # O(1/k^2) convergence via Nesterov momentum after a short burn-in.
        _poisson_like = _loss_name in ("poisson",)
        # Inverse Gaussian: 1/mu^2 gradient scaling causes Nesterov momentum
        # (and BB steps) to oscillate wildly (1.34e+04 diffs observed).
        # Disable both BB and momentum — use plain ISTA with fixed Lipschitz step.
        # _invgauss_like and _tweedie_like defined at top of loop body
        _gamma_like = _loss_name in ("gamma",)
        if _invgauss_like:
            bb_burn_in = max_iter + 1   # never switch to BB
        elif _tweedie_like:
            bb_burn_in = max(200, max_iter // 2)  # longer burn-in for Tweedie
        elif _gamma_like:
            bb_burn_in = max(50, max_iter // 8)  # short burn-in for gamma
        if _poisson_like or _invgauss_like:
            _momentum_burn_in = max_iter + 1   # never use momentum
        elif _tweedie_like:
            _momentum_burn_in = max(100, max_iter // 4)  # delayed momentum for Tweedie
        elif _gamma_like:
            _momentum_burn_in = max(30, max_iter // 10)  # delayed momentum for gamma
        else:
            _momentum_burn_in = 0  # momentum from the start (like standard FISTA)

        # For smooth penalties (L2/none) with Poisson, allow conservative
        # momentum after burn-in to improve convergence speed.
        _conservative_bb = False
        if _poisson_like and not _invgauss_like:
            _pen_name_bb = getattr(penalty, 'name', '')
            if _pen_name_bb in ("l2", "none", "", None):
                _momentum_burn_in = min(100, max_iter)
                _conservative_bb = True
        # Tweedie/gamma: use conservative momentum after burn-in
        if _tweedie_like or _gamma_like:
            _conservative_bb = True

        if iteration < _momentum_burn_in:
            t_k = 1.0
            beta = 0.0
            y_k = _copy_arr(coef)   # next gradient at current point, not extrapolated
        elif _conservative_bb:
            # Conservative momentum: fixed small beta to avoid explosion
            beta = 0.2
            y_k = coef + beta * (coef - coef_old)
            t_k = 1.0
        else:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = (t_k - 1.0) / t_new

            if use_restart and iteration > 0:
                # GPU-side comparison, only sync bool.
                _mc_dev = _dot_dev(y_k - coef_new, coef_new - coef_old)
                if backend == "torch":
                    import torch
                    _mc_positive = bool((_mc_dev > 0).item())
                elif backend == "cupy":
                    import cupy as cp
                    _mc_positive = bool((_mc_dev > 0).item())
                else:
                    _mc_positive = float(_mc_dev) > 0
                if _mc_positive:
                    t_k = 1.0
                    t_new = 1.0
                    beta = 0.0

            y_k = coef + beta * (coef - coef_old)
            t_k = t_new

        # --- Convergence check — deferred for GPU, every iteration for CPU. ---
        if _is_gpu:
            if iteration < 20 or iteration % 3 == 0:
                _conv_dev2 = _abs_sum_dev(coef - coef_old)
                if backend == "torch":
                    import torch
                    if bool((_conv_dev2 < tol).item()):
                        break
                elif backend == "cupy":
                    import cupy as cp
                    if bool((_conv_dev2 < tol).item()):
                        break
        else:
            _conv_dev2 = _abs_sum_dev(coef - coef_old)
            if float(_conv_dev2) < tol:
                break

    # Return best iterate if divergence was detected
    if _diverge_count > 0 and _coef_best is not None:
        coef = _coef_best

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"fista_bb_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}). "
            f"Consider increasing max_iter or using a different solver (newton, lbfgs, irls).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return coef, n_iter


def lbfgs_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=100,
    tol=1e-4,
    init_coef=None,
    history_size=10,
):
    """Limited-memory BFGS for smooth GLM objectives.

    The implementation keeps parameters, gradients, and curvature history on
    the input backend.  GPU-optimised path uses:
    - _fused_glm_value_and_gradient to avoid redundant X@coef
    - _dot_dev / _norm2_dev to stay on device
    - _sync_scalars to batch GPU→CPU transfers
    - _objective_value_dev + _device_leq for device-side line search
    """
    backend = _infer_backend(X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    if init_coef is not None:
        params = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        if backend == "numpy":
            params = np.zeros(n_features)
        elif backend == "cupy":
            import cupy as cp
            params = cp.zeros(
                n_features, dtype=X.dtype if hasattr(X, "dtype") else cp.float64
            )
        else:
            import torch
            params = torch.zeros(
                n_features,
                device=X.device if hasattr(X, "device") else "cpu",
                dtype=X.dtype if hasattr(X, "dtype") else torch.float64,
            )

    s_hist = []
    y_hist = []
    rho_hist = []

    _loss_name = getattr(loss, 'name', '')
    _use_fused = _loss_name in ('logistic', 'poisson', 'gamma',
                                'negative_binomial', 'tweedie', 'inverse_gaussian')

    # Initial gradient (fused to avoid redundant X@coef)
    if _use_fused:
        _init_val_dev, grad = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params)
        grad = grad + _smooth_penalty_gradient(penalty, params)
    else:
        grad = _objective_gradient(loss, penalty, X_proc, y_proc, params)

    if backend == "torch":
        import torch
        tol_dev = torch.tensor(tol, dtype=torch.float64, device=params.device)
    else:
        tol_dev = tol

    for iteration in range(max_iter):
        grad_norm_dev = _norm2_dev(grad)

        # Two-loop recursion — all dot products stay on device
        q = _copy_arr(grad)
        alphas = []
        for s_vec, y_vec, rho in reversed(list(zip(s_hist, y_hist, rho_hist))):
            alpha = rho * _dot_dev(s_vec, q)
            alphas.append(alpha)
            q = q - alpha * y_vec

        if y_hist:
            sy = _dot_dev(s_hist[-1], y_hist[-1])
            yy = _dot_dev(y_hist[-1], y_hist[-1])
            gamma = sy / yy if _device_gt(yy, 1e-30) else 1.0
        else:
            gamma = 1.0
        r = gamma * q

        for s_vec, y_vec, rho, alpha in zip(
            s_hist, y_hist, rho_hist, reversed(alphas)
        ):
            beta = rho * _dot_dev(y_vec, r)
            r = r + s_vec * (alpha - beta)

        direction = -r
        gdd_dev = _dot_dev(grad, direction)

        # Batch sync: grad_norm + grad_dot_dir
        gn, gdd = _sync_scalars(grad_norm_dev, gdd_dev, backend=backend)
        if gn < tol:
            break
        if gdd >= 0:
            direction = -grad
            gdd = -gn  # -||grad||^2

        # Line search — stays on device
        if _use_fused:
            old_val_dev, _ = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params)
            old_val_dev = old_val_dev + _smooth_penalty_value_dev(penalty, params)
        else:
            old_val_dev = _objective_value_dev(loss, penalty, X_proc, y_proc, params)

        step = 1.0
        params_new = params
        for _ in range(25):
            candidate = params + step * direction
            if _use_fused:
                cand_val_dev, _ = _fused_glm_value_and_gradient(loss, X_proc, y_proc, candidate)
                cand_val_dev = cand_val_dev + _smooth_penalty_value_dev(penalty, candidate)
            else:
                cand_val_dev = _objective_value_dev(loss, penalty, X_proc, y_proc, candidate)
            # Device-side comparison — single sync for the bool
            if _device_leq(cand_val_dev, old_val_dev + 1e-4 * step * gdd):
                params_new = candidate
                break
            step *= 0.5

        # Update gradient (fused)
        if _use_fused:
            _, grad_new = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params_new)
            grad_new = grad_new + _smooth_penalty_gradient(penalty, params_new)
        else:
            grad_new = _objective_gradient(loss, penalty, X_proc, y_proc, params_new)

        s_vec = params_new - params
        y_vec = grad_new - grad
        ys_dev = _dot_dev(y_vec, s_vec)
        s_norm_dev = _norm2_dev(s_vec)

        # Batch sync: ys + s_norm
        ys, s_norm = _sync_scalars(ys_dev, s_norm_dev, backend=backend)
        if ys > 1e-12:
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            rho_hist.append(1.0 / ys)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

        params = params_new
        grad = grad_new
        if s_norm < tol:
            break

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"lbfgs_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return params, n_iter


# =============================================================================
# ADMM Solver (three-backend: numpy/cupy/torch)
# =============================================================================


def _cg_solve(A_op, b, x0=None, max_iter=30, tol=1e-6):
    """Conjugate gradient for solving (A + rho*I)x = b.

    A_op(x) returns A(x) — used for the augmented Lagrangian subproblem.
    GPU-optimised: all dot products stay on device via _dot_dev.
    """
    backend = _infer_backend(b)
    if x0 is not None:
        x = _copy_arr(x0)
    else:
        x = _zeros_like(b)

    r = b - A_op(x)
    p = _copy_arr(r)
    rsold = _dot_dev(r, r)

    tol_sq = tol * tol
    if backend != "numpy" and _device_leq(rsold, tol_sq):
        return x
    if backend == "numpy" and rsold < tol_sq:
        return x

    for _ in range(max_iter):
        Ap = A_op(p)
        pAp = _dot_dev(p, Ap)
        if backend != "numpy":
            alpha = rsold / pAp if _device_gt(pAp, 1e-30) else 0.0
        else:
            alpha = rsold / pAp if pAp > 1e-30 else 0.0
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = _dot_dev(r, r)
        if backend != "numpy":
            if _device_leq(rsnew, tol_sq):
                break
            beta = rsnew / rsold if _device_gt(rsold, 1e-30) else 0.0
        else:
            if rsnew < tol_sq:
                break
            beta = rsnew / rsold if rsold > 1e-30 else 0.0
        p = r + beta * p
        rsold = rsnew

    return x


def admm_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=200,
    tol=1e-4,
    rho=1.0,
    adaptive_rho=True,
    cg_max_iter=30,
    cg_tol=1e-6,
    init_coef=None,
    sample_weight=None,
):
    """ADMM solver for penalized GLM optimization.

    Reformulates min_w f(Xw; y) + p(w) as:
        min_{w,z} f(Xw; y) + p(z)  s.t. w = z

    and solves via the alternating direction method of multipliers:
        w^{k+1} = argmin_w f(Xw; y) + (rho/2)||w - z^k + u^k||^2
        z^{k+1} = prox_{p/rho}(w^{k+1} + u^k)
        u^{k+1} = u^k + w^{k+1} - z^{k+1}

    The w-update is a smooth, strongly convex problem solved via conjugate
    gradient. The z-update reuses penalty.proximal(). Both are GPU-friendly:
    w-update uses dense matmuls (cuBLAS), z-update is element-wise.

    Supports numpy / cupy / torch backends via auto-detection of X.

    Parameters
    ----------
    loss : GLMLoss
    penalty : Penalty
    X, y : arrays
    max_iter : int
        Maximum ADMM outer iterations.
    tol : float
        Convergence tolerance for primal/dual residuals.
    rho : float
        Augmented Lagrangian penalty parameter.
    adaptive_rho : bool
        Adapt rho based on primal/dual residual balance.
    cg_max_iter : int
        Maximum CG iterations for w-update subproblem.
    cg_tol : float
        CG convergence tolerance.
    init_coef : array, optional
        Initial coefficients.
    sample_weight : array, optional

    Returns
    -------
    coef : array, n_iter : int
    """
    backend = _infer_backend(X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    # Initialize
    if init_coef is not None:
        w = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        if backend == "numpy":
            w = np.zeros(n_features)
        elif backend == "cupy":
            import cupy as cp
            w = cp.zeros(n_features)
        else:
            import torch
            w = torch.zeros(
                n_features, device=X.device, dtype=X.dtype
            )

    z = _copy_arr(w)
    u = _zeros_like(w)

    # Precompute sample_weight scaling if needed
    sw = None
    if sample_weight is not None:
        if backend == "cupy":
            import cupy as cp
            sw = cp.asarray(sample_weight)
        elif backend == "torch":
            import torch
            sw = torch.tensor(sample_weight, device=X.device, dtype=X.dtype)
        else:
            sw = np.asarray(sample_weight)

    def _grad_w(w_vec, z_cur, u_cur):
        """Gradient of f(w) + (rho/2)||w - z_cur + u_cur||^2 w.r.t. w."""
        g = loss.gradient(X_proc, y_proc, w_vec)
        if sw is not None:
            if g.ndim > 1:
                g = g * sw[:, None]
            else:
                g = g * sw
        g = g + rho * (w_vec - z_cur + u_cur)
        return g

    # Detect if loss has a constant Hessian (squared_error) — use Cholesky.
    # For GLM losses, use Nesterov-accelerated gradient descent.
    # When using Cholesky we pin rho (disable adaptive_rho) because the
    # precomputed _A_mat = XtX/n + rho*I would become stale if rho changed.
    loss_name = getattr(loss, 'name', '')
    use_cholesky = loss_name == "squared_error" and n_features <= 2000
    if use_cholesky:
        adaptive_rho = False

    if use_cholesky:
        _hess_const = loss.hessian(X_proc, y_proc, w)          # XtX / n
        _A_mat = _hess_const
        if hasattr(_hess_const, 'shape'):
            if backend == "numpy":
                _A_mat = _hess_const + rho * np.eye(n_features, dtype=_hess_const.dtype)
                _L = np.linalg.cholesky(_A_mat)
            elif backend == "cupy":
                import cupy as cp
                _A_mat = _hess_const + rho * cp.eye(n_features, dtype=_hess_const.dtype)
                _L = cp.linalg.cholesky(_A_mat)
            else:
                import torch
                _A_mat = _hess_const + rho * torch.eye(n_features, dtype=_hess_const.dtype, device=_hess_const.device)
                _L = torch.linalg.cholesky(_A_mat)
        else:
            use_cholesky = False

        # Precompute -grad_f(0) = Xty/n for squared_error (the constant part)
        _zero_coef = _zeros_like(w)
        _neg_grad_zero = -loss.gradient(X_proc, y_proc, _zero_coef)  # Xty/n
        if sw is not None:
            if _neg_grad_zero.ndim > 1:
                _neg_grad_zero = _neg_grad_zero * sw[:, None]
            else:
                _neg_grad_zero = _neg_grad_zero * sw

    else:
        # Gradient descent step: 1/(L_f + rho)
        L_f = loss.lipschitz(X_proc, w, y=y_proc)
        if L_f <= 0:
            L_f = 1.0
        lr_sub = 1.0 / (L_f + rho + 1e-8)

    for iteration in range(max_iter):
        z_old = _copy_arr(z)

        # --- w-update ---
        if use_cholesky:
            # Closed-form: (XtX/n + rho*I) w = Xty/n + rho*(z - u)
            rhs = _neg_grad_zero + rho * (z - u)
            if backend == "numpy":
                w = np.linalg.solve(_A_mat, rhs)
            elif backend == "cupy":
                w = cp.linalg.solve(_A_mat, rhs)
            else:
                w = torch.linalg.solve(_A_mat, rhs.unsqueeze(1)).squeeze(1)
        else:
            # Nesterov-accelerated gradient descent on the w-subproblem
            w_new = _copy_arr(w)
            w_mom = _copy_arr(w)
            t_mom = 1.0
            for _ in range(cg_max_iter):
                w_old_mom = _copy_arr(w_new)
                g_sub = _grad_w(w_mom, z, u)
                w_next = w_mom - lr_sub * g_sub
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
                w_mom = w_next + ((t_mom - 1.0) / t_new) * (w_next - w_new)
                w_new = w_next
                t_mom = t_new
                diff_dev = _abs_sum_dev(w_next - w_old_mom)
                if backend != "numpy":
                    if _device_leq(diff_dev, cg_tol * n_features):
                        break
                elif diff_dev < cg_tol * n_features:
                    break
            w = w_new

        # --- z-update: proximal operator ---
        z = penalty.proximal(w + u, 1.0 / rho, backend=backend)

        # --- u-update: dual ascent ---
        u = u + w - z

        # --- Adaptive rho + Convergence check (batched sync) ---
        rp_dev = _norm2_dev(w - z)
        rd_dev = _norm2_dev(z - z_old)
        rp, rd_raw = _sync_scalars(rp_dev, rd_dev, backend=backend)
        r_dual = rho * rd_raw

        if adaptive_rho:
            if rp > 10.0 * r_dual:
                rho = min(rho * 2.0, 1e4)
            elif r_dual > 10.0 * rp:
                rho = max(rho * 0.5, 1e-4)

        if rp < tol and r_dual < tol:
            break

    # Return z (penalized/feasible variable), not w (unconstrained).
    # At convergence w ≈ z, but z always satisfies the penalty structure.
    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"admm_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return z, n_iter
