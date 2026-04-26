"""
Unified solvers for GLMLoss + Penalty optimization.

minimize: loss(X, y, w) + penalty(w)

Supports numpy / cupy / torch backends via auto-detection.
"""

import numpy as np

from statgpu.backends import _to_numpy

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
    """Dot product."""
    if isinstance(a, np.ndarray):
        return float(a.dot(b))
    mod = type(a).__module__
    if mod.startswith("cupy"):
        import cupy as cp
        return float(a.dot(b))
    if mod.startswith("torch"):
        import torch
        return float(a.dot(b))
    return float(a.dot(b))


def _sum_sq(x):
    """Sum of squares."""
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
    raise ValueError(
        f"solver requires a smooth penalty, got penalty='{_penalty_name(penalty)}'."
    )


def _smooth_penalty_hessian(penalty, coef):
    """Return smooth penalty Hessian on the same backend as coef."""
    n = coef.shape[0]
    if penalty is None or _penalty_name(penalty) in ("none", "null"):
        return 0.0 * _eye_like(n, coef)
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


def fista_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=1000,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
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

    # Start with an initial Lipschitz estimate
    L = loss.lipschitz(X_proc, coef)
    if L <= 0:
        L = 1.0

    for iteration in range(max_iter):
        coef_old = _copy_arr(coef)

        # Backtracking line search for L
        step = 1.0 / L
        _fista_step = _get_fista_step_compiled() if backend == "torch" else None
        for _bt in range(20):
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

            if _fista_step is not None:
                w_tilde, _ = _fista_step_call(_fista_step, y_k, grad, step, coef_old, coef, 0.0)
            else:
                w_tilde = y_k - step * grad
            coef_new = penalty.proximal(w_tilde, step, backend=backend)

            # Check sufficient decrease condition (convert to numpy for scalar ops)
            q_new = _to_numpy(loss.value(X_proc, y_proc, coef_new))
            q_yk = _to_numpy(loss.value(X_proc, y_proc, y_k))
            grad_np = _to_numpy(grad)
            coef_new_np = _to_numpy(coef_new)
            y_k_np = _to_numpy(y_k)
            linear_approx = q_yk + _dot(grad_np, coef_new_np - y_k_np)
            quadratic_bound = linear_approx + 0.5 * L * _sum_sq(coef_new_np - y_k_np)

            if q_new <= quadratic_bound + 1e-14:
                break

            # Increase L (decrease step)
            L *= 1.5
            step = 1.0 / L

        coef = coef_new

        # Momentum update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        beta = (t_k - 1.0) / t_new

        if backend == "numpy":
            y_k = coef + beta * (coef - coef_old)
        elif backend == "cupy":
            y_k = coef + beta * (coef - coef_old)
        else:
            if _fista_step is not None:
                _, y_k = _fista_step_call(_fista_step, coef, coef, 0.0, coef_old, coef, beta)
            else:
                y_k = coef + beta * (coef - coef_old)
        t_k = t_new

        # Convergence check (compare in numpy)
        diff = _to_numpy(coef) - _to_numpy(coef_old)
        if np.sum(np.abs(diff)) < tol:
            break

    return coef, iteration + 1


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
    """Newton-Raphson solver.

    Supports numpy / cupy / torch backends via auto-detection of X.

    Requires: loss has hessian() and penalty is smooth.
    """
    backend = _infer_backend(X)
    n_features = X.shape[1]

    if init_coef is not None:
        params = _copy_arr(init_coef) if hasattr(init_coef, 'copy') or hasattr(init_coef, 'clone') else np.array(init_coef).copy()
    else:
        if backend == "numpy":
            params = np.zeros(n_features)
        elif backend == "cupy":
            import cupy as cp
            params = cp.zeros(n_features, dtype=X.dtype if hasattr(X, 'dtype') else cp.float64)
        else:
            import torch
            params = torch.zeros(n_features, device=X.device if hasattr(X, 'device') else 'cpu', dtype=X.dtype if hasattr(X, 'dtype') else torch.float64)

    _newton_step = _get_newton_step_compiled() if backend == "torch" else None
    for iteration in range(max_iter):
        params_old = _copy_arr(params)
        grad = _objective_gradient(loss, penalty, X, y, params)
        hess = loss.hessian(X, y, params) + _smooth_penalty_hessian(
            penalty, params
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

        if _newton_step is not None:
            params, norm_diff = _newton_step_call(_newton_step, params, direction, params_old)
            norm_diff = float(norm_diff.item())
        else:
            params = params - direction
            if backend == "numpy":
                norm_diff = np.linalg.norm(params - params_old)
            elif backend == "cupy":
                import cupy as cp
                norm_diff = float(cp.linalg.norm(params - params_old))
            else:
                import torch
                norm_diff = float(torch.linalg.norm(params - params_old).item())

        if norm_diff < tol:
            break

    return params, iteration + 1


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
    the input backend. Only scalar convergence and line-search values are
    copied to Python floats.
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
    grad = _objective_gradient(loss, penalty, X_proc, y_proc, params)

    for iteration in range(max_iter):
        grad_norm = _norm2(grad)
        if grad_norm < tol:
            break

        q = _copy_arr(grad)
        alphas = []
        for s_vec, y_vec, rho in reversed(list(zip(s_hist, y_hist, rho_hist))):
            alpha = rho * _dot(s_vec, q)
            alphas.append(alpha)
            q = q - alpha * y_vec

        if y_hist:
            sy = _dot(s_hist[-1], y_hist[-1])
            yy = _dot(y_hist[-1], y_hist[-1])
            gamma = sy / yy if yy > 1e-30 else 1.0
        else:
            gamma = 1.0
        r = gamma * q

        for s_vec, y_vec, rho, alpha in zip(
            s_hist, y_hist, rho_hist, reversed(alphas)
        ):
            beta = rho * _dot(y_vec, r)
            r = r + s_vec * (alpha - beta)

        direction = -r
        grad_dot_dir = _dot(grad, direction)
        if grad_dot_dir >= 0:
            direction = -grad
            grad_dot_dir = -_dot(grad, grad)

        old_value = _objective_value(loss, penalty, X_proc, y_proc, params)
        step = 1.0
        params_new = params
        for _ in range(25):
            candidate = params + step * direction
            candidate_value = _objective_value(
                loss, penalty, X_proc, y_proc, candidate
            )
            if candidate_value <= old_value + 1e-4 * step * grad_dot_dir:
                params_new = candidate
                break
            step *= 0.5

        grad_new = _objective_gradient(loss, penalty, X_proc, y_proc, params_new)
        s_vec = params_new - params
        y_vec = grad_new - grad
        ys = _dot(y_vec, s_vec)
        if ys > 1e-12:
            s_hist.append(_copy_arr(s_vec))
            y_hist.append(_copy_arr(y_vec))
            rho_hist.append(1.0 / ys)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

        params = params_new
        grad = grad_new
        if _norm2(s_vec) < tol:
            break

    return params, iteration + 1
