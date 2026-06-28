"""Proximal Newton solver for smooth loss + non-smooth penalty.

Solves: min f(x) + g(x)
where f is smooth (loss) and g is non-smooth (penalty).

Algorithm:
1. Compute Newton direction: d = -H^-1 @ (grad_f + prox_grad_g)
2. Line search: find step that decreases f(x + step*d) + g(x + step*d)
3. Update: x = x + step * d

Much faster than FISTA for problems where:
- f has a Hessian (Huber, Bisquare, Fair, CoxPH)
- g is non-smooth but has a proximal operator (L1, SCAD/MCP via LLA)

Typical convergence: 5-10 iterations vs 300+ for FISTA.
"""

__all__ = ["proximal_newton_solver"]

import warnings
import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._array_ops import (
    _copy_arr,
    _device_leq,
    _dot_dev,
    _norm2_dev,
    _sync_scalars,
    _zeros,
)
from statgpu.backends._utils import _to_float_scalar, _to_numpy
from ._utils import _smooth_penalty_gradient, _smooth_penalty_hessian


def proximal_newton_solver(
    loss,
    penalty,
    X,
    y,
    max_iter: int = 50,
    tol: float = 1e-6,
    init_coef=None,
    sample_weight=None,
):
    """Proximal Newton solver for smooth loss + non-smooth penalty.

    Parameters
    ----------
    loss : LossBase
        Must have gradient(), hessian(), fused_value_and_gradient().
    penalty : Penalty
        Non-smooth penalty with proximal() method.
    X, y : array
        Data (preprocessed).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    init_coef : array, optional
        Initial coefficients.
    sample_weight : array, optional

    Returns
    -------
    params : array
        Optimized coefficients.
    n_iter : int
        Number of iterations.
    """
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    if init_coef is not None:
        params = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        params = _zeros(n_features, backend, ref_tensor=X_proc)

    # Check if loss has hessian
    if not getattr(loss, 'has_hessian', False):
        raise ValueError(
            f"Proximal Newton requires loss with hessian, "
            f"got '{getattr(loss, 'name', '?')}' which has has_hessian=False."
        )

    # Pre-allocate ridge matrix (reused every iteration)
    _n = n_features
    if backend == "numpy":
        _ridge = 1e-10 * np.eye(_n, dtype=np.float64)
    elif backend == "cupy":
        import cupy as cp
        _ridge = 1e-10 * cp.eye(_n, dtype=cp.float64)
    else:
        import torch
        _ridge = 1e-10 * torch.eye(_n, dtype=torch.float64,
                                    device=params.device if hasattr(params, 'device') else 'cpu')

    # Check if loss supports fused gradient+hessian
    _has_fused = hasattr(loss, 'fused_gradient_and_hessian')

    for iteration in range(max_iter):
        params_old = _copy_arr(params)

        # Gradient and Hessian of smooth loss
        if _has_fused:
            loss_grad, loss_hess = loss.fused_gradient_and_hessian(
                X_proc, y_proc, params, sample_weight=sample_weight
            )
        else:
            loss_grad = loss.gradient(X_proc, y_proc, params, sample_weight=sample_weight)
            loss_hess = loss.hessian(X_proc, y_proc, params, sample_weight=sample_weight)

        # Add smooth penalty gradient/hessian only for smooth penalties.
        # Non-smooth penalties (L1, AdaptiveL1) are handled by proximal operator.
        _pen_name = getattr(penalty, 'name', '')
        _is_smooth_pen = _pen_name in ('l2', 'none', 'null', '', 'elasticnet')
        if _is_smooth_pen:
            grad = loss_grad + _smooth_penalty_gradient(penalty, params)
            hess = loss_hess + _smooth_penalty_hessian(penalty, params)
        else:
            grad = loss_grad
            hess = loss_hess
        hess = 0.5 * (hess + hess.T)

        # Check convergence via gradient norm
        grad_norm_dev = _norm2_dev(grad)
        (grad_norm,) = _sync_scalars(grad_norm_dev, backend=backend)
        if grad_norm <= tol:
            break

        # Newton direction with ridge for stability (use pre-allocated _ridge)
        hess_reg = hess + _ridge

        try:
            if backend == "numpy":
                direction = np.linalg.solve(hess_reg, grad)
            elif backend == "cupy":
                import cupy as cp
                direction = cp.linalg.solve(hess_reg, grad)
            else:
                import torch
                direction = torch.linalg.solve(hess_reg, grad.unsqueeze(1)).squeeze(1)
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to gradient descent if Hessian is singular/ill-conditioned
            direction = grad
        except RuntimeError as e:
            # Only catch singular/ill-conditioned errors, re-raise others (OOM, device mismatch, etc.)
            err_msg = str(e).lower()
            if "singular" in err_msg or "ill-conditioned" in err_msg or "not invertible" in err_msg:
                direction = grad
            else:
                raise

        # Armijo backtracking line search with proximal step
        obj_old_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params_old, sample_weight=sample_weight)
        _has_pen_value = hasattr(penalty, 'value')
        if _has_pen_value:
            pen_old = float(_to_numpy(penalty.value(params_old[:n_features])))
        else:
            pen_old = 0.0
            if iteration == 0:
                warnings.warn(
                    f"proximal_newton: penalty '{getattr(penalty, 'name', '?')}' "
                    f"has no value() method. Armijo condition ignores penalty value.",
                    RuntimeWarning, stacklevel=2,
                )
        gdd_dev = _dot_dev(grad, direction)
        gdd = _to_float_scalar(gdd_dev)

        if not np.isfinite(gdd) or gdd <= 0.0:
            # Fall back to steepest descent
            direction = grad
            gdd = grad_norm * grad_norm

        step = 1.0
        accepted = False
        for _bt in range(25):
            # Trial point: params_old - step * direction
            params_try = params_old - step * direction

            # Apply proximal operator (handles non-smooth penalty)
            if hasattr(penalty, 'proximal'):
                # For weighted L1 (AdaptiveL1 from LLA): proximal is soft-threshold
                params_try = penalty.proximal(params_try, step, backend=backend)

            try:
                obj_try_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params_try, sample_weight=sample_weight)
                pen_try = float(_to_numpy(penalty.value(params_try[:n_features]))) if _has_pen_value else 0.0

                # Composite Armijo: f(x_new) + g(x_new) <= f(x_old) + g(x_old) + c*step*gdd
                if _device_leq(obj_try_dev + pen_try, obj_old_dev + pen_old - 1e-4 * step * gdd):
                    params = params_try
                    accepted = True
                    break
            except (ValueError, RuntimeError, FloatingPointError):
                pass
            step *= 0.5

        if not accepted:
            params = params_old
            warnings.warn(
                f"proximal_newton line search failed (iter={iteration}).",
                RuntimeWarning, stacklevel=2,
            )
            break

    n_iter = iteration + 1
    return params, n_iter
