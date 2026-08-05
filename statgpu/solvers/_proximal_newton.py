"""Proximal Newton solver for smooth loss + non-smooth penalty.

Solves smooth loss plus a smooth penalty with Newton updates.

A general non-smooth proximal-Newton step requires solving the Hessian-metric
proximal subproblem. The historical Euclidean-prox approximation optimized a
different objective (and double-counted L2/ElasticNet curvature). Until a
metric proximal subproblem solver is implemented, non-smooth penalties are
explicitly delegated to the backend-native FISTA solver with a warning.
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
from ._utils import (
    _runtime_error_is_singular,
    _smooth_penalty_gradient,
    _smooth_penalty_hessian,
    _validate_sample_weight,
    _as_backend_vector,
)


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
    """Newton solver for smooth penalties with explicit FISTA delegation.

    L2/no-penalty objectives use Newton updates. A non-smooth penalty emits a
    ``RuntimeWarning`` and is delegated to ``fista_solver`` because the
    Hessian-metric proximal subproblem is not implemented.

    Parameters
    ----------
    loss : LossBase
        Must expose the operations required by the selected solver path.
    penalty : Penalty or None
        L2/None for Newton; non-smooth penalties are delegated to FISTA.
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
    _pen_name = str(getattr(penalty, "name", "none")).lower()
    _is_smooth_pen = _pen_name in ("l2", "none", "null", "")
    if not _is_smooth_pen:
        warnings.warn(
            "proximal_newton_solver delegates non-smooth penalties to "
            "fista_solver because the Hessian-metric proximal subproblem is "
            "not implemented; this preserves the declared objective.",
            RuntimeWarning,
            stacklevel=2,
        )
        from ._fista import fista_solver

        return fista_solver(
            loss,
            penalty,
            X,
            y,
            max_iter=max_iter,
            tol=tol,
            init_coef=init_coef,
            sample_weight=sample_weight,
        )

    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    _validate_sample_weight(sample_weight, X_proc.shape[0])
    n_features = X_proc.shape[1]

    if init_coef is not None:
        params = _as_backend_vector(init_coef, backend, X_proc)
    else:
        params = _zeros(n_features, backend, ref_tensor=X_proc)

    # Check if loss has hessian
    if not getattr(loss, 'has_hessian', False):
        raise ValueError(
            f"Proximal Newton requires loss with hessian, "
            f"got '{getattr(loss, 'name', '?')}' which has has_hessian=False."
        )

    # Pre-allocate a dtype/device-compatible ridge matrix.
    _n = n_features
    _dtype = getattr(params, "dtype", np.float64)
    if backend == "numpy":
        _ridge = 1e-10 * np.eye(_n, dtype=_dtype)
    elif backend == "cupy":
        import cupy as cp
        _ridge = 1e-10 * cp.eye(_n, dtype=_dtype)
    else:
        import torch
        _ridge = 1e-10 * torch.eye(
            _n,
            dtype=_dtype,
            device=params.device if hasattr(params, "device") else "cpu",
        )

    # Check if loss supports fused gradient+hessian
    _has_fused = hasattr(loss, 'fused_gradient_and_hessian')
    iteration = -1  # max_iter=0 returns the initialized coefficient vector

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

        # Only smooth penalties reach this path. Their gradient and
        # curvature are included exactly once in the Newton system.
        grad = loss_grad + _smooth_penalty_gradient(penalty, params)
        hess = loss_hess + _smooth_penalty_hessian(penalty, params)
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
        except np.linalg.LinAlgError:
            # Fall back only for a true singular/ill-conditioned Hessian.
            direction = grad
        except RuntimeError as exc:
            if not _runtime_error_is_singular(exc):
                raise
            direction = grad

        # Armijo backtracking line search with proximal step
        obj_old_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params_old, sample_weight=sample_weight)
        _has_pen_value = hasattr(penalty, 'value')
        if _has_pen_value:
            pen_old = float(_to_numpy(penalty.value(params_old[:n_features])))
        else:
            pen_old = 0.0
            if iteration == 0 and _pen_name not in ("none", "null", ""):
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

            # Smooth penalty terms are already represented in the Newton
            # direction; applying their proximal operator here would count the
            # same penalty a second time.
            try:
                obj_try_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params_try, sample_weight=sample_weight)
                pen_try = float(_to_numpy(penalty.value(params_try[:n_features]))) if _has_pen_value else 0.0

                # Composite Armijo: f(x_new) + g(x_new) <= f(x_old) + g(x_old) + c*step*gdd
                if _device_leq(obj_try_dev + pen_try, obj_old_dev + pen_old - 1e-4 * step * gdd):
                    params = params_try
                    accepted = True
                    break
            except FloatingPointError:
                pass
            except RuntimeError as exc:
                # Only swallow trial-point numerical failures; infrastructure
                # and device errors remain visible to the caller.
                err_msg = str(exc).lower()
                if not any(
                    marker in err_msg
                    for marker in ("overflow", "invalid value", "nan")
                ):
                    raise
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
