"""Quantile Group SCAD/MCP via LLA, IRLS majorization, and convex ADMM.

The automatic Quantile Group SCAD/MCP route is a non-convex LLA algorithm.
At a fixed LLA derivative it has a convex Adaptive-Group-Lasso surrogate.  The
pinball data-fit is majorized by the same IRLS quadratic used by the maintained
Quantile solvers; each weighted least-squares surrogate is then solved by the
existing squared-error ADMM engine.  For squared error that ADMM path uses a
direct Cholesky w-update and the penalty's backend-native group proximal map.

This solver is intentionally private to the automatic estimator/CV route.  It
does not change the public low-level ``fista_lla_path`` contract, and explicit
``solver='fista'`` requests remain explicit FISTA.
"""

from __future__ import annotations

import copy
import inspect
import warnings

import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._array_ops import _abs_sum_dev, _copy_arr, _zeros
from statgpu.backends._utils import xp_ones
from statgpu.glm_core._squared import SquaredErrorLoss
from ._admm import admm_solver
from ._convergence import ConvergenceWarning
from ._fista_lla_group_contract import _group_surrogate_factory
from ._utils import _validate_sample_weight


_GROUP_NONCONVEX_NAMES = frozenset(
    {"group_mcp", "gmcp", "group_scad", "gscad"}
)


def _external_warning_stacklevel() -> int:
    """Return a warning stacklevel pointing past internal statgpu wrappers."""
    frame = inspect.currentframe()
    if frame is None:
        return 2
    frame = frame.f_back
    level = 1
    try:
        while frame is not None:
            module_name = str(frame.f_globals.get("__name__", ""))
            is_internal = module_name == "statgpu" or module_name.startswith("statgpu.")
            if not is_internal:
                return level
            frame = frame.f_back
            level += 1
    finally:
        del frame
    return 2


def _backend_array(value, *, ref, xp, backend):
    if backend == "torch":
        return xp.as_tensor(value, dtype=ref.dtype, device=ref.device)
    return xp.asarray(value, dtype=ref.dtype)


def _backend_scalar(value, *, ref, xp, backend):
    if backend == "torch":
        return xp.tensor(value, dtype=ref.dtype, device=ref.device)
    return xp.asarray(value, dtype=ref.dtype)


def _normalized_sample_weight(sample_weight, n_samples, X_work, xp, backend):
    if sample_weight is None:
        return None
    sw = _backend_array(sample_weight, ref=X_work, xp=xp, backend=backend).reshape(-1)
    total = xp.sum(sw)
    total_f = float(_to_numpy(total))
    return sw * (float(n_samples) / total_f)


def _initial_params(
    init_coef,
    init_intercept,
    *,
    n_features,
    fit_intercept,
    X_work,
    xp,
    backend,
):
    n_aug = n_features + (1 if fit_intercept else 0)
    if init_coef is None:
        return _zeros(n_aug, backend, ref_tensor=X_work)

    params = _backend_array(init_coef, ref=X_work, xp=xp, backend=backend).reshape(-1)
    if fit_intercept and int(params.shape[0]) == n_features:
        intercept0 = 0.0 if init_intercept is None else float(init_intercept)
        extra = _backend_array([intercept0], ref=X_work, xp=xp, backend=backend)
        params = xp.cat([params, extra]) if backend == "torch" else xp.concatenate([params, extra])
    if int(params.shape[0]) != n_aug:
        raise ValueError(
            f"Quantile Group LLA init_coef has length {int(params.shape[0])}, "
            f"expected {n_aug}"
        )
    return _copy_arr(params)


def _quantile_irls_weights(loss, X_work, y, params, sw, xp, backend):
    """Return the maintained Quantile IRLS/MM observation weights."""
    tau = float(getattr(loss, "_tau", getattr(loss, "quantile", 0.5)))
    residual = y - X_work @ params
    abs_residual = xp.abs(residual)
    eps = _backend_scalar(1e-8, ref=abs_residual, xp=xp, backend=backend)
    abs_safe = xp.maximum(abs_residual, eps)
    if backend == "torch":
        neg = (residual < 0).to(abs_residual.dtype)
    else:
        neg = (residual < 0).astype(abs_residual.dtype)
    asym = tau + (1.0 - 2.0 * tau) * neg
    obs_weight = asym / abs_safe
    if sw is not None:
        obs_weight = obs_weight * sw
    cap = _backend_scalar(1.0e10, ref=obs_weight, xp=xp, backend=backend)
    return xp.minimum(obs_weight, cap)


def _all_zero(values) -> bool:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return bool(array.size and np.all(array == 0.0))


def _flat_irls_boundary_converged(
    loss,
    X_work,
    y_dev,
    params,
    *,
    tol,
    sample_weight,
    fit_intercept,
    xp,
) -> bool:
    """Disambiguate ``n_iter == max_iter`` without accepting an extra iterate.

    ``QuantileLoss.irls`` returns only ``(coef, n_iter)``.  A solve that first
    satisfies its parameter-change criterion on the last allowed iteration and
    a genuinely exhausted solve therefore both report ``n_iter == max_iter``.
    Starting from the returned point, one additional one-step IRLS probe tells
    those states apart.  The probe is diagnostic only: its coefficient is not
    accepted and is not included in the public iteration count.  Its norm stays
    on the active backend; only the final scalar is synchronized.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        probe, _ = loss.irls(
            X_work,
            y_dev,
            penalty=None,
            max_iter=1,
            tol=float(tol),
            init_coef=params,
            sample_weight=sample_weight,
            fit_intercept=fit_intercept,
        )
    delta_dev = xp.linalg.norm(probe - params)
    delta = float(_to_numpy(delta_dev))
    return bool(np.isfinite(delta) and delta < float(tol))


def quantile_group_proximal_irls_lla_solver(
    loss,
    penalty,
    X,
    y,
    alpha_path,
    *,
    max_lla_per_step=6,
    lla_tol=1e-6,
    max_iter=1000,
    tol=1e-4,
    fit_intercept=True,
    sample_weight=None,
    init_coef=None,
    init_intercept=None,
    fail_on_target_nonconvergence=False,
):
    """Solve automatic Quantile Group SCAD/MCP through stable convex surrogates."""
    if str(getattr(loss, "name", "")).lower() != "quantile":
        raise ValueError("quantile_group_proximal_irls_lla_solver requires QuantileLoss")
    penalty_name = str(getattr(penalty, "name", "")).lower()
    if penalty_name not in _GROUP_NONCONVEX_NAMES:
        raise ValueError(
            "quantile_group_proximal_irls_lla_solver requires Group SCAD/MCP"
        )

    backend = _resolve_backend("auto", X)
    if backend == "torch":
        import torch as xp
    elif backend == "cupy":
        import cupy as xp
    else:
        xp = np

    if backend == "torch":
        X_dev = xp.as_tensor(X, dtype=xp.float64, device=getattr(X, "device", None))
        y_dev = xp.as_tensor(y, dtype=xp.float64, device=X_dev.device).reshape(-1)
    else:
        X_dev = xp.asarray(X, dtype=xp.float64)
        y_dev = xp.asarray(y, dtype=xp.float64).reshape(-1)

    n_samples, n_features = int(X_dev.shape[0]), int(X_dev.shape[1])
    if int(y_dev.shape[0]) != n_samples:
        raise ValueError("Quantile Group LLA response length must match X rows")
    _validate_sample_weight(sample_weight, n_samples)

    if fit_intercept:
        ones = xp_ones((n_samples, 1), dtype=X_dev.dtype, xp=xp, ref_arr=X_dev)
        X_work = xp.concatenate([X_dev, ones], axis=1)
    else:
        X_work = X_dev

    params = _initial_params(
        init_coef,
        init_intercept,
        n_features=n_features,
        fit_intercept=fit_intercept,
        X_work=X_work,
        xp=xp,
        backend=backend,
    )
    sw = _normalized_sample_weight(sample_weight, n_samples, X_work, xp, backend)
    factory = _group_surrogate_factory(penalty)
    quadratic_loss = SquaredErrorLoss()
    total_iter = 0
    n_continuation = int(len(alpha_path))

    for cont_i, cont_alpha in enumerate(alpha_path):
        pen_step = copy.copy(penalty)
        pen_step.alpha = float(cont_alpha)
        irls_limit = max_iter[cont_i] if isinstance(max_iter, (list, tuple)) else max_iter
        irls_limit = max(1, int(irls_limit))
        flat_tol = min(float(tol), 1e-8)
        admm_limit = max(500, min(2000, 2 * irls_limit))
        admm_tol = max(float(tol), 1e-7)
        is_final_continuation = cont_i == n_continuation - 1
        lla_converged = False
        flat_irls_exhausted = False
        active_irls_exhausted = False

        for _lla_iter in range(int(max_lla_per_step)):
            feature_params = params[:n_features]
            lla_feature = pen_step.lla_weights(feature_params)
            lla_feature_np = np.asarray(
                _to_numpy(lla_feature), dtype=np.float64
            ).reshape(-1)
            if int(lla_feature_np.size) != n_features:
                raise ValueError(
                    "Quantile Group LLA derivative vector must match feature count"
                )
            before_lla = _copy_arr(params)

            if _all_zero(lla_feature_np):
                # A flat surrogate owns the current target state. Any active
                # IRLS exhaustion from an earlier LLA step is now historical.
                active_irls_exhausted = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    params, used_iter = loss.irls(
                        X_work,
                        y_dev,
                        penalty=None,
                        max_iter=irls_limit,
                        tol=flat_tol,
                        init_coef=None,
                        sample_weight=sw,
                        fit_intercept=fit_intercept,
                    )
                total_iter += int(used_iter)

                at_budget_boundary = (
                    is_final_continuation and int(used_iter) >= irls_limit
                )
                if at_budget_boundary:
                    boundary_converged = _flat_irls_boundary_converged(
                        loss,
                        X_work,
                        y_dev,
                        params,
                        tol=flat_tol,
                        sample_weight=sw,
                        fit_intercept=fit_intercept,
                        xp=xp,
                    )
                    flat_irls_exhausted = not boundary_converged
                else:
                    flat_irls_exhausted = False

                refreshed = pen_step.lla_weights(params[:n_features])
                if _all_zero(_to_numpy(refreshed)):
                    lla_converged = True
                    break
            else:
                # A later active surrogate owns the current target state, so a
                # previously exhausted flat solve cannot poison its verdict.
                flat_irls_exhausted = False
                active_irls_converged = False
                factory_values = (
                    np.concatenate([lla_feature_np, np.zeros(1, dtype=np.float64)])
                    if fit_intercept
                    else lla_feature_np
                )
                inner_penalty = factory(factory_values)

                for _irls_iter in range(irls_limit):
                    params_old = _copy_arr(params)
                    obs_weight = _quantile_irls_weights(
                        loss, X_work, y_dev, params, sw, xp, backend
                    )
                    sqrt_weight = xp.sqrt(obs_weight)
                    X_quad = X_work * sqrt_weight[:, None]
                    y_quad = y_dev * sqrt_weight

                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        params, inner_iter = admm_solver(
                            quadratic_loss,
                            inner_penalty,
                            X_quad,
                            y_quad,
                            max_iter=admm_limit,
                            tol=admm_tol,
                            rho=1.0,
                            adaptive_rho=False,
                            init_coef=params,
                            sample_weight=None,
                        )
                    total_iter += int(inner_iter)

                    delta_dev = xp.max(xp.abs(params - params_old))
                    if float(_to_numpy(delta_dev)) < float(tol):
                        active_irls_converged = True
                        break

                # Unlike the loss-owned flat IRLS call, this loop owns its
                # stopping comparison directly, so exhausting the budget is
                # unambiguous. A small outer LLA change must not convert an
                # unconverged active IRLS target into a successful candidate.
                active_irls_exhausted = (
                    is_final_continuation and not active_irls_converged
                )

            lla_delta = _abs_sum_dev(params - before_lla)
            if float(_to_numpy(lla_delta)) < float(lla_tol):
                lla_converged = True
                break

        if is_final_continuation:
            if flat_irls_exhausted:
                base_message = (
                    "Quantile Group Proximal IRLS-LLA flat target reached "
                    f"max_iter={irls_limit} in Quantile IRLS at "
                    f"alpha={float(cont_alpha):.12g}"
                )
                if fail_on_target_nonconvergence:
                    raise FloatingPointError(
                        base_message
                        + "; the CV candidate was not scored because target convergence was not established."
                    )
                warnings.warn(
                    base_message
                    + "; returning the final iterate. Increase max_iter for a stricter convergence check.",
                    ConvergenceWarning,
                    stacklevel=_external_warning_stacklevel(),
                )
            elif active_irls_exhausted:
                base_message = (
                    "Quantile Group Proximal IRLS-LLA active target reached "
                    f"max_iter={irls_limit} in Quantile IRLS at "
                    f"alpha={float(cont_alpha):.12g}"
                )
                if fail_on_target_nonconvergence:
                    raise FloatingPointError(
                        base_message
                        + "; the CV candidate was not scored because target convergence was not established."
                    )
                warnings.warn(
                    base_message
                    + "; returning the final iterate. Increase max_iter for a stricter convergence check.",
                    ConvergenceWarning,
                    stacklevel=_external_warning_stacklevel(),
                )
            elif not lla_converged:
                base_message = (
                    "Quantile Group Proximal IRLS-LLA reached "
                    f"max_lla_per_step={int(max_lla_per_step)} at the target "
                    f"alpha={float(cont_alpha):.12g}"
                )
                if fail_on_target_nonconvergence:
                    raise FloatingPointError(
                        base_message
                        + "; the CV candidate was not scored because target convergence was not established."
                    )
                warnings.warn(
                    base_message
                    + "; returning the final iterate. Increase max_lla_iters or relax lla_tol if needed.",
                    ConvergenceWarning,
                    stacklevel=_external_warning_stacklevel(),
                )

    params_np = np.asarray(_to_numpy(params), dtype=np.float64).reshape(-1)
    if fit_intercept:
        return params_np[:n_features].copy(), float(params_np[n_features]), total_iter
    return params_np.copy(), 0.0, total_iter


__all__ = ["quantile_group_proximal_irls_lla_solver"]
