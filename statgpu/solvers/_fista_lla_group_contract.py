"""Exact group-LLA surrogate scaling and Quantile LLA inner-solver ownership.

Group MCP/SCAD local-linear approximation (LLA) produces derivatives with
respect to each group norm.  The matching convex surrogate is

    sum_g D_g ||beta_g||_2.

``AdaptiveGroupLassoPenalty(alpha=1, weights=D_g/sqrt(p_g))`` represents this
surrogate exactly.  The wrapper below owns that mapping for every public Group
MCP/SCAD ``fista_lla_path`` call.

For smooth/robust losses the historical fused FISTA-LLA implementation remains
unchanged.  Quantile/check loss is deliberately different: its subgradient is
discontinuous, and the historical fixed-step FISTA inner loop can exhaust its
iteration budget without converging.  Small backend roundoff differences can
then be amplified by the non-convex outer LLA path.  Quantile LLA therefore
uses its standard IRLS quadratic majorization and solves each resulting convex
L1/Group-Lasso surrogate with the maintained squared-error FISTA engine.  This
keeps scalar and group Quantile LLA on one objective-consistent numerical core.
When the LLA derivative is identically zero, the target surrogate is exactly
unpenalized Quantile regression and closes through ``QuantileLoss.irls()``.
"""

from __future__ import annotations

import copy
import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._array_ops import _copy_arr, _zeros
from statgpu.backends._utils import xp_ones
from statgpu.penalties import AdaptiveGroupLassoPenalty, AdaptiveL1Penalty
from ._fista_lla import fista_lla_path as _base_fista_lla_path


_GROUP_NONCONVEX_NAMES = frozenset(
    {"group_mcp", "gmcp", "group_scad", "gscad"}
)


class _GroupFISTALossProxy:
    """Delegate a loss while disabling the generic proximal-Newton branch."""

    has_hessian = False

    def __init__(self, loss):
        self._loss = loss

    def __getattr__(self, name):
        return getattr(self._loss, name)


def _group_surrogate_factory(scad_penalty):
    groups = getattr(scad_penalty, "_group_indices", None)
    if groups is None:
        raise ValueError("group penalty must define group indices for LLA")
    group_indices = [np.asarray(group, dtype=np.int64) for group in groups]
    group_sizes = np.asarray([len(group) for group in group_indices], dtype=float)
    if np.any(group_sizes <= 0):
        raise ValueError("group penalty contains an empty group")

    inner_penalty = AdaptiveGroupLassoPenalty(
        groups=group_indices,
        alpha=1.0,
        weights=np.ones(len(group_indices), dtype=float),
    )
    # Quantile/smooth LLA can append one unpenalized intercept coordinate to
    # the numerical design. Public group specifications remain feature-only.
    inner_penalty._allow_trailing_unpenalized_intercept = True

    def factory(per_coordinate_derivatives):
        values = np.asarray(per_coordinate_derivatives, dtype=np.float64).ravel()
        group_weights = np.empty(len(group_indices), dtype=np.float64)
        for group_id, (indices, size) in enumerate(
            zip(group_indices, group_sizes)
        ):
            if indices.size == 0 or int(indices.max()) >= values.size:
                raise ValueError("LLA derivative vector is shorter than group indices")
            derivatives = values[indices]
            if derivatives.size != int(size):
                raise ValueError("LLA derivative vector is shorter than group indices")
            if not np.all(np.isfinite(derivatives)):
                raise FloatingPointError("group LLA derivatives must be finite")
            reference = float(derivatives[0])
            if not np.allclose(
                derivatives,
                reference,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "group LLA derivatives must be constant within each group"
                )
            if reference < -1e-12:
                raise ValueError("group LLA derivatives must be non-negative")
            group_weights[group_id] = max(reference, 0.0) / np.sqrt(size)
        inner_penalty.set_weights(group_weights)
        return inner_penalty

    return factory


def _scalar_surrogate_factory(per_coordinate_derivatives):
    """Return the exact Adaptive-L1 surrogate without weight renormalization."""
    values = np.asarray(per_coordinate_derivatives, dtype=np.float64).ravel()
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("scalar LLA derivatives must be finite")
    if np.any(values < -1e-12):
        raise ValueError("scalar LLA derivatives must be non-negative")
    values = np.maximum(values, 0.0)
    return AdaptiveL1Penalty(
        alpha=1.0,
        normalize=False,
        weights=values,
    )


def _backend_scalar(value, ref, xp, backend):
    if backend == "torch":
        return xp.tensor(value, dtype=ref.dtype, device=ref.device)
    return xp.asarray(value, dtype=ref.dtype)


def _to_backend_vector(value, ref, xp, backend):
    if backend == "torch":
        return xp.as_tensor(value, dtype=ref.dtype, device=ref.device)
    return xp.asarray(value, dtype=ref.dtype)


def _augment_initial_params(
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

    params = _to_backend_vector(init_coef, X_work, xp, backend).reshape(-1)
    if fit_intercept and int(params.shape[0]) == n_features:
        intercept0 = 0.0 if init_intercept is None else float(init_intercept)
        extra = _to_backend_vector([intercept0], X_work, xp, backend)
        params = xp.cat([params, extra]) if backend == "torch" else xp.concatenate([params, extra])
    if int(params.shape[0]) != n_aug:
        raise ValueError(
            f"Quantile LLA init_coef has length {int(params.shape[0])}, expected {n_aug}"
        )
    return _copy_arr(params)


def _normalized_sample_weight(sample_weight, n_samples, X_work, xp, backend):
    if sample_weight is None:
        return None
    sw = _to_backend_vector(sample_weight, X_work, xp, backend).reshape(-1)
    if int(sw.shape[0]) != int(n_samples):
        raise ValueError("sample_weight must have length n_samples")
    total = xp.sum(sw)
    total_f = float(_to_numpy(total))
    if not np.isfinite(total_f) or total_f <= 0.0:
        raise ValueError("sample_weight must have a finite positive sum")
    return sw * (float(n_samples) / total_f)


def _quantile_irls_weights(loss, X_work, y, params, sw, xp, backend):
    """Build the same Quantile IRLS observation weights as the maintained solver."""
    tau = float(getattr(loss, "_tau", getattr(loss, "quantile", 0.5)))
    residual = y - X_work @ params
    abs_residual = xp.abs(residual)
    eps = _backend_scalar(1e-8, abs_residual, xp, backend)
    abs_safe = xp.maximum(abs_residual, eps)
    if backend == "torch":
        neg = (residual < 0).to(abs_residual.dtype)
    else:
        neg = (residual < 0).astype(abs_residual.dtype)
    asym = tau + (1.0 - 2.0 * tau) * neg
    weights = asym / abs_safe
    if sw is not None:
        weights = weights * sw
    # Match the scalar Proximal-IRLS safety cap.
    cap = _backend_scalar(1.0e10, weights, xp, backend)
    return xp.minimum(weights, cap)


def _surrogate_is_zero(derivatives):
    values = np.asarray(derivatives, dtype=np.float64).ravel()
    return bool(values.size and np.all(values == 0.0))


def _quantile_fista_lla_path(
    loss,
    nonconvex_penalty,
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
    init_coef=None,
    init_intercept=None,
    return_path=False,
):
    """Quantile LLA via IRLS majorization and convex FISTA subproblems."""
    backend = _resolve_backend("auto", X)
    if backend == "torch":
        import torch as xp
    elif backend == "cupy":
        import cupy as xp
    else:
        xp = np

    X_dev = _to_backend_vector(X, X, xp, backend)
    y_dev = _to_backend_vector(y, X_dev, xp, backend).reshape(-1)
    n_samples, n_features = X_dev.shape
    if int(y_dev.shape[0]) != int(n_samples):
        raise ValueError("Quantile LLA response length must match X rows")

    if fit_intercept:
        ones = xp_ones((n_samples, 1), dtype=X_dev.dtype, xp=xp, ref_arr=X_dev)
        X_work = xp.concatenate([X_dev, ones], axis=1)
    else:
        X_work = X_dev

    params = _augment_initial_params(
        init_coef,
        init_intercept,
        n_features=n_features,
        fit_intercept=fit_intercept,
        X_work=X_work,
        xp=xp,
        backend=backend,
    )
    sw = _normalized_sample_weight(sample_weight, n_samples, X_work, xp, backend)

    penalty_name = str(getattr(nonconvex_penalty, "name", "")).lower()
    is_group = penalty_name in _GROUP_NONCONVEX_NAMES
    if is_group:
        factory = lla_penalty_factory or _group_surrogate_factory(nonconvex_penalty)
    else:
        factory = lla_penalty_factory or _scalar_surrogate_factory

    from statgpu.glm_core._squared import SquaredErrorLoss
    from ._fista import fista_solver

    quadratic_loss = SquaredErrorLoss()
    total_iter = 0
    path_records = [] if return_path else None

    def split_current(current):
        current_np = np.asarray(_to_numpy(current), dtype=np.float64).reshape(-1)
        if fit_intercept:
            return current_np[:n_features].copy(), float(current_np[n_features])
        return current_np.copy(), 0.0

    for cont_i, cont_alpha in enumerate(alpha_path):
        pen_step = copy.copy(nonconvex_penalty)
        pen_step.alpha = float(cont_alpha)
        irls_limit = max_iter[cont_i] if isinstance(max_iter, (list, tuple)) else max_iter
        irls_limit = max(1, int(irls_limit))
        # Each WLS surrogate is convex and warm-started.  A bounded inner solve
        # is enough to make the IRLS/MM step descend while keeping the overall
        # continuation cost predictable on large problems.
        fista_limit = max(50, min(250, irls_limit))
        fista_tol = min(max(float(tol) * 0.1, 1e-10), 1e-7)

        for _ in range(int(max_lla_per_step)):
            # Public group penalties are feature-dimensional.  Never pass the
            # augmented intercept coordinate into their LLA derivative API.
            feature_params = params[:n_features]
            lla_feature = pen_step.lla_weights(feature_params)
            lla_feature_np = np.asarray(
                _to_numpy(lla_feature), dtype=np.float64
            ).reshape(-1)
            if int(lla_feature_np.size) != int(n_features):
                raise ValueError(
                    "Quantile LLA derivative vector must match feature count"
                )

            if fit_intercept:
                lla_factory_values = np.concatenate(
                    [lla_feature_np, np.zeros(1, dtype=np.float64)]
                )
            else:
                lla_factory_values = lla_feature_np
            inner_penalty = factory(lla_factory_values)
            before_lla = _copy_arr(params)

            if _surrogate_is_zero(lla_feature_np):
                # Flat SCAD/MCP region: the exact LLA target is unpenalized
                # weighted Quantile regression.
                params, used_iter = loss.irls(
                    X_work,
                    y_dev,
                    penalty=None,
                    max_iter=irls_limit,
                    tol=min(float(tol), 1e-8),
                    init_coef=params,
                    sample_weight=sw,
                    fit_intercept=fit_intercept,
                )
                total_iter += int(used_iter)
            else:
                # Quantile IRLS/MM: at each iterate build the weighted least-
                # squares quadratic majorizer, then solve that convex L1/group
                # surrogate through the maintained squared-error FISTA engine.
                for _irls_iter in range(irls_limit):
                    params_old = _copy_arr(params)
                    obs_weight = _quantile_irls_weights(
                        loss, X_work, y_dev, params, sw, xp, backend
                    )
                    sqrt_weight = xp.sqrt(obs_weight)
                    X_quad = X_work * sqrt_weight[:, None]
                    y_quad = y_dev * sqrt_weight
                    params, inner_iter = fista_solver(
                        quadratic_loss,
                        inner_penalty,
                        X_quad,
                        y_quad,
                        max_iter=fista_limit,
                        tol=fista_tol,
                        init_coef=params,
                        sample_weight=None,
                        cv_mode=False,
                    )
                    total_iter += int(inner_iter)
                    delta_dev = xp.max(xp.abs(params - params_old))
                    if float(_to_numpy(delta_dev)) < float(tol):
                        break

            lla_delta = xp.max(xp.abs(params - before_lla))
            if float(_to_numpy(lla_delta)) < float(lla_tol):
                break

        if path_records is not None:
            coef_rec, intercept_rec = split_current(params)
            path_records.append(
                {
                    "alpha": float(cont_alpha),
                    "coef": coef_rec,
                    "intercept": intercept_rec,
                    "n_iter": int(total_iter),
                }
            )

    coef_np, intercept = split_current(params)
    if return_path:
        path = {
            "alpha": np.asarray([r["alpha"] for r in path_records], dtype=np.float64),
            "coef": np.vstack([r["coef"] for r in path_records]).astype(
                np.float64, copy=False
            ),
            "intercept": np.asarray(
                [r["intercept"] for r in path_records], dtype=np.float64
            ),
            "n_iter": np.asarray([r["n_iter"] for r in path_records], dtype=np.int64),
        }
        return coef_np, intercept, total_iter, path
    return coef_np, intercept, total_iter


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
    init_coef=None,
    init_intercept=None,
    return_path=False,
):
    """Run LLA with exact group scaling and Quantile-specific majorization."""
    is_quantile = str(getattr(loss, "name", "")).lower() == "quantile"
    penalty_name = str(getattr(scad_penalty, "name", "")).lower()

    if is_quantile:
        # Scalar and group Quantile LLA share the same IRLS/MM numerical core;
        # only the convex surrogate penalty factory differs.
        factory = lla_penalty_factory
        if penalty_name in _GROUP_NONCONVEX_NAMES:
            factory = _group_surrogate_factory(scad_penalty)
        return _quantile_fista_lla_path(
            loss,
            scad_penalty,
            X,
            y,
            alpha_path,
            max_lla_per_step=max_lla_per_step,
            lla_tol=lla_tol,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            sample_weight=sample_weight,
            lla_penalty_factory=factory,
            init_coef=init_coef,
            init_intercept=init_intercept,
            return_path=return_path,
        )

    if penalty_name in _GROUP_NONCONVEX_NAMES:
        lla_penalty_factory = _group_surrogate_factory(scad_penalty)
        loss = _GroupFISTALossProxy(loss)

    return _base_fista_lla_path(
        loss,
        scad_penalty,
        X,
        y,
        alpha_path,
        max_lla_per_step=max_lla_per_step,
        lla_tol=lla_tol,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        sample_weight=sample_weight,
        lla_penalty_factory=lla_penalty_factory,
        init_coef=init_coef,
        init_intercept=init_intercept,
        return_path=return_path,
    )
