"""Group-penalty contract wrapper for the fused FISTA-LLA path.

The base fused solver expects an optional factory mapping the current penalty's
per-coordinate LLA derivatives to an inner convex penalty. Group MCP/SCAD
provide the derivative with respect to each group norm, repeated on the group's
original feature coordinates. The matching convex surrogate is

    sum_g D_g ||beta_g||_2.

``AdaptiveGroupLassoPenalty(alpha=1, weights=D_g/sqrt(p_g))`` represents this
surrogate exactly. The historical estimator caller instead took an L2 norm of
the repeated derivatives and used the target regularization strength again,
producing ``alpha_target * p_g * D_g``. Direct public solver calls without a
factory fell back to coordinate-wise Adaptive L1. Both paths optimize the wrong
surrogate and are normalized here.

The generic proximal-Newton inner loop is intentionally disabled for group
nonconvex LLA. Its Armijo condition is based on a smooth Newton direction plus a
post-hoc group proximal map; on valid Huber Group MCP/SCAD problems it can reject
all trial steps, restore the old iterate, and return without a failure status.
The group-aware fixed-step FISTA path uses the loss Lipschitz/step-scale contract
and the exact weighted Group Lasso proximal operator, so convergence is
observable through actual proximal updates rather than a silently stalled
Newton step.

Quantile is a narrower exception. Its pinball loss is non-smooth, and the fused
fixed-step inner loop can exhaust its iteration limit while still moving enough
for tiny backend roundoff differences to select different non-convex LLA
trajectories. Quantile Group SCAD/MCP therefore keeps the same outer LLA
algorithm but solves each convex Adaptive-Group-Lasso surrogate with the
maintained generic Group FISTA/backtracking engine. When every LLA derivative is
zero, the surrogate is exactly unpenalized Quantile regression and closes
through ``QuantileLoss.irls()`` instead of returning an unconverged first-order
iterate.
"""

from __future__ import annotations

import copy
import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._array_ops import _abs_sum_dev, _copy_arr, _zeros
from statgpu.backends._utils import xp_ones
from statgpu.penalties import AdaptiveGroupLassoPenalty
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


class _QuantileWeightedStepScaleProxy:
    """Retain analytic weights across Quantile FISTA-LLA step refreshes.

    The fused engine supplies ``sample_weight`` on its initial step-scale call
    but omits it on periodic refreshes. Quantile's step scale follows the
    normalized weighted objective, so every public Quantile FISTA-LLA call must
    reuse the same backend-native weights. This proxy is Quantile-only; all
    other losses retain their previously validated behavior.
    """

    def __init__(self, loss):
        self._loss = loss
        self._sample_weight = None

    def __getattr__(self, name):
        return getattr(self._loss, name)

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        if sample_weight is not None:
            self._sample_weight = sample_weight
        effective_weight = (
            sample_weight if sample_weight is not None else self._sample_weight
        )
        return self._loss.lipschitz(
            X,
            coef,
            y=y,
            sample_weight=effective_weight,
        )


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
    # The numerical design can append one unpenalized intercept coordinate.
    # Public group penalties remain exact-dimensional; only this private
    # surrogate opts into that one-coordinate extension.
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


def _quantile_group_fista_lla_path(
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
    """Quantile Group LLA with convergent convex-surrogate ownership.

    Each LLA surrogate is an Adaptive Group Lasso problem. Non-zero surrogates
    are delegated to the maintained generic Group FISTA solver (including its
    backtracking/failure semantics); an exactly zero surrogate is ordinary
    weighted Quantile regression and is solved by ``loss.irls``.
    """
    backend = _resolve_backend("auto", X)
    if backend == "torch":
        import torch as xp
    elif backend == "cupy":
        import cupy as xp
    else:
        xp = np

    n_samples, n_features = X.shape
    if fit_intercept:
        ones = xp_ones((n_samples, 1), dtype=X.dtype, xp=xp, ref_arr=X)
        X_work = xp.concatenate([X, ones], axis=1)
        n_aug = n_features + 1
    else:
        X_work = X
        n_aug = n_features

    if init_coef is None:
        params = _zeros(n_aug, backend, ref_tensor=X_work)
    else:
        if backend == "torch":
            params = xp.as_tensor(init_coef, dtype=X_work.dtype, device=X_work.device)
        else:
            params = xp.asarray(init_coef, dtype=X_work.dtype)
        if fit_intercept and int(params.shape[0]) == n_features:
            intercept0 = 0.0 if init_intercept is None else float(init_intercept)
            if backend == "torch":
                params = xp.cat(
                    [
                        params,
                        xp.tensor([intercept0], dtype=X_work.dtype, device=X_work.device),
                    ]
                )
            else:
                params = xp.concatenate(
                    [params, xp.asarray([intercept0], dtype=X_work.dtype)]
                )
        else:
            params = _copy_arr(params)

    factory = lla_penalty_factory or _group_surrogate_factory(scad_penalty)
    total_iter = 0
    path_records = [] if return_path else None

    def _split(current):
        current_np = np.asarray(_to_numpy(current), dtype=np.float64).reshape(-1)
        if fit_intercept:
            return current_np[:n_features].copy(), float(current_np[n_features])
        return current_np.copy(), 0.0

    for cont_i, cont_alpha in enumerate(alpha_path):
        pen_step = copy.copy(scad_penalty)
        pen_step.alpha = float(cont_alpha)
        mi = max_iter[cont_i] if isinstance(max_iter, (list, tuple)) else max_iter
        mi = int(mi)

        for _ in range(int(max_lla_per_step)):
            lla_weights = pen_step.lla_weights(params)
            lla_weights_np = np.asarray(_to_numpy(lla_weights), dtype=np.float64)
            inner_penalty = factory(lla_weights_np)
            before = _copy_arr(params)

            group_weights = np.asarray(
                getattr(inner_penalty, "_group_weights", ()), dtype=np.float64
            )
            zero_surrogate = bool(
                group_weights.size and np.all(group_weights == 0.0)
            )

            if zero_surrogate:
                # The LLA surrogate contains no active penalty. Quantile IRLS
                # is the maintained full weighted solver for this exact convex
                # objective and avoids accepting a max-iteration FISTA iterate.
                params, used_iter = loss.irls(
                    X_work,
                    y,
                    penalty=None,
                    max_iter=mi,
                    tol=min(float(tol), 1e-8),
                    init_coef=params,
                    sample_weight=sample_weight,
                    fit_intercept=fit_intercept,
                )
            else:
                from ._fista import fista_solver

                params, used_iter = fista_solver(
                    loss,
                    inner_penalty,
                    X_work,
                    y,
                    max_iter=mi,
                    tol=tol,
                    init_coef=params,
                    sample_weight=sample_weight,
                    cv_mode=False,
                )

            total_iter += int(used_iter)
            delta = float(_to_numpy(_abs_sum_dev(params - before)))
            if delta < float(lla_tol):
                break

        if path_records is not None:
            coef_rec, intercept_rec = _split(params)
            path_records.append(
                {
                    "alpha": float(cont_alpha),
                    "coef": coef_rec,
                    "intercept": intercept_rec,
                    "n_iter": int(total_iter),
                }
            )

    coef_np, intercept = _split(params)
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
    """Run the fused LLA path with exact Group MCP/SCAD surrogate scaling."""
    is_quantile = str(getattr(loss, "name", "")).lower() == "quantile"
    if is_quantile:
        loss = _QuantileWeightedStepScaleProxy(loss)

    penalty_name = str(getattr(scad_penalty, "name", "")).lower()
    if penalty_name in _GROUP_NONCONVEX_NAMES:
        lla_penalty_factory = _group_surrogate_factory(scad_penalty)
        if is_quantile:
            return _quantile_group_fista_lla_path(
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
