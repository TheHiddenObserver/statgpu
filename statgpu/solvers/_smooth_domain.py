"""Private helpers for weighted smooth solvers and loss-owned domains.

The public Newton/L-BFGS APIs stay generic.  Losses may optionally expose
private ``_loss_domain_*`` hooks to provide an interior start, validate an
iterate, and cap a line-search step.  Losses without those hooks keep the
historical unconstrained behavior.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._utils import _get_xp

from ._utils import _as_backend_vector, _validate_sample_weight


class _LossDomainError(RuntimeError):
    """Fail-hard numerical loss-domain error for maintained smooth solvers."""


def _scalar_bool(value) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def _prepare_analytic_sample_weight(
    sample_weight,
    n_samples,
    backend,
    ref_arr,
):
    """Validate, align, and classify analytic weights for smooth solvers.

    Omitted, uniform, and historically effectively-uniform weights execute the
    unweighted objective.  Genuine non-uniform weights remain backend-native.
    Classification happens only after alignment to the executed design dtype
    and device so Newton, L-BFGS, initialization, and domain masking agree.
    """
    if sample_weight is None:
        return None

    _validate_sample_weight(sample_weight, n_samples)
    values = _as_backend_vector(sample_weight, backend, ref_arr).reshape(-1)
    if backend == "torch":
        import torch

        uniform_dev = (
            torch.allclose(values, values[0])
            if torch.is_floating_point(values)
            else torch.all(values == values[0])
        )
    else:
        xp = _get_xp(backend)
        uniform_dev = (
            xp.allclose(values, values[0])
            if getattr(values.dtype, "kind", "") == "f"
            else xp.all(values == values[0])
        )
    return None if _scalar_bool(uniform_dev) else values


def _domain_feasible(loss, X, coef, sample_weight=None) -> bool:
    checker = getattr(loss, "_loss_domain_is_feasible", None)
    if checker is None:
        return True
    return bool(checker(X, coef, sample_weight=sample_weight))


def _initial_smooth_params(
    loss,
    X,
    y,
    *,
    backend,
    n_features,
    init_coef=None,
    sample_weight=None,
):
    """Return a validated explicit or loss-generated initial parameter vector."""
    from ._utils import _zeros

    if init_coef is not None:
        params = _as_backend_vector(init_coef, backend, X)
        if not _domain_feasible(loss, X, params, sample_weight=sample_weight):
            raise ValueError(
                f"Explicit init_coef is outside the maintained optimization "
                f"domain for loss='{getattr(loss, 'name', '?')}'."
            )
        return params

    initializer = getattr(loss, "_loss_domain_initial_point", None)
    if initializer is None:
        params = _zeros(n_features, backend, ref_tensor=X)
    else:
        params = initializer(X, y, sample_weight=sample_weight)
        if params is None:
            params = _zeros(n_features, backend, ref_tensor=X)
        else:
            params = _as_backend_vector(params, backend, X)

    if not _domain_feasible(loss, X, params, sample_weight=sample_weight):
        raise _LossDomainError(
            f"loss='{getattr(loss, 'name', '?')}' did not produce a numerically "
            "certified smooth-domain start."
        )
    return params


def _floating_eps(ref_arr) -> float:
    backend = _resolve_backend("auto", ref_arr)
    if backend == "torch":
        import torch

        dtype = ref_arr.dtype if torch.is_floating_point(ref_arr) else torch.float64
        return float(torch.finfo(dtype).eps)
    dtype = getattr(ref_arr, "dtype", np.dtype(np.float64))
    try:
        return float(np.finfo(dtype).eps)
    except (TypeError, ValueError):
        return float(np.finfo(np.float64).eps)


def _domain_step_floor(ref_arr) -> float:
    """Smallest domain-capped step treated as numerically meaningful."""
    return max(64.0 * _floating_eps(ref_arr), 1e-15)


def _domain_max_step(loss, X, coef, delta, sample_weight=None):
    """Return the loss-owned maximum step for the final additive direction."""
    cap_fn = getattr(loss, "_loss_domain_max_step", None)
    if cap_fn is None:
        return None
    cap = cap_fn(X, coef, delta, sample_weight=sample_weight)
    if cap is None:
        return None
    cap = float(cap)
    if not np.isfinite(cap) or cap <= 0.0:
        raise _LossDomainError(
            f"loss='{getattr(loss, 'name', '?')}' has no positive interior "
            "line-search step for the final search direction."
        )
    if cap <= _domain_step_floor(X):
        raise _LossDomainError(
            f"loss='{getattr(loss, 'name', '?')}' is pinned to the maintained "
            "smooth-domain boundary before gradient convergence."
        )
    return cap
