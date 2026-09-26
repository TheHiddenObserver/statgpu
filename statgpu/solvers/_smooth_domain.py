"""Private helpers for weighted smooth solvers and loss-owned domains.

The public Newton/L-BFGS APIs stay generic. Losses may optionally expose
private ``_loss_domain_*`` hooks to provide an interior start, validate an
iterate, and cap a line-search step. Losses without those hooks keep the
historical unconstrained behavior.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._utils import _get_xp, xp_asarray

from ._utils import _as_backend_vector, _native_sample_weight


# Preserve the historical relative tolerance for "effectively uniform" weights
# while making the classification invariant to positive global rescaling and
# observation ordering.
_EFFECTIVELY_UNIFORM_WEIGHT_RTOL = 1e-5


class _LossDomainError(RuntimeError):
    """Fail-hard numerical loss-domain error for maintained smooth solvers."""


def _scalar_bool(value) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def _aligned_weight_dtype(ref_arr, backend):
    """Choose the floating dtype consumed by the executed smooth objective."""
    if backend == "torch":
        import torch

        return ref_arr.dtype if torch.is_floating_point(ref_arr) else torch.float64
    dtype = getattr(ref_arr, "dtype", np.dtype(np.float64))
    if getattr(dtype, "kind", "") == "f":
        return dtype
    return _get_xp(backend).float64


def _effectively_uniform_weights(values, backend) -> bool:
    """Return whether aligned non-negative weights are relatively uniform.

    The criterion uses the full weight range rather than one observation as an
    ``allclose`` reference. This makes the classification symmetric under row
    permutations and homogeneous under positive global rescaling:

        max(w) - min(w) <= rtol * max(w).
    """
    xp = _get_xp(backend)
    w_min = xp.min(values)
    w_max = xp.max(values)
    return _scalar_bool(
        (w_max - w_min) <= _EFFECTIVELY_UNIFORM_WEIGHT_RTOL * w_max
    )


def _validate_analytic_weight_shape_values(sample_weight, n_samples):
    """Validate smooth-solver weights without forming their raw floating sum.

    For finite non-negative weights, ``sum(w) > 0`` is equivalent to at least
    one strictly positive element. Checking that fact through ``max(w)`` avoids
    rejecting an otherwise valid normalized analytic-weight problem merely
    because a float16/float32/float64 raw reduction overflows after a positive
    global rescaling. Other solvers retain their existing validator semantics;
    this helper is private to the scale-normalizing Newton/L-BFGS path.
    """
    _source_backend, source_xp, source_values = _native_sample_weight(sample_weight)
    if int(source_values.ndim) != 1 or int(source_values.shape[0]) != int(n_samples):
        raise ValueError("sample_weight must be 1D with length n_samples")
    try:
        finite = source_xp.all(source_xp.isfinite(source_values))
        negative = source_xp.any(source_values < 0)
        positive = source_xp.any(source_values > 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight must contain real finite values") from exc
    if not _scalar_bool(finite):
        raise ValueError("sample_weight must contain only finite values")
    if _scalar_bool(negative):
        raise ValueError("sample_weight must be non-negative")
    if not _scalar_bool(positive):
        raise ValueError("sample_weight must have a finite positive sum")


def _prepare_analytic_sample_weight(
    sample_weight,
    n_samples,
    backend,
    ref_arr,
):
    """Validate, normalize, align, and classify analytic smooth-solver weights.

    Analytic weights define a normalized objective, so multiplying every weight
    by one positive constant must not change either the objective or the path
    classification. Validate finite/non-negative/positive-mass input without a
    raw sum, then move the weights to the executed backend in float64 and divide
    by their maximum. The resulting vector lies in ``[0, 1]`` with at least one
    exact 1, avoiding scale-induced reduction overflow and cast overflow before
    conversion to the numerical design dtype.

    Omitted, uniform, and historically effectively-uniform weights execute the
    unweighted objective. Genuine non-uniform weights remain backend-native.
    Integral design matrices cannot truncate fractional weights because the
    execution dtype is promoted to float64 in that case.
    """
    if sample_weight is None:
        return None

    _validate_analytic_weight_shape_values(sample_weight, n_samples)
    xp = _get_xp(backend)

    # Normalize in backend-native float64 *before* casting to the design dtype.
    # The source-side validation guarantees finite non-negative values and at
    # least one positive entry; no raw sum must be representable at this stage.
    wide = xp_asarray(
        sample_weight,
        dtype=xp.float64,
        xp=xp,
        ref_arr=ref_arr,
    ).reshape(-1)
    wide_max = xp.max(wide)
    if not _scalar_bool(xp.isfinite(wide_max)) or not _scalar_bool(wide_max > 0):
        raise ValueError(
            "sample_weight must be representable as finite positive float64 "
            "values on the executed smooth-solver backend"
        )
    wide = wide / wide_max

    values = xp_asarray(
        wide,
        dtype=_aligned_weight_dtype(ref_arr, backend),
        xp=xp,
        ref_arr=ref_arr,
    ).reshape(-1)
    return None if _effectively_uniform_weights(values, backend) else values


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
                f"Explicit init_coef is outside the supported optimization "
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
            f"loss='{getattr(loss, 'name', '?')}' is pinned to the "
            "smooth-domain boundary before gradient convergence."
        )
    return cap
