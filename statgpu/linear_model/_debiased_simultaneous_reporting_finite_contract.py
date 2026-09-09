"""Fail-closed reporting guard for debiased simultaneous inference.

The centered/intercept-inclusive CPU path and the centered CuPy/Torch-native
path already reject non-finite bootstrap max-|Z| draws before publishing a joint
interval.  The historical feature-only CPU helper predates that contract and can
instead let a non-finite bootstrap draw reach ``np.quantile``, leaving a NaN
critical value / joint interval while marking simultaneous inference enabled.

Keep the historical numerical helper and target-family semantics unchanged, but
validate every successful publication at the final method boundary.  This makes
the public simultaneous-inference surface fail closed consistently across
feature-only/intercept-inclusive and CPU/GPU execution paths.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


_SIMULTANEOUS_REPORTING_FINITE_MARKER = (
    "__statgpu_pr138_simultaneous_reporting_finite__"
)
_ORIGINAL_SIMULTANEOUS = (
    PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap
)


def _clear_failed_simultaneous_publication(model) -> None:
    """Remove result-bearing joint state after a rejected publication."""
    model._conf_int_simultaneous = None
    model._simultaneous_enabled = False
    for name in (
        "_simultaneous_critical_value",
        "_simultaneous_target_mask",
    ):
        model.__dict__.pop(name, None)


def _validate_simultaneous_publication(model) -> None:
    """Reject a non-representable joint result after the numerical helper returns."""
    if not bool(getattr(model, "_simultaneous_enabled", False)):
        return

    critical = getattr(model, "_simultaneous_critical_value", None)
    conf_int = getattr(model, "_conf_int_simultaneous", None)
    valid_critical = False
    try:
        critical_value = float(critical)
        valid_critical = np.isfinite(critical_value) and critical_value >= 0.0
    except (TypeError, ValueError, OverflowError):
        critical_value = float("nan")

    valid_conf_int = False
    if conf_int is not None:
        try:
            conf_arr = np.asarray(conf_int, dtype=np.float64)
            valid_conf_int = (
                conf_arr.ndim == 2
                and conf_arr.shape[1] == 2
                and bool(np.all(np.isfinite(conf_arr)))
                and bool(np.all(conf_arr[:, 0] <= conf_arr[:, 1]))
            )
        except (TypeError, ValueError, OverflowError):
            valid_conf_int = False

    if valid_critical and valid_conf_int:
        return

    _clear_failed_simultaneous_publication(model)
    if not valid_critical:
        raise FloatingPointError(
            "simultaneous debiased inference produced a non-finite or negative "
            "critical value"
        )
    raise FloatingPointError(
        "simultaneous debiased inference produced non-finite, malformed, or "
        "reversed confidence intervals"
    )


@functools.wraps(_ORIGINAL_SIMULTANEOUS)
def _compute_simultaneous_ci_maxz_bootstrap(self):
    result = _ORIGINAL_SIMULTANEOUS(self)
    _validate_simultaneous_publication(self)
    return result


def install_debiased_simultaneous_reporting_finite_contract() -> None:
    current = PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap
    if getattr(current, _SIMULTANEOUS_REPORTING_FINITE_MARKER, False):
        return
    setattr(
        _compute_simultaneous_ci_maxz_bootstrap,
        _SIMULTANEOUS_REPORTING_FINITE_MARKER,
        True,
    )
    PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap = (
        _compute_simultaneous_ci_maxz_bootstrap
    )


__all__ = [
    "install_debiased_simultaneous_reporting_finite_contract",
    "_validate_simultaneous_publication",
]
