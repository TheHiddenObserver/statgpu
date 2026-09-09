"""Fail-closed reporting guard for centered debiased intercept inference.

PR #138 gives fit-intercept sparse-Gaussian debiased inference a coherent
original-coordinate intercept.  The feature-only debiased calculation can still
reach non-representable reporting arrays after finite inputs (for example through
nodewise-precision overflow).  Validate the final centered marginal reporting
surface before downstream simultaneous inference treats it as a valid input.

Statistics are intentionally not required to be finite: a zero standard error
with a nonzero estimate can have a meaningful signed-infinite z statistic.  The
published parameter estimates, standard errors, p-values, and confidence
intervals themselves must be finite and internally valid.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.linear_model import (
    _post_selection_ols_review_fix_contract as _weighted_contract,
)


_FINALIZER_MARKER = "__statgpu_pr138_debiased_marginal_finite__"
_ORIGINAL_FINALIZER = _weighted_contract._finalize_weighted_debiased_result


def _validate_centered_marginal_result(result) -> None:
    params = np.asarray(result.params, dtype=np.float64).reshape(-1)
    bse = np.asarray(result.bse, dtype=np.float64).reshape(-1)
    pvalues = np.asarray(result.pvalues, dtype=np.float64).reshape(-1)
    conf_int = np.asarray(result.conf_int, dtype=np.float64)

    n_params = int(params.shape[0])
    if (
        bse.shape != (n_params,)
        or pvalues.shape != (n_params,)
        or conf_int.shape != (n_params, 2)
    ):
        raise RuntimeError(
            "centered debiased inference reporting arrays have inconsistent shapes"
        )

    if not (
        np.all(np.isfinite(params))
        and np.all(np.isfinite(bse))
        and np.all(np.isfinite(pvalues))
        and np.all(np.isfinite(conf_int))
    ):
        raise FloatingPointError(
            "centered debiased inference produced non-finite parameter estimates, "
            "standard errors, p-values, or confidence intervals"
        )
    if np.any(bse < 0.0):
        raise FloatingPointError(
            "centered debiased inference produced a negative standard error"
        )
    if np.any((pvalues < 0.0) | (pvalues > 1.0)):
        raise FloatingPointError(
            "centered debiased inference produced a p-value outside [0, 1]"
        )
    if np.any(conf_int[:, 0] > conf_int[:, 1]):
        raise FloatingPointError(
            "centered debiased inference produced a reversed confidence interval"
        )


@functools.wraps(_ORIGINAL_FINALIZER)
def _finalize_weighted_debiased_result(
    model,
    *,
    X_arr,
    y_work,
    X_work,
    row_scale,
    coef_native,
    backend_name: str,
    original_intercept: bool,
    simultaneous_requested: bool,
    sample_weighted: bool = True,
):
    result = _ORIGINAL_FINALIZER(
        model,
        X_arr=X_arr,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef_native,
        backend_name=backend_name,
        original_intercept=original_intercept,
        simultaneous_requested=simultaneous_requested,
        sample_weighted=sample_weighted,
    )
    if original_intercept:
        _validate_centered_marginal_result(result)
    return result


def install_debiased_marginal_finite_contract() -> None:
    current = _weighted_contract._finalize_weighted_debiased_result
    if getattr(current, _FINALIZER_MARKER, False):
        return
    setattr(_finalize_weighted_debiased_result, _FINALIZER_MARKER, True)
    _weighted_contract._finalize_weighted_debiased_result = (
        _finalize_weighted_debiased_result
    )


__all__ = [
    "install_debiased_marginal_finite_contract",
    "_validate_centered_marginal_result",
]
