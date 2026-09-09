"""Exact uniform-weight identity for the reviewed LassoCV selector.

Analytic weights are invariant to multiplication by a common positive constant.
In particular, ``sample_weight=c * ones(n)`` is the same statistical problem as
omitting weights.  The corrected non-uniform weighted CV path uses an algebraic
row transformation, but taking that path for exactly uniform weights can still
introduce avoidable floating-point differences relative to the maintained
unweighted fast-fold implementation.

Keep this identity exact: uniform positive weights delegate to the already
maintained unweighted selector, while genuinely non-uniform weights continue
through the corrected weighted objective contract.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model.cv._device import (
    resolve_cv_backend,
    validate_cv_sample_weight,
)
from statgpu.linear_model.wrappers import _lasso as _lasso_impl


_UNIFORM_WEIGHT_MARKER = "__statgpu_pr138_lassocv_uniform_weight_identity__"
_REVIEWED_SELECT = _lasso_impl._select_lasso_alpha_cv


def _uniform_positive_weight(X, sample_weight, device) -> bool:
    if sample_weight is None:
        return False

    (
        _device_name,
        _backend_name,
        backend,
        use_gpu,
        _gpu_input_cupy,
        _gpu_input_torch,
    ) = resolve_cv_backend(device, X)
    n_samples = int(X.shape[0])
    validated = validate_cv_sample_weight(sample_weight, n_samples)

    if use_gpu:
        weight = backend.asarray(validated, dtype=backend.float64).reshape(-1)
        first = weight[0]
        span = backend.max(backend.abs(weight - first))
        first_value = float(np.asarray(_to_numpy(first), dtype=np.float64))
        span_value = float(np.asarray(_to_numpy(span), dtype=np.float64))
    else:
        weight = np.asarray(validated, dtype=np.float64).reshape(-1)
        first_value = float(weight[0])
        span_value = float(np.max(np.abs(weight - first_value)))

    return np.isfinite(first_value) and first_value > 0.0 and span_value == 0.0


@functools.wraps(_REVIEWED_SELECT)
def _select_lasso_alpha_cv(
    X,
    y,
    *,
    alphas=None,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state=None,
    sample_weight=None,
    fit_intercept: bool = False,
    device="cpu",
    max_iter: int = 3000,
    tol: float = 1e-4,
    cpu_solver: str = "coordinate_descent",
    method: str = "standard",
    cd_kkt_check_every=None,
    gpu_cv_mixed_precision: bool = True,
    return_details: bool = False,
    cache_key=None,
):
    effective_weight = (
        None
        if _uniform_positive_weight(X, sample_weight, device)
        else sample_weight
    )
    return _REVIEWED_SELECT(
        X,
        y,
        alphas=alphas,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
        cv_folds=cv_folds,
        cv_splits=cv_splits,
        random_state=random_state,
        sample_weight=effective_weight,
        fit_intercept=fit_intercept,
        device=device,
        max_iter=max_iter,
        tol=tol,
        cpu_solver=cpu_solver,
        method=method,
        cd_kkt_check_every=cd_kkt_check_every,
        gpu_cv_mixed_precision=gpu_cv_mixed_precision,
        return_details=return_details,
        cache_key=cache_key,
    )


def install_lassocv_uniform_weight_identity_contract():
    current = _lasso_impl._select_lasso_alpha_cv
    if getattr(current, _UNIFORM_WEIGHT_MARKER, False):
        return
    setattr(_select_lasso_alpha_cv, _UNIFORM_WEIGHT_MARKER, True)
    _lasso_impl._select_lasso_alpha_cv = _select_lasso_alpha_cv


__all__ = [
    "install_lassocv_uniform_weight_identity_contract",
    "_uniform_positive_weight",
]
