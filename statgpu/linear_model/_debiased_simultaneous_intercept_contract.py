"""Correct intercept inclusion for debiased max-|Z| simultaneous inference.

The maintained multiplier bootstrap historically included the intercept in the
reported simultaneous-CI target rows but not in the bootstrap maximum itself.
For the centered weighted/unweighted debiased paths used by PR #138, recover the
intercept influence from the same raw weighted design and bread that produce its
marginal standard error, then include that standardized influence score in the
max-|Z| statistic.

Feature-only simultaneous inference is delegated unchanged.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._utils import _get_xp
from statgpu.linear_model import (
    _post_selection_ols_fifth_review_contract as _fifth_contract,
)
from statgpu.linear_model import (
    _post_selection_ols_review_fix_contract as _weighted_contract,
)
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


_SIMULTANEOUS_MARKER = "__statgpu_pr138_simultaneous_intercept_maxz__"
_FINALIZER_MARKER = "__statgpu_pr138_simultaneous_intercept_finalizer__"
_INVALIDATION_MARKER = "__statgpu_pr138_simultaneous_intercept_invalidation__"
_ORIGINAL_SIMULTANEOUS = (
    PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap
)
_ORIGINAL_FINALIZER = _weighted_contract._finalize_weighted_debiased_result
_ORIGINAL_INVALIDATE = _fifth_contract._invalidate_failed_sparse_inference_fit


def _compute_simultaneous_ci_maxz_bootstrap(self):
    """Include the original-coordinate intercept in the max-|Z| statistic."""
    include_intercept = bool(
        getattr(
            self,
            "simultaneous_include_intercept",
            getattr(self, "_simultaneous_include_intercept", False),
        )
    )
    if not (include_intercept and bool(self._effective_intercept)):
        return _ORIGINAL_SIMULTANEOUS(self)

    if self._debiased_M_cpu is None:
        return
    if self._y is None or self._resid is None or self._bse is None:
        return
    X = self._X_design
    if X is None:
        return

    influence = getattr(self, "_debiased_intercept_influence_cpu", None)
    if influence is None:
        raise RuntimeError(
            "simultaneous_include_intercept=True requires intercept influence "
            "from the fitted debiased working problem; refusing a feature-only "
            "critical value for an intercept-inclusive interval."
        )

    X = np.asarray(X, dtype=np.float64)
    X_feat = X[:, 1:]
    n, p = X_feat.shape
    M = np.asarray(self._debiased_M_cpu, dtype=np.float64)
    if M.shape != (p, p):
        raise RuntimeError(
            "debiased precision matrix does not match the simultaneous feature design"
        )
    resid = np.asarray(self._resid, dtype=np.float64).reshape(-1)
    influence = np.asarray(influence, dtype=np.float64).reshape(-1)
    if resid.shape[0] != n or influence.shape[0] != n:
        raise RuntimeError(
            "simultaneous intercept influence/residual state does not match nobs"
        )

    bse = np.asarray(self._bse, dtype=np.float64).reshape(-1)
    params = np.asarray(self._params, dtype=np.float64).reshape(-1)
    if bse.shape[0] != p + 1 or params.shape[0] != p + 1:
        raise RuntimeError(
            "simultaneous intercept target layout does not match debiased parameters"
        )
    se_intercept = float(bse[0])
    se_feat = bse[1:]

    alpha_sim = float(
        getattr(self, "simultaneous_alpha", getattr(self, "_simultaneous_alpha", 0.05))
    )
    B = int(
        getattr(
            self,
            "simultaneous_n_bootstrap",
            getattr(self, "_simultaneous_n_bootstrap", 1000),
        )
    )
    rng = np.random.default_rng(
        getattr(
            self,
            "simultaneous_random_state",
            getattr(self, "_simultaneous_random_state", None),
        )
    )

    chunk = min(256, B)
    max_stats = np.empty(B, dtype=np.float64)
    filled = 0
    while filled < B:
        bsz = min(chunk, B - filled)
        xi = rng.standard_normal(size=(bsz, n))
        multiplier_resid = xi * resid.reshape(1, -1)

        feature_score = (multiplier_resid @ X_feat) @ M.T / float(max(n, 1))
        z_feature = feature_score / (se_feat.reshape(1, -1) + 1e-30)

        intercept_score = multiplier_resid @ influence
        z_intercept = intercept_score / (se_intercept + 1e-30)

        feature_max = np.max(np.abs(z_feature), axis=1)
        max_stats[filled : filled + bsz] = np.maximum(
            np.abs(z_intercept),
            feature_max,
        )
        filled += bsz

    critical = float(np.quantile(max_stats, 1.0 - alpha_sim))
    conf_sim = np.array(self._conf_int, copy=True, dtype=np.float64)
    conf_sim[:, 0] = params - critical * bse
    conf_sim[:, 1] = params + critical * bse

    self._conf_int_simultaneous = conf_sim
    self._simultaneous_critical_value = critical
    self._simultaneous_enabled = True
    self._simultaneous_target_mask = np.ones(params.shape[0], dtype=bool)


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
    """Prepare intercept influence before intercept-inclusive simultaneous inference."""
    needs_intercept_maxz = bool(
        simultaneous_requested
        and original_intercept
        and getattr(
            model,
            "simultaneous_include_intercept",
            getattr(model, "_simultaneous_include_intercept", False),
        )
    )
    if not needs_intercept_maxz:
        model.__dict__.pop("_debiased_intercept_influence_cpu", None)
        return _ORIGINAL_FINALIZER(
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

    # First publish marginal debiased inference without invoking the historical
    # feature-only simultaneous helper.
    result = _ORIGINAL_FINALIZER(
        model,
        X_arr=X_arr,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef_native,
        backend_name=backend_name,
        original_intercept=original_intercept,
        simultaneous_requested=False,
        sample_weighted=sample_weighted,
    )

    xp = _get_xp(backend_name)
    feature_block = X_arr * row_scale.reshape(-1, 1)
    intercept_block = row_scale.reshape(-1, 1)
    if backend_name == "torch":
        full_design = xp.cat([intercept_block, feature_block], dim=1)
    else:
        full_design = xp.concatenate([intercept_block, feature_block], axis=1)
    bread_inv = _weighted_contract._inverse_or_pinv(
        full_design.T @ full_design,
        backend_name,
    )
    influence_native = full_design @ bread_inv[:, 0]
    model._debiased_intercept_influence_cpu = np.asarray(
        _to_numpy(influence_native),
        dtype=np.float64,
    ).reshape(-1)

    model._compute_simultaneous_ci_maxz_bootstrap()
    result.simultaneous_conf_int = np.asarray(
        model._conf_int_simultaneous,
        dtype=np.float64,
    ).copy()
    result.simultaneous_method = getattr(model, "simultaneous_method", None)
    result.simultaneous_alpha = getattr(model, "simultaneous_alpha", None)
    result.simultaneous_n_bootstrap = getattr(
        model,
        "simultaneous_n_bootstrap",
        None,
    )
    result.simultaneous_critical_value = getattr(
        model,
        "_simultaneous_critical_value",
        None,
    )
    result.simultaneous_target_mask = np.asarray(
        model._simultaneous_target_mask,
        dtype=bool,
    ).copy()
    result.apply_to(model)
    return result


def _invalidate_failed_sparse_inference_fit(estimator) -> None:
    _ORIGINAL_INVALIDATE(estimator)
    estimator.__dict__.pop("_debiased_intercept_influence_cpu", None)


def install_debiased_simultaneous_intercept_contract() -> None:
    current_sim = PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap
    if not getattr(current_sim, _SIMULTANEOUS_MARKER, False):
        setattr(_compute_simultaneous_ci_maxz_bootstrap, _SIMULTANEOUS_MARKER, True)
        PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap = (
            _compute_simultaneous_ci_maxz_bootstrap
        )

    current_finalizer = _weighted_contract._finalize_weighted_debiased_result
    if not getattr(current_finalizer, _FINALIZER_MARKER, False):
        setattr(_finalize_weighted_debiased_result, _FINALIZER_MARKER, True)
        _weighted_contract._finalize_weighted_debiased_result = (
            _finalize_weighted_debiased_result
        )

    current_invalidator = _fifth_contract._invalidate_failed_sparse_inference_fit
    if not getattr(current_invalidator, _INVALIDATION_MARKER, False):
        setattr(_invalidate_failed_sparse_inference_fit, _INVALIDATION_MARKER, True)
        _fifth_contract._invalidate_failed_sparse_inference_fit = (
            _invalidate_failed_sparse_inference_fit
        )


__all__ = ["install_debiased_simultaneous_intercept_contract"]
