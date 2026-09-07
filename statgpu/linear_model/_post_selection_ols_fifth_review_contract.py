"""Late-review closure for post-selection OLS migration edge cases.

This contract keeps several cross-path public boundaries aligned with #137:

* pre-fit migration detection recognizes public ``Penalty`` objects as well as
  string penalty names, so warning normalization and AUTO native-input routing
  are identical for both constructor forms and follow the current public value
  rather than stale resolved state from a previous fit;
* a no-intercept fit with an empty active set preserves the caller's requested
  covariance/reference-distribution semantics instead of silently claiming
  ``nonrobust`` Student-t inference;
* weighted sparse-Gaussian CPU debiased inference uses the same analytic-weight
  average-loss working problem as the maintained CuPy/Torch paths, restoring
  global weight-scale invariance and CPU/GPU statistical-definition parity.

The project installs review contracts after the primary implementation modules;
patch the already-bound runtime owners so public wrappers and generic penalized
estimators observe one contract.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model import _penalized_inference_api_contract as _api_contract
from statgpu.linear_model import (
    _post_selection_ols_review_fix_contract as _weighted_review_contract,
)
from statgpu.linear_model.penalized import _post_selection_ols as _post_selection
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._fit_mixin import _validate_sample_weight_backend
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


_FIFTH_REVIEW_MARKER = "__statgpu_pr138_fifth_review_contract__"
_CPU_DEBIASED_MARKER = "__statgpu_pr138_weighted_cpu_debiased_contract__"
_SPARSE_GAUSSIAN_PENALTIES = frozenset({"l1", "elasticnet", "en"})
_ORIGINAL_POST_SELECTION = _post_selection.compute_post_selection_ols_inference
_ORIGINAL_CPU_DEBIASED = PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference


def _supports_sparse_gaussian_migration(self) -> bool:
    """Recognize sparse Gaussian scope from the current public constructor state."""
    loss_name = str(getattr(self, "loss", "squared_error")).strip().lower()
    penalty_obj = getattr(self, "penalty", None)
    if penalty_obj is None:
        penalty_obj = getattr(self, "_penalty", "")
    penalty_name = str(getattr(penalty_obj, "name", penalty_obj)).strip().lower()
    return loss_name == "squared_error" and penalty_name in _SPARSE_GAUSSIAN_PENALTIES


@functools.wraps(_ORIGINAL_POST_SELECTION)
def _compute_post_selection_ols_inference(model, X, y, sample_weight=None):
    result = _ORIGINAL_POST_SELECTION(model, X, y, sample_weight=sample_weight)
    metadata = dict(getattr(result, "metadata", {}) or {})

    # With no intercept and no selected features, every public parameter slot is
    # an inactive compatibility placeholder. There is no covariance matrix to
    # compute, but the result must not rewrite an HC/HAC request to nonrobust/t.
    if int(metadata.get("n_selected", -1)) == 0 and not bool(model._effective_intercept):
        cov_type = str(getattr(model, "_cov_type", "nonrobust")).strip().lower()
        distribution = "t" if cov_type == "nonrobust" else "normal"
        result.cov_type = cov_type
        result.distribution = distribution
        result.statistic_name = "t" if distribution == "t" else "z"
        result.df = (
            float(metadata["refit_df_resid"])
            if distribution == "t" and "refit_df_resid" in metadata
            else None
        )
        result.apply_to(model)
    return result


@functools.wraps(_ORIGINAL_CPU_DEBIASED)
def _compute_post_fit_debiased_inference(self, X, y, sample_weight=None):
    """Run weighted CPU debiased inference on the shared average-loss problem."""
    backend_name = str(getattr(self, "_selected_backend_name", "numpy")).lower()
    if (
        sample_weight is None
        or backend_name != "numpy"
        or not _supports_sparse_gaussian_migration(self)
    ):
        return _ORIGINAL_CPU_DEBIASED(
            self,
            X,
            y,
            sample_weight=sample_weight,
        )

    X_arr = np.asarray(_to_numpy(X), dtype=np.float64)
    y_arr = np.asarray(_to_numpy(y), dtype=np.float64).reshape(-1)
    sw_arr = np.asarray(_to_numpy(sample_weight), dtype=np.float64).reshape(-1)
    n_samples = int(X_arr.shape[0])
    n_eff = _validate_sample_weight_backend(sw_arr, n_samples, "numpy")
    original_intercept = bool(self._effective_intercept)
    X_work, y_work, _, _ = PenalizedLinearRegression._weighted_sparse_gpu_working_data(
        X_arr,
        y_arr,
        sw_arr,
        fit_intercept=original_intercept,
        xp=np,
        n_eff=n_eff,
    )
    row_scale = np.sqrt(sw_arr * (float(n_samples) / float(n_eff)))

    saved_use_intercept = self._use_intercept
    simultaneous_attr = hasattr(self, "enable_simultaneous_inference")
    simultaneous_requested = (
        bool(getattr(self, "enable_simultaneous_inference", False))
        if simultaneous_attr
        else False
    )
    self._use_intercept = False
    if simultaneous_attr:
        self.enable_simultaneous_inference = False
    try:
        _ORIGINAL_CPU_DEBIASED(self, X_work, y_work, sample_weight=None)
    finally:
        self._use_intercept = saved_use_intercept
        if simultaneous_attr:
            self.enable_simultaneous_inference = simultaneous_requested

    coef_native = np.asarray(self.coef_, dtype=np.float64).reshape(-1)
    _weighted_review_contract._finalize_weighted_debiased_result(
        self,
        X_arr=X_arr,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef_native,
        backend_name="numpy",
        original_intercept=original_intercept,
        simultaneous_requested=simultaneous_requested,
    )
    return None


def install_post_selection_ols_fifth_review_contract():
    """Install the late-review closure once."""
    current = _api_contract._supports_sparse_gaussian_migration
    if not getattr(current, _FIFTH_REVIEW_MARKER, False):
        setattr(_supports_sparse_gaussian_migration, _FIFTH_REVIEW_MARKER, True)
        _api_contract._supports_sparse_gaussian_migration = _supports_sparse_gaussian_migration

        setattr(_compute_post_selection_ols_inference, _FIFTH_REVIEW_MARKER, True)
        _post_selection.compute_post_selection_ols_inference = _compute_post_selection_ols_inference
        _api_contract.compute_post_selection_ols_inference = _compute_post_selection_ols_inference

    current_debiased = PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference
    if not getattr(current_debiased, _CPU_DEBIASED_MARKER, False):
        setattr(_compute_post_fit_debiased_inference, _CPU_DEBIASED_MARKER, True)
        PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference = (
            _compute_post_fit_debiased_inference
        )


__all__ = ["install_post_selection_ols_fifth_review_contract"]
