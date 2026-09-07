"""Late-review closure for post-selection OLS migration edge cases.

This contract keeps several cross-path public boundaries aligned with #137:

* pre-fit migration detection recognizes public ``Penalty`` objects as well as
  string penalty names, so warning normalization and AUTO native-input routing
  are identical for both constructor forms and follow the current public value
  rather than stale resolved state from a previous fit;
* sparse-Gaussian inference-enabled estimators fail closed when the outer public
  finite-input guard rejects a refit before the inner fit transaction starts;
* a no-intercept fit with an empty active set preserves the caller's requested
  covariance/reference-distribution semantics instead of silently claiming
  ``nonrobust`` Student-t inference;
* sparse-Gaussian debiased inference uses one centered average-loss working
  problem whenever an intercept is fitted, with analytic weights represented by
  the same ``sqrt(w * n / sum(w))`` row scaling on NumPy/CuPy/Torch. This makes
  omitted weights, all-ones weights, and globally rescaled weights statistically
  consistent while preserving backend-native numerical inference.

The project installs review contracts after the primary implementation modules;
patch the already-bound runtime owners so public wrappers and generic penalized
estimators observe one contract.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._utils import _get_xp, xp_asarray
from statgpu.linear_model import _penalized_inference_api_contract as _api_contract
from statgpu.linear_model import (
    _post_selection_ols_review_fix_contract as _weighted_review_contract,
)
from statgpu.linear_model.penalized import _post_selection_ols as _post_selection
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._fit_mixin import _validate_sample_weight_backend
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
from statgpu.linear_model.penalized._no_inference_cleanup_contract import (
    _invalidate_failed_no_inference_fit,
)


_FIFTH_REVIEW_MARKER = "__statgpu_pr138_fifth_review_contract__"
_CPU_DEBIASED_MARKER = "__statgpu_pr138_centered_cpu_debiased_contract__"
_GPU_DEBIASED_MARKER = "__statgpu_pr138_centered_gpu_debiased_contract__"
_SPARSE_RESET_MARKER = "__statgpu_pr138_sparse_inference_finite_reset__"
_SPARSE_GAUSSIAN_PENALTIES = frozenset({"l1", "elasticnet", "en"})
_ORIGINAL_POST_SELECTION = _post_selection.compute_post_selection_ols_inference
_ORIGINAL_CPU_DEBIASED = PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference
_ORIGINAL_CUPY_DEBIASED = PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu
_ORIGINAL_TORCH_DEBIASED = PenalizedGeneralizedLinearModel._compute_inference_debiased_torch


def _supports_sparse_gaussian_migration(self) -> bool:
    """Recognize sparse Gaussian scope from the current public constructor state."""
    loss_name = str(getattr(self, "loss", "squared_error")).strip().lower()
    penalty_obj = getattr(self, "penalty", None)
    if penalty_obj is None:
        penalty_obj = getattr(self, "_penalty", "")
    penalty_name = str(getattr(penalty_obj, "name", penalty_obj)).strip().lower()
    return loss_name == "squared_error" and penalty_name in _SPARSE_GAUSSIAN_PENALTIES


def _invalidate_failed_sparse_inference_fit(estimator) -> None:
    """Clear every result-bearing sparse-inference state after rejected refit."""
    _invalidate_failed_no_inference_fit(estimator)
    estimator._penalty = None
    estimator._loss = None
    estimator._lla_n_iters_ = 0
    estimator._init_coef = None
    if hasattr(estimator, "_init_intercept"):
        estimator._init_intercept = None
    estimator._conf_int_simultaneous = None
    estimator._simultaneous_enabled = False
    estimator._debiased_M_cpu = None
    for name in (
        "_simultaneous_critical_value",
        "_simultaneous_target_mask",
        "_simultaneous_method",
        "_simultaneous_alpha",
        "_simultaneous_n_bootstrap",
    ):
        estimator.__dict__.pop(name, None)


def _install_sparse_inference_public_validation_reset() -> None:
    """Extend the existing finite-validation reset to sparse Gaussian inference."""
    current_reset = getattr(PenalizedGeneralizedLinearModel, "_reset_fit_state", None)
    if getattr(current_reset, _SPARSE_RESET_MARKER, False):
        return

    def _reset_fit_state(self):
        sparse_inference = (
            bool(getattr(self, "compute_inference", False))
            and _supports_sparse_gaussian_migration(self)
        )
        # Preserve the already-installed allocator/device cleanup and unrelated
        # L2/no-inference reset semantics exactly once.
        result = current_reset(self) if callable(current_reset) else None
        if sparse_inference:
            _invalidate_failed_sparse_inference_fit(self)
        return result

    if callable(current_reset):
        functools.update_wrapper(_reset_fit_state, current_reset)
    setattr(_reset_fit_state, _SPARSE_RESET_MARKER, True)
    _reset_fit_state._statgpu_original = current_reset
    PenalizedGeneralizedLinearModel._reset_fit_state = _reset_fit_state


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


def _debiased_working_data_numpy(self, X, y, sample_weight):
    X_arr = np.asarray(_to_numpy(X), dtype=np.float64)
    y_arr = np.asarray(_to_numpy(y), dtype=np.float64).reshape(-1)
    n_samples = int(X_arr.shape[0])
    sample_weighted = sample_weight is not None
    if sample_weighted:
        sw_arr = np.asarray(_to_numpy(sample_weight), dtype=np.float64).reshape(-1)
        n_eff = _validate_sample_weight_backend(sw_arr, n_samples, "numpy")
    else:
        sw_arr = np.ones(n_samples, dtype=np.float64)
        n_eff = float(n_samples)

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
    return (
        X_arr,
        X_work,
        y_work,
        row_scale,
        original_intercept,
        sample_weighted,
    )


@functools.wraps(_ORIGINAL_CPU_DEBIASED)
def _compute_post_fit_debiased_inference(self, X, y, sample_weight=None):
    """Run CPU debiased inference on the shared centered average-loss problem."""
    backend_name = str(getattr(self, "_selected_backend_name", "numpy")).lower()
    if backend_name != "numpy" or not _supports_sparse_gaussian_migration(self):
        return _ORIGINAL_CPU_DEBIASED(
            self,
            X,
            y,
            sample_weight=sample_weight,
        )

    (
        X_arr,
        X_work,
        y_work,
        row_scale,
        original_intercept,
        sample_weighted,
    ) = _debiased_working_data_numpy(self, X, y, sample_weight)

    # With neither intercept nor weights, the maintained CPU implementation is
    # already exactly the required working problem.
    if not original_intercept and not sample_weighted:
        return _ORIGINAL_CPU_DEBIASED(self, X, y, sample_weight=None)

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
        sample_weighted=sample_weighted,
    )
    return None


def _compute_centered_gpu_debiased(
    self,
    X,
    y,
    coef,
    *,
    backend_name: str,
    original,
):
    """Center an unweighted intercept design before backend-native debiasing."""
    if not _supports_sparse_gaussian_migration(self) or not bool(self._effective_intercept):
        return original(self, X, y, coef)

    xp = _get_xp(backend_name)
    X_arr = xp_asarray(X, dtype=np.float64, xp=xp, ref_arr=X)
    y_arr = xp_asarray(y, dtype=np.float64, xp=xp, ref_arr=X).reshape(-1)
    coef_arr = xp_asarray(coef, dtype=X_arr.dtype, xp=xp, ref_arr=X_arr).reshape(-1)
    n_samples = int(X_arr.shape[0])
    X_work = X_arr - xp.mean(X_arr, axis=0)
    y_work = y_arr - xp.mean(y_arr)
    row_scale = xp_asarray(
        np.ones(n_samples, dtype=np.float64),
        dtype=X_arr.dtype,
        xp=xp,
        ref_arr=X_arr,
    ).reshape(-1)

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
        original(self, X_work, y_work, coef_arr)
    finally:
        self._use_intercept = saved_use_intercept
        if simultaneous_attr:
            self.enable_simultaneous_inference = simultaneous_requested

    _weighted_review_contract._finalize_weighted_debiased_result(
        self,
        X_arr=X_arr,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef_arr,
        backend_name=backend_name,
        original_intercept=True,
        simultaneous_requested=simultaneous_requested,
        sample_weighted=False,
    )
    return None


@functools.wraps(_ORIGINAL_CUPY_DEBIASED)
def _compute_inference_debiased_gpu(self, X_gpu, y_gpu, coef_gpu):
    return _compute_centered_gpu_debiased(
        self,
        X_gpu,
        y_gpu,
        coef_gpu,
        backend_name="cupy",
        original=_ORIGINAL_CUPY_DEBIASED,
    )


@functools.wraps(_ORIGINAL_TORCH_DEBIASED)
def _compute_inference_debiased_torch(self, X_torch, y_torch, coef_torch):
    return _compute_centered_gpu_debiased(
        self,
        X_torch,
        y_torch,
        coef_torch,
        backend_name="torch",
        original=_ORIGINAL_TORCH_DEBIASED,
    )


def install_post_selection_ols_fifth_review_contract():
    """Install the late-review closure once."""
    current = _api_contract._supports_sparse_gaussian_migration
    if not getattr(current, _FIFTH_REVIEW_MARKER, False):
        setattr(_supports_sparse_gaussian_migration, _FIFTH_REVIEW_MARKER, True)
        _api_contract._supports_sparse_gaussian_migration = _supports_sparse_gaussian_migration

        setattr(_compute_post_selection_ols_inference, _FIFTH_REVIEW_MARKER, True)
        _post_selection.compute_post_selection_ols_inference = _compute_post_selection_ols_inference
        _api_contract.compute_post_selection_ols_inference = _compute_post_selection_ols_inference

    _install_sparse_inference_public_validation_reset()

    current_cpu = PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference
    if not getattr(current_cpu, _CPU_DEBIASED_MARKER, False):
        setattr(_compute_post_fit_debiased_inference, _CPU_DEBIASED_MARKER, True)
        PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference = (
            _compute_post_fit_debiased_inference
        )

    current_gpu = PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu
    if not getattr(current_gpu, _GPU_DEBIASED_MARKER, False):
        setattr(_compute_inference_debiased_gpu, _GPU_DEBIASED_MARKER, True)
        setattr(_compute_inference_debiased_torch, _GPU_DEBIASED_MARKER, True)
        PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu = (
            _compute_inference_debiased_gpu
        )
        PenalizedGeneralizedLinearModel._compute_inference_debiased_torch = (
            _compute_inference_debiased_torch
        )


__all__ = ["install_post_selection_ols_fifth_review_contract"]
