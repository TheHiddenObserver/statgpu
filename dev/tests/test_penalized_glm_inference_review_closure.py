"""Fresh-review closure tests for penalized-GLM inference boundaries."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from statgpu.linear_model import Lasso, PenalizedGLM_CV, PenalizedPoissonRegression
from statgpu.linear_model import _penalized_glm_inference_fit_transaction as _tx
from statgpu.linear_model.penalized._inference_mixin import _PenalizedInferenceMixin


def _poisson_data(seed=14251, n=84, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.45, size=(n, p))
    beta = np.array([0.24, -0.16, 0.11])[:p]
    mu = np.exp(0.12 + X @ beta)
    y = rng.poisson(mu).astype(float)
    return X, y


def test_weighted_penalized_glm_cv_auto_uses_newton_selection_and_refit():
    X, y = _poisson_data()
    weights = np.linspace(0.45, 1.75, X.shape[0])

    cv = PenalizedGLM_CV(
        loss="poisson",
        penalty="l2",
        alpha_grid=np.array([0.08, 0.03]),
        cv=2,
        random_state=7,
        device="cpu",
        solver="auto",
        max_iter=1200,
        tol=1e-7,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y, sample_weight=weights)

    assert cv.get_params(deep=False)["solver"] == "auto"
    assert getattr(cv, "_solver", None) == "auto"
    assert cv.estimator_._selected_solver == "newton"
    assert cv.inference_method_ == "m_estimation"
    assert cv.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.penalty_selection_adjusted_ is False
    assert np.all(np.isfinite(np.asarray(cv._bse)))
    assert np.all(np.isfinite(np.asarray(cv._pvalues)))


def test_no_penalty_alias_canonicalizes_to_zero_l2_before_inference_resolution():
    X, y = _poisson_data(seed=14252)
    model = PenalizedPoissonRegression(
        penalty="none",
        alpha=3.0,
        solver="fista",
        device="cpu",
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
        max_iter=1200,
        tol=1e-7,
    ).fit(X, y)

    assert getattr(model._penalty, "name", None) == "l2"
    assert float(getattr(model._penalty, "alpha", np.nan)) == 0.0
    assert model.inference_method_ == "m_estimation"
    assert model.inference_target_ == "unpenalized_population_coefficient"


def test_residual_bootstrap_rejects_too_few_draws_and_invalidates_fit():
    rng = np.random.default_rng(14253)
    X = rng.normal(size=(48, 3))
    y = 0.3 + X @ np.array([0.7, -0.35, 0.2]) + rng.normal(scale=0.3, size=48)
    model = Lasso(
        alpha=0.04,
        device="cpu",
        inference_method="bootstrap",
        n_bootstrap=1,
        compute_inference=True,
        max_iter=600,
    )

    with pytest.raises(ValueError, match="n_bootstrap must be an integer >= 2"):
        model.fit(X, y)

    assert not getattr(model, "_fitted", False)
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._inference_result is None
    assert model._selected_solver is None
    assert model._selected_backend_name is None
    assert model._selected_backend_device is None
    assert model._feature_names is None
    assert model._design_info is None


def test_sandwich_alignment_reuses_cross_backend_and_concrete_device_helpers():
    source = inspect.getsource(_tx._align_sandwich_inputs_to_fit_backend)

    # CuPy execution must reuse the BaseEstimator DLPack-aware conversion path,
    # then pin the result to the fit-recorded concrete CUDA ordinal.
    assert "self._to_array" in source
    assert "backend=\"cupy\"" in source
    assert "_cupy_asarray_on_device" in source
    assert "device_id" in source

    # Torch execution must likewise use the DLPack-aware helper with the exact
    # recorded device instead of torch.as_tensor() on an arbitrary container.
    assert "self._to_torch" in source
    assert "device=device" in source
    assert "dtype=torch.float64" in source


def test_penalized_sandwich_device_sensitive_arrays_are_reference_bound():
    source = inspect.getsource(
        _PenalizedInferenceMixin._compute_penalized_sandwich_inference
    )

    # Physical CUDA exposed that public NumPy coef_ was recreated on Torch CPU.
    # Lock all parameter/curvature materialization to the aligned design device.
    assert "ref_arr=X_design" in source
    assert "xp_zeros(" in source
    assert "xp_full(" in source
    assert "curvature_diag(self.coef_)" in source
    assert "params = xp.concatenate([intercept_native, coef_native])" in source


def test_sandwich_reference_distributions_follow_parameter_device():
    from statgpu.inference import _sandwich as sandwich

    critical_source = inspect.getsource(sandwich._normal_critical_value)
    chi2_source = inspect.getsource(sandwich._chi2_sf)
    assert "getattr(ref_arr, \"device\", None)" in critical_source
    assert "device=device_label" in chi2_source


def test_final_execution_boundary_installer_is_idempotent():
    sandwich_before = _PenalizedInferenceMixin._compute_penalized_sandwich_inference
    bootstrap_before = _PenalizedInferenceMixin._compute_post_fit_bootstrap_inference
    lasso_fit_before = Lasso.fit
    _tx.install_penalized_glm_inference_fit_transaction()
    assert _PenalizedInferenceMixin._compute_penalized_sandwich_inference is sandwich_before
    assert _PenalizedInferenceMixin._compute_post_fit_bootstrap_inference is bootstrap_before
    assert Lasso.fit is lasso_fit_before
