import types

import numpy as np
import pytest

from statgpu.linear_model import (
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
)


def _problem(seed=8138):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 5))
    y = 0.25 + X @ np.array([1.0, -0.7, 0.45, 0.0, 0.0]) + rng.normal(
        scale=0.35,
        size=X.shape[0],
    )
    return X, y


def _assert_sparse_fit_state_cleared(model):
    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.n_iter_ == 0
    assert model._penalty is None
    assert model._loss is None
    assert model._params is None
    assert model._bse is None
    assert model._tvalues is None
    assert model._zvalues is None
    assert model._pvalues is None
    assert model._conf_int is None
    assert model._inference_result is None
    assert model._selected_solver is None
    assert model._selected_backend_name is None
    assert model._selected_backend_device is None
    assert model._conf_int_simultaneous is None
    assert model._simultaneous_enabled is False
    assert model._debiased_M_cpu is None
    assert not hasattr(model, "n_features_in_")
    for name in (
        "_post_selection_X_design",
        "_post_selection_y",
        "_post_selection_resid",
        "_post_selection_scale",
        "_post_selection_df_resid",
        "_post_selection_nobs",
    ):
        assert not hasattr(model, name)


@pytest.mark.parametrize("inference_method", ["post_selection_ols", "debiased"])
def test_sparse_inference_finite_validation_failure_invalidates_prior_fit(inference_method):
    X, y = _problem()
    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.04,
        fit_intercept=True,
        solver="fista",
        inference_method=inference_method,
        compute_inference=True,
        device="cpu",
        max_iter=4000,
        tol=1e-9,
    ).fit(X, y)

    assert model._fitted is True
    assert model.coef_ is not None
    assert model._inference_result is not None
    assert model._selected_backend_name == "numpy"
    if inference_method == "post_selection_ols":
        assert hasattr(model, "_post_selection_resid")
    else:
        assert model._debiased_M_cpu is not None

    X_bad = X.copy()
    X_bad[7, 2] = np.nan
    with pytest.raises((ValueError, FloatingPointError)):
        model.fit(X_bad, y)

    _assert_sparse_fit_state_cleared(model)


@pytest.mark.parametrize("inference_method", ["post_selection_ols", "debiased"])
def test_typed_sparse_postfit_inference_failure_invalidates_current_fit(inference_method):
    X, y = _problem(seed=8139)
    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.04,
        fit_intercept=True,
        solver="fista",
        inference_method=inference_method,
        compute_inference=True,
        device="cpu",
        max_iter=4000,
        tol=1e-9,
    ).fit(X, y)
    assert model._fitted is True

    def fail_postfit(self, X_arg, y_arg, sample_weight=None):
        assert self.coef_ is not None
        raise RuntimeError("synthetic post-fit inference failure")

    model._compute_post_fit_gaussian_inference = types.MethodType(fail_postfit, model)
    with pytest.raises(RuntimeError, match="synthetic post-fit inference failure"):
        model.fit(X, y)

    _assert_sparse_fit_state_cleared(model)


def test_generic_sparse_postfit_inference_failure_invalidates_current_fit():
    X, y = _problem(seed=8140)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l1",
        alpha=0.04,
        fit_intercept=True,
        solver="fista",
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=4000,
        tol=1e-9,
    ).fit(X, y)
    assert model._fitted is True

    def fail_postfit(self, X_arg, y_arg, sample_weight=None):
        assert self.coef_ is not None
        raise RuntimeError("synthetic generic inference failure")

    model._compute_post_fit_gaussian_inference = types.MethodType(fail_postfit, model)
    with pytest.raises(RuntimeError, match="synthetic generic inference failure"):
        model.fit(X, y)

    _assert_sparse_fit_state_cleared(model)
