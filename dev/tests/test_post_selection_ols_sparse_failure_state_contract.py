import sys
import types

import numpy as np
import pytest

from statgpu.inference._results import ParameterInferenceResult
from statgpu.linear_model import (
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
)
from statgpu.linear_model import _post_selection_ols_fifth_review_contract as fifth_contract


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


def test_direct_post_selection_nonfinite_report_is_removed_before_error(monkeypatch):
    X, y = _problem(seed=8141)
    model = PenalizedLinearRegression(
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
    penalized_coef = model.coef_.copy()

    def publish_bad_result(owner, X_arg, y_arg, sample_weight=None):
        n_params = X_arg.shape[1] + 1
        params = np.zeros(n_params, dtype=np.float64)
        bse = np.ones(n_params, dtype=np.float64)
        bse[1] = np.inf
        result = ParameterInferenceResult(
            method="post_selection_ols",
            feature_names=owner._inference_feature_names(),
            params=params,
            bse=bse,
            statistic=np.zeros(n_params, dtype=np.float64),
            statistic_name="t",
            pvalues=np.full(n_params, 0.5, dtype=np.float64),
            conf_int=np.column_stack([-np.ones(n_params), np.ones(n_params)]),
            cov_type="nonrobust",
            distribution="t",
            df=float(X_arg.shape[0] - n_params),
            metadata={
                "n_selected": n_params - 1,
                "refit_df_resid": X_arg.shape[0] - n_params,
                "refit_scale": 1.0,
            },
        )
        result.apply_to(owner)
        owner._post_selection_resid = np.ones(X_arg.shape[0], dtype=np.float64)
        return result

    monkeypatch.setattr(fifth_contract, "_ORIGINAL_POST_SELECTION", publish_bad_result)
    with pytest.raises(FloatingPointError, match="non-finite"):
        fifth_contract._compute_post_selection_ols_inference(model, X, y)

    assert model._fitted is True
    np.testing.assert_allclose(model.coef_, penalized_coef, rtol=0, atol=0)
    assert model._inference_result is None
    assert model._bse is None
    assert model._pvalues is None
    assert model._conf_int is None
    assert not hasattr(model, "_post_selection_resid")


def test_post_selection_cupy_numerical_call_uses_recorded_fit_device(monkeypatch):
    events = []

    class FakeDevice:
        def __init__(self, device_id):
            self.device_id = int(device_id)

        def __enter__(self):
            events.append(("enter", self.device_id))
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", self.device_id))
            return False

    fake_cupy = types.SimpleNamespace(
        cuda=types.SimpleNamespace(Device=FakeDevice)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    model = types.SimpleNamespace(
        _selected_backend_name="cupy",
        _selected_backend_device="cuda:4",
    )
    sentinel = object()

    def delegate(owner, X_arg, y_arg, sample_weight=None):
        assert owner is model
        assert events == [("enter", 4)]
        return sentinel

    monkeypatch.setattr(fifth_contract, "_ORIGINAL_POST_SELECTION", delegate)
    result = fifth_contract._run_post_selection_on_fit_device(
        model,
        object(),
        object(),
        sample_weight=object(),
    )

    assert result is sentinel
    assert events == [("enter", 4), ("exit", 4)]
