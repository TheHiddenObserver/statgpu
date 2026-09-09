import numpy as np
import pytest

from statgpu.linear_model import LassoCV


def test_lassocv_debiased_outer_inference_surface_matches_final_refit():
    rng = np.random.default_rng(20260910)
    n = 132
    X = rng.normal(size=(n, 4)) + np.array([1.7, -0.8, 0.6, 1.2])
    beta = np.array([1.0, -0.65, 0.35, 0.15])
    y = 0.9 + X @ beta + rng.normal(scale=0.5, size=n)
    weight = rng.uniform(0.4, 1.8, size=n)

    model = LassoCV(
        alphas=np.array([0.12, 0.07, 0.04]),
        cv=3,
        fit_intercept=True,
        compute_inference=True,
        inference_method="debiased",
        device="cpu",
        solver="fista",
        max_iter=5000,
        tol=1e-8,
        random_state=17,
    ).fit(X, y, sample_weight=weight)

    final = model.estimator_
    assert final is not None
    assert model._inference_result is final._inference_result
    np.testing.assert_allclose(model._params, final._params, rtol=0, atol=0)
    np.testing.assert_allclose(model._bse, final._bse, rtol=0, atol=0)
    np.testing.assert_allclose(model._tvalues, final._tvalues, rtol=0, atol=0)
    np.testing.assert_allclose(model._pvalues, final._pvalues, rtol=0, atol=0)
    np.testing.assert_allclose(model._conf_int, final._conf_int, rtol=0, atol=0)

    # Prediction ownership remains penalized even though inference reporting uses
    # the coherent debiased parameter vector.
    np.testing.assert_allclose(model.coef_, final.coef_, rtol=0, atol=0)
    assert model.intercept_ == pytest.approx(final.intercept_, rel=0, abs=0)

    n_eff = float(np.sum(weight))
    x_mean = np.sum(weight[:, None] * X, axis=0) / n_eff
    y_mean = float(np.sum(weight * y) / n_eff)
    expected_prediction_intercept = y_mean - float(x_mean @ model.coef_)
    expected_inference_intercept = y_mean - float(x_mean @ model._params[1:])
    assert model.intercept_ == pytest.approx(
        expected_prediction_intercept,
        rel=0,
        abs=2e-9,
    )
    assert model._params[0] == pytest.approx(
        expected_inference_intercept,
        rel=0,
        abs=2e-9,
    )
    assert model._inference_result.metadata["intercept_estimator"] == "centered_debiased"


def test_lassocv_reset_clears_complete_outer_inference_surface():
    rng = np.random.default_rng(20260911)
    X = rng.normal(size=(96, 3)) + np.array([1.2, -0.5, 0.8])
    y = 0.7 + X @ np.array([0.9, -0.55, 0.25]) + rng.normal(scale=0.45, size=96)

    model = LassoCV(
        alphas=np.array([0.1, 0.05]),
        cv=3,
        compute_inference=True,
        inference_method="debiased",
        device="cpu",
        max_iter=4000,
        tol=1e-8,
    ).fit(X, y)
    assert model._inference_result is not None
    assert model._params is not None

    model._reset_cv_fit_state()

    assert model.estimator_ is None
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._fitted is False
    for attr in (
        "_params",
        "_bse",
        "_pvalues",
        "_tvalues",
        "_zvalues",
        "_conf_int",
        "_inference_result",
    ):
        assert not hasattr(model, attr)
