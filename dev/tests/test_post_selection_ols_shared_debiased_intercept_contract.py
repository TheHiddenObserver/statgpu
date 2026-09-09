import numpy as np
import pytest

from statgpu.linear_model import ElasticNet, Lasso, PenalizedGeneralizedLinearModel


def _problem(seed=20260913):
    rng = np.random.default_rng(seed)
    n = 132
    X = rng.normal(size=(n, 4)) + np.array([1.7, -0.9, 0.55, 1.25])
    beta = np.array([1.0, -0.65, 0.4, 0.18])
    y = 0.95 + X @ beta + rng.normal(scale=0.45, size=n)
    weights = rng.uniform(0.3, 2.0, size=n)
    return X, y, weights


def _make_estimator(kind):
    common = dict(
        alpha=0.11,
        fit_intercept=True,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        device="cpu",
        max_iter=6000,
        tol=1e-9,
    )
    if kind == "lasso":
        return Lasso(**common)
    if kind == "elasticnet":
        return ElasticNet(l1_ratio=0.65, **common)
    if kind == "generic_l1":
        return PenalizedGeneralizedLinearModel(
            loss="squared_error",
            penalty="l1",
            **common,
        )
    raise AssertionError(f"unknown fixture kind {kind!r}")


@pytest.mark.parametrize("kind", ["lasso", "elasticnet", "generic_l1"])
def test_shared_debiased_intercept_obeys_feature_translation(kind):
    X, y, weights = _problem()
    shift = np.array([0.8, -1.15, 0.35, 1.45])

    original = _make_estimator(kind).fit(X, y, sample_weight=weights)
    translated = _make_estimator(kind).fit(X + shift, y, sample_weight=weights)

    # Prediction remains owned by the penalized fit.
    np.testing.assert_allclose(
        translated.coef_,
        original.coef_,
        rtol=0,
        atol=3e-8,
    )
    assert translated.intercept_ == pytest.approx(
        original.intercept_ - float(shift @ original.coef_),
        rel=0,
        abs=3e-8,
    )

    # Inference reporting is parameterized by the debiased slope vector.
    np.testing.assert_allclose(
        translated._params[1:],
        original._params[1:],
        rtol=0,
        atol=3e-8,
    )
    assert translated._params[0] == pytest.approx(
        original._params[0] - float(shift @ original._params[1:]),
        rel=0,
        abs=4e-8,
    )

    for model in (original, translated):
        result = model._inference_result
        assert result is not None
        assert result.method == "debiased"
        assert result.metadata["intercept_estimator"] == "centered_debiased"
        assert result.metadata["intercept_influence"] == "centered_nodewise"
        np.testing.assert_allclose(result.params, model._params, rtol=0, atol=0)
