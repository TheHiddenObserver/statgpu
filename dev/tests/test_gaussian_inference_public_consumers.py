"""Public-consumer regression coverage for issue #127."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import LinearRegression, PenalizedGeneralizedLinearModel, Ridge
from statgpu.linear_model.penalized._inference_mixin import _PenalizedInferenceMixin
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


def _data():
    X = np.asarray(
        [
            [-2.0, 0.5],
            [-1.0, 1.0],
            [0.0, 1.5],
            [1.0, 2.0],
            [2.0, 2.5],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    coef = np.asarray([0.75, -0.4])
    intercept = 1.2
    y = intercept + X @ coef + np.asarray([0.2, -0.1, 0.15, -0.25, 0.05, -0.05])
    return X, y, coef, intercept


def _prepare_l2_model(model, backend_name="torch"):
    X, y, coef, intercept = _data()
    model._penalty = model._resolve_penalty()
    model._selected_backend_name = backend_name
    model.coef_ = coef.copy()
    model.intercept_ = float(intercept)
    return X, y


def test_linear_regression_nonrobust_matches_statsmodels_public_surface():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(12701)
    X = rng.normal(size=(180, 4))
    y = 0.7 + X @ np.asarray([0.8, -0.3, 0.4, 0.2]) + rng.normal(scale=0.35, size=180)

    ours = LinearRegression(
        device="cpu", compute_inference=True, cov_type="nonrobust"
    ).fit(X, y)
    ref = sm.OLS(y, sm.add_constant(X)).fit()

    np.testing.assert_allclose(
        np.r_[ours.intercept_, ours.coef_], ref.params, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(ours._bse, ref.bse, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(ours._tvalues, ref.tvalues, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(ours._pvalues, ref.pvalues, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(
        ours._conf_int, ref.conf_int(alpha=0.05), rtol=1e-8, atol=1e-10
    )
    assert ours._inference_result.metadata["numerical_backend"] == "numpy"
    assert ours._inference_result.metadata["reporting_boundary"] == "post_numerical_inference"


def test_linear_regression_hc3_matches_statsmodels_normal_reference():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(12702)
    X = rng.normal(size=(220, 3))
    noise_scale = 0.15 + 0.55 * np.abs(X[:, 0])
    y = -0.2 + X @ np.asarray([0.6, -0.4, 0.25]) + rng.normal(
        scale=noise_scale, size=220
    )

    ours = LinearRegression(
        device="cpu", compute_inference=True, cov_type="hc3"
    ).fit(X, y)
    ref = sm.OLS(y, sm.add_constant(X)).fit().get_robustcov_results(
        cov_type="HC3", use_t=False
    )

    np.testing.assert_allclose(ours._bse, ref.bse, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ours._tvalues, ref.tvalues, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ours._pvalues, ref.pvalues, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(
        ours._conf_int, ref.conf_int(alpha=0.05), rtol=1e-7, atol=1e-9
    )


def test_ridge_formula_inference_matches_filtered_direct_fit():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(12703)
    n = 140
    X = rng.normal(size=(n, 3))
    y = 0.5 + X @ np.asarray([0.7, -0.5, 0.25]) + rng.normal(scale=0.25, size=n)
    w = rng.uniform(0.2, 2.5, size=n)
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    frame["y"] = y
    frame.loc[[5, 44], "x2"] = np.nan
    frame.loc[[23], "y"] = np.nan
    keep = frame[["y", "x1", "x2", "x3"]].notna().all(axis=1).to_numpy()

    formula = Ridge(
        alpha=0.08, device="cpu", compute_inference=True, cov_type="hc3"
    ).fit(formula="y ~ x1 + x2 + x3", data=frame, sample_weight=w)
    direct = Ridge(
        alpha=0.08, device="cpu", compute_inference=True, cov_type="hc3"
    ).fit(X[keep], y[keep], sample_weight=w[keep])

    np.testing.assert_allclose(formula.coef_, direct.coef_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        formula.intercept_, direct.intercept_, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(formula._bse, direct._bse, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        formula._pvalues, direct._pvalues, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        formula._conf_int, direct._conf_int, rtol=1e-8, atol=1e-10
    )


def test_public_generic_pglm_l2_uses_selected_torch_backend():
    pytest.importorskip("torch")
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        device="cpu",
        compute_inference=True,
    )
    X, y = _prepare_l2_model(model, backend_name="torch")
    model._compute_post_fit_gaussian_inference(X, y)

    result = model._inference_result
    assert result is not None
    assert result.metadata["numerical_backend"] == "torch"
    assert result.metadata["numerical_device"] == "cpu"
    assert result.metadata["reporting_boundary"] == "post_numerical_inference"
    assert isinstance(model._X_design, np.ndarray)
    assert isinstance(model._resid, np.ndarray)
    assert isinstance(model._bse, np.ndarray)


def test_typed_penalized_linear_uses_same_public_router():
    assert (
        PenalizedLinearRegression._compute_post_fit_gaussian_inference
        is PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    )


def test_non_l2_branch_still_delegates_to_shared_inference_mixin(monkeypatch):
    calls = []

    def _fake_shared(self, X, y, sample_weight=None):
        calls.append((self, X, y, sample_weight))

    monkeypatch.setattr(
        _PenalizedInferenceMixin,
        "_compute_post_fit_gaussian_inference",
        _fake_shared,
    )
    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.1,
        device="cpu",
        compute_inference=True,
    )
    model._penalty = model._resolve_penalty()
    X, y, _, _ = _data()
    model._compute_post_fit_gaussian_inference(X, y)

    assert len(calls) == 1
    assert calls[0][0] is model


def test_non_gaussian_branch_still_delegates_to_shared_inference_mixin(monkeypatch):
    calls = []

    def _fake_shared(self, X, y, sample_weight=None):
        calls.append((self, X, y, sample_weight))

    monkeypatch.setattr(
        _PenalizedInferenceMixin,
        "_compute_post_fit_gaussian_inference",
        _fake_shared,
    )
    model = PenalizedGeneralizedLinearModel(
        loss="poisson",
        penalty="l2",
        alpha=0.1,
        device="cpu",
        compute_inference=True,
    )
    model._penalty = model._resolve_penalty()
    X, y, _, _ = _data()
    model._compute_post_fit_gaussian_inference(X, np.abs(y) + 1.0)

    assert len(calls) == 1
    assert calls[0][0] is model


@pytest.mark.parametrize("backend_name", [None, "unknown-backend"])
def test_l2_inference_fails_closed_without_valid_fit_backend_provenance(backend_name):
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        device="cpu",
        compute_inference=True,
    )
    X, y, coef, intercept = _data()
    model._penalty = model._resolve_penalty()
    model._selected_backend_name = backend_name
    model.coef_ = coef.copy()
    model.intercept_ = float(intercept)

    with pytest.raises(RuntimeError, match="backend provenance|executed backend"):
        model._compute_post_fit_gaussian_inference(X, y)
    assert model._inference_result is None
    assert model._bse is None
