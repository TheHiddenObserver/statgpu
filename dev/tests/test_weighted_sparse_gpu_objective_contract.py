import numpy as np
import pytest

from statgpu.inference._results import DebiasedInferenceResult
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
import statgpu.linear_model._post_selection_ols_review_fix_contract as review_fix


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_weighted_sparse_gpu_working_problem_matches_cpu_objective(fit_intercept):
    rng = np.random.default_rng(138)
    X = rng.normal(size=(37, 5))
    y = 0.4 + X @ np.array([1.2, -0.7, 0.4, 0.0, 0.2])
    y += rng.normal(scale=0.3, size=X.shape[0])
    weights = rng.uniform(0.2, 2.4, size=X.shape[0])
    n_eff = float(np.sum(weights))

    X_work, y_work, X_mean, y_mean = (
        PenalizedLinearRegression._weighted_sparse_gpu_working_data(
            X,
            y,
            weights,
            fit_intercept=fit_intercept,
            xp=np,
            n_eff=n_eff,
        )
    )

    if fit_intercept:
        expected_X_mean = np.average(X, axis=0, weights=weights)
        expected_y_mean = float(np.average(y, weights=weights))
        X_centered = X - expected_X_mean
        y_centered = y - expected_y_mean
        np.testing.assert_allclose(X_mean, expected_X_mean, rtol=0, atol=1e-15)
        assert y_mean == pytest.approx(expected_y_mean, rel=0, abs=1e-15)
    else:
        assert X_mean is None
        assert y_mean is None
        X_centered = X
        y_centered = y

    sqrt_w = np.sqrt(weights)
    expected_X_weighted = X_centered * sqrt_w[:, None]
    expected_y_weighted = y_centered * sqrt_w
    n_samples = X.shape[0]

    np.testing.assert_allclose(
        X_work.T @ X_work / n_samples,
        expected_X_weighted.T @ expected_X_weighted / n_eff,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        X_work.T @ y_work / n_samples,
        expected_X_weighted.T @ expected_y_weighted / n_eff,
        rtol=2e-14,
        atol=2e-14,
    )
    assert float(y_work @ y_work / n_samples) == pytest.approx(
        float(expected_y_weighted @ expected_y_weighted / n_eff),
        rel=2e-14,
        abs=2e-14,
    )


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_weighted_sparse_gpu_working_problem_is_weight_scale_invariant(fit_intercept):
    rng = np.random.default_rng(139)
    X = rng.normal(size=(29, 4))
    y = rng.normal(size=29)
    weights = rng.uniform(0.1, 1.9, size=29)

    baseline = PenalizedLinearRegression._weighted_sparse_gpu_working_data(
        X,
        y,
        weights,
        fit_intercept=fit_intercept,
        xp=np,
        n_eff=float(np.sum(weights)),
    )
    scaled_weights = 17.0 * weights
    scaled = PenalizedLinearRegression._weighted_sparse_gpu_working_data(
        X,
        y,
        scaled_weights,
        fit_intercept=fit_intercept,
        xp=np,
        n_eff=float(np.sum(scaled_weights)),
    )

    np.testing.assert_allclose(scaled[0], baseline[0], rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(scaled[1], baseline[1], rtol=2e-15, atol=2e-15)
    if fit_intercept:
        np.testing.assert_allclose(scaled[2], baseline[2], rtol=2e-15, atol=2e-15)
        assert scaled[3] == pytest.approx(baseline[3], rel=2e-15, abs=2e-15)


def test_weighted_sparse_gpu_review_fix_preserves_backend_debiased_inference(monkeypatch):
    rng = np.random.default_rng(140)
    X = rng.normal(size=(41, 3))
    y = rng.normal(size=41)
    weights = rng.uniform(0.3, 2.2, size=41)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l1",
        alpha=0.05,
        fit_intercept=True,
        compute_inference=True,
        inference_method="debiased",
        solver="fista",
        device="cpu",
    )
    model._penalty = model._resolve_penalty()
    model._loss = model._resolve_loss()
    model._selected_solver = "fista"
    model._selected_backend_name = "numpy"
    model._selected_backend_device = "cpu"
    original_cache = {"contract": "preserve"}
    model._cv_cache = original_cache
    captured = {}

    def fake_base_fit(
        self,
        X_work,
        y_work,
        sample_weight=None,
        backend_name="cupy",
    ):
        assert sample_weight is None
        assert backend_name == "numpy"
        assert self._effective_intercept is False
        # This is the regression guard: the first weighted fix incorrectly
        # disabled inference here, causing the outer fit to fall through to the
        # CPU-oriented post-fit debiased helper.
        assert self._compute_inference_enabled is True
        assert not hasattr(self, "_cv_cache")
        captured["X_work"] = np.asarray(X_work).copy()
        captured["y_work"] = np.asarray(y_work).copy()

        coef = np.array([0.35, -0.15, 0.08], dtype=np.float64)
        theta = coef + np.array([0.01, -0.005, 0.002], dtype=np.float64)
        bse = np.array([0.08, 0.07, 0.09], dtype=np.float64)
        statistic = theta / bse
        pvalues = np.array([0.02, 0.04, 0.3], dtype=np.float64)
        conf_int = np.column_stack([theta - 1.96 * bse, theta + 1.96 * bse])
        self.coef_ = coef
        self.intercept_ = 0.0
        self._debiased_M_cpu = np.eye(coef.size)
        result = DebiasedInferenceResult(
            method="debiased",
            params=theta,
            bse=bse,
            statistic=statistic,
            statistic_name="z",
            pvalues=pvalues,
            conf_int=conf_int,
            distribution="normal",
            precision_method="nodewise_lasso",
            metadata={
                "backend_path": "numpy_debiased",
                "precision_cache_hit": False,
            },
        )
        result.apply_to(self)
        resid = np.asarray(y_work) - np.asarray(X_work) @ coef
        self._X_design = np.asarray(X_work).copy()
        self._y = np.asarray(y_work).copy()
        self._resid = resid
        self._nobs = int(X_work.shape[0])
        self._df_resid = int(X_work.shape[0] - X_work.shape[1])

    monkeypatch.setattr(review_fix, "_BASE_GPU_FIT", fake_base_fit)

    model._fit_gpu_backend(
        X,
        y,
        sample_weight=weights,
        backend_name="numpy",
    )

    assert model._effective_intercept is True
    assert model._compute_inference_enabled is True
    assert model._cv_cache is original_cache
    expected_X_mean = np.average(X, axis=0, weights=weights)
    expected_y_mean = float(np.average(y, weights=weights))
    expected_intercept = expected_y_mean - expected_X_mean @ model.coef_
    assert model.intercept_ == pytest.approx(expected_intercept, rel=0, abs=1e-14)
    assert model._df_resid == X.shape[0] - X.shape[1] - 1

    X_expected, y_expected, _, _ = (
        PenalizedLinearRegression._weighted_sparse_gpu_working_data(
            X,
            y,
            weights,
            fit_intercept=True,
            xp=np,
            n_eff=float(np.sum(weights)),
        )
    )
    np.testing.assert_allclose(captured["X_work"], X_expected, rtol=0, atol=0)
    np.testing.assert_allclose(captured["y_work"], y_expected, rtol=0, atol=0)

    result = model._inference_result
    assert result.method == "debiased"
    assert result.metadata["backend_path"] == "numpy_debiased_weighted"
    assert result.metadata["sample_weighted"] is True
    assert result.metadata["reporting_boundary"] == "post_numerical_inference"
    assert model._params.shape == (X.shape[1] + 1,)
    assert model._bse.shape == model._params.shape
    assert model._pvalues.shape == model._params.shape
    assert model._conf_int.shape == (model._params.size, 2)
    assert model._params[0] == pytest.approx(expected_intercept, rel=0, abs=1e-14)
    assert np.all(np.isfinite(model._bse))
