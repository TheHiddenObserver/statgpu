import numpy as np
import pytest

from statgpu.inference._results import DebiasedInferenceResult
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
import statgpu.linear_model._debiased_intercept_parameterization_contract as intercept_contract
import statgpu.linear_model._post_selection_ols_review_fix_contract as review_fix


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_weighted_sparse_gpu_review_fix_preserves_backend_debiased_inference(
    monkeypatch,
    backend_name,
):
    torch = None
    if backend_name == "torch":
        torch = pytest.importorskip("torch")

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
    model._selected_backend_name = backend_name
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
        assert backend_name == model._selected_backend_name
        assert self._effective_intercept is False
        assert self._compute_inference_enabled is True
        assert not hasattr(self, "_cv_cache")
        captured["X_work"] = _as_numpy(X_work).copy()
        captured["y_work"] = _as_numpy(y_work).copy()

        coef = np.array([0.35, -0.15, 0.08], dtype=np.float64)
        theta = coef + np.array([0.01, -0.005, 0.002], dtype=np.float64)
        captured["theta"] = theta.copy()
        bse = np.array([0.08, 0.07, 0.09], dtype=np.float64)
        statistic = theta / bse
        pvalues = np.array([0.02, 0.04, 0.3], dtype=np.float64)
        conf_int = np.column_stack([theta - 1.96 * bse, theta + 1.96 * bse])
        self.coef_ = coef
        self.intercept_ = 0.0
        self._debiased_M_cpu = np.eye(coef.size)
        if backend_name == "torch":
            self.__dict__[intercept_contract._NATIVE_M] = torch.eye(
                coef.size,
                dtype=torch.float64,
            )
            self.__dict__[intercept_contract._NATIVE_THETA] = torch.as_tensor(
                theta,
                dtype=torch.float64,
            )
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
                "backend_path": f"{backend_name}_debiased",
                "precision_cache_hit": False,
            },
        )
        result.apply_to(self)
        X_np = _as_numpy(X_work)
        y_np = _as_numpy(y_work)
        resid = y_np - X_np @ coef
        self._X_design = X_np.copy()
        self._y = y_np.copy()
        self._resid = resid
        self._nobs = int(X_np.shape[0])
        self._df_resid = int(X_np.shape[0] - X_np.shape[1])

    monkeypatch.setattr(review_fix, "_BASE_GPU_FIT", fake_base_fit)

    if backend_name == "torch":
        X_input = torch.as_tensor(X, dtype=torch.float64)
        y_input = torch.as_tensor(y, dtype=torch.float64)
        weights_input = torch.as_tensor(weights, dtype=torch.float64)
    else:
        X_input = X
        y_input = y
        weights_input = weights

    model._fit_gpu_backend(
        X_input,
        y_input,
        sample_weight=weights_input,
        backend_name=backend_name,
    )

    assert model._effective_intercept is True
    assert model._compute_inference_enabled is True
    assert model._cv_cache is original_cache
    expected_X_mean = np.average(X, axis=0, weights=weights)
    expected_y_mean = float(np.average(y, weights=weights))
    expected_prediction_intercept = expected_y_mean - expected_X_mean @ model.coef_
    expected_inference_intercept = expected_y_mean - expected_X_mean @ captured["theta"]
    assert model.intercept_ == pytest.approx(
        expected_prediction_intercept,
        rel=0,
        abs=1e-14,
    )
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
    np.testing.assert_allclose(captured["X_work"], X_expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(captured["y_work"], y_expected, rtol=1e-14, atol=1e-14)

    result = model._inference_result
    assert result.method == "debiased"
    assert result.metadata["backend_path"] == f"{backend_name}_debiased_weighted"
    assert result.metadata["sample_weighted"] is True
    assert result.metadata["numerical_backend"] == backend_name
    assert result.metadata["reporting_boundary"] == "post_numerical_inference"
    assert result.metadata["intercept_estimator"] == "centered_debiased"
    assert result.metadata["intercept_influence"] == "centered_nodewise"
    assert model._params.shape == (X.shape[1] + 1,)
    assert model._bse.shape == model._params.shape
    assert model._pvalues.shape == model._params.shape
    assert model._conf_int.shape == (model._params.size, 2)
    assert model._params[0] == pytest.approx(
        expected_inference_intercept,
        rel=0,
        abs=1e-14,
    )
    assert not hasattr(model, "_debiased_intercept_influence_cpu")
    assert np.all(np.isfinite(model._bse))


@pytest.mark.parametrize(
    "loss_name,solver_name",
    [("logistic", "fista"), ("poisson", "fista_bb")],
)
@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
def test_weighted_sparse_review_fix_does_not_hijack_non_gaussian_losses(
    monkeypatch,
    loss_name,
    solver_name,
    backend_name,
):
    rng = np.random.default_rng(141)
    X = rng.normal(size=(32, 3))
    if loss_name == "logistic":
        y = (rng.random(32) > 0.4).astype(float)
    else:
        y = rng.poisson(np.exp(0.1 + 0.1 * X[:, 0])).astype(float)
    weights = rng.uniform(0.2, 1.8, size=32)
    model = PenalizedGeneralizedLinearModel(
        loss=loss_name,
        penalty="l1",
        alpha=0.02,
        fit_intercept=True,
        compute_inference=False,
        solver=solver_name,
        device="cpu",
    )
    model._penalty = model._resolve_penalty()
    model._loss = model._resolve_loss()
    model._selected_solver = solver_name
    captured = {}

    def fail_gaussian_transform(*args, **kwargs):
        raise AssertionError("non-Gaussian weighted fit entered Gaussian transform")

    def fake_base_fit(
        self,
        X_arg,
        y_arg,
        sample_weight=None,
        backend_name="cupy",
    ):
        captured["X"] = X_arg
        captured["y"] = y_arg
        captured["sample_weight"] = sample_weight
        captured["backend_name"] = backend_name

    monkeypatch.setattr(
        PenalizedLinearRegression,
        "_weighted_sparse_gpu_working_data",
        staticmethod(fail_gaussian_transform),
    )
    monkeypatch.setattr(review_fix, "_BASE_GPU_FIT", fake_base_fit)

    model._fit_gpu_backend(
        X,
        y,
        sample_weight=weights,
        backend_name=backend_name,
    )

    assert captured["X"] is X
    assert captured["y"] is y
    assert captured["sample_weight"] is weights
    assert captured["backend_name"] == backend_name
