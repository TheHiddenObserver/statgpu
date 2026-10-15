"""Contract tests for Issue #145 backend-native Gaussian residual bootstrap."""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest

from statgpu.inference._results import ParameterInferenceResult
from statgpu.linear_model import (
    ElasticNet,
    Lasso,
    PenalizedGLM_CV,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
)
from statgpu.linear_model import (
    _gaussian_residual_bootstrap_backend_contract as _bootstrap,
)


def _data(seed=145, n=56, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([0.8, -0.45, 0.25])[:p]
    y = 0.35 + X @ beta + rng.normal(scale=0.25, size=n)
    return X, y


def _cpu_model(
    *,
    penalty="l1",
    alpha=0.06,
    l1_ratio=0.5,
    seed=17,
    B=8,
    cov_type="nonrobust",
    **kwargs,
):
    model = PenalizedLinearRegression(
        penalty=penalty,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=kwargs.pop("fit_intercept", True),
        device="cpu",
        solver=kwargs.pop("solver", "fista"),
        max_iter=kwargs.pop("max_iter", 1200),
        tol=kwargs.pop("tol", 1e-8),
        compute_inference=True,
        inference_method="bootstrap",
        cov_type=cov_type,
        **kwargs,
    )
    model.n_bootstrap = B
    model.bootstrap_random_state = seed
    return model


def test_resample_schedule_is_backend_neutral_reproducible_and_hashed():
    left = _bootstrap._draw_resample_indices(11, 7, 1234)
    right = _bootstrap._draw_resample_indices(11, 7, 1234)
    other = _bootstrap._draw_resample_indices(11, 7, 1235)
    assert left.dtype == np.int64
    assert left.shape == (7, 11)
    np.testing.assert_array_equal(left, right)
    assert not np.array_equal(left, other)
    assert _bootstrap._schedule_sha256(left) == _bootstrap._schedule_sha256(right)
    assert _bootstrap._schedule_sha256(left) != _bootstrap._schedule_sha256(other)
    assert len(_bootstrap._schedule_sha256(left)) == 64


def test_cpu_bootstrap_preserves_contract_diagnostics_and_is_reproducible():
    X, y = _data()
    first = _cpu_model(seed=77, B=10).fit(X, y)
    second = _cpu_model(seed=77, B=10).fit(X, y)

    assert first.inference_requested_method_ == "bootstrap"
    assert first.inference_resolved_method_ == "residual_bootstrap"
    assert first.inference_method_ == "residual_bootstrap"
    assert first.inference_target_ == "penalized_coefficient_distribution"
    assert first.penalty_conditioning_ == "fixed_penalty"
    assert first._inference_result.method == "residual_bootstrap"
    metadata = first._inference_result.metadata
    assert metadata["resampling_scope"] == "unweighted_gaussian_residual"
    assert metadata["resampling_schedule"] == "numpy_generator_control_plane"
    assert len(metadata["resampling_schedule_sha256"]) == 64
    assert metadata["refit_penalty"] == "l1"
    assert metadata["numerical_backend"] == "numpy"
    assert metadata["numerical_device"] == "cpu"
    assert metadata["reporting_backend"] == "numpy"
    assert metadata["reporting_boundary"] == "post_numerical_inference"
    assert metadata["n_bootstrap"] == 10
    assert metadata["random_state"] == 77
    assert first._X_design.shape == (X.shape[0], X.shape[1] + 1)
    assert first._y.shape == (X.shape[0],)
    assert first._resid.shape == (X.shape[0],)
    assert np.isfinite(first.rsquared)
    np.testing.assert_allclose(first._bse, second._bse, rtol=0, atol=0)
    np.testing.assert_allclose(first._pvalues, second._pvalues, rtol=0, atol=0)
    np.testing.assert_allclose(first._conf_int, second._conf_int, rtol=0, atol=0)
    assert (
        first._inference_result.metadata["resampling_schedule_sha256"]
        == second._inference_result.metadata["resampling_schedule_sha256"]
    )


def test_child_factory_preserves_solver_controls_and_disables_inference():
    owner = PenalizedLinearRegression(
        penalty="elasticnet",
        alpha=0.05,
        l1_ratio=0.35,
        fit_intercept=False,
        device="cpu",
        solver="fista",
        lipschitz_L=2.75,
        max_iter=321,
        tol=2e-7,
        stopping="objective",
        compute_inference=True,
        inference_method="bootstrap",
    )
    child = _bootstrap._make_child_refit(owner, backend="numpy")
    assert str(getattr(child.penalty, "name", child.penalty)).lower() in (
        "elasticnet",
        "en",
    )
    assert child.alpha == pytest.approx(0.05)
    assert child.l1_ratio == pytest.approx(0.35)
    assert child.fit_intercept is False
    assert child.solver == "fista"
    assert child.lipschitz_L == pytest.approx(2.75)
    assert child.max_iter == 321
    assert child.tol == pytest.approx(2e-7)
    assert child.stopping == "objective"
    assert child.compute_inference is False


def test_cpu_elasticnet_bootstrap_preserves_penalty_family():
    X, y = _data(seed=146)
    model = _cpu_model(
        penalty="elasticnet", alpha=0.05, l1_ratio=0.35, seed=5, B=6
    ).fit(X, y)
    assert model._inference_result.metadata["refit_penalty"] in (
        "elasticnet",
        "en",
    )
    assert np.all(np.isfinite(model._bse))
    assert np.all(np.isfinite(model._conf_int))


@pytest.mark.parametrize("penalty", ["scad", "mcp"])
def test_cpu_nonconvex_bootstrap_preserves_claimed_penalty_rows(penalty):
    X, y = _data(seed=1461, n=44, p=2)
    model = _cpu_model(
        penalty=penalty,
        alpha=0.03,
        seed=9,
        B=3,
        max_iter=160,
        tol=1e-6,
        max_lla_iters=3,
        lla_tol=1e-6,
    ).fit(X, y)
    assert model.inference_resolved_method_ == "residual_bootstrap"
    assert model._inference_result.metadata["refit_penalty"] == penalty
    assert np.all(np.isfinite(model._bse))
    assert np.all(np.isfinite(model._conf_int))


def test_weighted_bootstrap_remains_fail_closed():
    X, y = _data(seed=147)
    model = _cpu_model(B=4)
    weights = np.linspace(0.5, 1.5, X.shape[0])
    with pytest.raises(NotImplementedError, match="Weighted Gaussian residual-bootstrap"):
        model.fit(X, y, sample_weight=weights)
    assert not getattr(model, "_fitted", False)
    assert model._inference_result is None


@pytest.mark.parametrize("cov_type", ["hc0", "hac"])
def test_robust_or_hac_bootstrap_semantics_remain_fail_closed(cov_type):
    X, y = _data(seed=1471)
    model = _cpu_model(B=3, cov_type=cov_type)
    with pytest.raises(NotImplementedError, match="nonrobust|robust/HAC"):
        model.fit(X, y)
    assert not getattr(model, "_fitted", False)
    assert model._inference_result is None


def test_non_gaussian_bootstrap_remains_fail_closed():
    X, y_cont = _data(seed=1472)
    y = (y_cont > np.median(y_cont)).astype(float)
    model = PenalizedLogisticRegression(
        penalty="l1",
        alpha=0.04,
        device="cpu",
        solver="fista",
        max_iter=250,
        compute_inference=True,
        inference_method="bootstrap",
    )
    with pytest.raises(NotImplementedError, match="Gaussian residual bootstrap only"):
        model.fit(X, y)
    assert not getattr(model, "_fitted", False)
    assert model._inference_result is None


def test_cox_cv_inference_remains_fail_closed():
    X, _ = _data(seed=1473, n=20, p=2)
    y = np.column_stack(
        [np.linspace(1.0, 20.0, X.shape[0]), np.tile([0.0, 1.0], X.shape[0] // 2)]
    )
    cv = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l1",
        alpha_grid=np.array([0.1, 0.05]),
        cv=2,
        device="cpu",
        compute_inference=True,
        inference_method="bootstrap",
    )
    with pytest.raises(NotImplementedError, match="estimation-only"):
        cv.fit(X, y)


def test_too_few_draws_remains_fail_closed():
    X, y = _data(seed=148)
    model = _cpu_model(B=1)
    with pytest.raises(ValueError, match="n_bootstrap must be an integer >= 2"):
        model.fit(X, y)
    assert not getattr(model, "_fitted", False)
    assert model.coef_ is None
    assert model._inference_result is None


def test_torch_backend_consumes_native_bootstrap_responses(monkeypatch):
    torch = pytest.importorskip("torch")
    X_np, y_np = _data(seed=149, n=24, p=2)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    owner = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.04,
        fit_intercept=True,
        device="cpu",
        solver="fista",
        compute_inference=False,
    )
    owner._selected_backend_name = "torch"
    owner._selected_backend_device = "cpu"
    owner.coef_ = np.array([0.4, -0.2], dtype=np.float64)
    owner.intercept_ = 0.1
    owner._params = np.array([0.1, 0.4, -0.2], dtype=np.float64)
    owner.n_bootstrap = 4
    owner.bootstrap_random_state = 22
    owner.cov_type = "nonrobust"
    seen = []

    class _Child:
        def fit(self, X_value, y_value):
            assert isinstance(X_value, torch.Tensor)
            assert isinstance(y_value, torch.Tensor)
            assert X_value.device.type == "cpu"
            assert y_value.device.type == "cpu"
            seen.append(y_value.detach().clone())
            self._selected_backend_name = "torch"
            self._selected_backend_device = "cpu"
            self._selected_solver = "fista"
            y_mean = float(torch.mean(y_value))
            self._params = np.array([y_mean, 0.4, -0.2], dtype=np.float64)
            return self

    monkeypatch.setattr(
        _bootstrap, "_make_child_refit", lambda *_args, **_kwargs: _Child()
    )
    _bootstrap._backend_native_gaussian_residual_bootstrap(owner, X, y)
    assert len(seen) == 4
    assert owner._inference_result.metadata["numerical_backend"] == "torch"
    assert owner._inference_result.metadata["numerical_device"] == "cpu"
    assert owner._inference_result.metadata["child_selected_solvers"] == ["fista"]
    assert owner._X_design.shape == (24, 3)
    assert np.all(np.isfinite(owner._bse))


def test_exact_cupy_device_context_is_entered_before_child_fit(monkeypatch):
    entered = []

    class _FakeDevice:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            entered.append(self.index)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=_FakeDevice))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    with _bootstrap._child_device_context("cupy", "cuda:3"):
        entered.append("body")
    assert entered == [3, "body"]


def test_exact_torch_device_context_is_entered_before_child_fit(monkeypatch):
    torch = pytest.importorskip("torch")
    entered = []

    @contextlib.contextmanager
    def fake_device(target):
        entered.append(str(target))
        yield

    monkeypatch.setattr(torch.cuda, "device", fake_device)
    with _bootstrap._child_device_context("torch", "cuda:2"):
        entered.append("body")
    assert entered == ["cuda:2", "body"]


def test_backend_child_provenance_mismatch_fails_closed():
    class _Child:
        _selected_backend_name = "numpy"
        _selected_backend_device = "cpu"

    with pytest.raises(RuntimeError, match="changed execution provenance"):
        _bootstrap._assert_child_provenance(
            _Child(), backend="cupy", device="cuda:0"
        )


def test_child_refit_failure_invalidates_outer_fit(monkeypatch):
    X, y = _data(seed=150)

    class _FailingChild:
        def fit(self, *_args, **_kwargs):
            raise RuntimeError("synthetic bootstrap child failure")

    monkeypatch.setattr(
        _bootstrap, "_make_child_refit", lambda *_args, **_kwargs: _FailingChild()
    )
    model = _cpu_model(B=3)
    with pytest.raises(RuntimeError, match="synthetic bootstrap child failure"):
        model.fit(X, y)
    assert not model._fitted
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._inference_result is None
    assert model.inference_method_ is None


def test_failed_bootstrap_refit_does_not_leak_prior_success(monkeypatch):
    X, y = _data(seed=151)
    model = _cpu_model(B=3).fit(X, y)
    assert model._fitted and model._inference_result is not None

    class _FailingChild:
        def fit(self, *_args, **_kwargs):
            raise RuntimeError("synthetic second bootstrap failure")

    monkeypatch.setattr(
        _bootstrap, "_make_child_refit", lambda *_args, **_kwargs: _FailingChild()
    )
    with pytest.raises(RuntimeError, match="synthetic second bootstrap failure"):
        model.fit(X, y)
    assert not model._fitted
    assert model.coef_ is None
    assert model._inference_result is None
    assert model.inference_method_ is None


def test_cv_bootstrap_runs_only_on_selected_final_refit(monkeypatch):
    X, y = _data(seed=152, n=42, p=2)
    bootstrap_calls = []
    child_calls = []
    original_bootstrap = _bootstrap._backend_native_gaussian_residual_bootstrap

    class _FastChild:
        def __init__(self, owner):
            self.owner = owner

        def fit(self, _X, _y):
            child_calls.append(1)
            self._selected_backend_name = self.owner._selected_backend_name
            self._selected_backend_device = self.owner._selected_backend_device
            self._selected_solver = "fista"
            self._params = np.asarray(self.owner._params, dtype=float).copy()
            return self

    def recording_bootstrap(owner, X_value, y_value):
        bootstrap_calls.append(float(owner.alpha))
        return original_bootstrap(owner, X_value, y_value)

    monkeypatch.setattr(
        _bootstrap, "_make_child_refit", lambda owner, **_kwargs: _FastChild(owner)
    )
    monkeypatch.setattr(
        _bootstrap, "_backend_native_gaussian_residual_bootstrap", recording_bootstrap
    )
    cv = PenalizedGLM_CV(
        loss="squared_error",
        penalty="l1",
        alpha_grid=np.array([0.08, 0.04]),
        cv=2,
        random_state=3,
        device="cpu",
        solver="fista",
        max_iter=250,
        tol=1e-6,
        compute_inference=True,
        inference_method="bootstrap",
        cov_type="nonrobust",
    ).fit(X, y)
    assert len(bootstrap_calls) == 1
    assert bootstrap_calls[0] == pytest.approx(cv.alpha_)
    assert len(child_calls) == 200
    assert cv._inference_result is cv.estimator_._inference_result
    assert cv.inference_requested_method_ == "bootstrap"
    assert cv.inference_resolved_method_ == "residual_bootstrap"
    assert cv.inference_method_ == "residual_bootstrap"
    assert cv.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.penalty_selection_adjusted_ is False
    assert cv.estimator_.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.estimator_.penalty_selection_adjusted_ is False
    assert cv._inference_result.metadata["selected_alpha"] == pytest.approx(cv.alpha_)


def test_formula_and_array_routes_match_for_gaussian_bootstrap():
    pd = pytest.importorskip("pandas")
    X, y = _data(seed=153, n=48, p=2)
    frame = pd.DataFrame(X, columns=["x1", "x2"])
    frame["y"] = y
    common = dict(
        loss="squared_error",
        penalty="l1",
        alpha=0.04,
        device="cpu",
        solver="fista",
        max_iter=500,
        tol=1e-8,
        compute_inference=True,
        inference_method="bootstrap",
    )
    array_model = PenalizedGeneralizedLinearModel(**common)
    array_model.n_bootstrap = 4
    array_model.bootstrap_random_state = 31
    array_model.fit(X, y)
    formula_model = PenalizedGeneralizedLinearModel(**common)
    formula_model.n_bootstrap = 4
    formula_model.bootstrap_random_state = 31
    formula_model.fit(formula="y ~ x1 + x2", data=frame)
    np.testing.assert_allclose(array_model.coef_, formula_model.coef_, atol=1e-8)
    np.testing.assert_allclose(array_model._bse, formula_model._bse, atol=1e-8)
    np.testing.assert_allclose(array_model._conf_int, formula_model._conf_int, atol=1e-8)
    assert formula_model._inference_result.feature_names == ["(Intercept)", "x1", "x2"]


@pytest.mark.parametrize("wrapper", [Lasso, ElasticNet])
def test_specialized_sparse_gaussian_wrappers_keep_explicit_bootstrap_surface(wrapper):
    X, y = _data(seed=154, n=44, p=2)
    kwargs = dict(
        alpha=0.04,
        device="cpu",
        max_iter=500,
        tol=1e-8,
        compute_inference=True,
        inference_method="bootstrap",
    )
    if wrapper is ElasticNet:
        kwargs["l1_ratio"] = 0.4
    model = wrapper(**kwargs)
    model.n_bootstrap = 3
    model.bootstrap_random_state = 12
    model.fit(X, y)
    assert model.inference_requested_method_ == "bootstrap"
    assert model.inference_resolved_method_ == "residual_bootstrap"
    assert model._inference_result.metadata["numerical_backend"] == "numpy"


def test_installer_is_idempotent():
    before = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    _bootstrap.install_backend_native_gaussian_residual_bootstrap()
    middle = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    _bootstrap.install_backend_native_gaussian_residual_bootstrap()
    after = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    assert before is middle is after


def test_publish_contract_metadata_survives_backend_extension():
    X, y = _data(seed=155, n=28, p=2)
    model = _cpu_model(B=3, seed=5).fit(X, y)
    result = model._inference_result
    assert isinstance(result, ParameterInferenceResult)
    assert result.metadata["inference_requested_method"] == "bootstrap"
    assert result.metadata["inference_resolved_method"] == "residual_bootstrap"
    assert result.metadata["inference_target"] == "penalized_coefficient_distribution"
    assert result.metadata["penalty_conditioning"] == "fixed_penalty"
    assert result.metadata["penalty_selection_adjusted"] is None
