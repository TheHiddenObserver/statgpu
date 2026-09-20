"""Issue #163 regressions for truthful penalized Quantile solver identity."""

from __future__ import annotations

import importlib
import inspect
import sys
import types

import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)
from statgpu.linear_model.penalized import _penalized_quantile as _typed_quantile_mod
from statgpu.linear_model.penalized import _predict_mixin as _predict_mixin_mod
from statgpu.linear_model.penalized import _quantile_solver_contract as _quantile_solver_contract
import statgpu.backends._utils as _backend_utils


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedQuantileRegression(
            quantile=0.3,
            penalty="l2",
            alpha=0.02,
            device="cpu",
        ),
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.3},
            penalty="l2",
            alpha=0.02,
            device="cpu",
        ),
    ],
)
def test_quantile_cupy_prediction_reuses_recorded_fit_device(
    monkeypatch, factory
):
    model = factory()
    model.coef_ = np.asarray([0.5, -0.25], dtype=np.float64)
    model.intercept_ = 0.2
    model._selected_backend_name = "cupy"
    model._selected_backend_device = "cuda:4"

    state = {"current": None, "entered": [], "targets": []}

    class FakeDevice:
        def __init__(self, device_id):
            self.device_id = int(device_id)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device_id
            state["entered"].append(self.device_id)
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    fake_cupy = types.SimpleNamespace(
        cuda=types.SimpleNamespace(Device=FakeDevice)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    def fake_align(value, target_device, dtype=None):
        state["targets"].append(int(target_device))
        return np.asarray(value, dtype=dtype)

    monkeypatch.setattr(
        _backend_utils,
        "_cupy_asarray_on_device",
        fake_align,
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, device: np.asarray(value),
    )

    X = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)
    raw = model._quantile_cupy_linear_prediction(X)

    np.testing.assert_allclose(
        raw,
        X @ model.coef_ + model.intercept_,
        rtol=0.0,
        atol=0.0,
    )
    assert state["entered"] == [4]
    assert state["targets"] == [4, 4, 4]


def test_penalized_quantile_intercept_column_uses_design_cupy_device(monkeypatch):
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha=0.02,
        device="cpu",
    )
    state = {"current": None, "entered": [], "ones_device": None}

    class FakeDeviceContext:
        def __init__(self, device_id):
            self.device_id = int(device_id)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device_id
            state["entered"].append(self.device_id)
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    def fake_ones(n, dtype=None):
        state["ones_device"] = state["current"]
        return np.ones(n, dtype=dtype)

    fake_cupy = types.SimpleNamespace(
        ones=fake_ones,
        cuda=types.SimpleNamespace(Device=FakeDeviceContext),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    class FakeDevice:
        id = 6

    class FakeRef:
        device = FakeDevice()
        dtype = np.dtype("float64")

    result = model._ones(4, "cupy", FakeRef())

    np.testing.assert_array_equal(result, np.ones(4, dtype=np.float64))
    assert state["entered"] == [6]
    assert state["ones_device"] == 6


class _NoImplicitNumpyArray:
    """Backend-like response container that forbids NumPy implicit conversion."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.shape = self.values.shape

    def __array__(self, *args, **kwargs):
        raise TypeError("implicit host conversion is forbidden")


def _data(seed=16301, n=96, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.75, -0.3], dtype=np.float64)[:p]
    y = (0.25 + X @ beta + rng.laplace(scale=0.18, size=n)).astype(np.float64)
    return X, y


@pytest.mark.parametrize(
    ("penalty", "penalty_kwargs"),
    [
        ("l1", None),
        ("group_lasso", {"groups": [[0], [1]]}),
    ],
)
def test_quantile_fit_rejects_response_length_mismatch_before_backend(
    monkeypatch, penalty, penalty_kwargs
):
    X, y = _data(seed=16364, n=32, p=2)
    model = PenalizedQuantileRegression(
        quantile=0.3,
        penalty=penalty,
        penalty_kwargs=penalty_kwargs,
        alpha=0.02,
        solver="auto",
        device="cpu",
        compute_inference=False,
    )

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("invalid response length must fail before backend work")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match="Response length must match"):
        model.fit(X, y[:1])


def test_quantile_fit_single_column_response_matches_one_dimensional_response():
    X, y = _data(seed=16365, n=40, p=2)
    common = dict(
        quantile=0.35,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    one_dimensional = PenalizedQuantileRegression(**common).fit(X, y)
    single_column = PenalizedQuantileRegression(**common).fit(X, y[:, None])

    np.testing.assert_allclose(
        single_column.coef_, one_dimensional.coef_, rtol=0.0, atol=0.0
    )
    assert single_column.intercept_ == pytest.approx(
        one_dimensional.intercept_, rel=0.0, abs=0.0
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.5},
            penalty="l2",
            alpha=0.03,
            solver="auto",
            device="cpu",
            max_iter=300,
            tol=1e-8,
        ),
        lambda: PenalizedQuantileRegression(
            quantile=0.5,
            penalty="l2",
            alpha=0.03,
            solver="auto",
            device="cpu",
            max_iter=300,
            tol=1e-8,
        ),
    ],
)
def test_l2_quantile_auto_reports_and_executes_irls(factory):
    X, y = _data()
    model = factory().fit(X, y)

    assert model._selected_solver == "irls"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.n_iter_ >= 1


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("max_iter", True, "max_iter must be a positive integer"),
        ("tol", 0.0, "tol must be a finite positive number"),
        ("tol", "1e-6", "tol must be a finite positive number"),
        (
            "lipschitz_L",
            0.0,
            "lipschitz_L must be None or a finite positive number",
        ),
        (
            "lipschitz_L",
            True,
            "lipschitz_L must be None or a finite positive number",
        ),
        (
            "lipschitz_L",
            "1.0",
            "lipschitz_L must be None or a finite positive number",
        ),
    ],
)
def test_quantile_direct_public_stopping_controls_fail_closed(name, value, message):
    X, y = _data(seed=16311)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    setattr(model, name, value)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_rejected_quantile_refit_clears_all_fit_derived_state():
    X, y = _data(seed=16333, n=64)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=200,
        tol=1e-8,
    ).fit(X, y)

    assert model._loss is not None
    assert model._penalty is not None
    assert model._nobs == X.shape[0]
    assert model._df_resid is not None

    model.tol = 0.0
    with pytest.raises(ValueError, match="tol must be a finite positive number"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.n_iter_ == 0
    assert model._selected_solver is None
    assert model._selected_backend_name is None
    assert model._selected_backend_device is None
    assert model._loss is None
    assert model._penalty is None
    assert model._nobs is None
    assert model._df_resid is None
    assert model._X_design is None
    assert model._y is None
    assert model._resid is None
    assert model._scale is None
    assert not hasattr(model, "n_features_in_")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"cv": True},
            "cv must be an integer greater than or equal to 2",
        ),
        (
            {"cv": 2.5},
            "cv must be an integer greater than or equal to 2",
        ),
        (
            {"cv": "2"},
            "cv must be an integer greater than or equal to 2",
        ),
        (
            {"acknowledge_approx": 1},
            "acknowledge_approx must be boolean",
        ),
        (
            {"refine_top_k": True},
            "refine_top_k must be a positive integer",
        ),
        (
            {"refine_top_k": 1.5},
            "refine_top_k must be a positive integer",
        ),
        (
            {"refine_top_k": "3"},
            "refine_top_k must be a positive integer",
        ),
    ],
)
def test_quantile_cv_constructor_controls_fail_before_coercion(kwargs, message):
    constructor = dict(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
    )
    constructor.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PenalizedGLM_CV(**constructor)


def test_quantile_cv_constructor_guard_is_idempotent_under_reload():
    before = PenalizedGLM_CV.__init__
    marker = _quantile_solver_contract._CV_INIT_VALIDATE_MARKER
    assert getattr(before, marker, False)

    reloaded = importlib.reload(_quantile_solver_contract)
    after = PenalizedGLM_CV.__init__

    assert after is before
    assert getattr(after, reloaded._CV_INIT_VALIDATE_MARKER, False)
    with pytest.raises(
        ValueError,
        match="cv must be an integer greater than or equal to 2",
    ):
        PenalizedGLM_CV(
            loss="quantile",
            penalty="l2",
            alpha_grid=np.asarray([0.03], dtype=np.float64),
            cv=2.5,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("tol", True, "tol must be a finite positive number"),
        ("tol", "1e-6", "tol must be a finite positive number"),
    ],
)
def test_quantile_cv_public_stopping_controls_fail_closed(name, value, message):
    X, y = _data(seed=16312, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    setattr(cv, name, value)

    with pytest.raises(ValueError, match=message):
        cv.fit(X, y)


def test_quantile_direct_public_device_replacement_is_authoritative_before_backend(
    monkeypatch,
):
    X, y = _data(seed=16320)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    model.device = "cuda"

    def capture_backend(backend="auto"):
        assert model._device == Device.CUDA
        raise RuntimeError("device sync sentinel")

    monkeypatch.setattr(model, "_get_backend", capture_backend)
    with pytest.raises(RuntimeError, match="device sync sentinel"):
        model.fit(X, y)


def test_quantile_cv_public_device_replacement_is_authoritative_before_routing(
    monkeypatch,
):
    X, y = _data(seed=16321, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    cv.device = "torch"
    seen = {}

    def capture_device(self, *args, **kwargs):
        seen["device"] = self._device
        raise RuntimeError("device routing sentinel")

    monkeypatch.setattr(PenalizedGLM_CV, "_effective_cv_device", capture_device)
    with pytest.raises(RuntimeError, match="device routing sentinel"):
        cv.fit(X, y)

    assert seen["device"] == Device.TORCH
    assert cv._device == Device.TORCH


@pytest.mark.parametrize("owner_kind", ["direct", "cv"])
def test_quantile_invalid_public_device_replacement_fails_closed(owner_kind):
    X, y = _data(seed=16322, n=72)
    if owner_kind == "direct":
        owner = PenalizedQuantileRegression(
            quantile=0.4,
            penalty="l2",
            alpha=0.02,
            solver="auto",
            device="cpu",
        )
    else:
        owner = PenalizedGLM_CV(
            loss="quantile",
            loss_kwargs={"quantile": 0.4},
            penalty="l2",
            alpha_grid=np.asarray([0.03], dtype=np.float64),
            cv=2,
            solver="auto",
            device="cpu",
        )

    owner.device = "not-a-device"
    with pytest.raises(ValueError, match="device must be one of"):
        owner.fit(X, y)


def test_generic_quantile_direct_public_loss_kwargs_replacement_is_authoritative():
    X, y = _data(seed=16335)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    model.loss_kwargs = {"quantile": 0.8}
    model.fit(X, y)

    assert model._loss_kwargs == {"quantile": 0.8}
    assert getattr(model._loss, "_tau", None) == pytest.approx(0.8)


def test_quantile_direct_public_fit_intercept_replacement_is_authoritative():
    X, y = _data(seed=16315)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        fit_intercept=True,
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    model.fit_intercept = False
    model.fit(X, y)

    assert model._fit_intercept is False
    assert model._effective_intercept is False
    assert model.intercept_ == 0.0


def test_quantile_direct_invalid_public_fit_intercept_fails_before_backend(monkeypatch):
    X, y = _data(seed=16316)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        fit_intercept=True,
        device="cpu",
    )
    model.fit_intercept = "False"

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match="fit_intercept must be boolean"):
        model.fit(X, y)


def test_invalid_direct_quantile_refit_control_clears_prior_fit_state():
    X, y = _data(seed=16317)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    ).fit(X, y)
    assert model._fitted is True
    assert model.coef_ is not None

    model.max_iter = 0
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._selected_solver is None


def test_invalid_quantile_cv_refit_control_clears_prior_selection_state():
    X, y = _data(seed=16318, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    ).fit(X, y)
    assert cv._fitted is True
    assert cv.estimator_ is not None

    cv.tol = "1e-6"
    with pytest.raises(ValueError, match="tol must be a finite positive number"):
        cv.fit(X, y)

    assert cv._fitted is False
    assert cv.alpha_ is None
    assert cv.estimator_ is None
    assert cv.coef_ is None
    assert cv.intercept_ is None
    assert cv.cv_results_ is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedQuantileRegression(
            quantile=0.3,
            penalty="l2",
            alpha=0.02,
            solver="irls",
            device="cpu",
        ),
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.3},
            penalty="l2",
            alpha=0.02,
            solver="auto",
            device="cpu",
        ),
    ],
)
def test_quantile_predict_rejects_non_2d_or_wrong_width_before_backend(
    monkeypatch, factory
):
    model = factory()
    model.coef_ = np.asarray([0.5, -0.2], dtype=np.float64)
    model.intercept_ = 0.1

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("invalid prediction shape must fail before backend work")

    monkeypatch.setattr(model, "_prediction_backend_name", forbidden_backend)

    with pytest.raises(ValueError, match="two-dimensional"):
        model.predict(np.asarray([1.0, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="same number of features"):
        model.predict(np.ones((3, 1), dtype=np.float64))


@pytest.mark.parametrize(
    ("factory", "module"),
    [
        (
            lambda: PenalizedQuantileRegression(
                quantile=0.3,
                penalty="l2",
                alpha=0.02,
                solver="irls",
                device="cpu",
            ),
            _typed_quantile_mod,
        ),
        (
            lambda: PenalizedGeneralizedLinearModel(
                loss="quantile",
                loss_kwargs={"quantile": 0.3},
                penalty="l2",
                alpha=0.02,
                solver="auto",
                device="cpu",
            ),
            _predict_mixin_mod,
        ),
    ],
)
def test_quantile_score_uses_explicit_reporting_conversion_for_backend_y(
    monkeypatch, factory, module
):
    X, y = _data(seed=16361, n=24)
    backend_y = _NoImplicitNumpyArray(y)
    model = factory()
    model.predict = lambda X_arg, return_cpu=True: np.zeros_like(y)

    original_to_numpy = module._to_numpy

    def reporting_to_numpy(value):
        if value is backend_y:
            return backend_y.values
        return original_to_numpy(value)

    monkeypatch.setattr(module, "_to_numpy", reporting_to_numpy)

    score = model.score(X, backend_y)
    assert np.isfinite(score)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedQuantileRegression(
            quantile=0.3,
            penalty="l2",
            alpha=0.02,
            solver="irls",
            device="cpu",
        ),
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.3},
            penalty="l2",
            alpha=0.02,
            solver="auto",
            device="cpu",
        ),
    ],
)
def test_quantile_score_rejects_complex_response_before_prediction(
    monkeypatch, factory
):
    X, y = _data(seed=16370, n=20)
    model = factory()

    def forbidden_predict(*args, **kwargs):
        raise AssertionError("non-real score response must fail before prediction")

    monkeypatch.setattr(model, "predict", forbidden_predict)
    with pytest.raises(ValueError, match="real numeric values"):
        model.score(X, y.astype(np.complex128) + 1j)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedQuantileRegression(
            quantile=0.3,
            penalty="l2",
            alpha=0.02,
            solver="irls",
            device="cpu",
        ),
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.3},
            penalty="l2",
            alpha=0.02,
            solver="auto",
            device="cpu",
        ),
    ],
)
def test_quantile_score_rejects_non_1d_response_before_prediction(
    monkeypatch, factory
):
    X, y = _data(seed=16362, n=20)
    model = factory()

    def forbidden_predict(*args, **kwargs):
        raise AssertionError("invalid response shape must fail before prediction")

    monkeypatch.setattr(model, "predict", forbidden_predict)
    with pytest.raises(ValueError, match="one-dimensional"):
        model.score(X, y[:, None])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedQuantileRegression(
            quantile=0.3,
            penalty="l2",
            alpha=0.02,
            solver="irls",
            device="cpu",
        ),
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.3},
            penalty="l2",
            alpha=0.02,
            solver="auto",
            device="cpu",
        ),
    ],
)
def test_quantile_score_rejects_response_length_mismatch(factory):
    X, y = _data(seed=16363, n=20)
    model = factory()
    model.predict = lambda X_arg, return_cpu=True: np.zeros(X.shape[0], dtype=np.float64)

    with pytest.raises(ValueError, match="same number of observations as X"):
        model.score(X, y[:1])


def test_generic_quantile_score_rejects_invalid_weights_before_prediction(monkeypatch):
    X = np.array(
        [[-1.0, 0.2], [0.0, -0.1], [0.5, 0.4], [1.0, -0.3]],
        dtype=np.float64,
    )
    y = np.array([-0.4, 0.1, 0.35, 0.8], dtype=np.float64)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=200,
        tol=1e-8,
    ).fit(X, y)

    def forbidden_predict(*args, **kwargs):
        raise AssertionError("invalid score weights must fail before prediction")

    monkeypatch.setattr(model, "predict", forbidden_predict)
    with pytest.raises(ValueError, match="sample_weight"):
        model.score(
            X,
            y,
            sample_weight=np.array([1.0, -0.2, 1.0, 1.0]),
        )


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.array([1.0, -0.2, 1.0, 1.0]),
        np.array([1.0, np.nan, 1.0, 1.0]),
        np.zeros(4, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    ],
)
def test_generic_quantile_score_rejects_invalid_sample_weight(sample_weight):
    X = np.array(
        [[-1.0, 0.2], [0.0, -0.1], [0.5, 0.4], [1.0, -0.3]],
        dtype=np.float64,
    )
    y = np.array([-0.4, 0.1, 0.35, 0.8], dtype=np.float64)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=200,
        tol=1e-8,
    ).fit(X, y)

    with pytest.raises(ValueError, match="sample_weight"):
        model.score(X, y, sample_weight=sample_weight)


def test_explicit_irls_l2_matches_auto_quantile_fit():
    X, y = _data(seed=16302)
    common = dict(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        device="cpu",
        max_iter=400,
        tol=1e-9,
    )
    auto = PenalizedQuantileRegression(solver="auto", **common).fit(X, y)
    explicit = PenalizedQuantileRegression(solver="irls", **common).fit(X, y)

    assert auto._selected_solver == explicit._selected_solver == "irls"
    np.testing.assert_allclose(auto.coef_, explicit.coef_, rtol=0.0, atol=0.0)
    assert auto.intercept_ == pytest.approx(explicit.intercept_, rel=0.0, abs=0.0)


def test_explicit_fista_l2_quantile_executes_true_fista(monkeypatch):
    from statgpu.losses import QuantileLoss

    X, y = _data(seed=16303)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError("explicit solver='fista' must not execute Quantile IRLS")

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista",
        device="cpu",
        max_iter=1200,
        tol=1e-7,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.n_iter_ >= 1


def test_explicit_quantile_fista_propagates_public_lipschitz_control(monkeypatch):
    import statgpu.solvers as solvers

    X, y = _data(seed=163031)
    captured = {}

    def fake_fista(loss, penalty, X_fit, y_fit, **kwargs):
        captured["lipschitz_L"] = kwargs.get("lipschitz_L")
        return np.zeros(X_fit.shape[1], dtype=np.float64), 1

    monkeypatch.setattr(solvers, "fista_solver", fake_fista)

    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.03,
        solver="fista",
        lipschitz_L=3.25,
        device="cpu",
        fit_intercept=False,
        max_iter=20,
        tol=1e-7,
    ).fit(X, y)

    assert captured["lipschitz_L"] == pytest.approx(3.25)
    assert model._selected_solver == "fista"
    assert model.n_iter_ == 1


def test_explicit_fista_bb_l2_quantile_fails_before_backend_fit(monkeypatch):
    X, y = _data(seed=16303)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista_bb",
        device="cpu",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("backend fit must not run for rejected explicit solver")

    monkeypatch.setattr(model, "_fit_cpu", forbidden)
    with pytest.raises(ValueError, match="not supported for L2/no-penalty Quantile objectives"):
        model.fit(X, y)


def test_sparse_quantile_auto_remains_fista_family():
    X, y = _data(seed=16304)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l1",
        alpha=0.03,
        solver="auto",
        device="cpu",
        max_iter=500,
        tol=1e-7,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert np.all(np.isfinite(model.coef_))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("lla", False, "require lla=True"),
        ("lla", "False", "lla must be boolean"),
        ("max_lla_iters", 0, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", False, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", 2, "max_lla_iters must be at least 3"),
        ("lla_tol", 0.0, "lla_tol must be a finite positive number"),
        ("lla_tol", "1e-6", "lla_tol must be a finite positive number"),
    ],
)
def test_scalar_nonconvex_quantile_public_lla_controls_fail_closed(
    name, value, message
):
    X, y = _data(seed=16313, n=64)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    setattr(model, name, value)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_scalar_quantile_small_max_iter_is_never_expanded_by_continuation(
    monkeypatch,
):
    import statgpu.solvers as solvers

    X, y = _data(seed=16319, n=64)
    seen = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["max_iter"] = list(kwargs["max_iter"])
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(solvers, "proximal_irls_quantile_solver", fake_solver)
    PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=7,
        tol=1e-6,
        max_lla_iters=6,
    ).fit(X, y)

    assert seen["max_iter"] == [1, 1, 7]


def test_scalar_quantile_three_lla_budget_maps_to_one_update_per_continuation(
    monkeypatch,
):
    import statgpu.solvers as solvers

    X, y = _data(seed=16314, n=64)
    seen = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["n_steps"] = len(alpha_path)
        seen["max_lla_per_step"] = kwargs["max_lla_per_step"]
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(solvers, "proximal_irls_quantile_solver", fake_solver)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
        max_lla_iters=3,
    ).fit(X, y)

    assert seen == {"n_steps": 3, "max_lla_per_step": 1}
    assert model._selected_solver == "proximal_irls_cd"


def test_nonconvex_quantile_auto_reports_dedicated_proximal_irls_cd():
    X, y = _data(seed=16307, n=64)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=160,
        tol=1e-6,
    ).fit(X, y)

    assert model._selected_solver == "proximal_irls_cd"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)


def test_explicit_scalar_scad_solver_error_precedes_irrelevant_lla_controls():
    X, y = _data(seed=16337, n=48)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="fista",
        device="cpu",
        lla=False,
        max_lla_iters=1,
        lla_tol="unused-for-rejected-explicit-route",
    )

    with pytest.raises(ValueError, match="dedicated Proximal IRLS-CD"):
        model.fit(X, y)


@pytest.mark.parametrize("solver", ["fista", "fista_bb", "admm"])
def test_explicit_solver_does_not_silently_replace_quantile_scad(
    monkeypatch, solver
):
    X, y = _data(seed=16308, n=48)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver=solver,
        device="cpu",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("backend fit must not run for rejected explicit solver")

    monkeypatch.setattr(model, "_fit_cpu", forbidden)
    with pytest.raises(ValueError, match="dedicated Proximal IRLS-CD"):
        model.fit(X, y)


def test_quantile_cv_l2_uses_irls_for_candidates_and_final_refit():
    X, y = _data(seed=16305, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="l2",
        alpha_grid=np.array([0.04, 0.02], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=250,
        tol=1e-8,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "irls"
    assert cv.estimator_._selected_solver == "irls"
    assert cv.alpha_ in {0.04, 0.02}
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_cv_sparse_path_and_final_refit_remain_fista():
    X, y = _data(seed=16306, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="l1",
        alpha_grid=np.array([0.04], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=350,
        tol=1e-7,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "fista"
    assert cv.estimator_._selected_solver == "fista"
    assert cv.alpha_ == pytest.approx(0.04)
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_cv_scad_reports_dedicated_solver_for_candidates_and_refit():
    X, y = _data(seed=16309, n=60)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="scad",
        alpha_grid=np.array([0.025], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=140,
        tol=1e-6,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "proximal_irls_cd"
    assert cv.estimator_._selected_solver == "proximal_irls_cd"
    assert cv.alpha_ == pytest.approx(0.025)
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_formula_route_preserves_auto_irls_identity_and_numerics():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")

    X, y = _data(seed=16310, n=84)
    df = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})
    common = dict(
        quantile=0.45,
        penalty="l2",
        alpha=0.025,
        solver="auto",
        device="cpu",
        max_iter=350,
        tol=1e-9,
    )

    array_model = PenalizedQuantileRegression(**common).fit(X, y)
    formula_model = PenalizedQuantileRegression(**common).fit(
        formula="y ~ x1 + x2", data=df
    )

    assert array_model._selected_solver == "irls"
    assert formula_model._selected_solver == "irls"
    assert array_model._selected_backend_name == formula_model._selected_backend_name == "numpy"
    np.testing.assert_allclose(
        formula_model.coef_, array_model.coef_, rtol=0.0, atol=2e-10
    )
    assert formula_model.intercept_ == pytest.approx(
        array_model.intercept_, rel=0.0, abs=2e-10
    )


def test_quantile_solver_contract_installer_is_idempotent_and_signature_safe():
    from statgpu.linear_model.penalized import _fit_mixin
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    before_policy = _fit_mixin._preferred_penalized_glm_solver
    before_validate = PenalizedGeneralizedLinearModel._validate_solver_penalty
    before_direct_fit = PenalizedGeneralizedLinearModel.fit
    before_cv_fit = PenalizedGLM_CV.fit
    before_resolve_penalty = PenalizedGeneralizedLinearModel._resolve_penalty
    policy_signature = inspect.signature(before_policy)
    validate_signature = inspect.signature(before_validate)

    contract.install_quantile_solver_contract()

    assert _fit_mixin._preferred_penalized_glm_solver is before_policy
    assert PenalizedGeneralizedLinearModel._validate_solver_penalty is before_validate
    assert PenalizedGeneralizedLinearModel.fit is before_direct_fit
    assert PenalizedGLM_CV.fit is before_cv_fit
    assert PenalizedGeneralizedLinearModel._resolve_penalty is before_resolve_penalty
    assert inspect.signature(before_policy) == policy_signature
    assert inspect.signature(before_validate) == validate_signature
    assert hasattr(before_policy, "__wrapped__")
    assert hasattr(before_validate, "__wrapped__")
    assert getattr(
        before_direct_fit, contract._DIRECT_FIT_SOLVER_SYNC_MARKER, False
    )
    assert getattr(before_cv_fit, contract._CV_FIT_SOLVER_SYNC_MARKER, False)
    assert getattr(
        before_resolve_penalty, contract._RESOLVE_PENALTY_MARKER, False
    )


def test_quantile_solver_contract_reload_preserves_layered_explicit_solver_semantics(
    monkeypatch,
):
    import importlib

    from statgpu.losses import QuantileLoss
    from statgpu.linear_model.penalized import _fit_mixin
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    before_direct_fit = PenalizedGeneralizedLinearModel.fit
    before_cv_fit = PenalizedGLM_CV.fit
    before_backend_fit = _fit_mixin._PenalizedFitMixin._fit_loss_backend

    reloaded = importlib.reload(contract)

    assert PenalizedGeneralizedLinearModel.fit is before_direct_fit
    assert PenalizedGLM_CV.fit is before_cv_fit
    assert _fit_mixin._PenalizedFitMixin._fit_loss_backend is before_backend_fit

    X, y = _data(seed=16338, n=48)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError(
            "reload must not make explicit smooth Quantile FISTA fall back to IRLS"
        )

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista",
        device="cpu",
        max_iter=300,
        tol=1e-6,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert model._selected_backend_name == "numpy"

    rejected = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista_bb",
        device="cpu",
    )
    with pytest.raises(ValueError, match="not supported|does not support"):
        rejected.fit(X, y)

    assert getattr(
        reloaded._validate_quantile_solver_request,
        "_statgpu_quantile_smooth_fista_validator_contract",
        False,
    )


def test_quantile_solver_installer_is_import_order_safe_in_fresh_interpreter():
    import json
    import subprocess
    import sys

    code = r'''
import json
from statgpu.linear_model.penalized import _fit_mixin
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
payload = {
    "l2_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "l2", backend_name="numpy"
    ),
    "sparse_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "l1", backend_name="numpy"
    ),
    "nonconvex_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "scad", backend_name="numpy"
    ),
    "policy_wrapped": hasattr(
        _fit_mixin._preferred_penalized_glm_solver, "__wrapped__"
    ),
    "validate_wrapped": hasattr(
        PenalizedGeneralizedLinearModel._validate_solver_penalty, "__wrapped__"
    ),
}
print(json.dumps(payload, sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {
        "l2_auto": "irls",
        "nonconvex_auto": "proximal_irls_cd",
        "policy_wrapped": True,
        "sparse_auto": "fista",
        "validate_wrapped": True,
    }
