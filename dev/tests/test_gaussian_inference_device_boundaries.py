"""Regression tests for Gaussian inference backend/device boundaries."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import statgpu.linear_model._gaussian_inference as gi


class _HostTransferOnlyArray:
    """GPU-like array that forbids implicit NumPy coercion."""

    def __init__(self, value):
        self._value = np.asarray(value)

    def __array__(self, *args, **kwargs):
        raise TypeError("implicit NumPy conversion is forbidden")

    def get(self):
        return self._value.copy()


def test_numpy_boundary_uses_explicit_backend_to_host_conversion():
    source = _HostTransferOnlyArray([1.0, 2.0, 3.0])

    converted = gi._as_backend_array(source, "numpy")

    np.testing.assert_array_equal(converted, np.asarray([1.0, 2.0, 3.0]))
    assert converted.dtype == np.float64


def test_build_fit_state_uses_first_native_torch_device_for_host_design():
    torch = pytest.importorskip("torch")

    X = np.asarray(
        [[-1.0, 0.5], [0.0, 1.0], [1.0, 1.5], [2.0, 2.0]],
        dtype=np.float64,
    )
    coef = torch.tensor([0.6, -0.25], dtype=torch.float64)
    y = np.asarray(X @ coef.numpy() + [0.1, -0.1, 0.05, -0.05])

    state = gi.build_gaussian_fit_state(
        X,
        y,
        coef,
        0.0,
        False,
        backend="auto",
    )

    assert state.backend == "torch"
    assert state.device == "cpu"
    assert isinstance(state.X_design, torch.Tensor)
    assert state.X_design.device.type == "cpu"
    assert state.params.device.type == "cpu"


def test_direct_inference_uses_first_native_torch_device_for_host_design():
    torch = pytest.importorskip("torch")

    X_design = np.asarray(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    params = torch.tensor([0.5, 0.2], dtype=torch.float64)
    resid = torch.tensor([0.1, -0.1, 0.05, -0.05], dtype=torch.float64)
    scale = torch.tensor(0.02, dtype=torch.float64)

    result = gi.compute_gaussian_inference(
        X_design,
        params,
        resid,
        scale,
        df_resid=2,
        cov_type="nonrobust",
        backend="auto",
    )

    assert result is not None
    assert result.metadata["numerical_backend"] == "torch"
    assert result.metadata["numerical_device"] == "cpu"


def test_cupy_reference_inference_runs_on_statistic_device(monkeypatch):
    state = {"current": 0, "entered": [], "reference_calls": 0}

    class FakeArray:
        def __init__(self, device):
            self.device = types.SimpleNamespace(id=int(device))

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device
            state["entered"].append(self.device)
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    fake_cupy = types.SimpleNamespace(
        abs=lambda value: value,
        cuda=types.SimpleNamespace(Device=FakeDevice),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    statistic = FakeArray(1)
    expected = (object(), object())

    def fake_reference(
        statistic_abs,
        *,
        distribution,
        alpha,
        backend,
        xp,
        df=None,
        device=None,
    ):
        state["reference_calls"] += 1
        assert statistic_abs is statistic
        assert distribution == "normal"
        assert alpha == 0.05
        assert backend == "cupy"
        assert xp is fake_cupy
        assert device is None
        assert state["current"] == 1
        return expected

    monkeypatch.setattr(gi, "two_sided_reference_inference", fake_reference)

    result = gi._reference_inference(
        statistic,
        distribution="normal",
        alpha=0.05,
        backend="cupy",
        device="cuda:1",
    )

    assert result == expected
    assert state["entered"] == [1]
    assert state["reference_calls"] == 1
    assert state["current"] == 0


def _torch_gaussian_contract_model(monkeypatch):
    torch = pytest.importorskip("torch")
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
    )
    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="torch")
    )
    monkeypatch.setattr(
        model, "_auto_backend_override", lambda backend_name, X: backend_name
    )
    monkeypatch.setattr(
        model,
        "_select_solver",
        lambda loss, backend_name=None, X=None: "newton",
    )
    return torch, model


def test_pglm_inference_reuses_post_alignment_response(monkeypatch):
    torch, model = _torch_gaussian_contract_model(monkeypatch)

    class ResponseBeforeAlignment:
        def __init__(self, value):
            self.value = value
            self.shape = value.shape
            self.ndim = value.ndim

        def to(self, device=None, dtype=None):
            return self.value.to(device=device, dtype=dtype)

    real_to_array = model._to_array

    def staged_to_array(value, device=None, backend=None):
        converted = real_to_array(value, device=device, backend=backend)
        if getattr(converted, "ndim", None) == 1:
            return ResponseBeforeAlignment(converted)
        return converted

    monkeypatch.setattr(model, "_to_array", staged_to_array)
    dispatched = {}

    def fake_fit_torch(X, y, sample_weight=None):
        assert isinstance(y, torch.Tensor)
        dispatched["X"] = X
        dispatched["y"] = y
        model._native_fit_coef = torch.zeros(
            X.shape[1], dtype=X.dtype, device=X.device
        )
        model._native_fit_intercept = None
        model.coef_ = None
        model.intercept_ = None
        model._params = None
        model._df_resid = int(X.shape[0] - X.shape[1])

    def assert_final_arrays(X, y, sample_weight=None):
        assert X is dispatched["X"]
        assert y is dispatched["y"]

    monkeypatch.setattr(model, "_fit_torch", fake_fit_torch)
    monkeypatch.setattr(
        model, "_compute_post_fit_gaussian_inference", assert_final_arrays
    )

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    model.fit(X, y)

    assert model._fitted is True


def test_pglm_inference_promotes_reused_float32_arrays_to_native_fit_dtype(monkeypatch):
    torch, model = _torch_gaussian_contract_model(monkeypatch)
    dispatched = {}

    def fake_fit_torch(X, y, sample_weight=None):
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32
        dispatched["X"] = X
        dispatched["y"] = y
        model._native_fit_coef = torch.zeros(
            X.shape[1], dtype=torch.float64, device=X.device
        )
        model._native_fit_intercept = None
        model.coef_ = None
        model.intercept_ = None
        model._params = None
        model._df_resid = int(X.shape[0] - X.shape[1])

    def assert_solver_precision(X, y, sample_weight=None):
        assert X.dtype == torch.float64
        assert y.dtype == torch.float64
        assert X.device == dispatched["X"].device
        assert y.device == dispatched["y"].device

    monkeypatch.setattr(model, "_fit_torch", fake_fit_torch)
    monkeypatch.setattr(
        model, "_compute_post_fit_gaussian_inference", assert_solver_precision
    )

    X = np.arange(18.0, dtype=np.float32).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    model.fit(X, y)

    assert model._fitted is True


def test_pglm_cupy_context_entry_failure_cleans_up_once(monkeypatch):
    import statgpu.linear_model.penalized._fit_mixin as fit_mixin_module
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []

    class FakeArray:
        def __init__(self, shape, device=1):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.device = types.SimpleNamespace(id=int(device))

    class FailingDevice:
        def __init__(self, device):
            self.device = int(device)

        def __enter__(self):
            raise RuntimeError("synthetic CuPy context-entry failure")

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_cupy = types.SimpleNamespace(
        cuda=types.SimpleNamespace(Device=FailingDevice)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        fit_mixin_module,
        "_cupy_asarray_on_device",
        lambda value, target_device, dtype=None: value,
    )

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
    )
    model._fitted = True
    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="cupy")
    )
    monkeypatch.setattr(
        model, "_auto_backend_override", lambda backend_name, X: backend_name
    )
    monkeypatch.setattr(
        model,
        "_select_solver",
        lambda loss, backend_name=None, X=None: "newton",
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, device=None, backend=None: FakeArray(np.asarray(value).shape),
    )
    monkeypatch.setattr(
        model, "_cleanup_cuda_memory", lambda: events.append("cleanup")
    )
    monkeypatch.setattr(
        model, "_fit_gpu", lambda *args, **kwargs: events.append("fit")
    )

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    with pytest.raises(RuntimeError, match="synthetic CuPy context-entry failure"):
        model.fit(X, y)

    assert events == ["cleanup"]
    assert model._native_fit_coef is None
    assert model._native_fit_intercept is None
    assert model._fitted is False


def test_pglm_exact_precomputed_gpu_cleanup_is_not_repeated(monkeypatch):
    import statgpu.linear_model.penalized._fit_mixin as fit_mixin_module
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []

    class FakeArray:
        def __init__(self, shape, device=1):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.device = types.SimpleNamespace(id=int(device))

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=FakeDevice))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        fit_mixin_module,
        "_cupy_asarray_on_device",
        lambda value, target_device, dtype=None: value,
    )

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        solver="exact",
        compute_inference=True,
        gpu_memory_cleanup=True,
    )
    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="cupy")
    )
    monkeypatch.setattr(
        model, "_auto_backend_override", lambda backend_name, X: backend_name
    )
    monkeypatch.setattr(
        model,
        "_select_solver",
        lambda loss, backend_name=None, X=None: "exact",
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, device=None, backend=None: FakeArray(np.asarray(value).shape),
    )
    monkeypatch.setattr(
        model, "_cleanup_cuda_memory", lambda: events.append("cleanup")
    )

    def consume_precomputed_state(*args, **kwargs):
        assert model._inference_precomputed is True
        model._inference_precomputed = False

    monkeypatch.setattr(
        model, "_compute_post_fit_gaussian_inference", consume_precomputed_state
    )

    def fake_fit_gpu(X, y, sample_weight=None):
        events.append("fit")
        model._inference_precomputed = True
        model._cleanup_cuda_memory()

    monkeypatch.setattr(model, "_fit_gpu", fake_fit_gpu)

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    model.fit(X, y)

    assert events == ["fit", "cleanup"]
    assert model._fitted is True


@pytest.mark.parametrize(
    "penalty_name", ["l2", "l2_squared", "ridge", "none", "null", "", None]
)
def test_gaussian_fit_transaction_contract_covers_resolved_l2_spellings(penalty_name):
    from statgpu.linear_model import PenalizedGeneralizedLinearModel
    from statgpu.linear_model.penalized import _gaussian_fit_transaction_contract as contract

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=penalty_name,
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
    )

    assert contract._needs_gaussian_conversion_contract(model) is True


def test_penalized_linear_weighted_diagnostics_use_raw_outcomes():
    from statgpu.linear_model import PenalizedLinearRegression

    X = np.asarray(
        [
            [-2.0, 0.5],
            [-1.0, 1.5],
            [0.0, -0.5],
            [1.0, 2.0],
            [2.0, 0.25],
            [3.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([-1.2, 0.1, 0.4, 2.2, 1.6, 3.8], dtype=np.float64)
    weights = np.asarray([1.0, 4.0, 2.0, 9.0, 3.0, 5.0], dtype=np.float64)

    model = PenalizedLinearRegression(
        penalty="l2",
        alpha=0.15,
        fit_intercept=True,
        device="cpu",
        solver="exact",
        compute_inference=True,
    )
    model.fit(X, y, sample_weight=weights)

    pred = np.asarray(model.predict(X), dtype=float)
    raw_resid = y - pred
    y_mean = np.average(y, weights=weights)
    ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
    ss_res = float(np.sum(weights * raw_resid ** 2))
    expected_r2 = 1.0 - ss_res / ss_tot
    k = X.shape[1]
    expected_f = ((ss_tot - ss_res) / k) / (ss_res / model._df_resid)

    np.testing.assert_allclose(model._y, y, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(model._raw_resid, raw_resid, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model._sample_weight_fit, weights, rtol=0.0, atol=0.0)
    assert model.rsquared == pytest.approx(expected_r2, rel=1e-10, abs=1e-10)
    assert model.fvalue == pytest.approx(expected_f, rel=1e-10, abs=1e-10)


def test_pglm_failed_refit_clears_stale_prediction_state():
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(-1.0, 1.0, 6)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
    )
    model.fit(X, y)
    assert model._fitted is True
    assert model.coef_ is not None
    assert model._inference_result is not None

    with pytest.raises(ValueError, match=r"Either formula\+data or X\+y"):
        model.fit()

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._inference_result is None
    assert model._selected_backend_name is None
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict(X)
