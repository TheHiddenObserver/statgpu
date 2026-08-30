"""Regression coverage for accelerator provenance and Gaussian cleanup/reporting boundaries."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from statgpu.backends._validation import check_finite


def test_cupy_finite_reduction_failure_records_backend_provenance(monkeypatch):
    class FakeCuPyArray:
        __module__ = "cupy"
        device = types.SimpleNamespace(id=1)

    def fail_isfinite(value):
        raise RuntimeError("synthetic CuPy finite-check allocation failure")

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        types.SimpleNamespace(isfinite=fail_isfinite),
    )

    with pytest.raises(RuntimeError, match="synthetic CuPy") as exc_info:
        check_finite(FakeCuPyArray(), name="X")

    assert getattr(exc_info.value, "_statgpu_finite_backend", None) == "cupy"
    assert getattr(exc_info.value, "_statgpu_finite_device", None) == "cuda:1"


def test_cuda_torch_finite_reduction_failure_records_backend_provenance(monkeypatch):
    strided = object()

    class FakeTorchTensor:
        __module__ = "torch"
        layout = strided
        device = types.SimpleNamespace(type="cuda", index=2)

    def fail_isfinite(value):
        raise RuntimeError("synthetic Torch finite-check allocation failure")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(strided=strided, isfinite=fail_isfinite),
    )

    with pytest.raises(RuntimeError, match="synthetic Torch") as exc_info:
        check_finite(FakeTorchTensor(), name="X")

    assert getattr(exc_info.value, "_statgpu_finite_backend", None) == "torch"
    assert getattr(exc_info.value, "_statgpu_finite_device", None) == "cuda:2"


def _exercise_cleanup_device_helper(monkeypatch, helper):
    state = {"current": 0, "entered": [], "cleaned": []}

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
            return False

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        types.SimpleNamespace(cuda=types.SimpleNamespace(Device=FakeDevice)),
    )

    helper(
        lambda: state["cleaned"].append(state["current"]),
        "cuda:1",
    )

    assert state["entered"] == [1]
    assert state["cleaned"] == [1]
    assert state["current"] == 0


def test_cupy_cleanup_helper_enters_executed_device(monkeypatch):
    from statgpu.linear_model.penalized import _gaussian_fit_transaction_contract as contract

    _exercise_cleanup_device_helper(monkeypatch, contract._run_cupy_cleanup_on_device)


def test_no_inference_cupy_cleanup_helper_enters_executed_device(monkeypatch):
    from statgpu.linear_model.penalized import _no_inference_cleanup_contract as contract

    _exercise_cleanup_device_helper(monkeypatch, contract._run_cupy_cleanup_on_device)


def test_linear_cleanup_context_entry_failure_preserves_fit_error(monkeypatch):
    import statgpu.linear_model.wrappers._linear as linear_module
    from statgpu.linear_model import LinearRegression

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(-1.0, 1.0, 6)
    model = LinearRegression(
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
    ).fit(X, y)
    assert model._fitted is True

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
            raise RuntimeError("synthetic cleanup context-entry failure")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        types.SimpleNamespace(cuda=types.SimpleNamespace(Device=FailingDevice)),
    )
    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="cupy")
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, backend=None: FakeArray(np.asarray(value).shape),
    )
    monkeypatch.setattr(
        linear_module,
        "_cupy_asarray_on_device",
        lambda value, target_device, dtype=None: (_ for _ in ()).throw(
            RuntimeError("synthetic response alignment failure")
        ),
    )
    monkeypatch.setattr(
        model, "_cleanup_cuda_memory", lambda: events.append("cleanup")
    )
    monkeypatch.setattr(
        model,
        "_fit_gpu",
        lambda *args, **kwargs: pytest.fail("backend dispatch must not start"),
    )

    with pytest.raises(RuntimeError, match="synthetic response alignment failure"):
        model.fit(X, y)

    assert events == ["cleanup"]
    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._selected_backend_name is None


def test_linear_outer_finite_cleanup_uses_recorded_input_device(monkeypatch):
    from statgpu.linear_model import LinearRegression

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.linspace(-1.0, 1.0, 6)
    model = LinearRegression(
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
    ).fit(X, y)

    # Emulate stale provenance from a prior successful cuda:0 fit. The rejected
    # refit below is a non-finite CuPy input on cuda:1, whose validation
    # temporaries must be cleaned on cuda:1 instead of following this stale state.
    model._selected_backend_name = "cupy"
    model._selected_backend_device = "cuda:0"
    state = {"current": 0, "entered": [], "cleaned": []}

    class FakeCuPyArray:
        __module__ = "cupy"
        device = types.SimpleNamespace(id=1)

    class FakeFiniteReduction:
        def all(self):
            return self

        def item(self):
            return False

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
            return False

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        types.SimpleNamespace(
            isfinite=lambda value: FakeFiniteReduction(),
            cuda=types.SimpleNamespace(Device=FakeDevice),
        ),
    )
    monkeypatch.setattr(
        model,
        "_cleanup_cuda_memory",
        lambda: state["cleaned"].append(state["current"]),
    )

    with pytest.raises(ValueError, match="X must contain only finite values"):
        model.fit(FakeCuPyArray(), y)

    assert state["entered"] == [1]
    assert state["cleaned"] == [1]
    assert state["current"] == 0
    assert model._fitted is False
    assert model.coef_ is None
    assert model._selected_backend_name is None
    assert model._selected_backend_device is None


def test_weighted_reporting_state_substitutes_raw_response_only():
    from statgpu.linear_model._gaussian_inference import GaussianFitState
    from statgpu.linear_model.penalized import _gaussian_fit_transaction_contract as contract

    weighted_y = object()
    raw_y = object()
    resid = object()
    state = GaussianFitState(
        X_design=object(),
        y=weighted_y,
        resid=resid,
        scale=object(),
        nobs=3,
        df_resid=1,
        params=object(),
        backend="cupy",
        device="cuda:1",
    )

    unweighted = contract._reporting_state_for_dispatch(
        state,
        {"X": object(), "y": raw_y, "sample_weight": None},
    )
    assert unweighted is state
    assert unweighted.y is weighted_y

    weighted = contract._reporting_state_for_dispatch(
        state,
        {"X": object(), "y": raw_y, "sample_weight": object()},
    )
    assert weighted is not state
    assert weighted.y is raw_y
    assert weighted.resid is resid
    assert state.y is weighted_y


def test_diagnostic_state_reuses_existing_unweighted_response_snapshot(monkeypatch):
    from statgpu.linear_model.penalized import _gaussian_fit_transaction_contract as contract

    raw_y = np.asarray([1.0, 2.0, 3.0])
    estimator = types.SimpleNamespace(
        _y=raw_y.copy(),
        _resid=np.asarray([0.1, -0.2, 0.3]),
    )
    dispatch_y = object()

    monkeypatch.setattr(
        contract,
        "_to_numpy",
        lambda value: (_ for _ in ()).throw(
            AssertionError("unweighted response must not be transferred twice")
        ),
    )

    contract._store_weighted_diagnostic_state(
        estimator,
        {"X": object(), "y": dispatch_y, "sample_weight": None},
    )

    np.testing.assert_array_equal(estimator._y, raw_y)
    np.testing.assert_array_equal(estimator._raw_resid, estimator._resid)
    assert estimator._sample_weight_fit is None


def test_weighted_diagnostic_state_transfers_weights_but_not_response(monkeypatch):
    from statgpu.linear_model.penalized import _gaussian_fit_transaction_contract as contract

    raw_y = np.asarray([1.0, 2.0, 3.0])
    weights = np.asarray([1.0, 4.0, 9.0])
    raw_resid = np.asarray([0.5, -0.25, 0.1])
    weighted_resid = raw_resid * np.sqrt(weights)
    estimator = types.SimpleNamespace(
        _y=raw_y.copy(),
        _resid=weighted_resid.copy(),
    )
    dispatch_y = object()
    calls = []

    def explicit_to_numpy(value):
        calls.append(value)
        if value is dispatch_y:
            raise AssertionError("weighted response must not be transferred twice")
        assert value is weights
        return value

    monkeypatch.setattr(contract, "_to_numpy", explicit_to_numpy)

    contract._store_weighted_diagnostic_state(
        estimator,
        {"X": object(), "y": dispatch_y, "sample_weight": weights},
    )

    assert calls == [weights]
    np.testing.assert_array_equal(estimator._y, raw_y)
    np.testing.assert_array_equal(estimator._sample_weight_fit, weights)
    np.testing.assert_allclose(estimator._raw_resid, raw_resid)
