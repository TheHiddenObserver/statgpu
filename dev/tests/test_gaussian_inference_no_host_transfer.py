"""Behavioral guard for the issue #127 numerical/reporting boundary."""

from __future__ import annotations

import numpy as np
import pytest


def test_pglm_torch_numerics_finish_before_any_reporting_snapshot(monkeypatch):
    torch = pytest.importorskip("torch")

    import statgpu.linear_model._gaussian_inference as gi
    import statgpu.linear_model.penalized._base as pglm_base
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    phase = {"reporting_allowed": False, "gi_snapshots": 0, "pglm_snapshots": 0}
    real_gi_to_numpy = gi._to_numpy
    real_pglm_to_numpy = pglm_base._to_numpy

    def guarded_gi_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError(
                "Gaussian numerical inference attempted a host snapshot before "
                "reference-distribution work completed"
            )
        phase["gi_snapshots"] += 1
        return real_gi_to_numpy(value)

    def guarded_pglm_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError(
                "PGLM reporting state attempted a host snapshot before numerical "
                "Gaussian inference completed"
            )
        phase["pglm_snapshots"] += 1
        return real_pglm_to_numpy(value)

    def fake_reference_inference(
        statistic_abs,
        *,
        distribution,
        alpha,
        backend,
        xp,
        df=None,
        device=None,
    ):
        assert distribution == "t"
        assert backend == "torch"
        assert str(device) == "cpu"
        assert isinstance(statistic_abs, torch.Tensor)
        assert statistic_abs.device.type == "cpu"
        assert alpha == pytest.approx(0.05)
        assert df is not None and int(df) > 0
        phase["reporting_allowed"] = True
        return (
            torch.full_like(statistic_abs, 0.25),
            torch.tensor(2.0, dtype=torch.float64),
        )

    monkeypatch.setattr(gi, "_to_numpy", guarded_gi_to_numpy)
    monkeypatch.setattr(pglm_base, "_to_numpy", guarded_pglm_to_numpy)
    monkeypatch.setattr(
        gi, "two_sided_reference_inference", fake_reference_inference
    )

    X = np.asarray(
        [[-2.0, 0.5], [-1.0, 1.0], [0.0, 1.5], [1.0, 2.0], [2.0, 2.5], [3.0, 3.0]],
        dtype=np.float64,
    )
    coef = np.asarray([0.75, -0.4])
    intercept = 1.2
    y = intercept + X @ coef + np.asarray([0.2, -0.1, 0.15, -0.25, 0.05, -0.05])

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        device="cpu",
        compute_inference=True,
    )
    model._penalty = model._resolve_penalty()
    model._selected_backend_name = "torch"
    model._selected_backend_device = "cpu"
    model._native_fit_coef = torch.as_tensor(coef, dtype=torch.float64)
    model._native_fit_intercept = torch.tensor(intercept, dtype=torch.float64)
    # Reporting fields are intentionally absent: the GPU router must consume
    # the native fit state and only populate them after numerical inference.
    model.coef_ = None
    model.intercept_ = None

    model._compute_post_fit_gaussian_inference(X, y)

    assert phase["reporting_allowed"] is True
    assert phase["gi_snapshots"] == 5  # params, bse, statistic, pvalue, CI
    assert phase["pglm_snapshots"] == 5  # design, y, residual, params, scale
    assert model._inference_result.metadata["numerical_backend"] == "torch"
    assert model._inference_result.metadata["reporting_boundary"] == "post_numerical_inference"


def test_cupy_device_helper_explicitly_copies_cross_device_native_arrays(monkeypatch):
    import sys
    import types

    from statgpu.backends._utils import _cupy_asarray_on_device

    state = {"current": 0, "copies": 0}

    class FakeArray:
        def __init__(self, device, dtype=np.float64):
            self.device = types.SimpleNamespace(id=int(device))
            self.dtype = np.dtype(dtype)

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    def fake_copy(value):
        state["copies"] += 1
        return FakeArray(state["current"], value.dtype)

    def fake_asarray(value, dtype=None):
        # Deliberately emulate a no-copy asarray for an existing same-dtype
        # native array. The helper must explicitly copy before reaching here.
        if isinstance(value, FakeArray) and (dtype is None or np.dtype(dtype) == value.dtype):
            return value
        return FakeArray(state["current"], dtype or np.float64)

    fake_cupy = types.SimpleNamespace(
        ndarray=FakeArray,
        cuda=types.SimpleNamespace(Device=FakeDevice),
        copy=fake_copy,
        asarray=fake_asarray,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    source = FakeArray(0)
    moved = _cupy_asarray_on_device(source, 1, dtype=np.float64)
    assert moved.device.id == 1
    assert source.device.id == 0
    assert state == {"current": 0, "copies": 1}

    same = _cupy_asarray_on_device(moved, 1, dtype=np.float64)
    assert same is moved
    assert state == {"current": 0, "copies": 1}


def test_torch_l2_fit_defers_parameter_reporting_until_gaussian_inference(monkeypatch):
    torch = pytest.importorskip("torch")

    import statgpu.linear_model._gaussian_inference as gi
    import statgpu.linear_model.penalized._base as pglm_base
    import statgpu.linear_model.penalized._fit_mixin as fit_mixin
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    phase = {"reporting_allowed": False}
    real_fit_to_numpy = fit_mixin._to_numpy

    def guarded_fit_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError("fit parameters crossed to host before Gaussian inference")
        return real_fit_to_numpy(value)

    def fake_reference_inference(
        statistic_abs, *, distribution, alpha, backend, xp, df=None, device=None
    ):
        assert backend == "torch"
        phase["reporting_allowed"] = True
        return torch.full_like(statistic_abs, 0.25), torch.tensor(2.0, dtype=torch.float64)

    monkeypatch.setattr(fit_mixin, "_to_numpy", guarded_fit_to_numpy)
    monkeypatch.setattr(gi, "two_sided_reference_inference", fake_reference_inference)

    X = torch.tensor(
        [[-2.0, 0.5], [-1.0, 1.1], [0.0, 1.4], [1.0, 2.2], [2.0, 2.4], [3.0, 3.1]],
        dtype=torch.float64,
    )
    y = 0.7 * X[:, 0] - 0.2 * X[:, 1] + torch.tensor(
        [0.2, -0.1, 0.15, -0.25, 0.05, -0.05], dtype=torch.float64
    )

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        solver="newton",
    )
    model._penalty = model._resolve_penalty()
    model._loss = model._resolve_loss()
    model._nobs = int(X.shape[0])
    model._selected_backend_name = "torch"
    model._selected_backend_device = "cpu"

    model._fit_loss_backend(X, y, None, "newton", "torch")
    assert model.coef_ is None
    assert model._params is None
    assert isinstance(model._native_fit_coef, torch.Tensor)
    assert model._native_fit_coef.device.type == "cpu"

    model._compute_post_fit_gaussian_inference(X, y)
    assert phase["reporting_allowed"] is True
    assert isinstance(model.coef_, np.ndarray)
    assert model._native_fit_coef is None
    assert model._native_fit_intercept is None


def test_gpu_cleanup_is_called_after_post_fit_inference(monkeypatch):
    import types

    pytest.importorskip("torch")
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
    )

    monkeypatch.setattr(model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="torch"))
    monkeypatch.setattr(model, "_auto_backend_override", lambda backend_name, X: backend_name)
    monkeypatch.setattr(model, "_select_solver", lambda loss, backend_name=None, X=None: "newton")

    def fake_fit_torch(X, y, sample_weight=None):
        events.append("fit")
        model._native_fit_coef = X.new_zeros(X.shape[1])
        model._native_fit_intercept = None
        model.coef_ = None
        model.intercept_ = None
        model._params = None
        model._df_resid = int(X.shape[0] - X.shape[1])

    monkeypatch.setattr(model, "_fit_torch", fake_fit_torch)
    monkeypatch.setattr(
        model,
        "_compute_post_fit_gaussian_inference",
        lambda X, y, sample_weight=None: events.append("inference"),
    )
    monkeypatch.setattr(model, "_cleanup_torch_memory", lambda: events.append("cleanup"))

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    model.fit(X, y)

    assert events[-2:] == ["inference", "cleanup"]


def test_gpu_cleanup_runs_when_post_fit_inference_raises(monkeypatch):
    import types

    torch = pytest.importorskip("torch")
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
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

    def fake_fit_torch(X, y, sample_weight=None):
        events.append("fit")
        model._native_fit_coef = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
        model._native_fit_intercept = None
        model.coef_ = None
        model.intercept_ = None
        model._params = None
        model._df_resid = int(X.shape[0] - X.shape[1])

    def failing_inference(X, y, sample_weight=None):
        events.append("inference")
        raise RuntimeError("synthetic post-fit inference failure")

    monkeypatch.setattr(model, "_fit_torch", fake_fit_torch)
    monkeypatch.setattr(
        model, "_compute_post_fit_gaussian_inference", failing_inference
    )
    monkeypatch.setattr(
        model, "_cleanup_torch_memory", lambda: events.append("cleanup")
    )

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    # Simulate a failed refit of an estimator that was previously successful.
    model._fitted = True
    with pytest.raises(RuntimeError, match="synthetic post-fit inference failure"):
        model.fit(X, y)

    assert events[-2:] == ["inference", "cleanup"]
    assert model._native_fit_coef is None
    assert model._native_fit_intercept is None
    assert model._fitted is False


def test_gpu_cleanup_runs_when_backend_fit_itself_raises(monkeypatch):
    import types

    torch = pytest.importorskip("torch")
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []
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
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="torch")
    )
    monkeypatch.setattr(
        model, "_auto_backend_override", lambda backend_name, X: backend_name
    )
    monkeypatch.setattr(
        model,
        "_select_solver",
        lambda loss, backend_name=None, X=None: "exact",
    )

    def failing_fit_torch(X, y, sample_weight=None):
        events.append("fit")
        model._native_fit_coef = torch.zeros(
            X.shape[1], dtype=X.dtype, device=X.device
        )
        raise RuntimeError("synthetic backend fit/inference failure")

    monkeypatch.setattr(model, "_fit_torch", failing_fit_torch)
    monkeypatch.setattr(
        model,
        "_compute_post_fit_gaussian_inference",
        lambda X, y, sample_weight=None: events.append("post-fit-inference"),
    )
    monkeypatch.setattr(
        model, "_cleanup_torch_memory", lambda: events.append("cleanup")
    )

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    with pytest.raises(RuntimeError, match="synthetic backend fit/inference failure"):
        model.fit(X, y)

    assert events == ["fit", "cleanup"]
    assert model._native_fit_coef is None
    assert model._native_fit_intercept is None
    assert model._fitted is False


def test_linear_regression_cupy_fit_aligns_y_and_weights_to_x_device(monkeypatch):
    import sys
    import types

    import statgpu.linear_model.wrappers._linear as linear_module
    from statgpu._config import Device
    from statgpu.linear_model import LinearRegression

    state = {"current": 0, "helper_targets": [], "fit_called": False}

    class FakeArray:
        def __init__(self, shape, device):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.device = types.SimpleNamespace(id=int(device))

        def get(self):
            return np.zeros(self.shape, dtype=np.float64)

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    fake_cupy = types.SimpleNamespace(
        cuda=types.SimpleNamespace(Device=FakeDevice)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    X_token = object()
    y_token = object()
    X_native = FakeArray((6, 2), 1)
    y_other_device = FakeArray((6,), 0)
    weights = np.ones(6, dtype=np.float64)

    model = LinearRegression(device="cpu", compute_inference=False)
    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="cupy")
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, backend=None: (
            X_native if value is X_token else y_other_device if value is y_token else value
        ),
    )
    monkeypatch.setattr(model, "_get_compute_device", lambda: Device.CUDA)

    def fake_transfer(value, target_device, dtype=None):
        state["helper_targets"].append(int(target_device))
        shape = getattr(value, "shape", (6,))
        return FakeArray(shape, target_device)

    monkeypatch.setattr(
        linear_module, "_cupy_asarray_on_device", fake_transfer
    )

    def fake_fit_gpu(X, y, sample_weight=None):
        assert X.device.id == 1
        assert y.device.id == 1
        assert sample_weight.device.id == 1
        state["fit_called"] = True

    monkeypatch.setattr(model, "_fit_gpu", fake_fit_gpu)
    model.fit(X_token, y_token, sample_weight=weights)

    assert state["fit_called"] is True
    assert state["helper_targets"] == [1, 1]
    assert state["current"] == 0
    assert model._selected_backend_device == "cuda:1"


def test_pglm_post_fit_inference_reuses_converted_fit_arrays(monkeypatch):
    import types

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

    fitted = {}

    def fake_fit_torch(X, y, sample_weight=None):
        fitted["X"] = X
        fitted["y"] = y
        model._native_fit_coef = torch.zeros(
            X.shape[1], dtype=X.dtype, device=X.device
        )
        model._native_fit_intercept = None
        model.coef_ = None
        model.intercept_ = None
        model._params = None
        model._df_resid = int(X.shape[0] - X.shape[1])

    def assert_reused_arrays(X, y, sample_weight=None):
        assert X is fitted["X"]
        assert y is fitted["y"]

    monkeypatch.setattr(model, "_fit_torch", fake_fit_torch)
    monkeypatch.setattr(
        model, "_compute_post_fit_gaussian_inference", assert_reused_arrays
    )

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    model.fit(X, y)

    assert model._fitted is True


def test_pglm_conversion_failure_invalidates_refit_and_runs_cleanup(monkeypatch):
    import types

    pytest.importorskip("torch")
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []
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

    real_to_array = model._to_array
    calls = {"count": 0}

    def failing_to_array(value, device=None, backend=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic backend conversion failure")
        return real_to_array(value, device=device, backend=backend)

    monkeypatch.setattr(model, "_to_array", failing_to_array)
    monkeypatch.setattr(
        model, "_cleanup_torch_memory", lambda: events.append("cleanup")
    )
    monkeypatch.setattr(
        model, "_fit_torch", lambda *args, **kwargs: events.append("fit")
    )

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    with pytest.raises(RuntimeError, match="synthetic backend conversion failure"):
        model.fit(X, y)

    assert events == ["cleanup"]
    assert model._native_fit_coef is None
    assert model._native_fit_intercept is None
    assert model._fitted is False
