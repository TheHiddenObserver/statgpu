"""Covariance estimator device policy, array ownership, and numerical regression.

Torch CPU tests execute real tensors. Unavailable-backend tests mock only
availability, never pretend to execute a GPU. The physical CUDA tests skip when
the corresponding runtime (or second GPU) is unavailable.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu._config import Device, _device_manager
from statgpu.backends import _to_numpy
from statgpu.covariance import (
    OAS,
    EmpiricalCovariance,
    GraphicalLasso,
    GraphicalLassoCV,
    LedoitWolf,
    MinCovDet,
    ShrunkCovariance,
)

ESTIMATORS = (
    EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance,
    GraphicalLasso, GraphicalLassoCV, MinCovDet,
)


@pytest.fixture(autouse=True)
def automatic_global_device(monkeypatch):
    monkeypatch.setattr(_device_manager, "_current_device", Device.AUTO)


@pytest.fixture
def X():
    data = np.random.default_rng(192).normal(size=(45, 3))
    data[:, 1] += 0.45 * data[:, 0]
    return data


def _model(cls, device="auto"):
    params = {"device": device}
    if cls is GraphicalLasso:
        params.update(alpha=0.08, max_iter=100, tol=1e-7)
    if cls is GraphicalLassoCV:
        params.update(alphas=[0.03, 0.12], cv=3, random_state=7, max_iter=100, tol=1e-7)
    if cls is MinCovDet:
        params.update(support_fraction=0.75, random_state=7)
    return cls(**params)


def _array_attributes(model):
    names = ("covariance_", "precision_", "location_", "support_",
             "raw_covariance_", "raw_location_", "dist_")
    return [getattr(model, name) for name in names if hasattr(model, name)]


def _assert_numpy_model(model):
    assert model._backend_name == "numpy"
    assert all(isinstance(value, np.ndarray) for value in _array_attributes(model))


def _assert_parity(actual, expected, data):
    for got, want in zip(_array_attributes(actual), _array_attributes(expected)):
        assert_allclose(_to_numpy(got), _to_numpy(want), rtol=5e-6, atol=5e-7)
    assert_allclose(actual.score(data), expected.score(data), rtol=5e-6, atol=5e-7)
    assert_allclose(actual.mahalanobis(data), expected.mahalanobis(data), rtol=5e-6, atol=5e-7)
    assert isinstance(actual.predict(data), np.ndarray)
    if isinstance(actual, GraphicalLassoCV):
        assert actual.alpha_ == expected.alpha_
        assert_allclose([r["scores"] for r in actual.cv_results_],
                        [r["scores"] for r in expected.cv_results_], rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_explicit_cpu_converts_real_torch_input_to_numpy(cls, X):
    torch = pytest.importorskip("torch")
    expected = _model(cls, "cpu").fit(X)
    # A differentiable CPU tensor is still ordinary estimator input, not an
    # authorization to change its requested compute library.
    tensor = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    actual = _model(cls, "cpu").fit(tensor)
    _assert_numpy_model(actual)
    _assert_parity(actual, expected, tensor)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("device", ["torch", "cuda"])
@pytest.mark.parametrize("tensor_input", [False, True], ids=["numpy", "torch_cpu"])
def test_unavailable_explicit_gpu_raises_before_covariance_work(
    cls, device, tensor_input, X, monkeypatch
):
    torch = pytest.importorskip("torch")
    from statgpu.backends._cupy import CuPyBackend
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(CuPyBackend, "is_available", lambda self: False)
    data = torch.as_tensor(X, dtype=torch.float64) if tensor_input else X
    requirement = "PyTorch.*CUDA" if device == "torch" else "CuPy.*CUDA"
    model = _model(cls, device)
    with pytest.raises(RuntimeError, match=requirement):
        model.fit(data)
    assert not model._fitted
    assert not hasattr(model, "covariance_")


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_true_auto_preserves_real_torch_cpu_numerics(cls, X):
    torch = pytest.importorskip("torch")
    expected = _model(cls, "cpu").fit(X)
    actual = _model(cls).fit(torch.as_tensor(X, dtype=torch.float64))
    assert actual._backend_name == "torch"
    for value in _array_attributes(actual):
        assert isinstance(value, torch.Tensor)
        assert value.device.type == "cpu"
    _assert_parity(actual, expected, X)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_true_auto_numpy_falls_back_when_no_gpu_is_available(cls, X, monkeypatch):
    monkeypatch.setattr(_device_manager, "_cupy_available", False)
    monkeypatch.setattr(_device_manager, "_torch_available", False)
    monkeypatch.setattr(_device_manager, "_cuda_available", False)
    _assert_numpy_model(_model(cls).fit(X))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_global_cpu_overrides_native_torch_input_for_auto(cls, X, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    _assert_numpy_model(_model(cls).fit(torch.as_tensor(X, dtype=torch.float64)))


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("device", [Device.TORCH, Device.CUDA])
def test_global_explicit_gpu_is_strict_but_estimator_cpu_overrides_it(cls, device, X, monkeypatch):
    torch = pytest.importorskip("torch")
    from statgpu.backends._cupy import CuPyBackend
    monkeypatch.setattr(_device_manager, "_current_device", device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(CuPyBackend, "is_available", lambda self: False)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="CUDA"):
        _model(cls).fit(tensor)
    _assert_numpy_model(_model(cls, "cpu").fit(tensor))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_score_and_distances_keep_fitted_backend_after_global_change(cls, X, monkeypatch):
    torch = pytest.importorskip("torch")
    model = _model(cls).fit(torch.as_tensor(X, dtype=torch.float64))
    expected_score = model.score(X)
    expected_distance = model.mahalanobis(X)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
    calls = []
    original = model._prepare_covariance_input

    def observe(data, **kwargs):
        result = original(data, **kwargs)
        calls.append((result[0], result[2].device.type))
        return result

    monkeypatch.setattr(model, "_prepare_covariance_input", observe)
    assert_allclose(model.score(X), expected_score, rtol=0, atol=0)
    assert_allclose(model.predict(X), expected_distance, rtol=0, atol=0)
    assert calls == [("torch", "cpu"), ("torch", "cpu")]
    assert model._backend_name == "torch"


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_clone_and_set_params_preserve_requested_device_and_reset_fit(cls, X):
    torch = pytest.importorskip("torch")
    from sklearn.base import clone
    model = _model(cls, Device.CPU).fit(torch.as_tensor(X, dtype=torch.float64))
    copied = clone(model)
    assert copied.get_params(deep=False)["device"] is Device.CPU
    assert not copied._fitted
    assert copied.set_params(device="auto") is copied
    copied.fit(torch.as_tensor(X, dtype=torch.float64))
    assert copied._backend_name == "torch"
    copied.set_params(device="cpu")
    assert copied.get_params(deep=False)["device"] == "cpu"
    assert not copied._fitted
    with pytest.raises(RuntimeError, match="fitted"):
        copied.score(X)
    _assert_numpy_model(copied.fit(torch.as_tensor(X, dtype=torch.float64)))


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("device", ["cuda:0", "torch:0"])
def test_indexed_device_strings_are_not_estimator_device_values(cls, device):
    with pytest.raises(ValueError, match=device):
        _model(cls, device)


@pytest.mark.parametrize("tensor_input", [False, True])
def test_cv_all_folds_and_final_refit_use_initial_backend(tensor_input, X, monkeypatch):
    torch = pytest.importorskip("torch")
    data = torch.as_tensor(X, dtype=torch.float64) if tensor_input else X
    device = "auto" if tensor_input else "cpu"
    expected = _model(GraphicalLassoCV, device).fit(data)
    calls = []
    original = GraphicalLasso._fit_prepared

    def observe(self, backend_name, xp, prepared):
        calls.append((backend_name, len(prepared), type(prepared)))
        # Configuration changes cannot switch a CV run midway through scoring
        # and final refit after its input backend was already selected.
        monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
        return original(self, backend_name, xp, prepared)

    monkeypatch.setattr(GraphicalLasso, "_fit_prepared", observe)
    actual = _model(GraphicalLassoCV, device).fit(data)
    backend_name = "torch" if tensor_input else "numpy"
    array_type = torch.Tensor if tensor_input else np.ndarray
    assert calls == [(backend_name, 30, array_type)] * 6 + [(backend_name, 45, array_type)]
    _assert_parity(actual, expected, data)


def _cuda_runtime(device, minimum_devices=1):
    if device == "torch":
        xp = pytest.importorskip("torch")
        if not xp.cuda.is_available() or xp.cuda.device_count() < minimum_devices:
            pytest.skip(f"physical Torch CUDA test requires {minimum_devices} CUDA device(s)")
    else:
        xp = pytest.importorskip("cupy")
        try:
            count = xp.cuda.runtime.getDeviceCount()
        except (xp.cuda.runtime.CUDARuntimeError, RuntimeError, OSError) as exc:
            pytest.skip(f"physical CuPy CUDA runtime unavailable: {exc}")
        if count < minimum_devices:
            pytest.skip(f"physical CuPy test requires {minimum_devices} CUDA device(s)")
    return xp


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("device", ["torch", "cuda"])
@pytest.mark.parametrize("input_kind", ["numpy", "torch_cpu", "native_gpu"])
def test_physical_cuda_routing_and_numerical_parity(cls, device, input_kind, X):
    xp = _cuda_runtime(device)
    if input_kind == "torch_cpu":
        torch = pytest.importorskip("torch")
        data = torch.as_tensor(X, dtype=torch.float64, device="cpu")
    elif input_kind == "native_gpu":
        data = xp.asarray(X, dtype=xp.float64, device="cuda") if device == "torch" else xp.asarray(X)
    else:
        data = X
    expected = _model(cls, "cpu").fit(X)
    actual = _model(cls, device).fit(data)
    assert actual._backend_name == ("torch" if device == "torch" else "cupy")
    for value in _array_attributes(actual):
        if device == "torch":
            assert isinstance(value, xp.Tensor) and value.is_cuda
        else:
            assert isinstance(value, xp.ndarray)
    _assert_parity(actual, expected, X)
    # Evaluation must not make CPU tensors select Torch CPU after a GPU fit.
    torch = pytest.importorskip("torch")
    _, _, prepared = actual._prepare_covariance_input(torch.as_tensor(X), fitted=True)
    if device == "torch":
        assert prepared.device == actual.covariance_.device
    else:
        assert prepared.device.id == actual.covariance_.device.id


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("target", ["torch", "cuda"])
def test_physical_cross_library_gpu_conversion_obeys_requested_backend(cls, target, X):
    torch = _cuda_runtime("torch")
    cp = _cuda_runtime("cuda")
    data = cp.asarray(X) if target == "torch" else torch.as_tensor(X, device="cuda")
    actual = _model(cls, target).fit(data)
    assert actual._backend_name == ("torch" if target == "torch" else "cupy")
    _assert_parity(actual, _model(cls, "cpu").fit(X), X)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("device", ["torch", "cuda"])
def test_physical_nondefault_device_is_preserved_for_fit_cv_and_score(cls, device, X):
    xp = _cuda_runtime(device, minimum_devices=2)
    with xp.cuda.device(1) if device == "torch" else xp.cuda.Device(1):
        data = xp.asarray(X, dtype=xp.float64, device="cuda:1") if device == "torch" else xp.asarray(X)
    with xp.cuda.device(0) if device == "torch" else xp.cuda.Device(0):
        actual = _model(cls, device).fit(data)
        for value in _array_attributes(actual):
            assert (value.device.index if device == "torch" else value.device.id) == 1
        _assert_parity(actual, _model(cls, "cpu").fit(X), X)
        test_data = xp.asarray(X, dtype=xp.float64, device="cuda:0") if device == "torch" else xp.asarray(X)
        _, _, prepared = actual._prepare_covariance_input(test_data, fitted=True)
        assert (prepared.device.index if device == "torch" else prepared.device.id) == 1
        assert np.isfinite(actual.score(test_data))


def test_graphical_cpu_matches_aligned_sklearn_reference(X):
    from sklearn.covariance import GraphicalLasso as Reference
    actual = _model(GraphicalLasso, "cpu").fit(X)
    expected = Reference(alpha=0.08, max_iter=100, tol=1e-7).fit(X)
    assert_allclose(actual.covariance_, expected.covariance_, rtol=2e-5, atol=2e-6)
    assert_allclose(actual.precision_, expected.precision_, rtol=2e-5, atol=2e-6)


def test_cv_rejects_a_fold_with_only_one_training_observation():
    with pytest.raises(ValueError, match="at least 2 samples"):
        GraphicalLassoCV(alphas=[0.0], cv=2, device="cpu").fit(np.array([[0.0], [1.0]]))


def test_fitted_torch_exact_index_conversion_routing_without_gpu_execution(X, monkeypatch):
    """A call sentinel checks routing only; no CUDA execution is simulated."""
    from types import SimpleNamespace

    import statgpu.covariance._empirical as module
    model = EmpiricalCovariance(device="torch")
    ref = SimpleNamespace(is_cuda=True, device="cuda:3")
    model.covariance_ = ref
    model._backend_name = "torch"
    xp = SimpleNamespace(float64="float64")
    backend = SimpleNamespace(xp=xp, is_available=lambda: True)
    monkeypatch.setattr(module, "get_backend", lambda **kwargs: backend)
    calls = []
    marker = object()

    def to_torch(data, device):
        calls.append(device)
        assert data is X
        return marker

    def cast(data, **kwargs):
        assert data is marker and kwargs["ref_arr"] is ref
        return marker

    monkeypatch.setattr(model, "_to_torch", to_torch)
    monkeypatch.setattr(module, "xp_asarray", cast)
    assert model._prepare_covariance_input(X, fitted=True)[2] is marker
    assert calls == ["cuda:3"]


def test_fitted_cupy_conversion_enters_fitted_index_before_allocating(X, monkeypatch):
    """Context sentinels check ordering only, not physical GPU allocation."""
    from types import SimpleNamespace

    import statgpu.covariance._empirical as module
    calls = []

    class DeviceContext:
        id = 3

        def __enter__(self):
            calls.append("enter:3")

        def __exit__(self, *args):
            calls.append("exit:3")

    model = EmpiricalCovariance(device="cuda")
    model.covariance_ = SimpleNamespace(device=DeviceContext())
    model._backend_name = "cupy"
    xp = SimpleNamespace(float64="float64")
    backend = SimpleNamespace(xp=xp, is_available=lambda: True)
    monkeypatch.setattr(module, "get_backend", lambda **kwargs: backend)
    marker = object()

    def convert(data, **kwargs):
        assert data is X and calls == ["enter:3"]
        calls.append("convert")
        return marker

    def cast(data, target, **kwargs):
        assert data is marker and target == 3
        calls.append("cast:3")
        return marker

    monkeypatch.setattr(model, "_to_array", convert)
    monkeypatch.setattr(module, "_cupy_asarray_on_device", cast)
    assert model._prepare_covariance_input(X, fitted=True)[2] is marker
    assert calls == ["enter:3", "convert", "cast:3", "exit:3"]
