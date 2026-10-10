"""Call-local covariance normalization reuse without weakening public validation.

Conversion counts are structural evidence, not wall-clock speedup claims.
Physical-source tests remain separate from real Torch CPU checks.
"""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from test_covariance_torch_inputs import (
    ESTIMATORS,
    INPUT_KINDS,
    QUERIES,
    _assert_fit_matches,
    _assert_numpy_model,
    _assert_unchanged,
    _logical_numpy,
    _model,
    _require_cuda,
    _snapshot,
    _tensor,
)

from statgpu._base import BaseEstimator, refresh_public_finite_validation_contracts
from statgpu._config import Device, _device_manager
from statgpu.covariance import EmpiricalCovariance
from statgpu.covariance import _empirical


@pytest.fixture(autouse=True)
def automatic_global_device(monkeypatch):
    monkeypatch.setattr(_device_manager, "_current_device", Device.AUTO)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def X():
    values = np.random.default_rng(192).normal(size=(45, 3))
    values[:, 1] += 0.45 * values[:, 0]
    return values


def _watch_conversion(monkeypatch, torch, tensor):
    """Count the helper and actual source-shape CPU/cast/NumPy operations."""
    calls = {"helper": [], "cpu": [], "cast": [], "numpy": []}
    original_helper = _empirical._torch_to_covariance_numpy
    original_cpu, original_to = torch.Tensor.cpu, torch.Tensor.to
    original_numpy = torch.Tensor.numpy

    def convert(value):
        calls["helper"].append(value)
        return original_helper(value)

    def cpu(value, *args, **kwargs):
        if value.shape == tensor.shape:
            calls["cpu"].append(value.device)
        return original_cpu(value, *args, **kwargs)

    def cast(value, *args, **kwargs):
        if value.shape == tensor.shape and (
            kwargs.get("dtype") == torch.float64 or torch.float64 in args
        ):
            calls["cast"].append(value.device)
        return original_to(value, *args, **kwargs)

    def numpy(value, *args, **kwargs):
        if value.shape == tensor.shape:
            calls["numpy"].append(value.device)
        return original_numpy(value, *args, **kwargs)

    monkeypatch.setattr(_empirical, "_torch_to_covariance_numpy", convert)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "to", cast)
    monkeypatch.setattr(torch.Tensor, "numpy", numpy)
    return calls


def _assert_one_conversion(calls, tensor, torch):
    assert len(calls["helper"]) == 1 and calls["helper"][0] is tensor
    assert calls["cpu"] == [tensor.device]
    assert calls["cast"] == [torch.device("cpu")]
    assert calls["numpy"] == [torch.device("cpu")]


def _invoke(model, method, tensor, style):
    if style == "keyword":
        return getattr(model, method)(X=tensor)
    return getattr(model, method)(tensor)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("method", ("fit",) + QUERIES)
@pytest.mark.parametrize("style", ["positional", "keyword"])
def test_numpy_bound_public_call_converts_torch_once(
    cls, kind, method, style, X, torch, monkeypatch
):
    tensor = _tensor(torch, X, kind)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    actual = _model(cls)
    if method != "fit":
        actual.fit(values)
        # Queries must retain the fitted NumPy backend after policy changes.
        monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    with monkeypatch.context() as patch:
        calls = _watch_conversion(patch, torch, tensor)
        result = _invoke(actual, method, tensor, style)
        _assert_one_conversion(calls, tensor, torch)
    if method == "fit":
        assert result is actual
        _assert_fit_matches(actual, expected)
    else:
        assert_allclose(result, getattr(expected, method)(values), rtol=1e-12, atol=1e-12)
    _assert_numpy_model(actual)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_global_cpu_reuse_does_not_override_later_auto_refit(
    cls, X, torch, monkeypatch
):
    tensor = _tensor(torch, X, "bfloat16")
    values = _logical_numpy(tensor)
    model = _model(cls, "auto")
    monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    with monkeypatch.context() as patch:
        calls = _watch_conversion(patch, torch, tensor)
        model.fit(tensor)
        _assert_one_conversion(calls, tensor, torch)
    _assert_numpy_model(model)
    monkeypatch.setattr(_device_manager, "_current_device", Device.AUTO)
    model.fit(tensor)
    assert model._backend_name == "torch"
    assert model.covariance_.device == tensor.device
    _assert_fit_matches(model, _model(cls).fit(values), native=True)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("method", ("fit",) + QUERIES)
def test_native_fp8_reuses_one_source_device_widening(
    cls, method, X, torch, monkeypatch
):
    tensor = _tensor(torch, X, "float8_e5m2")
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    model = _model(cls, "auto")
    if method != "fit":
        model.fit(torch.tensor(values))
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    original_to = torch.Tensor.to
    widened = []

    def cast(value, *args, **kwargs):
        if value.dtype == torch.float8_e5m2:
            assert value.device == tensor.device and not value.requires_grad
            widened.append(value.device)
        return original_to(value, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("native Torch input was converted through NumPy")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", cast)
        patch.setattr(_empirical, "_torch_to_covariance_numpy", forbidden)
        result = getattr(model, method)(tensor)
        assert widened == [tensor.device]
    assert model._backend_name == "torch"
    assert model.covariance_.device == tensor.device
    if method == "fit":
        _assert_fit_matches(model, expected, native=True)
    else:
        assert_allclose(result, getattr(expected, method)(values), rtol=5e-6, atol=5e-7)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_reuse_is_call_local_and_clone_has_no_prepared_input(
    cls, X, torch, monkeypatch
):
    clone = pytest.importorskip("sklearn.base").clone
    model = _model(cls).fit(X)
    tensor = _tensor(torch, X, "float32")
    state_keys = set(vars(model))
    for _ in range(2):
        tensor.add_(0.25)
        values = _logical_numpy(tensor)
        expected = model.predict(values)
        with monkeypatch.context() as patch:
            calls = _watch_conversion(patch, torch, tensor)
            actual = model.predict(tensor)
            _assert_one_conversion(calls, tensor, torch)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert set(vars(model)) == state_keys
    copied = clone(model)
    assert not copied._fitted and not hasattr(copied, "covariance_")
    with monkeypatch.context() as patch:
        calls = _watch_conversion(patch, torch, tensor)
        copied.fit(tensor)
        _assert_one_conversion(calls, tensor, torch)
    tensor[0, 0] = float("nan")
    for method in ("fit",) + QUERIES:
        with pytest.raises(ValueError, match="X must contain only finite values"):
            getattr(model, method)(tensor)
    tensor[0, 0] = 0
    assert np.all(np.isfinite(model.predict(tensor)))


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("method", ["fit", "score"])
def test_reused_X_does_not_skip_ignored_y_validation(
    cls, method, X, torch, monkeypatch
):
    tensor = _tensor(torch, X, "bfloat16")
    model = _model(cls).fit(X)
    with monkeypatch.context() as patch:
        calls = _watch_conversion(patch, torch, tensor)
        with pytest.raises(ValueError) as error:
            getattr(model, method)(X=tensor, y=np.array([np.inf]))
        assert str(error.value) == "y must contain only finite values; found NaN or infinite values"
        _assert_one_conversion(calls, tensor, torch)
    # No failed-call normalization survives into a retry with the same object.
    tensor[0, 0] = float("nan")
    with pytest.raises(ValueError, match="X must contain only finite values"):
        getattr(model, method)(tensor, y=np.ones(1))


@pytest.mark.parametrize("representation", ["numpy", "list", "pandas"])
def test_host_input_identity_remains_unchanged(representation, X, monkeypatch):
    value = X if representation == "numpy" else X.tolist()
    if representation == "pandas":
        value = pytest.importorskip("pandas").DataFrame(X)
    original = EmpiricalCovariance._prepare_covariance_input
    received = []

    def prepare(self, X, **kwargs):
        received.append(X)
        return original(self, X, **kwargs)

    monkeypatch.setattr(EmpiricalCovariance, "_prepare_covariance_input", prepare)
    model = EmpiricalCovariance(device="cpu").fit(value)
    for method in QUERIES:
        getattr(model, method)(value)
    assert len(received) == 4 and all(item is value for item in received)


def test_default_and_legacy_validation_only_hooks_keep_input_identity(X):
    class IdentityEstimator(BaseEstimator):
        def fit(self, X, /, y=None, *, sample_weight=None):
            self.received = (X, y, sample_weight)
            return self

        def predict(self, X):
            return X

    class LegacyEstimator(IdentityEstimator):
        def _check_public_input_finite(self, value, **kwargs):
            super()._check_public_input_finite(value, **kwargs)
            # Existing validation-only overrides return None implicitly.

    y, weight = np.ones(len(X)), np.ones(len(X))
    for cls in (IdentityEstimator, LegacyEstimator):
        model = cls(device="cpu").fit(X, y=y, sample_weight=weight)
        assert all(got is want for got, want in zip(model.received, (X, y, weight)))
        assert model.predict(X=X) is X


def test_normalized_handoff_preserves_other_arguments_and_validation(X):
    class NormalizingEstimator(BaseEstimator):
        def _check_public_input_finite(self, value, *, name, method_name):
            super()._check_public_input_finite(value, name=name, method_name=method_name)
            return np.asarray(value) if name == "X" else None

        def fit(self, X, /, y=None, *, sample_weight=None):
            self.received = (X, y, sample_weight)
            return self

        def predict(self, X):
            return X

    values, y, weight = X.tolist(), np.ones(len(X)), np.ones(len(X))
    model = NormalizingEstimator(device="cpu").fit(values, y, sample_weight=weight)
    assert_array_equal(model.received[0], X)
    assert model.received[1] is y and model.received[2] is weight
    assert_array_equal(model.predict(X=values), X)
    with pytest.raises(ValueError, match="y must contain only finite values"):
        model.fit(values, np.array([np.nan]), sample_weight=weight)


def test_all_non_covariance_consumers_keep_default_hook_and_installer_identity():
    descendants, pending = set(), list(BaseEstimator.__subclasses__())
    while pending:
        cls = pending.pop()
        if cls in descendants:
            continue
        descendants.add(cls)
        pending.extend(cls.__subclasses__())
    concrete = {
        cls for cls in descendants
        if cls.__module__.startswith("statgpu.") and not inspect.isabstract(cls)
        and not issubclass(cls, EmpiricalCovariance)
    }
    assert len(concrete) >= 64
    wrappers = {}
    for cls in concrete:
        assert cls._check_public_input_finite is BaseEstimator._check_public_input_finite
        for name in dir(cls):
            method = getattr(cls, name)
            if getattr(method, "__statgpu_finite_validation__", False):
                wrappers[cls, name] = method
    refresh_public_finite_validation_contracts()
    refresh_public_finite_validation_contracts()
    assert all(getattr(cls, name) is original for (cls, name), original in wrappers.items())


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("index", [0, 1], ids=["single_gpu", "nondefault_gpu"])
@pytest.mark.parametrize("kind", ["float32", "bfloat16", "float8_e5m2"])
def test_physical_cuda_source_converts_once_per_public_call(
    cls, index, kind, X, torch, monkeypatch
):
    _require_cuda(torch, index + 1)
    # No conversion/kernel skips on the physical boundary, including FP8.
    tensor = torch.tensor(X, device=f"cuda:{index}", dtype=torch.float64).to(
        dtype=getattr(torch, kind)
    )
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    model = _model(cls)
    for method in ("fit",) + QUERIES:
        with monkeypatch.context() as patch:
            calls = _watch_conversion(patch, torch, tensor)
            result = getattr(model, method)(tensor)
            _assert_one_conversion(calls, tensor, torch)
        if method == "fit":
            _assert_fit_matches(model, expected)
        else:
            assert_allclose(result, getattr(expected, method)(values), rtol=1e-12, atol=1e-12)
    _assert_unchanged(tensor, before)
