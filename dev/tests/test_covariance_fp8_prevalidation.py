"""Covariance normalization must precede the public finite-input guard.

The CPU emulator reproduces a missing raw FP8 ``isfinite`` kernel and proves
ordering only. Physical CUDA tests are separate, unpatched-kernel evidence;
they skip for missing hardware, never for an FP8 conversion/kernel failure.
The existing physical matrix in test_covariance_torch_inputs stays unchanged.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from test_covariance_torch_inputs import (
    ESTIMATORS,
    QUERIES,
    _assert_fit_matches,
    _assert_numpy_model,
    _assert_queries_match,
    _assert_unchanged,
    _attributes,
    _logical_numpy,
    _model,
    _require_cuda,
    _snapshot,
    _tensor,
)

from statgpu._base import BaseEstimator
from statgpu._config import Device, _device_manager
from statgpu.backends._validation import check_finite
from statgpu.covariance import EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV


@pytest.fixture(autouse=True)
def automatic_global_device(monkeypatch):
    monkeypatch.setattr(_device_manager, "_current_device", Device.AUTO)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def X():
    data = np.random.default_rng(192).normal(size=(45, 3))
    data[:, 1] += 0.45 * data[:, 0]
    return data


@pytest.fixture
def missing_fp8_isfinite(torch, monkeypatch):
    """Emulate only the missing kernel; every other Torch operation is real."""
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("CPU ordering emulator needs Torch float8_e5m2")
    original = torch.isfinite
    calls = []

    def isfinite(value, *args, **kwargs):
        calls.append((value.dtype, value.device))
        if value.dtype == torch.float8_e5m2:
            raise NotImplementedError('emulated "isfinite" missing for Float8_e5m2')
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch, "isfinite", isfinite)
    return calls


def _policy(policy, monkeypatch):
    if policy == "global_cpu":
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    return "cpu" if policy == "explicit_cpu" else "auto"


def _assert_native_model(model, torch, device):
    assert model._backend_name == "torch"
    for name, value in _attributes(model).items():
        assert isinstance(value, torch.Tensor) and value.device == device
        assert value.dtype == (torch.bool if name == "support_" else torch.float64)


def _fitted_snapshot(model):
    return {name: (value, value.clone() if hasattr(value, "clone") else value.copy())
            for name, value in _attributes(model).items()}


def _assert_fitted_unchanged(model, before, torch):
    for name, (original, values) in before.items():
        actual = getattr(model, name)
        assert actual is original
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, values, rtol=0, atol=0)
        else:
            assert_array_equal(actual, values)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("policy", ["explicit_cpu", "global_cpu", "native_auto"])
def test_missing_fp8_kernel_fit_normalizes_before_public_guard(
    cls, policy, X, torch, monkeypatch, missing_fp8_isfinite
):
    tensor = _tensor(torch, X, "float8_e5m2")
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    values.setflags(write=False)
    expected = _model(cls).fit(values)
    device = _policy(policy, monkeypatch)
    native = policy == "native_auto"
    actual = _model(cls, device)
    checked = _observe_native_guard(actual, torch, monkeypatch, tensor.device) if native else []
    actual.fit(X=tensor)
    if native:
        assert checked
        _assert_native_model(actual, torch, tensor.device)
        assert missing_fp8_isfinite[0] == (torch.float64, tensor.device)
    else:
        _assert_numpy_model(actual)
        assert missing_fp8_isfinite == []
    _assert_fit_matches(actual, expected, native=native)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
    _assert_queries_match(actual, expected, tensor, values, native=native)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("method", QUERIES)
def test_missing_fp8_kernel_numpy_trained_query_normalizes_before_guard(
    cls, method, X, torch, monkeypatch, missing_fp8_isfinite
):
    model = _model(cls).fit(X)
    tensor = _tensor(torch, X[::3] + 0.2, "float8_e5m2")
    before = _snapshot(tensor)
    fitted = _fitted_snapshot(model)
    expected = getattr(model, method)(_logical_numpy(tensor))
    monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    result = getattr(model, method)(X=tensor)
    assert isinstance(result, float if method == "score" else np.ndarray)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert missing_fp8_isfinite == []
    _assert_numpy_model(model)
    _assert_fitted_unchanged(model, fitted, torch)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("policy", ["explicit_cpu", "native_auto"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_missing_fp8_kernel_nonfinite_retains_public_error_and_ownership(
    cls, policy, bad_value, X, torch, monkeypatch, missing_fp8_isfinite
):
    data = X.copy()
    data[1, 0] = bad_value
    tensor = _tensor(torch, data, "float8_e5m2")
    before = _snapshot(tensor)
    device = _policy(policy, monkeypatch)
    model = _model(cls, device)
    message = "X must contain only finite values; found NaN or infinite values"
    with pytest.raises(ValueError) as caught:
        model.fit(tensor)
    assert str(caught.value) == message
    assert getattr(caught.value, "_statgpu_finite_backend", None) is None
    assert not model._fitted and not hasattr(model, "covariance_")
    good = _tensor(torch, X, "float8_e5m2")
    model.fit(good)
    fitted = _fitted_snapshot(model)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
    for method in QUERIES:
        with pytest.raises(ValueError) as caught:
            getattr(model, method)(tensor)
        assert str(caught.value) == message
        _assert_fitted_unchanged(model, fitted, torch)
    # Isolate query prevalidation from any earlier Torch-input fit.
    numpy_trained = _model(cls).fit(X)
    numpy_fitted = _fitted_snapshot(numpy_trained)
    monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    for method in QUERIES:
        with pytest.raises(ValueError) as caught:
            getattr(numpy_trained, method)(tensor)
        assert str(caught.value) == message
        _assert_numpy_model(numpy_trained)
        _assert_fitted_unchanged(numpy_trained, numpy_fitted, torch)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("kind", ["bfloat16", "requires_grad", "noncontiguous", "lazy_negative"])
def test_numpy_guard_conversion_orders_detach_cpu_cast_and_resolve(
    kind, X, torch, monkeypatch
):
    tensor = _tensor(torch, X, kind)
    before = _snapshot(tensor)
    expected = _model(EmpiricalCovariance).fit(_logical_numpy(tensor))
    calls = []
    original_cpu, original_to = torch.Tensor.cpu, torch.Tensor.to
    original_numpy = torch.Tensor.numpy
    original_isfinite = np.isfinite

    def cpu(self, *args, **kwargs):
        assert not self.requires_grad
        calls.append("cpu")
        return original_cpu(self, *args, **kwargs)

    def to(self, *args, **kwargs):
        if kwargs.get("dtype") == torch.float64 or torch.float64 in args:
            assert self.device.type == "cpu"
            calls.append("float64")
        return original_to(self, *args, **kwargs)

    def numpy(self, *args, **kwargs):
        assert not self.requires_grad and not self.is_neg()
        assert self.dtype == torch.float64 and self.device.type == "cpu"
        calls.append("numpy")
        return original_numpy(self, *args, **kwargs)

    def isfinite(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.shape == X.shape:
            calls.append("finite")
        return original_isfinite(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", cpu)
        patch.setattr(torch.Tensor, "to", to)
        patch.setattr(torch.Tensor, "numpy", numpy)
        patch.setattr(np, "isfinite", isfinite)
        actual = _model(EmpiricalCovariance).fit(tensor)
    assert calls[:4] == ["cpu", "float64", "numpy", "finite"]
    _assert_fit_matches(actual, expected)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("policy", ["explicit_cpu", "native_auto"])
@pytest.mark.parametrize("view", ["requires_grad", "noncontiguous"])
def test_missing_fp8_kernel_preserves_gradient_and_backing_storage(
    policy, view, X, torch, monkeypatch, missing_fp8_isfinite
):
    base = _tensor(torch, np.repeat(X, 2, axis=1), "float8_e5m2")
    tensor = base[:, ::2] if view == "noncontiguous" else base[:, ::2].clone()
    tensor.requires_grad_(True)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(EmpiricalCovariance).fit(values)
    actual = _model(EmpiricalCovariance, _policy(policy, monkeypatch)).fit(tensor)
    _assert_fit_matches(actual, expected, native=policy == "native_auto")
    _assert_queries_match(actual, expected, tensor, values, native=policy == "native_auto")
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("previous_backend", ["numpy", "torch"])
def test_missing_fp8_kernel_refit_resolves_new_policy_not_stale_fitted_backend(
    cls, previous_backend, X, torch, monkeypatch, missing_fp8_isfinite
):
    tensor = _tensor(torch, X, "float8_e5m2")
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    if previous_backend == "numpy":
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    model = _model(cls, "auto").fit(tensor)
    assert model._backend_name == previous_backend
    missing_fp8_isfinite.clear()
    native = previous_backend == "numpy"
    monkeypatch.setattr(_device_manager, "_current_device", Device.AUTO if native else Device.CPU)
    model.fit(tensor)
    if native:
        _assert_native_model(model, torch, tensor.device)
        assert missing_fp8_isfinite[0] == (torch.float64, tensor.device)
    else:
        _assert_numpy_model(model)
        assert missing_fp8_isfinite == []
    _assert_fit_matches(model, expected, native=native)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (4, 0), (2, 3, 1)])
def test_missing_fp8_kernel_retains_fit_shape_errors(
    cls, shape, torch, missing_fp8_isfinite
):
    tensor = _tensor(torch, np.zeros(shape), "float8_e5m2")
    with pytest.raises(ValueError, match="sample|feature|2D|two-dimensional"):
        _model(cls).fit(tensor)


@pytest.mark.parametrize("method", QUERIES)
@pytest.mark.parametrize("shape, message", [((0, 3), "sample"), ((4, 0), "feature"),
                                           ((4, 2), "features"), ((2, 3, 1), "two-dimensional")])
def test_missing_fp8_kernel_retains_query_shape_errors(
    method, shape, message, X, torch, missing_fp8_isfinite
):
    model = _model(EmpiricalCovariance).fit(X)
    tensor = _tensor(torch, np.zeros(shape), "float8_e5m2")
    with pytest.raises(ValueError, match=message):
        getattr(model, method)(tensor)


@pytest.mark.parametrize("policy", ["explicit_cpu", "global_cpu", "native_auto"])
def test_missing_fp8_kernel_cv_preserves_folds_and_selected_refit(
    policy, X, torch, monkeypatch, missing_fp8_isfinite
):
    tensor = _tensor(torch, X, "float8_e5m2")
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(GraphicalLassoCV).fit(values)
    device = _policy(policy, monkeypatch)
    native = policy == "native_auto"
    backend = "torch" if native else "numpy"
    fits, scores = [], []
    original_fit, original_score = GraphicalLasso._fit_prepared, GraphicalLasso.score

    def fit(self, backend_name, xp, prepared):
        assert backend_name == backend
        assert prepared.dtype == (torch.float64 if native else np.float64)
        fits.append((self.alpha, _logical_numpy(prepared) if native else prepared.copy()))
        monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
        return original_fit(self, backend_name, xp, prepared)

    def score(self, prepared, y=None):
        assert self._backend_name == backend
        scores.append(_logical_numpy(prepared) if native else prepared.copy())
        return original_score(self, prepared, y=y)

    monkeypatch.setattr(GraphicalLasso, "_fit_prepared", fit)
    monkeypatch.setattr(GraphicalLasso, "score", score)
    actual = _model(GraphicalLassoCV, device).fit(tensor)
    folds = np.array_split(np.random.RandomState(7).permutation(len(values)), 3)
    expected_fits, expected_scores = [], []
    for alpha in (0.03, 0.12):
        for i, test in enumerate(folds):
            train = np.concatenate([folds[j] for j in range(3) if j != i])
            expected_fits.append((alpha, values[train]))
            expected_scores.append(values[test])
    expected_fits.append((actual.alpha_, values))
    assert len(fits) == 7 and len(scores) == 6
    for (alpha, got), (expected_alpha, want) in zip(fits, expected_fits):
        assert alpha == expected_alpha
        assert_array_equal(got, want)
    for got, want in zip(scores, expected_scores):
        assert_array_equal(got, want)
    _assert_fit_matches(actual, expected, native=native)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("reset_name", ["_reset_fit_state", "_reset_cv_fit_state"])
@pytest.mark.parametrize("failure_stage", ["conversion", "finite"])
def test_public_prevalidation_error_preserves_identity_and_fit_reset(
    reset_name, failure_stage, X, torch, monkeypatch
):
    model = _model(EmpiricalCovariance, "auto").fit(torch.tensor(X))
    tensor = _tensor(torch, X, "float8_e5m2")
    error = RuntimeError("synthetic public prevalidation failure")
    calls = []

    def reset():
        calls.append("reset")
        model._fitted = False

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(model, reset_name, reset, raising=False)
    if failure_stage == "conversion":
        monkeypatch.setattr(torch.Tensor, "to", fail)
    else:
        monkeypatch.setattr(torch, "isfinite", fail)
    with pytest.raises(RuntimeError) as caught:
        model.fit(tensor)
    assert caught.value is error
    assert calls == ["reset"] and not model._fitted


def test_shared_finite_guard_remains_identity_for_non_covariance(
    X, torch, missing_fp8_isfinite
):
    class IdentityEstimator(BaseEstimator):
        def fit(self, X, y=None):
            self.received = X
            return self

        def predict(self, X):
            return X

    tensor = _tensor(torch, X, "float8_e5m2")
    model = IdentityEstimator(device="cpu")
    for action in (lambda: check_finite(tensor, name="X"),
                   lambda: model.fit(tensor)):
        with pytest.raises(NotImplementedError, match="emulated.*Float8_e5m2"):
            action()
    assert all(dtype == torch.float8_e5m2 for dtype, _ in missing_fp8_isfinite)
    assert not hasattr(model, "received")
    ordinary = _tensor(torch, X, "bfloat16")
    assert check_finite(ordinary, name="X") is ordinary
    assert model.fit(ordinary).received is ordinary


@pytest.mark.parametrize("method", ["fit", "score"])
def test_covariance_fp8_normalization_does_not_extend_to_ignored_y(
    method, X, torch, missing_fp8_isfinite
):
    model = _model(EmpiricalCovariance).fit(X)
    tensor = _tensor(torch, X, "float8_e5m2")
    # y is statistically ignored here, but the established shared public
    # finite contract still validates it without covariance-X normalization.
    with pytest.raises(NotImplementedError, match="emulated.*Float8_e5m2"):
        getattr(model, method)(X=tensor, y=tensor)
    assert missing_fp8_isfinite == [(torch.float8_e5m2, tensor.device)]


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("method", ["fit", "score"])
def test_covariance_ignored_nonfinite_y_still_checks_public_guard(
    cls, method, X, torch, missing_fp8_isfinite
):
    model = _model(cls).fit(X)
    tensor = _tensor(torch, X, "float8_e5m2")
    # Covariance methods ignore y themselves, so this detects a disabled or
    # X-only public guard that would evade the later numerical X validation.
    with pytest.raises(ValueError, match="y must contain only finite values"):
        getattr(model, method)(tensor, y=np.array([np.nan]))


@pytest.mark.parametrize("policy", ["explicit_cpu", "native_auto"])
@pytest.mark.parametrize("kind", ["complex", "sparse", "quantized", "other_fp8"])
def test_unsupported_torch_inputs_reach_original_finite_guard_unchanged(
    policy, kind, X, torch, monkeypatch
):
    base = torch.tensor(X)
    if kind == "complex":
        tensor = torch.complex(base, torch.ones_like(base))
    elif kind == "sparse":
        tensor = base.to_sparse().coalesce()
    elif kind == "quantized":
        tensor = torch.quantize_per_tensor(base.float(), scale=0.1, zero_point=0,
                                           dtype=torch.qint8)
    else:
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("other-FP8 boundary control needs float8_e4m3fn")
        tensor = base.to(torch.float8_e4m3fn)
    error = RuntimeError("original unsupported finite operation")
    seen = []

    def fail(value):
        seen.append(value)
        raise error

    monkeypatch.setattr(torch, "isfinite", fail)
    with pytest.raises(RuntimeError) as caught:
        _model(EmpiricalCovariance, _policy(policy, monkeypatch)).fit(tensor)
    assert caught.value is error
    assert len(seen) == 1 and seen[0].dtype == tensor.dtype
    if kind == "sparse":
        assert seen[0].data_ptr() == tensor.values().data_ptr()
    else:
        assert seen[0] is tensor


@pytest.mark.parametrize("device", ["torch", "cuda"])
@pytest.mark.parametrize("kind", ["bfloat16", "float8_e5m2"])
def test_missing_fp8_kernel_unavailable_explicit_backend_does_not_fallback(
    device, kind, X, torch, monkeypatch, missing_fp8_isfinite
):
    from statgpu.backends._cupy import CuPyBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(CuPyBackend, "is_available", lambda self: False)
    tensor = _tensor(torch, X, kind)
    model = _model(EmpiricalCovariance, device)
    # CuPy-target Torch FP8 is deliberately outside the new contract: its
    # original raw finite check can fail before availability is resolved.
    message = "emulated.*Float8_e5m2" if device == "cuda" and kind == "float8_e5m2" else "CUDA"
    error_type = NotImplementedError if "emulated" in message else RuntimeError
    with pytest.raises(error_type, match=message):
        model.fit(tensor)
    assert not model._fitted and not hasattr(model, "covariance_")


def _physical_fp8(torch, values, index):
    _require_cuda(torch, index + 1)
    # Deliberately no hasattr/operation-failure skip on the physical boundary.
    return torch.tensor(values, dtype=torch.float64, device=f"cuda:{index}").to(
        dtype=torch.float8_e5m2
    )


def _observe_native_guard(model, torch, monkeypatch, source_device):
    """Forbid host transfer only inside validation, not public NumPy output."""
    original_guard = model._check_public_input_finite
    original_cpu, original_isfinite = torch.Tensor.cpu, torch.isfinite
    active = [False]
    checked = []

    def cpu(self, *args, **kwargs):
        assert not active[0], "native finite validation copied a tensor to CPU"
        return original_cpu(self, *args, **kwargs)

    def isfinite(value, *args, **kwargs):
        if active[0]:
            assert value.dtype == torch.float64
            assert value.device == source_device
            checked.append(value.device)
        return original_isfinite(value, *args, **kwargs)

    def guard(value, *, name, method_name):
        active[0] = True
        try:
            return original_guard(value, name=name, method_name=method_name)
        finally:
            active[0] = False

    monkeypatch.setattr(model, "_check_public_input_finite", guard)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch, "isfinite", isfinite)
    return checked


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("index", [0, 1], ids=["single_gpu", "nondefault_gpu"])
def test_physical_cuda_fp8_native_guard_stays_on_source_device(
    cls, index, X, torch, monkeypatch
):
    tensor = _physical_fp8(torch, X, index)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    actual = _model(cls, "auto")
    checked = _observe_native_guard(actual, torch, monkeypatch, tensor.device)
    with torch.cuda.device(0):
        actual.fit(tensor)
        _assert_native_model(actual, torch, tensor.device)
        _assert_fit_matches(actual, expected, native=True)
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
        _assert_queries_match(actual, expected, tensor, values, native=True)
        assert torch.cuda.current_device() == 0
    assert len(checked) >= 4
    _assert_unchanged(tensor, before)
    torch.cuda.synchronize(index)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("policy", ["explicit_cpu", "global_cpu"])
def test_physical_cuda_fp8_numpy_trained_query_boundary(
    cls, policy, X, torch, monkeypatch
):
    tensor = _physical_fp8(torch, X, 0)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    actual = _model(cls, _policy(policy, monkeypatch)).fit(tensor)
    _assert_numpy_model(actual)
    _assert_fit_matches(actual, expected)
    monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    _assert_queries_match(actual, expected, tensor, values)
    _assert_queries_match(expected, expected, tensor, values)
    _assert_unchanged(tensor, before)
    torch.cuda.synchronize(0)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("policy", ["explicit_cpu", "native_auto"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_physical_cuda_fp8_nonfinite_preserves_error_provenance(
    cls, policy, bad_value, X, torch, monkeypatch
):
    data = X.copy()
    data[1, 0] = bad_value
    tensor = _physical_fp8(torch, data, 0)
    before = _snapshot(tensor)
    model = _model(cls, _policy(policy, monkeypatch))
    native = policy == "native_auto"
    if native:
        _observe_native_guard(model, torch, monkeypatch, tensor.device)
    message = "X must contain only finite values; found NaN or infinite values"
    fitted = None
    for method in ("fit",) + QUERIES:
        if method != "fit" and not model._fitted:
            model.fit(_physical_fp8(torch, X, 0))
            fitted = _fitted_snapshot(model)
            monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
        with pytest.raises(ValueError) as caught:
            getattr(model, method)(tensor)
        assert str(caught.value) == message
        assert caught.value._statgpu_finite_backend == "torch"
        assert caught.value._statgpu_finite_device == "cuda:0"
        if fitted is not None:
            _assert_fitted_unchanged(model, fitted, torch)
    numpy_trained = _model(cls).fit(X)
    numpy_fitted = _fitted_snapshot(numpy_trained)
    monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    for method in QUERIES:
        with pytest.raises(ValueError) as caught:
            getattr(numpy_trained, method)(tensor)
        assert str(caught.value) == message
        assert caught.value._statgpu_finite_backend == "torch"
        assert caught.value._statgpu_finite_device == "cuda:0"
        _assert_numpy_model(numpy_trained)
        _assert_fitted_unchanged(numpy_trained, numpy_fitted, torch)
    _assert_unchanged(tensor, before)
    torch.cuda.synchronize(0)


@pytest.mark.parametrize("policy", ["explicit_cpu", "native_auto"])
@pytest.mark.parametrize("failure_stage", ["conversion", "finite"])
@pytest.mark.parametrize("index", [0, 1], ids=["single_gpu", "nondefault_gpu"])
def test_physical_cuda_fp8_failure_preserves_identity_provenance_and_reset(
    policy, failure_stage, index, X, torch, monkeypatch
):
    tensor = _physical_fp8(torch, X, index)
    model = _model(EmpiricalCovariance, _policy(policy, monkeypatch))
    error = RuntimeError("synthetic FP8 prevalidation allocation failure")
    resets = []
    monkeypatch.setattr(model, "_reset_fit_state", lambda: resets.append("reset"),
                        raising=False)

    def fail(*args, **kwargs):
        raise error

    if failure_stage == "conversion":
        monkeypatch.setattr(torch.Tensor, "to", fail)
    elif policy == "native_auto":
        monkeypatch.setattr(torch, "isfinite", fail)
    else:
        monkeypatch.setattr(np, "isfinite", fail)
    with pytest.raises(RuntimeError) as caught:
        model.fit(tensor)
    assert caught.value is error and resets == ["reset"]
    assert error._statgpu_finite_backend == "torch"
    assert error._statgpu_finite_device == f"cuda:{index}"
    assert not model._fitted and not hasattr(model, "covariance_")
