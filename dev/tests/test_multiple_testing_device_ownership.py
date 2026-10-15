"""Call-local Torch device ownership for the multiple-testing public APIs.

Routing tests do not require Torch. Numerical tests use real Torch CPU, including
simulated CUDA availability, and separately gated physical one-/two-GPU cases.
Mocked CUDA routing is not evidence of physical CUDA numerical execution.
"""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from statgpu.backends import _base, _factory
from statgpu.inference import _multiple_testing as mt

ADJUST_METHODS = ("bh", "by", "holm", "bonferroni", "hochberg")
COMBINE_METHODS = ("fisher", "cauchy", "stouffer")
AXES = (None, 0, 1, -1)
VALUES = np.array([[0.01, 0.20, 0.70, 0.90],
                   [0.04, 0.03, 0.40, 0.60],
                   [0.50, 0.06, 0.02, 0.80]])


def _factory_state():
    return (vars(_factory._torch_backend).copy(),
            vars(_factory._torch_cpu_backend).copy())


@pytest.fixture
def fake_torch(monkeypatch):
    """Only the routing protocol; no dependency on Torch or physical hardware."""
    class Device:
        def __init__(self, value):
            self.value = str(value)
            self.type = self.value.split(":", 1)[0]
            self.index = (int(self.value.split(":", 1)[1])
                          if ":" in self.value else None)

        def __str__(self):
            return self.value

    class Tensor:
        def __init__(self, device, dtype="float64"):
            self.device = Device(device)
            self.dtype = dtype

    state = SimpleNamespace(available=True, current=0, reads=0)

    def current_device():
        state.reads += 1
        return state.current

    def forbidden_set_device(*args, **kwargs):
        pytest.fail("multiple-testing routing must not change CUDA's current device")

    module = ModuleType("torch")
    module.Tensor = Tensor
    module.device = Device
    module.cuda = SimpleNamespace(
        is_available=lambda: state.available,
        current_device=current_device,
        set_device=forbidden_set_device,
    )
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setattr(_base, "_TORCH_MOD", module)
    monkeypatch.setattr(_base, "_TORCH_CHECKED", True)
    return module, state


@pytest.mark.parametrize("backend", ["auto", "torch"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:3"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("available", [False, True])
def test_tensor_device_owns_routing_without_consulting_cuda_default(
    fake_torch, backend, device, dtype, available
):
    torch, state = fake_torch
    state.available = available
    state.current = 1
    pvalues = torch.Tensor(device, dtype)
    before = _factory_state()

    first = mt._get_multiple_testing_backend(backend, pvalues)
    second = mt._get_multiple_testing_backend(backend, pvalues)

    assert first.name == "torch"
    assert first._device == second._device == device
    assert first is not second
    assert first is not _factory._torch_backend
    assert first is not _factory._torch_cpu_backend
    assert state.reads == 0
    assert state.current == 1
    assert pvalues.device.type == device.split(":")[0]
    assert str(pvalues.device) == device
    assert pvalues.dtype == dtype
    assert _factory_state() == before


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("pvalues", [[0.01, 0.4], np.array([0.01, 0.4])])
def test_non_tensor_explicit_torch_captures_concrete_default_device(
    fake_torch, available, pvalues
):
    _, state = fake_torch
    state.available = available
    state.current = 2
    before = _factory_state()
    first = mt._get_multiple_testing_backend("torch", pvalues)
    assert first._device == ("cuda:2" if available else "cpu")
    assert state.reads == int(available)

    # A subsequent CUDA default change must not retarget an existing call.
    state.current = 0
    second = mt._get_multiple_testing_backend("torch", pvalues)
    assert first._device == ("cuda:2" if available else "cpu")
    assert second._device == ("cuda:0" if available else "cpu")
    assert first is not second
    assert _factory_state() == before


@pytest.mark.parametrize("backend", ["auto", "torch"])
@pytest.mark.parametrize("device", ["mps", "meta", "xpu:0"])
def test_unsupported_tensor_devices_fail_explicitly(fake_torch, backend, device):
    torch, state = fake_torch
    before = _factory_state()
    with pytest.raises(ValueError, match="float64"):
        mt._get_multiple_testing_backend(backend, torch.Tensor(device))
    assert state.reads == 0
    assert _factory_state() == before


@pytest.mark.parametrize("requested,expected", [
    ("auto", "numpy"), ("numpy", "numpy"), ("cpu", "numpy"),
    (" NumPy ", "numpy"), ("cupy", "cupy"), ("cuda", "cupy"),
    ("gpu", "cupy"),
])
def test_other_backend_and_legacy_alias_resolution_is_preserved(
    monkeypatch, requested, expected
):
    sentinel = object()
    calls = []

    def factory(name):
        calls.append(name)
        return sentinel

    monkeypatch.setattr(mt, "get_backend", factory)
    assert mt._get_multiple_testing_backend(requested, VALUES) is sentinel
    assert calls == [expected]


@pytest.mark.parametrize("function,method", [
    *[(mt.adjust_pvalues, method) for method in ADJUST_METHODS],
    *[(mt.multipletests, method) for method in ADJUST_METHODS],
    *[(mt.combine_pvalues, method) for method in COMBINE_METHODS],
])
def test_every_public_consumer_uses_call_local_resolution(monkeypatch, function, method):
    calls = []

    def resolve(backend, pvalues):
        calls.append((backend, pvalues))
        return _factory.get_backend("numpy")

    monkeypatch.setattr(mt, "_get_multiple_testing_backend", resolve)
    function(VALUES, method=method, backend="torch")
    assert len(calls) == 1
    assert calls[0][0] == "torch"
    assert calls[0][1] is VALUES


@pytest.fixture(params=[
    "cpu_with_cuda_available",
    pytest.param("cuda", marks=pytest.mark.torch_cuda),
    pytest.param("cuda_noncurrent", marks=pytest.mark.torch_cuda),
])
def torch_owner(request, monkeypatch, physical_gpu_required):
    """CPU ownership is exercised even when CUDA is advertised as available."""
    mode = request.param
    if mode != "cpu_with_cuda_available":
        try:
            import torch
        except ImportError:
            if physical_gpu_required:
                pytest.fail("STATGPU_REQUIRE_PHYSICAL_GPU=1 but Torch is not installed")
            pytest.skip("Torch is not installed")
        if not torch.cuda.is_available():
            if physical_gpu_required:
                pytest.fail("STATGPU_REQUIRE_PHYSICAL_GPU=1 but Torch CUDA is unavailable")
            pytest.skip("physical Torch CUDA is unavailable")
        current = torch.cuda.current_device()
        if mode == "cuda_noncurrent":
            if torch.cuda.device_count() < 2:
                pytest.skip("non-current-device execution requires two physical CUDA GPUs")
            index = (current + 1) % torch.cuda.device_count()
        else:
            index = current
        yield torch, torch.device("cuda", index)
        assert torch.cuda.current_device() == current
        return

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def forbidden_current_device():
        pytest.fail("CPU p-value tensors must not query the current CUDA device")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden_current_device)
    yield torch, torch.device("cpu")


def _tensor_snapshot(tensor):
    return (tensor.detach().clone(), tensor.device, tensor.dtype, tensor.shape,
            tensor.stride(), tensor.data_ptr(), tensor._version)


def _assert_unchanged(torch, tensor, snapshot):
    value, device, dtype, shape, stride, pointer, version = snapshot
    torch.testing.assert_close(tensor, value, rtol=0, atol=0, equal_nan=True)
    assert (tensor.device, tensor.dtype, tensor.shape, tensor.stride(),
            tensor.data_ptr(), tensor._version) == (
                device, dtype, shape, stride, pointer, version)


def _assert_outputs(torch, outputs, expected, device, adjustment=False):
    assert len(outputs) == 2
    for index, (actual, baseline) in enumerate(zip(outputs, expected)):
        assert isinstance(actual, torch.Tensor)
        assert actual.device == device
        assert actual.dtype == (torch.bool if adjustment and index == 0 else torch.float64)
        assert tuple(actual.shape) == np.shape(baseline)
        if adjustment and index == 0:
            np.testing.assert_array_equal(actual.detach().cpu().numpy(), baseline)
        else:
            np.testing.assert_allclose(actual.detach().cpu().numpy(), baseline,
                                       rtol=2e-7, atol=2e-10)


@pytest.mark.parametrize("method", ADJUST_METHODS)
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_adjustments_and_multipletests_preserve_tensor_ownership(
    torch_owner, method, axis, dtype, backend
):
    torch, device = torch_owner
    # Transposition also tests noncontiguous input without changing its layout.
    pvalues = torch.tensor(VALUES, dtype=getattr(torch, dtype), device=device).T
    before = _tensor_snapshot(pvalues)
    factory_before = _factory_state()
    expected = mt.adjust_pvalues(pvalues.cpu().numpy(), method=method,
                                axis=axis, backend="numpy")
    for function in (mt.adjust_pvalues, mt.multipletests):
        outputs = function(pvalues, method=method, axis=axis, backend=backend)
        _assert_outputs(torch, outputs, expected, device, adjustment=True)
        _assert_unchanged(torch, pvalues, before)
    assert _factory_state() == factory_before


@pytest.mark.parametrize("method,weighted", [
    ("fisher", False), ("cauchy", False), ("cauchy", True),
    ("stouffer", False), ("stouffer", True),
])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_combinations_preserve_tensor_and_weight_ownership(
    torch_owner, method, weighted, axis, dtype, backend
):
    torch, device = torch_owner
    pvalues = torch.tensor(VALUES, dtype=getattr(torch, dtype), device=device).T
    before = _tensor_snapshot(pvalues)
    m = pvalues.numel() if axis is None else pvalues.shape[axis]
    weights = (torch.arange(1, m + 1, dtype=getattr(torch, dtype), device=device)
               if weighted else None)
    weights_before = _tensor_snapshot(weights) if weighted else None
    factory_before = _factory_state()
    expected = mt.combine_pvalues(
        pvalues.cpu().numpy(), method=method, axis=axis, backend="numpy",
        weights=weights.cpu().numpy() if weighted else None,
    )
    outputs = mt.combine_pvalues(pvalues, method=method, weights=weights,
                                 axis=axis, backend=backend)
    _assert_outputs(torch, outputs, expected, device)
    _assert_unchanged(torch, pvalues, before)
    if weighted:
        _assert_unchanged(torch, weights, weights_before)
    assert _factory_state() == factory_before


@pytest.mark.parametrize("method", ["cauchy", "stouffer"])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("backend", ["auto", "torch"])
@pytest.mark.parametrize("direction", ["cpu_pvalues", "cuda_pvalues", "other_cuda_weights"])
@pytest.mark.torch_cuda
def test_cross_device_weights_follow_pvalues(
    torch_cuda_available, method, axis, backend, direction
):
    import torch

    current = torch.cuda.current_device()
    if direction == "other_cuda_weights" and torch.cuda.device_count() < 2:
        pytest.skip("cross-GPU weights require two physical CUDA GPUs")
    cuda = torch.device("cuda", current)
    p_device = torch.device("cpu") if direction == "cpu_pvalues" else cuda
    w_device = cuda if direction == "cpu_pvalues" else torch.device("cpu")
    if direction == "other_cuda_weights":
        w_device = torch.device("cuda", (current + 1) % torch.cuda.device_count())
    pvalues = torch.tensor(VALUES, dtype=torch.float32, device=p_device)
    m = pvalues.numel() if axis is None else pvalues.shape[axis]
    weights = torch.arange(1, m + 1, dtype=torch.float64, device=w_device)
    p_before, w_before = _tensor_snapshot(pvalues), _tensor_snapshot(weights)
    factory_before = _factory_state()
    expected = mt.combine_pvalues(pvalues.cpu().numpy(), method=method, axis=axis,
                                 weights=weights.cpu().numpy(), backend="numpy")
    outputs = mt.combine_pvalues(pvalues, method=method, weights=weights,
                                 axis=axis, backend=backend)
    _assert_outputs(torch, outputs, expected, p_device)
    _assert_unchanged(torch, pvalues, p_before)
    _assert_unchanged(torch, weights, w_before)
    assert torch.cuda.current_device() == current
    assert _factory_state() == factory_before


@pytest.mark.parametrize("function,method", [
    *[(mt.adjust_pvalues, method) for method in ADJUST_METHODS],
    (mt.multipletests, "bh"),
    *[(mt.combine_pvalues, method) for method in COMBINE_METHODS],
])
@pytest.mark.parametrize("axis", [None, 1])
def test_nan_is_rejected_without_moving_or_mutating_input(torch_owner, function, method, axis):
    torch, device = torch_owner
    pvalues = torch.tensor([[0.01, float("nan")], [0.1, 0.8]],
                           dtype=torch.float32, device=device)
    before = _tensor_snapshot(pvalues)
    with pytest.raises(ValueError, match="finite"):
        function(pvalues, method=method, axis=axis)
    _assert_unchanged(torch, pvalues, before)


@pytest.mark.parametrize("method", ADJUST_METHODS + COMBINE_METHODS)
@pytest.mark.parametrize("axis", [None, 1])
def test_endpoint_clipping_and_ownership_match_numpy(torch_owner, method, axis):
    torch, device = torch_owner
    pvalues = torch.tensor([[0.0, 0.03], [1.0, 0.75]],
                           dtype=torch.float64, device=device)
    function = mt.adjust_pvalues if method in ADJUST_METHODS else mt.combine_pvalues
    before = _tensor_snapshot(pvalues)
    expected = function(pvalues.cpu().numpy(), method=method, axis=axis, backend="numpy")
    outputs = function(pvalues, method=method, axis=axis)
    _assert_outputs(torch, outputs, expected, device, adjustment=method in ADJUST_METHODS)
    _assert_unchanged(torch, pvalues, before)


def test_calls_do_not_consume_rng_or_mutate_cached_backends(torch_owner):
    torch, device = torch_owner
    pvalues = torch.tensor(VALUES, dtype=torch.float64, device=device)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    factory_before = _factory_state()
    for _ in range(2):
        for method in ADJUST_METHODS:
            mt.adjust_pvalues(pvalues, method=method)
        mt.multipletests(pvalues)
        for method in COMBINE_METHODS:
            mt.combine_pvalues(pvalues, method=method)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert all(torch.equal(before, after) for before, after in
                   zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert _factory_state() == factory_before


@pytest.mark.parametrize("function,method", [
    (mt.adjust_pvalues, "holm"), (mt.multipletests, "by"),
    *[(mt.combine_pvalues, method) for method in COMBINE_METHODS],
])
def test_explicit_torch_for_non_tensor_uses_cpu_when_cuda_unavailable(monkeypatch, function, method):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    expected = function(VALUES, method=method, backend="numpy")
    outputs = function(VALUES.tolist(), method=method, backend="torch")
    _assert_outputs(torch, outputs, expected, torch.device("cpu"),
                    adjustment=function is not mt.combine_pvalues)


@pytest.mark.parametrize("function,method", [
    (mt.adjust_pvalues, "holm"), (mt.multipletests, "by"),
    *[(mt.combine_pvalues, method) for method in COMBINE_METHODS],
])
@pytest.mark.torch_cuda
def test_explicit_torch_for_non_tensor_uses_current_cuda(torch_cuda_available, function, method):
    import torch

    current = torch.cuda.current_device()
    expected = function(VALUES, method=method, backend="numpy")
    outputs = function(VALUES, method=method, backend="torch")
    _assert_outputs(torch, outputs, expected, torch.device("cuda", current),
                    adjustment=function is not mt.combine_pvalues)
    assert torch.cuda.current_device() == current


@pytest.mark.parametrize("function", [mt.adjust_pvalues, mt.multipletests, mt.combine_pvalues])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_real_meta_tensor_is_rejected_before_conversion(function, backend):
    torch = pytest.importorskip("torch")
    pvalues = torch.empty(3, device="meta", dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        function(pvalues, backend=backend)
    assert pvalues.device.type == "meta"
    assert pvalues.dtype == torch.float32


@pytest.mark.torch_cuda
def test_captured_cuda_backend_keeps_allocations_on_original_device(torch_cuda_available):
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("changing the current CUDA device requires two physical CUDA GPUs")
    original = torch.cuda.current_device()
    device = torch.device("cuda", original)
    before = _factory_state()
    backend = mt._get_multiple_testing_backend("torch", VALUES)
    with torch.cuda.device((original + 1) % torch.cuda.device_count()):
        changed_current = torch.cuda.current_device()
        for array in (backend.asarray(VALUES, dtype=backend.float64),
                      backend.arange(4, dtype=backend.float64),
                      backend.full((4,), 0.25, dtype=backend.float64)):
            assert array.device == device
            assert array.dtype == torch.float64
        assert torch.cuda.current_device() == changed_current
    assert torch.cuda.current_device() == original
    assert _factory_state() == before
