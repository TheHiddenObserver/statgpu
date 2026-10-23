"""Torch host ingress contracts, separate from physical CuPy execution evidence.

The CPU conversion spies exercise only boundary dispatch, never emulate a GPU
or run a public estimator. Numerical/device regressions require real CuPy CUDA.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.nonparametric import KernelDensityEstimator, fit_kernel_regression
from statgpu.nonparametric.kernel_smoothing import _kde, _kernel_common


INPUT_KINDS = (
    "int64", "float16", "float32", "float64", "bfloat16",
    "strided", "transposed", "requires_grad", "negative_view", "real_conjugate",
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _host_tensor(torch, value, kind):
    dtype = getattr(torch, kind) if kind in INPUT_KINDS[:5] else torch.float64
    tensor = torch.tensor(value, dtype=dtype)
    if kind == "strided" or (kind == "transposed" and tensor.ndim == 1):
        tensor = tensor.repeat_interleave(2, dim=0)[::2]
        assert not tensor.is_contiguous()
    elif kind == "transposed":
        tensor = tensor.T.contiguous().T
        assert not tensor.is_contiguous()
    elif kind == "requires_grad":
        tensor.requires_grad_()
    elif kind == "negative_view":
        # Logical values equal the input, but direct .numpy() must fail until
        # the lazy negative bit is resolved. Weights remain nonnegative.
        tensor = torch._neg_view(-tensor)
        assert tensor.is_neg()
    elif kind == "real_conjugate":
        # Real conjugation is a no-op; genuinely complex views are rejected.
        tensor = tensor.conj()
    return tensor


def _host_value(tensor):
    return tensor.detach().resolve_conj().resolve_neg().double().numpy().copy()


class _ConversionSpy:
    """Record conversion arguments only. This is not a numerical backend."""

    __name__ = "cupy"
    float64 = np.float64

    def __init__(self):
        self.calls = []
        self.result = object()

    def asarray(self, value, dtype=None):
        self.calls.append((value, dtype))
        return self.result


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("reference", ["none", "torch_input", "cupy_target"])
def test_cpu_conversion_contract_exports_resolved_host_storage(torch, monkeypatch, kind, reference):
    value = _host_tensor(torch, np.arange(24).reshape(8, 3), kind)
    expected = _host_value(value)
    original = value.detach().clone()
    state = (value.dtype, value.stride(), value.requires_grad, value.is_neg(), value.is_conj())
    spy = _ConversionSpy()
    target_calls = []

    def on_target(data, device, dtype=None):
        target_calls.append(device)
        return spy.asarray(data, dtype=dtype)

    def forbidden(*args, **kwargs):
        raise AssertionError("host ingress must not call .cpu() or implicit __array__")

    monkeypatch.setattr(_kernel_common, "_cupy_asarray_on_device", on_target)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "__array__", forbidden)
    ref = {"none": None, "torch_input": value,
           "cupy_target": SimpleNamespace(device=SimpleNamespace(id=7))}[reference]
    result = _kernel_common._as_backend_float64(value, spy, ref_arr=ref)

    assert result is spy.result
    assert target_calls == ([7] if reference == "cupy_target" else [])
    assert len(spy.calls) == 1
    host, dtype = spy.calls[0]
    assert isinstance(host, np.ndarray) and host.dtype == np.float64
    assert dtype is np.float64  # Never rewrite the CuPy dtype to torch.float64.
    np.testing.assert_array_equal(host, expected)
    assert torch.equal(value, original)
    assert state == (value.dtype, value.stride(), value.requires_grad, value.is_neg(), value.is_conj())


@pytest.mark.parametrize("name", ["samples", "points", "weights"])
def test_cpu_conversion_contract_rejects_complex_conjugate_before_cast(torch, monkeypatch, name):
    value = torch.tensor([1 + 2j, 3 - 4j]).conj()
    assert value.is_conj()
    spy = _ConversionSpy()

    def forbidden(*args, **kwargs):
        raise AssertionError("complex input must be rejected before host export")

    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    with pytest.raises(ValueError, match=f"{name} must be real-valued"):
        _kernel_common._as_backend_float64(value, spy, ref_arr=value, name=name)
    assert not spy.calls


def test_cpu_conversion_contract_does_not_export_non_cpu_tensor(torch, monkeypatch):
    # A real meta tensor proves the host-only dispatch condition without
    # pretending that this test allocates or computes anything on CUDA.
    value = torch.empty((3, 2), device="meta")
    spy = _ConversionSpy()

    def forbidden(*args, **kwargs):
        raise AssertionError("non-CPU input must not be exported to host")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "__array__", forbidden)
    result = _kernel_common._as_backend_float64(value, spy, ref_arr=value)
    assert result is spy.result
    assert len(spy.calls) == 1
    assert spy.calls[0][0] is value and spy.calls[0][1] is np.float64


def test_cpu_conversion_contract_does_not_silently_densify_sparse_tensor(torch, monkeypatch):
    value = torch.tensor([[1., 0.], [0., 2.], [3., 0.]]).to_sparse()
    spy = _ConversionSpy()

    def forbidden(*args, **kwargs):
        raise AssertionError("host ingress must not silently densify sparse storage")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    with pytest.raises((TypeError, RuntimeError), match="[Ss]parse|layout"):
        _kernel_common._as_backend_float64(value, spy, ref_arr=value)
    assert not spy.calls


@pytest.fixture
def physical_cupy():
    cp = pytest.importorskip("cupy", reason="physical CuPy CUDA required; no emulation")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("physical CuPy CUDA required")
        cp.cuda.Device().synchronize()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"working physical CuPy CUDA required: {exc}")
    return cp


def _data():
    rng = np.random.default_rng(254)
    return (rng.normal(size=(24, 2)) * 4, rng.normal(size=(7, 2)) * 2,
            np.arange(24) % 7)


def _assert_cupy_arrays(cp, model, outputs, target):
    for value in (model.samples_, model.weights_, model.covariance_,
                  model.inv_covariance_, model._samples_proj_, model._samples_quad_, *outputs):
        assert isinstance(value, cp.ndarray)
        assert value.device.id == target and value.dtype == cp.float64


def _forbid_native_host_exports(monkeypatch, cp, torch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native CuPy computation attempted host-array export")

    original_numpy = torch.Tensor.numpy

    def host_numpy(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad and not tensor.is_conj() and not tensor.is_neg()
        assert tensor.dtype == torch.float64
        return original_numpy(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "__array__", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", host_numpy)
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    monkeypatch.setattr(_kde, "_to_numpy", forbidden)
    monkeypatch.setattr(_kernel_common, "_to_numpy", forbidden)


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_physical_cupy_kde_accepts_torch_cpu_samples_queries_weights(torch, physical_cupy, monkeypatch, kind):
    cp = physical_cupy
    x, q, w = [_host_tensor(torch, value, kind) for value in _data()]
    expected_x, expected_q, expected_w = map(_host_value, (x, q, w))
    snapshots = [value.detach().clone() for value in (x, q, w)]
    expected = KernelDensityEstimator(
        weights=expected_w, bandwidth=.7, backend="numpy", device="cpu",
    ).fit(expected_x)
    target = cp.cuda.Device().id
    with monkeypatch.context() as patch:
        _forbid_native_host_exports(patch, cp, torch)
        model = KernelDensityEstimator(
            weights=w, bandwidth=.7, backend="cupy", device="cuda",
        ).fit(x)
        pdf, logpdf = model.pdf(q, batch_size=3), model.logpdf(q, batch_size=2)
        score = model.score(q)
        model.fit(x)
        refit = model.pdf(q, batch_size=3)
        assert cp.cuda.Device().id == target
        _assert_cupy_arrays(cp, model, (pdf, logpdf, refit), target)
    np.testing.assert_allclose(cp.asnumpy(model.samples_), expected_x, rtol=0, atol=0)
    np.testing.assert_allclose(cp.asnumpy(pdf), expected.pdf(expected_q), rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(cp.asnumpy(logpdf), expected.logpdf(expected_q), rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(cp.asnumpy(refit), cp.asnumpy(pdf), rtol=0, atol=0)
    assert score == pytest.approx(expected.score(expected_q), rel=2e-10, abs=2e-12)
    for value, original in zip((x, q, w), snapshots):
        assert torch.equal(value, original)


def test_physical_cupy_host_queries_weights_follow_training_device(torch, physical_cupy, monkeypatch):
    cp = physical_cupy
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two physical CuPy CUDA devices required for target-device regression")
    x, q, w = _data()
    expected = KernelDensityEstimator(
        weights=w, bandwidth=.7, backend="numpy", device="cpu",
    ).fit(x)
    with cp.cuda.Device(1):
        dx = cp.asarray(x)
        # Host training samples should allocate on the current CuPy device.
        host_model = KernelDensityEstimator(backend="cupy", device="cuda", bandwidth=.7).fit(
            _host_tensor(torch, x, "requires_grad"),
        )
        assert host_model.samples_.device.id == 1
    dq, dw = [_host_tensor(torch, value, "negative_view") for value in (q, w)]
    with cp.cuda.Device(0), monkeypatch.context() as patch:
        _forbid_native_host_exports(patch, cp, torch)
        model = KernelDensityEstimator(
            weights=dw, bandwidth=.7, backend="cupy", device="cuda",
        ).fit(dx)
        pdf, logpdf = model.pdf(dq), model.logpdf(dq)
        _assert_cupy_arrays(cp, model, (pdf, logpdf), 1)
        assert cp.cuda.Device().id == 0
    np.testing.assert_allclose(cp.asnumpy(pdf), expected.pdf(q), rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(cp.asnumpy(logpdf), expected.logpdf(q), rtol=2e-10, atol=2e-12)


def test_physical_cupy_shared_regression_inputs_preserve_results(torch, physical_cupy):
    cp = physical_cupy
    x, q, w = _data()
    y = np.sin(x[:, 0]) + .2 * x[:, 1]
    expected = fit_kernel_regression(x, y, weights=w, bandwidth=.7, backend="numpy")
    dx, dq, dw = [_host_tensor(torch, value, "requires_grad") for value in (x, q, w)]
    # Targets use a different boundary, outside this repair's scope.
    actual = fit_kernel_regression(dx, y, weights=dw, bandwidth=.7, backend="cupy")
    output = actual.predict(dq)
    assert isinstance(output, cp.ndarray) and output.dtype == cp.float64
    assert output.device == actual.samples_.device
    np.testing.assert_allclose(cp.asnumpy(output), expected.predict(q), rtol=2e-10, atol=2e-12)


def test_physical_cupy_torch_cuda_ingress_never_roundtrips_host(torch, physical_cupy, monkeypatch):
    cp = physical_cupy
    if not torch.cuda.is_available():
        pytest.skip("physical Torch CUDA required for cross-library no-host regression")
    x, q, w = _data()
    target = cp.cuda.Device().id
    dx, dq, dw = [torch.tensor(value, dtype=torch.float64, device=f"cuda:{target}")
                  for value in (x, q, w)]
    expected = KernelDensityEstimator(
        weights=w, bandwidth=.7, backend="numpy", device="cpu",
    ).fit(x)
    with monkeypatch.context() as patch:
        _forbid_native_host_exports(patch, cp, torch)
        model = KernelDensityEstimator(
            weights=dw, bandwidth=.7, backend="cupy", device="cuda",
        ).fit(dx)
        pdf, logpdf = model.pdf(dq), model.logpdf(dq)
        _assert_cupy_arrays(cp, model, (pdf, logpdf), target)
    np.testing.assert_allclose(cp.asnumpy(pdf), expected.pdf(q), rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(cp.asnumpy(logpdf), expected.logpdf(q), rtol=2e-10, atol=2e-12)
