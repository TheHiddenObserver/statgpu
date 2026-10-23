"""Weighted KDE backend parity; Torch CPU is algebra evidence, not CUDA evidence."""

import math

import numpy as np
import pytest

from statgpu.nonparametric import KernelDensityEstimator, fit_kde
from statgpu.nonparametric.kernel_smoothing import _kde, _kernel_common


KERNEL_DIMENSIONS = [
    (kernel, dimension)
    for kernel in ("gaussian", "rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cosine", "optcosine")
    for dimension in ((1,) if kernel in ("cosine", "optcosine") else (1, 2, 8))
]


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def data(dimension, weights_kind):
    rng = np.random.default_rng(187)
    samples = rng.normal(size=(36, dimension))
    points = np.concatenate([samples[:4], samples[:3] + 0.13, np.full((1, dimension), 20.0)])
    weights = None if weights_kind == "none" else np.arange(1.0, 37.0)
    if weights_kind == "zeros":
        weights[::3] = 0.0
    return samples, points, weights


@pytest.mark.parametrize("kernel,dimension", KERNEL_DIMENSIONS)
@pytest.mark.parametrize("weights_kind", ["none", "unequal", "zeros"])
def test_torch_cpu_weighted_fit_pdf_logpdf_matches_numpy(torch, kernel, dimension, weights_kind):
    samples, points, weights = data(dimension, weights_kind)
    expected = KernelDensityEstimator(
        bandwidth=0.8, kernel=kernel, weights=weights, backend="numpy", device="cpu"
    ).fit(samples)
    dx = torch.asarray(samples, dtype=torch.float64)
    dq = torch.asarray(points, dtype=torch.float64)
    dw = None if weights is None else torch.asarray(weights, dtype=torch.float64)
    actual = KernelDensityEstimator(
        bandwidth=0.8, kernel=kernel, weights=dw, backend="torch", device="cpu"
    ).fit(dx)
    for field in ("samples_", "weights_", "covariance_", "inv_covariance_"):
        value = getattr(actual, field)
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float64 and value.device == dx.device
        np.testing.assert_allclose(value.numpy(), getattr(expected, field), rtol=2e-12, atol=2e-13)
    for batch_size in (1, 5, 100):
        for method in ("pdf", "logpdf"):
            value = getattr(actual, method)(dq, batch_size=batch_size)
            assert value.device == dx.device and value.dtype == torch.float64
            np.testing.assert_allclose(
                value.numpy(), getattr(expected, method)(points, batch_size=batch_size),
                # sqrt amplifies last-bit quadratic-form noise at a triangular
                # kernel's exact apex; other kernels have smooth derivatives.
                rtol=3e-7 if kernel == "triangular" else 2e-11,
                atol=2e-11,
            )
    # Constructor weights and every caller-owned tensor remain untouched.
    np.testing.assert_array_equal(dx.numpy(), samples)
    np.testing.assert_array_equal(dq.numpy(), points)
    if dw is not None:
        np.testing.assert_array_equal(dw.numpy(), weights)


@pytest.mark.parametrize("kind", ["list", "numpy_int", "numpy_float32", "numpy_float64", "numpy_view", "torch_int", "torch_float32", "torch_float64", "torch_view"])
def test_torch_accepts_host_native_integer_float_and_noncontiguous_inputs(torch, kind):
    x = np.arange(36).reshape(12, 3) % np.array([7, 11, 13])
    w = np.arange(12) % 4
    q = x[::3]
    def convert(value):
        if kind == "list":
            return value.tolist()
        dtype = np.int64 if "int" in kind else np.float32 if "float32" in kind else np.float64
        array = np.array(value, dtype=dtype)
        if kind == "numpy_view":
            # Both positive- and negative-stride host views are valid inputs.
            return array[::-1]
        if kind == "torch_view":
            tensor = torch.asarray(np.repeat(array, 2, axis=0))
            return tensor[::2]
        return torch.asarray(array) if kind.startswith("torch") else array
    xx, ww, qq = map(convert, (x, w, q))
    reference_x, reference_w, reference_q = (v[::-1] for v in (x, w, q)) if kind == "numpy_view" else (x, w, q)
    expected = fit_kde(reference_x, weights=reference_w, bandwidth=0.8, backend="numpy")
    actual = KernelDensityEstimator(weights=ww, bandwidth=0.8, backend="torch", device="cpu").fit(xx)
    np.testing.assert_allclose(actual.pdf(qq).numpy(), expected.pdf(reference_q), rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(actual.logpdf(qq).numpy(), expected.logpdf(reference_q), rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_zero_weight_gaussian_tail_agrees_with_positive_weight_refit(torch, backend):
    # The closest row has no mass. Centering logsumexp at that row used to
    # underflow all positive-weight terms even though the log density is finite.
    samples = np.array([-0.2, 0.0, 0.2, 90.0])
    weights = np.array([1.0, 2.0, 3.0, 0.0])
    query = np.array([90.0, 100.0])
    model = KernelDensityEstimator(weights=weights, bandwidth=0.4, backend=backend, device="cpu").fit(samples)
    positive = fit_kde(samples[:3], weights=weights[:3], bandwidth=0.4, backend="numpy")
    actual = model.logpdf(query)
    if backend == "torch":
        actual = actual.numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, positive.logpdf(query), rtol=2e-13, atol=1e-7)


def test_weighted_gaussian_matches_scipy_and_vector_query(torch):
    stats = pytest.importorskip("scipy.stats")
    x, q, w = data(3, "zeros")
    model = KernelDensityEstimator(weights=torch.asarray(w), bandwidth=0.7, backend="torch", device="cpu").fit(torch.asarray(x))
    reference = stats.gaussian_kde(x[w > 0].T, weights=w[w > 0], bw_method=0.7)
    np.testing.assert_allclose(model.pdf(q).numpy(), reference(q.T), rtol=2e-10, atol=1e-13)
    np.testing.assert_allclose(model.logpdf(q).numpy(), reference.logpdf(q.T), rtol=2e-10, atol=1e-9)
    np.testing.assert_allclose(model.pdf(q[0]).numpy(), model.pdf(q[:1]).numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(model.logpdf(q[0].tolist()).numpy(), model.logpdf(q[:1]).numpy(), rtol=0, atol=0)
    assert model.pdf(np.empty((0, 3))).shape == (0,)


def test_torch_fit_refit_density_and_score_never_export_numeric_arrays(torch, monkeypatch):
    x, q, w = data(3, "zeros")
    dx, dq, dw = [torch.asarray(v) for v in (x, q, w)]
    expected = fit_kde(x, weights=w, bandwidth=0.8, backend="numpy")
    def forbidden(*args, **kwargs):
        raise AssertionError("native weighted KDE attempted host-array conversion")
    original_asarray = np.asarray
    def guard_asarray(value, *args, **kwargs):
        if isinstance(value, torch.Tensor):
            forbidden()
        return original_asarray(value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "numpy", forbidden)
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(_kde, "_to_numpy", forbidden)
        patch.setattr(_kernel_common, "_to_numpy", forbidden)
        patch.setattr(np, "asarray", guard_asarray)
        model = KernelDensityEstimator(weights=dw, bandwidth=0.8, backend="torch", device="cpu").fit(dx)
        pdf = model.pdf(dq)
        logpdf = model.logpdf(dq)
        score = model.score(dq)
        model.fit(dx)  # Exercise repeated fit.
        refit = model.pdf(dq)
    np.testing.assert_allclose(pdf.numpy(), expected.pdf(q), rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(logpdf.numpy(), expected.logpdf(q), rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(refit.numpy(), pdf.numpy(), rtol=0, atol=0)
    assert score == pytest.approx(float(expected.score(q)), rel=2e-12)


@pytest.mark.parametrize("weights,match", [
    ([1, 2], "same length"), ([0, 0, 0], "positive"),
    ([1, -1, 2], "non-negative"), ([1, math.nan, 2], "finite"),
    ([1, math.inf, 2], "finite"), ([1e308, 1e308, 1e308], "positive"),
    ([0, 0, 1], "degrees of freedom"),
])
def test_torch_weight_validation(torch, weights, match):
    with pytest.raises(ValueError, match=match):
        KernelDensityEstimator(weights=torch.asarray(weights, dtype=torch.float64), bandwidth=0.8, backend="torch", device="cpu").fit([0, 1, 2])


@pytest.mark.parametrize("kernel", ["cosine", "optcosine"])
def test_torch_one_dimensional_kernel_restrictions(torch, kernel):
    with pytest.raises(ValueError, match="only 1D"):
        KernelDensityEstimator(kernel=kernel, bandwidth=0.8, backend="torch", device="cpu").fit(np.arange(12).reshape(6, 2))


@pytest.mark.parametrize("device,backend", [("torch", "numpy"), ("cuda", "numpy"), ("torch", "cupy"), ("cuda", "torch"), ("cpu", "cupy")])
def test_conflicting_explicit_device_backend_is_rejected(device, backend):
    with pytest.raises(ValueError, match="requires backend|incompatible"):
        KernelDensityEstimator(device=device, backend=backend, bandwidth=0.8).fit([0, 1, 2])


@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_explicit_torch_device_never_falls_back_to_cpu(torch, monkeypatch, backend):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*no CPU fallback"):
        KernelDensityEstimator(device="torch", backend=backend, bandwidth=0.8).fit([0, 1, 2])


def test_explicit_cupy_device_never_falls_back_to_torch_or_numpy(monkeypatch):
    import statgpu.backends._factory as factory
    monkeypatch.setattr(factory._cupy_backend, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*no CPU fallback"):
        KernelDensityEstimator(device="cuda", bandwidth=0.8).fit([0, 1, 2])


def test_global_explicit_torch_device_is_enforced(torch, monkeypatch):
    from statgpu._config import Device, _device_manager
    monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*no CPU fallback"):
        fit_kde([0, 1, 2], bandwidth=0.8, backend="torch")


def test_functional_torch_backend_preserves_torch_cpu_device(torch):
    model = fit_kde(torch.arange(6), weights=[0, 1, 2, 3, 4, 5], backend="torch", bandwidth=0.8)
    assert model.samples_.device.type == "cpu"
    assert model.pdf([1, 2]).device.type == "cpu"


@pytest.fixture
def torch_cuda(torch):
    if not torch.cuda.is_available():
        pytest.skip("physical Torch CUDA is required; CPU algebra tests are separate")
    return torch


@pytest.mark.parametrize("kernel,dimension", KERNEL_DIMENSIONS)
@pytest.mark.parametrize("input_kind", ["host", "native"])
def test_physical_torch_cuda_weighted_fit_and_native_device_affinity(torch_cuda, monkeypatch, kernel, dimension, input_kind):
    torch = torch_cuda
    x, q, w = data(dimension, "zeros")
    expected = fit_kde(x, weights=w, bandwidth=0.8, kernel=kernel, backend="numpy")
    device = torch.device("cuda", torch.cuda.current_device())
    dx, dq, dw = [torch.asarray(v, device=device) for v in (x, q, w)]
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA KDE exported a numeric array")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        model = KernelDensityEstimator(
            weights=w if input_kind == "host" else dw, bandwidth=0.8,
            kernel=kernel, backend="torch", device="torch",
        ).fit(x if input_kind == "host" else dx)
        pdf, logpdf = model.pdf(dq), model.logpdf(dq)
    for value in (model.samples_, model.weights_, model.covariance_, model.inv_covariance_, pdf, logpdf):
        assert value.device == device and value.dtype == torch.float64
    tolerance = 3e-7 if kernel == "triangular" else 2e-10
    np.testing.assert_allclose(pdf.cpu().numpy(), expected.pdf(q), rtol=tolerance, atol=2e-11)
    np.testing.assert_allclose(logpdf.cpu().numpy(), expected.logpdf(q), rtol=tolerance, atol=2e-9)


@pytest.mark.parametrize("dimension", [1, 8])
def test_torch_zero_weight_nearest_row_has_no_gaussian_tail_mass(torch, dimension):
    samples, _, weights = data(dimension, "unequal")
    samples = np.concatenate([samples, np.full((1, dimension), 90.0)])
    weights = np.r_[weights, 0.0]
    query = np.full((2, dimension), 90.0)
    expected = fit_kde(samples[:-1], weights=weights[:-1], bandwidth=0.4, backend="numpy")
    actual = fit_kde(torch.asarray(samples), weights=torch.asarray(weights), bandwidth=0.4, backend="torch")
    values = actual.logpdf(torch.asarray(query)).numpy()
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, expected.logpdf(query), rtol=2e-12, atol=2e-8)


@pytest.mark.parametrize("input_name", ["samples", "weights", "points"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_real_valued_input_contract_is_enforced(torch, input_name, backend):
    samples = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 3.0])
    points = np.array([0.2, 0.7])
    if input_name == "samples":
        samples = samples.astype(complex) + 1j
    elif input_name == "weights":
        weights = weights.astype(complex) + 1j
    else:
        points = points.astype(complex) + 1j
    with pytest.raises(ValueError, match="real-valued"):
        model = KernelDensityEstimator(weights=weights, bandwidth=0.5, backend=backend, device="cpu").fit(samples)
        model.pdf(points)


def test_physical_torch_cuda_relocates_queries_weights_and_preserves_training_device(torch_cuda):
    torch = torch_cuda
    if torch.cuda.device_count() < 2:
        pytest.skip("two physical CUDA devices required for native relocation test")
    x, q, w = data(2, "zeros")
    expected = fit_kde(x, weights=w, bandwidth=0.8, backend="numpy")
    with torch.cuda.device(0):
        dx = torch.asarray(x, device="cuda:1")
        dq = torch.asarray(q, device="cuda:0")
        dw = torch.asarray(w, device="cuda:0")
        model = KernelDensityEstimator(weights=dw, bandwidth=0.8, backend="torch", device="torch").fit(dx)
        output = model.pdf(dq)
        assert torch.cuda.current_device() == 0
    for value in (model.samples_, model.weights_, model.covariance_, output):
        assert value.device == dx.device
    assert dq.device.index == 0 and dw.device.index == 0
    np.testing.assert_allclose(output.cpu().numpy(), expected.pdf(q), rtol=2e-10, atol=2e-12)


def test_physical_cupy_relocates_queries_weights_and_restores_ambient_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("two physical CUDA devices required for native relocation test")
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"physical CUDA unavailable: {error}")
    x, q, w = data(2, "zeros")
    expected = fit_kde(x, weights=w, bandwidth=0.8, backend="numpy")
    with cp.cuda.Device(1):
        dx = cp.asarray(x)
    with cp.cuda.Device(0):
        dq, dw = cp.asarray(q), cp.asarray(w)
        model = KernelDensityEstimator(weights=dw, bandwidth=0.8, backend="cupy", device="cuda").fit(dx)
        output = model.pdf(dq)
        assert cp.cuda.Device().id == 0
    for value in (model.samples_, model.weights_, model.covariance_, output):
        assert value.device.id == dx.device.id
    assert dq.device.id == 0 and dw.device.id == 0
    np.testing.assert_allclose(cp.asnumpy(output), expected.pdf(q), rtol=2e-10, atol=2e-12)


def test_numpy_metadata_exports_native_projection_without_mutating_selector(torch):
    x, _, w = data(3, "unequal")
    model = KernelDensityEstimator(weights=torch.asarray(w), bandwidth="nrd", backend="torch", device="cpu").fit(torch.asarray(x))
    original = model.bandwidth_info_.details["projection_vector"]
    assert isinstance(original, torch.Tensor)
    before = original.clone()
    metadata = model.to_numpy_metadata()
    projection = metadata["bandwidth_selection"]["details"]["projection_vector"]
    assert isinstance(projection, np.ndarray)
    np.testing.assert_array_equal(projection, before.numpy())
    assert model.bandwidth_info_.details["projection_vector"] is original
    assert torch.equal(original, before)
