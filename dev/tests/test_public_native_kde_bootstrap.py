"""Public KDE bootstrap acceptance: real backends, never GPU emulation.

Torch CPU is useful numerical coverage, not CUDA evidence. Physical CUDA tests
skip explicitly when their library/device is unavailable. Fixed indices isolate
refitting semantics from deliberately different backend RNG algorithms.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import importlib

import numpy as np
import pytest

import statgpu
from statgpu.nonparametric import fit_kde, kde_bootstrap_confidence_interval, kde_confidence_interval

SELECTORS = ('scott', 'silverman', 'nrd', 'nrd0', 'ucv', 'bcv', 'sj', 'sj-ste', 'sj-dpi')
KERNELS = ('gaussian', 'rectangular', 'triangular', 'epanechnikov', 'biweight', 'triweight', 'cosine', 'optcosine')
FIELDS = ('estimate', 'lower', 'upper', 'bootstrap_samples')
INGRESS_CONTAINERS = ('list', 'tuple', 'numpy', 'torch-cpu', 'native-strided')


class Backend:
    def __init__(self, name, module, device):
        self.name, self.xp, self.device = name, module, device

    def array(self, value, dtype=None):
        if value is None:
            return None
        if self.name == 'torch':
            return self.xp.as_tensor(value, dtype=dtype or self.xp.float64, device=self.device)
        return self.xp.asarray(value, dtype=dtype or self.xp.float64)

    def host(self, value):
        if self.name == 'torch':
            return value.detach().cpu().numpy()
        if self.name == 'cupy':
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def context(self):
        if self.name == 'cupy':
            return self.xp.cuda.Device(int(self.device.split(':')[1]))
        if self.name == 'torch' and self.device != 'cpu':
            return self.xp.cuda.device(self.device)
        return nullcontext()


@contextmanager
def requested_backend_device(backend, monkeypatch):
    """Request the tier publicly, then restore even an unresolved AUTO setting."""
    from statgpu import _config
    with monkeypatch.context() as local:
        # get_device() resolves AUTO and therefore cannot restore caller policy.
        local.setattr(_config._device_manager, '_current_device', _config._get_configured_device())
        statgpu.set_device('cpu' if backend.device == 'cpu' else
                          'torch' if backend.name == 'torch' else 'cuda')
        yield


def assert_backend_array(value, backend, dtype=None):
    dtype = backend.xp.float64 if dtype is None else dtype
    if backend.name == 'torch':
        assert backend.xp.is_tensor(value)
        assert value.device == backend.xp.device(backend.device)
    else:
        assert isinstance(value, backend.xp.ndarray)
        if backend.name == 'cupy':
            assert value.device.id == int(backend.device.split(':')[1])
    assert value.dtype == dtype


def ingress_array(value, container, backend):
    if value is None:
        return None
    if container == 'list':
        return value.tolist()
    if container == 'tuple':
        return tuple(map(tuple, value)) if value.ndim == 2 else tuple(value)
    if container == 'numpy':
        return value
    if container == 'torch-cpu':
        torch = pytest.importorskip('torch', reason='Torch CPU ingress requires real Torch')
        return torch.as_tensor(value, dtype=torch.float64, device='cpu')
    buffer = np.zeros((len(value)*2,) + value.shape[1:])
    buffer[::2] = value
    result = backend.array(buffer)[::2]
    if backend.name == 'torch':
        assert not result.is_contiguous()
    else:
        assert not result.flags.c_contiguous
    return result


def observe_ingress_devices(monkeypatch, backend):
    """Observe real arrays without converting numerical stages back to the host."""
    from statgpu.nonparametric.kernel_smoothing import _kde as kde, _native_bootstrap as native
    observed = dict(models=[], points=[], cdfs=[], indices=[], generators=[])
    original_fit, original_points = kde.fit_kde, kde._as_points_2d
    original_cdf, original_draw, original_rng = native._probability_cdf, native._draw_indices, native._make_rng

    def fit(*args, **kwargs):
        model = original_fit(*args, **kwargs)
        assert model.backend_ == backend.name
        assert_backend_array(model.samples_, backend)
        assert_backend_array(model.weights_, backend)
        observed['models'].append(model)
        return model

    def points(*args, **kwargs):
        assert_backend_array(kwargs['ref_arr'], backend)
        result = original_points(*args, **kwargs)
        assert_backend_array(result, backend)
        observed['points'].append(result)
        return result

    def cdf(weights, xp):
        assert xp is backend.xp
        assert_backend_array(weights, backend)
        result = original_cdf(weights, xp)
        assert_backend_array(result, backend)
        observed['cdfs'].append(result)
        return result

    def rng(random_state, xp, reference):
        assert xp is backend.xp
        assert_backend_array(reference, backend)
        result = original_rng(random_state, xp, reference)
        if backend.name == 'torch':
            assert result.device == backend.xp.device(backend.device)
        observed['generators'].append(result)
        return result

    def draw(cdf, rng, n_samples, xp):
        assert xp is backend.xp
        assert_backend_array(cdf, backend)
        result = original_draw(cdf, rng, n_samples, xp)
        assert_backend_array(result, backend, xp.int64)
        assert result.shape == (n_samples,)
        observed['indices'].append(result)
        return result

    monkeypatch.setattr(kde, 'fit_kde', fit)
    monkeypatch.setattr(kde, '_as_points_2d', points)
    monkeypatch.setattr(native, '_probability_cdf', cdf)
    monkeypatch.setattr(native, '_make_rng', rng)
    monkeypatch.setattr(native, '_draw_indices', draw)
    return observed


def assert_ingress_observed(observed, backend, dimension, resamples):
    native = backend.name != 'numpy'
    # NumPy's historical 1D Gaussian shortcut does not construct refit models.
    expected_models = 1 if not native and dimension == 1 else resamples + 1
    assert len(observed['models']) == expected_models
    assert len(observed['points']) == expected_models + 1
    assert len(observed['cdfs']) == int(native)
    assert len(observed['generators']) == int(native)
    assert len(observed['indices']) == (resamples if native else 0)


@pytest.fixture(params=('numpy', 'torch-cpu', 'cupy-cuda', 'torch-cuda'))
def backend(request):
    if request.param == 'numpy':
        return Backend('numpy', np, 'cpu')
    name = request.param.split('-')[0]
    xp = pytest.importorskip(name, reason=name + ' is not installed; no emulation')
    if request.param == 'torch-cpu':
        return Backend('torch', xp, 'cpu')
    if name == 'torch':
        if not xp.cuda.is_available():
            pytest.skip('physical Torch CUDA unavailable; Torch CPU is a separate test tier')
        device = 'cuda:' + str(xp.cuda.current_device())
        xp.cuda.synchronize(device=device)
    else:
        try:
            if xp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip('physical CuPy CUDA unavailable')
            device = 'cuda:' + str(xp.cuda.runtime.getDevice())
            xp.cuda.Device(int(device.split(':')[1])).synchronize()
        except xp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip('physical CuPy CUDA unavailable: ' + str(exc))
    return Backend(name, xp, device)


def fixture_data(dimension=1, weights='unequal'):
    rng = np.random.default_rng(517)
    x = rng.normal(size=(24, dimension))
    q = rng.normal(scale=.8, size=(7, dimension))
    w = None if weights == 'none' else np.linspace(.3, 2., len(x))
    if weights == 'sparse':
        w[::4] = 0.
    # Deterministic, deliberately different covariances and unequal-weight fits.
    positive = np.arange(len(x)) if w is None else np.flatnonzero(w)
    indices = np.stack([np.resize(np.roll(positive, k), len(x)) for k in (1, 5, 11)])
    indices[1, :8] = positive[3]
    indices[2, 10:] = positive[7]
    return (x[:, 0] if dimension == 1 else x,
            q[:, 0] if dimension == 1 else q, w, indices)


def oracle(x, q, w, indices, bandwidth=.55, kernel='gaussian', level=.9, numpy_fast=False):
    probabilities = np.full(len(x), 1. / len(x)) if w is None else w / np.sum(w)
    original = fit_kde(x, weights=w, bandwidth=bandwidth, kernel=kernel, backend='numpy')
    # Historical CPU 1D Gaussian shortcut fixes the original selected factor.
    draw_bandwidth = original.bandwidth_factor_ if numpy_fast else bandwidth
    draws = np.stack([fit_kde(x[row], weights=probabilities[row], bandwidth=draw_bandwidth,
        kernel=kernel, backend='numpy').pdf(q) for row in indices])
    lower, upper = np.quantile(draws, [(1.-level)/2., (1.+level)/2.], axis=0)
    return dict(estimate=original.pdf(q), lower=lower, upper=upper, bootstrap_samples=draws)


def install_fixed_indices(monkeypatch, backend, indices):
    calls = []
    if backend.name == 'numpy':
        class FixedGenerator:
            def choice(self, n, size=None, replace=True, p=None):
                assert n == len(indices[0]) and size == n and replace
                row = indices[len(calls)].copy()
                calls.append(row)
                return row
        monkeypatch.setattr(np.random, 'default_rng', lambda *a, **k: FixedGenerator())
    else:
        module = importlib.import_module('statgpu.nonparametric.kernel_smoothing._native_bootstrap')
        def draw(cdf, rng, n_samples, xp):
            assert xp is backend.xp and n_samples == len(indices[0])
            assert_backend_array(cdf, backend)
            row = indices[len(calls)].copy()
            calls.append(row)
            result = backend.array(row, dtype=backend.xp.int64)
            assert_backend_array(result, backend, backend.xp.int64)
            return result
        monkeypatch.setattr(module, '_draw_indices', draw)
    return calls


def assert_result(result, expected, n, q, backend, rtol=3e-8, atol=2e-10):
    assert result.backend == backend.name
    assert result.n_resamples == n and result.confidence_level == .9
    for field in FIELDS:
        value = getattr(result, field)
        assert isinstance(value, np.ndarray) and value.dtype == np.float64
        assert value.shape == ((n, q) if field == 'bootstrap_samples' else (q,))
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, expected[field], rtol=rtol, atol=atol)
    assert np.all(result.lower <= result.upper)
    assert np.all(result.bootstrap_samples >= 0.)


@pytest.mark.parametrize('kernel,dimension', [(k, d) for k in KERNELS for d in (1, 2, 4)
    if d == 1 or k not in ('cosine', 'optcosine')])
@pytest.mark.parametrize('weights', ('none', 'unequal', 'sparse'))
def test_public_numeric_kernel_dimension_weight_matrix(backend, monkeypatch, kernel, dimension, weights):
    x, q, w, indices = fixture_data(dimension, weights)
    expected = oracle(x, q, w, indices, kernel=kernel)
    with backend.context():
        arrays = [backend.array(value) for value in (x, q, w)]
        before = [None if value is None else backend.host(value).copy() for value in arrays]
        calls = install_fixed_indices(monkeypatch, backend, indices)
        actual = kde_bootstrap_confidence_interval(*arrays[:2], weights=arrays[2], bandwidth=.55,
            kernel=kernel, backend=backend.name, n_resamples=len(indices), confidence_level=.9,
            random_state=182, return_bootstrap_samples=True, batch_size=3)
        assert_result(actual, expected, len(indices), len(q), backend)
        assert len(calls) == len(indices)
        for value, original in zip(arrays, before):
            if original is not None:
                np.testing.assert_array_equal(backend.host(value), original)


@pytest.mark.parametrize('selector', SELECTORS)
@pytest.mark.parametrize('kernel', ('gaussian','epanechnikov'))
@pytest.mark.parametrize('dimension', (1, 2))
@pytest.mark.parametrize('weights', ('none', 'unequal', 'sparse'))
def test_public_every_selector_refits_sampled_rows(backend, monkeypatch, selector, kernel, dimension, weights):
    x, q, w, indices = fixture_data(dimension, weights)
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    with backend.context():
        calls = install_fixed_indices(monkeypatch, backend, indices)
        def call():
            return kde_bootstrap_confidence_interval(backend.array(x), backend.array(q),
                weights=backend.array(w), bandwidth=selector, kernel=kernel, backend=backend.name,
                n_resamples=len(indices), confidence_level=.9, random_state=4,
                return_bootstrap_samples=True, batch_size=4)
        actual, records = bench.capture_selector_models(call,backend)
    if backend.name == 'numpy':
        expected = oracle(x,q,w,indices,bandwidth=selector,kernel=kernel,numpy_fast=dimension == 1 and kernel == 'gaussian')
    else:
        problem = bench.Problem(n=len(x),queries=len(q),resamples=3,dimension=dimension,
            bandwidth=selector,kernel=kernel,confidence_level=.9)
        expected, _ = bench.reference_for_indices(dict(samples=x,points=q,weights=w),problem,
            indices,backend.name,records)
        expected['bootstrap_samples'] = expected.pop('draws')
    # Independent NumPy scalar rules align the actual native PCA/tie policy.
    assert_result(actual, expected, len(indices), len(q), backend, rtol=3e-5, atol=3e-7)
    assert len(calls) == len(indices)


@pytest.mark.parametrize('container', INGRESS_CONTAINERS)
@pytest.mark.parametrize('dimension', (1, 2, 3))
@pytest.mark.parametrize('weights', ('none', 'unequal', 'sparse'))
def test_ingress_and_noncontiguous_arrays(backend, monkeypatch, container, dimension, weights):
    x, q, w, indices = fixture_data(dimension, weights)
    # The NumPy oracle must run before the explicit global GPU request.
    expected = oracle(x, q, w, indices)
    with backend.context(), requested_backend_device(backend, monkeypatch), monkeypatch.context() as local:
        arrays = [ingress_array(value, container, backend) for value in (x, q, w)]
        calls = install_fixed_indices(local, backend, indices)
        observed = observe_ingress_devices(local, backend)
        actual = kde_bootstrap_confidence_interval(*arrays[:2], weights=arrays[2], bandwidth=.55,
            backend=backend.name, n_resamples=3, confidence_level=.9,
            return_bootstrap_samples=True, batch_size=2)
        assert_ingress_observed(observed, backend, dimension, 3)
        assert len(calls) == 3
    assert_result(actual, expected, 3, len(q), backend)


@pytest.mark.parametrize('container', INGRESS_CONTAINERS)
@pytest.mark.parametrize('dimension', (1, 2, 3))
@pytest.mark.parametrize('weights', ('none', 'unequal', 'sparse'))
@pytest.mark.parametrize('backend', ('torch-cpu', 'cupy-cuda', 'torch-cuda'), indirect=True)
def test_native_ingress_real_rng_keeps_every_stage_on_requested_device(
        backend, monkeypatch, container, dimension, weights):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from statgpu.nonparametric.kernel_smoothing import _kde as kde, _native_bootstrap as native
    x, q, w, _ = fixture_data(dimension, weights)
    runtime = bench.PublicRuntime(backend.name, torch_cpu=backend.device == 'cpu',
        ordinal=0 if backend.device == 'cpu' else int(backend.device.split(':')[1]))
    problem = bench.Problem(n=len(x), queries=len(q), resamples=3, dimension=dimension,
        bandwidth=.55, confidence_level=.9, weights=weights)
    # Capture production source identities before installing observing wrappers.
    specs = [bench.stage(native.native_bootstrap, 'native-bootstrap', ('exp', 'quantile')),
        bench.stage(kde.KernelDensityEstimator.fit, 'native-fit', ('sum', 'linalg.inv'), minimum_calls=4),
        bench.stage(native._probability_cdf, 'probability-cdf', ('cumsum',)),
        bench.stage(native._draw_indices, 'native-weighted-rng', ('searchsorted',), minimum_calls=3)]
    source_audit = bench.audit_native_export_sources()
    with backend.context(), requested_backend_device(backend, monkeypatch), monkeypatch.context() as local:
        arrays = [ingress_array(value, container, backend) for value in (x, q, w)]
        observed = observe_ingress_devices(local, backend)
        with bench.PublicTrace(runtime, specs, problem) as trace:
            trace.source_audit = source_audit
            actual = kde_bootstrap_confidence_interval(*arrays[:2], weights=arrays[2], bandwidth=.55,
                backend=backend.name, n_resamples=3, confidence_level=.9, random_state=182,
                return_bootstrap_samples=True, batch_size=2)
        runtime.sync(runtime.device)
        report = trace.report()
        assert report['status'] == 'observed'
        assert report['host_conversion_guard']['status'] == 'pass'
        assert len(report['rng_generation']['outputs']) == 3
        assert_ingress_observed(observed, backend, dimension, 3)
    # Only the independent oracle exports observed rows, after native execution.
    indices = np.stack([backend.host(row) for row in observed['indices']])
    expected = bench.gaussian_reference(x, q, w, indices, .55, .9)
    expected['bootstrap_samples'] = expected.pop('draws')
    assert_result(actual, expected, 3, len(q), backend)


@pytest.mark.parametrize('container', ('list', 'tuple', 'numpy'))
@pytest.mark.parametrize('weights', ('none', 'unequal'))
def test_explicit_torch_host_ingress_with_auto_policy_stays_on_cpu(monkeypatch, container, weights):
    """An explicit array library is not an implicit request for CUDA."""
    from statgpu import _config
    from dev.benchmarks.benchmark_public_native_kde_bootstrap import gaussian_reference
    torch = pytest.importorskip('torch', reason='Torch CPU contract requires real Torch')
    cpu = Backend('torch', torch, 'cpu')
    x, q, w, _ = fixture_data(2, weights)
    arrays = [ingress_array(value, container, cpu) for value in (x, q, w)]
    with monkeypatch.context() as local:
        local.setattr(_config._device_manager, '_current_device', _config._get_configured_device())
        statgpu.set_device('auto')
        observed = observe_ingress_devices(local, cpu)
        actual = kde_bootstrap_confidence_interval(*arrays[:2], weights=arrays[2], bandwidth=.55,
            backend='torch', n_resamples=3, confidence_level=.9, random_state=182,
            return_bootstrap_samples=True, batch_size=2)
        assert_ingress_observed(observed, cpu, 2, 3)
    indices = np.stack([cpu.host(row) for row in observed['indices']])
    expected = gaussian_reference(x, q, w, indices, .55, .9)
    expected['bootstrap_samples'] = expected.pop('draws')
    assert_result(actual, expected, 3, len(q), cpu)


@pytest.mark.parametrize('raise_inside', (False, True))
def test_requested_backend_device_restores_unresolved_auto(monkeypatch, raise_inside):
    from statgpu import _config
    with monkeypatch.context() as local:
        local.setattr(_config._device_manager, '_current_device', _config._get_configured_device())
        statgpu.set_device('auto')
        expected_exception = pytest.raises(RuntimeError, match='scope sentinel') if raise_inside else nullcontext()
        with expected_exception:
            with requested_backend_device(Backend('numpy', np, 'cpu'), monkeypatch):
                assert _config._get_configured_device() is _config.Device.CPU
                if raise_inside:
                    raise RuntimeError('scope sentinel')
        assert _config._get_configured_device() is _config.Device.AUTO


@pytest.mark.parametrize('seed', (0, 42, 2**96 + 271, np.int64(811)))
def test_rng_reproducibility_chunk_invariance_and_global_isolation(backend, seed):
    x, q, w, _ = fixture_data(weights='sparse')
    with backend.context():
        args = (backend.array(x), backend.array(q))
        kwargs = dict(weights=backend.array(w), bandwidth=.55, backend=backend.name,
            random_state=seed, n_resamples=5, confidence_level=.9, return_bootstrap_samples=True)
        numpy_state = np.random.get_state()
        torch_state = backend.xp.random.get_rng_state().clone() if backend.name == 'torch' else None
        if backend.name == 'torch' and backend.device != 'cpu':
            cuda_state = backend.xp.cuda.get_rng_state(backend.device).clone()
        first = kde_bootstrap_confidence_interval(*args, batch_size=1, **kwargs)
        second = kde_bootstrap_confidence_interval(*args, batch_size=7, **kwargs)
        after = np.random.get_state()
        assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
        np.testing.assert_array_equal(numpy_state[1], after[1])
        if torch_state is not None:
            assert backend.xp.equal(torch_state, backend.xp.random.get_rng_state())
        if backend.name == 'torch' and backend.device != 'cpu':
            assert backend.xp.equal(cuda_state, backend.xp.cuda.get_rng_state(backend.device))
    for field in FIELDS:
        np.testing.assert_allclose(getattr(first, field), getattr(second, field), rtol=3e-12, atol=3e-13)


def test_generator_argument_advances_only_declared_seed_material(backend):
    if backend.name == 'numpy':
        pytest.skip('NumPy keeps historical choice draws; native seed-material contract is separate')
    x, q, _, _ = fixture_data()
    generator = np.random.default_rng(75)
    replay = np.random.default_rng(75)
    replay.integers(0, 2**32, size=4, dtype=np.uint32)
    with backend.context():
        kde_bootstrap_confidence_interval(backend.array(x), backend.array(q), backend=backend.name,
            bandwidth=.55, random_state=generator, n_resamples=3)
    assert generator.bit_generator.state == replay.bit_generator.state


@pytest.mark.parametrize('retained', (False, True))
def test_general_wrapper_and_public_metadata(backend, retained):
    x, q, w, _ = fixture_data(2)
    with backend.context():
        args = (backend.array(x), backend.array(q))
        kwargs = dict(weights=backend.array(w), bandwidth=.55, backend=backend.name, n_resamples=3,
            random_state=404, return_bootstrap_samples=retained, batch_size=2)
        direct = kde_bootstrap_confidence_interval(*args, **kwargs)
        general = kde_confidence_interval(*args, method='bootstrap', **kwargs)
    for field in ('points', 'estimate', 'lower', 'upper'):
        np.testing.assert_array_equal(getattr(direct, field), getattr(general, field))
    assert direct.points.shape == q.shape
    assert direct.metadata['method'] == 'bootstrap' and direct.metadata['n_features'] == 2
    assert (direct.bootstrap_samples is not None) == retained
    if retained:
        np.testing.assert_array_equal(direct.bootstrap_samples, general.bootstrap_samples)


@pytest.mark.parametrize('invalid', ({'n_resamples': 0}, {'confidence_level': np.nan},
    {'confidence_level': 1.}, {'method': 'bca'}, {'bandwidth': 0.}, {'batch_size': 0},
    {'random_state': -1}, {'random_state': 1.5}, {'weights': [0.]*24},
    {'weights': [-1.]+[1.]*23}))
def test_public_invalid_inputs_fail_explicitly(backend, invalid):
    x, q, _, _ = fixture_data()
    with backend.context(), pytest.raises((ValueError, TypeError)):
        kwargs = dict(bandwidth=.55, backend=backend.name, n_resamples=2)
        kwargs.update(invalid)
        kde_bootstrap_confidence_interval(backend.array(x), backend.array(q), **kwargs)


@pytest.mark.parametrize('kernel', ('cosine', 'optcosine'))
def test_one_dimensional_kernel_boundary(backend, kernel):
    x, q, _, _ = fixture_data(2)
    with backend.context(), pytest.raises(ValueError, match='1D'):
        kde_bootstrap_confidence_interval(backend.array(x), backend.array(q),
            kernel=kernel, backend=backend.name, n_resamples=2)


def test_native_numeric_gaussian_matches_independent_covariance_formula(backend, monkeypatch):
    from dev.benchmarks.benchmark_public_native_kde_bootstrap import gaussian_reference
    x, q, w, indices = fixture_data(3)
    expected = gaussian_reference(x, q, w, indices, .55, .9)
    with backend.context():
        install_fixed_indices(monkeypatch, backend, indices)
        result = kde_bootstrap_confidence_interval(backend.array(x), backend.array(q),
            weights=backend.array(w), bandwidth=.55, backend=backend.name,
            n_resamples=3, confidence_level=.9, return_bootstrap_samples=True)
    for key, field in (('estimate', 'estimate'), ('lower', 'lower'), ('upper', 'upper'), ('draws', 'bootstrap_samples')):
        np.testing.assert_allclose(getattr(result, field), expected[key], rtol=3e-8, atol=2e-10)


@pytest.mark.parametrize('query_count', (0, 1))
@pytest.mark.parametrize('resamples', (1, 3))
def test_empty_and_singleton_query_and_replicate_shapes(backend, query_count, resamples):
    x, q, w, _ = fixture_data()
    with backend.context():
        result = kde_bootstrap_confidence_interval(backend.array(x), backend.array(q[:query_count]),
            weights=backend.array(w), bandwidth=.55, backend=backend.name, random_state=123,
            n_resamples=resamples, return_bootstrap_samples=True)
    assert result.points.shape == (query_count,)
    for field in ('estimate', 'lower', 'upper'):
        assert getattr(result, field).shape == (query_count,)
    assert result.bootstrap_samples.shape == (resamples, query_count)
    if resamples == 1:
        np.testing.assert_array_equal(result.lower, result.bootstrap_samples[0])
        np.testing.assert_array_equal(result.upper, result.bootstrap_samples[0])


def test_native_workspace_rejects_before_random_draw_and_preserves_problem(backend, monkeypatch):
    if backend.name == 'numpy':
        pytest.skip('managed native bootstrap workspace, not a NumPy memory contract')
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    x, q, w, _ = fixture_data()
    with backend.context():
        def forbidden(*args, **kwargs):
            raise AssertionError('RNG must not run before budget rejection')
        with monkeypatch.context() as local:
            local.setattr(native, '_draw_indices', forbidden)
            with pytest.raises(MemoryError, match='workspace'):
                kde_bootstrap_confidence_interval(backend.array(x), backend.array(q),
                    weights=backend.array(w), backend=backend.name, bandwidth=.55,
                    n_resamples=3, workspace_bytes=1)
        minimum = native.plan_workspace(len(x), 1, len(q), 3, 1)['estimated_managed_bytes']
        options = dict(weights=backend.array(w), backend=backend.name, bandwidth=.55,
            random_state=737, n_resamples=3, return_bootstrap_samples=True)
        narrow = kde_bootstrap_confidence_interval(backend.array(x), backend.array(q),
            workspace_bytes=minimum, **options)
        wide = kde_bootstrap_confidence_interval(backend.array(x), backend.array(q), **options)
    assert narrow.n_resamples == 3 and narrow.bootstrap_samples.shape == (3, len(q))
    assert narrow.metadata['query_chunk'] == 1
    for field in FIELDS:
        np.testing.assert_allclose(getattr(narrow,field), getattr(wide,field), rtol=3e-12, atol=3e-13)


def test_workspace_accounts_for_selector_managed_arrays():
    from statgpu.nonparametric.kernel_smoothing._native_bootstrap import plan_workspace
    numeric = plan_workspace(100, 2, 7, 3, 1, bandwidth=.5)
    selector = plan_workspace(100, 2, 7, 3, 1, bandwidth='sj-dpi')
    assert selector['selector_reserve_bytes'] > 0
    assert selector['estimated_managed_bytes'] > numeric['estimated_managed_bytes']
    with pytest.raises(MemoryError):
        plan_workspace(100, 2, 7, 3, 1, workspace_bytes=numeric['estimated_managed_bytes'], bandwidth='sj-dpi')


def test_production_package_runs_without_dev_tree(tmp_path):
    """An isolated copied package must not import either benchmark candidate."""
    import pathlib
    import shutil
    import subprocess
    import sys
    package = pathlib.Path(__file__).resolve().parents[2] / 'statgpu'
    shutil.copytree(package, tmp_path/'statgpu', ignore=shutil.ignore_patterns('__pycache__'))
    code = """
import importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
class RejectDev(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dev' or fullname.startswith('dev.'):
            raise AssertionError('production imports development code: ' + fullname)
sys.meta_path.insert(0, RejectDev())
from statgpu.nonparametric import kde_bootstrap_confidence_interval
from statgpu.nonparametric.kernel_smoothing import _native_bootstrap
assert pathlib.Path(_native_bootstrap.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]))
import numpy as np
backends = ['numpy']
try:
    import torch
except ImportError:
    pass
else:
    backends.append('torch')
for backend in backends:
    result = kde_bootstrap_confidence_interval([-2.,-1.,.2,.7,1.4,2.], [-.5,.5],
        weights=[1.,2.,0.,3.,2.,1.], bandwidth=.5, backend=backend, random_state=7,
        n_resamples=3, return_bootstrap_samples=True)
    assert result.bootstrap_samples.shape == (3,2)
    assert np.isfinite(result.bootstrap_samples).all()
print('isolated public bootstrap passed: ' + ','.join(backends))
"""
    completed = subprocess.run([sys.executable,'-I','-c',code,str(tmp_path)], cwd=tmp_path,
        capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'isolated public bootstrap passed' in completed.stdout


def test_benchmark_captures_actual_numpy_rng_and_checks_analytic_reference():
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    problem = replace(bench.PRESETS['tiny'], dimension=2, weights='sparse')
    data = bench.make_data(problem)
    identity = bench.data_identity(data)
    reference, correctness, rng = bench.diagnose(bench.PublicRuntime('numpy'), data, problem, 3, None)
    assert correctness['status'] == 'pass' and correctness['reference']['reference_type'] == 'analytic'
    assert rng['same_seed_indices_reproduced'] is True
    assert reference['draws'].shape == (problem.resamples, problem.queries)
    assert bench.data_identity(data) == identity


def test_benchmark_torch_cpu_observes_rng_arithmetic_and_rejects_host_conversion(monkeypatch):
    pytest.importorskip('torch')
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    runtime = bench.PublicRuntime('torch', torch_cpu=True)
    problem = bench.PRESETS['tiny']
    data = bench.make_data(problem)
    reference, _, _ = bench.diagnose(runtime, data, problem, 3, None)
    value, trace = bench.trace_public(runtime, data, problem, 3, None)
    bench.check(value, reference, problem)
    assert trace['device'] == 'cpu' and trace['status'] == 'observed'
    assert len(trace['rng_generation']['outputs']) == problem.resamples
    original = native._draw_indices
    def host_sampling(cdf, rng, n_samples, xp):
        getattr(cdf, 'cpu')()
        return original(cdf, rng, n_samples, xp)
    monkeypatch.setattr(native, '_draw_indices', host_sampling)
    with pytest.raises(bench.TraceIncomplete, match='exported to host'):
        bench.trace_public(runtime, data, problem, 3, None)


def test_benchmark_rejects_fake_native_primitive(monkeypatch):
    torch = pytest.importorskip('torch')
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    original = torch.sum
    def disguised(*args, **kwargs):
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'sum', disguised)
    problem = bench.PRESETS['tiny']
    with pytest.raises(bench.TraceIncomplete, match='Untrusted replacement'):
        bench.trace_public(bench.PublicRuntime('torch',torch_cpu=True), bench.make_data(problem), problem, 3, None)


def test_benchmark_cpu_only_is_partial_and_uninstrumented(monkeypatch):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    factory = np.random.default_rng
    original = bench.make_call
    capture_active = {'value': False}
    original_capture = bench.capture_realized
    def capture(call, runtime):
        capture_active['value'] = True
        try:
            return original_capture(call, runtime)
        finally:
            capture_active['value'] = False
    def make_call(*args):
        call = original(*args)
        def checked():
            if not capture_active['value']:
                assert np.random.default_rng is factory
            return call()
        return checked
    monkeypatch.setattr(bench,'capture_realized',capture)
    monkeypatch.setattr(bench,'make_call',make_call)
    result = bench.run_benchmark(bench.PRESETS['tiny'],cpu_only=True,warmup=0,repeats=1)
    assert result['rows'][0]['status'] == 'pass'
    assert result['validation']['status'] == 'partial'
    assert result['validation']['physical_gpu_claim'] is False
    assert result['performance']['eligible'] is False and result['performance']['comparisons'] == []
    assert all(row['status'] == 'unavailable' for row in result['rows'][1:])


def test_benchmark_failed_correctness_prevents_timing(monkeypatch):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    def fail(*args):
        raise AssertionError('oracle rejects')
    monkeypatch.setattr(bench, 'diagnose', fail)
    result = bench.run_benchmark(bench.PRESETS['tiny'],cpu_only=True)
    assert result['rows'][0]['status'] == 'failed'
    assert 'timing' not in result['rows'][0]
    assert not result['performance']['eligible']


def test_benchmark_fingerprint_covers_production_and_validator():
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    source = bench.fingerprint()
    assert source['exact_import_paths']
    assert not source['missing_sources']
    for path in ('statgpu/nonparametric/kernel_smoothing/_native_bootstrap.py',
        'statgpu/nonparametric/kernel_smoothing/_native_bandwidth.py',
        'dev/tests/test_public_native_kde_bootstrap.py',
        'dev/benchmarks/benchmark_public_native_kde_bootstrap.py'):
        assert path in source['content_hashes']


@pytest.mark.parametrize('dtype', ('int64', 'float32'))
def test_public_dtype_promotion_and_input_immutability(backend, monkeypatch, dtype):
    x, q, w, indices = fixture_data()
    if dtype == 'int64':
        x = np.rint(5*x).astype(dtype)
        q = np.rint(5*q).astype(dtype)
        w = np.arange(1, len(x)+1, dtype=dtype)
    else:
        x, q, w = (value.astype(dtype) for value in (x, q, w))
    expected = oracle(x.astype(float), q.astype(float), w.astype(float), indices)
    with backend.context():
        arrays = [backend.array(value,dtype=getattr(backend.xp,dtype)) for value in (x,q,w)]
        before = [backend.host(value).copy() for value in arrays]
        install_fixed_indices(monkeypatch,backend,indices)
        result = kde_bootstrap_confidence_interval(*arrays[:2],weights=arrays[2],bandwidth=.55,
            backend=backend.name,n_resamples=3,confidence_level=.9,return_bootstrap_samples=True)
        assert_result(result,expected,3,len(q),backend)
        for value, original in zip(arrays,before):
            np.testing.assert_array_equal(backend.host(value),original)
            assert backend.host(value).dtype == original.dtype


@pytest.mark.parametrize('seed_kind', ('none', 'seed-sequence'))
def test_public_none_and_seed_sequence_rng_inputs(backend, seed_kind):
    x, q, w, _ = fixture_data()
    random_state = None if seed_kind == 'none' else np.random.SeedSequence(617)
    with backend.context():
        args = (backend.array(x),backend.array(q))
        kwargs = dict(weights=backend.array(w),bandwidth=.55,backend=backend.name,
            random_state=random_state,n_resamples=3,return_bootstrap_samples=True)
        result = kde_bootstrap_confidence_interval(*args,**kwargs)
        assert np.isfinite(result.bootstrap_samples).all()
        if seed_kind == 'seed-sequence':
            repeated = kde_bootstrap_confidence_interval(*args,**kwargs)
            np.testing.assert_array_equal(result.bootstrap_samples,repeated.bootstrap_samples)


def synthetic_complete_benchmark():
    """Synthetic metadata for gate tests only, never an execution result."""
    import copy
    from dataclasses import asdict
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    problem = asdict(bench.PRESETS['tiny'])
    source = dict(git_commit='synthetic',working_tree='clean',all_sources_tracked=True,
        missing_sources=[],exact_import_paths=True,worktree_fingerprint='synthetic')
    result = dict(problem=problem,source=source,source_after=copy.deepcopy(source),
        data_identity={'samples':'synthetic'},data_identity_after={'samples':'synthetic'},
        timing=dict(warmup=2,repeats=7,scope='synthetic'),environment={'environment_id':'synthetic'},rows=[])
    for name in bench.BACKENDS:
        device = 'cpu' if name == 'numpy' else 'cuda:1'
        row = dict(backend=name,status='pass',device=device,correctness={'status':'pass'},
            rng_validation=dict(status='pass',same_seed_indices_reproduced=True,seed=problem['seed'],
                index_shape=[problem['resamples'],problem['n']],indices_sha256='a'*64,
                cdf_max_abs_error=0.,dkw_epsilon=float(np.sqrt(np.log(2./1e-6)/(2.*problem['resamples']*problem['n']))),dkw_alpha=1e-6),
            timing=dict(instrumented=False,samples_seconds=[.1]*7,median_seconds=.1,synchronized_device=device))
        if name != 'numpy':
            identity = dict(backend=name,device=device,dtype='float64',kind='array',shape=[])
            stages = []
            for label,count,operations in [('native-bootstrap',1,()),('probability-cdf',1,('cumsum',)),
                ('native-weighted-rng',problem['resamples'],('searchsorted',)),('native-percentile',1,('quantile',)),
                ('native-fit',problem['resamples']+1,('sum','linalg.inv')),
                ('weighted-covariance',problem['resamples']+1,('sum',)),
                ('native-density',problem['resamples']+1,('exp',))]:
                stages += [dict(stage=label,completed=True,operations=[dict(operation=op,backend=name,
                    inputs=[identity.copy()],outputs=[identity.copy()]) for op in operations]) for _ in range(count)]
            row['operation_evidence'] = dict(status='observed',backend=name,device=device,stages=stages,
                host_conversion_guard={'status':'pass','hooks':['synthetic'],'source_audit':{'status':'pass'}},rng_generation=dict(status='observed',
                    backend=name,device=device,outputs=[dict(identity,shape=[problem['n']]) for _ in range(problem['resamples'])]))
        result['rows'].append(row)
    return result


def test_benchmark_complete_synthetic_fixture_and_policy_noncomparability():
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    result = bench.finalize(synthetic_complete_benchmark())
    assert result['validation']['status'] == 'complete'
    assert len(result['performance']['comparisons']) == 2
    different = synthetic_complete_benchmark()
    different['problem']['bandwidth'] = 'scott'
    bench.finalize(different)
    assert different['validation']['status'] == 'complete'
    assert different['performance']['selector_policy_comparable'] is False
    assert different['performance']['comparisons'] == []


def test_benchmark_scalar_control_protocol_survives_json_roundtrip():
    """Synthetic gate protocol, not physical GPU execution evidence."""
    import json
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from dev.benchmarks.learning_execution_trace import identities
    result = synthetic_complete_benchmark()
    trace = result['rows'][1]['operation_evidence']
    native = dict(backend='cupy', device='cuda:1', dtype='float64', kind='array', shape=[])
    # Source-derived CuPy slogdet where signatures, not a recorded remote event.
    for value in (0., -float('inf')):
        trace['stages'][0]['operations'].append(dict(operation='where', backend='cupy',
            inputs=[dict(native, dtype='bool'), identities(np.float64(value))[0], native.copy()],
            outputs=[native.copy()]))
    result = json.loads(json.dumps(result))
    assert bench.finalize(result)['validation']['status'] == 'complete'


@pytest.mark.parametrize('failure', ('scalar-only-input', 'host-scalar-output', 'host-0d-input',
    'host-0d-output', 'host-array-input', 'host-array-output', 'foreign-backend-input',
    'foreign-backend-output', 'wrong-device-input', 'wrong-device-output',
    'forged-native-scalar-input', 'forged-native-scalar-output', 'scalar-with-array-shape',
    'missing-input-kind', 'unknown-input-kind', 'missing-output-kind', 'unknown-output-kind'))
def test_benchmark_scalar_control_protocol_fails_closed(failure):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from dev.benchmarks.learning_execution_trace import identities
    result = synthetic_complete_benchmark()
    event = result['rows'][1]['operation_evidence']['stages'][1]['operations'][0]
    scalar = identities(np.float64(0.))[0]
    event['inputs'].append(scalar)
    if failure == 'scalar-only-input': event['inputs'] = [scalar]
    elif failure == 'host-scalar-output': event['outputs'].append(scalar)
    elif failure == 'host-0d-input': event['inputs'].append(identities(np.asarray(0.))[0])
    elif failure == 'host-0d-output': event['outputs'].append(identities(np.asarray(0.))[0])
    elif failure == 'host-array-input': event['inputs'].append(identities(np.zeros(2))[0])
    elif failure == 'host-array-output': event['outputs'].append(identities(np.zeros(2))[0])
    elif failure == 'foreign-backend-input': event['inputs'][0]['backend'] = 'torch'
    elif failure == 'foreign-backend-output': event['outputs'][0]['backend'] = 'torch'
    elif failure == 'wrong-device-input': event['inputs'][0]['device'] = 'cuda:0'
    elif failure == 'wrong-device-output': event['outputs'][0]['device'] = 'cuda:0'
    elif failure == 'forged-native-scalar-input': event['inputs'][0]['kind'] = 'numpy-scalar'
    elif failure == 'forged-native-scalar-output': event['outputs'][0]['kind'] = 'numpy-scalar'
    elif failure == 'scalar-with-array-shape': scalar['shape'] = []
    elif failure == 'missing-input-kind': event['inputs'][0].pop('kind')
    elif failure == 'unknown-input-kind': event['inputs'][0]['kind'] = 'unknown'
    elif failure == 'missing-output-kind': event['outputs'][0].pop('kind')
    elif failure == 'unknown-output-kind': event['outputs'][0]['kind'] = 'unknown'
    assert bench.finalize(result)['validation']['status'] == 'partial'
    assert result['performance']['comparisons'] == []


@pytest.mark.parametrize('value', ([1., 2.], (1., 2.), [[1., 2.]], [],
                                 np.ones(2), np.asarray(0.), np.float64(1.)))
def test_public_trace_host_guard_rejects_host_array_operands(value):
    """CPU observer-only call; no accelerator module or synthetic GPU array."""
    from types import SimpleNamespace
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    trace = bench.PublicTrace(SimpleNamespace(name='cupy'), [], bench.PRESETS['tiny'])
    trace.active[1] = {'operations': []}
    guarded = trace._wrap(np.sum, 'numpy', 'sum')
    with pytest.raises(bench.TraceIncomplete, match='host array numerical operation'):
        guarded(value)


def test_public_trace_host_guard_keeps_python_scalar_control_without_gpu_evidence():
    from types import SimpleNamespace
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    trace = bench.PublicTrace(SimpleNamespace(name='cupy'), [], bench.PRESETS['tiny'])
    trace.active[1] = {'operations': []}
    result = trace._wrap(np.sqrt, 'numpy', 'sqrt')(4.)
    assert result == 2.
    event = trace.active[1]['operations'][0]
    assert event['backend'] == 'numpy' and event['inputs'] == []
    assert event['outputs'][0]['kind'] == 'numpy-scalar'


@pytest.mark.parametrize('shape', ((), (2,)))
@pytest.mark.parametrize('backend', ('numpy', 'torch'))
def test_benchmark_guard_claim_cannot_hide_foreign_array_event(shape, backend):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from dev.benchmarks.learning_execution_trace import identities
    result = synthetic_complete_benchmark()
    identity = dict(identities(np.ones(shape))[0], backend=backend)
    result['rows'][1]['operation_evidence']['stages'][0]['operations'].append(
        dict(operation='sum', backend=backend, inputs=[identity], outputs=[identity.copy()]))
    assert bench.finalize(result)['validation']['status'] == 'partial'
    assert result['performance']['comparisons'] == []


def test_benchmark_host_scalar_control_is_permitted_but_never_native_evidence():
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from dev.benchmarks.learning_execution_trace import identities
    result = synthetic_complete_benchmark()
    event = dict(operation='sqrt', backend='numpy', inputs=[], outputs=identities(np.float64(2.)))
    trace = result['rows'][1]['operation_evidence']
    trace['stages'][0]['operations'].append(event)
    assert bench.finalize(result)['validation']['status'] == 'complete'
    trace['stages'][1]['operations'] = [dict(event, operation='cumsum')]
    assert bench.finalize(result)['validation']['status'] == 'partial'


@pytest.mark.parametrize('failure', ('missing-backend','cpu-device','trace','stage','stage-op','operand-device',
    'rng','rng-hash','rng-seed','rng-law','rng-shape','rng-output','rng-output-shape','rng-output-device',
    'host-guard','correctness','timing','sync','source','dirty','data','protocol'))
def test_benchmark_gate_fails_closed(failure):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    result = synthetic_complete_benchmark()
    row = result['rows'][2]
    trace = row['operation_evidence']
    if failure == 'missing-backend': result['rows'].pop(1)
    elif failure == 'cpu-device': row['device'] = 'cpu'
    elif failure == 'trace': trace['status'] = 'missing'
    elif failure == 'stage': trace['stages'] = []
    elif failure == 'stage-op': trace['stages'][1]['operations'] = []
    elif failure == 'operand-device': trace['stages'][1]['operations'][0]['inputs'][0]['device'] = 'cpu'
    elif failure == 'rng': row['rng_validation']['same_seed_indices_reproduced'] = False
    elif failure == 'rng-hash': row['rng_validation']['indices_sha256'] = 'invalid'
    elif failure == 'rng-seed': row['rng_validation']['seed'] = -1
    elif failure == 'rng-law': row['rng_validation']['cdf_max_abs_error'] = float('nan')
    elif failure == 'rng-shape': row['rng_validation']['index_shape'] = [1,1]
    elif failure == 'rng-output': trace['rng_generation']['outputs'] = []
    elif failure == 'rng-output-shape': trace['rng_generation']['outputs'][0]['shape'] = [1]
    elif failure == 'rng-output-device': trace['rng_generation']['outputs'][0]['device'] = 'cpu'
    elif failure == 'host-guard': trace['host_conversion_guard']['status'] = 'missing'
    elif failure == 'correctness': row['correctness']['status'] = 'fail'
    elif failure == 'timing': row['timing']['instrumented'] = True
    elif failure == 'sync': row['timing']['synchronized_device'] = 'cuda:0'
    elif failure == 'source': result['source_after']['git_commit'] = 'changed'
    elif failure == 'dirty': result['source']['working_tree'] = 'dirty'
    elif failure == 'data': result['data_identity_after'] = {}
    elif failure == 'protocol': result['timing']['repeats'] = 1
    bench.finalize(result)
    assert result['validation']['status'] == 'partial'
    assert not result['validation']['physical_gpu_claim']
    assert not result['performance']['eligible']
    assert result['performance']['comparisons'] == []


def test_benchmark_rejects_high_dimension_before_any_allocation(monkeypatch):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    def forbidden(*args,**kwargs):
        raise AssertionError('guard must precede data generation or runtime construction')
    monkeypatch.setattr(bench,'make_data',forbidden)
    monkeypatch.setattr(bench,'PublicRuntime',forbidden)
    problem = replace(bench.PRESETS['tiny'],n=2,queries=1,resamples=1,dimension=10000)
    with pytest.raises(ValueError,match='guard'):
        bench.run_benchmark(problem,cpu_only=True)


@pytest.mark.parametrize('method', ('cpu','numpy','tolist','to','__array__'))
def test_benchmark_rejects_additional_torch_host_export_routes(monkeypatch,method):
    pytest.importorskip('torch')
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    runtime = bench.PublicRuntime('torch',torch_cpu=True)
    problem = bench.PRESETS['tiny']
    original = native._draw_indices
    def exporting(*args,**kwargs):
        result = original(*args,**kwargs)
        exporter = getattr(result,method)
        exporter('cpu') if method == 'to' else exporter()
        return result
    monkeypatch.setattr(native,'_draw_indices',exporting)
    with pytest.raises(bench.TraceIncomplete,match='exported to host'):
        bench.trace_public(runtime,bench.make_data(problem),problem,3,None)


def test_explicit_uniform_weights_match_absent_weights(backend):
    x,q,_,_ = fixture_data(2,weights='none')
    with backend.context():
        args = (backend.array(x),backend.array(q))
        kwargs = dict(bandwidth='silverman',kernel='triweight',backend=backend.name,
            random_state=95,n_resamples=3,return_bootstrap_samples=True)
        absent = kde_bootstrap_confidence_interval(*args,**kwargs)
        explicit = kde_bootstrap_confidence_interval(*args,weights=backend.array(np.ones(len(x))),**kwargs)
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(absent,field),getattr(explicit,field))


def test_physical_nondefault_stream_keeps_rng_and_refits_on_input_device(backend):
    if backend.device == 'cpu':
        pytest.skip('physical accelerator stream test; CPU is not emulation')
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    runtime = bench.PublicRuntime(backend.name,ordinal=int(backend.device.split(':')[1]))
    problem = replace(bench.PRESETS['tiny'],dimension=2,weights='sparse')
    data = bench.make_data(problem)
    with backend.context():
        if backend.name == 'torch':
            stream = backend.xp.cuda.Stream(device=backend.device)
            context = backend.xp.cuda.stream(stream)
        else:
            stream = backend.xp.cuda.Stream(non_blocking=True)
            context = stream
        with context:
            value,indices = bench.capture_realized(bench.make_call(runtime,data,problem,3,None),runtime)
        stream.synchronize()
    expected,_ = bench.reference_for_indices(data,problem,indices,backend.name)
    bench.check(value,expected,problem)


def test_benchmark_source_audit_rejects_unhookable_direct_array_export(monkeypatch):
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    original = native._draw_indices
    def direct_get(*args,**kwargs):
        result = original(*args,**kwargs)
        result.get()
        return result
    monkeypatch.setattr(native,'_draw_indices',direct_get)
    with pytest.raises(bench.TraceIncomplete,match='exported to host'):
        bench.audit_native_export_sources()


def test_benchmark_source_audit_accepts_defined_contextvar_diagnostics():
    from dev.benchmarks.benchmark_public_native_kde_bootstrap import audit_native_export_sources
    report = audit_native_export_sources()
    assert report['status'] == 'pass'
    assert any(name.endswith('.projection_warning_diagnostics') for name in report['functions'])


def test_benchmark_source_audit_rejects_locally_shadowed_contextvar(monkeypatch):
    from contextvars import ContextVar
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    monkeypatch.setitem(globals(), '_audit_context', ContextVar('audit_context'))
    original = native._draw_indices
    def shadowed_get(*args,**kwargs):
        _audit_context = original(*args,**kwargs)
        _audit_context.get()
        return _audit_context
    monkeypatch.setattr(native,'_draw_indices',shadowed_get)
    with pytest.raises(bench.TraceIncomplete,match='exported to host'):
        bench.audit_native_export_sources()


def test_benchmark_source_audit_rejects_reassigned_global_contextvar(monkeypatch):
    from contextvars import ContextVar
    from dev.benchmarks import benchmark_public_native_kde_bootstrap as bench
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    monkeypatch.setitem(globals(), '_audit_context', ContextVar('audit_context'))
    original = native._draw_indices
    def reassigned_get(*args,**kwargs):
        global _audit_context
        _audit_context = original(*args,**kwargs)
        _audit_context.get()
        return _audit_context
    monkeypatch.setattr(native,'_draw_indices',reassigned_get)
    with pytest.raises(bench.TraceIncomplete,match='exported to host'):
        bench.audit_native_export_sources()
