"""CPU algebra/planner tests plus explicit physical-CuPy tests (no emulation claim)."""
import math
import numpy as np
import pytest

from dev.benchmarks.native_kde_bootstrap import _compute, _density, native_kde_bootstrap, plan_workspace
from statgpu.nonparametric import fit_kde


def fixture_data(kind='uniform'):
    x = np.array([-1.3, -.4, .2, .8, 1.5, 2.1])
    q = np.linspace(-2., 3., 7)
    w = {'uniform': np.ones(6), 'unequal': np.arange(1., 7.),
         'sparse': np.array([0., 2., 0., 3., 4., 1.])}[kind]
    w /= w.sum()
    indices = np.array([[1, 3, 3, 4, 5, 1], [5, 5, 4, 3, 1, 4],
                        [1, 1, 1, 1, 1, 1]], dtype=np.int64)
    return x, q, w, indices


def reference(x, q, w, indices):
    estimate = fit_kde(x, weights=w, bandwidth=.4, backend='numpy').pdf(q)
    draws = []
    for row in indices:
        sw = w[row].copy()
        sw /= sw.sum()
        draws.append(fit_kde(x[row], weights=sw, bandwidth=.4, backend='numpy').pdf(q))
    draws = np.asarray(draws)
    bounds = np.quantile(draws, [.025, .975], axis=0)
    return dict(estimate=estimate, draws=draws, lower=bounds[0], upper=bounds[1])


@pytest.mark.parametrize('kind', ['uniform', 'unequal', 'sparse'])
@pytest.mark.parametrize('chunk', [1, 3, 7])
def test_cpu_algebra_matches_existing_refits(kind, chunk):
    x, q, w, indices = fixture_data(kind)
    before = [a.copy() for a in (x, q, w, indices)]
    actual = _compute(x, q, indices, w, .4, .95, chunk, np)
    expected = reference(x, q, w, indices)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=1e-13)
    for a, original in zip((x, q, w, indices), before):
        np.testing.assert_array_equal(a, original)


def test_weighted_refits_are_not_silently_unweighted():
    x, q, w, indices = fixture_data('unequal')
    correct = _compute(x, q, indices, w, .4, .95, 2, np)
    unweighted = _compute(x, q, indices, np.ones(6)/6, .4, .95, 2, np)
    assert not np.allclose(correct['draws'][0], unweighted['draws'][0])


def test_constant_analytic_density_and_single_draw_query():
    x = np.ones(4)
    q = np.array([1.])
    w = np.ones(4)/4
    actual = _compute(x, q, np.zeros((1, 4), dtype=np.int64), w, .4, .95, 1, np)
    expected = 1 / (math.sqrt(1e-12 * .4**2) * math.sqrt(2*math.pi))
    for key in actual:
        np.testing.assert_allclose(actual[key], expected, rtol=2e-15)


@pytest.mark.parametrize('n,q,b', [(2, 1, 1), (20, 13, 5), (1000, 512, 100)])
def test_workspace_minimum_and_chunk_limit(n, q, b):
    minimum = plan_workspace(n, q, b, 10**12, query_chunk=1)['estimated_bytes']
    with pytest.raises(MemoryError):
        plan_workspace(n, q, b, minimum-1)
    assert plan_workspace(n, q, b, minimum)['query_chunk'] == 1
    if q > 1:
        with pytest.raises(MemoryError):
            plan_workspace(n, q, b, minimum, query_chunk=2)


@pytest.mark.parametrize('bad', [0, -1, 2.5, True, '5'])
def test_invalid_budget(bad):
    with pytest.raises(ValueError):
        plan_workspace(10, 3, 2, bad)


def test_retain_draws_is_budgeted():
    assert plan_workspace(10, 3, 2, 100000)['retained_draw_bytes'] == 2*3*8
    with pytest.raises(MemoryError):
        plan_workspace(10, 10000, 10000, 1024**2)


@pytest.mark.parametrize('kind', ['zero_resample', 'degenerate', 'overflow'])
def test_cpu_algebra_failure_contracts(kind):
    x, q, w, indices = fixture_data('sparse')
    if kind == 'zero_resample':
        indices[0] = 0
    elif kind == 'degenerate':
        w[:] = 0; w[1] = 1
    else:
        x *= 1e200
    with np.errstate(all='ignore'), pytest.raises(ValueError):
        _compute(x, q, indices, w, .4, .95, 2, np)


@pytest.fixture
def cp():
    cp = pytest.importorskip('cupy', reason='physical CuPy/CUDA required')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('physical CUDA device required')
        value = cp.sum(cp.arange(4, dtype=cp.float64))
        cp.cuda.Device().synchronize()
        assert float(value) == 6
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA unavailable: {exc}')
    return cp


@pytest.mark.parametrize('kind', ['uniform', 'unequal', 'sparse'])
@pytest.mark.parametrize('chunk', [1, 3, 7])
def test_physical_cupy_parity_and_host_conversion_guard(cp, monkeypatch, kind, chunk):
    x, q, w, indices = fixture_data(kind)
    device = cp.cuda.Device().id
    dx, dq, dw, di = map(cp.asarray, (x, q, w, indices))
    expected = reference(x, q, w, indices)
    original = cp.asnumpy
    def forbidden(*args, **kwargs):
        raise AssertionError('candidate attempted host-array conversion')
    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', forbidden)
        result = native_kde_bootstrap(dx, dq, di, weights=dw, query_chunk=chunk)
    cp.cuda.Device(device).synchronize()
    for key, value in result.items():
        assert isinstance(value, cp.ndarray) and value.device.id == device
        assert value.dtype == cp.float64
        np.testing.assert_allclose(original(value), expected[key], rtol=2e-11, atol=1e-12)
    for value, before in zip((dx, dq, dw, di), (x, q, w, indices)):
        np.testing.assert_array_equal(original(value), before)


@pytest.mark.parametrize('failure', ['host', 'dtype', 'shape', 'indices_float', 'indices_range',
    'zero_weights', 'negative_weights', 'nonfinite', 'budget', 'bandwidth_overflow',
    'bandwidth_underflow', 'confidence', 'zero_sampled_weights'])
def test_physical_cupy_invalid_contracts(cp, failure):
    x, q, w, indices = fixture_data('sparse')
    dx, dq, dw, di = map(cp.asarray, (x, q, w, indices))
    kwargs = dict(weights=dw)
    if failure == 'host': dx = x
    elif failure == 'dtype': dx = dx.astype(cp.float32)
    elif failure == 'shape': dx = dx.reshape(3, 2)
    elif failure == 'indices_float': di = di.astype(cp.float64)
    elif failure == 'indices_range': di[0, 0] = x.size
    elif failure == 'zero_weights': dw[:] = 0
    elif failure == 'negative_weights': dw[0] = -1
    elif failure == 'nonfinite': dx[0] = cp.nan
    elif failure == 'budget': kwargs['workspace_bytes'] = 1
    elif failure == 'bandwidth_overflow': kwargs['bandwidth'] = 1e200
    elif failure == 'bandwidth_underflow': kwargs['bandwidth'] = 1e-200
    elif failure == 'confidence': kwargs['confidence_level'] = 1
    elif failure == 'zero_sampled_weights': di[0] = 0
    with pytest.raises((ValueError, TypeError, MemoryError)):
        native_kde_bootstrap(dx, dq, di, **kwargs)


def test_physical_cupy_budget_rejected_before_numerical_allocation(cp, monkeypatch):
    x, q, w, indices = fixture_data()
    dx, dq, di = map(cp.asarray, (x, q, indices))
    def forbidden(*args, **kwargs):
        raise AssertionError('allocation or numerical validation before budget rejection')
    monkeypatch.setattr(cp, 'isfinite', forbidden)
    monkeypatch.setattr(cp, 'empty', forbidden)
    with pytest.raises(MemoryError):
        native_kde_bootstrap(dx, dq, di, workspace_bytes=1)


def test_physical_cupy_mixed_device_rejected(cp):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip('two physical CUDA devices required for mixed-device rejection')
    with cp.cuda.Device(0):
        x = cp.arange(6, dtype=cp.float64)
        indices = cp.arange(6, dtype=cp.int64)[None, :]
    with cp.cuda.Device(1):
        q = cp.arange(3, dtype=cp.float64)
    with pytest.raises(ValueError, match='same CUDA device'):
        native_kde_bootstrap(x, q, indices)


def test_cpu_distance_overflow_fails_even_when_final_density_would_be_finite():
    # Direct squaring previously silently erased a material tail contribution.
    x = np.array([-5e153, 5e153])
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='squared distance'):
        _density(x, np.array([.5, .5]), np.array([1.5e154]), 1., 1, np)


def test_physical_cupy_distance_overflow_fails(cp):
    x = cp.asarray([-5e153, 5e153], dtype=cp.float64)
    q = cp.asarray([1.5e154], dtype=cp.float64)
    indices = cp.asarray([[0, 1], [1, 0]], dtype=cp.int64)
    with pytest.raises(ValueError, match='squared distance'):
        native_kde_bootstrap(x, q, indices, bandwidth=1.)


def test_physical_cupy_unweighted_constant_views_single_draw_query(cp):
    # Accepted column-shaped, strided inputs; no dtype/device conversion needed.
    x = cp.ones((8, 2), dtype=cp.float64)[::2, :1]
    q = cp.ones((2, 2), dtype=cp.float64)[::2, :1]
    indices = cp.zeros((1, 4), dtype=cp.int64)
    minimum = plan_workspace(4, 1, 1, 10**6, 1)['estimated_bytes']
    result = native_kde_bootstrap(x, q, indices, workspace_bytes=minimum)
    expected = 1 / (math.sqrt(1e-12 * .4**2) * math.sqrt(2*math.pi))
    for value in result.values():
        np.testing.assert_allclose(cp.asnumpy(value), expected, rtol=2e-14)


def test_physical_cupy_automatic_budget_chunk_invariance(cp):
    x, q, w, indices = fixture_data('unequal')
    dx, dq, dw, di = map(cp.asarray, (x, q, w, indices))
    minimum = plan_workspace(x.size, q.size, len(indices), 10**6, 1)['estimated_bytes']
    narrow = native_kde_bootstrap(dx, dq, di, weights=dw, workspace_bytes=minimum)
    wide = native_kde_bootstrap(dx, dq, di, weights=dw, workspace_bytes=10**6)
    for key in narrow:
        cp.testing.assert_allclose(narrow[key], wide[key], rtol=2e-13, atol=1e-13)


@pytest.mark.parametrize('location', [1e9, 1e12])
@pytest.mark.parametrize('spread', [0., 1e-6])
def test_cpu_location_scale_cancellation_rejected(location, spread):
    x = np.full(6, location) + np.arange(6)*spread
    with pytest.raises(ValueError, match='location-to-scale'):
        _compute(x, np.array([location]), np.arange(6)[None, :], np.ones(6)/6,
                 .4, .95, 1, np)


@pytest.mark.parametrize('location', [1e9, 1e12])
def test_physical_cupy_location_scale_cancellation_rejected(cp, location):
    x = cp.full(6, location, dtype=cp.float64)
    q = cp.asarray([location], dtype=cp.float64)
    indices = cp.arange(6, dtype=cp.int64)[None, :]
    with pytest.raises(ValueError, match='location-to-scale'):
        native_kde_bootstrap(x, q, indices)


@pytest.mark.parametrize('draw_count', [1, 3])
def test_single_query_quantiles_use_vector_reduction(draw_count):
    # CPU route regression, not a substitute for physical CuPy execution.
    x, q, w, indices = fixture_data('unequal')
    q = q[3:4]
    indices = indices[:draw_count]
    calls = []

    class TrackedNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def quantile(self, values, levels, axis=None):
            calls.append((values.shape, axis))
            return np.quantile(values, levels, axis=axis)

    actual = _compute(x, q, indices, w, .4, .95, 1, TrackedNumpy())
    expected = reference(x, q, w, indices)
    assert calls == [((draw_count,), None)]
    for key, value in actual.items():
        assert value.shape == expected[key].shape
        np.testing.assert_allclose(value, expected[key], rtol=2e-12, atol=1e-13)


@pytest.mark.parametrize('layout', ['vector', 'strided_column'])
@pytest.mark.parametrize('draw_count', [1, 3])
@pytest.mark.parametrize('kind', ['constant', 'unequal'])
def test_physical_cupy_single_query_quantile_regression(cp, layout, draw_count, kind):
    # B=1 analytic and B>1 interpolated bounds; retain every returned array
    # through unrelated allocator activity before checking its value.
    x, q, w, indices = fixture_data('unequal')
    q = q[3:4]
    indices = indices[:draw_count]
    if kind == 'constant':
        x = np.ones_like(x)
        q = np.ones_like(q)
        w = np.ones_like(w) / w.size
    expected = reference(x, q, w, indices)
    dx, dq, dw, di = map(cp.asarray, (x, q, w, indices))
    if layout == 'strided_column':
        storage_x = cp.empty((x.size * 2, 2), dtype=cp.float64)
        storage_q = cp.empty((2, 2), dtype=cp.float64)
        dx, dq = storage_x[::2, :1], storage_q[::2, :1]
        dx[:, 0], dq[:, 0] = cp.asarray(x), cp.asarray(q)
    minimum = plan_workspace(x.size, 1, draw_count, 10**6, 1)['estimated_bytes']
    saved = []
    for poison in (0., .25, 1e200):
        buffers = [cp.full(64, poison, dtype=cp.float64) for _ in range(8)]
        cp.cuda.Device().synchronize()
        del buffers
        saved.append(native_kde_bootstrap(dx, dq, di, weights=dw,
                                         workspace_bytes=minimum))
    cp.cuda.Device().synchronize()
    for result in saved:
        for key, value in result.items():
            assert value.shape == expected[key].shape
            assert value.device.id == dx.device.id
            np.testing.assert_allclose(cp.asnumpy(value), expected[key],
                                       rtol=2e-11, atol=1e-12)
        assert bool(cp.all(result['lower'] <= result['upper']))
