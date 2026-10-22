"""Native RNG contracts: CPU planner/algebra checks are not GPU evidence."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from dev.benchmarks.native_kde_rng import (
    _probability_cdf, _sample_indices, native_kde_bootstrap_rng,
    plan_rng_workspace, validate_seed,
)
from dev.benchmarks.benchmark_native_kde_bootstrap import Problem, analytic_reference, check


@pytest.mark.parametrize('seed', [0, 1, 2**32-1, np.int64(42)])
def test_explicit_seed(seed):
    assert validate_seed(seed) == int(seed)


@pytest.mark.parametrize('seed', [None, True, False, -1, 2**32, 1.5, '1', np.nan])
def test_bad_seed_before_device_import(seed):
    with pytest.raises(ValueError, match='seed'):
        native_kde_bootstrap_rng(None, None, seed=seed)


@pytest.mark.parametrize('n,q,b', [(2, 1, 1), (32, 7, 5), (4096, 512, 100)])
def test_budget_includes_endogenous_indices_and_uniforms(n, q, b):
    plan = plan_rng_workspace(n, q, b, 10**12, 1)
    assert plan['index_bytes'] == 8*n*b
    assert plan['rng_buffer_reserve_bytes'] >= 24*n*b
    assert plan['estimated_bytes'] > plan['rng_buffer_reserve_bytes']
    minimum = plan['estimated_bytes']
    assert plan_rng_workspace(n, q, b, minimum)['query_chunk'] == 1
    with pytest.raises(MemoryError):
        plan_rng_workspace(n, q, b, minimum-1)


@pytest.mark.parametrize('name', ['n', 'q', 'b', 'budget', 'query_chunk'])
@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_invalid_planner(name, bad):
    args = dict(n=4, q=3, b=2, budget=10**6, query_chunk=1)
    args[name] = bad
    with pytest.raises(ValueError):
        plan_rng_workspace(**args)


@pytest.mark.parametrize('weights', [None, np.ones(5), np.array([0., .25, 0., .75, 0.])])
def test_cdf_normalized_preserves_support(weights):
    cdf = _probability_cdf(weights, 5, np)
    assert cdf[-1] == 1 and np.all(np.diff(cdf) >= 0)
    if weights is not None:
        np.testing.assert_array_equal(np.diff(np.r_[0., cdf]) == 0, weights == 0)


@pytest.mark.parametrize('weights', [np.zeros(3), np.array([-1., 1., 1.]),
    np.array([np.nan, 1., 1.]), np.array([np.inf, 1., 1.]),
    np.array([1., 1e-300, 1.]), np.array([1., 1., 1e-300]),
    np.array([1e-300, 1e300, 1e300])])
def test_invalid_or_unrepresentable_probabilities_fail(weights):
    with np.errstate(all='ignore'), pytest.raises(ValueError):
        _probability_cdf(weights, 3, np)


def test_exact_inverse_cdf_boundaries_skip_all_zero_bins():
    cdf = _probability_cdf(np.array([0., .25, 0., .75, 0.]), 5, np)
    values = np.array([[0., np.nextafter(.25, 0), .25, .5, np.nextafter(1., 0)]])
    class RNG:
        def random_sample(self, shape, dtype):
            assert shape == values.shape and dtype is np.float64
            return values.copy()
    xp = SimpleNamespace(random=SimpleNamespace(RandomState=lambda seed: RNG()),
        float64=np.float64, int64=np.int64, searchsorted=np.searchsorted, min=np.min, max=np.max)
    np.testing.assert_array_equal(_sample_indices(cdf, 1, 42, xp), [[1, 1, 3, 3, 3]])
    values[0, -1] = 1.
    with pytest.raises(ValueError, match='support'):
        _sample_indices(cdf, 1, 42, xp)


@pytest.fixture
def cp():
    cp = pytest.importorskip('cupy', reason='physical CuPy/CUDA required')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('physical CUDA device required')
        assert float(cp.sum(cp.arange(4, dtype=cp.float64))) == 6
        cp.cuda.Device().synchronize()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA unavailable: {exc}')
    return cp


def fixture(cp, kind='none', singleton=False):
    x = cp.asarray([-1.3, -.4, .2, .8, 1.5, 2.1], dtype=cp.float64)
    q = cp.asarray([.1] if singleton else [-1., .1, .7, 2.], dtype=cp.float64)
    w = None if kind == 'none' else cp.asarray(
        [0., 2., 0., 3., 4., 1.] if kind == 'sparse' else [1., 2., 3., 4., 5., 6.], dtype=cp.float64)
    return x, q, w


@pytest.mark.parametrize('kind', ['none', 'unequal', 'sparse'])
@pytest.mark.parametrize('b,q1', [(1, True), (7, True), (7, False)])
def test_physical_rng_realized_oracle_and_repeat(cp, kind, b, q1):
    x, q, w = fixture(cp, kind, q1)
    before = [None if a is None else a.copy() for a in (x, q, w)]
    result, indices = native_kde_bootstrap_rng(x, q, weights=w, seed=21, n_resamples=b, return_indices=True)
    again, repeated = native_kde_bootstrap_rng(x, q, weights=w, seed=21, n_resamples=b, return_indices=True, query_chunk=1)
    cp.testing.assert_array_equal(indices, repeated)
    p = Problem(n=x.size, queries=q.size, resamples=b)
    data = dict(samples=cp.asnumpy(x), points=cp.asnumpy(q), weights=None if w is None else cp.asnumpy(w), indices=cp.asnumpy(indices))
    check({k: cp.asnumpy(v) for k, v in result.items()}, analytic_reference(data, p), p)
    for key in result:
        assert result[key].device.id == x.device.id and result[key].dtype == cp.float64
        cp.testing.assert_allclose(again[key], result[key], rtol=2e-12, atol=1e-13)
    for a, old in zip((x, q, w), before):
        if a is not None:
            cp.testing.assert_array_equal(a, old)
    if w is not None:
        assert bool(cp.all(w[indices] > 0))


def test_physical_rng_global_stream_unchanged_and_different_seed(cp):
    x, q, w = fixture(cp)
    # Restore the preexisting global generator object after this test.
    old = cp.random.get_random_state()
    try:
        cp.random.seed(109)
        expected = cp.random.random_sample(12)
        cp.random.seed(109)
        _, a = native_kde_bootstrap_rng(x, q, n_resamples=10, seed=1, return_indices=True)
        actual = cp.random.random_sample(12)
        _, b = native_kde_bootstrap_rng(x, q, n_resamples=10, seed=2, return_indices=True)
        cp.testing.assert_array_equal(expected, actual)
        assert not bool(cp.all(a == b))
    finally:
        cp.random.set_random_state(old)


def test_physical_rng_weighted_frequencies_and_zero_support(cp):
    cdf = _probability_cdf(cp.asarray([0., 1., 0., 3., 0.]), 5, cp)
    indices = _sample_indices(cdf, 20000, 123, cp)
    counts = cp.asnumpy(cp.bincount(indices.reshape(-1), minlength=5))
    assert counts[[0, 2, 4]].sum() == 0
    # Deterministic seed, generous >6-sigma band; checks law, not bitstream parity.
    assert abs(counts[1] / indices.size - .25) < .01
    assert abs(counts[3] / indices.size - .75) < .01


def test_physical_rng_views_constant_retention_nondefault_stream(cp):
    x = cp.ones((8, 2), dtype=cp.float64)[::2, :1]
    q = cp.ones((2, 2), dtype=cp.float64)[::2, :1]
    minimum = plan_rng_workspace(4, 1, 1, 10**6, 1)['estimated_bytes']
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.Device(x.device.id).synchronize()  # publish default-stream inputs
    with stream:
        first, idx = native_kde_bootstrap_rng(x, q, n_resamples=1, seed=17, workspace_bytes=minimum, return_indices=True)
        for seed in range(5):
            native_kde_bootstrap_rng(x, q, n_resamples=1, seed=seed, workspace_bytes=minimum)
    stream.synchronize()
    expected = 1 / (math.sqrt(1e-12 * .4**2) * math.sqrt(2*math.pi))
    for value in first.values():
        np.testing.assert_allclose(cp.asnumpy(value), expected, rtol=2e-14)
    assert idx.shape == (1, 4)


def test_physical_rng_budget_rejected_before_rng(cp):
    x, q, w = fixture(cp)
    with patch.object(cp.random, 'RandomState', side_effect=AssertionError('RNG was allocated')):
        with pytest.raises(MemoryError):
            native_kde_bootstrap_rng(x, q, workspace_bytes=1)


@pytest.mark.parametrize('kind', ['host', 'float32', 'matrix', 'nan', 'bad_weights', 'cdf_collapse'])
def test_physical_rng_invalid_input(cp, kind):
    x, q, w = fixture(cp, 'unequal')
    error = (TypeError, ValueError)
    if kind == 'host': x = np.arange(6, dtype=np.float64)
    elif kind == 'float32': x = x.astype(cp.float32)
    elif kind == 'matrix': x = x.reshape(2, 3)
    elif kind == 'nan': x[0] = cp.nan
    elif kind == 'bad_weights': w[0] = -1
    elif kind == 'cdf_collapse': w[-1] = 1e-300
    with pytest.raises(error):
        native_kde_bootstrap_rng(x, q, weights=w)


def test_physical_rng_mixed_device_rejected(cp):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip('two physical CUDA devices required for mixed-device rejection')
    with cp.cuda.Device(0): x = cp.arange(6, dtype=cp.float64)
    with cp.cuda.Device(1): q = cp.arange(3, dtype=cp.float64)
    with pytest.raises(ValueError, match='same CUDA device'):
        native_kde_bootstrap_rng(x, q)


def test_physical_rng_no_numeric_host_conversion(cp):
    x, q, w = fixture(cp, 'unequal')
    with patch.object(cp, 'asnumpy', side_effect=AssertionError('host conversion')):
        result = native_kde_bootstrap_rng(x, q, weights=w, n_resamples=3, seed=4)
    assert all(isinstance(value, cp.ndarray) for value in result.values())


def test_physical_rng_exact_cdf_boundaries(cp):
    cdf = _probability_cdf(cp.asarray([0., .25, 0., .75, 0.]), 5, cp)
    u = cp.asarray([[0., np.nextafter(.25, 0), .25, .5, np.nextafter(1., 0)]])
    class Fixed:
        def random_sample(self, shape, dtype):
            assert shape == u.shape
            return u.copy()
    with patch.object(cp.random, 'RandomState', return_value=Fixed()):
        actual = _sample_indices(cdf, 1, 0, cp)
    cp.testing.assert_array_equal(actual, cp.asarray([[1, 1, 3, 3, 3]]))
