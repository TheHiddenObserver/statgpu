"""Scoped dfn=2 F survival regression tests, including physical CUDA cases.

Torch CPU is an explicitly selected test backend, not CUDA evidence. General
numerator df and vector shape parameters are not repaired by this change.
"""
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from statgpu.inference import _distributions_backend as distributions


@pytest.fixture(params=['numpy', 'torch-cpu', 'cupy-cuda', 'torch-cuda'])
def f_backend(request, monkeypatch):
    name = request.param
    if name == 'numpy':
        backend, device, xp = 'numpy', None, np
    elif name.startswith('torch'):
        xp = pytest.importorskip('torch')
        backend = 'torch'
        if name == 'torch-cuda':
            if not xp.cuda.is_available():
                pytest.skip('Physical Torch CUDA unavailable; no CPU substitution')
            device = f'cuda:{xp.cuda.current_device()}'
        else:
            device = 'cpu'
        # f_oneway historically creates its scalar distribution without device.
        # Keep that test on the explicitly selected Torch device, including CPU.
        monkeypatch.setattr(distributions, '_get_torch_device', lambda: device)
    else:
        xp = pytest.importorskip('cupy')
        try:
            if xp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip('Physical CuPy CUDA unavailable; no CPU substitution')
        except xp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f'Physical CuPy CUDA unavailable: {exc}')
        backend, device = 'cupy', None
    dist = distributions.get_distribution('f', backend=backend, device=device)

    def array(value):
        if backend == 'torch':
            return xp.as_tensor(value, dtype=xp.float64, device=device)
        return xp.asarray(value, dtype=xp.float64)

    def host(value):
        if backend == 'torch':
            assert value.device == xp.device(device)
            assert value.dtype == xp.float64
            return value.detach().cpu().numpy()
        if backend == 'cupy':
            assert isinstance(value, xp.ndarray)
            assert value.device.id == xp.cuda.runtime.getDevice()
            assert value.dtype == xp.float64
            return value.get()
        assert np.asarray(value).dtype == np.float64
        return np.asarray(value)

    return backend, dist, array, host


@pytest.mark.parametrize('dfd', [.25, 1., 2.5, 7., 251.89076961668133, 381., 10000.])
def test_exact_family_matches_independent_scipy(f_backend, dfd):
    _, dist, array, host = f_backend
    x = np.logspace(-12, 8, 91).reshape(7, 13)
    actual = host(dist.sf(array(x), 2, dfd))
    np.testing.assert_allclose(actual, stats.f.sf(x, 2, dfd), rtol=3e-12, atol=2e-15)
    assert np.all(np.diff(actual.ravel()) <= 0)


def test_old_lut_knots_and_neighbors(f_backend):
    _, dist, array, host = f_backend
    # Independent reconstruction of old beta grid; do not invoke the LUT.
    knots = np.array([.01, .01 + .98/8001, .01 + 68*.98/8001, .25, .99])
    z = np.concatenate([np.nextafter(knots, 0), knots,
                        np.nextafter(knots, 1), knots + .49*.98/8001])
    x = 381*z/(2*(1-z))
    actual = host(dist.sf(array(x), 2, 381))
    np.testing.assert_allclose(actual, stats.f.sf(x, 2, 381), rtol=3e-12, atol=1e-15)


@pytest.mark.parametrize('dfd', [.01, 2.5, 381.])
def test_support_nan_scalar_and_empty_shapes(f_backend, dfd):
    _, dist, array, host = f_backend
    x = np.array([-np.inf, -1., -0., 0., 1., np.inf, np.nan])
    np.testing.assert_allclose(host(dist.sf(array(x), 2, dfd)),
                               stats.f.sf(x, 2, dfd), rtol=2e-13, equal_nan=True)
    assert host(dist.sf(array(1.), 2, dfd)).shape == ()
    assert host(dist.sf(array(np.empty((2, 0))), 2, dfd)).shape == (2, 0)


def test_cancellation_and_overflow_extremes(f_backend):
    _, dist, array, host = f_backend
    cases = [(50., 381.), (100., 381.), (1e150, .01),
             (np.finfo(float).max, .01), (np.finfo(float).max, 1.),
             (np.finfo(float).max, 1e-300), (1., 1e300),
             (np.finfo(float).max, np.nextafter(0., 1.))]
    for x, dfd in cases:
        # Decimal provides an independent high-precision analytic oracle even
        # where the naive 2*x/dfd formula overflows or 1-CDF rounds to zero.
        with localcontext() as ctx:
            ctx.prec = 700
            dx, dd = Decimal.from_float(x), Decimal.from_float(dfd)
            expected = float((-(dd/2) * (1+2*dx/dd).ln()).exp())
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            actual = host(dist.sf(array(x), 2, dfd))
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0)
        assert float(actual) > 0


@pytest.mark.parametrize('use_lut', [False, True])
def test_exact_family_never_calls_beta_or_host_fallback(f_backend, monkeypatch, use_lut):
    backend, dist, array, host = f_backend
    dist._sf.use_lut = use_lut
    x = array([.5, 3.5466470234207264, 100.])
    expected = stats.f.sf([.5, 3.5466470234207264, 100.], 2, 381)

    def forbidden(*args, **kwargs):
        raise AssertionError('Exact F survival must stay native and bypass beta/LUT')

    with monkeypatch.context() as patch:
        patch.setattr(dist._sf, 'betainc', forbidden)
        patch.setattr(distributions, '_get_torch_betainc_lut', forbidden)
        patch.setattr(dist._sf, 'to_numpy', forbidden)
        patch.setattr(stats.f, 'sf', forbidden)
        if backend == 'torch':
            patch.setattr(dist._sf._torch.Tensor, 'cpu', forbidden)
            patch.setattr(dist._sf._torch.Tensor, 'numpy', forbidden)
        actual = dist.sf(x, 2, 381)
    np.testing.assert_allclose(host(actual), expected, rtol=3e-13, atol=0)


def test_vector_degrees_of_freedom_remain_unsupported(f_backend):
    _, dist, array, _ = f_backend
    for dfn, dfd in [(array([2., 3.]), 381.), (2., array([250., 381.]))]:
        with pytest.raises((TypeError, ValueError)):
            dist.sf(array([1., 2.]), dfn, dfd)


def test_other_numerator_df_retains_existing_path(f_backend, monkeypatch):
    _, dist, array, host = f_backend
    seen = []

    def cdf(x, dfn, dfd):
        seen.append((dfn, dfd))
        return array([.2, .8])

    monkeypatch.setattr(dist, 'cdf', cdf)
    np.testing.assert_allclose(host(dist.sf(array([1., 2.]), 3.5, 17.25)), [.8, .2])
    assert seen == [(3.5, 17.25)]


@pytest.mark.parametrize('operation', ['f_oneway', 'f_welch'])
@pytest.mark.parametrize('n', [128, 1024])
def test_fullscan_anova_fixture_independent_reference(f_backend, operation, n):
    from statgpu import anova

    backend, _, array, _ = f_backend
    rng = np.random.default_rng(20261020)
    groups = [rng.normal(loc=i*.2, scale=1+i*.15, size=n) for i in range(3)]
    actual = getattr(anova, operation)(*[array(g) for g in groups], backend=backend)
    if operation == 'f_oneway':
        ref = stats.f_oneway(*groups)
        statistic, pvalue, dfd = ref.statistic, ref.pvalue, 3*n-3
    else:
        # Independent Welch equations, not the production implementation or
        # its F helper; supports environments without optional statsmodels.
        means = np.array([np.mean(g) for g in groups])
        weights = np.array([len(g)/np.var(g, ddof=1) for g in groups])
        adjustment = np.sum((1-weights/weights.sum())**2/(n-1))
        statistic = np.sum(weights*(means-np.average(means, weights=weights))**2)
        statistic = statistic/2/(1+adjustment/4)
        dfd = (8/3)/adjustment
        pvalue = stats.f.sf(statistic, 2, dfd)
    # Preserve the failed benchmark's acceptance thresholds exactly.
    np.testing.assert_allclose(actual.statistic, statistic, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual.pvalue, pvalue, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual.df_within, dfd, rtol=1e-12)
    assert actual.df_between == 2


def test_two_way_anova_shared_survival_consumer(f_backend):
    from statgpu.anova import f_twoway

    backend, _, array, _ = f_backend
    rng = np.random.default_rng(1234)
    cells = rng.normal(size=(2, 3, 24)) + np.arange(3)[None, :, None]*.3
    actual = f_twoway([[array(cells[i, j]) for j in range(3)] for i in range(2)],
                      backend=backend)
    mean = cells.mean()
    ma, mb, mc = cells.mean(axis=(1, 2)), cells.mean(axis=(0, 2)), cells.mean(axis=2)
    mse = np.sum((cells-mc[:, :, None])**2)/(6*23)
    f_b = (48*np.sum((mb-mean)**2)/2)/mse
    f_ab = (24*np.sum((mc-ma[:, None]-mb[None, :]+mean)**2)/2)/mse
    np.testing.assert_allclose([actual.factor_b_statistic, actual.interaction_statistic],
                               [f_b, f_ab], rtol=2e-13)
    np.testing.assert_allclose([actual.factor_b_pvalue, actual.interaction_pvalue],
                               stats.f.sf([f_b, f_ab], 2, 6*23), rtol=2e-13)


def test_regression_and_panel_shared_survival_consumers():
    from types import SimpleNamespace
    from statgpu.linear_model._stats import RegressionResults
    from statgpu.panel._diagnostics import _classical_model_f

    rng = np.random.default_rng(75)
    design = np.column_stack([np.ones(80), rng.normal(size=(80, 2))])
    y = design @ [1., .25, -.4] + rng.normal(size=80)
    params = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y-design@params
    rss, tss = resid@resid, np.sum((y-y.mean())**2)
    expected_f = ((tss-rss)/2)/(rss/77)
    expected_p = stats.f.sf(expected_f, 2, 77)
    model = SimpleNamespace(_y=y, _X_design=design)
    result = RegressionResults(model, params, resid, rss/77, len(y), 77)
    np.testing.assert_allclose([result.fvalue, result.f_pvalue],
                               [expected_f, expected_p], rtol=2e-13)
    panel_f, panel_p, df, _ = _classical_model_f(
        y, design, params, xp=np, df_resid=77, has_constant=True)
    assert df == (2., 77.)
    np.testing.assert_allclose([panel_f, panel_p], [expected_f, expected_p], rtol=2e-13)
