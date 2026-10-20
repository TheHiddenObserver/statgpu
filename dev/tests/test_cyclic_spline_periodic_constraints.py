"""Independent analytic regressions for issue #216 (no finite differences)."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from statgpu.nonparametric import cyclic_cubic_spline_basis


def _verify(x, knots, actual):
    a, b = x.min(), x.max()
    # Express the independent reference in dimensionless coordinates so
    # derivative comparisons have the same physical meaning across units.
    z, k = (x-a)/(b-a), (knots-a)/(b-a)
    ref = BSpline(np.r_[np.zeros(4), k, np.ones(4)], np.eye(len(k)+4), 3)
    base = ref(z)
    coefficients = np.linalg.lstsq(base, actual, rcond=None)[0]
    assert actual.shape == (len(x), len(k)+1)
    assert actual.dtype == np.float64
    assert np.isfinite(actual).all()
    assert np.linalg.matrix_rank(actual) == len(k)+1
    assert_allclose(base @ coefficients, actual, atol=2e-12, rtol=2e-12)
    constraints = np.vstack([ref.derivative(d)(0)-ref.derivative(d)(1) for d in (0, 1, 2)])
    assert_allclose(constraints @ coefficients, 0, atol=2e-9, rtol=0)
    expected = null_space(constraints / np.linalg.norm(constraints, axis=1)[:,None])
    q, _ = np.linalg.qr(coefficients)
    assert_allclose(q @ q.T, expected @ expected.T, atol=2e-12, rtol=2e-12)
    weights = np.linalg.lstsq(actual, np.ones(len(x)), rcond=None)[0]
    assert_allclose(actual @ weights, 1, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('knots', [[.5], [.1,.7], [.07,.2,.55,.91], np.linspace(.1,.9,10)])
@pytest.mark.parametrize('offset,scale', [(0.,1.), (0.,1e-12), (0.,1e12), (-123.,7.), (1e6,3.)])
def test_analytic_periodicity_and_complete_space(knots, offset, scale):
    x = offset + scale*np.linspace(0,1,501)
    k = offset + scale*np.asarray(knots)
    _verify(x, k, cyclic_cubic_spline_basis(x, k))


def test_unsorted_points_and_float32_inputs():
    x = np.linspace(-2,3,501,dtype=np.float32)[::-1].copy()
    k = np.array([-1.5,-.3,1.7,2.5],dtype=np.float32)
    _verify(x.astype(float), k.astype(float), cyclic_cubic_spline_basis(x,k))


@pytest.mark.parametrize('x,knots', [([], [.5]), ([0,1], []), ([0,0], [.5]),
    ([0,np.nan,1],[.5]), ([0,1],[np.inf]), ([0,1],[.7,.3]),
    ([0,1],[.5,.5]), ([0,1],[0]), ([0,1],[1]), ([0,1],[-.1])])
def test_invalid_input_rejected(x,knots):
    with pytest.raises(ValueError):
        cyclic_cubic_spline_basis(x,knots)


def test_unresolvable_knot_spacing_rejected():
    with pytest.raises(ValueError, match="three independent"):
        cyclic_cubic_spline_basis(np.linspace(0,1,20), [1e-20,.5])


def test_unresolvable_constraint_rank_rejected(monkeypatch):
    original = np.linalg.svd
    def deficient(a, **kw):
        u,s,v = original(a,**kw)
        s[-1] = 0
        return u,s,v
    monkeypatch.setattr(np.linalg,'svd',deficient)
    with pytest.raises(ValueError,match='three independent'):
        cyclic_cubic_spline_basis(np.linspace(0,1,20),[.2,.7])


def test_svd_failure_not_silently_retried_on_host(monkeypatch):
    def fail(*args,**kw):
        raise np.linalg.LinAlgError('SVD failed')
    monkeypatch.setattr(np.linalg,'svd',fail)
    with pytest.raises(np.linalg.LinAlgError,match='SVD failed'):
        cyclic_cubic_spline_basis(np.linspace(0,1,20),[.2,.7])


@pytest.mark.parametrize('backend', ['cupy','torch_cpu','torch_cuda'])
def test_backend_periodic_space_dtype_and_ownership(backend, monkeypatch):
    x=np.linspace(-2,3,501); k=np.array([-1.5,-.3,1.7,2.5])
    if backend=='cupy':
        xp=pytest.importorskip('cupy')
        try:
            count=xp.cuda.runtime.getDeviceCount()
        except xp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA runtime unavailable')
        if not count: pytest.skip('No physical CUDA device')
        dx,dk=xp.asarray(x),xp.asarray(k)
        seen=[]
        original=xp.linalg.svd
        def native_svd(a, **kwargs):
            seen.append(a.device)
            return original(a, **kwargs)
        monkeypatch.setattr(xp.linalg, 'svd', native_svd)
        result=cyclic_cubic_spline_basis(dx,dk,xp=xp)
        assert len(seen)==1 and seen[0].id==dx.device.id
        assert result.device.id==dx.device.id
        assert result.dtype==xp.float64
        actual=xp.asnumpy(result)
    else:
        xp=pytest.importorskip('torch')
        device='cuda' if backend=='torch_cuda' else 'cpu'
        if device=='cuda' and not xp.cuda.is_available():
            pytest.skip('No physical CUDA device')
        dx=xp.asarray(x,device=device);dk=xp.asarray(k,device=device)
        seen=[]
        original=xp.linalg.svd
        def native_svd(a, **kwargs):
            seen.append(a.device)
            return original(a, **kwargs)
        monkeypatch.setattr(xp.linalg, 'svd', native_svd)
        result=cyclic_cubic_spline_basis(dx,dk,xp=xp)
        assert seen==[dx.device]
        assert result.device==dx.device and result.dtype==xp.float64
        actual=result.detach().cpu().numpy()
    _verify(x,k,actual)
