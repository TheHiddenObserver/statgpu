"""Native sampler regression contracts for issue #173.

Sampling remains CPU RNG plus native-backend output conversion, not GPU RNG.
"""
import numpy as np
import pytest
from statgpu.inference._distributions_backend import get_distribution, f

SHAPES = {
    'norm': {}, 't': {'df': 7}, 'uniform': {}, 'expon': {},
    'cauchy': {}, 'laplace': {}, 'logistic': {}, 'chi2': {'df': 7},
    'gamma': {'a': 2}, 'beta': {'a': 2, 'b': 3},
    'f': {'dfn': 5, 'dfd': 10}, 'weibull_min': {'c': 2},
    'lognorm': {'s': .5}, 'poisson': {'mu': 3}, 'binom': {'n': 9, 'p': .4},
}

@pytest.fixture(params=['numpy', 'cupy', 'torch_cpu', 'torch_cuda'])
def backend(request):
    name = request.param
    if name == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            available = cp.cuda.runtime.getDeviceCount() > 0
        except cp.cuda.runtime.CUDARuntimeError:
            available = False
        if not available:
            pytest.skip('physical CuPy CUDA unavailable')
    if name.startswith('torch'):
        torch = pytest.importorskip('torch')
        if name == 'torch_cuda' and not torch.cuda.is_available():
            pytest.skip('physical Torch CUDA unavailable')
    return name

@pytest.mark.parametrize('name', SHAPES)
@pytest.mark.parametrize('dtype', [None, np.float32, np.dtype('float64')])
@pytest.mark.parametrize('size', [None, (2, 3), 0])
def test_native_sampling_dtype_shape_and_rng(name, dtype, size, backend):
    family = 'torch' if backend.startswith('torch') else backend
    device = ('cuda' if backend == 'torch_cuda' else 'cpu') if family == 'torch' else None
    dist = get_distribution(name, backend=family, device=device)
    np.random.seed(173)
    reference = get_distribution(name, backend='numpy').rvs(**SHAPES[name], size=size)
    np.random.seed(173)
    out = dist.rvs(**SHAPES[name], size=size, dtype=dtype)
    expected_dtype = np.dtype('float64' if dtype is None else dtype)
    if family == 'torch':
        import torch
        assert isinstance(out, torch.Tensor)
        assert out.device.type == device
        assert out.dtype == (torch.float32 if expected_dtype == np.float32 else torch.float64)
        host = out.cpu().numpy()
    elif family == 'cupy':
        import cupy as cp
        assert isinstance(out, cp.ndarray)
        assert out.device.id == cp.cuda.runtime.getDevice()
        host = cp.asnumpy(out)
    else:
        assert isinstance(out, np.ndarray)
        host = out
    assert host.dtype == expected_dtype
    assert host.shape == (() if size is None else (size,) if isinstance(size, int) else size)
    np.testing.assert_array_equal(host, reference.astype(expected_dtype))

@pytest.mark.parametrize('size', [None, 12, (3, 4)])
def test_f_sampling_matches_numpy_public_parameters(size):
    np.random.seed(173)
    expected = np.random.f(dfnum=5, dfden=10, size=size)
    np.random.seed(173)
    actual = f.rvs(dfn=5, dfd=10, size=size, backend='numpy')
    np.testing.assert_array_equal(actual, expected)

@pytest.mark.parametrize('name', ['poisson', 'binom'])
def test_explicit_integer_sampling_dtype(name):
    out = get_distribution(name, backend='numpy').rvs(**SHAPES[name], size=10, dtype=np.int64)
    assert out.dtype == np.int64


def test_torch_native_dtype():
    torch = pytest.importorskip('torch')
    out = get_distribution('f', backend='torch', device='cpu').rvs(5, 10, size=3, dtype=torch.float32)
    assert out.dtype == torch.float32 and out.device.type == 'cpu'
