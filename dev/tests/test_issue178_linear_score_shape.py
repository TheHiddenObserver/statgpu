"""Regression tests for single-response scoring shape normalization (#178)."""
import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from statgpu import LinearRegression
from statgpu.linear_model.wrappers import _linear


@pytest.fixture(params=['cpu', 'cuda', 'torch'])
def native(request):
    device = request.param
    try:
        if device == 'cuda':
            import cupy as cp
            cp.zeros(1)
            return device, cp.asarray
        if device == 'torch':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Torch CUDA unavailable')
            return device, lambda x: torch.as_tensor(x, dtype=torch.float64, device='cuda')
        return device, np.asarray
    except (ImportError, RuntimeError) as exc:
        if os.environ.get('STATGPU_REQUIRE_PHYSICAL_GPU') == '1':
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def _r2(y, pred):
    residual = np.sum((y - pred) ** 2, axis=0)
    total = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    return float(np.mean(np.divide(total - residual, total,
                                  out=np.zeros_like(total), where=total > 0)))


@pytest.mark.parametrize('fit_column', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('perfect', [False, True])
def test_single_response_forms_equal_independent_r2(native, monkeypatch, fit_column, weighted, perfect):
    device, array = native
    X = np.arange(10., dtype=float).reshape(-1, 1)
    y = 1 + 2 * X[:, 0]
    if not perfect:
        y = y + np.sin(X[:, 0])
    weight = np.linspace(.3, 2., len(y)) if weighted else None
    model = LinearRegression(device=device, compute_inference=False).fit(
        array(X), array(y[:, None] if fit_column else y),
        sample_weight=None if weight is None else array(weight))
    query = X if perfect else X + .4
    pred = query @ model.coef_ + model.intercept_
    expected = _r2(y, pred)
    original = _linear._normalize_score_response
    seen = []
    def checked(response, prediction):
        normalized = original(response, prediction)
        seen.append((tuple(normalized.shape), tuple(prediction.shape)))
        assert type(normalized).__module__.split('.')[0] == type(prediction).__module__.split('.')[0]
        if device == 'torch':
            assert normalized.device == prediction.device
        if device == 'cuda':
            assert normalized.device.id == prediction.device.id
        return normalized
    monkeypatch.setattr(_linear, '_normalize_score_response', checked)
    assert_allclose(model.score(array(query), array(y)), expected, rtol=1e-11, atol=1e-11)
    assert_allclose(model.score(array(query), array(y[:, None])), expected, rtol=1e-11, atol=1e-11)
    assert seen == [((10,), (10,)), ((10,), (10,))]
    if perfect:
        assert_allclose(expected, 1., atol=1e-13)


@pytest.mark.parametrize('shape', [(), (1,), (9,), (1, 10), (10, 2), (10, 1, 1), (10, 10)])
def test_invalid_single_response_shapes_fail_before_broadcast(native, shape):
    device, array = native
    X = np.arange(10.).reshape(-1, 1)
    model = LinearRegression(device=device, compute_inference=False).fit(array(X), array(1 + 2 * X[:, 0]))
    with pytest.raises(ValueError, match='sample and target shape'):
        model.score(array(X), array(np.ones(shape)))


def test_multi_response_requires_matching_target_dimensions(native):
    device, array = native
    rng = np.random.default_rng(178)
    X = rng.normal(size=(20, 2))
    y = X @ np.array([[2., -.3], [.2, 1.]]) + rng.normal(scale=.2, size=(20, 2))
    model = LinearRegression(device=device, compute_inference=False).fit(array(X), array(y))
    pred = X @ model.coef_.T + model.intercept_
    assert_allclose(model.score(array(X), array(y)), _r2(y, pred), atol=1e-11)
    for bad in (y[:, 0], y[:, :1], y.T, np.ones((1, 2)), np.ones((20, 2, 1))):
        with pytest.raises(ValueError, match='sample and target shape'):
            model.score(array(X), array(bad))
    with pytest.raises(ValueError, match='sample and target shape'):
        model.score(array(X[:2]), array(y[0]))


def test_constant_response_keeps_existing_zero_score(native):
    device, array = native
    X = np.arange(8.).reshape(-1, 1)
    y = np.ones((8, 1))
    model = LinearRegression(device=device, compute_inference=False).fit(array(X), array(y))
    assert model.score(array(X), array(y)) == 0.
