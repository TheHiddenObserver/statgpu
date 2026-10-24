"""Bounded regression coverage for the selected L2 CV refit's native solve.

Torch CPU exercises backend mechanics only, never physical CUDA evidence.
The public CUDA cases skip deterministically without a working GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from statgpu.backends import NumpyBackend, _to_numpy
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import _penalized_cv as cv


def problem(weighted):
    rng = np.random.default_rng(918)
    X = rng.normal(size=(24, 3)) + [1.0, -2.0, 0.5]
    y = 2.2 + X @ np.array([0.8, -0.4, 1.2]) + rng.normal(scale=.1, size=24)
    weights = np.linspace(.3, 2.0, len(y)) if weighted else None
    if weighted:
        weights[0] = 0.0
    return X, y, weights


def analytic_ridge(X, y, alpha, weights):
    w = np.ones(len(y)) if weights is None else weights
    design = np.column_stack([np.ones(len(y)), X])
    system = design.T @ (w[:, None] * design)
    system[1:, 1:] += w.sum() * alpha * np.eye(X.shape[1])
    return np.linalg.solve(system, design.T @ (w * y))


def backend_for(name):
    if name == 'numpy':
        return NumpyBackend()
    if name in {'torch_cpu', 'torch_cuda'}:
        torch = pytest.importorskip('torch')
        if name == 'torch_cuda' and not torch.cuda.is_available():
            pytest.skip('Physical Torch CUDA device unavailable')
        from statgpu.backends import TorchBackend
        return TorchBackend(device='cpu' if name == 'torch_cpu' else 'cuda:0')
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('Physical CuPy CUDA device unavailable')
    except Exception as exc:
        pytest.skip(f'Physical CuPy CUDA runtime unavailable: {exc}')
    from statgpu.backends import CuPyBackend
    return CuPyBackend()


@pytest.mark.parametrize('name', ['numpy', 'torch_cpu', 'torch_cuda', 'cupy'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('alpha', [0.0, .2])
def test_native_refit_matches_weighted_objective_without_host_solve(monkeypatch, name, weighted, alpha):
    backend = backend_for(name)
    X, y, weights = problem(weighted)
    expected = analytic_ridge(X, y, alpha, weights)
    native_X = backend.asarray(X, dtype=backend.float64)
    native_y = backend.asarray(y, dtype=backend.float64)
    native_w = None if weights is None else backend.asarray(weights, dtype=backend.float64)
    original_eigh = backend.eigh
    numerical_inputs = []

    def capture_eigh(gram):
        numerical_inputs.append(gram)
        return original_eigh(gram)

    monkeypatch.setattr(backend, 'eigh', capture_eigh)
    if name != 'numpy':
        def no_cpu(*args, **kwargs):
            raise AssertionError('accelerator helper must not execute the CPU Ridge solver')
        monkeypatch.setattr(cv, '_ridge_eig_single', no_cpu)
        monkeypatch.setattr(cv, '_to_numpy', no_cpu)
        monkeypatch.setattr(np.linalg, 'eigh', no_cpu)
    coef, intercept = cv._ridge_eig_single_backend(
        native_X, native_y, alpha, backend, sample_weight=native_w,
    )
    np.testing.assert_allclose(_to_numpy(coef), expected[1:], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(_to_numpy(intercept), expected[0], rtol=1e-11, atol=1e-11)
    if name != 'numpy':
        assert len(numerical_inputs) == 1
        assert str(numerical_inputs[0].dtype).endswith('float64')
        assert type(coef).__module__.split('.')[0] == ('torch' if name.startswith('torch') else 'cupy')
        assert coef.device == native_X.device


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('solver', ['auto', 'fista'])
@pytest.mark.parametrize('name', ['numpy', 'torch_cuda', 'cupy'])
def test_public_cv_selected_refit_contract(monkeypatch, name, solver, weighted):
    backend = backend_for(name)
    X, y, weights = problem(weighted)
    device = {'numpy': 'cpu', 'torch_cuda': 'torch', 'cupy': 'cuda'}[name]
    model = PenalizedGLM_CV(loss='squared_error', penalty='l2',
                            alpha_grid=[.05, .2], cv=2, device=device,
                            max_iter=3000, tol=1e-8, solver=solver)
    original = cv._ridge_eig_single_backend
    observed = []

    def capture(X, y, alpha, resolved_backend, sample_weight=None):
        result = original(X, y, alpha, resolved_backend, sample_weight)
        observed.append((resolved_backend.name, result[0], alpha))
        return result

    monkeypatch.setattr(cv, '_ridge_eig_single_backend', capture)
    native_X = backend.asarray(X, dtype=backend.float64)
    native_y = backend.asarray(y, dtype=backend.float64)
    native_w = None if weights is None else backend.asarray(weights, dtype=backend.float64)
    model.fit(native_X, native_y, sample_weight=native_w)
    assert len(observed) == 1
    assert observed[0][0] == backend.name
    assert observed[0][2] == model.alpha_
    expected = analytic_ridge(X, y, model.alpha_, weights)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.predict(native_X), X @ expected[1:] + expected[0], rtol=1e-10, atol=1e-10)
    assert isinstance(model.estimator_.coef_, np.ndarray)
    assert isinstance(model.estimator_.intercept_, float)
    assert model.estimator_._selected_backend_name == backend.name
    assert model.estimator_.fit_intercept is True
    assert model.estimator_._fitted is True


def test_explicit_torch_refit_does_not_silently_use_torch_cpu(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    X, y, _ = problem(False)
    model = PenalizedGLM_CV(device='torch')
    with pytest.raises(RuntimeError, match='requires.*cuda'):
        model._refit_best(X, y, .1)


def test_native_refit_propagates_numerical_failure(monkeypatch):
    backend = backend_for('torch_cpu')
    X, y, _ = problem(False)
    def failure(*args, **kwargs):
        raise RuntimeError('native eigensolve failed')
    monkeypatch.setattr(backend, 'eigh', failure)
    monkeypatch.setattr(cv, '_ridge_eig_single', lambda *a, **k: pytest.fail('CPU fallback'))
    with pytest.raises(RuntimeError, match='native eigensolve failed'):
        cv._ridge_eig_single_backend(X, y, .1, backend)
