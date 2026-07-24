'''Behavioral regressions for the complete PR #79 review fixes.'''

from __future__ import annotations

import builtins

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.survival import CoxPH


def _cox_sample(seed=7901, n=100, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.45, -0.2, p)
    failure = -np.log(rng.random(n)) / np.exp(X @ beta)
    censor = rng.exponential(np.median(failure), size=n)
    event = (failure <= censor).astype(np.int32)
    time = np.minimum(failure, censor)
    return X, time, event


def _minimal_fit_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4, dtype=np.int32)
    return X, time, event


def test_cpu_cox_line_search_failure_does_not_update_beta(monkeypatch):
    X, time, event = _minimal_fit_data()
    model = CoxPH(
        device='cpu', compute_inference=False, compute_cindex=False, max_iter=3
    )

    def derivatives(beta, *_args, **_kwargs):
        return np.ones_like(beta), -np.eye(beta.size)

    def objective(beta, *_args, **_kwargs):
        return 0.0 if np.array_equal(beta, np.zeros_like(beta)) else -1.0

    monkeypatch.setattr(model, '_compute_gradient_hessian', derivatives)
    monkeypatch.setattr(model, '_compute_log_likelihood', objective)
    model.fit(X, time=time, event=event)

    assert_allclose(model.coef_, np.zeros(1), atol=0.0)
    assert model._objective_history == [0.0]


def test_cpu_cox_line_search_failure_is_not_converged(monkeypatch):
    X, time, event = _minimal_fit_data()
    model = CoxPH(
        device='cpu', compute_inference=False, compute_cindex=False, max_iter=3
    )

    monkeypatch.setattr(
        model,
        '_compute_gradient_hessian',
        lambda beta, *_args, **_kwargs: (np.ones_like(beta), -np.eye(beta.size)),
    )
    monkeypatch.setattr(
        model,
        '_compute_log_likelihood',
        lambda beta, *_args, **_kwargs: (
            0.0 if np.array_equal(beta, np.zeros_like(beta)) else -1.0
        ),
    )
    model.fit(X, time=time, event=event)

    assert model.converged_ is False
    assert model.termination_reason_ == 'line_search_failed'
    assert model.final_kkt_normalized_ is not None


def test_cpu_cox_small_step_large_kkt_is_stalled(monkeypatch):
    X, time, event = _minimal_fit_data()
    model = CoxPH(
        device='cpu', compute_inference=False, compute_cindex=False, max_iter=3
    )
    monkeypatch.setattr(
        model,
        '_compute_gradient_hessian',
        lambda beta, *_args, **_kwargs: (
            np.ones_like(beta), -1e20 * np.eye(beta.size)
        ),
    )
    monkeypatch.setattr(
        model, '_compute_log_likelihood', lambda *_args, **_kwargs: 0.0
    )
    model.fit(X, time=time, event=event)

    assert model.converged_ is False
    assert model.termination_reason_ == 'stalled_with_large_kkt'
    assert model.final_kkt_normalized_ > 1e-7


def test_cpu_cox_final_kkt_overrides_false_success(monkeypatch):
    X, time, event = _minimal_fit_data()
    model = CoxPH(
        device='cpu', compute_inference=False, compute_cindex=False, max_iter=3
    )
    calls = {'count': 0}

    def derivatives(beta, *_args, **_kwargs):
        calls['count'] += 1
        gradient = np.zeros_like(beta) if calls['count'] == 2 else np.ones_like(beta)
        return gradient, -1e20 * np.eye(beta.size)

    monkeypatch.setattr(model, '_compute_gradient_hessian', derivatives)
    monkeypatch.setattr(
        model, '_compute_log_likelihood', lambda *_args, **_kwargs: 0.0
    )
    model.fit(X, time=time, event=event)

    assert calls['count'] >= 3
    assert model.converged_ is False
    assert model.termination_reason_ == 'stalled_with_large_kkt'


@pytest.mark.parametrize('backend', ['cpu', 'cupy', 'torch'])
def test_cpu_cupy_torch_termination_contract_matches(backend):
    X, time, event = _cox_sample(n=70, p=2)
    X_backend = X
    device = 'cpu'
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CuPy CUDA unavailable')
        X_backend = cp.asarray(X)
        device = 'cuda'
    elif backend == 'torch':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('Torch CUDA unavailable')
        X_backend = torch.as_tensor(X, dtype=torch.float64, device='cuda')
        device = 'torch'

    model = CoxPH(
        device=device, penalty=0.1, ties='efron', compute_cindex=False,
        max_iter=100, tol=1e-7,
    ).fit(X_backend, time=time, event=event)

    assert model.converged_ is True
    assert model.termination_reason_ == 'kkt_converged'
    assert model.final_kkt_normalized_ <= 1e-7
    assert model.n_iter_ == model._iterations


def test_cpu_penalized_objective_is_monotone_within_tolerance():
    X, time, event = _cox_sample()
    model = CoxPH(
        device='cpu', penalty=0.1, ties='efron', compute_cindex=False,
        max_iter=100, tol=1e-7,
    ).fit(X, time=time, event=event)

    history = np.asarray(model._objective_history)
    assert history.size >= 2
    assert np.all(np.diff(history) >= -1e-10)
    assert np.isclose(model._penalized_objective, history[-1], atol=1e-9)


def test_delayed_entry_penalty_and_robust_contracts_are_explicit():
    X, time, event = _cox_sample(n=45, p=2)
    entry = np.maximum(0.0, time * 0.25)
    # Penalised delayed-entry on CPU is unsupported (Guard 2).
    with pytest.raises(NotImplementedError, match='penalty'):
        CoxPH(device='cpu', penalty=0.1).fit(
            X, time=time, event=event, entry=entry
        )
    # Robust covariance with delayed entry + inference is unsupported (Guard 1).
    with pytest.raises(NotImplementedError, match='Robust/cluster'):
        CoxPH(device='cpu', cov_type='hc0', compute_inference=True).fit(
            X, time=time, event=event, entry=entry
        )
    # But robust covariance with delayed entry is allowed when inference is off.
    model = CoxPH(
        device='cpu', cov_type='hc0', compute_inference=False, compute_cindex=False,
    ).fit(X, time=time, event=event, entry=entry)
    assert model.coef_ is not None
    assert model._bse is None
    assert model._conf_int is None


def test_delayed_entry_missing_statsmodels_has_actionable_error(monkeypatch):
    X, time, event = _cox_sample(n=30, p=2)
    entry = np.maximum(0.0, time * 0.2)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith('statsmodels.duration'):
            raise ImportError('blocked for test')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    with pytest.raises(ImportError, match=r'statgpu\[survival\]'):
        CoxPH(device='cpu').fit(X, time=time, event=event, entry=entry)


def test_robust_strict_requires_exact_dependency(monkeypatch):
    X, time, event = _cox_sample(n=55, p=2)
    model = CoxPH(
        device='cpu', ties='efron', cov_type='hc0', inference_mode='strict',
        compute_cindex=False,
    )
    monkeypatch.setattr(
        model, '_score_residuals_via_statsmodels_if_available',
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match='inference_mode=approx'):
        model.fit(X, time=time, event=event)


def test_robust_strict_breslow_uses_internal_exact_residuals(monkeypatch):
    X, time, event = _cox_sample(n=55, p=2)
    model = CoxPH(
        device='cpu', ties='breslow', cov_type='hc0', inference_mode='strict',
        compute_cindex=False,
    )
    monkeypatch.setattr(
        model, '_score_residuals_via_statsmodels_if_available',
        lambda *_args, **_kwargs: None,
    )
    model.fit(X, time=time, event=event)

    assert model.inference_method_ == 'exact_breslow_score_sandwich'
    assert model.inference_backend_ == 'numpy'
    assert model.inference_approximate_ is False


def test_robust_approx_is_explicit_and_disclosed(monkeypatch):
    X, time, event = _cox_sample(n=55, p=2)
    model = CoxPH(
        device='cpu', ties='efron', cov_type='hc0', inference_mode='approx',
        compute_cindex=False,
    )
    monkeypatch.setattr(
        model, '_score_residuals_via_statsmodels_if_available',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('approx mode must not select the exact dependency')
        ),
    )
    model.fit(X, time=time, event=event)

    assert model.inference_method_ == 'event_row_score_sandwich'
    assert model.inference_backend_ == 'numpy'
    assert model.inference_approximate_ is True
    assert model.inference_fallback_reason_


def test_cpu_prediction_contract_validation_and_custom_times():
    X, time, event = _cox_sample(n=65, p=3)
    model = CoxPH(device='cpu', compute_cindex=False).fit(X, time=time, event=event)

    one = model.predict_risk_score(X[0])
    assert isinstance(one, np.ndarray)
    assert one.shape == (1,)
    with pytest.raises(ValueError, match='two-dimensional'):
        model.predict(X.reshape(5, 13, 3))
    with pytest.raises(ValueError, match='features'):
        model.predict(np.zeros((2, 2)))
    bad = X[:2].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN or infinite'):
        model.predict(bad)

    requested = np.array([0.0, np.median(model._unique_times), time.max() * 2.0])
    survival, returned_times = model.predict_survival(X[:2], times=requested)
    assert survival.shape == (2, 3)
    assert_allclose(returned_times, requested)
    assert_allclose(survival[:, 0], np.ones(2))


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
def test_gpu_prediction_is_native_and_does_not_require_full_host_transfer(backend):
    X, time, event = _cox_sample(n=55, p=2)
    if backend == 'cupy':
        xp = pytest.importorskip('cupy')
        if xp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CuPy CUDA unavailable')
        X_backend = xp.asarray(X)
        model = CoxPH(device='cuda', compute_cindex=False)
        native_type = xp.ndarray
    else:
        xp = pytest.importorskip('torch')
        if not xp.cuda.is_available():
            pytest.skip('Torch CUDA unavailable')
        X_backend = xp.as_tensor(X, dtype=xp.float64, device='cuda')
        model = CoxPH(device='torch', compute_cindex=False)
        native_type = xp.Tensor

    model.fit(X_backend, time=time, event=event)
    prediction = model.predict(X_backend[:4])
    survival, prediction_times = model.predict_survival(X_backend[:4])
    assert isinstance(prediction, native_type)
    assert isinstance(survival, native_type)
    assert isinstance(prediction_times, native_type)
    assert model.full_host_transfer_performed_ is False
    assert_allclose(
        model._to_numpy(prediction), np.exp(X[:4] @ model.coef_),
        rtol=1e-8, atol=1e-9,
    )


@pytest.mark.parametrize('backend', ['numpy', 'torch', 'cupy'])
def test_rbf_complex_inputs_fail_consistently(backend):
    from statgpu.nonparametric.kernel_methods._kernels import rbf_kernel

    X = np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0 - 1.0j]])
    if backend == 'numpy':
        X_backend, xp = X, np
    elif backend == 'torch':
        xp = pytest.importorskip('torch')
        X_backend = xp.as_tensor(X)
    else:
        xp = pytest.importorskip('cupy')
        if xp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CuPy CUDA unavailable')
        X_backend = xp.asarray(X)
    with pytest.raises(ValueError, match='complex-valued'):
        rbf_kernel(X_backend, xp=xp)


def test_torch_streaming_hessian_matches_grouped_reference():
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(79)
    X = torch.randn(24, 4, dtype=torch.float64, generator=generator)
    exp_eta = torch.exp(torch.randn(24, dtype=torch.float64, generator=generator))
    X_exp = X * exp_eta[:, None]
    first_idx = torch.tensor([0, 5, 13, 20], dtype=torch.int64)
    risk_at = torch.stack([exp_eta[index:].sum() for index in first_idx])
    risk_X = torch.flip(torch.cumsum(torch.flip(X_exp, dims=[0]), dim=0), dims=[0])
    weights = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    total = X_exp.T @ X

    model = CoxPH(device='cpu')
    actual = model._compute_hessian_grouped_streaming_torch(
        X, X_exp, total, risk_at, risk_X, first_idx, weights
    )
    expected = torch.zeros_like(total)
    for group, index_tensor in enumerate(first_idx):
        index = int(index_tensor)
        second = X_exp[index:].T @ X[index:]
        mean = risk_X[index] / risk_at[group]
        expected -= weights[group] * (second / risk_at[group] - torch.outer(mean, mean))

    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12)
    assert model._last_torch_hessian_peak_shape_ == (4, 4)


def test_cox_chi_square_tests_use_distribution_survival_function(monkeypatch):
    import statgpu.survival._cox as cox_module
    from scipy import stats

    class RecordingChiSquare:
        def __init__(self):
            self.calls = []

        def sf(self, value, *, df):
            self.calls.append((value, df))
            return stats.chi2.sf(value, df)

    distribution = RecordingChiSquare()
    monkeypatch.setattr(cox_module, 'chi2', distribution)
    X, time, event = _cox_sample(n=55, p=2)
    CoxPH(device='cpu', compute_cindex=False).fit(X, time=time, event=event)

    assert len(distribution.calls) >= 3
    assert all(df == X.shape[1] for _, df in distribution.calls)


@pytest.mark.gpu
def test_cupy_gaussian_inference_uses_cholesky_solves(monkeypatch):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CuPy CUDA unavailable')
    from statgpu.backends._gpu_inference_cupy import compute_inference_gpu

    monkeypatch.setattr(
        cp.linalg, 'inv',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('cp.linalg.inv must not be used after Cholesky')
        ),
    )
    X = cp.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    params = cp.asarray([0.5, 1.0])
    outputs = compute_inference_gpu(
        X, cp.zeros(4), cp.asarray(0.25), df_resid=2, params_gpu=params
    )
    assert all(bool(cp.all(cp.isfinite(value))) for value in outputs)
