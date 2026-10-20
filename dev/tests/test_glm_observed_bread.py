"""Observed inference bread is distinct from IG/NB Fisher solver curvature."""
import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.inference._sandwich import compute_bread_avg, m_estimation_inference
from statgpu.linear_model import PenalizedGeneralizedLinearModel, PenalizedGLM_CV


def data(family):
    rng = np.random.default_rng(286)
    X = rng.normal(scale=.35, size=(96, 3))
    mu = np.exp(.3 + X @ np.array([.2, -.3, .15]))
    y = (rng.wald(mu, 3.) if family == 'inverse_gaussian'
         else rng.negative_binomial(1/.7, 1/(1+.7*mu)).astype(float))
    return X, y, np.linspace(.4, 1.8, len(y))


def moments(family, X, y, beta):
    mu = np.exp(X @ beta)
    if family == 'inverse_gaussian':
        return (mu-y)/mu**2, (2*y-mu)/mu**2, 1/mu
    return (mu-y)/(1+.7*mu), mu*(1+.7*y)/(1+.7*mu)**2, mu/(1+.7*mu)


def native(backend, *arrays):
    if backend.startswith('torch'):
        torch = pytest.importorskip('torch')
        if backend == 'torch_cuda' and not torch.cuda.is_available():
            pytest.skip('physical Torch CUDA unavailable')
        device = 'cuda' if backend == 'torch_cuda' else 'cpu'
        return [torch.as_tensor(a, dtype=torch.float64, device=device) for a in arrays]
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('physical CuPy CUDA unavailable')
        return [cp.asarray(a) for a in arrays]
    return arrays


def host(a):
    if hasattr(a, 'detach'):
        return a.detach().cpu().numpy()
    return a.get() if hasattr(a, 'get') else np.asarray(a)


@pytest.mark.parametrize('family', ['inverse_gaussian', 'negative_binomial'])
@pytest.mark.parametrize('backend', ['numpy', 'torch_cpu', 'cupy', 'torch_cuda'])
@pytest.mark.parametrize('weighted', [False, True])
def test_observed_derivative_fisher_and_native_sandwich(family, backend, weighted):
    X, y, weights = data(family)
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.array([.3, .2, -.3, .15])
    w = weights if weighted else np.ones(len(y))
    g, h, fisher = moments(family, X, y, beta)
    H = X.T @ ((w*h)[:, None]*X)/w.sum()
    F = X.T @ ((w*fisher)[:, None]*X)/w.sum()
    curv = np.array([0., .07, .07, .07])
    B = np.linalg.inv(H + np.diag(curv))
    J = X.T @ (((w*g)**2)[:, None]*X)/w.sum()
    expected = B @ J @ B/w.sum()
    loss = get_glm_loss(family, **({'alpha': .7} if family == 'negative_binomial' else {}))
    xn, yn, bn, wn, cn = native(backend, X, y, beta, w, curv)
    sw = wn if weighted else None
    observed = loss.observed_hessian(xn, yn, bn, sw)
    assert type(observed) is type(xn)
    if backend.startswith('torch') or backend == 'cupy':
        assert observed.device == xn.device
    np.testing.assert_allclose(host(observed), H, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(host(loss.hessian(xn, yn, bn, sw)), F, rtol=2e-13, atol=2e-14)
    assert not np.allclose(H, F, rtol=1e-3, atol=1e-5)
    numerical = np.column_stack([
        (loss.gradient(X, y, beta+1e-6*e, sample_weight=w)
         - loss.gradient(X, y, beta-1e-6*e, sample_weight=w))/(2e-6)
        for e in np.eye(len(beta))])
    np.testing.assert_allclose(H, numerical, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(host(compute_bread_avg(loss, xn, yn, bn,
        sample_weight=sw, use_fisher=True)), np.linalg.inv(F), rtol=2e-12)
    for cov_type in ('hc0', 'hc1'):
        out = m_estimation_inference(loss, xn, yn, bn, cov_type=cov_type,
                                    sample_weight=sw, penalty_curvature_diag=cn)
        scale = len(y)/(len(y)-len(beta)) if cov_type == 'hc1' else 1.
        np.testing.assert_allclose(out['cov_params'], expected*scale, rtol=2e-12, atol=1e-14)
        assert out['hessian_type'] == 'observed'
        assert out['distribution'] == 'normal'


@pytest.mark.parametrize('family', ['inverse_gaussian', 'negative_binomial'])
@pytest.mark.parametrize('cov_type', ['hc0', 'hc1', 'nonrobust'])
def test_penalized_public_and_cv_final_refit(family, cov_type):
    X, y, weights = data(family)
    settings = {'loss_kwargs': {'alpha': .7}} if family == 'negative_binomial' else {}
    common = dict(loss=family, penalty='l2', device='cpu', solver='newton',
                  tol=1e-9, max_iter=2000, compute_inference=True, cov_type=cov_type, **settings)
    direct = PenalizedGeneralizedLinearModel(alpha=.04, **common).fit(X, y, sample_weight=weights)
    cv = PenalizedGLM_CV(alpha_grid=[.04, .08], cv=2, random_state=4, **common).fit(X, y, sample_weight=weights)
    assert cv.penalty_conditioning_ == 'cv_selected_penalty'
    assert cv.penalty_selection_adjusted_ is False
    assert cv._inference_result is cv.estimator_._inference_result
    # Public fits normalize analytic weights to sum n before inference.
    weights = weights / weights.mean()
    for model in (direct, cv.estimator_):
        params = np.r_[model.intercept_, model.coef_]
        design = np.column_stack([np.ones(len(y)), X])
        g, h, fisher = moments(family, design, y, params)
        curvature = fisher if cov_type == 'nonrobust' else h
        H = design.T @ ((weights*curvature)[:, None]*design)/weights.sum()
        B = np.linalg.inv(H + np.diag(np.r_[0., np.repeat(model.alpha, X.shape[1])]))
        if cov_type == 'nonrobust':
            mu = np.exp(design @ params)
            dispersion = (np.sum(weights*(y-mu)**2 / mu**3)/(len(y)-len(params))
                          if family == 'inverse_gaussian' else 1.)
            cov = dispersion*B/weights.sum()
        else:
            J = design.T @ (((weights*g)**2)[:, None]*design)/weights.sum()
            cov = B @ J @ B/weights.sum()
            if cov_type == 'hc1':
                cov *= len(y)/(len(y)-len(params))
        np.testing.assert_allclose(model._bse, np.sqrt(np.diag(cov)), rtol=2e-9, atol=2e-11)
        assert model.inference_method_ == 'm_estimation'
        assert model.inference_target_ == 'penalized_estimating_equation'


@pytest.mark.parametrize('family', ['inverse_gaussian', 'negative_binomial'])
def test_formula_alignment(family):
    pd = pytest.importorskip('pandas')
    X, y, _ = data(family)
    settings = {'loss_kwargs': {'alpha': .7}} if family == 'negative_binomial' else {}
    kw = dict(loss=family, penalty='l2', alpha=.04, device='cpu', solver='newton',
              tol=1e-9, max_iter=2000, compute_inference=True, cov_type='hc0', **settings)
    array = PenalizedGeneralizedLinearModel(**kw).fit(X, y)
    frame = pd.DataFrame(X, columns=['a', 'b', 'c']); frame['y'] = y
    formula = PenalizedGeneralizedLinearModel(**kw).fit(formula='y ~ a + b + c', data=frame)
    np.testing.assert_allclose(array._params, formula._params, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(array._bse, formula._bse, rtol=1e-10, atol=1e-12)


def test_duck_typed_nonrobust_fallback_is_solver_hessian():
    class LegacyLoss:
        def hessian(self, X, y, coef, sample_weight=None):
            return np.eye(X.shape[1])*2

        def observed_hessian(self, X, y, coef, sample_weight=None):
            return np.eye(X.shape[1])*4

    X = np.ones((4, 1)); y = np.ones(4); coef = np.ones(1)
    np.testing.assert_array_equal(compute_bread_avg(LegacyLoss(), X, y, coef,
                                                   use_fisher=True), [[.5]])
    np.testing.assert_array_equal(compute_bread_avg(LegacyLoss(), X, y, coef), [[.25]])
