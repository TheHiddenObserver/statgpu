"""Concrete CPU checks for the guide/panel plain-language explanations.

These are small executable examples and analytic checks, not GPU evidence or
claims about statistical coverage. Production code is unchanged.
"""
from pathlib import Path
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu import LinearRegression, RegressionDiagnostics
from statgpu.linear_model import Lasso, PenalizedPoissonRegression
from statgpu.panel import BetweenOLS, PanelOLS, PooledOLS

ROOT = Path(__file__).resolve().parents[2]


def _example(language, path, contains, namespace=None):
    text = (ROOT / 'docs' / language / path).read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    code = next(block for block in blocks if contains in block)
    ns = {} if namespace is None else dict(namespace)
    exec(compile(code, path, 'exec'), ns)
    return ns


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_quickstart_rows_features_and_three_predictions(language):
    ns = _example(language, 'getting-started/quickstart.md', 'size=(1000, 20)')
    X, y, model = ns['X'], ns['y'], ns['model']
    assert X.shape == (1000, 20)
    assert y.shape == (1000,)
    predicted = model.predict(X[:3])
    assert predicted.shape == (3,)
    assert_allclose(predicted, X[:3] @ model.coef_ + model.intercept_)
    assert model.score(X, y) > 0.99
    with pytest.raises(ValueError):
        model.predict(X[:3, :-1])


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_diagnostics_map_to_rows_and_nonintercept_columns(language):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(100, 2))
    y = 1 + X @ [2, -0.5] + rng.normal(scale=0.3, size=100)
    ns = _example(language, 'guides/regression-diagnostics.md', 'diag =', {'X': X, 'y': y})
    model, diag = ns['model'], ns['diag']
    design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    h = np.einsum('ij,ji->i', design, np.linalg.pinv(design))
    scale = residual @ residual / (100 - 3)
    external_scale = ((100 - 3) * scale - residual**2 / (1-h)) / (100 - 4)
    expected = [h, residual / np.sqrt(external_scale * (1-h)),
                residual**2 / (3 * scale) * h / (1-h)**2]
    actual = [diag.leverage, diag.externally_studentized_residuals, diag.cooks_distance]
    for result, reference in zip(actual, expected):
        assert result.shape == (100,)
        assert_allclose(result, reference, rtol=1e-9, atol=1e-11)
    expected_vif = 1 / (1 - np.corrcoef(X.T)[0, 1]**2)
    assert diag.vif().shape == (2,)
    assert_allclose(diag.vif(), [expected_vif, expected_vif])
    # A row permutation must permute row diagnostics, not the feature outputs.
    order = rng.permutation(100)
    permuted = RegressionDiagnostics(LinearRegression(device='cpu').fit(X[order], y[order]))
    assert_allclose(permuted.cooks_distance, diag.cooks_distance[order])
    assert_allclose(permuted.vif(), diag.vif())
    with pytest.raises(ValueError, match='missing fitted'):
        RegressionDiagnostics(LinearRegression(device='cpu'))


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_fixed_penalty_poisson_parameter_and_interval_order(language):
    ns = _example(language, 'guides/penalized-glm-inference.md', 'rng = np.random.default_rng(72)')
    model = ns['model']
    inference = {key: np.asarray(value) for key, value in ns['inference'].items()
                 if key in ('params', 'conf_int')}
    assert inference['params'].shape == (3,)
    assert inference['conf_int'].shape == (3, 2)
    params = np.r_[model.intercept_, model.coef_]
    assert_allclose(inference['params'], params)
    assert_allclose(inference['conf_int'].mean(axis=1), params)
    assert np.all(inference['conf_int'][:, 1] > inference['conf_int'][:, 0])
    assert model.inference_resolved_method_ == 'm_estimation'
    assert model.inference_target_ == 'penalized_estimating_equation'
    assert model.penalty_conditioning_ == 'fixed_penalty'
    unpenalized = PenalizedPoissonRegression(
        penalty='l2', alpha=0, compute_inference=True, inference_method='auto',
        cov_type='hc0', solver='lbfgs', device='cpu',
    ).fit(ns['X'], ns['y'], sample_weight=ns['w'])
    assert unpenalized.inference_target_ != model.inference_target_
    assert np.linalg.norm(unpenalized.coef_ - model.coef_) > 1e-4


def test_debiased_report_does_not_replace_prediction_coefficients():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(120, 3))
    y = 1 + X @ [2, -1, 0] + rng.normal(scale=0.2, size=120)
    options = dict(alpha=0.2, inference_method='debiased', device='cpu',
                   solver='fista', max_iter=2000, tol=1e-8)
    fitted = Lasso(compute_inference=True, **options).fit(X, y)
    prediction_only = Lasso(compute_inference=False, **options).fit(X, y)
    assert_allclose(fitted.coef_, prediction_only.coef_)
    assert_allclose(fitted.predict(X), X @ fitted.coef_ + fitted.intercept_)
    corrected = fitted._params
    assert corrected.shape == (4,)
    assert np.linalg.norm(corrected[1:] - fitted.coef_) > 0.05
    assert not np.allclose(fitted.predict(X), corrected[0] + X @ corrected[1:])
    assert_allclose(fitted._conf_int.mean(axis=1), corrected)


def test_averaging_or_robust_covariance_does_not_remove_entity_effect():
    entity = np.repeat(np.arange(12), 5)
    means = np.linspace(-2, 2, 12)
    x = means[entity] + np.tile(np.arange(-2, 3), 12)
    X = x[:, None]
    y = 1 + 2*x + 3*means[entity]
    between = BetweenOLS(device='cpu').fit(X, y, entity_ids=entity)
    fixed = PanelOLS(entity_effects=True, device='cpu').fit(X, y, entity_ids=entity)
    pooled = PooledOLS(device='cpu').fit(X, y, entity_ids=entity)
    robust = PooledOLS(device='cpu', cov_type='hc0').fit(X, y, entity_ids=entity)
    assert between.nobs == 12
    assert_allclose(between.coef_, [1, 5], atol=1e-12)
    assert_allclose(fixed.coef_, [2], atol=1e-12)
    assert pooled.coef_[1] > 2.5
    assert_allclose(robust.coef_, pooled.coef_)


def test_two_disconnected_groups_reject_cross_component_predictions():
    # Exactly the two sets of edges in both PanelOLS language editions.
    entity = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'])
    time = np.array([1, 2, 1, 2, 3, 4, 3, 4])
    X = np.random.default_rng(72).normal(size=(8, 1))
    entity_effect = np.repeat([0.2, 0.5, -0.3, 0.7], 2)
    y = 2*X[:, 0] + entity_effect + time*0.1
    model = PanelOLS(entity_effects=True, time_effects=True, device='cpu').fit(
        X, y, entity_ids=entity, time_ids=time)
    assert model.df_resid == 8 - (4 + 4 - 2) - 1
    assert_allclose(model.predict(X, entity_ids=entity, time_ids=time), y)
    for kwargs in [dict(entity_ids=['A'], time_ids=[3]),
                   dict(entity_ids=['A'], time_ids=[99]),
                   dict(entity_ids=['Z'], time_ids=[1]),
                   dict(entity_ids=['A']), dict(time_ids=[1])]:
        with pytest.raises(ValueError, match='both entity and time labels are known'):
            model.predict(X[:1], **kwargs)
    linear = X[:1] @ model.coef_
    assert_allclose(model.predict(X[:1]), linear)
    assert_allclose(model.predict(X[:1], entity_ids=['Z'], time_ids=[99]), linear)


def test_unobserved_pair_inside_one_component_is_allowed():
    entity = np.repeat(['A', 'B', 'C'], 3)
    time = np.tile([1, 2, 3], 3)
    keep = ~((entity == 'A') & (time == 3))
    entity, time = entity[keep], time[keep]
    X = np.random.default_rng(14).normal(size=(8, 1))
    effect = np.array([{'A': 0.2, 'B': 0.5, 'C': -0.3}[i] for i in entity])
    y = 2*X[:, 0] + effect + time*0.1
    model = PanelOLS(entity_effects=True, time_effects=True, device='cpu').fit(
        X, y, entity_ids=entity, time_ids=time)
    assert_allclose(model.predict([[0.4]], entity_ids=['A'], time_ids=[3]), [1.3], atol=1e-10)
