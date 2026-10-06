"""Sixth-pass connected Gaussian/Poisson inference documentation contracts.

CPU-only executable examples and analytic checks; the exact-Ridge offset defect
is a narrow strict xfail, not an approved numerical behavior.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.linear_model import (
    MCPRegression,
    PenalizedLinearRegression,
    PoissonRegression,
    Ridge,
    SCADRegression,
)

ROOT = Path(__file__).resolve().parents[2]


class _RidgeCenteredObjectiveMismatch(AssertionError):
    """Finite optimized exact coefficients do not solve the centered objective."""


def _example(language, page, marker):
    text = (ROOT / 'docs' / language / page).read_text()
    match = re.search(r'<!-- ' + re.escape(marker) + r' -->\s*```python\n(.*?)```',
                      text, re.DOTALL)
    assert match is not None
    scope = {}
    exec(compile(match[1], f'{language}/{page}:{marker}', 'exec'), scope)  # noqa: S102
    return scope


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_poisson_unpenalized_example_solves_score_equation(language):
    ns = _example(language, 'models/poisson-regression.md',
                  'learner-example: poisson-unpenalized')
    model, X, y = ns['model'], ns['X'], ns['y']
    np.testing.assert_allclose(model.coef_, [.377, -.154], atol=.0005)
    design = np.column_stack([np.ones(len(y)), X])
    np.testing.assert_allclose(design.T @ (model.predict(X) - y), 0., atol=1e-7)
    assert model._conf_int.shape == (3, 2)
    assert model._inference_result.method == 'm_estimation'
    assert model._inference_result.distribution == 'normal'
    assert isinstance(model.summary(), str)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('C', [0., 1.])
def test_ordinary_poisson_irls_documented_penalty_scale(weighted, C):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(80, 2))
    y = rng.poisson(np.exp(.2 + X @ np.array([.3, -.2])))
    w = np.linspace(.5, 2., len(y)) if weighted else np.ones(len(y))
    model = PoissonRegression(C=C, solver='irls', device='cpu', max_iter=1000,
                              tol=1e-10).fit(X, y, sample_weight=w if weighted else None)
    residual = w * (model.predict(X) - y) / w.sum()
    gradient = X.T @ residual
    if C:
        gradient += model.coef_ / (2 * C)
    np.testing.assert_allclose(gradient, 0., atol=2e-7)
    assert abs(residual.sum()) < 2e-7


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('cls,name', [(PoissonRegression, 'poisson-regression'),
                                     (SCADRegression, 'scad'), (MCPRegression, 'mcp'),
                                     (Ridge, 'ridge')])
def test_connected_model_parameter_tables_cover_live_constructor(language, cls, name):
    text = (ROOT / f'docs/{language}/models/{name}.md').read_text()
    for parameter in inspect.signature(cls).parameters:
        assert f'| `{parameter}` |' in text, (cls.__name__, parameter)
    if cls is PoissonRegression:
        assert '| `formula` |' not in text
        assert '| `data` |' not in text
        assert 'failed-ordinary-glm-refits' in text


@pytest.mark.parametrize('cls', [PoissonRegression, SCADRegression, MCPRegression])
def test_connected_wrapper_help_covers_constructor_parameters(cls):
    help_text = inspect.getdoc(cls)
    for parameter in inspect.signature(cls).parameters:
        assert re.search(r'^' + parameter + r'\s*:', help_text, re.MULTILINE), parameter


@pytest.mark.parametrize('cls', [SCADRegression, MCPRegression])
def test_specialized_nonconvex_wrapper_requires_generic_inference_interface(cls):
    assert 'inference_method' not in inspect.signature(cls).parameters
    with pytest.raises(TypeError, match='inference_method'):
        cls(inference_method='oracle')
    model = cls(device='cpu', compute_inference=True)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + rng.normal(size=30)
    with pytest.raises(NotImplementedError, match='auto does not silently select oracle'):
        model.fit(X, y)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_generic_gaussian_oracle_example_matches_active_set_ols(language):
    ns = _example(language, 'guides/inference-modes.md',
                  'inference-example: gaussian-nonconvex-oracle')
    model, X, y = ns['model'], ns['X'], ns['y']
    assert model._inference_result.method == 'oracle'
    active = np.flatnonzero(model._inference_result.metadata['active_set'])
    expected = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X[:, active]]), y,
                              rcond=None)[0]
    np.testing.assert_allclose(model._inference_result.params[np.r_[0, active + 1]],
                               expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.predict(X), model.intercept_ + X @ model.coef_)
    mcp = PenalizedLinearRegression(
        penalty='mcp', penalty_kwargs={'gamma': 3.0}, alpha=.1, device='cpu',
        compute_inference=True, inference_method='oracle',
    ).fit(X, y)
    assert mcp._inference_result.method == 'oracle'


def _ridge_problem(weighted):
    rng = np.random.default_rng(113)
    variation = rng.normal(size=(40, 3))
    y = .4 + variation @ np.array([1., -.5, .3]) + rng.normal(scale=.1, size=40)
    X = variation + 1e8
    w = np.linspace(.5, 2., len(y)) if weighted else np.ones(len(y))
    origin = np.average(X, axis=0, weights=w)
    centered = X - origin
    ymean = np.average(y, weights=w)
    gram = centered.T @ (w[:, None] * centered) + .1 * w.sum() * np.eye(3)
    coef = np.linalg.solve(gram, centered.T @ (w * (y - ymean)))
    return X, y, w, origin, coef, ymean + centered @ coef


@pytest.mark.xfail(strict=True, raises=_RidgeCenteredObjectiveMismatch,
                   reason='Issue #241: optimized exact Ridge raw-moment cancellation under large offsets')
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('inference', [False, True])
def test_exact_ridge_should_solve_translated_average_loss_objective(weighted, inference):
    X, y, w, _origin, expected, _prediction = _ridge_problem(weighted)
    model = Ridge(alpha=.1, solver='exact', device='cpu',
                  compute_inference=inference).fit(X, y, sample_weight=w if weighted else None)
    assert model.coef_.shape == expected.shape
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.predict(X)).all()
    # Attribute only this known raw-moment failure to #241. A different wrong
    # finite answer, wrong shape or NaN must fail rather than become an xfail.
    mean = np.average(X, axis=0, weights=w)
    ymean = np.average(y, weights=w)
    raw_gram = (X * w[:, None]).T @ X - w.sum() * np.outer(mean, mean)
    raw_rhs = (X * w[:, None]).T @ y - w.sum() * mean * ymean
    if not weighted:
        # Match the unweighted raw-moment expression's operation ordering.
        raw_gram = X.T @ X - len(y) * np.outer(X.mean(0), X.mean(0))
        raw_rhs = X.T @ y - len(y) * X.mean(0) * y.mean()
    raw_solution = np.linalg.solve(raw_gram + .1*w.sum()*np.eye(3), raw_rhs)
    _check_ridge_solution(model.coef_, expected, raw_solution)


def _check_ridge_solution(actual, centered_solution, raw_solution):
    assert actual.shape == centered_solution.shape
    assert np.isfinite(actual).all()
    if np.allclose(actual, centered_solution, atol=1e-7, rtol=1e-7):
        return
    assert np.allclose(actual, raw_solution, atol=1e-7, rtol=1e-7), (
        'A different finite numerical failure must not be hidden by the #241 xfail'
    )
    raise _RidgeCenteredObjectiveMismatch(
        f'exact coefficients {actual!r} match the unstable raw-moment solve '
        f'instead of the centered solution {centered_solution!r}'
    )


def test_ridge_xfail_guard_distinguishes_repair_and_unrelated_failures():
    expected = np.array([1., 2., 3.])
    raw = np.array([-1., 3., .1])
    assert _check_ridge_solution(expected, expected, raw) is None
    with pytest.raises(_RidgeCenteredObjectiveMismatch):
        _check_ridge_solution(raw, expected, raw)
    for bad in [np.zeros(2), np.full(3, np.nan), raw + .01]:
        with pytest.raises(AssertionError) as error:
            _check_ridge_solution(bad, expected, raw)
        assert error.type is AssertionError


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('inference', [False, True])
def test_training_origin_workaround_preserves_ridge_objective(weighted, inference):
    X, y, w, origin, expected, prediction = _ridge_problem(weighted)
    model = Ridge(alpha=.1, solver='exact', device='cpu',
                  compute_inference=inference).fit(X-origin, y, sample_weight=w if weighted else None)
    np.testing.assert_allclose(model.coef_, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.predict(X-origin), prediction, atol=1e-8, rtol=1e-8)
    if inference:
        assert np.isfinite(model._conf_int).all()
    fista = Ridge(alpha=.1, solver='fista', device='cpu', compute_inference=False,
                  max_iter=10000, tol=1e-10).fit(X, y, sample_weight=w if weighted else None)
    np.testing.assert_allclose(fista.coef_, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_ridge_centering_example_reuses_training_origin(language):
    ns = _example(language, 'models/ridge.md', 'learner-example: ridge-training-origin')
    model, X, y = ns['model'], ns['X_train'], ns['y_train']
    centered = X - ns['origin']
    ymean = y.mean()
    expected = np.linalg.solve(centered.T @ centered + len(y)*.1*np.eye(3),
                               centered.T @ (y-ymean))
    np.testing.assert_allclose(model.coef_, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(ns['prediction'], model.intercept_ + (ns['X_test']-ns['origin']) @ expected,
                               atol=1e-10, rtol=1e-10)
    assert ns['prediction'].shape == (20,)
    assert np.isfinite(model._conf_int).all()
    assert ns['original_intercept'] == pytest.approx(model.intercept_-ns['origin'] @ expected)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_connected_pages_describe_current_behavior_without_obsolete_inference_claims(language):
    scad = (ROOT / f'docs/{language}/models/scad.md').read_text()
    mcp = (ROOT / f'docs/{language}/models/mcp.md').read_text()
    ridge = (ROOT / f'docs/{language}/models/ridge.md').read_text()
    for text in [scad, mcp]:
        assert 'PenalizedLinearRegression' in text
        assert 'inference_method' in text
        assert 'scadmcp-active-set-inference' in text
    assert 'TO_DO.md' not in mcp
    assert 'bias decreasing as' not in mcp
    assert '偏差随' not in mcp
    assert 'PenalizedLinearRegression(loss=' not in ridge
    for obsolete in ('exact-source remote gate', 'Issue #127', 'fail closed', 'maintained validator'):
        assert obsolete not in ridge
    assert 'large-feature-offsets' in ridge
    assert 'training-derived origin' in inspect.getdoc(Ridge)
