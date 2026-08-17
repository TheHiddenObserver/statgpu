"""Regression coverage for numerically full-rank ill-conditioned panel fits."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, RandomEffects


def _shared_rcond(X):
    return max(X.shape) * np.finfo(np.float64).eps


def _assert_numerically_full_rank(X):
    singular = np.linalg.svd(X, compute_uv=False)
    cutoff = _shared_rcond(X) * singular.max()
    assert int(np.count_nonzero(singular > cutoff)) == X.shape[1]
    assert singular.max() / singular.min() > 1.0e7


def _near_collinear_columns(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    X = np.column_stack([x, x + 1.0e-7 * z])
    _assert_numerically_full_rank(X)
    return rng, X


def test_panelols_full_rank_near_collinearity_matches_shared_svd_solution():
    rng, X = _near_collinear_columns(200, 2026081701)
    beta = np.array([1.0, -1.0])
    y = X @ beta + 1.0e-9 * rng.normal(size=X.shape[0])
    expected = np.linalg.lstsq(X, y, rcond=_shared_rcond(X))[0]
    model = PanelOLS(cov_type="hc0").fit(X, y)
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=2e-8, atol=2e-8)
    assert_allclose(X @ model.coef_, X @ expected, rtol=0, atol=2e-12)


def test_between_full_rank_near_collinearity_matches_shared_svd_solution():
    rng, X = _near_collinear_columns(180, 2026081702)
    entity = np.arange(X.shape[0], dtype=np.int64)
    beta = np.array([1.0, -1.0])
    y = 0.35 + X @ beta + 1.0e-9 * rng.normal(size=X.shape[0])
    design = np.column_stack([np.ones(X.shape[0]), X])
    _assert_numerically_full_rank(design)
    expected = np.linalg.lstsq(design, y, rcond=_shared_rcond(design))[0]
    model = BetweenOLS(cov_type="hc0").fit(X, y, entity_ids=entity)
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=2e-8, atol=2e-8)
    assert_allclose(design @ model.coef_, design @ expected, rtol=0, atol=2e-12)


def test_first_difference_full_rank_near_collinearity_matches_shared_svd_solution():
    rng, differences = _near_collinear_columns(140, 2026081703)
    n_entities = differences.shape[0]
    entity = np.repeat(np.arange(n_entities), 2)
    time = np.tile(np.arange(2), n_entities)
    X = np.zeros((2 * n_entities, 2), dtype=np.float64)
    X[1::2] = differences
    alpha = rng.normal(scale=0.2, size=n_entities)
    beta = np.array([1.0, -1.0])
    y = np.repeat(alpha, 2)
    y[1::2] += differences @ beta + 1.0e-9 * rng.normal(size=n_entities)
    expected = np.linalg.lstsq(
        differences, y[1::2] - y[0::2], rcond=_shared_rcond(differences)
    )[0]
    model = FirstDifferenceOLS(cov_type="hc0").fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=2e-8, atol=2e-8)
    assert_allclose(
        differences @ model.coef_, differences @ expected, rtol=0, atol=2e-12
    )


def test_random_effects_full_rank_near_collinearity_matches_quasi_demeaned_svd():
    rng = np.random.default_rng(2026081704)
    n_entities, n_times = 60, 4
    entity = np.repeat(np.arange(n_entities), n_times)
    x = rng.normal(size=entity.size)
    z = rng.normal(size=entity.size)
    X = np.column_stack([np.ones(entity.size), x, x + 1.0e-7 * z])
    _assert_numerically_full_rank(X)
    alpha = np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)
    beta = np.array([0.4, 1.0, -1.0])
    y = X @ beta + alpha + 1.0e-9 * rng.normal(size=entity.size)
    model = RandomEffects(cov_type="hc0").fit(X, y, entity_ids=entity)
    theta = float(model.theta_)
    X_mean = np.repeat(
        X.reshape(n_entities, n_times, -1).mean(axis=1), n_times, axis=0
    )
    y_mean = np.repeat(y.reshape(n_entities, n_times).mean(axis=1), n_times)
    X_star = X - theta * X_mean
    y_star = y - theta * y_mean
    _assert_numerically_full_rank(X_star)
    expected = np.linalg.lstsq(X_star, y_star, rcond=_shared_rcond(X_star))[0]
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=3e-8, atol=3e-8)
    assert_allclose(X_star @ model.coef_, X_star @ expected, rtol=0, atol=3e-12)
