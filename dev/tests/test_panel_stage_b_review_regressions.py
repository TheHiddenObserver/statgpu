"""Regression tests for PR #122 review findings closed after physical promotion."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import RandomEffects
from statgpu.panel._diagnostic_context import (
    explicit_constant_column,
    fixed_effect_diagnostic_df,
)
from statgpu.panel._diagnostics import (
    _classical_model_f,
    _diagnostic_identity,
    _fingerprints_match,
    _hausman_quadratic,
    _pooling_f_from_sums,
)


def test_two_way_effect_rank_counts_disconnected_incidence_components():
    # Two disconnected incidence components:
    # {entities 0,1} x {times 0,1} and {entities 2,3} x {times 2,3}.
    entity = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    time = np.asarray([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    X_transformed = np.arange(1.0, 9.0).reshape(-1, 1)

    result = fixed_effect_diagnostic_df(
        X_transformed,
        xp=np,
        nobs=8,
        n_entities=4,
        n_times=4,
        entity_effects=True,
        time_effects=True,
        entity_codes=entity,
        time_codes=time,
    )

    assert result["incidence_components"] == 2
    assert result["effect_rank"] == 4 + 4 - 2
    assert result["rank_x"] == 1
    assert result["df_resid"] == 1
    assert result["df_total"] == 2


def test_hausman_identity_rejects_low_order_moment_collision():
    # These response vectors have the same sum, sum of squares, and
    # row-weighted sum.  The previous low-order fingerprint therefore collided.
    y_left = np.asarray([0.0, 1.0, 3.0, 2.0])
    y_right = np.asarray([0.0, 2.0, 1.0, 3.0])
    X = np.asarray(
        [
            [0.2, -0.4],
            [0.7, 0.1],
            [-0.3, 0.5],
            [1.1, -0.2],
        ]
    )
    entity = np.asarray([0, 0, 1, 1], dtype=np.int64)

    assert y_left.sum() == y_right.sum()
    assert np.dot(y_left, y_left) == np.dot(y_right, y_right)
    weights = np.arange(1.0, 5.0)
    assert np.dot(y_left, weights) == np.dot(y_right, weights)

    left = _diagnostic_identity(X, y_left, xp=np, entity_codes=entity)
    right = _diagnostic_identity(X, y_right, xp=np, entity_codes=entity)
    matched, reason = _fingerprints_match(left, right)

    assert not matched
    assert "content_digest" in reason
    assert (
        left["fingerprint"]["content_digest"]
        != right["fingerprint"]["content_digest"]
    )


def test_classical_model_f_reports_infinite_statistic_for_exact_fit():
    x = np.linspace(-1.0, 1.0, 8)
    X = np.column_stack([np.ones(x.size), x])
    params = np.asarray([1.25, -0.8])
    y = X @ params

    statistic, pvalue, df, metadata = _classical_model_f(
        y,
        X,
        params,
        xp=np,
        df_resid=x.size - np.linalg.matrix_rank(X),
        has_constant=True,
    )

    assert np.isinf(statistic)
    assert pvalue == 0.0
    assert df == (1.0, 6.0)
    assert metadata["exact_fit"] is True
    assert metadata["rss_restricted"] > 0.0
    assert metadata["rss_unrestricted"] <= metadata["rss_restricted"]


def test_classical_model_f_is_invariant_to_response_units():
    x = np.linspace(-1.0, 1.0, 12)
    X = np.column_stack([np.ones(x.size), x])
    y = 0.8 + 0.45 * x + np.asarray(
        [0.08, -0.04, 0.03, -0.06, 0.05, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.05]
    )
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    df_resid = x.size - np.linalg.matrix_rank(X)

    reference = _classical_model_f(
        y,
        X,
        params,
        xp=np,
        df_resid=df_resid,
        has_constant=True,
    )
    assert reference[0] is not None and np.isfinite(reference[0])

    for scale in (1e-8, 1e-12):
        candidate = _classical_model_f(
            scale * y,
            X,
            scale * params,
            xp=np,
            df_resid=df_resid,
            has_constant=True,
        )
        assert candidate[0] is not None and np.isfinite(candidate[0])
        assert_allclose(candidate[0], reference[0], rtol=1e-10, atol=1e-12)
        assert_allclose(candidate[1], reference[1], rtol=1e-10, atol=1e-14)
        assert candidate[2] == reference[2]


def test_hausman_quadratic_is_invariant_to_parameter_units():
    difference = np.asarray([0.2, -0.1])
    covariance_difference = np.asarray([[0.04, 0.01], [0.01, 0.09]])
    reference = _hausman_quadratic(difference, covariance_difference)
    assert reference.applicable

    for scale in (1e-8, 1e-12):
        candidate = _hausman_quadratic(
            scale * difference,
            (scale * scale) * covariance_difference,
        )
        assert candidate.applicable
        assert candidate.df == reference.df
        assert_allclose(candidate.statistic, reference.statistic, rtol=1e-10, atol=1e-12)
        assert_allclose(candidate.pvalue, reference.pvalue, rtol=1e-10, atol=1e-14)


def test_explicit_constant_detection_is_invariant_to_column_units():
    slope = np.linspace(-1.0, 1.0, 9)
    for scale in (1.0, 1e-8, 1e-12):
        X = np.column_stack([np.full(slope.size, scale), slope])
        assert explicit_constant_column(X, xp=np) == 0

    zero_column = np.column_stack([np.zeros(slope.size), slope])
    assert explicit_constant_column(zero_column, xp=np) is None


def test_pooling_f_reports_infinite_statistic_for_exact_effect_fit():
    result = _pooling_f_from_sums(
        rss_pooled=4.0,
        rss_effects=0.0,
        df_num=3,
        df_denom=10,
    )

    assert result.applicable
    assert np.isinf(result.statistic)
    assert result.pvalue == 0.0
    assert result.df == (3.0, 10.0)
    assert result.metadata["exact_fit"] is True


def test_pooling_f_both_exact_models_remains_inapplicable():
    result = _pooling_f_from_sums(
        rss_pooled=0.0,
        rss_effects=0.0,
        df_num=3,
        df_denom=10,
    )

    assert not result.applicable
    assert result.statistic is None
    assert result.pvalue is None
    assert "both zero" in result.reason


def test_random_effects_explicit_constant_uses_constant_aware_diagnostics():
    rng = np.random.default_rng(20260808)
    counts = np.asarray([5, 4, 3, 5, 4, 3])
    entity = np.repeat(np.arange(counts.size), counts)
    slope = rng.normal(size=entity.size)
    X = np.column_stack([np.ones(entity.size), slope])
    alpha = np.repeat(np.linspace(-0.4, 0.5, counts.size), counts)
    y = 1.7 + 0.65 * slope + alpha + rng.normal(scale=0.12, size=entity.size)

    model = RandomEffects().fit(X, y, entity_ids=entity)
    result = model.fit_statistics_

    assert result.metadata["has_explicit_constant"] is True
    assert result.metadata["constant_column_index"] == 0
    assert result.metadata["restricted_rank"] == 1
    assert result.metadata["model_f"]["rank_restricted"] == 1
    assert result.metadata["model_f"]["restricted_design_supplied"] is True
    assert result.f_df == (1.0, float(entity.size - np.linalg.matrix_rank(X)))

    level_resid = y - X @ model.coef_
    expected_overall = 1.0 - np.dot(level_resid, level_resid) / np.sum(
        (y - y.mean()) ** 2
    )
    assert_allclose(result.rsquared_overall, expected_overall, rtol=1e-12, atol=1e-12)


def test_random_effects_without_constant_preserves_uncentered_diagnostic_basis():
    rng = np.random.default_rng(20260809)
    entity = np.repeat(np.arange(6), 4)
    X = rng.normal(size=(entity.size, 2))
    y = X @ np.asarray([0.7, -0.3]) + np.repeat(
        np.linspace(-0.3, 0.4, 6), 4
    ) + rng.normal(scale=0.15, size=entity.size)

    model = RandomEffects().fit(X, y, entity_ids=entity)
    result = model.fit_statistics_

    assert result.metadata["has_explicit_constant"] is False
    assert result.metadata["constant_column_index"] is None
    assert result.metadata["restricted_rank"] == 0
    assert result.metadata["model_f"].get("restricted_design_supplied") is None
    assert result.metadata["model_f"]["rank_restricted"] == 0
