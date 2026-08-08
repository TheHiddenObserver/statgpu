"""Analytic contracts for Panel Tier-1 Stage B diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from statgpu.panel._diagnostics import (
    _bp_lm_from_components,
    _build_fit_statistics,
    _diagnostic_identity,
    _fingerprints_match,
    _hausman_quadratic,
    _pooling_f_from_sums,
)
from statgpu.panel._results import PanelFitStatistics, PanelTestResult


def test_pooling_f_matches_hand_formula():
    result = _pooling_f_from_sums(
        rss_pooled=18.0,
        rss_effects=12.0,
        df_num=3,
        df_denom=40,
        metadata={"constant_correction": False},
    )
    expected = ((18.0 - 12.0) / 3.0) / (12.0 / 40.0)
    assert isinstance(result, PanelTestResult)
    assert result.applicable
    assert_allclose(result.statistic, expected, rtol=0, atol=1e-14)
    assert_allclose(result.pvalue, stats.f.sf(expected, 3, 40), rtol=1e-12)
    assert result.df == (3.0, 40.0)
    assert result.metadata["classical_homoskedastic"] is True


def test_pooling_f_roundoff_negative_is_normalized_but_material_violation_is_not():
    tiny = _pooling_f_from_sums(
        rss_pooled=10.0 - 1e-14,
        rss_effects=10.0,
        df_num=2,
        df_denom=20,
    )
    assert tiny.applicable
    assert tiny.statistic == 0.0
    assert tiny.metadata["roundoff_normalized"] is True

    bad = _pooling_f_from_sums(
        rss_pooled=9.0,
        rss_effects=10.0,
        df_num=2,
        df_denom=20,
    )
    assert not bad.applicable
    assert "nested-model contract failed" in bad.reason


def test_bp_lm_matches_baltagi_li_unbalanced_formula():
    counts = np.asarray([2.0, 3.0, 4.0])
    group_sums = np.asarray([0.6, -0.2, 0.9])
    cp = 5.5
    nobs = int(counts.sum())
    a1 = np.sum(group_sums ** 2) / cp - 1.0
    m11 = np.sum(counts ** 2)
    lm1 = nobs * np.sqrt(1.0 / (2.0 * (m11 - nobs))) * a1
    expected = lm1 ** 2

    result = _bp_lm_from_components(
        nobs=nobs,
        residual_ss=cp,
        group_residual_sums=group_sums,
        group_counts=counts,
    )
    assert result.applicable
    assert result.df == 1.0
    assert_allclose(result.statistic, expected, rtol=0, atol=1e-14)
    assert_allclose(result.pvalue, stats.chi2.sf(expected, 1), rtol=1e-12)
    assert result.metadata["definition"].startswith("Baltagi-Li")


def test_bp_lm_rejects_singletons_only_and_zero_rss():
    singletons = _bp_lm_from_components(
        nobs=3,
        residual_ss=2.0,
        group_residual_sums=[0.1, -0.1, 0.0],
        group_counts=[1.0, 1.0, 1.0],
    )
    assert not singletons.applicable
    assert "repeated observations" in singletons.reason

    zero = _bp_lm_from_components(
        nobs=4,
        residual_ss=0.0,
        group_residual_sums=[0.0, 0.0],
        group_counts=[2.0, 2.0],
    )
    assert not zero.applicable
    assert "must be positive" in zero.reason


def test_hausman_full_rank_matches_quadratic_form():
    d = np.asarray([0.2, -0.1])
    D = np.asarray([[0.08, 0.01], [0.01, 0.05]])
    expected = float(d @ np.linalg.solve(D, d))
    result = _hausman_quadratic(d, D)
    assert result.applicable
    assert result.df == 2.0
    assert result.metadata["used_pinv"] is False
    assert_allclose(result.statistic, expected, rtol=1e-12)
    assert_allclose(result.pvalue, stats.chi2.sf(expected, 2), rtol=1e-12)


def test_hausman_singular_psd_uses_identified_range_and_rank_df():
    D = np.asarray([[2.0, 0.0], [0.0, 0.0]])
    result = _hausman_quadratic([1.0, 0.0], D)
    assert result.applicable
    assert result.df == 1.0
    assert result.metadata["used_pinv"] is True
    assert result.metadata["definition_extension"].startswith("singular PSD")
    assert_allclose(result.statistic, 0.5, rtol=0, atol=1e-14)

    outside = _hausman_quadratic([1.0, 0.1], D)
    assert not outside.applicable
    assert "outside the identified" in outside.reason


def test_hausman_indefinite_covariance_difference_is_explicitly_inapplicable():
    result = _hausman_quadratic([0.1, 0.2], [[1.0, 0.0], [0.0, -0.1]])
    assert not result.applicable
    assert "not positive semidefinite" in result.reason


def test_parameter_based_r2_and_adjusted_r2_are_hand_checkable():
    entity = np.repeat(np.arange(3), 3)
    X = np.column_stack(
        [
            np.ones(9),
            np.linspace(-1.0, 1.0, 9),
        ]
    )
    beta = np.asarray([1.2, 0.7])
    entity_shift = np.repeat(np.asarray([-0.4, 0.0, 0.5]), 3)
    noise = np.asarray([0.1, -0.1, 0.0] * 3)
    y = X @ beta + entity_shift + noise
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(np.sum((y - y.mean()) ** 2))

    result = _build_fit_statistics(
        y,
        X,
        beta,
        xp=np,
        entity_codes=entity,
        has_constant=True,
        rss_fit=rss,
        tss_fit=tss,
        df_resid=7,
        df_total=8,
        metadata={"fit_space": "pooled"},
    )
    assert isinstance(result, PanelFitStatistics)
    expected_overall = 1.0 - rss / tss
    expected_adj = 1.0 - (rss / 7.0) / (tss / 8.0)
    assert_allclose(result.rsquared_overall, expected_overall, rtol=1e-12)
    assert_allclose(result.rsquared_adj, expected_adj, rtol=1e-12)
    assert result.rsquared_within is not None
    assert result.rsquared_between is not None
    assert result.f_statistic is not None
    assert result.f_df == (1.0, 7.0)


def test_standardized_r2_zero_tss_is_zero_with_metadata():
    X = np.ones((6, 1))
    y = np.ones(6)
    result = _build_fit_statistics(
        y,
        X,
        np.asarray([1.0]),
        xp=np,
        entity_codes=np.repeat(np.arange(3), 2),
        has_constant=True,
        rss_fit=0.0,
        tss_fit=0.0,
        df_resid=5,
        df_total=5,
    )
    assert result.rsquared_overall == 0.0
    assert result.rsquared_between == 0.0
    assert result.rsquared_within == 0.0
    assert result.rsquared_adj == 0.0
    assert result.metadata["degenerate_total_ss"] == {
        "within": True,
        "between": True,
        "overall": True,
    }


def test_numerical_fingerprint_detects_row_order_and_data_changes():
    X = np.arange(12.0).reshape(6, 2) / 10.0
    y = np.linspace(-0.5, 0.5, 6)
    entity = np.asarray([0, 0, 1, 1, 2, 2])
    left = _diagnostic_identity(
        X,
        y,
        xp=np,
        entity_codes=entity,
        feature_names=["x1", "x2"],
    )
    same = _diagnostic_identity(
        X.copy(),
        y.copy(),
        xp=np,
        entity_codes=entity.copy(),
        feature_names=["x1", "x2"],
    )
    ok, reason = _fingerprints_match(left, same)
    assert ok
    assert reason == ""

    changed_y = y.copy()
    changed_y[-1] += 0.2
    different = _diagnostic_identity(
        X,
        changed_y,
        xp=np,
        entity_codes=entity,
        feature_names=["x1", "x2"],
    )
    ok, reason = _fingerprints_match(left, different)
    assert not ok
    assert "fingerprint mismatch" in reason

    order = np.asarray([1, 0, 2, 3, 4, 5])
    reordered = _diagnostic_identity(
        X[order],
        y[order],
        xp=np,
        entity_codes=entity[order],
        feature_names=["x1", "x2"],
    )
    ok, reason = _fingerprints_match(left, reordered)
    assert not ok
    assert "identity mismatch" in reason or "fingerprint mismatch" in reason
