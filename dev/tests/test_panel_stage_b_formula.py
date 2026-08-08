"""Formula/missing-row contracts for Panel Tier-1 Stage B diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import statgpu
from statgpu.panel import FamaMacBeth, PanelOLS, PooledOLS


def _frame(seed=1230):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 8, 4
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    x1 = rng.normal(size=entity.size)
    x2 = rng.normal(size=entity.size)
    alpha = np.repeat(np.linspace(-0.4, 0.5, n_entities), n_times)
    y = 0.7 + 0.85 * x1 - 0.3 * x2 + alpha + rng.normal(
        scale=0.15, size=entity.size
    )
    frame = pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "entity": entity, "time": time}
    )
    # Patsy will drop these rows. Keep at least three observations in every
    # affected period so Fama-MacBeth remains estimable after filtering.
    frame.loc[[1, 10], "x1"] = np.nan
    return frame


def _assert_fit_statistics(actual, expected):
    for field in (
        "rsquared_within",
        "rsquared_between",
        "rsquared_overall",
        "rsquared_adj",
        "f_statistic",
        "f_pvalue",
    ):
        a = getattr(actual, field)
        e = getattr(expected, field)
        if e is None:
            assert a is None
        else:
            assert_allclose(a, e, rtol=1e-9, atol=1e-11)
    assert actual.f_df == expected.f_df


def test_pooled_formula_aligns_entity_ids_before_bp_and_r2():
    frame = _frame()
    entity_full = frame["entity"].to_numpy()
    fitted = PooledOLS().fit(
        formula="y ~ x1 + x2",
        data=frame,
        entity_ids=entity_full,
    )

    retained = frame.dropna(subset=["x1", "x2", "y"])
    reference = PooledOLS().fit(
        retained[["x1", "x2"]].to_numpy(),
        retained["y"].to_numpy(),
        entity_ids=retained["entity"].to_numpy(),
    )
    assert_allclose(fitted.coef_, reference.coef_, rtol=1e-10, atol=1e-11)
    _assert_fit_statistics(fitted.fit_statistics_, reference.fit_statistics_)
    assert_allclose(
        fitted.breusch_pagan_lm_test().statistic,
        reference.breusch_pagan_lm_test().statistic,
        rtol=1e-9,
        atol=1e-11,
    )


def test_panelols_formula_effect_tokens_use_exact_retained_sample_for_pooling_f():
    frame = _frame(seed=1231)
    fitted = PanelOLS().fit(
        formula="y ~ x1 + x2 | entity",
        data=frame,
    )
    retained = frame.dropna(subset=["x1", "x2", "y"])
    reference = PanelOLS(entity_effects=True).fit(
        retained[["x1", "x2"]].to_numpy(),
        retained["y"].to_numpy(),
        entity_ids=retained["entity"].to_numpy(),
    )
    assert fitted.entity_effects is True
    assert_allclose(fitted.coef_, reference.coef_, rtol=1e-10, atol=1e-11)
    _assert_fit_statistics(fitted.fit_statistics_, reference.fit_statistics_)
    assert_allclose(
        fitted.pooling_f_test().statistic,
        reference.pooling_f_test().statistic,
        rtol=1e-9,
        atol=1e-11,
    )
    assert fitted.pooling_f_test().df == reference.pooling_f_test().df


def test_fama_macbeth_formula_aligns_optional_entity_ids_for_r2_only():
    frame = _frame(seed=1232)
    time_full = frame["time"].to_numpy()
    entity_full = frame["entity"].to_numpy()
    fitted = FamaMacBeth(cov_type="newey-west", bandwidth=1).fit(
        formula="y ~ x1 + x2",
        data=frame,
        time_ids=time_full,
        entity_ids=entity_full,
    )
    retained = frame.dropna(subset=["x1", "x2", "y"])
    reference = FamaMacBeth(cov_type="newey-west", bandwidth=1).fit(
        retained[["x1", "x2"]].to_numpy(),
        retained["y"].to_numpy(),
        time_ids=retained["time"].to_numpy(),
        entity_ids=retained["entity"].to_numpy(),
    )
    assert_allclose(
        np.asarray(fitted.coef_),
        np.asarray(reference.coef_),
        rtol=1e-10,
        atol=1e-11,
    )
    assert_allclose(
        fitted.fit_statistics_.rsquared_within,
        reference.fit_statistics_.rsquared_within,
        rtol=1e-9,
        atol=1e-11,
    )
    assert_allclose(
        fitted.fit_statistics_.rsquared_between,
        reference.fit_statistics_.rsquared_between,
        rtol=1e-9,
        atol=1e-11,
    )
    assert_allclose(
        fitted.fit_statistics_.rsquared_overall,
        reference.fit_statistics_.rsquared_overall,
        rtol=1e-9,
        atol=1e-11,
    )
    assert fitted.fit_statistics_.rsquared_adj is None
    assert fitted.fit_statistics_.f_statistic is None


def test_stage_b_diagnostic_api_is_available_from_panel_and_top_level():
    from statgpu.panel import (
        PanelFitStatistics,
        PanelTestResult,
        breusch_pagan_lm_test,
        hausman_test,
        pooling_f_test,
    )

    assert statgpu.PanelFitStatistics is PanelFitStatistics
    assert statgpu.PanelTestResult is PanelTestResult
    assert statgpu.hausman_test is hausman_test
    assert statgpu.pooling_f_test is pooling_f_test
    assert statgpu.breusch_pagan_lm_test is breusch_pagan_lm_test
