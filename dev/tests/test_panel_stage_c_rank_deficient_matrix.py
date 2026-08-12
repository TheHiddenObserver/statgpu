"""Cross-backend estimator coverage for the Stage-C rank-deficient fit policy."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


torch = pytest.importorskip("torch")


def _torch_arrays(X, y, entity, time):
    return (
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(entity, dtype=torch.int64),
        torch.as_tensor(time, dtype=torch.int64),
    )


def _assert_inference(actual, expected, *, rtol=2e-7, atol=2e-9):
    assert_allclose(actual.coef_, expected.coef_, rtol=rtol, atol=atol)
    assert_allclose(actual.bse_, expected.bse_, rtol=rtol, atol=atol)
    assert_allclose(actual.tvalues_, expected.tvalues_, rtol=rtol, atol=atol)
    assert_allclose(actual.pvalues_, expected.pvalues_, rtol=rtol, atol=atol)
    assert_allclose(actual.conf_int_, expected.conf_int_, rtol=rtol, atol=atol)
    assert_allclose(
        actual._panel_cov_params_raw,
        expected._panel_cov_params_raw,
        rtol=rtol,
        atol=atol,
    )


def test_exact_rank_deficient_estimator_matrix_torch_cpu_matches_numpy():
    """All Stage-C residual-OLS families share the same rank-deficient subspace."""
    rng = np.random.default_rng(12912)
    n_entities, n_times = 15, 4
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    x = rng.normal(size=entity.size)
    X = np.column_stack([x, 2.0 * x])
    alpha = np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)
    y = 0.6 * x + alpha + rng.normal(scale=0.2, size=entity.size)

    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    model_pairs = [
        (
            PooledOLS(cov_type="hc0").fit(X, y, entity_ids=entity),
            PooledOLS(cov_type="hc0").fit(X_t, y_t, entity_ids=entity_t),
        ),
        (
            PanelOLS(cov_type="hc0").fit(X, y),
            PanelOLS(cov_type="hc0").fit(X_t, y_t),
        ),
        (
            BetweenOLS(cov_type="hc0").fit(X, y, entity_ids=entity),
            BetweenOLS(cov_type="hc0").fit(X_t, y_t, entity_ids=entity_t),
        ),
        (
            FirstDifferenceOLS(cov_type="hc0").fit(
                X, y, entity_ids=entity, time_ids=time
            ),
            FirstDifferenceOLS(cov_type="hc0").fit(
                X_t, y_t, entity_ids=entity_t, time_ids=time_t
            ),
        ),
    ]

    X_re = np.column_stack([np.ones(entity.size), X])
    X_re_t = torch.as_tensor(X_re, dtype=torch.float64)
    model_pairs.append(
        (
            RandomEffects(cov_type="hc0").fit(X_re, y, entity_ids=entity),
            RandomEffects(cov_type="hc0").fit(
                X_re_t, y_t, entity_ids=entity_t
            ),
        )
    )

    for expected, actual in model_pairs:
        _assert_inference(actual, expected)
        expected_meta = expected._covariance_metadata
        actual_meta = actual._covariance_metadata
        assert expected_meta["design_rank"] < expected_meta["design_columns"]
        assert actual_meta["design_rank"] == expected_meta["design_rank"]
        assert actual_meta["design_columns"] == expected_meta["design_columns"]


@pytest.mark.parametrize("cov_type", ["nonrobust", "robust"])
def test_rank_deficient_df_and_identified_inference_are_column_space_invariant(cov_type):
    """Redundant columns cannot change identified fit/inference in rank extensions."""
    rng = np.random.default_rng(12913)
    n_entities, n_times = 18, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    keep = np.ones(entity.size, dtype=bool)
    keep[[1, 8, 17, 29, 44, 63, 78]] = False
    entity = entity[keep]
    time = time[keep]
    x = rng.normal(size=entity.size)
    alpha = np.repeat(rng.normal(scale=0.35, size=n_entities), n_times)[keep]
    y = 0.45 + 0.75 * x + alpha + rng.normal(scale=0.2, size=entity.size)

    X1 = x[:, None]
    X2 = np.column_stack([x, 2.0 * x])
    pairs = [
        (PanelOLS(cov_type=cov_type).fit(X1, y),
         PanelOLS(cov_type=cov_type).fit(X2, y), X1, X2),
        (BetweenOLS(cov_type=cov_type).fit(X1, y, entity_ids=entity),
         BetweenOLS(cov_type=cov_type).fit(X2, y, entity_ids=entity),
         np.column_stack([np.ones(len(y)), X1]),
         np.column_stack([np.ones(len(y)), X2])),
        (FirstDifferenceOLS(cov_type=cov_type).fit(X1, y, entity_ids=entity, time_ids=time),
         FirstDifferenceOLS(cov_type=cov_type).fit(X2, y, entity_ids=entity, time_ids=time),
         X1, X2),
    ]

    X1_re = np.column_stack([np.ones(len(y)), X1])
    X2_re = np.column_stack([np.ones(len(y)), X2])
    re_base = RandomEffects(cov_type=cov_type).fit(X1_re, y, entity_ids=entity)
    re_redundant = RandomEffects(cov_type=cov_type).fit(X2_re, y, entity_ids=entity)
    pairs.append((re_base, re_redundant, X1_re, X2_re))

    for base, redundant, design_base, design_redundant in pairs:
        assert base.df_resid == redundant.df_resid
        assert_allclose(
            design_redundant @ np.asarray(redundant.coef_),
            design_base @ np.asarray(base.coef_),
            rtol=2e-10, atol=2e-11,
        )
        cov_base = np.asarray(base._panel_cov_params_raw)
        cov_redundant = np.asarray(redundant._panel_cov_params_raw)
        assert_allclose(
            design_redundant @ cov_redundant @ design_redundant.T,
            design_base @ cov_base @ design_base.T,
            rtol=2e-8, atol=2e-10,
        )

    assert_allclose(re_redundant.variance_components_["sigma2_e"],
                    re_base.variance_components_["sigma2_e"], rtol=2e-11, atol=2e-13)
    assert_allclose(re_redundant.variance_components_["sigma2_a"],
                    re_base.variance_components_["sigma2_a"], rtol=2e-11, atol=2e-13)
    assert_allclose(re_redundant.theta_, re_base.theta_, rtol=2e-11, atol=2e-13)
