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
