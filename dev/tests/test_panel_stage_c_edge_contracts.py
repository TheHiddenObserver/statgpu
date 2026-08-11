"""Edge contracts for public Panel Stage-C covariance primitives."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import PooledOLS, driscoll_kraay_covariance
from statgpu.panel._covariance import clustered_covariance


def _fit_space(seed=12950, n_times=6):
    rng = np.random.default_rng(seed)
    time = np.tile(np.arange(n_times), 5)
    X = np.column_stack([np.ones(time.size), rng.normal(size=time.size)])
    resid = rng.normal(size=time.size)
    return X, resid, time


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0.0, 1.0, np.nan, 2.0]),
        np.array([0.0, 1.0, np.inf, 2.0]),
        np.array(["a", "b", None, "c"], dtype=object),
        np.array([np.datetime64("2026-01-01"), np.datetime64("NaT")]),
    ],
)
def test_group_label_factorization_rejects_missing_or_nonfinite_values(labels):
    n = len(labels)
    X = np.column_stack([np.ones(n), np.arange(float(n))])
    resid = np.linspace(-0.2, 0.3, n)
    with pytest.raises(ValueError, match="must not contain missing or non-finite values"):
        clustered_covariance(X, resid, labels)


def test_public_driscoll_kraay_rejects_missing_time_labels():
    X, resid, time = _fit_space()
    time = time.astype(float)
    time[7] = np.nan
    with pytest.raises(ValueError, match="must not contain missing or non-finite values"):
        driscoll_kraay_covariance(X, resid, time, bandwidth=2)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen"])
def test_truncated_kernels_do_not_silently_cap_oversized_bandwidth(kernel):
    X, resid, time = _fit_space(n_times=5)
    meta = {}
    cov_large = driscoll_kraay_covariance(
        X, resid, time, bandwidth=9, kernel=kernel, metadata=meta
    )
    cov_capped = driscoll_kraay_covariance(
        X, resid, time, bandwidth=4, kernel=kernel
    )
    assert meta["bandwidth"] == 9
    assert meta["max_weighted_lag"] == 4
    # The requested bandwidth remains in the Bartlett/Parzen weight formula;
    # silently replacing 9 by T-1=4 would make these matrices identical.
    assert not np.allclose(cov_large, cov_capped, rtol=1e-12, atol=1e-14)


def test_qs_oversized_bandwidth_remains_a_smoothing_scale():
    X, resid, time = _fit_space(n_times=5)
    meta = {}
    cov = driscoll_kraay_covariance(
        X, resid, time, bandwidth=9, kernel="qs", metadata=meta
    )
    assert np.all(np.isfinite(cov))
    assert meta["bandwidth"] == 9
    assert meta["all_observed_lags_weighted"] is True
    assert meta["max_weighted_lag"] == 4


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_cluster_primitives_reject_nonboolean_group_debias(value):
    X = np.column_stack([np.ones(8), np.arange(8.0)])
    resid = np.linspace(-0.2, 0.3, 8)
    groups = np.repeat(np.arange(4), 2)
    with pytest.raises(ValueError, match="group_debias must be boolean"):
        clustered_covariance(X, resid, groups, group_debias=value)



def test_dk_preserves_ordered_categorical_chronology():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(12951)
    labels = np.tile(np.array(["t1", "t2", "t10"], dtype=object), 9)
    numeric = np.tile(np.arange(3), 9)
    ordered = pd.Categorical(
        labels,
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    X = np.column_stack([np.ones(labels.size), rng.normal(size=labels.size)])
    resid = rng.normal(size=labels.size)

    actual = driscoll_kraay_covariance(
        X, resid, ordered, bandwidth=1, kernel="bartlett"
    )
    expected = driscoll_kraay_covariance(
        X, resid, numeric, bandwidth=1, kernel="bartlett"
    )
    lexical = driscoll_kraay_covariance(
        X, resid, np.asarray(ordered, dtype=object), bandwidth=1, kernel="bartlett"
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)
    assert not np.allclose(actual, lexical, rtol=1e-10, atol=1e-12)


def test_dk_ordered_categorical_rejects_missing_codes():
    pd = pytest.importorskip("pandas")
    labels = pd.Categorical(
        ["t1", "t2", None, "t10"],
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    X = np.column_stack([np.ones(4), np.arange(4.0)])
    resid = np.linspace(-0.2, 0.3, 4)
    with pytest.raises(ValueError, match="must not contain missing or non-finite values"):
        driscoll_kraay_covariance(X, resid, labels, bandwidth=1)


def test_pooled_formula_dk_preserves_ordered_categorical_after_row_alignment():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(12952)
    labels = np.tile(np.array(["t1", "t2", "t10"], dtype=object), 12)
    numeric = np.tile(np.arange(3), 12)
    ordered = pd.Categorical(
        labels,
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    x = rng.normal(size=labels.size)
    y = 0.4 + 0.7 * x + rng.normal(scale=0.2, size=labels.size)
    x_with_gap = x.copy()
    x_with_gap[5] = np.nan
    data = pd.DataFrame({"y": y, "x": x_with_gap})

    categorical_fit = PooledOLS(cov_type="dk", bandwidth=1).fit(
        formula="y ~ x", data=data, time_index=ordered
    )
    numeric_fit = PooledOLS(cov_type="dk", bandwidth=1).fit(
        formula="y ~ x", data=data, time_index=numeric
    )

    np.testing.assert_allclose(
        categorical_fit.coef_, numeric_fit.coef_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        categorical_fit.bse_, numeric_fit.bse_, rtol=2e-13, atol=2e-15
    )
    assert categorical_fit._covariance_metadata["n_periods"] == 3


def test_first_difference_preserves_ordered_categorical_chronology():
    pd = pytest.importorskip("pandas")
    from statgpu.panel import FirstDifferenceOLS

    rng = np.random.default_rng(12953)
    n_entities = 12
    numeric_time = np.tile(np.arange(3), n_entities)
    labels = np.tile(np.array(["t1", "t2", "t10"], dtype=object), n_entities)
    ordered_time = pd.Categorical(
        labels, categories=["t1", "t2", "t10"], ordered=True
    )
    entity = np.repeat(np.arange(n_entities), 3)
    X = rng.normal(size=(entity.size, 1))
    y = (
        0.8 * X[:, 0]
        + np.tile(np.array([0.0, 0.35, -0.2]), n_entities)
        + rng.normal(scale=0.15, size=entity.size)
    )

    numeric = FirstDifferenceOLS(cov_type="hc0").fit(
        X, y, entity_ids=entity, time_ids=numeric_time
    )
    categorical = FirstDifferenceOLS(cov_type="hc0").fit(
        X, y, entity_ids=entity, time_ids=ordered_time
    )
    np.testing.assert_allclose(categorical.coef_, numeric.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(categorical.bse_, numeric.bse_, rtol=2e-13, atol=2e-15)


def test_panel_rank_boundary_fit_matches_explicit_svd_policy():
    from statgpu.panel import PanelOLS

    rng = np.random.default_rng(12954)
    n, k = 100, 3
    q_left, _ = np.linalg.qr(rng.normal(size=(n, k)))
    q_right, _ = np.linalg.qr(rng.normal(size=(k, k)))
    X = q_left @ np.diag([10.0, 1.0, 1.0e-13]) @ q_right.T
    y = rng.normal(size=n)
    time = np.tile(np.arange(10), 10)
    U, singular_values, Vh = np.linalg.svd(X, full_matrices=False)
    cutoff = max(X.shape) * np.finfo(np.float64).eps * singular_values[0]
    retained = singular_values > cutoff
    assert int(np.sum(retained)) == 2
    inverse_values = np.zeros_like(singular_values)
    inverse_values[retained] = 1.0 / singular_values[retained]
    expected = ((Vh.T * inverse_values) @ U.T) @ y

    model = PanelOLS(cov_type="dk", bandwidth=2).fit(X, y, time_ids=time)
    np.testing.assert_allclose(model.coef_, expected, rtol=5e-11, atol=5e-13)
    assert model._covariance_metadata["design_rank"] == 2
    assert model.fit_statistics_.metadata["diagnostic_df"]["rank_x"] == 2


def test_panel_full_rank_fit_preserves_historical_solver_path(monkeypatch):
    import statgpu.panel._fixed_effects as fixed_effects_module
    from statgpu.panel import PanelOLS

    rng = np.random.default_rng(12955)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([0.4, -0.2, 0.7]) + rng.normal(scale=0.1, size=80)
    expected = np.linalg.solve(X.T @ X, X.T @ y)

    def _forbid_rank_deficient_solve(*args, **kwargs):
        raise AssertionError("full-rank PanelOLS entered the SVD rank-deficient solve")

    monkeypatch.setattr(fixed_effects_module, "panel_lstsq", _forbid_rank_deficient_solve)
    model = PanelOLS().fit(X, y)
    np.testing.assert_allclose(model.coef_, expected, rtol=2e-12, atol=2e-14)


def test_full_rank_pooling_diagnostic_preserves_historical_pinv_path(monkeypatch):
    import statgpu.panel._diagnostic_context as diagnostic_context

    rng = np.random.default_rng(12956)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([0.3, -0.5]) + rng.normal(scale=0.2, size=60)

    def _forbid_rank_deficient_solve(*args, **kwargs):
        raise AssertionError("full-rank pooling diagnostic entered SVD deficiency solve")

    monkeypatch.setattr(diagnostic_context, "panel_lstsq", _forbid_rank_deficient_solve)
    result = diagnostic_context.pooling_f_from_level_arrays(
        y, X, xp=np, rss_effects=float(np.sum(y * y)) + 1.0,
        df_resid_effects=50, has_constant=True,
    )
    assert result is not None
