"""Regression tests for covariance and panel module review findings."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from statgpu.covariance import (
    EmpiricalCovariance,
    GraphicalLasso,
    GraphicalLassoCV,
    MinCovDet,
)
from statgpu.panel import BetweenOLS, FamaMacBeth, FirstDifferenceOLS, PanelOLS, PooledOLS
from statgpu.panel._covariance import clustered_covariance, hac_covariance


def test_empirical_precision_is_inverse_without_unnecessary_jitter():
    rng = np.random.RandomState(100)
    X = rng.normal(size=(120, 5)) @ np.diag([1.0, 1.3, 0.8, 2.0, 0.7])
    model = EmpiricalCovariance().fit(X)
    assert_allclose(
        np.asarray(model.covariance_) @ np.asarray(model.precision_),
        np.eye(5),
        rtol=1e-12,
        atol=1e-12,
    )


def test_empirical_covariance_validates_feature_count():
    rng = np.random.RandomState(101)
    model = EmpiricalCovariance().fit(rng.normal(size=(30, 4)))
    with pytest.raises(ValueError, match="features"):
        model.score(rng.normal(size=(10, 3)))
    with pytest.raises(ValueError, match="features"):
        model.mahalanobis(rng.normal(size=(10, 3)))


def test_graphical_lasso_matches_sklearn_and_preserves_covariance_diagonal():
    from sklearn.covariance import GraphicalLasso as SkGraphicalLasso

    rng = np.random.RandomState(102)
    A = rng.normal(size=(6, 6))
    cov = A @ A.T + np.eye(6)
    X = rng.multivariate_normal(np.zeros(6), cov, size=350)

    alpha = 0.08
    actual = GraphicalLasso(alpha=alpha, max_iter=250, tol=1e-7).fit(X)
    expected = SkGraphicalLasso(alpha=alpha, max_iter=250, tol=1e-7).fit(X)

    assert_allclose(actual.covariance_, expected.covariance_, rtol=3e-3, atol=3e-3)
    assert_allclose(actual.precision_, expected.precision_, rtol=5e-3, atol=5e-3)
    empirical_diag = np.diag(np.cov(X, rowvar=False, bias=True))
    assert_allclose(np.diag(actual.covariance_), empirical_diag, rtol=1e-12, atol=1e-12)
    assert_allclose(
        np.asarray(actual.covariance_) @ np.asarray(actual.precision_),
        np.eye(X.shape[1]),
        rtol=1e-8,
        atol=1e-8,
    )


def test_graphical_lasso_input_contracts():
    X = np.arange(30.0).reshape(10, 3)
    with pytest.raises(ValueError, match="alpha"):
        GraphicalLasso(alpha=-0.1).fit(X)
    with pytest.raises(ValueError, match="max_iter"):
        GraphicalLasso(max_iter=0).fit(X)
    with pytest.raises(ValueError, match="tol"):
        GraphicalLasso(tol=0).fit(X)


def test_graphical_lasso_cv_validates_cv_and_alphas():
    X = np.arange(60.0).reshape(20, 3)
    with pytest.raises(ValueError, match="cv"):
        GraphicalLassoCV(cv=1).fit(X)
    with pytest.raises(ValueError, match="cv"):
        GraphicalLassoCV(cv=21).fit(X)
    with pytest.raises(ValueError, match="alphas"):
        GraphicalLassoCV(alphas=[]).fit(X)
    with pytest.raises(ValueError, match="alphas"):
        GraphicalLassoCV(alphas=[-0.1, 0.1]).fit(X)


def test_min_cov_det_validates_fraction_and_honors_assume_centered():
    rng = np.random.RandomState(103)
    X = rng.normal(loc=4.0, scale=1.0, size=(90, 3))
    with pytest.raises(ValueError, match="support_fraction"):
        MinCovDet(support_fraction=0).fit(X)
    with pytest.raises(ValueError, match="support_fraction"):
        MinCovDet(support_fraction=1.1).fit(X)

    centered_model = MinCovDet(assume_centered=True, random_state=0).fit(X)
    assert_allclose(centered_model.location_, np.zeros(3), atol=0, rtol=0)
    assert_allclose(centered_model.raw_location_, np.zeros(3), atol=0, rtol=0)


def test_clustered_covariance_string_labels_match_integer_labels():
    rng = np.random.RandomState(104)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    resid = rng.normal(size=40)
    codes = np.repeat(np.arange(8), 5)
    labels = np.asarray([f"firm-{v}" for v in codes], dtype=object)
    cov_codes = clustered_covariance(X, resid, codes, xp=np)
    cov_labels = clustered_covariance(X, resid, labels, xp=np)
    assert_allclose(cov_codes, cov_labels, rtol=1e-12, atol=1e-12)


def test_panel_hac_validates_kernel_and_bandwidth():
    X = np.column_stack([np.ones(12), np.arange(12.0)])
    resid = np.linspace(-1.0, 1.0, 12)
    with pytest.raises(ValueError, match="kernel"):
        hac_covariance(X, resid, kernel="uniform", xp=np)
    with pytest.raises(ValueError, match="bandwidth"):
        hac_covariance(X, resid, bandwidth=-1, xp=np)
    with pytest.raises(ValueError, match="bandwidth"):
        hac_covariance(X, resid, bandwidth=1.5, xp=np)


def _panel_frame_with_missing():
    entity = np.repeat(np.arange(8), 5)
    time = np.tile(np.arange(5), 8)
    x = 0.3 * entity + 0.2 * time
    y = 1.0 + 2.0 * x + 0.4 * entity - 0.1 * time
    frame = pd.DataFrame({"y": y, "x": x, "entity": entity, "time": time})
    frame.loc[7, "x"] = np.nan
    return frame


def test_panel_formula_aligns_fixed_effect_ids_after_patsy_drops_rows():
    frame = _panel_frame_with_missing()
    model = PanelOLS().fit(formula="y ~ x | entity + time", data=frame)
    assert model.nobs == len(frame) - 1
    assert np.all(np.isfinite(model.coef_))


def test_pooled_formula_aligns_cluster_after_missing_rows():
    frame = _panel_frame_with_missing()
    cluster = np.asarray([f"entity-{v}" for v in frame.entity], dtype=object)
    model = PooledOLS(cov_type="clustered").fit(
        formula="y ~ x", data=frame, cluster=cluster
    )
    assert model.nobs == len(frame) - 1
    assert np.all(np.isfinite(model.bse_))


def test_between_formula_aligns_entity_ids_after_missing_rows():
    frame = _panel_frame_with_missing()
    model = BetweenOLS().fit(
        formula="y ~ x", data=frame, entity_ids=frame.entity.to_numpy()
    )
    assert model.nobs == frame.entity.nunique()
    assert np.all(np.isfinite(model.coef_))


def test_first_difference_formula_aligns_ids_after_missing_rows():
    frame = _panel_frame_with_missing()
    model = FirstDifferenceOLS().fit(
        formula="y ~ x - 1",
        data=frame,
        entity_ids=frame.entity.to_numpy(),
        time_ids=frame.time.to_numpy(),
    )
    assert np.all(np.isfinite(model.coef_))


def test_fama_macbeth_formula_aligns_time_ids_and_requires_two_periods():
    frame = _panel_frame_with_missing()
    model = FamaMacBeth(min_obs_per_period=3).fit(
        formula="y ~ x", data=frame, time_ids=frame.time.to_numpy()
    )
    assert model.nobs == len(frame) - 1
    assert model.n_periods >= 2

    one_period = pd.DataFrame({"y": np.arange(6.0), "x": np.arange(6.0)})
    with pytest.raises(ValueError, match="at least 2 time periods"):
        FamaMacBeth(min_obs_per_period=2).fit(
            formula="y ~ x", data=one_period, time_ids=np.zeros(6, dtype=int)
        )


def test_panel_rank_deficiency_uses_stable_pseudoinverse():
    x = np.arange(20.0)
    X = np.column_stack([x, 2.0 * x])
    y = 1.0 + 3.0 * x
    pooled = PooledOLS().fit(X, y)
    assert np.all(np.isfinite(pooled.coef_))


def test_between_requires_positive_residual_degrees_of_freedom():
    X = np.arange(12.0).reshape(6, 2)
    y = np.arange(6.0)
    entity = np.array([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="degrees of freedom"):
        BetweenOLS().fit(X, y, entity_ids=entity)
