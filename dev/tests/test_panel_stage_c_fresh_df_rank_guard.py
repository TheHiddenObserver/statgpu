"""Fresh review regressions for FE df and rank-deficient HC guards."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

statsmodels = pytest.importorskip("statsmodels.api")

from statgpu.panel import PanelOLS, PooledOLS


def _panel(seed=2026081705):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 9, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.4, size=n_entities), n_times)
    tau = np.tile(np.linspace(-0.25, 0.3, n_times), n_entities)
    y = 0.6 * X[:, 0] - 0.35 * X[:, 1] + alpha + 0.2 * tau
    y += rng.normal(scale=0.18, size=entity.size)
    return X, y, entity, time


def _entity_dummy_design(X, entity):
    groups = np.unique(entity)
    dummies = np.column_stack([(entity == group).astype(float) for group in groups])
    return np.column_stack([X, dummies])


def _two_way_dummy_design(X, entity, time):
    entities = np.unique(entity)
    times = np.unique(time)
    entity_dummies = np.column_stack(
        [(entity == group).astype(float) for group in entities]
    )
    time_dummies = np.column_stack(
        [(time == period).astype(float) for period in times[1:]]
    )
    return np.column_stack([X, entity_dummies, time_dummies])


@pytest.mark.parametrize("cov_type", ["nonrobust", "robust"])
def test_one_way_panel_public_inference_matches_explicit_dummy_ols(cov_type):
    X, y, entity, _time = _panel()
    model = PanelOLS(entity_effects=True, cov_type=cov_type).fit(
        X, y, entity_ids=entity
    )
    design = _entity_dummy_design(X, entity)
    reference = statsmodels.OLS(y, design).fit()
    if cov_type == "robust":
        reference = reference.get_robustcov_results(cov_type="HC1")

    k = X.shape[1]
    assert model.df_resid == int(reference.df_resid)
    assert model.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert_allclose(model.coef_, reference.params[:k], rtol=2e-11, atol=2e-12)
    assert_allclose(
        model._panel_cov_params_raw,
        reference.cov_params()[:k, :k],
        rtol=2e-9,
        atol=2e-11,
    )
    assert_allclose(model.bse_, reference.bse[:k], rtol=2e-9, atol=2e-11)


def test_two_way_panel_nonrobust_df_and_covariance_match_explicit_dummies():
    X, y, entity, time = _panel(seed=2026081706)
    model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="nonrobust",
    ).fit(X, y, entity_ids=entity, time_ids=time)
    design = _two_way_dummy_design(X, entity, time)
    reference = statsmodels.OLS(y, design).fit()

    k = X.shape[1]
    assert model.df_resid == int(reference.df_resid)
    assert model.fit_statistics_.metadata["diagnostic_df"]["effect_rank"] == (
        len(np.unique(entity)) + len(np.unique(time)) - 1
    )
    assert_allclose(model.coef_, reference.params[:k], rtol=2e-11, atol=2e-12)
    assert_allclose(
        model._panel_cov_params_raw,
        reference.cov_params()[:k, :k],
        rtol=2e-9,
        atol=2e-11,
    )


@pytest.mark.parametrize("cov_type", ["hc2", "hc3"])
def test_rank_deficient_hc_with_unit_leverage_keeps_fit_and_disables_inference(cov_type):
    # PooledOLS adds an intercept.  Together with the duplicated first-row
    # indicator this design has rank 2 < 3 and h_0=1, so HC2/HC3 coordinate
    # covariance is undefined even though the least-squares fitted values exist.
    indicator = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X = np.column_stack([indicator, 2.0 * indicator])
    y = np.array([1.5, 0.2, -0.1, 0.4, 0.0, 0.3])

    model = PooledOLS(cov_type=cov_type).fit(X, y)
    assert model._coefficient_inference_available is False
    assert model.bse_ is None
    assert model.pvalues_ is None
    assert model.conf_int_ is None
    assert model._panel_cov_params_raw is None
    assert model._covariance_metadata["rank_deficient_covariance_unavailable"] is True
    assert model._covariance_metadata["design_rank"] == 2
    assert model._covariance_metadata["design_columns"] == 3
    pred = model.predict(X)
    assert np.all(np.isfinite(pred))
    with pytest.raises(ValueError, match="rank deficient"):
        model.summary()


def test_one_way_panel_standard_df_torch_cpu_matches_numpy():
    torch = pytest.importorskip("torch")
    X, y, entity, _time = _panel(seed=2026081707)
    expected = PanelOLS(entity_effects=True, cov_type="robust").fit(
        X, y, entity_ids=entity
    )
    actual = PanelOLS(entity_effects=True, cov_type="robust").fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        entity_ids=torch.as_tensor(entity, dtype=torch.int64),
    )
    assert actual.df_resid == expected.df_resid
    assert_allclose(actual.coef_, expected.coef_, rtol=2e-9, atol=2e-11)
    assert_allclose(actual.bse_, expected.bse_, rtol=2e-8, atol=2e-10)
