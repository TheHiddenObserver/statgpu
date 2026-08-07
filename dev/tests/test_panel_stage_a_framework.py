"""Direct contracts for the Panel P1 Stage-A shared substrate."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel._base import BasePanelModel
from statgpu.panel._covariance import (
    clustered_covariance,
    hac_covariance,
    ols_covariance,
    two_way_clustered_covariance,
)
from statgpu.panel._results import (
    PanelFitStatistics,
    PanelTestResult,
    build_panel_index_info,
)


class _DummyPanel(BasePanelModel):
    def fit(self, X, y=None, **fit_params):
        _, _, X_arr, y_arr = self._panel_prepare_numeric(X, y, validate_alpha=False)
        X_np = np.asarray(X_arr)
        y_np = np.asarray(y_arr)
        self.coef_ = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
        self.bse_ = np.ones_like(self.coef_)
        self.tvalues_ = self.coef_.copy()
        self.pvalues_ = np.zeros_like(self.coef_)
        self.conf_int_ = np.column_stack([self.coef_ - 1, self.coef_ + 1])
        self.nobs = X_np.shape[0]
        self.df_resid = X_np.shape[0] - X_np.shape[1]
        self.alpha = 0.05
        self._design_info = None
        self._feature_names = None
        self._formula_has_intercept = None
        self._fitted = True
        return self

    def predict(self, X):
        return self._panel_predict_linear(
            X,
            model_has_intercept=False,
            add_intercept=False,
            return_numpy=True,
        )


def test_panel_result_substrate_is_inert_until_stage_b():
    result = PanelTestResult(null="pooled model is adequate")
    assert result.statistic is None
    assert result.pvalue is None
    assert result.applicable is False
    assert result.reason is None

    fit = PanelFitStatistics()
    assert fit.rsquared_within is None
    assert fit.f_statistic is None


def test_panel_index_info_balanced_unbalanced_and_order():
    entity = np.asarray(["b", "b", "a", "a"], dtype=object)
    time = np.asarray([2, 1, 2, 1])
    info = build_panel_index_info(4, entity_ids=entity, time_ids=time)
    assert info.n_entities == 2
    assert info.n_times == 2
    assert info.is_balanced is True
    assert info.has_duplicate_pairs is False
    assert info.entity_labels.tolist() == ["a", "b"]
    assert info.entity_counts.tolist() == [2, 2]
    assert info.original_order.tolist() == [0, 1, 2, 3]

    unbalanced = build_panel_index_info(
        3, entity_ids=entity[:3], time_ids=time[:3]
    )
    assert unbalanced.is_balanced is False
    assert unbalanced.has_duplicate_pairs is False

    duplicate = build_panel_index_info(
        3, entity_ids=["a", "a", "b"], time_ids=[1, 1, 1]
    )
    assert duplicate.has_duplicate_pairs is True
    assert duplicate.is_balanced is False


def test_panel_index_info_validates_metadata_lengths():
    try:
        build_panel_index_info(4, entity_ids=[0, 1, 2])
    except ValueError as exc:
        assert "4 observations" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("length mismatch must fail")


def _ols_problem():
    X = np.asarray(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    )
    y = np.asarray([0.2, 1.1, 2.4, 2.7, 4.5])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    df = X.shape[0] - np.linalg.matrix_rank(X)
    scale = float(resid @ resid) / df
    return X, resid, scale, df


def test_ols_covariance_registry_preserves_nonrobust_and_hc1_formulas():
    X, resid, scale, df = _ols_problem()
    bread = np.linalg.pinv(X.T @ X)

    expected_nonrobust = scale * bread
    actual_nonrobust = ols_covariance(
        X,
        resid,
        cov_type="nonrobust",
        scale=scale,
        df_resid=df,
        xp=np,
        allowed=("nonrobust", "robust"),
    )
    assert_allclose(actual_nonrobust, expected_nonrobust, rtol=1e-13, atol=1e-14)

    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    expected_hc1 = bread @ meat @ bread * (X.shape[0] / df)
    actual_hc1 = ols_covariance(
        X,
        resid,
        cov_type="robust",
        df_resid=df,
        xp=np,
        allowed=("nonrobust", "robust"),
    )
    assert_allclose(actual_hc1, expected_hc1, rtol=1e-13, atol=1e-14)


def test_ols_covariance_registry_delegates_cluster_and_hac_exactly():
    X, resid, _, df = _ols_problem()
    cluster = np.asarray([0, 0, 1, 1, 1])
    two_way = np.column_stack([cluster, np.asarray([0, 1, 0, 1, 2])])

    assert_allclose(
        ols_covariance(X, resid, cov_type="clustered", cluster=cluster, df_resid=df, xp=np),
        clustered_covariance(X, resid, cluster, xp=np),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        ols_covariance(X, resid, cov_type="clustered", cluster=two_way, df_resid=df, xp=np),
        two_way_clustered_covariance(X, resid, two_way[:, 0], two_way[:, 1], xp=np),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        ols_covariance(X, resid, cov_type="hac", bandwidth=2, kernel="bartlett", df_resid=df, xp=np),
        hac_covariance(X, resid, bandwidth=2, kernel="bartlett", xp=np),
        rtol=0,
        atol=0,
    )


def test_covariance_registry_rejects_context_unsupported_name():
    X, resid, scale, df = _ols_problem()
    try:
        ols_covariance(
            X,
            resid,
            cov_type="hac",
            scale=scale,
            df_resid=df,
            xp=np,
            allowed=("nonrobust", "robust"),
        )
    except ValueError as exc:
        assert "not supported here" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("context-unsupported covariance must fail")


def test_base_panel_numeric_prediction_and_summary_contract():
    X = np.asarray([[1.0, 2.0], [2.0, -1.0], [0.5, 0.25], [3.0, 1.0]])
    y = X @ np.asarray([0.7, -0.3])
    model = _DummyPanel(device="cpu").fit(X, y)
    assert_allclose(model.predict(X), y, rtol=1e-12, atol=1e-12)

    summary = model._panel_summary(model_type="DummyPanel")
    payload = summary.to_dict()
    assert payload["model_type"] == "DummyPanel"
    assert payload["nobs"] == 4
    assert payload["df_resid"] == 2
    assert payload["feature_names"] == ["x0", "x1"]
