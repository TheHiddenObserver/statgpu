"""Stage-C public API, formula alignment, and metadata-order contracts."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

pd = pytest.importorskip("pandas")
patsy = pytest.importorskip("patsy")

from statgpu.inference._results import ParameterInferenceResult
from statgpu.panel import BetweenOLS, PanelOLS, PooledOLS, RandomEffects
from statgpu.panel._covariance import driscoll_kraay_covariance


def _frame(seed=12800):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 8, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    x = rng.normal(size=entity.size)
    z = rng.normal(size=entity.size)
    alpha = np.repeat(rng.normal(scale=0.4, size=n_entities), n_times)
    y = 0.7 * x - 0.25 * z + alpha + rng.normal(scale=0.2, size=entity.size)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "entity": entity,
            "time": time,
        }
    )


def test_random_effects_stage_c_options_do_not_shift_old_positional_arguments():
    model = RandomEffects(0.1, "cpu", 3, cov_type="hc2", bandwidth=2)
    assert model.alpha == 0.1
    assert str(model.device).lower().endswith("cpu")
    assert model.n_jobs == 3
    assert model.cov_type == "hc2"
    assert model.bandwidth == 2
    params = model.get_params()
    assert params["alpha"] == 0.1
    assert params["cov_type"] == "hc2"
    assert params["bandwidth"] == 2
    assert params["group_debias"] is False


def test_random_effects_hc1_alias_preserves_raw_constructor_identity_and_runtime_robust_contract():
    model = RandomEffects(cov_type="hc1")
    assert model.cov_type == "hc1"
    assert model._cov_type == "robust"
    params = model.get_params()
    assert params["cov_type"] == "hc1"


def test_panel_dk_formula_missing_rows_matches_explicit_filtered_array_fit():
    data = _frame(12801)
    data.loc[[3, 17], "x"] = np.nan
    formula_model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="dk",
        bandwidth=2,
    ).fit(formula="y ~ x + z | entity + time", data=data)

    keep = data[["y", "x", "z"]].notna().all(axis=1).to_numpy()
    X = data.loc[keep, ["x", "z"]].to_numpy()
    y = data.loc[keep, "y"].to_numpy()
    entity = data.loc[keep, "entity"].to_numpy()
    time = data.loc[keep, "time"].to_numpy()
    array_model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="dk",
        bandwidth=2,
    ).fit(X, y, entity_ids=entity, time_ids=time)

    assert_allclose(formula_model.coef_, array_model.coef_, rtol=2e-12, atol=2e-13)
    assert_allclose(formula_model.bse_, array_model.bse_, rtol=2e-11, atol=2e-13)
    assert formula_model._covariance_metadata["n_periods"] == len(np.unique(time))


def test_pooled_two_way_string_clusters_follow_formula_missing_row_filter():
    data = _frame(12802)
    data["firm_cluster"] = np.array([f"firm-{i}" for i in data["entity"]])
    data["calendar_cluster"] = np.array([f"t-{i}" for i in data["time"]])
    clusters = data[["firm_cluster", "calendar_cluster"]].to_numpy(dtype=object)
    data.loc[[5, 24], "z"] = np.nan

    formula_model = PooledOLS(
        cov_type="clustered",
        group_debias=True,
    ).fit(
        formula="y ~ x + z",
        data=data,
        cluster=clusters,
    )

    keep = data[["y", "x", "z"]].notna().all(axis=1).to_numpy()
    array_model = PooledOLS(
        cov_type="clustered",
        group_debias=True,
    ).fit(
        data.loc[keep, ["x", "z"]].to_numpy(),
        data.loc[keep, "y"].to_numpy(),
        cluster=clusters[keep],
    )
    assert_allclose(formula_model.coef_, array_model.coef_, rtol=2e-12, atol=2e-13)
    assert_allclose(formula_model.bse_, array_model.bse_, rtol=2e-11, atol=2e-13)
    assert formula_model._covariance_metadata["cluster_dimensions"] == 2


def test_random_effects_dk_formula_metadata_alignment_matches_constant_array_fit():
    data = _frame(12803)
    data.loc[[7, 20], "x"] = np.nan
    formula_model = RandomEffects(cov_type="dk", bandwidth=2).fit(
        formula="y ~ x + z | entity + time",
        data=data,
    )

    keep = data[["y", "x", "z"]].notna().all(axis=1).to_numpy()
    X = np.column_stack(
        [
            np.ones(int(np.sum(keep))),
            data.loc[keep, ["x", "z"]].to_numpy(),
        ]
    )
    array_model = RandomEffects(cov_type="dk", bandwidth=2).fit(
        X,
        data.loc[keep, "y"].to_numpy(),
        entity_ids=data.loc[keep, "entity"].to_numpy(),
        time_ids=data.loc[keep, "time"].to_numpy(),
    )
    assert_allclose(formula_model.coef_, array_model.coef_, rtol=2e-12, atol=2e-13)
    assert_allclose(formula_model.bse_, array_model.bse_, rtol=2e-11, atol=2e-13)
    summary = formula_model.summary()
    assert summary.feature_names == ["Intercept", "x", "z"]


@pytest.mark.parametrize("estimator", [PooledOLS, BetweenOLS])
def test_formula_added_constant_keeps_intercept_in_summary_and_inference_names(estimator):
    data = _frame(128035)
    kwargs = {}
    if estimator is BetweenOLS:
        kwargs["entity_ids"] = data["entity"].to_numpy()
    model = estimator(cov_type="hc0").fit(
        formula="y ~ x + z",
        data=data,
        **kwargs,
    )

    expected_names = ["Intercept", "x", "z"]
    summary = model.summary()
    assert summary.feature_names == expected_names
    assert len(summary.feature_names) == len(summary.coef) == len(model.coef_)
    assert model._feature_names == expected_names
    assert model._inference_result.feature_names == expected_names



def test_random_effects_formula_matches_patsy_categorical_interaction_transform_matrix():
    data = _frame(128034)
    data["group"] = pd.Categorical(
        np.where(
            data["time"].to_numpy() % 3 == 0,
            "base",
            np.where(data["time"].to_numpy() % 3 == 1, "mid", "high"),
        ),
        categories=["base", "mid", "high"],
    )
    main_formula = "y ~ x * C(group) + I(z ** 2)"
    y_design, X_design = patsy.dmatrices(
        main_formula, data, return_type="dataframe"
    )

    formula_model = RandomEffects().fit(
        formula=f"{main_formula} | entity",
        data=data,
    )
    array_model = RandomEffects().fit(
        X_design.to_numpy(),
        y_design.iloc[:, 0].to_numpy(),
        entity_ids=data["entity"].to_numpy(),
    )

    expected_names = list(X_design.design_info.column_names)
    assert formula_model._feature_names == expected_names
    assert "Intercept" in expected_names
    assert "C(group)[T.mid]" in expected_names
    assert "C(group)[T.high]" in expected_names
    assert "x:C(group)[T.mid]" in expected_names
    assert "x:C(group)[T.high]" in expected_names
    assert "I(z ** 2)" in expected_names
    assert_allclose(formula_model.coef_, array_model.coef_, rtol=2e-12, atol=2e-13)
    assert_allclose(formula_model.bse_, array_model.bse_, rtol=2e-11, atol=2e-13)
    assert_allclose(formula_model.pvalues_, array_model.pvalues_, rtol=2e-11, atol=2e-13)

    expected_prediction = X_design.to_numpy() @ np.asarray(formula_model.coef_)
    assert_allclose(
        formula_model.predict(data),
        expected_prediction,
        rtol=2e-12,
        atol=2e-13,
    )


@pytest.mark.parametrize(
    "column,expected_name",
    [("entity", "entity_ids"), ("time", "time_ids")],
)
def test_pipe_formula_rejects_missing_fixed_effect_labels(column, expected_name):
    data = _frame(128031)
    data.loc[5, column] = np.nan
    with pytest.raises(
        ValueError,
        match=rf"{expected_name}.*missing or non-finite",
    ):
        PanelOLS(entity_effects=True, time_effects=True).fit(
            formula="y ~ x + z | entity + time",
            data=data,
        )


def test_formula_aligned_cluster_rejects_missing_labels():
    data = _frame(128032)
    clusters = data["entity"].astype(object).to_numpy().copy()
    clusters[4] = None
    with pytest.raises(ValueError, match="clusters.*missing or non-finite"):
        PooledOLS(cov_type="clustered").fit(
            formula="y ~ x + z",
            data=data,
            cluster=clusters,
        )


def test_stage_c_inference_populates_unified_result_and_internal_attributes():
    data = _frame(128033)
    model = PooledOLS(cov_type="hc0").fit(
        data[["x", "z"]].to_numpy(),
        data["y"].to_numpy(),
    )
    assert isinstance(model._inference_result, ParameterInferenceResult)
    assert_allclose(model._params, model.coef_)
    assert_allclose(model._bse, model.bse_)
    assert_allclose(model._tvalues, model.tvalues_)
    assert_allclose(model._zvalues, model.tvalues_)
    assert_allclose(model._pvalues, model.pvalues_)
    assert_allclose(model._conf_int, model.conf_int_)

    adjusted = model.adjust_pvalues()
    assert adjusted["pvalues_adjusted"].shape == model.pvalues_.shape
    combined = model.combine_pvalues()
    assert np.isfinite(float(combined["pvalue"]))


def test_dk_is_invariant_to_row_permutation_within_and_across_time_groups():
    rng = np.random.default_rng(12804)
    time = np.tile(np.arange(7), 8)
    X = np.column_stack([np.ones(time.size), rng.normal(size=(time.size, 2))])
    resid = rng.normal(size=time.size)
    base = driscoll_kraay_covariance(
        X, resid, time, bandwidth=2, kernel="parzen"
    )
    order = rng.permutation(time.size)
    permuted = driscoll_kraay_covariance(
        X[order], resid[order], time[order], bandwidth=2, kernel="parzen"
    )
    assert_allclose(permuted, base, rtol=2e-12, atol=2e-13)


def test_dk_rejects_mixed_nonorderable_time_labels():
    X = np.column_stack([np.ones(4), np.arange(4.0)])
    resid = np.array([0.1, -0.2, 0.3, -0.1])
    time = np.array([0, "one", 2, "three"], dtype=object)
    with pytest.raises(ValueError, match="deterministic"):
        driscoll_kraay_covariance(X, resid, time, bandwidth=1)


def test_robust_random_effects_hausman_has_explicit_covariance_reason():
    data = _frame(12805)
    X = data[["x", "z"]].to_numpy()
    y = data["y"].to_numpy()
    entity = data["entity"].to_numpy()
    fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        X, y, entity_ids=entity
    )
    re = RandomEffects(cov_type="hc2").fit(X, y, entity_ids=entity)
    result = fe.hausman_test(re)
    assert result.applicable is False
    assert result.reason is not None
    assert "nonrobust RE covariance" in result.reason
