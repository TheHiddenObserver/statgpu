"""Hosted contract checks for the Stage-C physical GPU validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_RUNNER = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
_SPEC = importlib.util.spec_from_file_location("panel_stage_c_gpu_runner", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_stage_c_runner_numpy_reference_matrix_is_complete_and_executable():
    X, y, entity, time, clusters = _MOD._dataset()
    cases = _MOD._fit_cases(X, y, entity, time, clusters, "numpy")
    required = {
        "pooled_hc0", "pooled_hc2", "pooled_hc3",
        "pooled_cluster_one_way", "pooled_cluster_two_way_group_debias",
        "pooled_dk_bartlett", "pooled_dk_qs", "pooled_legacy_hac",
        "panel_entity_hc0", "panel_entity_hc2", "panel_entity_hc3",
        "panel_two_way_hc3", "panel_two_way_cluster_group_debias", "panel_two_way_dk",
        "random_effects_explicit_constant_robust",
        "random_effects_explicit_constant_hc0",
        "random_effects_explicit_constant_hc2",
        "random_effects_explicit_constant_hc3",
        "random_effects_cluster_two_way", "random_effects_dk",
        "between_hc0", "between_hc2", "between_hc3",
        "first_difference_hc0", "first_difference_hc2", "first_difference_hc3",
        "panel_entity_rank_deficient_nonrobust", "panel_entity_rank_deficient_robust",
        "between_rank_deficient_nonrobust", "between_rank_deficient_robust",
        "first_difference_rank_deficient_nonrobust", "first_difference_rank_deficient_robust",
        "random_effects_rank_deficient_nonrobust", "random_effects_rank_deficient_robust",
        "panel_rank_boundary_dk",
    }
    assert required <= set(cases)
    assert len(cases) == len(required)
    for name, model in cases.items():
        snap = _MOD._snapshot(model)
        assert np.all(np.isfinite(snap["coef"])), name
        assert np.all(np.isfinite(snap["bse"])), name
        assert np.all(np.isfinite(snap["covariance"])), name


def test_stage_c_runner_physically_requires_qs_all_lag_contract():
    X, y, entity, time, clusters = _MOD._dataset()
    cases = _MOD._fit_cases(X, y, entity, time, clusters, "numpy")
    qs = _MOD._snapshot(cases["pooled_dk_qs"])["covariance_metadata"]
    assert qs["kernel"] == "qs"
    assert qs["bandwidth"] == 2
    assert qs["n_periods"] == len(np.unique(time))
    assert qs["all_observed_lags_weighted"] is True
    assert qs["max_weighted_lag"] == len(np.unique(time)) - 1


def test_stage_c_runner_group_debias_and_panel_dk_metadata_are_auditable():
    X, y, entity, time, clusters = _MOD._dataset()
    cases = _MOD._fit_cases(X, y, entity, time, clusters, "numpy")
    clustered = _MOD._snapshot(
        cases["panel_two_way_cluster_group_debias"]
    )["covariance_metadata"]
    assert clustered["group_debias"] is True
    assert clustered["cluster_dimensions"] == 2
    assert len(clustered["cluster_group_counts"]) == 3
    assert len(clustered["group_debias_factors"]) == 3

    panel_dk = cases["panel_two_way_dk"]
    dk_meta = _MOD._snapshot(panel_dk)["covariance_metadata"]
    effect_rank = panel_dk.fit_statistics_.metadata["diagnostic_df"]["effect_rank"]
    assert dk_meta["extra_df"] == effect_rank
    assert dk_meta["rank_deficient_extension"] is False


def test_stage_c_runner_public_primitive_matrix_is_complete():
    X, y, entity, time, clusters = _MOD._dataset()
    values = _MOD._public_primitive_cases(
        X, y, entity, time, clusters, "numpy"
    )
    assert set(values) == {
        "cluster_group_debias",
        "driscoll_kraay_qs",
        "ill_conditioned_hc0",
        "ill_conditioned_hc2",
        "ill_conditioned_hc3",
        "ill_conditioned_dk",
        "rank_boundary_nonrobust",
        "rank_boundary_hc0",
        "rank_boundary_hc2",
        "rank_boundary_hc3",
        "rank_boundary_cluster",
        "rank_boundary_dk",
    }
    for value in values.values():
        arr = np.asarray(value, dtype=np.float64)
        assert arr.shape == (3, 3)
        assert np.all(np.isfinite(arr))


def test_stage_c_runner_rank_boundary_is_explicitly_rank_two():
    X, resid, time, _cluster = _MOD._rank_boundary_inputs()
    meta = {}
    cov = _MOD.ols_covariance(
        X, resid, cov_type="driscoll-kraay", time_ids=time, bandwidth=2, metadata=meta
    )
    assert np.all(np.isfinite(cov))
    assert meta["design_rank"] == 2
    assert meta["design_columns"] == 3
    assert meta["rank_deficient_extension"] is True



def test_stage_c_runner_rank_deficient_estimators_exercise_identified_rank_df():
    X, y, entity, time, clusters = _MOD._dataset()
    cases = _MOD._fit_cases(X, y, entity, time, clusters, "numpy")
    names = {
        "panel_entity_rank_deficient_nonrobust",
        "panel_entity_rank_deficient_robust",
        "between_rank_deficient_nonrobust",
        "between_rank_deficient_robust",
        "first_difference_rank_deficient_nonrobust",
        "first_difference_rank_deficient_robust",
        "random_effects_rank_deficient_nonrobust",
        "random_effects_rank_deficient_robust",
    }
    for name in names:
        model = cases[name]
        fit_rank = _MOD._fit_rank(model)
        assert fit_rank < len(np.asarray(model.coef_).ravel()), name
        assert model.df_resid > 0, name
        assert np.all(np.isfinite(model.bse_)), name
