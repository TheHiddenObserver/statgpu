"""Hosted contract checks for the Stage-C physical GPU validator."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np


_RUNNER = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
_SPEC = importlib.util.spec_from_file_location("panel_stage_c_gpu_runner", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_stage_c_runner_fresh_schema_version_is_explicit():
    assert _MOD.CORRECTNESS_SCHEMA_VERSION == 2


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
        assert np.all(np.isfinite(snap["covariance"])), name
        if snap["coefficient_inference_applicable"]:
            assert np.all(np.isfinite(snap["bse"])), name
            assert np.all(np.isfinite(snap["tvalues"])), name
            assert np.all(np.isfinite(snap["pvalues"])), name
            assert np.all(np.isfinite(snap["conf_int"])), name
            assert snap["coefficient_inference_reason"] is None
        else:
            assert snap["bse"] is None, name
            assert snap["tvalues"] is None, name
            assert snap["pvalues"] is None, name
            assert snap["conf_int"] is None, name
            assert "rank deficient" in snap["coefficient_inference_reason"], name


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
        assert np.all(np.isfinite(model._panel_cov_params_raw)), name
        assert model._coefficient_inference_available is False, name
        assert model.bse_ is None, name
        assert model.tvalues_ is None, name
        assert model.pvalues_ is None, name
        assert model.conf_int_ is None, name
        assert model._inference_result.metadata["applicable"] is False, name
        assert "rank deficient" in model._inference_result.metadata["reason"], name



def test_stage_c_runner_diagnostic_scale_audit_is_executable():
    audit = _MOD._diagnostic_scale_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    for field in (
        "pooling_f_statistic", "pooling_f_pvalue",
        "classical_f_statistic", "classical_f_pvalue",
        "bp_lm_statistic", "bp_lm_pvalue",
    ):
        assert np.isfinite(audit[field]), field


def test_stage_c_runner_zero_variance_inference_audit_is_executable():
    audit = _MOD._zero_variance_inference_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    assert audit["zero_coefficient_statistic"] == 0.0
    assert audit["positive_zero_variance_is_inf"] is True
    assert audit["negative_zero_variance_is_inf"] is True
    np.testing.assert_allclose(
        audit["tiny_positive_bse_statistics"],
        np.asarray([2.0, -3.0]),
        rtol=2.0e-15,
        atol=0.0,
    )


def test_stage_c_runner_covariance_extreme_scale_audit_is_executable():
    audit = _MOD._covariance_extreme_scale_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    for key in (
        "one_way",
        "two_way",
        "group_cancellation",
        "hac",
        "driscoll_kraay",
        "lag_accumulator_hac",
        "lag_accumulator_driscoll_kraay",
        "nonnested_two_way_structural_cancellation",
        "nonnested_two_way_safe_gram_cancellation",
        "nonnested_two_way_group_debias_cancellation",
    ):
        assert np.all(np.isfinite(np.asarray(audit[key], dtype=np.float64))), key


def test_stage_c_runner_hausman_scale_audit_is_executable():
    audit = _MOD._hausman_scale_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    assert audit["large_singular_range_rejected"] is True
    assert audit["dense_projection_range_rejected"] is True
    np.testing.assert_allclose(audit["dense_large_statistic"], 1.0, rtol=5e-13, atol=0.0)
    for label in ("large", "subnormal"):
        case = audit["cases"][label]
        assert case["df"] == 1.0
        assert np.isfinite(case["statistic"])
        assert np.isfinite(case["pvalue"])
        np.testing.assert_allclose(case["statistic"], 1.0, rtol=3e-12, atol=0.0)


def test_stage_c_runner_multiscale_grouping_audit_is_executable():
    audit = _MOD._multiscale_grouping_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    np.testing.assert_array_equal(
        np.asarray(audit["grouped"]),
        np.asarray([[1.0], [1.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(audit["one_way"]),
        np.asarray([[2.0]]),
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(audit["driscoll_kraay"]),
        np.asarray([[8.0 / 3.0]]),
        rtol=3e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(audit["deep_two_way"]),
        np.zeros((1, 1)),
        rtol=0.0,
        atol=0.0,
    )
    unsafe_amplitude = 1.0e200
    unsafe_low1 = 1.0e108
    unsafe_low2 = unsafe_low1 * (1.0 + 1.0e-3)
    np.testing.assert_allclose(
        np.asarray(audit["unsafe_cross_two_way"]),
        np.asarray(
            [[4.0 * unsafe_amplitude * (unsafe_low2 - unsafe_low1)]],
            dtype=np.float64,
        ),
        rtol=4e-12,
        atol=0.0,
    )
    tier_amplitude = 2.0 ** 660
    tier_tiny = 2.0 ** 350
    np.testing.assert_allclose(
        np.asarray(audit["third_tier_two_way"]),
        np.asarray([[-4.0 * tier_amplitude * tier_tiny]], dtype=np.float64),
        rtol=3e-12,
        atol=0.0,
    )


def test_stage_c_runner_registers_shared_mean_gpu_audit():
    audit_source = inspect.getsource(_MOD._cancellation_safe_mean_audit)
    assert "_scaled_mean" in audit_source
    assert "_scaled_group_means" in audit_source
    assert "1.0e-320" in audit_source

    main_source = inspect.getsource(_MOD.main)
    assert '"cancellation_safe_mean": _cancellation_safe_mean_audit(backend)' in main_source

def test_stage_c_runner_registers_nonfinite_covariance_gpu_guards():
    audit_source = inspect.getsource(_MOD._nonfinite_covariance_guard_audit)
    for token in (
        "clustered_covariance",
        "two_way_clustered_covariance",
        "hac_covariance",
        "driscoll_kraay_covariance",
        "accepted a NaN residual",
    ):
        assert token in audit_source
    main_source = inspect.getsource(_MOD.main)
    assert '"nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend)' in main_source

def test_stage_c_runner_registers_projection_created_dynamic_range_gpu_audit():
    audit_source = inspect.getsource(_MOD._projection_created_dynamic_range_audit)
    for token in (
        "stable_reduction_flags",
        "within_transform",
        "demean_variables",
        "raw_stability_flag",
        "post_entity_stability_flag",
    ):
        assert token in audit_source
    main_source = inspect.getsource(_MOD.main)
    assert '"projection_created_dynamic_range": _projection_created_dynamic_range_audit(backend)' in main_source



def test_stage_c_runner_registers_fe_effect_recovery_gpu_audit():
    audit_source = inspect.getsource(_MOD._fixed_effect_recovery_cancellation_audit)
    for token in (
        "PanelOLS",
        "_recover_two_way_effects",
        "demean_variables",
        "one-way FE prediction cancellation tail",
        "two-way FE recovery projection-created risk",
    ):
        assert token in audit_source
    main_source = inspect.getsource(_MOD.main)
    assert (
        '"fixed_effect_recovery_cancellation": _fixed_effect_recovery_cancellation_audit(backend)'
        in main_source
    )



def test_stage_c_runner_registers_two_way_effect_normalization_overflow_audit():
    audit_source = inspect.getsource(_MOD._two_way_effect_normalization_overflow_audit)
    for token in (
        "1.0e308",
        "_recover_two_way_effects",
        "two-way FE normalization overflow audit",
    ):
        assert token in audit_source
    main_source = inspect.getsource(_MOD.main)
    assert (
        '"two_way_effect_normalization_overflow": _two_way_effect_normalization_overflow_audit(backend)'
        in main_source
    )



def test_stage_c_runner_registers_common_scale_product_range_guard():
    audit_source = inspect.getsource(_MOD._common_scale_product_range_guard_audit)
    for token in (
        "1.0e308",
        "1.0e-100",
        "common-scale product range",
        "failed_closed",
    ):
        assert token in audit_source
    main_source = inspect.getsource(_MOD.main)
    assert (
        '"common_scale_product_range_guard": _common_scale_product_range_guard_audit(backend)'
        in main_source
    )
