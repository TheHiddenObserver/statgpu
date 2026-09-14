"""Static contract for the issue #150 physical CUDA validators."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


VALIDATOR = Path("dev/benchmarks/validate_glm_weighted_explicit_solvers_gpu.py")
VALIDATOR_V4 = Path("dev/benchmarks/validate_pr151_inverse_gamma_domain_gpu_v4.py")
VALIDATOR_V5 = Path("dev/benchmarks/validate_pr151_final_gpu_v5.py")
VALIDATOR_V6 = Path("dev/benchmarks/validate_pr151_final_gpu_v6.py")
ARTIFACT_V5 = Path("dev/reviews/pr151_final_gpu_v5.json")
ACCEPTED_V5_SOURCE_SHA = "9eb39cee2c0e691160f528ab687b7379d26f3e42"


def _load_validator(path=VALIDATOR, name="pr151_validator"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pr151_validator_schema_matrix_and_tolerances_are_frozen():
    module = _load_validator()
    assert module.SCHEMA_VERSION == 3
    assert module.SOLVER_TOL == 1.0e-8
    assert module.ATOL_COEF == 2.0e-5
    assert module.ATOL_INTERCEPT == 2.0e-5
    assert module.ATOL_WEIGHT_RESCALE == 2.0e-6
    assert module._SOLVERS == ("newton", "lbfgs")
    assert module._CASES == (
        "gaussian",
        "binomial",
        "poisson",
        "gamma_log",
        "gamma_inverse",
        "inverse_gaussian",
        "negative_binomial",
        "tweedie",
    )
    assert module._CONSUMER_CASES == (
        "negative_binomial",
        "gamma",
        "inverse_gaussian",
    )


def test_pr151_validator_promotes_solver_warnings_to_failures():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert 'warnings.simplefilter("error", ConvergenceWarning)' in source
    assert 'message="lbfgs_solver: line search failed.*"' in source
    assert '"convergence_warning": "error"' in source
    assert '"lbfgs_line_search_failure": "error"' in source


def test_pr151_validator_requires_all_shared_cv_parity_and_selected_alpha_identity():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "for case in _CONSUMER_CASES:" in source
    assert '"errors_vs_numpy": _assert_parity(' in source
    assert 'cv["selected_alpha"] != ref_cv["selected_alpha"]' in source
    assert '"selected_alpha_matches_numpy": True' in source
    assert '"consumer_case_count": len(_CONSUMER_CASES)' in source


def test_pr151_schema_v4_extends_v3_without_rewriting_historical_contract():
    module = _load_validator(VALIDATOR_V4, "pr151_validator_v4")
    assert module.SCHEMA_VERSION == 4
    assert module.SOLVER_TOL == 1.0e-8
    assert module.ATOL_COEF == 2.0e-5
    assert module.ATOL_INTERCEPT == 2.0e-5
    assert module.ATOL_WEIGHT_RESCALE == 2.0e-6
    assert module._SOLVERS == ("newton", "lbfgs")

    source = VALIDATOR_V4.read_text(encoding="utf-8")
    assert "v3.run(v3_path)" in source
    assert '"legacy_schema_v3": legacy' in source
    assert '"source_sha": source_sha' in source
    assert '"source_clean": True' in source
    assert 'default=Path("dev/reviews/pr151_inverse_gamma_domain_gpu_v4.json")' in source


def test_pr151_schema_v4_covers_inverse_gamma_domain_and_consumers():
    source = VALIDATOR_V4.read_text(encoding="utf-8")
    assert 'fit_intercept=False' in source
    assert 'loss_kwargs={"link": "inverse_power"}' in source
    assert '"ordinary_no_intercept": ordinary' in source
    assert '"cross_container": crossings' in source
    assert '"penalized_l2_no_intercept": penalized' in source
    assert '"smooth_l2_cv_intercept": cv_routes' in source
    assert '"gpu_negative_domain"' in source
    assert '"selected_alpha_matches_numpy": True' in source
    assert 'active contradictory design unexpectedly fitted' in source
    assert '"errors_vs_row_deletion": errors' in source
    assert '"eta_min_active"' in source
    assert '"eta_max_active"' in source


def test_pr151_schema_v5_extends_v4_for_final_review_closure():
    module = _load_validator(VALIDATOR_V5, "pr151_validator_v5")
    assert module.SCHEMA_VERSION == 5
    assert module.SOLVER_TOL == 1.0e-8
    assert module.ATOL_COEF == 2.0e-5
    assert module.ATOL_INTERCEPT == 2.0e-5
    assert module.ATOL_WEIGHT_RESCALE == 2.0e-6
    assert module._SOLVERS == ("newton", "lbfgs")
    assert module._EXTREME_WEIGHT_SCALES == (1.0e-200, 1.0e200)
    assert module._FLOAT32_OVERFLOW_SCALE == 3.0e38
    assert module._ROUNDOFF_SOLVER_TOL == 1.0e-9
    assert module._ROUNDOFF_WEIGHT_SCALE == 6.0

    source = VALIDATOR_V5.read_text(encoding="utf-8")
    assert "v4.run(v4_path)" in source
    assert '"legacy_schema_v4": legacy' in source
    assert '"source_sha": source_sha' in source
    assert '"source_clean": True' in source
    assert 'default=Path("dev/reviews/pr151_final_gpu_v5.json")' in source


def test_pr151_schema_v5_covers_review_found_numerical_edges():
    source = VALIDATOR_V5.read_text(encoding="utf-8")
    assert '"extreme_global_weight_rescaling"' in source
    assert '"float32_raw_sum_overflow"' in source
    assert '"integer_design_fractional_weights"' in source
    assert '"penalized_effective_uniform_inference"' in source
    assert '"penalized_gamma_lbfgs_roundoff_stopping"' in source
    assert '"gpu_domain_pinned_fail_closed"' in source
    assert "_penalized_gamma_roundoff_gate" in source
    assert '"roundoff_solver_tol": _ROUNDOFF_SOLVER_TOL' in source
    assert '"roundoff_weight_scale": _ROUNDOFF_WEIGHT_SCALE' in source
    assert "1.0e-200" in source
    assert "1.0e200" in source
    assert "3.0e38" in source
    assert "raw_float32_sum_overflow" in source
    assert "rng.integers" in source
    assert 'dtype=torch.int64' in source
    assert 'inference_method="auto"' in source
    assert '"m_estimation"' in source
    assert 'boundary surrogate' in source
    assert '"pinned to the maintained smooth-domain boundary"' in source
    assert '"no positive interior line-search step"' in source
    assert '"postfit_design_is_floating"' in source
    assert '"postfit_params_error_vs_coef"' in source
    assert '"loglikelihood"' in source


def test_pr151_schema_v5_accepted_artifact_identity_is_frozen():
    payload = json.loads(ARTIFACT_V5.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 5
    assert payload["status"] == "success"
    assert payload["source_clean"] is True
    assert payload["source_sha"] == ACCEPTED_V5_SOURCE_SHA
    assert payload["frozen_tolerances"]["roundoff_solver_tol"] == 1.0e-9
    assert payload["frozen_tolerances"]["roundoff_weight_scale"] == 6.0

    v4 = payload["legacy_schema_v4"]
    assert v4["schema_version"] == 4
    assert v4["status"] == "success"
    assert v4["source_clean"] is True
    assert v4["source_sha"] == ACCEPTED_V5_SOURCE_SHA

    v3 = v4["legacy_schema_v3"]
    assert v3["schema_version"] == 3
    assert v3["status"] == "success"
    assert v3["source_clean"] is True
    assert v3["source_sha"] == ACCEPTED_V5_SOURCE_SHA

    review = payload["review_closure"]
    assert "penalized_gamma_lbfgs_roundoff_stopping" in review
    assert "gpu_domain_pinned_fail_closed" in review
    assert set(review["gpu_domain_pinned_fail_closed"]) == {"cupy", "torch"}


def test_pr151_schema_v6_extends_v5_for_analytic_weight_inference_closure():
    module = _load_validator(VALIDATOR_V6, "pr151_validator_v6")
    assert module.SCHEMA_VERSION == 6
    assert module.SOLVER_TOL == 1.0e-8
    assert module.ATOL_COEF == 2.0e-5
    assert module.ATOL_INTERCEPT == 2.0e-5
    assert module.ATOL_WEIGHT_RESCALE == 2.0e-6
    assert module.ATOL_INFERENCE == 2.0e-5
    assert module._WEIGHT_SCALE == 7.25
    assert module._FLOAT32_OVERFLOW_SCALE == 3.0e38
    assert module._SOLVERS == ("newton", "lbfgs")

    source = VALIDATOR_V6.read_text(encoding="utf-8")
    assert "v5.run(v5_path)" in source
    assert '"legacy_schema_v5": legacy' in source
    assert '"source_sha": source_sha' in source
    assert '"source_clean": True' in source
    assert 'default=Path("dev/reviews/pr151_final_gpu_v6.json")' in source
    assert '"analytic_weight_nonrobust_inference_scale_invariance"' in source
    assert '"float32_raw_sum_overflow_inference"' in source
    assert "_analytic_weight_inference_gate" in source
    assert "_float32_raw_sum_overflow_inference_gate" in source
    assert 'cov_type="nonrobust"' in source
    assert 'inference_method="auto"' in source
    assert '("ordinary", _fit_ordinary)' in source
    assert '("penalized", _fit_penalized)' in source
    assert '("cupy", "cuda")' in source
    assert '("torch", "torch")' in source
    assert '"inference_weight_scale": _WEIGHT_SCALE' in source
    assert '"float32_overflow_scale": _FLOAT32_OVERFLOW_SCALE' in source
    assert '"raw_float32_sum_overflow": True' in source
