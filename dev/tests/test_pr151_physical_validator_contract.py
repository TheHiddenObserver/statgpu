"""Static contract for the issue #150 physical CUDA validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path


VALIDATOR = Path("dev/benchmarks/validate_glm_weighted_explicit_solvers_gpu.py")


def _load_validator():
    spec = importlib.util.spec_from_file_location("pr151_validator", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pr151_validator_schema_matrix_and_tolerances_are_frozen():
    module = _load_validator()
    assert module.SCHEMA_VERSION == 2
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


def test_pr151_validator_promotes_solver_warnings_to_failures():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert 'warnings.simplefilter("error", ConvergenceWarning)' in source
    assert 'message="lbfgs_solver: line search failed.*"' in source
    assert '"convergence_warning": "error"' in source
    assert '"lbfgs_line_search_failure": "error"' in source


def test_pr151_validator_requires_shared_cv_parity_and_selected_alpha_identity():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert '"errors_vs_numpy": _assert_parity(' in source
    assert 'cv["selected_alpha"] != ref_cv["selected_alpha"]' in source
    assert '"selected_alpha_matches_numpy": True' in source
