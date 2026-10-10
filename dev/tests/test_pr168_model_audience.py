"""Protect learner-first placement without weakening model limitations or API links."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _text(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


@pytest.mark.parametrize("language", ("en", "cn"))
def test_glm_keeps_weight_and_initialization_contract_without_solver_internals(language):
    page = _text(f"docs/{language}/models/generalized-linear-model.md")
    for token in ("sample_weight", 'solver="newton"', 'solver="lbfgs"',
                  'link="inverse_power"', r"\eta_i=b+x_i^\top\beta>0",
                  "solver-penalty-matrix.md", "solver-algorithms.md",
                  "linear-model-api.md#generalizedlinearmodel"):
        assert token in page
    assert "LossBase" not in page
    assert "Armijo" not in page
    developer = _text("dev/references/glm-weighted-solver-implementation.md")
    assert "LossBase" in developer and "Armijo" in developer
    assert "$Xd>0$" in developer


@pytest.mark.parametrize("language", ("en", "cn"))
def test_cox_keeps_actionable_transfer_and_screening_rules_with_linked_diagnostics(language):
    page = _text(f"docs/{language}/models/coxph.md")
    reference = _text(f"docs/{language}/reference/coxph-diagnostics.md")
    assert "../reference/coxph-diagnostics.md" in page
    for token in ('full_host_transfer_performed_=True', "RuntimeWarning",
                  "STATGPU_COXPHCV_TWO_STAGE", "STATGPU_COXPHCV_SUCCESSIVE_HALVING",
                  'staged_safety_strategy="single_pass_exhaustive"'):
        assert token in page
    for field in ("selection_cache_hit", "requested_fit_device",
                  "fold_backend_preparation_count_this_call",
                  "candidate_target_host_transfer_count_this_call",
                  "candidate_target_host_vector_transfer_count_this_call",
                  "selection_origin_device", "candidate_preparation_origin_device",
                  "scoring_device", "effective_device",
                  "cv_full_host_transfer_performed_",
                  "final_refit_full_host_transfer_performed_", "orchestration_device_"):
        assert field in reference
    assert "fold_backend_preparation_count_this_call" not in page
    assert "candidate_target_host_transfer_count_this_call" not in page
    developer = _text("dev/references/coxph-implementation-and-evidence.md")
    assert "two\nactual target vectors" in developer
    assert "candidate_target_host_vector_transfer_count" in developer


@pytest.mark.parametrize("language", ("en", "cn"))
def test_umap_actions_precede_advanced_graph_and_force_equations(language):
    text = _text(f"docs/{language}/unsupervised/umap.md")
    advanced = "## Advanced: graph and layout equations" if language == "en" else "## 进阶：近邻图与布局公式"
    first, detail = text.split(advanced, 1)
    assert "<!-- learner-example: umap -->" in first
    for token in ('n_components=1', 'nn_method="nndescent"', "NumPy 2", "float64",
                  'init="random"', "api-reference.md#umap"):
        assert token in first
    assert "n_samples * negative_sample_rate" not in first
    assert "n_samples * negative_sample_rate" in detail
    for formula in (r"\rho_i=d_{i1}", r"10^{-12}",
                    r"w_{ij}=v_{ij}+v_{ji}-v_{ij}v_{ji}", r"q_{ij}^2(y_i-y_j)"):
        assert formula in detail
    assert "smooth_knn_dist" in detail


@pytest.mark.parametrize("language", ("en", "cn"))
def test_mixture_example_and_cautions_precede_em_derivation(language):
    text = _text(f"docs/{language}/unsupervised/gaussian-mixture.md")
    advanced = "## Advanced: EM updates" if language == "en" else "## 进阶：EM 更新公式"
    first, detail = text.split(advanced, 1)
    for token in ("<!-- learner-example: gaussian-mixture -->", "reg_covar=0",
                  "lower_bound_", "score(X)", "converged_", "api-reference.md#gaussianmixture"):
        assert token in first
    assert r"\ell(\theta)" in first
    for formula in (r"a_{ik}", r"r_{ik}", r"n_k", r"\mu_k", r"\Sigma", r"\sigma_{kj}^{2}"):
        assert formula in detail
    anchor = "estimating-equation" if language == "en" else "估计方程"
    assert f'id="{anchor}"' in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_spline_accuracy_heading_does_not_imply_a_mode_parameter(language):
    text = _text(f"docs/{language}/models/splines.md")
    assert "## strict / approx" not in text
    anchor = "strict--approx-difference" if language == "en" else "strict--approx-区别"
    assert f'id="{anchor}"' in text
    for token in ("natural_cubic_spline_basis", "cyclic_cubic_spline_basis", "1e6",
                  "boundary_lo", "boundary_hi", "extrapolation", "knots_"):
        assert token in text
