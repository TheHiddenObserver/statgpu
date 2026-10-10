"""Keep model comparisons user-facing and developer validation catalogs separate."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "linear-regression", "ridge", "lasso", "elastic-net", "knockoff", "nonparametric",
)


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("model", MODELS)
def test_model_guides_link_to_validation_reference_without_script_catalogs(language, model):
    text = (ROOT / f"docs/{language}/models/{model}.md").read_text(encoding="utf-8")
    assert "dev/references/model-validation.md#" in text
    assert not re.search(r"dev/(?:tests|benchmarks|comparisons)/", text)
    assert not re.search(r"`test_[a-z_]+`", text)


def test_developer_reference_preserves_relocated_validation_entries():
    text = (ROOT / "dev/references/model-validation.md").read_text(encoding="utf-8")
    for name in (
        "test_external_consistency.py",
        "test_linear_estimation_and_inference_match_statsmodels",
        "test_linear_robust_covariance_matches_statsmodels",
        "test_linear_robust_covariance_gpu_matches_statsmodels",
        "test_linear_hac_covariance_matches_statsmodels",
        "test_ridge_weighted_consistency.py",
        "validate_post_selection_ols_gpu.py",
        "validate_gaussian_residual_bootstrap_gpu.py",
        "benchmark_lasso_inference_gpu_vs_cpu.py",
        "benchmark_lasso_cpu_gpu_tol.py",
        "compare_lasso_kkt_stopping.py",
        "test_lasso_debiased_inference.py",
        "test_nodewise_alpha_inference_contract.py",
        "test_post_selection_ols_inference_api.py",
        "test_penalized_solver_api_cleanup.py",
        "benchmark_knockoff_fixedx.py",
        "benchmark_knockoff_vs_baselines.py",
        "benchmark_knockoff_same_xk_parity.py",
        "benchmark_kde_vs_scipy.py",
        "benchmark_kernel_regression_vs_statsmodels.py",
        "benchmark_nonparametric_vs_r.py",
        "benchmark_nonparametric_comparison_suite.py",
    ):
        assert name in text, name
    for target in re.findall(r"\]\((\.\./[^)#]+)\)", text):
        assert (ROOT / "dev/references" / target).resolve().is_file(), target


@pytest.mark.parametrize("language", ["en", "cn"])
def test_user_comparison_settings_remain_in_model_guides(language):
    model_dir = ROOT / f"docs/{language}/models"
    linear = (model_dir / "linear-regression.md").read_text(encoding="utf-8")
    ridge = (model_dir / "ridge.md").read_text(encoding="utf-8")
    elastic = (model_dir / "elastic-net.md").read_text(encoding="utf-8")
    nonparametric = (model_dir / "nonparametric.md").read_text(encoding="utf-8")
    assert "statsmodels.OLS" in linear and "WLS" in linear
    assert "sklearn_alpha" in ridge and "statgpu_alpha" in ridge
    assert "l1_ratio=0" in elastic and "Ridge" in elastic
    assert "gaussian_kde" in nonparametric and "statsmodels" in nonparametric
