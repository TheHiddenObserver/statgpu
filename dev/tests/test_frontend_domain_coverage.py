from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_SOURCE_DATE = date(2026, 6, 1)
TARGET_CATEGORIES = {
    "robust_quantile",
    "unsupervised",
    "ordered",
    "nonparametric",
    "panel",
    "covariance",
    "anova",
}


@pytest.fixture(scope="module")
def canonical_output():
    from dev.benchmarks.frontend_data.cli import generate
    from dev.benchmarks.frontend_data.registry import load_manifest

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    output, report, inventory = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=manifest,
        strict_sources=True,
    )
    return output, report, inventory, manifest


def test_published_categories_have_runs(canonical_output):
    output, _, _, _ = canonical_output
    counts = {
        category: sum(category in run["category_ids"] for run in output["runs"])
        for category in TARGET_CATEGORIES
    }
    assert all(count > 0 for count in counts.values()), counts


def test_dashboard_uses_only_june_2026_or_later_sources(canonical_output):
    output, _, _, manifest = canonical_output
    assert manifest["minimum_source_date"] == "2026-06-01"
    assert len(manifest["sources"]) == 8

    manifest_dates = {
        source["source_id"]: date.fromisoformat(source["source_date"])
        for source in manifest["sources"]
    }
    assert all(source_date >= MINIMUM_SOURCE_DATE for source_date in manifest_dates.values())

    registered_ids = set(manifest_dates)
    generated_ids = {run["source"]["source_id"] for run in output["runs"]}
    assert generated_ids <= registered_ids
    assert all("202604" not in source_id for source_id in generated_ids)
    assert all(
        not run["source"].get("date", "").startswith("2026-04")
        for run in output["runs"]
    )


def test_recent_linear_results_are_visible(canonical_output):
    output, _, _, _ = canonical_output
    recent = [
        run
        for run in output["runs"]
        if "linear_models" in run["category_ids"]
        and run["source"]["file"] in {
            "penalized_glm_perf_20260622.json",
            "glm_solver_20260623.json",
        }
        and run.get("loss") == "squared_error"
    ]
    assert recent, "2026-06-22/23 squared-error runs must appear under linear_models"
    assert {run["backend"] for run in recent if run["backend"]} >= {"numpy", "cupy", "torch"}


def test_quantile_inference_has_all_backends(canonical_output):
    output, _, _, _ = canonical_output
    quantile_inference = [
        run
        for run in output["runs"]
        if run["model_id"] == "QuantileRegression"
        and run.get("variant") in {"kernel", "bootstrap"}
        and run.get("metrics", {}).get("inference")
    ]
    assert {run["backend"] for run in quantile_inference} >= {"numpy", "cupy", "torch"}
    assert {run.get("variant") for run in quantile_inference} == {"kernel", "bootstrap"}


def test_anova_has_all_functions_backends_and_scipy(canonical_output):
    output, _, _, _ = canonical_output
    anova_runs = [run for run in output["runs"] if "anova" in run["category_ids"]]
    assert anova_runs
    assert {run["model_id"] for run in anova_runs} >= {
        "OneWayANOVA",
        "TwoWayANOVA",
        "WelchANOVA",
        "TukeyHSD",
        "BonferroniCorrection",
    }
    statgpu_backends = {
        run["backend"]
        for run in anova_runs
        if run["framework"] == "statgpu" and run["backend"]
    }
    assert statgpu_backends >= {"numpy", "cupy", "torch"}
    assert any(run["framework"] == "scipy" for run in anova_runs)
    assert any(
        run["model_id"] == "OneWayANOVA"
        and run.get("metrics", {}).get("validation")
        for run in anova_runs
    )


def test_missing_domain_sources_are_manifest_registered(canonical_output):
    _, _, _, manifest = canonical_output
    parsers = {source["parser"] for source in manifest["sources"]}
    assert {
        "loss_functions_benchmark",
        "ordered_inference_benchmark",
        "unsupervised_benchmark",
        "new_modules_with_anova_benchmark",
        "p2_benchmark",
    } <= parsers


def test_current_external_frameworks_are_reachable(canonical_output):
    output, _, _, _ = canonical_output
    frameworks = {run["framework"] for run in output["runs"]}
    assert {"linearmodels", "pygam", "sklearn", "statsmodels", "scipy"} <= frameworks
    assert frameworks.isdisjoint({"glmnet", "lifelines", "scikit_survival", "knockpy"})


def test_domain_models_are_present(canonical_output):
    output, _, _, _ = canonical_output
    model_ids = {model["model_id"] for model in output["models"]}
    assert {
        "QuantileRegression",
        "RobustRegression",
        "OrderedLogit",
        "OrderedProbit",
        "PCA",
        "KMeans",
        "PanelOLS",
        "RandomEffects",
        "GAM",
        "EmpiricalCovariance",
        "OneWayANOVA",
        "TwoWayANOVA",
        "WelchANOVA",
        "TukeyHSD",
        "BonferroniCorrection",
    } <= model_ids
