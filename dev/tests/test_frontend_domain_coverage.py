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
    "survival",
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
        and run["source"]["file"]
        in {
            "penalized_glm_perf_20260622.json",
            "glm_solver_20260623.json",
        }
        and run.get("loss") == "squared_error"
    ]
    assert recent, "2026-06-22/23 squared-error runs must appear under linear_models"
    assert {run["backend"] for run in recent if run["backend"]} >= {
        "numpy",
        "cupy",
        "torch",
    }


def test_quantile_inference_has_all_backends(canonical_output):
    output, _, _, _ = canonical_output
    quantile_inference = [
        run
        for run in output["runs"]
        if run["model_id"] == "QuantileRegression"
        and run.get("variant") in {"kernel", "bootstrap"}
        and run.get("metrics", {}).get("inference")
    ]
    assert {run["backend"] for run in quantile_inference} >= {
        "numpy",
        "cupy",
        "torch",
    }
    assert {run.get("variant") for run in quantile_inference} == {
        "kernel",
        "bootstrap",
    }


def test_pr74_exposes_all_nonquantile_inference_methods(canonical_output):
    output, _, _, _ = canonical_output
    expected_variants = {
        "hc0-sandwich",
        "oracle-inference",
        "bootstrap-inference",
    }
    rows = [
        run
        for run in output["runs"]
        if run["source"]["file"] == "ordered_inference_pr74.json"
        and run.get("variant") in expected_variants
    ]

    assert len(rows) == 18
    assert {run["variant"] for run in rows} == expected_variants
    assert {run["backend"] for run in rows} == {"numpy", "cupy", "torch"}
    assert {run["scale"]["label"] for run in rows} == {"500×5", "2K×10"}
    assert {run["source"]["parser_version"] for run in rows} == {"2.0"}
    assert all(
        run.get("parameters", {}).get("timing_scope") == "fit_plus_inference"
        for run in rows
    )
    assert all(run.get("metrics", {}).get("inference") for run in rows)

    method_contract = {
        "hc0-sandwich": ("PenalizedLogisticRegression", "logistic", "l2"),
        "oracle-inference": ("PenalizedLogisticRegression", "logistic", "scad"),
        "bootstrap-inference": ("PenalizedLinearRegression", "squared_error", "l1"),
    }
    for variant, (model_id, loss, penalty) in method_contract.items():
        variant_rows = [run for run in rows if run["variant"] == variant]
        assert {run["model_id"] for run in variant_rows} == {model_id}
        assert {run["loss"] for run in variant_rows} == {loss}
        assert {run["penalty"] for run in variant_rows} == {penalty}


def test_survival_includes_breslow_timing_speedup_and_validation(canonical_output):
    output, _, _, _ = canonical_output
    breslow = [
        run
        for run in output["runs"]
        if "survival" in run["category_ids"]
        and run["model_id"] == "CoxPH"
        and run.get("variant") == "breslow"
        and run["source"]["file"] == "loss_functions_20260623.json"
    ]
    assert len(breslow) == 20
    assert {run["framework"] for run in breslow} == {"statgpu", "statsmodels"}
    assert {
        run["backend"]
        for run in breslow
        if run["framework"] == "statgpu"
    } == {"numpy", "cupy", "torch"}
    assert all(run.get("metrics", {}).get("timing") for run in breslow)

    statgpu = [run for run in breslow if run["framework"] == "statgpu"]
    assert all(
        run.get("metrics", {}).get("speedup", {}).get("reference_framework")
        == "statsmodels"
        for run in statgpu
    )
    assert all(
        run["metrics"]["speedup"]["reported_semantics"] == "reported_by_runner"
        for run in statgpu
    )
    validated_backends = {
        run["backend"]
        for run in statgpu
        if run.get("metrics", {}).get("validation")
    }
    assert validated_backends == {"numpy", "cupy"}


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


def test_gam_exposes_both_complete_comparison_matrices(canonical_output):
    output, _, _, _ = canonical_output
    gam_runs = [
        run
        for run in output["runs"]
        if run["model_id"] == "GAM"
        and run.get("variant") in {"pygam-comparison", "aligned-pygam"}
        and run["source"]["file"] == "new_modules_full_20260624.json"
    ]
    assert len(gam_runs) == 24
    assert {run["variant"] for run in gam_runs} == {
        "pygam-comparison",
        "aligned-pygam",
    }
    assert {run["scale"]["label"] for run in gam_runs} == {
        "1K×3",
        "10K×5",
        "100K×10",
    }
    assert {run["solver"] for run in gam_runs} == {"fixed_lam"}
    assert {run["source"]["parser_version"] for run in gam_runs} == {"1.4"}

    for variant in ("pygam-comparison", "aligned-pygam"):
        variant_rows = [run for run in gam_runs if run["variant"] == variant]
        assert len(variant_rows) == 12
        for label in ("1K×3", "10K×5", "100K×10"):
            rows = [run for run in variant_rows if run["scale"]["label"] == label]
            assert len(rows) == 4
            assert {run["framework"] for run in rows} == {"statgpu", "pygam"}
            assert {
                run["backend"]
                for run in rows
                if run["framework"] == "statgpu"
            } == {"numpy", "cupy", "torch"}
            assert all(run.get("metrics", {}).get("timing") for run in rows)
            assert all(
                run.get("metrics", {}).get("speedup", {}).get("reference_framework")
                == "pygam"
                for run in rows
                if run["framework"] == "statgpu"
            )


def test_panel_exposes_complete_aligned_scale_matrix(canonical_output):
    output, _, _, _ = canonical_output
    panel_runs = [
        run
        for run in output["runs"]
        if "panel" in run["category_ids"]
        and run.get("variant") == "aligned-linearmodels"
        and run["source"]["file"] == "new_modules_full_20260624.json"
    ]
    assert len(panel_runs) == 16
    assert {run["scale"]["label"] for run in panel_runs} == {
        "10K×10",
        "100K×20",
    }
    assert {run["model_id"] for run in panel_runs} == {
        "PanelOLS",
        "RandomEffects",
    }

    for model_id in ("PanelOLS", "RandomEffects"):
        for label in ("10K×10", "100K×20"):
            rows = [
                run
                for run in panel_runs
                if run["model_id"] == model_id
                and run["scale"]["label"] == label
            ]
            assert len(rows) == 4
            assert {run["framework"] for run in rows} == {
                "statgpu",
                "linearmodels",
            }
            assert {
                run["backend"]
                for run in rows
                if run["framework"] == "statgpu"
            } == {"numpy", "cupy", "torch"}
            assert all(run.get("metrics", {}).get("timing") for run in rows)
            assert all(
                run.get("metrics", {}).get("speedup", {}).get("reference_framework")
                == "linearmodels"
                for run in rows
                if run["framework"] == "statgpu"
            )

    assert {run["source"]["parser_version"] for run in panel_runs} == {"1.4"}


def test_unsupervised_exposes_complete_source_matrix(canonical_output):
    output, _, _, _ = canonical_output
    rows = [
        run
        for run in output["runs"]
        if "unsupervised" in run["category_ids"]
        and run["source"]["file"] == "unsupervised_20260627.json"
    ]
    assert len(rows) == 131
    assert {run["source"]["parser_version"] for run in rows} == {"2.0"}

    expected_model_counts = {
        "PCA": 12,
        "KMeans": 12,
        "GaussianMixture": 12,
        "NMF": 12,
        "TruncatedSVD": 12,
        "IncrementalPCA": 12,
        "AgglomerativeClustering": 8,
        "DBSCAN": 24,
        "UMAP": 6,
        "TSNE": 3,
        "MiniBatchKMeans": 9,
        "MiniBatchNMF": 9,
    }
    assert {
        model_id: sum(run["model_id"] == model_id for run in rows)
        for model_id in expected_model_counts
    } == expected_model_counts

    pca = [run for run in rows if run["model_id"] == "PCA"]
    assert {run["scale"]["label"] for run in pca} == {
        "1K×20",
        "10K×50",
        "100K×50",
    }
    assert all(
        run["scale"]["n_features"] == 50
        for run in pca
        if run["scale"]["n_samples"] == 100_000
    )

    dbscan = [run for run in rows if run["model_id"] == "DBSCAN"]
    assert {run.get("variant") for run in dbscan} == {"10d", "50d"}
    assert {run["scale"]["label"] for run in dbscan} == {
        "1K×10",
        "10K×10",
        "100K×10",
        "1K×50",
        "10K×50",
        "100K×50",
    }


def test_generated_bundle_has_expected_complete_run_count(canonical_output):
    output, report, _, _ = canonical_output
    assert len(output["runs"]) == 1774
    assert report["runs_generated"] == 1774


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
        "CoxPH",
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
