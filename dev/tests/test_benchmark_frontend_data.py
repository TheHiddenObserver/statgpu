from __future__ import annotations
"""Tests for the benchmark frontend data generator."""

import json
import math
import sys
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


@pytest.fixture
def generator():
    from dev.benchmarks.generate_benchmark_data import generate

    return generate


@pytest.fixture
def results_dir():
    return repo_root / "results"


@pytest.fixture
def manifest():
    from dev.benchmarks.frontend_data.registry import load_manifest

    return load_manifest(repo_root)


class TestGenerateBenchmarkData:
    """Integration tests for the June-only fallback registry."""

    def test_generate_produces_valid_output(self, generator, results_dir):
        output, report, inventory = generator(results_dir)
        assert output["schema_version"] == "1.1.0"
        assert len(output["environments"]) >= 1
        assert len(output["categories"]) == 12
        assert output["models"]
        assert output["runs"]
        assert output["frameworks"]
        assert output["comparisons"]
        assert output["meta"]["generation_id"]
        assert report["files_parsed"] == 2
        assert report["runs_generated"] == len(output["runs"])
        assert inventory["registered_sources"] == 2

    def test_all_runs_have_required_fields(self, generator, results_dir):
        output, _, _ = generator(results_dir)
        required = {
            "run_id",
            "env_id",
            "category_ids",
            "model_id",
            "framework",
            "backend",
            "scale",
            "source",
            "comparison_id",
            "case_id",
            "method_config_id",
        }
        for run in output["runs"]:
            assert required <= run.keys(), run.get("run_id", "?")
            assert "source_id" in run["source"]

    def test_no_duplicate_run_ids(self, generator, results_dir):
        output, _, _ = generator(results_dir)
        run_ids = [run["run_id"] for run in output["runs"]]
        assert len(run_ids) == len(set(run_ids))

    def test_framework_backend_consistency(self, generator, results_dir):
        output, _, _ = generator(results_dir)
        for run in output["runs"]:
            if run["framework"] == "statgpu":
                assert run["backend"] in {"numpy", "cupy", "torch"}
            else:
                assert run["backend"] is None

    def test_metrics_are_finite_and_valid(self, generator, results_dir):
        output, _, _ = generator(results_dir)

        def check_finite(value, path=""):
            if isinstance(value, float):
                assert not math.isnan(value), path
                assert not math.isinf(value), path
            elif isinstance(value, dict):
                for key, item in value.items():
                    check_finite(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    check_finite(item, f"{path}[{index}]")

        for run in output["runs"]:
            timing = run.get("metrics", {}).get("timing")
            if timing:
                assert timing["fit_time_ms"] >= 0
                if "std_ms" in timing:
                    assert timing["std_ms"] >= 0
                if "min_ms" in timing and "max_ms" in timing:
                    assert timing["min_ms"] <= timing["max_ms"]
            speedup = run.get("metrics", {}).get("speedup")
            if speedup:
                assert speedup["value"] > 0
                assert "reference_backend" in speedup
                assert "reference_framework" in speedup
                assert "reported_semantics" in speedup
            check_finite(run, run["run_id"])

    def test_penalized_glm_parser_has_three_backends(self, generator, results_dir):
        output, _, _ = generator(results_dir)
        runs = [run for run in output["runs"] if "penalized_glm" in run["category_ids"]]
        assert runs
        assert {run["backend"] for run in runs if run["backend"]} >= {
            "numpy",
            "cupy",
            "torch",
        }
        assert any(run.get("loss") == "squared_error" for run in runs)

    def test_solver_parser_has_dispatch_and_manual_runs(self, generator, results_dir):
        output, _, _ = generator(results_dir)
        runs = [
            run
            for run in output["runs"]
            if run["source"]["parser"] == "parse_glm_solver_benchmark_v1"
            and run.get("metrics", {}).get("speedup")
        ]
        assert runs
        assert {run.get("solver_kind") for run in runs} >= {"dispatch", "manual"}

    def test_category_and_scale_contracts(self, generator, results_dir):
        output, _, _ = generator(results_dir)
        for run in output["runs"]:
            assert isinstance(run["category_ids"], list)
            assert run["category_ids"]
            scale = run["scale"]
            assert {"scale_key", "n_samples", "n_features", "label"} <= scale.keys()
            assert scale["n_samples"] > 0
            assert scale["n_features"] > 0

    def test_catalog_total_is_computed(self, generator, results_dir):
        _, _, inventory = generator(results_dir)
        expected = sum(
            1
            for path in results_dir.rglob("*.json")
            if "benchmark_frontend_sources" not in path.relative_to(results_dir).parts
        )
        assert inventory["catalog_total"] == expected


class TestManifestMode:
    """Integration tests for the canonical June-or-later manifest."""

    def test_manifest_loads_with_exact_current_sources(self, manifest):
        assert manifest is not None
        assert manifest["minimum_source_date"] == "2026-06-01"
        assert len(manifest["sources"]) == 8
        assert all(source.get("source_date") for source in manifest["sources"])

    def test_canonical_generate(self, generator, manifest, results_dir):
        output, report, inventory = generator(
            results_dir,
            manifest=manifest,
            deterministic=True,
            strict_sources=True,
        )
        assert output["runs"]
        assert output["schema_version"] == "1.1.0"
        assert output["frameworks"]
        assert output["comparisons"]
        assert output["meta"]["generation_id"]
        assert report["files_seen"] == 8
        assert report["files_parsed"] == 8
        assert inventory["registered_sources"] == 8
        assert inventory["available_sources"] == 8
        assert inventory["parsed_sources"] == 8
        assert not any(
            run["source"]["source_id"].startswith("transitional:")
            for run in output["runs"]
        )

    def test_computed_speedups_reference_matching_timing_runs(
        self, generator, manifest, results_dir
    ):
        output, _, _ = generator(results_dir, manifest=manifest)
        runs_by_id = {run["run_id"]: run for run in output["runs"]}
        computed = [
            run
            for run in output["runs"]
            if run.get("metrics", {}).get("speedup", {}).get("reported_semantics")
            == "computed"
        ]
        assert computed
        for run in computed:
            speedup = run["metrics"]["speedup"]
            reference = runs_by_id[speedup["reference_run_id"]]
            assert reference["framework"] == speedup["reference_framework"]
            assert reference["backend"] == speedup["reference_backend"]
            expected = (
                reference["metrics"]["timing"]["fit_time_ms"]
                / run["metrics"]["timing"]["fit_time_ms"]
            )
            assert speedup["value"] == pytest.approx(expected, rel=1e-4, abs=1e-4)

    def test_canonical_frameworks_present(self, generator, manifest, results_dir):
        output, _, _ = generator(results_dir, manifest=manifest)
        framework_ids = {item["framework_id"] for item in output["frameworks"]}
        assert {"statgpu", "sklearn", "linearmodels", "pygam"} <= framework_ids
        assert framework_ids.isdisjoint(
            {"glmnet", "statsmodels", "lifelines", "scikit_survival", "knockpy"}
        )

    def test_canonical_models(self, generator, manifest, results_dir):
        output, _, _ = generator(results_dir, manifest=manifest)
        model_ids = {model["model_id"] for model in output["models"]}
        assert {"CoxPH", "QuantileRegression", "PanelOLS"} <= model_ids
        assert {"LassoCV", "ElasticNet"}.isdisjoint(model_ids)


def test_parse_family_penalty_solver_handles_underscored_solver():
    from dev.benchmarks.frontend_data.canonical import parse_family_penalty_solver

    assert parse_family_penalty_solver("inverse_gaussian_group_lasso_fista_bb") == (
        "inverse_gaussian",
        "group_lasso",
        "fista_bb",
    )


def test_make_scale_label_preserves_fractional_thousands():
    from dev.benchmarks.frontend_data.canonical import make_scale_label

    assert make_scale_label(1500, 20) == "1.5K×20"


def test_manifest_registry_allows_unhashed_sources():
    from dev.benchmarks.frontend_data.registry import build_registry_from_manifest

    registry = build_registry_from_manifest(
        {
            "sources": [
                {
                    "path": "results/example.json",
                    "parser": "knockoff_benchmark",
                    "env_id": "test",
                    "source_id": "example-20260101-000000000000",
                }
            ]
        }
    )
    assert "sha256" not in registry["results/example.json"]


def test_manifest_date_policy_rejects_old_and_undated_sources():
    from dev.benchmarks.frontend_data.registry import validate_manifest_source_dates

    with pytest.raises(ValueError, match="missing source_date"):
        validate_manifest_source_dates(
            {"minimum_source_date": "2026-06-01", "sources": [{"source_id": "x"}]}
        )
    with pytest.raises(ValueError, match="minimum allowed date"):
        validate_manifest_source_dates(
            {
                "minimum_source_date": "2026-06-01",
                "sources": [{"source_id": "x", "source_date": "2026-04-30"}],
            }
        )


def test_strict_mode_requires_hash_for_required_source(generator, tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "catalog_total": 1,
        "environments": {"test": {"label": "Test", "gpu": "none", "cpu": "test"}},
        "sources": [
            {
                "path": str(source),
                "parser": "knockoff_benchmark",
                "env_id": "test",
                "source_id": "example-20260101-000000000000",
                "required": True,
            }
        ],
    }
    with pytest.raises(ValueError, match="missing SHA256"):
        generator(tmp_path, manifest=manifest, strict_sources=True)


def test_load_manifest_propagates_invalid_json(tmp_path):
    from dev.benchmarks.frontend_data.registry import load_manifest

    manifest_dir = tmp_path / "dev" / "benchmarks"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "frontend_sources.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_manifest(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
