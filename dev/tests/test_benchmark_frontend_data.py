from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def generator():
    from dev.benchmarks.generate_benchmark_data import generate

    return generate


@pytest.fixture(scope="module")
def results_dir():
    return REPO_ROOT / "results"


@pytest.fixture(scope="module")
def manifest():
    from dev.benchmarks.frontend_data.registry import load_manifest

    return load_manifest(REPO_ROOT)


class TestGenerateBenchmarkData:
    """Transitional discovery-mode generator tests."""

    def test_generate_produces_valid_output(self, generator, results_dir):
        output, report, inventory = generator(results_dir, deterministic=True)
        assert output["schema_version"] == "1.1.0"
        assert output["runs"]
        assert report["runs_generated"] == len(output["runs"])
        assert inventory["catalog_total"] >= 1

    def test_all_runs_have_required_fields(self, generator, results_dir):
        output, _, _ = generator(results_dir, deterministic=True)
        for run in output["runs"]:
            assert run["run_id"]
            assert run["env_id"]
            assert run["category_ids"]
            assert run["model_id"]
            assert run["framework"]
            assert "backend" in run
            assert run["scale"]["scale_key"]
            assert run["source"]["file"]
            assert run["metrics"]

    def test_no_duplicate_run_ids(self, generator, results_dir):
        output, _, _ = generator(results_dir, deterministic=True)
        run_ids = [run["run_id"] for run in output["runs"]]
        assert len(run_ids) == len(set(run_ids))

    def test_framework_backend_consistency(self, generator, results_dir):
        output, _, _ = generator(results_dir, deterministic=True)
        external = {
            framework["framework_id"]
            for framework in output["frameworks"]
            if framework["external"]
        }
        for run in output["runs"]:
            if run["framework"] in external:
                assert run["backend"] is None

    def test_metrics_are_finite_and_valid(self, generator, results_dir):
        output, _, _ = generator(results_dir, deterministic=True)

        def visit(value):
            if isinstance(value, dict):
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)
            elif isinstance(value, float):
                assert math.isfinite(value)

        visit(output)
        for run in output["runs"]:
            timing = run.get("metrics", {}).get("timing")
            if timing:
                assert timing["fit_time_ms"] > 0

    def test_penalized_glm_parser_has_three_backends(self, generator, results_dir):
        output, _, _ = generator(results_dir, deterministic=True)
        rows = [
            run
            for run in output["runs"]
            if run["source"]["file"] == "penalized_glm_bench_perf_2026-06-22.json"
            and run["framework"] == "statgpu"
        ]
        if rows:
            assert {run["backend"] for run in rows} == {"numpy", "cupy", "torch"}

    def test_solver_parser_has_dispatch_and_manual_runs(self, generator, results_dir):
        output, _, _ = generator(results_dir, deterministic=True)
        rows = [
            run
            for run in output["runs"]
            if run["source"]["file"] == "glm_solver_benchmark_2026-06-23.json"
        ]
        if rows:
            kinds = {run.get("solver_kind") for run in rows}
            assert {"dispatch", "manual"} <= kinds

    def test_category_and_scale_contracts(self, generator, results_dir):
        output, _, _ = generator(results_dir, deterministic=True)
        for run in output["runs"]:
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
        assert len(manifest["sources"]) == 9
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
        assert report["files_seen"] == 9
        assert report["files_parsed"] == 9
        assert inventory["registered_sources"] == 9
        assert inventory["available_sources"] == 9
        assert inventory["parsed_sources"] == 9
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
        assert {"statgpu", "sklearn", "statsmodels", "linearmodels", "pygam"} <= framework_ids
        assert framework_ids.isdisjoint(
            {"glmnet", "lifelines", "scikit_survival", "knockpy"}
        )

    def test_canonical_models(self, generator, manifest, results_dir):
        output, _, _ = generator(results_dir, manifest=manifest)
        model_ids = {model["model_id"] for model in output["models"]}
        assert {"CoxPH", "QuantileRegression", "PanelOLS"} <= model_ids
        assert {
            "RidgeCV",
            "LassoCV",
            "ElasticNetCV",
            "LogisticRegressionCV",
            "PenalizedGLM_CV",
            "CoxPHCV",
        } <= model_ids
        # The old April non-CV ElasticNet benchmark remains excluded; only the
        # current six-family CV package is promoted by #91.
        assert "ElasticNet" not in model_ids


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


def test_manifest_registry_allows_unhashed_sources(tmp_path):
    from dev.benchmarks.frontend_data.registry import validate_manifest_sources

    source = tmp_path / "sample.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "sources": [
            {
                "source_id": "sample",
                "path": "sample.json",
                "parser": "comprehensive_validation",
                "env_id": "env",
                "required": True,
                "source_date": "2026-06-01",
            }
        ]
    }
    issues = validate_manifest_sources(tmp_path, manifest, strict_sources=False)
    assert not any(issue["severity"] == "error" for issue in issues)


def test_manifest_date_policy_rejects_old_and_undated_sources(tmp_path):
    from dev.benchmarks.frontend_data.registry import validate_manifest_sources

    (tmp_path / "old.json").write_text("{}", encoding="utf-8")
    (tmp_path / "undated.json").write_text("{}", encoding="utf-8")
    manifest = {
        "minimum_source_date": "2026-06-01",
        "sources": [
            {
                "source_id": "old",
                "path": "old.json",
                "parser": "comprehensive_validation",
                "env_id": "env",
                "required": True,
                "source_date": "2026-05-31",
            },
            {
                "source_id": "undated",
                "path": "undated.json",
                "parser": "comprehensive_validation",
                "env_id": "env",
                "required": True,
            },
        ],
    }
    issues = validate_manifest_sources(tmp_path, manifest, strict_sources=False)
    codes = {issue["code"] for issue in issues}
    assert "source_before_minimum_date" in codes
    assert "missing_source_date" in codes


def test_direct_manifest_generation_rejects_pre_june_sources(tmp_path):
    from dev.benchmarks.generate_benchmark_data import generate

    source = tmp_path / "old.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "minimum_source_date": "2026-06-01",
        "environments": {"env": {"label": "env", "gpu": "", "cpu": ""}},
        "frameworks": {},
        "comparisons": {},
        "sources": [
            {
                "source_id": "old",
                "path": "old.json",
                "parser": "comprehensive_validation",
                "env_id": "env",
                "required": True,
                "source_date": "2026-05-31",
            }
        ],
    }
    with pytest.raises(ValueError, match="source_before_minimum_date"):
        generate(tmp_path, manifest=manifest, strict_sources=True)


def test_strict_mode_requires_hash_for_required_source(tmp_path):
    from dev.benchmarks.frontend_data.registry import validate_manifest_sources

    source = tmp_path / "sample.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "minimum_source_date": "2026-06-01",
        "sources": [
            {
                "source_id": "sample",
                "path": "sample.json",
                "parser": "comprehensive_validation",
                "env_id": "env",
                "required": True,
                "source_date": "2026-06-01",
            }
        ],
    }
    issues = validate_manifest_sources(tmp_path, manifest, strict_sources=True)
    assert any(issue["code"] == "missing_source_hash" for issue in issues)


def test_load_manifest_propagates_invalid_json(tmp_path):
    from dev.benchmarks.frontend_data.registry import load_manifest

    (tmp_path / "dev" / "benchmarks").mkdir(parents=True)
    (tmp_path / "dev" / "benchmarks" / "frontend_sources.json").write_text(
        "{ invalid", encoding="utf-8"
    )
    with pytest.raises(json.JSONDecodeError):
        load_manifest(tmp_path)
