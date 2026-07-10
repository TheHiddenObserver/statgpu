"""Tests for the benchmark frontend data generator."""
import json
import sys
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import pytest  # noqa: E402


@pytest.fixture
def generator():
    from dev.benchmarks.generate_benchmark_data import generate
    return generate


@pytest.fixture
def results_dir():
    return repo_root / "results"


class TestGenerateBenchmarkData:
    """Integration tests for the data generator."""

    def test_generate_produces_valid_output(self, generator, results_dir):
        output, report = generator(results_dir)
        assert output["schema_version"] == "1.1.0"
        assert "generated" in output
        assert len(output["environments"]) >= 1
        assert len(output["categories"]) == 12
        assert len(output["models"]) >= 1
        assert len(output["runs"]) > 0
        assert "frameworks" in output
        assert "comparisons" in output
        assert "generation_id" in output["meta"]
        assert report["files_parsed"] >= 1
        assert report["runs_generated"] == len(output["runs"])

    def test_all_runs_have_required_fields(self, generator, results_dir):
        output, _ = generator(results_dir)
        required = ["run_id", "env_id", "category_ids", "model_id", "framework", "backend", "scale", "source",
                    "comparison_id", "case_id", "method_config_id"]
        for run in output["runs"]:
            for field in required:
                assert field in run, f"{run.get('run_id', '?')} missing {field}"
            assert "source_id" in run["source"], f"{run['run_id']} source missing source_id"

    def test_no_duplicate_run_ids(self, generator, results_dir):
        output, _ = generator(results_dir)
        ids = [r["run_id"] for r in output["runs"]]
        assert len(ids) == len(set(ids)), f"Found {len(ids) - len(set(ids))} duplicate run_ids"

    def test_framework_backend_consistency(self, generator, results_dir):
        output, _ = generator(results_dir)
        for run in output["runs"]:
            fw = run["framework"]
            bk = run["backend"]
            if fw == "statgpu":
                assert bk in ("numpy", "cupy", "torch"), f"{run['run_id']}: statgpu has invalid backend {bk}"
            else:
                assert bk is None, f"{run['run_id']}: external framework {fw} has backend {bk}"

    def test_timing_metrics_valid(self, generator, results_dir):
        output, _ = generator(results_dir)
        for run in output["runs"]:
            t = run.get("metrics", {}).get("timing")
            if t is None:
                continue
            assert t["fit_time_ms"] >= 0, f"{run['run_id']}: negative fit_time_ms"
            if "std_ms" in t:
                assert t["std_ms"] >= 0, f"{run['run_id']}: negative std_ms"
            if "min_ms" in t and "max_ms" in t:
                assert t["min_ms"] <= t["max_ms"], f"{run['run_id']}: min_ms > max_ms"

    def test_speedup_metrics_valid(self, generator, results_dir):
        output, _ = generator(results_dir)
        for run in output["runs"]:
            s = run.get("metrics", {}).get("speedup")
            if s is None:
                continue
            assert s["value"] >= 0, f"{run['run_id']}: negative speedup"
            assert "reference_backend" in s, f"{run['run_id']}: speedup missing reference_backend"
            assert "reference_framework" in s, f"{run['run_id']}: speedup missing reference_framework"
            assert "reported_semantics" in s, f"{run['run_id']}: speedup missing reported_semantics"

    def test_no_nan_or_inf(self, generator, results_dir):
        import math
        output, _ = generator(results_dir)

        def check(obj, path=""):
            if isinstance(obj, float):
                assert not math.isnan(obj), f"NaN at {path}"
                assert not math.isinf(obj), f"Inf at {path}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check(v, f"{path}[{i}]")

        for run in output["runs"]:
            check(run, run.get("run_id", "?"))

    def test_parse_penalized_glm_bench_perf(self, generator, results_dir):
        """Verify penalized_glm_bench_perf parser produces expected runs."""
        output, _ = generator(results_dir)
        # Check we have penalized_glm runs with timing
        glm_runs = [r for r in output["runs"] if "penalized_glm" in r["category_ids"]]
        assert len(glm_runs) > 0, "Should have penalized_glm runs"

        timing_runs = [r for r in glm_runs if "timing" in r.get("metrics", {})]
        assert len(timing_runs) > 0, "Should have timing data for penalized_glm"

        # Check families
        families = set(r.get("loss", "") for r in timing_runs)
        assert "logistic" in families or "squared_error" in families

        # Check backends
        backends = set(r["backend"] for r in timing_runs if r["backend"])
        for bk in ("numpy", "cupy", "torch"):
            assert bk in backends, f"Should have {bk} backend data"

    def test_parse_glm_solver_benchmark(self, generator, results_dir):
        """Verify glm_solver_benchmark parser produces speedup runs."""
        output, _ = generator(results_dir)
        speedup_runs = [
            r for r in output["runs"]
            if "speedup" in r.get("metrics", {})
            and r["source"]["parser"] == "parse_glm_solver_benchmark_v1"
        ]
        assert len(speedup_runs) > 0, "Should have solver speedup runs"

        # Check solver_kinds
        kinds = set(r.get("solver_kind") for r in speedup_runs)
        assert "dispatch" in kinds, "Should have dispatch (auto) solver runs"
        assert "manual" in kinds, "Should have manual solver runs"

    def test_parse_elasticnet_benchmark(self, generator, results_dir):
        """Verify elasticnet benchmark parser produces glmnet/external runs."""
        output, _ = generator(results_dir)
        ext_runs = [r for r in output["runs"] if r["framework"] == "glmnet"]
        assert len(ext_runs) > 0, "Should have glmnet comparison runs"

        # glmnet runs should have timing
        for r in ext_runs:
            assert "timing" in r.get("metrics", {}), f"glmnet run {r['run_id']} should have timing"

    def test_category_ids_is_array(self, generator, results_dir):
        output, _ = generator(results_dir)
        for run in output["runs"]:
            assert isinstance(run["category_ids"], list), f"{run['run_id']}: category_ids should be list"
            assert len(run["category_ids"]) >= 1, f"{run['run_id']}: category_ids should have at least 1 entry"

    def test_scale_has_all_fields(self, generator, results_dir):
        output, _ = generator(results_dir)
        for run in output["runs"]:
            s = run["scale"]
            assert "scale_key" in s
            assert "n_samples" in s
            assert "n_features" in s
            assert "label" in s
            assert s["n_samples"] > 0
            assert s["n_features"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
