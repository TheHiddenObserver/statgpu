from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_schema_pins_supported_version() -> None:
    schema_path = REPO_ROOT / "dev" / "benchmarks" / "benchmark_frontend_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"] == {"const": "1.1.0"}


def test_schema_rejects_unsupported_version() -> None:
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate, validate_against_schema

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None

    output, _, _ = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=manifest,
    )
    output["schema_version"] = "1.0.0"

    errors = validate_against_schema(output)
    assert any("schema_version" in error and "1.1.0" in error for error in errors)


def test_manifest_declares_every_generated_framework() -> None:
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None

    output, _, _ = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=manifest,
    )
    generated_frameworks = {run["framework"] for run in output["runs"]}

    assert generated_frameworks <= set(manifest["frameworks"])


def test_transitional_inventory_catalog_total_matches_results_tree() -> None:
    from dev.benchmarks.generate_benchmark_data import generate

    results_dir = REPO_ROOT / "results"
    _, _, inventory = generate(results_dir)
    expected = sum(
        1
        for path in results_dir.rglob("*.json")
        if "benchmark_frontend_sources" not in path.relative_to(results_dir).parts
    )

    assert inventory["catalog_total"] == expected


def test_public_validator_rejects_zero_timing_for_log_axis() -> None:
    from dev.benchmarks.generate_benchmark_data import validate_output

    output = {
        "runs": [
            {
                "run_id": "zero-timing",
                "env_id": "test",
                "category_ids": ["linear_models"],
                "model_id": "ExampleModel",
                "framework": "statgpu",
                "backend": "numpy",
                "scale": {
                    "scale_key": "n10_p2",
                    "n_samples": 10,
                    "n_features": 2,
                    "label": "10×2",
                },
                "source": {
                    "file": "example.json",
                },
                "metrics": {
                    "timing": {
                        "fit_time_ms": 0,
                    },
                },
            }
        ]
    }

    errors = validate_output(output)
    assert any(
        "timing.fit_time_ms must be a finite number > 0" in error
        for error in errors
    )


def _minimal_timing_output(fit_time_ms: object) -> dict:
    return {
        "runs": [
            {
                "run_id": "timing-regression",
                "env_id": "remote-p100",
                "category_ids": ["linear_models"],
                "model_id": "ExampleModel",
                "framework": "statgpu",
                "backend": "numpy",
                "scale": {},
                "source": {},
                "metrics": {"timing": {"fit_time_ms": fit_time_ms}},
            }
        ]
    }


def test_core_validator_rejects_nonpositive_and_nonnumeric_fit_time() -> None:
    from dev.benchmarks.frontend_data import validate_output

    for bad_value in (0, -1, "0", None, True):
        errors = validate_output(_minimal_timing_output(bad_value))
        assert any(
            "timing.fit_time_ms must be a finite number > 0" in error
            for error in errors
        )


def test_wrapper_exports_core_timing_validator() -> None:
    from dev.benchmarks import generate_benchmark_data as wrapper
    from dev.benchmarks.frontend_data import validate_output as core_validate_output

    assert wrapper.validate_output is core_validate_output


def test_schema_rejects_zero_fit_time() -> None:
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate, validate_against_schema

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    output, _, _ = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=manifest,
    )
    run = next(run for run in output["runs"] if run.get("metrics", {}).get("timing"))
    run["metrics"]["timing"]["fit_time_ms"] = 0

    errors = validate_against_schema(output)
    assert any("fit_time_ms" in error for error in errors)
