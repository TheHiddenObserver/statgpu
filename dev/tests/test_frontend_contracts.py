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


def test_schema_validation_requires_jsonschema(monkeypatch) -> None:
    from dev.benchmarks.frontend_data import cli

    monkeypatch.setattr(cli, "_get_jsonschema_validator", lambda: None)
    errors = cli.validate_against_schema({})

    assert any("jsonschema>=4.0 is required" in error for error in errors)


def test_transactional_write_rolls_back_all_prior_assets(
    tmp_path, monkeypatch
) -> None:
    import os

    import pytest

    from dev.benchmarks.frontend_data.cli import _write_transactional

    data_path = tmp_path / "benchmark_data.json"
    report_path = tmp_path / "parse_report.json"
    inventory_path = tmp_path / "source_inventory.json"
    old_contents = {
        data_path: "old-data",
        report_path: "old-report",
        inventory_path: "old-inventory",
    }
    for path, content in old_contents.items():
        path.write_text(content, encoding="utf-8")

    real_replace = os.replace
    failed = False

    def fail_data_replacement(source, target):
        nonlocal failed
        if Path(target) == data_path and not failed:
            failed = True
            raise OSError("simulated data replacement failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_data_replacement)

    with pytest.raises(OSError, match="simulated data replacement failure"):
        _write_transactional(
            data_path,
            report_path,
            inventory_path,
            {"generation_id": "new"},
            {"generation_id": "new"},
            {"generation_id": "new"},
        )

    assert {
        path: path.read_text(encoding="utf-8")
        for path in old_contents
    } == old_contents


def _semantic_speedup_output() -> dict:
    reference = {
        "run_id": "reference",
        "env_id": "remote-p100",
        "category_ids": ["linear_models"],
        "model_id": "ExampleModel",
        "comparison_id": "comparison",
        "case_id": "case-a",
        "method_config_id": "method-a",
        "variant": "variant-a",
        "loss": "squared_error",
        "penalty": "l2",
        "solver": "auto",
        "framework": "statgpu",
        "backend": "numpy",
        "scale": {
            "scale_key": "n10_p2",
            "n_samples": 10,
            "n_features": 2,
            "label": "10×2",
        },
        "source": {"file": "reference.json", "source_id": "reference-source"},
        "metrics": {"timing": {"fit_time_ms": 10.0}},
    }
    measured = {
        **reference,
        "run_id": "measured",
        "backend": "cupy",
        "source": {"file": "measured.json", "source_id": "measured-source"},
        "metrics": {
            "timing": {"fit_time_ms": 5.0},
            "speedup": {
                "value": 2.0,
                "reference_run_id": "reference",
                "reference_framework": "statgpu",
                "reference_backend": "numpy",
                "reported_semantics": "computed",
            },
        },
    }
    return {
        "environments": [
            {"env_id": "remote-p100"},
            {"env_id": "other-env"},
        ],
        "models": [
            {"model_id": "ExampleModel", "category_ids": ["linear_models"]},
            {"model_id": "OtherModel", "category_ids": ["linear_models"]},
        ],
        "frameworks": [
            {
                "framework_id": "statgpu",
                "backend_policy": "required",
            }
        ],
        "comparisons": [
            {"comparison_id": "comparison"},
            {"comparison_id": "other-comparison"},
        ],
        "runs": [reference, measured],
    }


def test_speedup_reference_requires_complete_chart_cell_identity() -> None:
    from copy import deepcopy

    from dev.benchmarks.frontend_data.cli import validate_semantic

    mutations = [
        (("comparison_id",), "other-comparison"),
        (("env_id",), "other-env"),
        (("model_id",), "OtherModel"),
        (("case_id",), "case-b"),
        (("method_config_id",), "method-b"),
        (("variant",), "variant-b"),
        (("loss",), "huber"),
        (("penalty",), "l1"),
        (("solver",), "fista"),
        (("scale", "scale_key"), "n20_p2"),
    ]

    for path, value in mutations:
        output = deepcopy(_semantic_speedup_output())
        reference = output["runs"][0]
        target = reference
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

        errors = validate_semantic(output)
        assert any(
            "speedup reference has different chart-cell identity" in error
            for error in errors
        ), path
