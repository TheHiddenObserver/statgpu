from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SOURCE_PATH = (
    REPO_ROOT
    / "results"
    / "benchmark_frontend_sources"
    / "panel_stage_b_pr122_p100_20260809.json"
)


def _synthetic_source(tmp_path: Path, *, include_numeric: bool) -> Path:
    data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    for backend in ("cupy", "torch"):
        diagnostics = data["backend_results"][backend]["diagnostics"]
        template = dict(next(iter(diagnostics.values())))
        template["status"] = "success"
        template["applicable"] = True
        template["reason"] = None
        if include_numeric:
            template["statistic"] = 1.1965942530850693
            template["pvalue"] = 0.2740034414267718
            template["df"] = 1.0
        diagnostics["hausman_applicable_nonzero_effect"] = template

    synthetic = tmp_path / (
        "panel_stage_b_with_applicable_hausman.json"
        if include_numeric
        else "panel_stage_b_missing_applicable_numeric.json"
    )
    synthetic.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return synthetic


def test_parser_distinguishes_dedicated_applicable_hausman_fixture(tmp_path) -> None:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )

    runs, _models, warnings = parse_panel_stage_b_physical_validation(
        _synthetic_source(tmp_path, include_numeric=True),
        "synthetic-pr122-applicable",
    )

    assert warnings == []
    dedicated = [
        run for run in runs if run["variant"] == "hausman-applicable-nonzero-effect"
    ]
    assert len(dedicated) == 2
    assert {run["backend"] for run in dedicated} == {"cupy", "torch"}
    assert {run["scale"]["n_samples"] for run in dedicated} == {48}
    assert {run["scale"]["n_features"] for run in dedicated} == {1}
    assert {run["parameters"]["parameterization"] for run in dedicated} == {"standard"}
    assert {
        run["parameters"]["diagnostic_fixture"] for run in dedicated
    } == {"nonzero-effect-applicable"}
    assert all(run["parameters"]["applicable"] is True for run in dedicated)
    assert {run["parameters"]["statistic"] for run in dedicated} == {
        1.1965942530850693
    }
    assert {run["parameters"]["pvalue"] for run in dedicated} == {
        0.2740034414267718
    }
    assert {run["parameters"]["df"] for run in dedicated} == {1.0}
    assert all(run["metrics"]["validation"]["status"] == "pass" for run in dedicated)
    assert all(
        {
            check["metric"]: check["status"]
            for check in run["metrics"]["validation"]["checks"]
        }["hausman_applicable_statistic_pvalue_df"]
        == "pass"
        for run in dedicated
    )

    standard_balanced = [
        run for run in runs if run["variant"] == "hausman-balanced"
    ]
    assert len(standard_balanced) == 2
    dedicated_methods = {run["method_config_id"] for run in dedicated}
    standard_methods = {run["method_config_id"] for run in standard_balanced}
    assert dedicated_methods.isdisjoint(standard_methods)
    dedicated_cases = {run["case_id"] for run in dedicated}
    standard_cases = {run["case_id"] for run in standard_balanced}
    assert dedicated_cases.isdisjoint(standard_cases)


def test_parser_fails_closed_when_applicable_numeric_evidence_is_missing(tmp_path) -> None:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )

    runs, _models, warnings = parse_panel_stage_b_physical_validation(
        _synthetic_source(tmp_path, include_numeric=False),
        "synthetic-pr122-missing-numeric",
    )

    assert warnings == []
    dedicated = [
        run for run in runs if run["variant"] == "hausman-applicable-nonzero-effect"
    ]
    assert len(dedicated) == 2
    assert all(run["metrics"]["validation"]["status"] == "fail" for run in dedicated)
    assert all(
        {
            check["metric"]: check["status"]
            for check in run["metrics"]["validation"]["checks"]
        }["hausman_applicable_statistic_pvalue_df"]
        == "fail"
        for run in dedicated
    )
