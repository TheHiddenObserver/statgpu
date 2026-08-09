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


def test_parser_distinguishes_dedicated_applicable_hausman_fixture(tmp_path) -> None:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )

    data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    for backend in ("cupy", "torch"):
        diagnostics = data["backend_results"][backend]["diagnostics"]
        template = dict(next(iter(diagnostics.values())))
        template["status"] = "success"
        template["applicable"] = True
        diagnostics["hausman_applicable_nonzero_effect"] = template

    synthetic = tmp_path / "panel_stage_b_with_applicable_hausman.json"
    synthetic.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    runs, _models, warnings = parse_panel_stage_b_physical_validation(
        synthetic, "synthetic-pr122-applicable"
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
    assert all(
        "hausman_applicable_statistic_pvalue_df"
        in {check["metric"] for check in run["metrics"]["validation"]["checks"]}
        for run in dedicated
    )

    standard_balanced = [
        run for run in runs if run["variant"] == "hausman-balanced"
    ]
    assert len(standard_balanced) == 2
    dedicated_methods = {run["method_config_id"] for run in dedicated}
    standard_methods = {run["method_config_id"] for run in standard_balanced}
    assert dedicated_methods.isdisjoint(standard_methods)
