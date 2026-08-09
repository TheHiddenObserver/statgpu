"""Regression coverage for findings raised after PR #122 became ready for review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from statgpu.panel import BetweenOLS, PanelOLS


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_PATH = (
    REPO_ROOT
    / "results"
    / "benchmark_frontend_sources"
    / "panel_stage_b_pr122_p100_20260809.json"
)


def test_disconnected_two_way_fe_uses_component_rank_before_df_gate() -> None:
    # Three disconnected entity-time incidence components: two 2x2 blocks and
    # one singleton.  The true nuisance rank is N + T - C = 7, whereas the
    # historical count uses (N - 1) + (T - 1) = 8.
    entity = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int64)
    time = np.asarray([0, 1, 0, 1, 2, 3, 2, 3, 4], dtype=np.int64)
    X = np.asarray([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0]).reshape(-1, 1)
    y = np.asarray([1.0, -1.0, -1.0, 1.0, 2.0, -2.0, -2.0, 2.0, 0.0])

    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X,
        y,
        entity_ids=entity,
        time_ids=time,
    )

    metadata = model.fit_statistics_.metadata
    diagnostic_df = metadata["diagnostic_df"]
    assert metadata["legacy_df_resid"] == 0
    assert metadata["public_df_resid_basis"] == "component-aware"
    assert model.df_resid == 1
    assert diagnostic_df["incidence_components"] == 3
    assert diagnostic_df["effect_rank"] == 7
    assert diagnostic_df["rank_x"] == 1
    assert diagnostic_df["df_resid"] == 1
    assert np.all(np.isfinite(model.bse_))


def _parse_mutated_source(tmp_path: Path, mutate) -> tuple[list[dict], list[str]]:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )

    data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "mutated_panel_stage_b.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    runs, _models, warnings = parse_panel_stage_b_physical_validation(
        path, "test-pr122-aggregate-failure"
    )
    return runs, warnings


def test_panel_stage_b_parser_rejects_failed_source_summary(tmp_path: Path) -> None:
    runs, warnings = _parse_mutated_source(
        tmp_path,
        lambda data: data.__setitem__("status", "failed"),
    )

    assert len(runs) == 42
    assert any("physical validation status is not success" in item for item in warnings)
    assert all(run["metrics"]["validation"]["status"] == "fail" for run in runs)
    assert all(
        any(
            check["metric"] == "source_validation_status_success"
            and check["status"] == "fail"
            for check in run["metrics"]["validation"]["checks"]
        )
        for run in runs
    )
    estimator_runs = [run for run in runs if "inference" in run["metrics"]]
    assert estimator_runs
    assert all(run["metrics"]["inference"]["ok"] is False for run in estimator_runs)


def test_panel_stage_b_parser_rejects_failed_backend_summary(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["backend_results"]["cupy"]["status"] = "failed"

    runs, warnings = _parse_mutated_source(tmp_path, mutate)
    cupy_runs = [run for run in runs if run["backend"] == "cupy"]
    torch_runs = [run for run in runs if run["backend"] == "torch"]

    assert len(cupy_runs) == 21
    assert len(torch_runs) == 21
    assert any("cupy backend validation status is not success" in item for item in warnings)
    assert all(run["metrics"]["validation"]["status"] == "fail" for run in cupy_runs)
    assert all(run["metrics"]["validation"]["status"] == "pass" for run in torch_runs)
    assert all(
        any(
            check["metric"] == "backend_validation_status_success"
            and check["status"] == "fail"
            for check in run["metrics"]["validation"]["checks"]
        )
        for run in cupy_runs
    )


def test_between_ols_keeps_detailed_class_contract_docstring() -> None:
    doc = BetweenOLS.__doc__ or ""
    assert "Collapses the data to entity means" in doc
    assert "Parameters" in doc
    assert "Attributes" in doc
    assert "fit_statistics_" in doc
