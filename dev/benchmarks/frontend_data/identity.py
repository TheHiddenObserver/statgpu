from __future__ import annotations
"""Run identity and chart cell identity helpers."""

import json


def identity_json(identity: list) -> str:
    """Serialize an identity list to canonical JSON for hashing."""
    return json.dumps(identity, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def run_identity(run: dict) -> list:
    """Canonical RunIdentity. Used for accumulator (A2+), run ID (A2+), preservation."""
    return [
        run["source"].get("source_id", run["source"]["file"]),
        run.get("case_id", "default"),
        run.get("method_config_id", "default"),
        run["env_id"],
        run["model_id"],
        run.get("variant"),
        run.get("implementation"),
        run.get("loss"),
        run.get("penalty"),
        run.get("solver"),
        run["framework"],
        run["backend"],
        run["scale"]["scale_key"],
    ]


def chart_cell_identity(run: dict, *, include_session: bool) -> list:
    """Chart cell uniqueness. EXCLUDES source_id. Must match TypeScript chartGroupIdentity."""
    base = [
        run.get("comparison_id", run["source"].get("source_id", run["source"]["file"])),
        run["env_id"],
        run["model_id"],
        run.get("case_id", "default"),
        run.get("method_config_id", "default"),
        run.get("variant"),
        run.get("loss"),
        run.get("penalty"),
        run.get("solver"),
        run["scale"]["scale_key"],
    ]
    # NOTE: implementation, framework, backend belong in chart_series_identity
    if include_session:
        return [run.get("comparison_id", run["source"].get("source_id", run["source"]["file"])),
                run.get("benchmark_session_id")] + base[1:]
    return base


def chart_series_identity(run: dict) -> list:
    """Series identity: framework, backend, implementation."""
    return [
        run["framework"],
        run.get("backend"),
        run.get("implementation"),
    ]


def timing_cell_identity(run: dict, *, include_session: bool) -> list:
    """Cell = group + series."""
    return chart_cell_identity(run, include_session=include_session) + chart_series_identity(run)


def speedup_cell_identity(run: dict, *, include_session: bool) -> list:
    """Cell = group + series."""
    return chart_cell_identity(run, include_session=include_session) + chart_series_identity(run)
