#!/usr/bin/env python3
"""
Generate unified benchmark data JSON for the statgpu frontend dashboard.

Usage:
    python dev/benchmarks/generate_benchmark_data.py \
        --out frontend/public/data/benchmark_data.json \
        --report frontend/public/data/parse_report.json \
        --inventory-out frontend/public/data/source_inventory.json

    python dev/benchmarks/generate_benchmark_data.py --check  # validate only

This is a thin wrapper — implementation lives in dev.benchmarks.frontend_data.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path for 'from dev.benchmarks.frontend_data import ...'
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dev.benchmarks import frontend_data as _frontend_data
from dev.benchmarks.frontend_data import (
    generate,
    validate_output,
    validate_against_schema,
    get_git_sha,
    CATEGORIES,
    BACKEND_MAP,
    FRAMEWORK_MAP,
    SCALE_CONFIG,
    SOLVER_KIND_MAP,
    SOLVER_DISPLAY_MAP,
    FAMILY_MODEL_MAP,
    SPEEDUP_REFERENCE_BY_SOURCE,
    make_scale_key,
    make_scale_label,
    make_run_id,
    _short_hash,
    parse_family_penalty_solver,
    normalize_utf8_bytes,
    source_sha256,
)


def validate_output(output: dict) -> list[str]:
    """Validate the normalized bundle, including log-axis timing safety.

    A zero wall-clock duration is not a meaningful measured benchmark and cannot
    be represented by the dashboard's logarithmic timing axis. Treat every
    emitted timing as strictly positive rather than silently dropping the bar.
    """
    errors = _frontend_data.validate_output(output)
    for run in output.get("runs", []):
        timing = run.get("metrics", {}).get("timing")
        if timing is None:
            continue
        fit_time_ms = timing.get("fit_time_ms")
        if not isinstance(fit_time_ms, (int, float)) or fit_time_ms <= 0:
            errors.append(
                f"{run.get('run_id', '?')}: timing.fit_time_ms must be > 0 "
                f"for logarithmic charting ({fit_time_ms})"
            )
    return errors


def main() -> None:
    from dev.benchmarks.frontend_data import cli as _cli

    _cli.main()


__all__ = [
    "generate",
    "validate_output",
    "validate_against_schema",
    "main",
    "get_git_sha",
    "CATEGORIES",
    "BACKEND_MAP",
    "FRAMEWORK_MAP",
    "SCALE_CONFIG",
    "SOLVER_KIND_MAP",
    "SOLVER_DISPLAY_MAP",
    "FAMILY_MODEL_MAP",
    "SPEEDUP_REFERENCE_BY_SOURCE",
    "make_scale_key",
    "make_scale_label",
    "make_run_id",
    "_short_hash",
    "parse_family_penalty_solver",
    "normalize_utf8_bytes",
    "source_sha256",
]

if __name__ == "__main__":
    main()
