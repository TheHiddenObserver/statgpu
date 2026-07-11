#!/usr/bin/env python3
"""
Generate unified benchmark data JSON for the statgpu frontend dashboard.

Usage:
    python dev/benchmarks/generate_benchmark_data.py \
        --out frontend/public/data/benchmark_data.json \
        --report frontend/public/data/parse_report.json

    python dev/benchmarks/generate_benchmark_data.py --check  # validate only

This is a thin wrapper — implementation lives in dev.benchmarks.frontend_data.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path for 'from dev.benchmarks.frontend_data import ...'
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dev.benchmarks.frontend_data import (
    generate,
    validate_output,
    validate_against_schema,
    main,
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
