"""Benchmark frontend data generation package."""

from .cli import (
    generate,
    validate_output,
    validate_against_schema,
    main,
    get_git_sha,
)
from .canonical import (
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
