from __future__ import annotations
"""Benchmark frontend data generation package.

The public package installs two canonical contract hooks before exporting the CLI:

* manifest SHA256 values are injected before generation IDs are computed;
* JSON Schema format declarations are enforced with the validator's format
  checker rather than treated as annotations.

Keeping the hooks at package initialization makes the script wrapper, the
package entry point, and normal package imports share one validation contract.
"""

from pathlib import Path
from typing import Optional

from . import cli as _cli


_original_inject_canonical_fields = _cli._inject_canonical_fields


def _inject_canonical_fields_with_hashes(
    runs: list[dict], manifest: dict, results_dir: Path
) -> None:
    """Inject canonical identities plus the exact registered source hash."""
    _original_inject_canonical_fields(runs, manifest, results_dir)
    hashes_by_source_id = {
        source["source_id"]: source["sha256"]
        for source in manifest.get("sources", [])
        if source.get("sha256")
    }
    for run in runs:
        source = run.get("source", {})
        source_hash = hashes_by_source_id.get(source.get("source_id"))
        if source_hash:
            source["sha256"] = source_hash


def validate_against_schema(output: dict) -> list[str]:
    """Validate the full schema, including declared date-time formats."""
    schema = _cli._load_schema()
    if schema is None:
        return ["Schema file not found; JSON Schema validation is required"]

    validator_class = _cli._get_jsonschema_validator()
    if validator_class is None:
        return [
            "jsonschema[format]>=4.0 is required for benchmark bundle "
            "validation; install statgpu[dev] or statgpu[validation]"
        ]

    validator = validator_class(
        schema,
        format_checker=validator_class.FORMAT_CHECKER,
    )
    schema_errors = sorted(
        validator.iter_errors(output),
        key=lambda error: list(error.path),
    )
    return [
        f"{' → '.join(str(part) for part in error.absolute_path)}: "
        f"{error.message}"
        for error in schema_errors[:50]
    ]


# Patch the CLI module before exposing its entry points. cli.generate resolves
# _inject_canonical_fields at call time, while cli.main resolves both generate
# and validate_against_schema from its module globals.
_cli._inject_canonical_fields = _inject_canonical_fields_with_hashes
_cli.validate_against_schema = validate_against_schema

generate = _cli.generate
validate_output = _cli.validate_output
main = _cli.main
get_git_sha = _cli.get_git_sha
_write_transactional = _cli._write_transactional

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
