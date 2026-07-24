#!/usr/bin/env python3
"""Render the final PR79 Core Accuracy Gate report from validated JSON.

Tests may import::

    from dev.benchmarks.pr79.emit_final_report import (
        ReportValidationError,
        emit_report,
        load_json_strict,
        render_markdown,
        validate_aggregated_report,
    )
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# –– public API ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class ReportValidationError(ValueError):
    """Raised when a validated report cannot be rendered as PASS."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard / non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> dict:
    """Load a JSON file, rejecting NaN, Infinity, and other non-standard values."""
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle, parse_constant=_reject_json_constant)

    if not isinstance(result, dict):
        raise ReportValidationError("validated report must be a JSON object")

    return result


def validate_aggregated_report(report: Mapping[str, Any]) -> None:
    """Raise ReportValidationError if *report* is not a canonical PASS."""
    if report.get("validated_schema_version") != "pr79-validated-accuracy-1.0":
        raise ReportValidationError("unsupported validated schema version")

    if report.get("status") != "pass":
        raise ReportValidationError("cannot render a failed accuracy Gate as final")

    if report.get("canonical_eligible") is not True:
        raise ReportValidationError(
            "PASS claim requires clean exact-head canonical evidence"
        )

    sha = report.get("validated_git_sha", "")
    if not _SHA_PATTERN.match(sha):
        raise ReportValidationError("SHA-256 mismatch: validated_git_sha must be a 40-char hex SHA")

    # Cross-check validated_git_sha against embedded repository provenance
    raw_prov = report.get("repository_provenance", {}).get("raw")
    if isinstance(raw_prov, Mapping):
        for snapshot_key in ("initial", "final"):
            snapshot = raw_prov.get(snapshot_key)
            if isinstance(snapshot, Mapping):
                prov_sha = snapshot.get("git_sha", "")
                if prov_sha and _SHA_PATTERN.match(prov_sha) and prov_sha != sha:
                    raise ReportValidationError(
                        "SHA-256 mismatch: validated_git_sha does not match "
                        f"repository provenance ({snapshot_key})"
                    )

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise ReportValidationError("validated summary is missing")

    if summary.get("unresolved", 0) != 0:
        raise ReportValidationError("validated report contains unresolved checks")

    total = summary.get("total_checks", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    if total != passed + failed:
        raise ReportValidationError(
            f"passed ({passed}) + failed ({failed}) do not derive from total_checks ({total})"
        )


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a canonical PASS report as Markdown."""
    validate_aggregated_report(report)

    summary = report["summary"]
    sha = report.get("validated_git_sha", report.get("validated_code_sha", ""))
    meaningful = summary.get("meaningful_parity_passed", summary.get("passed", 0))
    meaningful_total = summary.get("meaningful_parity_checks", summary.get("total_checks", meaningful))
    not_comparable = summary.get("rank_def_non_identifiable", summary.get("not_comparable", 0))
    final_passed = summary.get("final_state_contracts_passed", 0)
    final_total = summary.get("final_state_contracts", final_passed)

    return (
        f"# PR79 Core Accuracy Gate\n\n"
        f"**Validated code SHA:** `{sha}`\n\n"
        f"## Verdict\n\n"
        f"**{summary.get('gate_verdict', 'PASS')}**\n\n"
        f"## Summary\n\n"
        f"| Category | Passed | Total |\n"
        f"|---|---:|---:|\n"
        f"| Meaningful parity | {meaningful} | {meaningful_total} |\n"
        f"| Final-state contracts | {final_passed} | {final_total} |\n"
        f"| Not comparable | {not_comparable} | {not_comparable} |\n"
        f"| Unresolved | {summary.get('unresolved', 0)} | 0 |\n"
    )


def emit_report(
    validated: Mapping[str, Any],
    output_json: Path,
    output_markdown: Path,
) -> None:
    """Validate *validated* and write JSON + Markdown to *output_json* / *output_markdown*."""
    validate_aggregated_report(validated)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(validated, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    with output_markdown.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(validated))


# –– CLI ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="full", help="preset configuration name")
    parser.add_argument("--validated", type=Path, help="path to validated JSON")
    parser.add_argument("--output-json", type=Path, help="output JSON path")
    parser.add_argument("--output-markdown", type=Path, help="output Markdown path")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    validated_path = args.validated
    output_json = args.output_json
    output_markdown = args.output_markdown

    config = args.config
    if validated_path is None:
        if config == "full":
            validated_path = Path("results/pr79/accuracy/full_validated_results.json")
            output_json = output_json or Path("results/pr79/final/final_accuracy_report.json")
            output_markdown = output_markdown or Path("results/pr79/final/final_accuracy_report.md")
        else:
            validated_path = Path(f"results/pr79/accuracy/{config}_validated_results.json")
            output_json = output_json or Path(f"results/pr79/accuracy/{config}_final_report.json")
            output_markdown = output_markdown or Path(f"results/pr79/accuracy/{config}_final_report.md")

    report = load_json_strict(validated_path)
    actual_config = report.get("configuration", report.get("config_name", ""))
    if actual_config and actual_config != config:
        print(
            f"configuration does not match: requested {config}, validated has {actual_config}",
            file=sys.stderr,
        )
        return 1

    emit_report(report, output_json, output_markdown)
    print(f"Report written to {output_json} and {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
