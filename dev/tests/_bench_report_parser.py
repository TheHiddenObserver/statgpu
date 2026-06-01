"""Parse full-matrix benchmark text output into JSON/Markdown summaries.

This script is intentionally standalone so it can summarize remote benchmark
logs without importing statgpu or touching benchmark execution code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


GROUP_RE = re.compile(
    r"^\s*\[(?P<family>[^+\]]+)\+(?P<penalty>[^|]+)"
    r"\|\s*n=(?P<n>\d+),p=(?P<p>\d+)"
    r"\s*\|\s*solvers=(?P<solvers>[^\]]+)\]"
)

ROW_RE = re.compile(
    r"^\s*(?P<solver>[A-Za-z0-9_]+)\s+"
    r"(?P<backend>CPU|cpu|numpy|cupy|torch|cuda)\s+"
    r"(?P<time_ms>[-+0-9.eE]+)\s+"
    r"(?P<iters>[-+0-9.eE]+)\s+"
    r"(?P<nnz>[-+0-9.eE]+)\s+"
    r"(?P<coef_norm>[-+0-9.eE]+)\s+"
    r"(?P<vs_cpu>\S+)\s+"
    r"(?P<speedup>\S+)"
)

WARN_RE = re.compile(r"(^|\W)(WARN|WARNING)(\W|$)", re.IGNORECASE)
ERROR_RE = re.compile(r"(^|\W)(ERROR|FAILED|FAIL)(\W|$)", re.IGNORECASE)

SECTION_SUMMARY_RE = re.compile(
    r"^\s*(?P<section>Section\s+\S+):\s+"
    r"(?P<passed>\d+)/(?P<total>\d+)\s+passed"
    r"(?:\s+\(max diff:\s*(?P<max_diff>[-+0-9.eE]+)\))?.*"
    r"\[(?P<status>[^\]]+)\]"
)

TOTAL_RE = re.compile(
    r"^\s*TOTAL:\s+(?P<passed>\d+)/(?P<total>\d+)\s+passed\s+\[(?P<status>[^\]]+)\]"
)


def _float_or_none(value: str) -> Optional[float]:
    token = value.strip().rstrip("x")
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: str) -> Optional[int]:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(parsed)


def _top(rows: Iterable[Dict[str, Any]], key: str, reverse: bool, limit: int = 10):
    finite = [row for row in rows if row.get(key) is not None]
    return sorted(finite, key=lambda row: row[key], reverse=reverse)[:limit]


def _count_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def parse_benchmark_text(text: str, precision_threshold: float = 1e-3) -> Dict[str, Any]:
    """Parse benchmark output text into a structured summary."""
    rows: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    section_summaries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    errors: List[str] = []
    current_group: Optional[Dict[str, Any]] = None
    total_summary: Optional[Dict[str, Any]] = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        group_match = GROUP_RE.match(line)
        if group_match:
            current_group = {
                "family": group_match.group("family").strip(),
                "penalty": group_match.group("penalty").strip(),
                "n": int(group_match.group("n")),
                "p": int(group_match.group("p")),
                "solvers": [
                    part.strip()
                    for part in group_match.group("solvers").split(",")
                    if part.strip()
                ],
                "line": lineno,
            }
            groups.append(current_group)
            continue

        row_match = ROW_RE.match(line)
        if row_match and current_group is not None:
            row = {
                **{k: current_group[k] for k in ("family", "penalty", "n", "p")},
                "solver": row_match.group("solver"),
                "backend": row_match.group("backend").lower(),
                "time_ms": _float_or_none(row_match.group("time_ms")),
                "iters": _int_or_none(row_match.group("iters")),
                "nnz": _int_or_none(row_match.group("nnz")),
                "coef_norm": _float_or_none(row_match.group("coef_norm")),
                "vs_cpu": _float_or_none(row_match.group("vs_cpu")),
                "speedup": _float_or_none(row_match.group("speedup")),
                "line": lineno,
            }
            rows.append(row)
            continue

        section_match = SECTION_SUMMARY_RE.match(line)
        if section_match:
            section_summaries.append(
                {
                    "section": section_match.group("section"),
                    "passed": int(section_match.group("passed")),
                    "total": int(section_match.group("total")),
                    "max_diff": _float_or_none(section_match.group("max_diff") or ""),
                    "status": section_match.group("status"),
                    "line": lineno,
                }
            )
            continue

        total_match = TOTAL_RE.match(line)
        if total_match:
            total_summary = {
                "passed": int(total_match.group("passed")),
                "total": int(total_match.group("total")),
                "status": total_match.group("status"),
                "line": lineno,
            }
            continue

        if WARN_RE.search(line):
            warnings.append(f"{lineno}: {line.strip()}")
        if ERROR_RE.search(line):
            errors.append(f"{lineno}: {line.strip()}")

    gpu_rows = [row for row in rows if row["backend"] not in ("cpu", "numpy")]
    precision_alerts = [
        row for row in gpu_rows
        if row["vs_cpu"] is not None and row["vs_cpu"] > precision_threshold
    ]
    slow_gpu_rows = [
        row for row in gpu_rows
        if row["speedup"] is not None and row["speedup"] < 1.0
    ]
    fast_gpu_rows = [
        row for row in gpu_rows
        if row["speedup"] is not None and row["speedup"] >= 1.0
    ]

    return {
        "counts": {
            "groups": len(groups),
            "rows": len(rows),
            "gpu_rows": len(gpu_rows),
            "precision_alerts": len(precision_alerts),
            "slow_gpu_rows": len(slow_gpu_rows),
            "fast_gpu_rows": len(fast_gpu_rows),
            "warnings": len(warnings),
            "errors": len(errors),
        },
        "backend_counts": _count_by(rows, "backend"),
        "family_counts": _count_by(rows, "family"),
        "penalty_counts": _count_by(rows, "penalty"),
        "section_summaries": section_summaries,
        "total_summary": total_summary,
        "top_precision_alerts": _top(precision_alerts, "vs_cpu", reverse=True),
        "slowest_gpu_rows": _top(slow_gpu_rows, "speedup", reverse=False),
        "fastest_gpu_rows": _top(fast_gpu_rows, "speedup", reverse=True),
        "warnings": warnings[:50],
        "errors": errors[:50],
        "rows": rows,
    }


def summary_to_markdown(summary: Dict[str, Any]) -> str:
    """Render a compact Markdown report."""
    counts = summary["counts"]
    lines = [
        "# Benchmark Summary",
        "",
        "## Counts",
        "",
        f"- Groups: {counts['groups']}",
        f"- Rows: {counts['rows']} ({counts['gpu_rows']} GPU rows)",
        f"- Precision alerts: {counts['precision_alerts']}",
        f"- Slow GPU rows: {counts['slow_gpu_rows']}",
        f"- Fast GPU rows: {counts['fast_gpu_rows']}",
        f"- Warnings: {counts['warnings']}",
        f"- Errors: {counts['errors']}",
        "",
    ]

    if summary.get("backend_counts"):
        backend_counts = ", ".join(
            f"{name}={count}" for name, count in summary["backend_counts"].items()
        )
        lines.extend(["## Backend Rows", "", f"- {backend_counts}", ""])

    if summary["section_summaries"]:
        lines.extend(["## Section Results", ""])
        for item in summary["section_summaries"]:
            max_diff = item["max_diff"]
            suffix = f", max diff {max_diff:.3g}" if max_diff is not None else ""
            lines.append(
                f"- {item['section']}: {item['passed']}/{item['total']} "
                f"passed [{item['status']}]{suffix}"
            )
        lines.append("")

    total = summary.get("total_summary")
    if total is not None:
        lines.extend([
            "## Total",
            "",
            f"- {total['passed']}/{total['total']} passed [{total['status']}]",
            "",
        ])

    def add_table(title: str, rows: List[Dict[str, Any]], metric: str):
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["No rows.", ""])
            return
        lines.append("| Family | Penalty | n,p | Solver | Backend | Metric | Time ms | Speedup |")
        lines.append("|---|---|---:|---|---|---:|---:|---:|")
        for row in rows:
            metric_value = row.get(metric)
            metric_text = "" if metric_value is None else f"{metric_value:.3g}"
            speedup = row.get("speedup")
            speedup_text = "" if speedup is None else f"{speedup:.2f}x"
            lines.append(
                f"| {row['family']} | {row['penalty']} | {row['n']},{row['p']} "
                f"| {row['solver']} | {row['backend']} | {metric_text} "
                f"| {row['time_ms']:.1f} | {speedup_text} |"
            )
        lines.append("")

    add_table("Largest CPU Differences", summary["top_precision_alerts"], "vs_cpu")
    add_table("Slowest GPU Rows", summary["slowest_gpu_rows"], "speedup")
    add_table("Fastest GPU Rows", summary["fastest_gpu_rows"], "speedup")

    if summary["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {item}" for item in summary["errors"])
        lines.append("")

    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in summary["warnings"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Benchmark output text file")
    parser.add_argument("--summary-json", type=Path, help="Write structured JSON summary")
    parser.add_argument("--summary-md", type=Path, help="Write Markdown summary")
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=1e-3,
        help="Flag GPU rows whose vs_CPU exceeds this threshold",
    )
    parser.add_argument(
        "--fail-on-alerts",
        action="store_true",
        help=(
            "Exit non-zero when errors, precision alerts, failed sections, or a "
            "failed TOTAL status are present"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print Markdown to stdout")
    args = parser.parse_args(argv)

    text = args.input.read_text(encoding="utf-8", errors="replace")
    summary = parse_benchmark_text(text, precision_threshold=args.precision_threshold)
    markdown = summary_to_markdown(summary)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text(markdown, encoding="utf-8")
    if not args.quiet:
        print(markdown)
    if args.fail_on_alerts:
        failed_sections = any(
            item.get("passed") != item.get("total")
            for item in summary.get("section_summaries", [])
        )
        total = summary.get("total_summary")
        failed_total = total is not None and total.get("passed") != total.get("total")
        if (
            summary["counts"]["errors"] > 0
            or summary["counts"]["precision_alerts"] > 0
            or failed_sections
            or failed_total
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
