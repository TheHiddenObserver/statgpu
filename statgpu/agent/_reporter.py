"""Report generation: dict, markdown, JSON, and notebook output."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "NA"
    if abs(numeric) >= 1000 or (abs(numeric) < 0.001 and numeric != 0.0):
        return f"{numeric:.3e}"
    return f"{numeric:.4f}"


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def to_markdown(result, max_terms: int = 12) -> str:
    """Generate markdown report from AnalysisResult."""
    lines = [
        "# statgpu Automatic Analysis Report",
        "",
        "## Data Profile",
        f"- Task: `{result.profile.task_type}`",
        f"- Samples: {result.profile.n_samples}",
        f"- Features: {result.profile.n_features}",
        f"- Target: {result.profile.target_name or 'None'}",
        f"- Device: `{result.profile.device}`",
    ]
    if result.profile.dropped_rows:
        lines.append(f"- Dropped rows: {result.profile.dropped_rows}")
    if result.profile.imputed_values:
        lines.append(f"- Imputed feature values: {result.profile.imputed_values}")
    if result.profile.target_summary:
        summary = ", ".join(
            f"{key}={_format_metric(value)}"
            for key, value in result.profile.target_summary.items()
        )
        lines.append(f"- Target summary: {summary}")

    lines.extend(["", "## Agent Plan"])
    lines.append("- Agents: " + ", ".join(f"`{name}`" for name in result.plan.agents))
    lines.append("- Methods: " + ", ".join(f"`{name}`" for name in result.plan.methods))
    for reason in result.plan.rationale:
        lines.append(f"- {reason}")

    lines.extend(["", "## Results"])
    for model in result.models:
        lines.append(f"### {model.name}")
        if model.error:
            lines.append(f"- Error: {model.error}")
            continue
        if model.metrics:
            metric_text = ", ".join(
                f"{key}={_format_metric(value)}"
                for key, value in model.metrics.items()
            )
            lines.append(f"- Metrics: {metric_text}")
        if model.diagnostics:
            diag_text = ", ".join(
                f"{key}={_format_metric(value)}"
                for key, value in model.diagnostics.items()
                if isinstance(value, (str, int, float, np.integer, np.floating))
                or value is None
            )
            if diag_text:
                lines.append(f"- Diagnostics: {diag_text}")
        if model.coefficients:
            # Determine if we have adjusted p-values
            has_adj = any(row.get("adj_p_value") is not None for row in model.coefficients)
            if has_adj:
                lines.append("")
                lines.append("| term | estimate | std_error | statistic | p_value | adj_p_value | rejected | interval |")
                lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
                for row in model.coefficients[:max_terms]:
                    interval = _format_interval(row)
                    rejected = "Yes" if row.get("rejected") else "No"
                    lines.append(
                        f"| {row.get('term', '')} | {_format_metric(row.get('estimate'))} | "
                        f"{_format_metric(row.get('std_error'))} | {_format_metric(row.get('statistic'))} | "
                        f"{_format_metric(row.get('p_value'))} | {_format_metric(row.get('adj_p_value'))} | "
                        f"{rejected} | {interval} |"
                    )
            else:
                lines.append("")
                lines.append("| term | estimate | std_error | statistic | p_value | interval |")
                lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
                for row in model.coefficients[:max_terms]:
                    interval = _format_interval(row)
                    lines.append(
                        f"| {row.get('term', '')} | {_format_metric(row.get('estimate'))} | "
                        f"{_format_metric(row.get('std_error'))} | {_format_metric(row.get('statistic'))} | "
                        f"{_format_metric(row.get('p_value'))} | {interval} |"
                    )
        if model.feature_importance:
            lines.append("")
            lines.append("**Feature Importance**:")
            lines.append("| feature | importance |")
            lines.append("| --- | ---: |")
            for fi in model.feature_importance[:max_terms]:
                lines.append(f"| {fi['feature']} | {fi['importance']:.4f} |")
        for warning in model.warnings:
            lines.append(f"- Warning: {warning}")

    # Validation trace
    if hasattr(result, "validation_trace") and result.validation_trace:
        lines.extend(["", "## Self-Correction Trace"])
        for entry in result.validation_trace:
            lines.append(f"- Round {entry['round']}: issues={entry['issues']}, correction={entry.get('correction', {}).get('action', 'N/A')}")

    if result.warnings:
        lines.extend(["", "## Validation Warnings"])
        for warning in result.warnings:
            lines.append(f"- {warning}")

    if result.recommendations:
        lines.extend(["", "## Recommended Next Checks"])
        for recommendation in result.recommendations:
            lines.append(f"- {recommendation}")
    return "\n".join(lines) + "\n"


def _format_interval(row: Dict[str, Any]) -> str:
    if row.get("ci_low") is not None and row.get("ci_high") is not None:
        return f"[{_format_metric(row.get('ci_low'))}, {_format_metric(row.get('ci_high'))}]"
    return "NA"


def save_markdown(result, path: str, max_terms: int = 12) -> None:
    """Save markdown report to file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(to_markdown(result, max_terms=max_terms))


def save_json(result, path: str, include_estimators: bool = False) -> None:
    """Save complete analysis result as JSON artifact."""
    import statgpu

    artifact = {
        "version": statgpu.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result.to_dict(include_estimators=include_estimators),
        "provenance": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "device_used": result.profile.device,
        },
    }
    if hasattr(result, "validation_trace"):
        artifact["validation_trace"] = _json_ready(result.validation_trace)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(artifact), handle, indent=2)


def save_notebook(result, data_source: str, output_path: str) -> None:
    """Generate a Jupyter notebook documenting the analysis."""
    try:
        import nbformat as nbf
    except ImportError:
        raise ImportError("pip install nbformat for notebook generation")

    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Overview
    cells.append(nbf.v4.new_markdown_cell(
        "# statgpu Automatic Analysis Report\n\n"
        f"**Data source**: `{data_source}`\n"
        f"**Task**: {result.profile.task_type}\n"
        f"**Samples**: {result.profile.n_samples}\n"
        f"**Features**: {result.profile.n_features}\n"
        f"**Device**: {result.profile.device}"
    ))

    # Cell 2: Agent Plan
    cells.append(nbf.v4.new_markdown_cell(
        "## Agent Plan\n\n"
        f"**Methods**: {', '.join(result.plan.methods)}\n\n"
        + "\n".join(f"- {r}" for r in result.plan.rationale)
    ))

    # Cell 3: Model Results
    for model in result.models:
        if model.error:
            cells.append(nbf.v4.new_markdown_cell(
                f"### {model.name}\n\n**Error**: {model.error}"
            ))
            continue

        lines = [f"### {model.name}"]
        if model.metrics:
            lines.append("**Metrics**: " + ", ".join(
                f"{k}={_format_metric(v)}" for k, v in model.metrics.items()
            ))
        if model.coefficients:
            lines.append("\n**Coefficients**:\n")
            lines.append("| term | estimate | p_value |")
            lines.append("| --- | ---: | ---: |")
            for row in model.coefficients[:12]:
                lines.append(
                    f"| {row.get('term', '')} | {_format_metric(row.get('estimate'))} | "
                    f"{_format_metric(row.get('p_value'))} |"
                )
        cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))

    # Cell 4: Validation Trace
    if hasattr(result, "validation_trace") and result.validation_trace:
        cells.append(nbf.v4.new_markdown_cell(
            "## Self-Correction Trace\n\n" +
            "\n".join(f"- Round {e['round']}: {e['issues']}" for e in result.validation_trace)
        ))

    # Cell 5: Warnings and Recommendations
    if result.warnings or result.recommendations:
        lines = ["## Warnings & Recommendations"]
        if result.warnings:
            lines.append("\n### Warnings")
            for w in result.warnings:
                lines.append(f"- {w}")
        if result.recommendations:
            lines.append("\n### Recommendations")
            for r in result.recommendations:
                lines.append(f"- {r}")
        cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))

    nb.cells = cells
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
