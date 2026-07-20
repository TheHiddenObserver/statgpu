#!/usr/bin/env python3
"""One-shot PR #76 review-fix patch, executed by a temporary branch workflow."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    file_path = ROOT / path
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match in {path}, found {count}: {old[:80]!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    file_path = ROOT / path
    text = file_path.read_text(encoding="utf-8")
    if marker in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    file_path.write_text(text + addition, encoding="utf-8")


# ---------------------------------------------------------------------------
# Python canonical contract: own the behavior in cli.py, not package hooks.
# ---------------------------------------------------------------------------
replace_once(
    "dev/benchmarks/frontend_data/cli.py",
    '''            run["source"]["source_id"] = manifest_src["source_id"]
            run["source"]["original_path"] = manifest_src.get("original_path", "")
            run["comparison_id"] = manifest_src.get("comparison_id", manifest_src["source_id"])
''',
    '''            run["source"]["source_id"] = manifest_src["source_id"]
            run["source"]["original_path"] = manifest_src.get("original_path", "")
            if manifest_src.get("sha256"):
                run["source"]["sha256"] = manifest_src["sha256"]
            run["comparison_id"] = manifest_src.get("comparison_id", manifest_src["source_id"])
''',
)

replace_once(
    "dev/benchmarks/frontend_data/cli.py",
    "    validator = Validator(schema)\n",
    "    validator = Validator(schema, format_checker=Validator.FORMAT_CHECKER)\n",
)

replace_once(
    "dev/benchmarks/frontend_data/cli.py",
    '''    fw_policy = {f["framework_id"]: f.get("backend_policy", "forbidden") for f in output.get("frameworks", [])}
    runs_by_id = {run.get("run_id"): run for run in runs}

    for run in runs:
''',
    '''    fw_policy = {f["framework_id"]: f.get("backend_policy", "forbidden") for f in output.get("frameworks", [])}
    runs_by_id = {run.get("run_id"): run for run in runs}
    manifest_sources_by_id = (
        {source.get("source_id"): source for source in manifest.get("sources", [])}
        if manifest
        else {}
    )

    for run in runs:
''',
)

replace_once(
    "dev/benchmarks/frontend_data/cli.py",
    '''        # Canonical IDs
        if run.get("source", {}).get("source_id", "").startswith("transitional:"):
            if manifest:
                errors.append(f"{rid}: transitional source_id in canonical mode")
        if run.get("case_id", "").startswith("legacy-"):
            if manifest:
                errors.append(f"{rid}: legacy case_id in canonical mode")
''',
    '''        # Canonical IDs and source provenance
        source = run.get("source", {})
        source_id = source.get("source_id", "")
        if source_id.startswith("transitional:"):
            if manifest:
                errors.append(f"{rid}: transitional source_id in canonical mode")
        if run.get("case_id", "").startswith("legacy-"):
            if manifest:
                errors.append(f"{rid}: legacy case_id in canonical mode")
        if manifest:
            manifest_source = manifest_sources_by_id.get(source_id)
            if manifest_source is None:
                errors.append(f"{rid}: source_id '{source_id}' not found in manifest")
            else:
                expected_hash = manifest_source.get("sha256")
                if expected_hash and source.get("sha256") != expected_hash:
                    errors.append(
                        f"{rid}: source.sha256 does not match manifest for '{source_id}'"
                    )
''',
)

init_path = ROOT / "dev/benchmarks/frontend_data/__init__.py"
init_path.write_text(
    '''from __future__ import annotations
"""Benchmark frontend data generation package."""

from .cli import (
    generate,
    validate_output,
    validate_against_schema,
    main,
    get_git_sha,
    _write_transactional,
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
''',
    encoding="utf-8",
)

append_once(
    "dev/tests/test_frontend_contracts.py",
    "def test_semantic_validation_rejects_manifest_source_hash_mismatch()",
    '''\n\ndef test_semantic_validation_rejects_manifest_source_hash_mismatch() -> None:
    from dev.benchmarks.frontend_data.cli import validate_semantic
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    output, _, _ = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=manifest,
    )
    output["runs"][0]["source"]["sha256"] = "0" * 64

    errors = validate_semantic(output, manifest=manifest)
    assert any("source.sha256 does not match manifest" in error for error in errors)
''',
)

# ---------------------------------------------------------------------------
# Shared chart identity: solver-family identity deliberately excludes solver.
# ---------------------------------------------------------------------------
append_once(
    "frontend/src/identity.ts",
    "export function chartSolverFamilyIdentity",
    '''\n\n/** Chart family identity used when Focused mode prefers Auto per comparable family. */
export function chartSolverFamilyIdentity(
  run: Run,
  includeSession: boolean,
): readonly unknown[] {
  const common: unknown[] = [
    run.comparison_id,
    run.env_id,
    run.model_id,
    run.case_id,
    run.method_config_id,
    run.variant ?? null,
    run.loss ?? null,
    run.penalty ?? null,
    run.scale.scale_key,
  ];
  return includeSession
    ? [run.comparison_id, run.benchmark_session_id ?? null, ...common.slice(1)]
    : common;
}
''',
)

# ---------------------------------------------------------------------------
# Timing chart: retain manual-only families and make multi-scale labels unique.
# ---------------------------------------------------------------------------
replace_once(
    "frontend/src/charts/TimingChart.ts",
    "import { chartGroupIdentity } from '../identity';\n",
    "import { chartGroupIdentity, chartSolverFamilyIdentity } from '../identity';\n",
)
replace_once(
    "frontend/src/charts/TimingChart.ts",
    '''function groupKey(run: Run): string {
  return JSON.stringify(chartGroupIdentity(run, false));
}
''',
    '''function groupKey(run: Run): string {
  return JSON.stringify(chartGroupIdentity(run, false));
}

function solverFamilyKey(run: Run): string {
  return JSON.stringify(chartSolverFamilyIdentity(run, false));
}
''',
)
replace_once(
    "frontend/src/charts/TimingChart.ts",
    "function selectTimingRuns(runs: Run[], state: AppState): TimingSelection {\n",
    "export function selectTimingRuns(runs: Run[], state: AppState): TimingSelection {\n",
)
replace_once(
    "frontend/src/charts/TimingChart.ts",
    '''  const dispatchGroupKeys = new Set(
    focused
      .filter(
        (run) =>
          run.framework === 'statgpu' &&
          (run.solver_kind === 'dispatch' || run.solver === 'auto'),
      )
      .map(groupKey),
  );
  if (dispatchGroupKeys.size > 0) {
    focused = focused.filter((run) => dispatchGroupKeys.has(groupKey(run)));
    notes.push('Auto/best solver groups');
  }
''',
    '''  const dispatchRuns = focused.filter(
    (run) =>
      run.framework === 'statgpu' &&
      (run.solver_kind === 'dispatch' || run.solver === 'auto'),
  );
  const dispatchGroupKeys = new Set(dispatchRuns.map(groupKey));
  const familiesWithDispatch = new Set(dispatchRuns.map(solverFamilyKey));
  if (familiesWithDispatch.size > 0) {
    const before = focused.length;
    focused = focused.filter(
      (run) =>
        !familiesWithDispatch.has(solverFamilyKey(run)) ||
        dispatchGroupKeys.has(groupKey(run)),
    );
    if (focused.length < before) notes.push('Auto/best solver groups');
  }
''',
)
replace_once(
    "frontend/src/charts/TimingChart.ts",
    "function formatGroupLabel(run: Run, focused: boolean): string {\n",
    "export function formatGroupLabel(run: Run, focused: boolean, includeScale = false): string {\n",
)
replace_once(
    "frontend/src/charts/TimingChart.ts",
    "    return [model, penalty, solverPart].filter(Boolean).join(' · ');\n",
    "    return [model, penalty, solverPart, includeScale ? run.scale.label : null]\n      .filter(Boolean)\n      .join(' · ');\n",
)
replace_once(
    "frontend/src/charts/TimingChart.ts",
    '''  const seriesMeta = new Map<string, TimingSeries>();
  const isFocused = state.chartViewMode === 'focused';

  for (const run of timingRuns) {
''',
    '''  const seriesMeta = new Map<string, TimingSeries>();
  const isFocused = state.chartViewMode === 'focused';
  const includeScaleInFocusedLabel = isFocused && state.selectedScaleKeys.size > 1;

  for (const run of timingRuns) {
''',
)
replace_once(
    "frontend/src/charts/TimingChart.ts",
    "        label: formatGroupLabel(run, isFocused),\n",
    "        label: formatGroupLabel(run, isFocused, includeScaleInFocusedLabel),\n",
)

# ---------------------------------------------------------------------------
# Speedup chart: same per-family Auto preference and unique focused labels.
# ---------------------------------------------------------------------------
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    "import { CHART_STYLE } from '../utils/theme';\n",
    "import { CHART_STYLE } from '../utils/theme';\nimport { chartGroupIdentity, chartSolverFamilyIdentity } from '../identity';\n",
)
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    '''function formatSeries(run: Run): string {
''',
    '''function groupKey(run: Run): string {
  return JSON.stringify(chartGroupIdentity(run, false));
}

function solverFamilyKey(run: Run): string {
  return JSON.stringify(chartSolverFamilyIdentity(run, false));
}

function formatSeries(run: Run): string {
''',
)
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    "function formatRunLabel(run: Run, focused: boolean): string {\n",
    "export function formatRunLabel(run: Run, focused: boolean, includeScale = false): string {\n",
)
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    '''  if (focused) {
    return `${[model, penalty, formatFocusedSeries(run)].filter(Boolean).join(' · ')}${reported}`;
  }
''',
    '''  if (focused) {
    const solver = run.solver_display ?? run.solver ?? 'unknown';
    const solverPart = run.solver === 'auto' || run.solver_kind === 'dispatch'
      ? null
      : solver;
    return `${[
      model,
      penalty,
      solverPart,
      formatFocusedSeries(run),
      includeScale ? run.scale.label : null,
    ].filter(Boolean).join(' · ')}${reported}`;
  }
''',
)
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    "function selectSpeedupRuns(runs: Run[], state: AppState): SpeedupSelection {\n",
    "export function selectSpeedupRuns(runs: Run[], state: AppState): SpeedupSelection {\n",
)
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    '''  const dispatchRows = focused.filter(
    (run) => run.solver_kind === 'dispatch' || run.solver === 'auto',
  );
  if (dispatchRows.length > 0) {
    focused = dispatchRows;
    notes.push('Auto/best solver rows');
  }
''',
    '''  const dispatchRows = focused.filter(
    (run) => run.solver_kind === 'dispatch' || run.solver === 'auto',
  );
  const dispatchGroupKeys = new Set(dispatchRows.map(groupKey));
  const familiesWithDispatch = new Set(dispatchRows.map(solverFamilyKey));
  if (familiesWithDispatch.size > 0) {
    const before = focused.length;
    focused = focused.filter(
      (run) =>
        !familiesWithDispatch.has(solverFamilyKey(run)) ||
        dispatchGroupKeys.has(groupKey(run)),
    );
    if (focused.length < before) notes.push('Auto/best solver rows');
  }
''',
)
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    '''  const isFocused = state.chartViewMode === 'focused';
  const limit = isFocused ? 18 : state.speedupChartLimit;
''',
    '''  const isFocused = state.chartViewMode === 'focused';
  const includeScaleInFocusedLabel = isFocused && state.selectedScaleKeys.size > 1;
  const limit = isFocused ? 18 : state.speedupChartLimit;
''',
)
replace_once(
    "frontend/src/charts/SpeedupChart.ts",
    "        data: displayRuns.map((run) => formatRunLabel(run, isFocused)),\n",
    "        data: displayRuns.map((run) =>\n          formatRunLabel(run, isFocused, includeScaleInFocusedLabel)),\n",
)

print("Applied PR #76 review fixes")
