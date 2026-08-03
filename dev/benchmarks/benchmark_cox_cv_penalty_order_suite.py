#!/usr/bin/env python3
"""Canonical exact-source suite for CoxPHCV custom-grid order semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


INNER_RUNNER = "dev/benchmarks/benchmark_cox_cv_penalty_order_gpu.py"
SOURCE_FILES = (
    "dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py",
    "dev/benchmarks/benchmark_cox_cv_penalty_order_gpu.py",
    "dev/tests/test_pr80_cox_cv_grid_failed_refit_state.py",
    "dev/tests/test_pr80_cox_cv_penalty_order_contract.py",
    "dev/tests/test_pr80_cox_cv_penalty_order_docs.py",
    "dev/tests/test_pr80_cox_cv_penalty_order_integration.py",
    "dev/tests/test_pr80_cox_cv_penalty_order_suite_contract.py",
    "dev/tests/test_pr80_cox_cv_scalar_detail_consistency.py",
    "statgpu/backends/__init__.py",
    "statgpu/cross_validation/_base.py",
    "statgpu/cross_validation/_grid_validation.py",
    "statgpu/linear_model/penalized/__init__.py",
    "statgpu/linear_model/penalized/_penalized_cox.py",
    "statgpu/linear_model/penalized/_penalized_cox_cv.py",
    "statgpu/linear_model/penalized/_penalized_cox_public_contract.py",
    "statgpu/survival/__init__.py",
    "statgpu/survival/_cox.py",
    "statgpu/survival/_cox_cv.py",
    "statgpu/survival/_cox_cv_penalty_order_contract.py",
    "statgpu/survival/_cox_fit_adapter.py",
    "statgpu/survival/_risk_sets.py",
)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_dirty_excluding_output(output):
    output_path = output.resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    try:
        output_relative = output_path.relative_to(root).as_posix()
    except ValueError:
        output_relative = None
    retained = []
    for line in _git("status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"') if len(line) >= 4 else ""
        if output_relative is not None and path == output_relative:
            continue
        retained.append(line)
    return bool(retained)


def _run_inner(head):
    with tempfile.TemporaryDirectory(prefix="statgpu-cox-cv-order-") as temp_dir:
        output = Path(temp_dir) / "inner.json"
        completed = subprocess.run(
            [sys.executable, INNER_RUNNER, "--output", str(output)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if not output.is_file():
            return {
                "returncode": int(completed.returncode),
                "passed": False,
                "error": "inner runner did not create JSON output",
                "stdout_tail": completed.stdout[-4000:],
            }
        try:
            inner = json.loads(output.read_text())
        except Exception as exc:
            return {
                "returncode": int(completed.returncode),
                "passed": False,
                "error": f"invalid JSON: {type(exc).__name__}: {exc}",
                "stdout_tail": completed.stdout[-4000:],
            }

        failures = list(inner.get("gate_failures") or [])
        if completed.returncode != 0:
            failures.append(f"inner returncode={completed.returncode}")
        if inner.get("source_commit") != head:
            failures.append(
                "inner source_commit mismatch: "
                f"expected {head}, got {inner.get('source_commit')}"
            )
        if not bool(inner.get("source_clean", False)):
            failures.append("inner source_clean is false")
        if not bool(inner.get("source_clean_after", False)):
            failures.append("inner source_clean_after is false")
        backends = inner.get("backends") or {}
        for name in ("cupy", "torch"):
            if not bool((backends.get(name) or {}).get("passed", False)):
                failures.append(f"inner {name} backend did not pass")

        return {
            "returncode": int(completed.returncode),
            "schema_version": inner.get("schema_version"),
            "source_commit": inner.get("source_commit"),
            "source_clean": inner.get("source_clean"),
            "source_clean_after": inner.get("source_clean_after"),
            "backends": backends,
            "gate_failures": failures,
            "passed": not failures,
            "stdout_tail": completed.stdout[-4000:] if failures else "",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)

    head = _git("rev-parse", "HEAD")
    dirty_before = bool(_git("status", "--porcelain"))
    missing_sources = [path for path in SOURCE_FILES if not Path(path).is_file()]
    report = {
        "schema_version": 1,
        "validation_tier": "remote-full-canonical-suite",
        "source_commit": head,
        "source_clean": not dirty_before,
        "source_sha256": {
            path: _sha256(path)
            for path in SOURCE_FILES
            if Path(path).is_file()
        },
        "command": (
            "python dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py "
            "--output <path>"
        ),
        "inner_runner": None,
        "gate_failures": [],
    }
    if dirty_before:
        report["gate_failures"].append("source tree is dirty before suite")
    if missing_sources:
        report["gate_failures"].append(
            "missing source files: " + ", ".join(missing_sources)
        )
    if not report["gate_failures"]:
        report["inner_runner"] = _run_inner(head)
        if not report["inner_runner"]["passed"]:
            report["gate_failures"].append("inner GPU order runner failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    dirty_after = _tree_dirty_excluding_output(output)
    report["source_clean_after"] = not dirty_after
    if dirty_after:
        report["gate_failures"].append("source tree is dirty after suite")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
