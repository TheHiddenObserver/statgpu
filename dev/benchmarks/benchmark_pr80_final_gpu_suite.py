#!/usr/bin/env python3
"""Final exact-head physical-GPU promotion suite for PR #80."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


CHILD_SUITES = (
    "dev/benchmarks/benchmark_pr80_group_gpu_suite.py",
    "dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py",
    "dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py",
)
SOURCE_FILES = (
    "dev/benchmarks/benchmark_pr80_final_gpu_suite.py",
    "dev/benchmarks/benchmark_pr80_group_gpu_suite.py",
    "dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py",
    "dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py",
    "dev/tests/test_pr80_final_gpu_suite_contract.py",
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


def _run_child(path, head):
    with tempfile.TemporaryDirectory(prefix="statgpu-pr80-final-") as temp_dir:
        output = Path(temp_dir) / (Path(path).stem + ".json")
        completed = subprocess.run(
            [sys.executable, path, "--output", str(output)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if not output.is_file():
            return {
                "suite": path,
                "returncode": int(completed.returncode),
                "passed": False,
                "error": "child suite did not create JSON output",
                "stdout_tail": completed.stdout[-4000:],
            }
        try:
            child = json.loads(output.read_text())
        except Exception as exc:
            return {
                "suite": path,
                "returncode": int(completed.returncode),
                "passed": False,
                "error": f"invalid JSON: {type(exc).__name__}: {exc}",
                "stdout_tail": completed.stdout[-4000:],
            }

        failures = list(child.get("gate_failures") or [])
        if completed.returncode != 0:
            failures.append(f"returncode={completed.returncode}")
        if child.get("source_commit") != head:
            failures.append(
                "source_commit mismatch: "
                f"expected {head}, got {child.get('source_commit')}"
            )
        if not bool(child.get("source_clean", False)):
            failures.append("source_clean is false")
        if not bool(child.get("source_clean_after", False)):
            failures.append("source_clean_after is false")

        return {
            "suite": path,
            "returncode": int(completed.returncode),
            "schema_version": child.get("schema_version"),
            "validation_tier": child.get("validation_tier"),
            "source_commit": child.get("source_commit"),
            "source_clean": child.get("source_clean"),
            "source_clean_after": child.get("source_clean_after"),
            "gate_failures": failures,
            "passed": not failures,
            "report": child,
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
        "validation_tier": "remote-full-final-promotion-suite",
        "source_commit": head,
        "source_clean": not dirty_before,
        "source_sha256": {
            path: _sha256(path)
            for path in SOURCE_FILES
            if Path(path).is_file()
        },
        "command": (
            "python dev/benchmarks/benchmark_pr80_final_gpu_suite.py "
            "--output <path>"
        ),
        "child_suites": {},
        "gate_failures": [],
    }
    if dirty_before:
        report["gate_failures"].append("source tree is dirty before final suite")
    if missing_sources:
        report["gate_failures"].append(
            "missing source files: " + ", ".join(missing_sources)
        )

    if not report["gate_failures"]:
        for path in CHILD_SUITES:
            result = _run_child(path, head)
            report["child_suites"][path] = result
            if not result["passed"]:
                report["gate_failures"].append(f"{path}: failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    dirty_after = _tree_dirty_excluding_output(output)
    report["source_clean_after"] = not dirty_after
    if dirty_after:
        report["gate_failures"].append("source tree is dirty after final suite")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
