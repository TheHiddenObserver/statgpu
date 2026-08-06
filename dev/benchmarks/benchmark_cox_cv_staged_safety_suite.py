#!/usr/bin/env python3
"""Canonical exact-source suite for CoxPHCV staged-screening safety."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    from ._exact_source_runtime import prepare_exact_source_runtime
except ImportError:
    try:
        from dev.benchmarks._exact_source_runtime import (
            prepare_exact_source_runtime,
        )
    except ImportError:  # direct or importlib file execution
        import importlib.util

        _helper_path = Path(__file__).with_name("_exact_source_runtime.py")
        _helper_spec = importlib.util.spec_from_file_location(
            "_statgpu_exact_source_runtime",
            _helper_path,
        )
        if _helper_spec is None or _helper_spec.loader is None:
            raise ImportError(f"cannot load exact-source helper: {_helper_path}")
        _helper_module = importlib.util.module_from_spec(_helper_spec)
        _helper_spec.loader.exec_module(_helper_module)
        prepare_exact_source_runtime = (
            _helper_module.prepare_exact_source_runtime
        )


INNER_RUNNER = "dev/benchmarks/benchmark_cox_cv_staged_safety_gpu.py"
SOURCE_FILES = (
    "dev/benchmarks/_exact_source_runtime.py",
    "dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py",
    "dev/benchmarks/benchmark_cox_cv_staged_safety_gpu.py",
    "dev/tests/test_pr80_cox_cv_split_lifecycle_contract.py",
    "dev/tests/test_pr80_cox_cv_staged_safety_contract.py",
    "dev/tests/test_pr80_target_transfer_overflow_cache.py",
    "dev/tests/test_pr80_cox_cv_staged_safety_suite_contract.py",
    "dev/tests/test_pr80_exact_source_runtime_provenance.py",
    "statgpu/survival/__init__.py",
    "statgpu/survival/_cox.py",
    "statgpu/survival/_cox_cv.py",
    "statgpu/survival/_cox_cv_penalty_order_contract.py",
    "statgpu/survival/_cox_cv_split_lifecycle_contract.py",
    "statgpu/survival/_cox_cv_staged_safety_contract.py",
    "statgpu/survival/_risk_sets.py",
)


def _git(root, *args):
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_dirty_excluding_output(root, output):
    output_path = output.resolve()
    try:
        output_relative = output_path.relative_to(root).as_posix()
    except ValueError:
        output_relative = None
    retained = []
    for line in _git(root, "status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"') if len(line) >= 4 else ""
        if output_relative is not None and path == output_relative:
            continue
        retained.append(line)
    return bool(retained)


def _run_inner(head, *, root, runtime_env):
    with tempfile.TemporaryDirectory(prefix="statgpu-cox-cv-staged-") as temp_dir:
        output = Path(temp_dir) / "inner.json"
        completed = subprocess.run(
            [sys.executable, str(root / INNER_RUNNER), "--output", str(output)],
            cwd=root,
            env=runtime_env,
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
        if not bool((inner.get("cross_backend") or {}).get("passed", False)):
            failures.append("inner cross-backend parity did not pass")

        failures = list(dict.fromkeys(failures))
        return {
            "returncode": int(completed.returncode),
            "schema_version": inner.get("schema_version"),
            "source_commit": inner.get("source_commit"),
            "source_clean": inner.get("source_clean"),
            "source_clean_after": inner.get("source_clean_after"),
            "backends": backends,
            "cross_backend": inner.get("cross_backend"),
            "gate_failures": failures,
            "passed": not failures,
            "stdout_tail": completed.stdout[-4000:] if failures else "",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()

    root, runtime_env, provenance, provenance_failures = (
        prepare_exact_source_runtime(
            (
                "statgpu",
                "statgpu.survival",
                "statgpu.survival._cox_cv",
                "statgpu.survival._cox_cv_staged_safety_contract",
            )
        )
    )
    head = _git(root, "rev-parse", "HEAD")
    dirty_before = bool(_git(root, "status", "--porcelain"))
    missing_sources = [
        path for path in SOURCE_FILES if not (root / path).is_file()
    ]
    report = {
        "schema_version": 2,
        "validation_tier": "remote-full-canonical-suite",
        "source_commit": head,
        "source_clean": not dirty_before,
        "runtime_import_provenance": provenance,
        "source_sha256": {
            path: _sha256(root / path)
            for path in SOURCE_FILES
            if (root / path).is_file()
        },
        "command": (
            "python dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py "
            "--output <path>"
        ),
        "inner_runner": None,
        "gate_failures": list(provenance_failures),
    }
    if dirty_before:
        report["gate_failures"].append("source tree is dirty before suite")
    if missing_sources:
        report["gate_failures"].append(
            "missing source files: " + ", ".join(missing_sources)
        )
    if not report["gate_failures"]:
        report["inner_runner"] = _run_inner(
            head,
            root=root,
            runtime_env=runtime_env,
        )
        if not report["inner_runner"]["passed"]:
            report["gate_failures"].append("inner staged safety runner failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    dirty_after = _tree_dirty_excluding_output(root, output)
    report["source_clean_after"] = not dirty_after
    if dirty_after:
        report["gate_failures"].append("source tree is dirty after suite")
    report["gate_failures"] = list(dict.fromkeys(report["gate_failures"]))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
