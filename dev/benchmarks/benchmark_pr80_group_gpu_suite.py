#!/usr/bin/env python3
"""Canonical exact-source physical-GPU suite for PR #80 group penalties.

This outer gate binds all current correctness boundaries to one clean commit and
runs the specialized CuPy/Torch runners for:
- canonical and legacy-pickle Group Lasso layouts;
- exact Group Lasso objectives on correlated/weighted designs;
- Group MCP/SCAD LLA layout and surrogate scaling;
- weighted Group MCP/SCAD direct fit and CV;
- penalty-object CV alpha, constructor isolation, Adaptive Group Lasso, and
  selected final penalty snapshots.

The outer manifest is authoritative even when a historical sub-runner retains a
narrower local manifest. Every sub-report must bind to the same commit, report a
clean source tree, and contain no gate failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


RUNNERS = (
    "dev/benchmarks/benchmark_group_layout_gpu.py",
    "dev/benchmarks/benchmark_group_lasso_objective_gpu.py",
    "dev/benchmarks/benchmark_group_nonconvex_layout_gpu.py",
    "dev/benchmarks/benchmark_group_nonconvex_weighted_gpu.py",
    "dev/benchmarks/benchmark_group_cv_object_gpu.py",
)

SOURCE_FILES = (
    "dev/benchmarks/benchmark_pr80_group_gpu_suite.py",
    *RUNNERS,
    "dev/tests/test_pr80_adaptive_group_lipschitz_contract.py",
    "dev/tests/test_pr80_adaptive_group_penalty_contract.py",
    "dev/tests/test_pr80_adaptive_group_public_capability_contract.py",
    "dev/tests/test_pr80_group_clone_contract.py",
    "dev/tests/test_pr80_group_cv_list_input_contract.py",
    "dev/tests/test_pr80_group_cv_object_alpha_contract.py",
    "dev/tests/test_pr80_group_dimension_contract.py",
    "dev/tests/test_pr80_group_failed_refit_state_contract.py",
    "dev/tests/test_pr80_group_formula_contract.py",
    "dev/tests/test_pr80_group_inference_contract.py",
    "dev/tests/test_pr80_group_input_contract.py",
    "dev/tests/test_pr80_group_lasso_exact_objective_contract.py",
    "dev/tests/test_pr80_group_lasso_explicit_solver_contract.py",
    "dev/tests/test_pr80_group_lasso_nonquadratic_contract.py",
    "dev/tests/test_pr80_group_lasso_weighted_contract.py",
    "dev/tests/test_pr80_group_layout_contract.py",
    "dev/tests/test_pr80_group_library_clone_contract.py",
    "dev/tests/test_pr80_group_lla_surrogate_contract.py",
    "dev/tests/test_pr80_group_nonconvex_capability_contract.py",
    "dev/tests/test_pr80_group_nonconvex_convergence_contract.py",
    "dev/tests/test_pr80_group_nonconvex_hyperparameter_contract.py",
    "dev/tests/test_pr80_group_nonconvex_layout_contract.py",
    "dev/tests/test_pr80_group_nonconvex_pickle_contract.py",
    "dev/tests/test_pr80_group_nonconvex_weighted_contract.py",
    "dev/tests/test_pr80_group_penalty_object_isolation_contract.py",
    "dev/tests/test_pr80_group_warm_start_transaction_contract.py",
    "statgpu/glm_core/_solver_utils.py",
    "statgpu/linear_model/penalized/__init__.py",
    "statgpu/linear_model/penalized/_base.py",
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "statgpu/linear_model/penalized/_group_penalty_model_contract.py",
    "statgpu/linear_model/penalized/_penalized_cv.py",
    "statgpu/penalties/__init__.py",
    "statgpu/penalties/_categories.py",
    "statgpu/penalties/_group_clone_contract.py",
    "statgpu/penalties/_group_dimension_contract.py",
    "statgpu/penalties/_group_lasso.py",
    "statgpu/penalties/_group_lasso_layout.py",
    "statgpu/penalties/_group_mcp.py",
    "statgpu/penalties/_group_nonconvex_layout.py",
    "statgpu/penalties/_group_scad.py",
    "statgpu/solvers/__init__.py",
    "statgpu/solvers/_adaptive_group_lipschitz_contract.py",
    "statgpu/solvers/_admm.py",
    "statgpu/solvers/_fista.py",
    "statgpu/solvers/_fista_bb.py",
    "statgpu/solvers/_fista_lla.py",
    "statgpu/solvers/_fista_lla_group_contract.py",
    "statgpu/solvers/_utils.py",
)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_subrunner(path, head):
    with tempfile.TemporaryDirectory(prefix="statgpu-pr80-group-") as temp_dir:
        output = Path(temp_dir) / (Path(path).stem + ".json")
        completed = subprocess.run(
            [sys.executable, path, "--output", str(output)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if not output.exists():
            return {
                "runner": path,
                "returncode": int(completed.returncode),
                "passed": False,
                "error": "runner did not create its JSON output",
                "stdout_tail": completed.stdout[-4000:],
            }
        try:
            subreport = json.loads(output.read_text())
        except Exception as exc:
            return {
                "runner": path,
                "returncode": int(completed.returncode),
                "passed": False,
                "error": f"invalid JSON: {type(exc).__name__}: {exc}",
                "stdout_tail": completed.stdout[-4000:],
            }

        failures = list(subreport.get("gate_failures") or [])
        source_commit = subreport.get("source_commit")
        source_clean = bool(subreport.get("source_clean", False))
        if source_commit != head:
            failures.append(
                f"source_commit mismatch: expected {head}, got {source_commit}"
            )
        if not source_clean:
            failures.append("sub-runner source_clean is false")
        if completed.returncode != 0:
            failures.append(f"runner returncode={completed.returncode}")

        return {
            "runner": path,
            "returncode": int(completed.returncode),
            "source_commit": source_commit,
            "source_clean": source_clean,
            "schema_version": subreport.get("schema_version"),
            "gate_failures": failures,
            "backends": subreport.get("backends"),
            "api_contract": subreport.get("api_contract"),
            "passed": not failures,
            "stdout_tail": completed.stdout[-4000:] if failures else "",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
            "python dev/benchmarks/benchmark_pr80_group_gpu_suite.py "
            "--output <path>"
        ),
        "subrunners": {},
        "gate_failures": [],
    }

    if dirty_before:
        report["gate_failures"].append("source tree is dirty before suite")
    if missing_sources:
        report["gate_failures"].append(
            "missing source files: " + ", ".join(missing_sources)
        )

    if not report["gate_failures"]:
        for runner in RUNNERS:
            result = _run_subrunner(runner, head)
            report["subrunners"][runner] = result
            if not result["passed"]:
                report["gate_failures"].append(f"{runner}: failed")

    dirty_after = bool(_git("status", "--porcelain"))
    report["source_clean_after"] = not dirty_after
    if dirty_after:
        report["gate_failures"].append("source tree is dirty after suite")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
