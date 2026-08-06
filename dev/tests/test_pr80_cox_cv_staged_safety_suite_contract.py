"""Hosted structural contract for the staged-safety physical GPU suite."""

from __future__ import annotations

import ast
from pathlib import Path


def _assignment(tree, name):
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")


def test_staged_safety_suite_binds_runtime_tests_and_runner():
    suite = Path("dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py")
    tree = ast.parse(suite.read_text())
    inner = _assignment(tree, "INNER_RUNNER")
    source_files = _assignment(tree, "SOURCE_FILES")

    assert inner == "dev/benchmarks/benchmark_cox_cv_staged_safety_gpu.py"
    required = {
        suite.as_posix(),
        inner,
        "dev/tests/test_pr80_cox_cv_staged_safety_contract.py",
        "dev/tests/test_pr80_cox_cv_staged_safety_suite_contract.py",
        "statgpu/survival/__init__.py",
        "statgpu/survival/_cox_cv.py",
        "statgpu/survival/_cox_cv_penalty_order_contract.py",
        "statgpu/survival/_cox_cv_staged_safety_contract.py",
    }
    assert required.issubset(set(source_files))
    assert len(source_files) == len(set(source_files))
    assert all(Path(path).is_file() for path in source_files)
