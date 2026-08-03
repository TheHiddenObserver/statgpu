"""Hosted structural gate for the canonical Cox CV GPU order suite."""

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


def test_canonical_cox_cv_order_suite_binds_runtime_cache_tests_and_runner():
    suite = Path("dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py")
    tree = ast.parse(suite.read_text())
    inner = _assignment(tree, "INNER_RUNNER")
    source_files = _assignment(tree, "SOURCE_FILES")

    assert inner == "dev/benchmarks/benchmark_cox_cv_penalty_order_gpu.py"
    required = {
        suite.as_posix(),
        inner,
        "dev/tests/test_pr80_cox_cv_grid_failed_refit_state.py",
        "dev/tests/test_pr80_cox_cv_penalty_order_contract.py",
        "dev/tests/test_pr80_cox_cv_penalty_order_integration.py",
        "statgpu/cross_validation/_base.py",
        "statgpu/cross_validation/_grid_validation.py",
        "statgpu/survival/__init__.py",
        "statgpu/survival/_cox_cv.py",
        "statgpu/survival/_cox_cv_penalty_order_contract.py",
        "statgpu/survival/_risk_sets.py",
    }
    assert required.issubset(set(source_files))
    assert all(Path(path).is_file() for path in source_files)
