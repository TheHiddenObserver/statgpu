"""Hosted structural contract for the final PR #80 GPU promotion suite."""

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


def test_final_gpu_suite_runs_both_canonical_exact_head_suites():
    final_suite = Path("dev/benchmarks/benchmark_pr80_final_gpu_suite.py")
    tree = ast.parse(final_suite.read_text())
    child_suites = _assignment(tree, "CHILD_SUITES")
    source_files = _assignment(tree, "SOURCE_FILES")

    assert child_suites == (
        "dev/benchmarks/benchmark_pr80_group_gpu_suite.py",
        "dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py",
    )
    required = {
        final_suite.as_posix(),
        *child_suites,
        "dev/tests/test_pr80_final_gpu_suite_contract.py",
    }
    assert required.issubset(set(source_files))
    assert len(source_files) == len(set(source_files))
    assert all(Path(path).is_file() for path in source_files)
