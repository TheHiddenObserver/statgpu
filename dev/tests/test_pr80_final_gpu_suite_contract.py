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


def _dict_assignment_value(tree, assignment_name, key):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for dict_key, dict_value in zip(node.value.keys, node.value.values):
            if isinstance(dict_key, ast.Constant) and dict_key.value == key:
                return ast.literal_eval(dict_value)
    raise AssertionError(f"missing {assignment_name}[{key!r}]")


def test_final_gpu_suite_runs_all_canonical_exact_head_suites():
    final_suite = Path("dev/benchmarks/benchmark_pr80_final_gpu_suite.py")
    tree = ast.parse(final_suite.read_text())
    child_suites = _assignment(tree, "CHILD_SUITES")
    source_files = _assignment(tree, "SOURCE_FILES")

    assert child_suites == (
        "dev/benchmarks/benchmark_pr80_group_gpu_suite.py",
        "dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py",
        "dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py",
    )
    required = {
        final_suite.as_posix(),
        *child_suites,
        "dev/tests/test_pr80_final_gpu_suite_contract.py",
    }
    assert required.issubset(set(source_files))
    assert len(source_files) == len(set(source_files))
    assert all(Path(path).is_file() for path in source_files)
    assert _dict_assignment_value(tree, "report", "schema_version") == 3
