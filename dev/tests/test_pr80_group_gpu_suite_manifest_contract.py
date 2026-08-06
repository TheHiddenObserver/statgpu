"""Hosted structural contract for the canonical physical-GPU suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_suite_module():
    path = Path("dev/benchmarks/benchmark_pr80_group_gpu_suite.py")
    spec = importlib.util.spec_from_file_location("pr80_group_gpu_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_group_gpu_suite_has_complete_existing_manifest():
    suite = _load_suite_module()
    assert len(suite.RUNNERS) == 5
    assert len(set(suite.RUNNERS)) == len(suite.RUNNERS)
    assert len(set(suite.SOURCE_FILES)) == len(suite.SOURCE_FILES)

    missing = [path for path in suite.SOURCE_FILES if not Path(path).is_file()]
    assert missing == []
    assert set(suite.RUNNERS).issubset(set(suite.SOURCE_FILES))

    required = {
        "statgpu/linear_model/penalized/_group_penalty_model_contract.py",
        "statgpu/penalties/_categories.py",
        "statgpu/penalties/_group_lasso_layout.py",
        "statgpu/penalties/_group_nonconvex_layout.py",
        "statgpu/solvers/_adaptive_group_lipschitz_contract.py",
        "statgpu/solvers/_fista_lla_group_contract.py",
        "dev/tests/test_pr80_adaptive_group_public_capability_contract.py",
        "dev/tests/test_pr80_group_cv_object_alpha_contract.py",
        "dev/tests/test_pr80_group_failed_refit_state_contract.py",
        "dev/tests/test_pr80_group_nonconvex_weighted_contract.py",
    }
    assert required.issubset(set(suite.SOURCE_FILES))


def test_canonical_group_gpu_suite_subrunner_names_are_exact():
    suite = _load_suite_module()
    assert suite.RUNNERS == (
        "dev/benchmarks/benchmark_group_layout_gpu.py",
        "dev/benchmarks/benchmark_group_lasso_objective_gpu.py",
        "dev/benchmarks/benchmark_group_nonconvex_layout_gpu.py",
        "dev/benchmarks/benchmark_group_nonconvex_weighted_gpu.py",
        "dev/benchmarks/benchmark_group_cv_object_gpu.py",
    )
