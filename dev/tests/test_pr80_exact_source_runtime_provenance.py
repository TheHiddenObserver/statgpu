"""Exact-checkout runtime import provenance contracts for GPU suites."""

from __future__ import annotations

import os
from pathlib import Path

from dev.benchmarks._exact_source_runtime import prepare_exact_source_runtime


SUITES = (
    "dev/benchmarks/benchmark_pr80_final_gpu_suite.py",
    "dev/benchmarks/benchmark_pr80_group_gpu_suite.py",
    "dev/benchmarks/benchmark_cox_cv_penalty_order_suite.py",
    "dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py",
)


def test_runtime_probe_hashes_modules_from_current_checkout():
    root, runtime_env, provenance, failures = prepare_exact_source_runtime(
        (
            "statgpu",
            "statgpu.survival",
            "statgpu.survival._cox_cv",
        )
    )

    assert failures == []
    assert provenance["passed"] is True
    assert runtime_env["PYTHONNOUSERSITE"] == "1"
    assert Path(runtime_env["PYTHONPATH"].split(os.pathsep)[0]).resolve() == root
    for module in provenance["modules"].values():
        path = Path(module["path"]).resolve()
        assert path.is_relative_to(root)
        assert module["relative_path"] is not None
        assert len(module["sha256"]) == 64


def test_runtime_probe_precedes_conflicting_pythonpath(monkeypatch, tmp_path):
    fake = tmp_path / "conflict"
    package = fake / "statgpu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('conflicting statgpu import was selected')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(fake))

    root, runtime_env, provenance, failures = prepare_exact_source_runtime(
        ("statgpu",)
    )

    assert failures == []
    assert provenance["passed"] is True
    assert Path(
        provenance["modules"]["statgpu"]["path"]
    ).resolve() == (root / "statgpu" / "__init__.py").resolve()
    entries = runtime_env["PYTHONPATH"].split(os.pathsep)
    assert Path(entries[0]).resolve() == root
    assert Path(entries[1]).resolve() == fake.resolve()


def test_canonical_suites_pass_controlled_runtime_to_children():
    helper = "dev/benchmarks/_exact_source_runtime.py"
    provenance_test = "dev/tests/test_pr80_exact_source_runtime_provenance.py"
    for path in SUITES:
        source = Path(path).read_text()
        assert "prepare_exact_source_runtime" in source
        assert "runtime_import_provenance" in source
        assert "env=runtime_env" in source
        assert "cwd=root" in source
        assert helper in source
        assert provenance_test in source
