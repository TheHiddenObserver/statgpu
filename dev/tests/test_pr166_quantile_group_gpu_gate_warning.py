"""Physical-gate convergence warning contracts for PR #166."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_GATE_PATH = REPO_ROOT / "dev" / "benchmarks" / "run_quantile_group_lla_gpu_gate.py"
GROUP_VALIDATOR_PATH = REPO_ROOT / "dev" / "benchmarks" / "validate_quantile_group_lla_gpu.py"
SMOOTH_GATE_PATH = REPO_ROOT / "dev" / "benchmarks" / "run_quantile_smooth_fista_gpu_gate.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _warning_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "warning_runner.py"
    runner.write_text(
        """
import argparse
import warnings
from pathlib import Path
from statgpu.solvers._convergence import ConvergenceWarning

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
warnings.warn("sentinel convergence warning", ConvergenceWarning)
Path(args.output).write_text('{"status": "unexpected-success"}\\n', encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    return runner


def test_group_gate_and_validator_freeze_schema_v3_transfer_contract():
    gate = _load_module(GROUP_GATE_PATH, "pr166_quantile_group_gate_schema")
    validator = _load_module(
        GROUP_VALIDATOR_PATH,
        "pr166_quantile_group_validator_schema",
    )
    source = GROUP_VALIDATOR_PATH.read_text(encoding="utf-8")

    assert gate.GROUP_SCHEMA_VERSION == 3
    assert validator.SCHEMA_VERSION == 3
    assert "derivative_transfer_contract" in source
    assert "max_host_values_per_factory_transfer" in source
    assert "_explicit_group_fista_lla_transfer_probe" in source


def test_group_gate_promotes_statgpu_convergence_warning_to_failure(tmp_path):
    gate = _load_module(GROUP_GATE_PATH, "pr166_quantile_group_gate")
    runner = _warning_runner(tmp_path)
    output = tmp_path / "group-payload.json"

    with pytest.raises(subprocess.CalledProcessError):
        gate._run_inner(runner, output)

    assert not output.exists()


def test_smooth_gate_promotes_statgpu_convergence_warning_to_failure(
    tmp_path, monkeypatch
):
    gate = _load_module(SMOOTH_GATE_PATH, "pr166_quantile_smooth_gate")
    runner = _warning_runner(tmp_path)
    output = tmp_path / "smooth-payload.json"
    monkeypatch.setattr(gate, "RUNNER", runner)

    with pytest.raises(subprocess.CalledProcessError):
        gate._run_inner(output)

    assert not output.exists()
