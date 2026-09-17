"""Physical-gate convergence warning contract for PR #166."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "dev" / "benchmarks" / "run_quantile_group_lla_gpu_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("pr166_quantile_group_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_gate_promotes_statgpu_convergence_warning_to_failure(tmp_path):
    gate = _load_gate_module()
    runner = tmp_path / "warning_runner.py"
    output = tmp_path / "payload.json"
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

    with pytest.raises(subprocess.CalledProcessError):
        gate._run_inner(runner, output)

    assert not output.exists()
