"""Run the legacy native tests after real CPU-setting tests in one process."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NATIVE_TESTS = "dev/tests/test_three_backend_native_followup.py"


@pytest.mark.parametrize(
    "predecessor",
    [
        "dev/tests/test_linear.py::TestLinearRegression::test_basic_fit_cpu",
        (
            "dev/tests/test_ridge_inference.py::TestRidgeInferenceParams::"
            "test_compute_inference_false_no_bse"
        ),
    ],
    ids=["linear-cpu", "ridge-cpu"],
)
def test_native_auto_tests_restore_incoming_cpu_policy(predecessor, tmp_path):
    pytest.importorskip("torch")
    sentinel = tmp_path / "test_restored_device.py"
    sentinel.write_text(
        textwrap.dedent(
            """
            import numpy as np
            import torch

            from statgpu._config import Device, _device_manager
            from statgpu.covariance import EmpiricalCovariance


            def test_restored_cpu_policy_still_controls_native_input():
                assert _device_manager._current_device is Device.CPU
                data = torch.tensor(
                    [[0., 1.], [1., 0.], [2., 3.], [3., 1.]],
                    dtype=torch.float64,
                )
                fitted = EmpiricalCovariance().fit(data)
                assert isinstance(fitted.covariance_, np.ndarray)
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", predecessor, NATIVE_TESTS,
            str(sentinel), "-q", "--tb=short",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
