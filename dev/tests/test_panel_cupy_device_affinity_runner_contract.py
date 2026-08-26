"""Hosted contract for the focused Panel CuPy device-affinity physical gate."""

from __future__ import annotations

import pytest

from dev.benchmarks import validate_panel_cupy_device_affinity_gpu as device_gate


def test_cupy_device_affinity_runner_schema_and_target_selection():
    assert device_gate.SCHEMA_VERSION == 1
    assert device_gate._target_device(1, 0) == 0
    assert device_gate._target_device(2, 0) == 1
    assert device_gate._target_device(2, 1) == 0
    assert device_gate._target_device(4, 2) == 3


def test_cupy_device_affinity_runner_rejects_invalid_device_state():
    with pytest.raises(ValueError, match="at least one CUDA device"):
        device_gate._target_device(0, 0)
    with pytest.raises(ValueError, match="outside the reported device range"):
        device_gate._target_device(2, 2)
