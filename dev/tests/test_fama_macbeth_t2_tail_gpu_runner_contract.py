"""Hosted contract coverage for the focused Fama-MacBeth extreme t(2) GPU gate."""

from __future__ import annotations

import numpy as np
import pytest

from dev.benchmarks import validate_fama_macbeth_t2_tail_gpu as t2_gpu_gate
from statgpu.inference._reference_distribution import two_sided_reference_inference


def test_extreme_t2_tail_numpy_preserves_representable_subnormal_probability():
    statistic_value = t2_gpu_gate._EXTREME_STATISTIC
    pvalues, critical = two_sided_reference_inference(
        np.asarray([statistic_value], dtype=np.float64),
        distribution="t",
        alpha=0.05,
        backend="numpy",
        xp=np,
        df=2,
    )

    observed = float(np.asarray(pvalues)[0])
    expected = t2_gpu_gate._expected_tail(statistic_value)
    assert observed > 0.0
    assert np.isfinite(observed)
    np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=0.0)
    assert np.isfinite(float(np.asarray(critical)))


def test_extreme_t2_gpu_runner_contract_requires_both_cuda_backends():
    assert t2_gpu_gate.SCHEMA_VERSION == 1
    assert t2_gpu_gate._validate_acceptance_backends(["cupy", "torch"]) == [
        "cupy",
        "torch",
    ]
    with pytest.raises(ValueError, match="requires exactly both GPU backends"):
        t2_gpu_gate._validate_acceptance_backends(["cupy"])

    expected = t2_gpu_gate._expected_tail(t2_gpu_gate._EXTREME_STATISTIC)
    assert expected > 0.0
    assert expected == pytest.approx(1.0e-308, rel=2e-15, abs=0.0)
