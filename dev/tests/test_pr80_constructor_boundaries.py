"""Regression tests for Cox constructor-level public controls."""

from __future__ import annotations

import inspect

import pytest

from statgpu.survival import CoxPH, CoxPHCV


@pytest.mark.parametrize(
    ("parameter", "estimator"),
    [
        ("compute_inference", CoxPH),
        ("compute_cindex", CoxPH),
        ("gpu_memory_cleanup", CoxPH),
        ("compute_inference", CoxPHCV),
        ("gpu_memory_cleanup", CoxPHCV),
    ],
)
def test_truthy_boolean_strings_are_rejected_at_construction(parameter, estimator):
    with pytest.raises(ValueError, match=rf"{parameter} must be"):
        estimator(**{parameter: "False"})


@pytest.mark.parametrize("value", [False, True, 0, 1])
def test_coxph_constructor_accepts_explicit_boolean_controls(value):
    model = CoxPH(
        compute_inference=value,
        compute_cindex=value,
        gpu_memory_cleanup=value,
    )
    assert bool(model.compute_inference) is bool(value)
    assert model.compute_cindex is bool(value)
    assert model.gpu_memory_cleanup is bool(value)


@pytest.mark.parametrize("value", [False, True, 0, 1])
def test_coxphcv_constructor_preserves_clone_safe_boolean_inputs(value):
    model = CoxPHCV(
        compute_inference=value,
        gpu_memory_cleanup=value,
        penalties=[0.1],
        cv=2,
    )
    assert model.compute_inference is value
    assert model.gpu_memory_cleanup is value


def test_adapter_wrapping_preserves_public_constructor_signatures():
    cox_parameters = inspect.signature(CoxPH.__init__).parameters
    cv_parameters = inspect.signature(CoxPHCV.__init__).parameters
    assert "compute_cindex" in cox_parameters
    assert "gpu_memory_cleanup" in cox_parameters
    assert "penalties" in cv_parameters
    assert "gpu_memory_cleanup" in cv_parameters
