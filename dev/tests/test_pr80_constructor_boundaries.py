"""Regression tests for Cox constructor-level public controls."""

from __future__ import annotations

import inspect

import numpy as np
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
    assert model.compute_inference is value
    assert model.compute_cindex is value
    assert model.gpu_memory_cleanup is value


def test_coxph_constructor_preserves_clone_safe_cox_controls():
    penalty = np.float64(0.125)
    model = CoxPH(
        ties="EFRON",
        cov_type="HC1",
        inference_mode="STRICT",
        penalty=penalty,
        compute_inference=False,
    )
    assert model.ties == "EFRON"
    assert model.cov_type == "HC1"
    assert model.inference_mode == "STRICT"
    assert model.penalty is penalty


def test_coxph_set_params_preserves_validated_public_values():
    penalty = np.float64(0.125)
    model = CoxPH().set_params(
        ties="EFRON",
        cov_type="HC1",
        inference_mode="STRICT",
        penalty=penalty,
    )

    assert model.ties == "EFRON"
    assert model.cov_type == "HC1"
    assert model.inference_mode == "STRICT"
    assert model.penalty is penalty


def _cox_fit_sample(n=30, p=2):
    rng = np.random.default_rng(8080)
    X = rng.normal(size=(n, p))
    stop = rng.uniform(0.5, 3.0, size=n)
    event = (np.arange(n) % 3 != 0).astype(np.float64)
    return X, stop, event


def test_coxph_fit_does_not_rewrite_constructor_parameters():
    X, stop, event = _cox_fit_sample()
    penalty = np.float64(0.1)
    tol = np.float64(1e-7)
    max_iter = np.int64(40)
    model = CoxPH(
        ties="EFRON",
        cov_type="NONROBUST",
        inference_mode="STRICT",
        penalty=penalty,
        tol=tol,
        max_iter=max_iter,
        compute_inference=0,
        compute_cindex=0,
    )
    before = model.get_params().copy()
    model.fit(X, stop, event)
    after = model.get_params()

    assert after == before
    assert model.ties == "EFRON"
    assert model.cov_type == "NONROBUST"
    assert model.inference_mode == "STRICT"
    assert model.penalty is penalty
    assert model.tol is tol
    assert model.max_iter is max_iter
    assert model.compute_inference == 0
    assert model.compute_cindex == 0


def test_coxphcv_fit_does_not_rewrite_constructor_parameters():
    X, stop, event = _cox_fit_sample(n=36)
    tol = np.float64(1e-6)
    max_iter = np.int64(30)
    model = CoxPHCV(
        penalties=[0.1],
        cv=2,
        ties="EFRON",
        cov_type="NONROBUST",
        inference_mode="STRICT",
        tol=tol,
        max_iter=max_iter,
        compute_inference=0,
        gpu_memory_cleanup=0,
        random_state=3,
    )
    before = model.get_params().copy()
    model.fit(X, stop, event)
    after = model.get_params()

    assert after == before
    assert model.ties == "EFRON"
    assert model.cov_type == "NONROBUST"
    assert model.inference_mode == "STRICT"
    assert model.tol is tol
    assert model.max_iter is max_iter
    assert model.compute_inference == 0
    assert model.gpu_memory_cleanup == 0


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
