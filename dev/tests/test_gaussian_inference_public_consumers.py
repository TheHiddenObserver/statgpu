"""Public-consumer regression coverage for issue #127."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._inference_mixin import _PenalizedInferenceMixin
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


def _data():
    X = np.asarray(
        [
            [-2.0, 0.5],
            [-1.0, 1.0],
            [0.0, 1.5],
            [1.0, 2.0],
            [2.0, 2.5],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    coef = np.asarray([0.75, -0.4])
    intercept = 1.2
    y = intercept + X @ coef + np.asarray([0.2, -0.1, 0.15, -0.25, 0.05, -0.05])
    return X, y, coef, intercept


def _prepare_l2_model(model, backend_name="torch"):
    X, y, coef, intercept = _data()
    model._penalty = model._resolve_penalty()
    model._selected_backend_name = backend_name
    model.coef_ = coef.copy()
    model.intercept_ = float(intercept)
    return X, y


def test_public_generic_pglm_l2_uses_selected_torch_backend():
    pytest.importorskip("torch")
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        device="cpu",
        compute_inference=True,
    )
    X, y = _prepare_l2_model(model, backend_name="torch")
    model._compute_post_fit_gaussian_inference(X, y)

    result = model._inference_result
    assert result is not None
    assert result.metadata["numerical_backend"] == "torch"
    assert result.metadata["numerical_device"] == "cpu"
    assert result.metadata["reporting_boundary"] == "post_numerical_inference"
    assert isinstance(model._X_design, np.ndarray)
    assert isinstance(model._resid, np.ndarray)
    assert isinstance(model._bse, np.ndarray)


def test_typed_penalized_linear_uses_same_public_router():
    assert (
        PenalizedLinearRegression._compute_post_fit_gaussian_inference
        is PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    )


def test_non_l2_branch_still_delegates_to_shared_inference_mixin(monkeypatch):
    calls = []

    def _fake_shared(self, X, y, sample_weight=None):
        calls.append((self, X, y, sample_weight))

    monkeypatch.setattr(
        _PenalizedInferenceMixin,
        "_compute_post_fit_gaussian_inference",
        _fake_shared,
    )
    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.1,
        device="cpu",
        compute_inference=True,
    )
    model._penalty = model._resolve_penalty()
    X, y, _, _ = _data()
    model._compute_post_fit_gaussian_inference(X, y)

    assert len(calls) == 1
    assert calls[0][0] is model


def test_non_gaussian_branch_still_delegates_to_shared_inference_mixin(monkeypatch):
    calls = []

    def _fake_shared(self, X, y, sample_weight=None):
        calls.append((self, X, y, sample_weight))

    monkeypatch.setattr(
        _PenalizedInferenceMixin,
        "_compute_post_fit_gaussian_inference",
        _fake_shared,
    )
    model = PenalizedGeneralizedLinearModel(
        loss="poisson",
        penalty="l2",
        alpha=0.1,
        device="cpu",
        compute_inference=True,
    )
    model._penalty = model._resolve_penalty()
    X, y, _, _ = _data()
    model._compute_post_fit_gaussian_inference(X, np.abs(y) + 1.0)

    assert len(calls) == 1
    assert calls[0][0] is model


@pytest.mark.parametrize("backend_name", [None, "unknown-backend"])
def test_l2_inference_fails_closed_without_valid_fit_backend_provenance(backend_name):
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        device="cpu",
        compute_inference=True,
    )
    X, y, coef, intercept = _data()
    model._penalty = model._resolve_penalty()
    model._selected_backend_name = backend_name
    model.coef_ = coef.copy()
    model.intercept_ = float(intercept)

    with pytest.raises(RuntimeError, match="backend provenance|executed backend"):
        model._compute_post_fit_gaussian_inference(X, y)
    assert model._inference_result is None
    assert model._bse is None
