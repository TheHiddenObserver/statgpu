"""Regression tests for fail-closed Stage-C inference storage."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import _covariance
from statgpu.panel._base import BasePanelModel


class _DummyPanelModel(BasePanelModel):
    def __init__(self):
        super().__init__(device="cpu", n_jobs=None)
        self.alpha = 0.05

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        return np.asarray(X)


def test_negative_variance_guard_is_local_to_each_coefficient(monkeypatch):
    """A huge variance elsewhere must not mask a substantive negative variance."""

    def _fake_covariance(*args, **kwargs):
        return np.diag([1.0e14, -1.0])

    monkeypatch.setattr(_covariance, "ols_covariance", _fake_covariance)
    model = _DummyPanelModel()
    backend = model._get_backend(backend="auto")

    with pytest.raises(
        ValueError,
        match="materially negative diagonal variance",
    ):
        model._panel_store_ols_inference(
            np.eye(2),
            np.zeros(2),
            np.ones(2),
            scale=1.0,
            df_resid=1,
            backend=backend,
            cov_type="hc0",
            allowed=("hc0",),
            diag_floor=1.0e-30,
        )



def _store_with_mock_covariance(monkeypatch, covariance):
    def _fake_covariance(*args, **kwargs):
        return np.asarray(covariance, dtype=np.float64)

    monkeypatch.setattr(_covariance, "ols_covariance", _fake_covariance)
    model = _DummyPanelModel()
    backend = model._get_backend(backend="auto")
    model._panel_store_ols_inference(
        np.eye(2), np.zeros(2), np.ones(2), scale=1.0, df_resid=2,
        backend=backend, cov_type="hc0", allowed=("hc0",), diag_floor=1.0e-30,
    )
    return model


def test_negative_variance_guard_is_scale_equivariant(monkeypatch):
    """Outcome/parameter rescaling cannot change covariance validity."""
    material = np.array([[4.0e-14, 0.0], [0.0, -1.0e-28]])
    for multiplier in (1.0e-12, 1.0, 1.0e12):
        with pytest.raises(ValueError, match="materially negative diagonal variance"):
            _store_with_mock_covariance(monkeypatch, material * multiplier)

    # A large off-diagonal entry must never mask a negative variance. This also
    # guards against row/column-norm tolerances that change units under regressor
    # reparameterization.
    indefinite = np.array([[1.0, 1.0e8], [1.0e8, -1.0e-20]])
    with pytest.raises(ValueError, match="materially negative diagonal variance"):
        _store_with_mock_covariance(monkeypatch, indefinite)

    signed_zero = np.array([[4.0e-14, 0.0], [0.0, -0.0]])
    model = _store_with_mock_covariance(monkeypatch, signed_zero)
    assert np.all(np.isfinite(model.bse_))
    assert model.bse_[1] >= 0.0
