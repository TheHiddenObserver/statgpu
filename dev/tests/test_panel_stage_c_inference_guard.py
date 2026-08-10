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
