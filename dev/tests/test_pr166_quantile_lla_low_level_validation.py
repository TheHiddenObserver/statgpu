"""Low-level validation contracts for Quantile fista_lla_path in PR #166."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import L1Penalty, L2Penalty, SCADPenalty
from statgpu.solvers import fista_lla_path


@pytest.mark.parametrize(
    "penalty",
    [
        L1Penalty(alpha=0.04),
        L2Penalty(alpha=0.04),
    ],
)
def test_quantile_fista_lla_rejects_non_lla_penalty_before_loss_work(
    monkeypatch, penalty
):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "lipschitz", forbidden)

    with pytest.raises(ValueError, match="requires SCAD/MCP"):
        fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.04], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=20,
            tol=1e-6,
            fit_intercept=False,
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.asarray([1.0, -0.2, 1.0, 1.0]), "sample_weight must be non-negative"),
        (np.asarray([1.0, np.nan, 1.0, 1.0]), "sample_weight must contain only finite values"),
        (np.asarray([1.0, 1.0, 1.0]), "sample_weight must be 1D with length n_samples"),
    ],
)
def test_quantile_fista_lla_preserves_shared_sample_weight_validation(weights, message):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = SCADPenalty(alpha=0.04, a=3.7)

    with pytest.raises(ValueError, match=message):
        fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.04], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=20,
            tol=1e-6,
            fit_intercept=False,
            sample_weight=weights,
        )
