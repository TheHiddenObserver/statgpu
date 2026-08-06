"""Regression coverage for the all-censored Cox loss boundary."""

import numpy as np
from numpy.testing import assert_allclose

from statgpu.losses import CoxPartialLikelihoodLoss


def test_all_censored_cox_loss_has_zero_value_gradient_and_hessian():
    rng = np.random.default_rng(20260726)
    X = rng.normal(size=(24, 3))
    time = rng.exponential(size=24) + 0.1
    event = np.zeros(24, dtype=np.float64)
    y = {"time": time, "event": event}
    coef = np.array([0.3, -0.2, 0.1])

    loss = CoxPartialLikelihoodLoss(ties="efron")
    assert loss.value(X, y, coef) == 0.0
    assert_allclose(loss.gradient(X, y, coef), np.zeros(3))
    assert_allclose(loss.hessian(X, y, coef), np.zeros((3, 3)))
