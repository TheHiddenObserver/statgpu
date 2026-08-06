"""Invalid Cox CV grids must not preserve stale fitted state."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.survival import CoxPHCV


def test_invalid_custom_grid_clears_existing_coxphcv_state():
    rng = np.random.default_rng(12131)
    X = rng.normal(size=(18, 2))
    time = np.linspace(1.0, 18.0, 18)
    event = np.tile(np.array([1.0, 0.0, 1.0]), 6)

    model = CoxPHCV(
        penalties=[0.2, 0.1],
        cv=2,
        device="cpu",
        compute_inference=False,
    )
    # Seed the complete public/private state that a prior successful fit would
    # expose. The second fit must clear all of it before grid validation.
    model._fitted = True
    model.penalty_ = 0.1
    model.penalties_ = np.array([0.2, 0.1])
    model.cv_results_ = {"mean_pl": np.array([-1.0, -0.8])}
    model.best_score_ = -0.8
    model.coef_ = np.array([0.3, -0.2])
    model.hazard_ratios_ = np.exp(model.coef_)
    model.estimator_ = object()
    model._params = model.coef_.copy()
    model._bse = np.array([0.1, 0.1])
    model._inference_result = object()

    invalid = [True, 0.1]
    model.penalties = invalid
    with pytest.raises(ValueError, match="booleans"):
        model.fit(X, time, event)

    assert model.penalties is invalid
    assert model._fitted is False
    assert model.penalty_ is None
    assert model.penalties_ is None
    assert model.cv_results_ is None
    assert model.best_score_ is None
    assert model.coef_ is None
    assert model.hazard_ratios_ is None
    assert model.estimator_ is None
    assert model._params is None
    assert model._bse is None
    assert model._inference_result is None
