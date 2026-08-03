"""Transactional failed-refit state for public group estimators and CV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


def _data(seed=10601):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(72, 4))
    y = 0.4 + X @ np.array([0.8, -0.45, 0.25, 0.6])
    y += rng.normal(scale=0.07, size=X.shape[0])
    return X, y


def test_direct_group_failed_refit_clears_coefficients_and_formula_state():
    X, y = _data()
    frame = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
    frame["y"] = y
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": [[0, 1], [2, 3]]},
        alpha=0.08,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=3000,
        tol=1e-9,
    ).fit(formula="y ~ x0 + x1 + x2 + x3", data=frame)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert model._feature_names is not None
    assert model._design_info is not None
    assert model._selected_solver is not None

    model.penalty_kwargs = {"groups": [[0, 4], [1, 2, 3]]}
    with pytest.raises(ValueError, match="outside the design matrix"):
        model.fit(X, y)

    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._inference_result is None
    assert model._feature_names is None
    assert model._design_info is None
    assert model._formula_has_intercept is None
    assert model._use_intercept is None
    assert model._selected_solver is None
    assert model._selected_backend_name is None
    assert model._penalty is None
    assert model._loss is None
    assert model._fitted is False
    assert not hasattr(model, "n_features_in_")


def test_group_cv_failed_prevalidation_clears_previous_selection_and_refit():
    X, y = _data(seed=10602)
    cv = PenalizedGLM_CV(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": [[0, 1], [2, 3]]},
        alpha_grid=[0.16, 0.08],
        cv=2,
        random_state=31,
        device="cpu",
        max_iter=1200,
        tol=1e-8,
    ).fit(X, y)

    assert cv._fitted is True
    assert cv.alpha_ is not None
    assert cv.cv_results_ is not None
    assert cv.estimator_ is not None
    assert cv.coef_ is not None

    cv._penalty_kwargs = {"groups": [[0, 4], [1, 2, 3]]}
    with pytest.raises(ValueError, match="outside the design matrix"):
        cv.fit(X, y)

    assert cv._fitted is False
    assert cv.alpha_ is None
    assert cv.alpha_grid_ is None
    assert cv.best_score_ is None
    assert cv.cv_results_ is None
    assert cv.estimator_ is None
    assert cv.coef_ is None
    assert cv.intercept_ is None
    assert cv.cv_strategy_ is None
    assert cv.cv_selected_device_ is None
    with pytest.raises(RuntimeError, match="not fitted"):
        cv.predict(X)
