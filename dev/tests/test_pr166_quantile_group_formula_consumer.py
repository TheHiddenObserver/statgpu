"""Formula consumer coverage for the PR #166 Quantile group auto route."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statgpu.core.formula import FormulaParser
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


def test_weighted_quantile_group_scad_formula_matches_array_route():
    rng = np.random.default_rng(166601)
    n = 24
    data = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    beta = np.array([0.8, -0.45, 0.3, 0.15], dtype=np.float64)
    X_raw = data[["x0", "x1", "x2", "x3"]].to_numpy(dtype=np.float64)
    data["y"] = 0.2 + X_raw @ beta + rng.laplace(scale=0.15, size=n)
    weights = np.linspace(0.4, 1.8, n, dtype=np.float64)
    rng.shuffle(weights)

    formula = "y ~ x0 + x1 + x2 + x3"
    y_matrix, X_matrix, design_info = FormulaParser(formula).eval(data)
    names = list(design_info.column_names)
    intercept_position = names.index("Intercept")
    X_features = np.delete(X_matrix, intercept_position, axis=1)

    common = dict(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": [[0, 1], [2, 3]], "a": 3.7},
        alpha=0.04,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=80,
        tol=1e-5,
        max_lla_iters=9,
        lla_tol=1e-5,
    )

    formula_model = PenalizedGeneralizedLinearModel(
        **common,
        fit_intercept=False,
    ).fit(
        formula=formula,
        data=data,
        sample_weight=weights,
    )
    array_model = PenalizedGeneralizedLinearModel(
        **common,
        fit_intercept=True,
    ).fit(
        np.asarray(X_features, dtype=np.float64),
        np.asarray(y_matrix, dtype=np.float64).reshape(-1),
        sample_weight=weights,
    )

    assert formula_model._effective_intercept is True
    assert formula_model._formula_has_intercept is True
    assert formula_model._selected_solver == "group_proximal_irls_lla"
    assert array_model._selected_solver == "group_proximal_irls_lla"
    np.testing.assert_allclose(
        formula_model.coef_, array_model.coef_, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(
        formula_model.intercept_, array_model.intercept_, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(
        formula_model.predict(data),
        array_model.predict(X_features),
        rtol=2e-8,
        atol=2e-9,
    )
