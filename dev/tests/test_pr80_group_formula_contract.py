"""Formula-facing group penalty coverage and column-order contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statgpu.core.formula import FormulaParser
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


def test_group_lasso_formula_uses_final_patsy_feature_order_and_free_intercept():
    rng = np.random.default_rng(10501)
    n = 90
    data = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "cat": np.resize(np.array(["a", "b", "c"]), n),
        }
    )
    data["y"] = (
        0.7
        + 0.8 * data["x"]
        - 0.35 * data["z"]
        + 0.25 * (data["cat"] == "b").astype(float)
        - 0.2 * (data["cat"] == "c").astype(float)
        + rng.normal(scale=0.05, size=n)
    )
    formula = "y ~ x + z + C(cat)"
    parser = FormulaParser(formula)
    y_matrix, X_matrix, design_info = parser.eval(data)
    names = list(design_info.column_names)
    intercept_position = names.index("Intercept")
    X_features = np.delete(X_matrix, intercept_position, axis=1)
    feature_names = [name for name in names if name != "Intercept"]
    p = X_features.shape[1]
    assert p == 4

    groups = [[0, 1], [2, 3]]
    common = dict(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": groups},
        alpha=0.06,
        solver="auto",
        device="cpu",
        fit_intercept=False,  # formula syntax must override this flag
        compute_inference=False,
        max_iter=4000,
        tol=1e-10,
    )
    formula_model = PenalizedGeneralizedLinearModel(**common).fit(
        formula=formula,
        data=data,
    )
    array_model = PenalizedGeneralizedLinearModel(
        **{**common, "fit_intercept": True}
    ).fit(X_features, np.asarray(y_matrix).reshape(-1))

    assert formula_model._effective_intercept is True
    assert formula_model._formula_has_intercept is True
    assert formula_model._feature_names == feature_names
    assert formula_model._penalty.groups == ((0, 1), (2, 3))
    assert formula_model.coef_.shape == (p,)
    np.testing.assert_allclose(
        formula_model.coef_, array_model.coef_, rtol=2e-7, atol=2e-8
    )
    assert formula_model.intercept_ == pytest.approx(
        array_model.intercept_, rel=2e-7, abs=2e-8
    )
    np.testing.assert_allclose(
        formula_model.predict(data=data),
        array_model.predict(X_features),
        rtol=2e-7,
        atol=2e-8,
    )


def test_formula_group_completion_uses_expanded_feature_count():
    rng = np.random.default_rng(10502)
    n = 60
    data = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": np.resize(np.array(["a", "b", "c"]), n),
        }
    )
    data["y"] = (
        0.4
        + 0.6 * data["x"]
        + 0.2 * (data["cat"] == "b").astype(float)
        + rng.normal(scale=0.05, size=n)
    )

    with pytest.warns(UserWarning, match="Auto-adding 1 single-feature"):
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error",
            penalty="group_lasso",
            penalty_kwargs={"groups": [[0, 1]]},
            alpha=0.05,
            solver="auto",
            device="cpu",
            compute_inference=False,
            max_iter=3000,
            tol=1e-9,
        ).fit(formula="y ~ x + C(cat)", data=data)

    assert len(model._feature_names) == 3
    assert model._penalty.groups == ((0, 1), (2,))
    assert model.coef_.shape == (3,)
