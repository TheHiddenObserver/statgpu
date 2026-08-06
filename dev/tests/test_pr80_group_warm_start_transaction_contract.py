"""One-shot warm-start state for transactional group fits."""

from __future__ import annotations

import numpy as np

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


def test_group_lasso_pending_warm_start_reaches_solver_as_one_vector(monkeypatch):
    import statgpu.solvers as solvers

    rng = np.random.default_rng(10901)
    X = rng.normal(size=(36, 3))
    y = rng.normal(size=36)
    warm_coef = np.array([0.45, -0.25, 0.15])
    warm_intercept = 0.37
    observed = {}

    def fake_fista(
        loss,
        penalty,
        X_work,
        y_work,
        *,
        max_iter,
        tol,
        init_coef,
        sample_weight,
        **kwargs,
    ):
        observed["init"] = np.asarray(init_coef, dtype=float).copy()
        return np.asarray(init_coef, dtype=float).copy(), 1

    monkeypatch.setattr(solvers, "fista_solver", fake_fista)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": [[0, 1], [2]]},
        alpha=0.1,
        solver="fista",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
    )
    model._init_coef = warm_coef.copy()
    model._init_intercept = warm_intercept

    model.fit(X, y)

    np.testing.assert_allclose(
        observed["init"], np.append(warm_coef, warm_intercept)
    )
    np.testing.assert_allclose(model.coef_, warm_coef)
    assert model.intercept_ == warm_intercept
    assert model._init_coef is None
    assert model._init_intercept is None


def test_failed_group_fit_clears_both_warm_start_components():
    rng = np.random.default_rng(10902)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=29)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": [[0, 1], [2]]},
        alpha=0.1,
        solver="fista",
        device="cpu",
        compute_inference=False,
    )
    model._init_coef = np.array([0.2, -0.1, 0.05])
    model._init_intercept = 0.4

    try:
        model.fit(X, y)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched response length must fail")

    assert model._init_coef is None
    assert model._init_intercept is None
    assert model.coef_ is None
    assert model.intercept_ is None
