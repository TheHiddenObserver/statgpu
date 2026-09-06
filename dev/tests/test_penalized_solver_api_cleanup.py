import warnings

import numpy as np
import pytest

from statgpu.linear_model import Lasso, LassoCV, PenalizedLinearRegression, Ridge


def _regression_data(seed=123, n=80, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.5, -0.8, 0.0, 0.4, 0.0, 0.0])
    y = X @ beta + rng.normal(scale=0.25, size=n)
    return X, y


def test_direct_solver_is_authoritative_over_legacy_cpu_solver():
    X, y = _regression_data()
    with pytest.warns(FutureWarning, match="cpu_solver.*deprecated"):
        model = PenalizedLinearRegression(
            penalty="l1",
            alpha=0.05,
            device="cpu",
            solver="fista",
            cpu_solver="coordinate_descent",
            compute_inference=False,
            max_iter=500,
        ).fit(X, y)
    assert model._selected_solver == "fista"


def test_explicit_legacy_default_cpu_solver_still_warns():
    with pytest.warns(FutureWarning, match="Lasso.*cpu_solver.*deprecated"):
        Lasso(cpu_solver="coordinate_descent", device="cpu")


def test_default_direct_estimator_does_not_emit_cpu_solver_warning():
    X, y = _regression_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Ridge(alpha=0.2, device="cpu", compute_inference=False).fit(X, y)
    assert not any("cpu_solver" in str(item.message) for item in caught)


def test_lassocv_default_separates_cv_solver_from_final_refit_solver():
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        solver="fista",
        compute_inference=False,
        max_iter=400,
        random_state=7,
    ).fit(X, y)
    assert model.cv_solver_ == "coordinate_descent"
    assert model.estimator_._selected_solver == "fista"


def test_lassocv_explicit_cv_solver_controls_cv_path():
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        solver="fista",
        cv_solver="fista",
        compute_inference=False,
        max_iter=400,
        random_state=7,
    ).fit(X, y)
    assert model.cv_solver_ == "fista"
    assert model.estimator_._selected_solver == "fista"


def test_lassocv_deprecated_cpu_solver_alias_preserves_old_cpu_choice():
    X, y = _regression_data()
    with pytest.warns(FutureWarning, match="LassoCV.*cpu_solver"):
        model = LassoCV(
            alphas=[0.03, 0.08],
            cv=3,
            device="cpu",
            solver="fista",
            cpu_solver="fista",
            compute_inference=False,
            max_iter=400,
            random_state=7,
        ).fit(X, y)
    assert model.cv_solver_ == "fista"
    assert model.estimator_._selected_solver == "fista"


def test_lassocv_glmnet_preserves_legacy_cpu_solver_override_behavior():
    X, y = _regression_data()
    with pytest.warns(FutureWarning, match="LassoCV.*cpu_solver"):
        model = LassoCV(
            alphas=[0.03, 0.08],
            cv=3,
            device="cpu",
            solver="fista",
            cpu_solver="fista",
            method="glmnet",
            compute_inference=False,
            max_iter=400,
            random_state=7,
        ).fit(X, y)
    assert model.cv_solver_ == "coordinate_descent"
    assert model.estimator_._selected_solver == "fista"


def test_lassocv_rejects_conflicting_new_and_legacy_cv_solver_controls():
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        cv_solver="coordinate_descent",
        cpu_solver="fista",
        compute_inference=False,
    )
    with pytest.warns(FutureWarning, match="LassoCV.*cpu_solver"):
        with pytest.raises(ValueError, match="different CV solvers"):
            model.fit(X, y)


def test_lassocv_glmnet_rejects_non_coordinate_descent_cv_solver():
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        method="glmnet",
        cv_solver="fista",
        compute_inference=False,
    )
    with pytest.raises(ValueError, match="method='glmnet'"):
        model.fit(X, y)


def test_lassocv_cv_solver_is_sklearn_clone_visible():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = LassoCV(cv_solver="fista", cpu_solver=None, device="cpu")
    params = model.get_params(deep=False)
    assert params["cv_solver"] == "fista"
    assert params["cpu_solver"] is None

    cloned = clone(model)
    assert cloned.cv_solver == "fista"
    assert cloned.cpu_solver is None
    assert cloned.estimator_ is None


def test_direct_lasso_solver_coordinate_descent_is_selected_on_cpu():
    X, y = _regression_data()
    model = Lasso(
        alpha=0.05,
        device="cpu",
        solver="coordinate_descent",
        compute_inference=False,
        max_iter=500,
    ).fit(X, y)
    assert model._selected_solver == "coordinate_descent"
