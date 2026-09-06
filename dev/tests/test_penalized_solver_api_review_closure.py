import numpy as np
import pytest

import statgpu.linear_model.wrappers._lasso as lasso_impl
from statgpu.linear_model import Lasso, LassoCV


def _regression_data(seed=321, n=48, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.2, -0.7, 0.0, 0.35, 0.0])
    y = X @ beta + rng.normal(scale=0.2, size=n)
    return X, y


def test_direct_legacy_constructor_warning_points_to_caller():
    with pytest.warns(
        FutureWarning, match="Lasso.*cpu_solver.*deprecated"
    ) as caught:
        Lasso(cpu_solver="coordinate_descent", device="cpu")
    assert caught[0].filename == __file__


def test_lassocv_explicit_historical_default_warns_and_preserves_cpu_cv_solver():
    X, y = _regression_data()
    with pytest.warns(FutureWarning, match="LassoCV.*cpu_solver") as caught:
        model = LassoCV(
            alphas=[0.03, 0.08],
            cv=3,
            device="cpu",
            cpu_solver="coordinate_descent",
            compute_inference=False,
            max_iter=300,
            random_state=11,
        ).fit(X, y)
    assert caught[0].filename == __file__
    assert model.cv_solver_ == "coordinate_descent"


def test_lassocv_failed_selection_does_not_publish_cv_solver(monkeypatch):
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        compute_inference=False,
        random_state=11,
    )

    def fail_selection(*args, **kwargs):
        raise RuntimeError("synthetic CV selection failure")

    monkeypatch.setattr(lasso_impl, "_select_lasso_alpha_cv", fail_selection)
    with pytest.raises(RuntimeError, match="synthetic CV selection failure"):
        model.fit(X, y)

    assert model.cv_solver_ is None
