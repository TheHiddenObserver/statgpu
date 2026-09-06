import numpy as np
import pytest

import statgpu.linear_model.wrappers._lasso as lasso_impl
from statgpu._config import Device
from statgpu.linear_model import Lasso, LassoCV


def _regression_data(seed=321, n=48, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.2, -0.7, 0.0, 0.35, 0.0])
    y = X @ beta + rng.normal(scale=0.2, size=n)
    return X, y


def _assert_cv_state_cleared(model, X):
    assert model._fitted is False
    for attr in (
        "alpha_",
        "alphas_",
        "cv_results_",
        "mse_path_",
        "mean_mse_",
        "best_score_",
        "coef_",
        "intercept_",
        "n_iter_",
        "estimator_",
        "cv_solver_",
    ):
        assert getattr(model, attr) is None
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


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


def test_lassocv_failed_selection_clears_previous_fit_state(monkeypatch):
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        compute_inference=False,
        max_iter=300,
        random_state=11,
    ).fit(X, y)
    assert model._fitted is True
    assert model.estimator_ is not None
    assert model.cv_solver_ == "coordinate_descent"

    def fail_selection(*args, **kwargs):
        raise RuntimeError("synthetic CV selection failure")

    monkeypatch.setattr(lasso_impl, "_select_lasso_alpha_cv", fail_selection)
    with pytest.raises(RuntimeError, match="synthetic CV selection failure"):
        model.fit(X, y)

    _assert_cv_state_cleared(model, X)


def test_lassocv_failed_final_refit_does_not_publish_candidate_state(monkeypatch):
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        compute_inference=False,
        random_state=11,
    )

    def synthetic_selection(*args, **kwargs):
        return {
            "alpha": 0.03,
            "alphas": np.asarray([0.03, 0.08], dtype=np.float64),
            "mse_path": np.asarray(
                [[0.30, 0.31, 0.29], [0.42, 0.40, 0.41]],
                dtype=np.float64,
            ),
            "mean_mse": np.asarray([0.30, 0.41], dtype=np.float64),
        }

    def fail_refit(self, *args, **kwargs):
        raise RuntimeError("synthetic final refit failure")

    monkeypatch.setattr(lasso_impl, "_select_lasso_alpha_cv", synthetic_selection)
    monkeypatch.setattr(Lasso, "fit", fail_refit)

    with pytest.raises(RuntimeError, match="synthetic final refit failure"):
        model.fit(X, y)

    _assert_cv_state_cleared(model, X)


@pytest.mark.parametrize(
    ("device_name", "target_device", "backend_name"),
    [
        ("cuda", Device.CUDA, "cupy"),
        ("torch", Device.TORCH, "torch"),
    ],
)
def test_lassocv_explicit_gpu_fit_preserves_requested_cv_backend(
    monkeypatch,
    device_name,
    target_device,
    backend_name,
):
    X, y = _regression_data()
    sample_weight = np.ones(X.shape[0], dtype=np.float64)
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device=device_name,
        compute_inference=False,
        random_state=11,
    )

    # Exercise the public fit path without requiring physical CUDA. The strict
    # availability gate is replaced by a sentinel, while _to_array records the
    # exact backend identity that LassoCV requests before entering the legacy
    # selector.
    monkeypatch.setattr(model, "_get_backend", lambda: object())
    converted = {}

    def fake_to_array(value, device=None, backend=None):
        marker = object()
        converted[id(value)] = (marker, device, backend)
        return marker

    monkeypatch.setattr(model, "_to_array", fake_to_array)

    def synthetic_selection(X_cv, y_cv, **kwargs):
        assert X_cv is converted[id(X)][0]
        assert y_cv is converted[id(y)][0]
        assert kwargs["sample_weight"] is converted[id(sample_weight)][0]
        assert kwargs["device"] == device_name
        assert kwargs["cpu_solver"] == "fista"
        return {
            "alpha": 0.03,
            "alphas": np.asarray([0.03, 0.08], dtype=np.float64),
            "mse_path": np.asarray(
                [[0.30, 0.31, 0.29], [0.42, 0.40, 0.41]],
                dtype=np.float64,
            ),
            "mean_mse": np.asarray([0.30, 0.41], dtype=np.float64),
        }

    def successful_refit(self, *args, **kwargs):
        self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        self.n_iter_ = 1
        self._fitted = True
        return self

    monkeypatch.setattr(lasso_impl, "_select_lasso_alpha_cv", synthetic_selection)
    monkeypatch.setattr(Lasso, "fit", successful_refit)

    model.fit(X, y, sample_weight=sample_weight)

    assert model.cv_solver_ == "fista"
    assert len(converted) == 3
    for original in (X, y, sample_weight):
        _, converted_device, converted_backend = converted[id(original)]
        assert converted_device == target_device
        assert converted_backend == backend_name
