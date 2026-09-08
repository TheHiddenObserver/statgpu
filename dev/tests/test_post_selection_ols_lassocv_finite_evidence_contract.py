from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.linear_model import LassoCV
from statgpu.linear_model import _lassocv_device_affinity_contract as affinity
from statgpu.linear_model.cv._lasso_cv import _validate_lassocv_selection_details
from statgpu.linear_model.wrappers import _lasso as lasso_impl


def _details_with_partial_failure():
    return {
        "alpha": 0.1,
        "alphas": np.asarray([0.1, 0.05, 0.01], dtype=np.float64),
        "mse_path": np.asarray(
            [
                [0.01, np.nan, 0.01],
                [0.20, 0.22, 0.21],
                [0.40, 0.39, 0.38],
            ],
            dtype=np.float64,
        ),
        "mean_mse": np.asarray([0.01, 0.21, 0.39], dtype=np.float64),
    }


def test_lassocv_requires_complete_finite_fold_evidence():
    checked = _validate_lassocv_selection_details(
        _details_with_partial_failure(),
        n_samples=12,
    )

    assert checked["alpha"] == pytest.approx(0.05)
    assert np.isnan(checked["mean_mse"][0])
    np.testing.assert_allclose(checked["mean_mse"][1:], [0.21, 0.39])


def test_lassocv_fails_when_no_candidate_has_complete_finite_evidence():
    details = {
        "alpha": 0.1,
        "alphas": np.asarray([0.1, 0.05], dtype=np.float64),
        "mse_path": np.asarray(
            [[0.1, np.nan, 0.2], [np.nan, 0.3, 0.2]],
            dtype=np.float64,
        ),
        "mean_mse": np.asarray([0.15, 0.25], dtype=np.float64),
    }

    with pytest.raises(FloatingPointError, match="finite validation MSE on every fold"):
        _validate_lassocv_selection_details(details, n_samples=12)


def test_lassocv_single_alpha_degenerate_path_remains_unchanged():
    details = {
        "alpha": 0.1,
        "alphas": np.asarray([0.1], dtype=np.float64),
        "mse_path": np.asarray([[np.nan]], dtype=np.float64),
        "mean_mse": np.asarray([np.nan], dtype=np.float64),
    }

    checked = _validate_lassocv_selection_details(details, n_samples=12)
    assert checked is details


def test_lassocv_fit_applies_complete_evidence_before_final_refit(monkeypatch):
    X = np.zeros((12, 2), dtype=np.float64)
    y = np.zeros(12, dtype=np.float64)
    refit = {}

    def synthetic_selector(X_arg, y_arg, **kwargs):
        assert X_arg is X
        assert y_arg is y
        assert kwargs["return_details"] is True
        return _details_with_partial_failure()

    def successful_refit(self, X_arg, y_arg, sample_weight=None):
        refit["alpha"] = float(self.alpha)
        self.coef_ = np.zeros(X_arg.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        self.n_iter_ = 1
        self._fitted = True
        return self

    monkeypatch.setattr(lasso_impl, "_select_lasso_alpha_cv", synthetic_selector)
    monkeypatch.setattr(lasso_impl.Lasso, "fit", successful_refit)

    model = LassoCV(
        alphas=[0.1, 0.05, 0.01],
        cv=3,
        device="cpu",
        compute_inference=False,
    ).fit(X, y)

    assert refit["alpha"] == pytest.approx(0.05)
    assert model.alpha_ == pytest.approx(0.05)
    assert np.isnan(model.mean_mse_[0])
    np.testing.assert_allclose(model.mean_mse_[1:], [0.21, 0.39])


def test_lassocv_no_complete_evidence_fails_before_final_refit(monkeypatch):
    X = np.zeros((12, 2), dtype=np.float64)
    y = np.zeros(12, dtype=np.float64)
    details = {
        "alpha": 0.1,
        "alphas": np.asarray([0.1, 0.05], dtype=np.float64),
        "mse_path": np.asarray(
            [[0.1, np.nan, 0.2], [np.nan, 0.3, 0.2]],
            dtype=np.float64,
        ),
        "mean_mse": np.asarray([0.15, 0.25], dtype=np.float64),
    }

    monkeypatch.setattr(
        lasso_impl,
        "_select_lasso_alpha_cv",
        lambda X_arg, y_arg, **kwargs: details,
    )

    def forbidden_refit(self, *args, **kwargs):
        raise AssertionError("final Lasso refit must not run without complete CV evidence")

    monkeypatch.setattr(lasso_impl.Lasso, "fit", forbidden_refit)
    model = LassoCV(
        alphas=[0.1, 0.05],
        cv=3,
        device="cpu",
        compute_inference=False,
    )

    with pytest.raises(FloatingPointError, match="finite validation MSE on every fold"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.alpha_ is None
    assert model.estimator_ is None


def test_shared_selector_keeps_torch_cpu_inputs_transparent(monkeypatch):
    X = SimpleNamespace(device="cpu")
    y = object()
    observed = {}

    def synthetic_selector(X_arg, y_arg, *args, **kwargs):
        observed["X"] = X_arg
        observed["y"] = y_arg
        return 0.25

    monkeypatch.setattr(affinity, "_is_cupy_array", lambda value: False)
    monkeypatch.setattr(affinity, "_is_torch_array", lambda value: value is X)
    monkeypatch.setattr(affinity, "_ORIGINAL_SELECT", synthetic_selector)

    selected = affinity._select_lasso_alpha_cv_on_design_device(
        X,
        y,
        device="cpu",
    )

    assert selected == pytest.approx(0.25)
    assert observed == {"X": X, "y": y}
