import numpy as np
import pytest

from statgpu.linear_model import _lassocv_device_affinity_contract as affinity


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


def test_unweighted_lassocv_requires_complete_finite_fold_evidence():
    X = np.zeros((12, 2), dtype=np.float64)
    checked = affinity._validate_weighted_cv_selection_evidence(
        X,
        _details_with_partial_failure(),
        kwargs={"sample_weight": None, "cv_folds": 3},
    )

    assert checked["alpha"] == pytest.approx(0.05)
    assert np.isnan(checked["mean_mse"][0])
    np.testing.assert_allclose(checked["mean_mse"][1:], [0.21, 0.39])


def test_constant_weight_and_unweighted_calls_share_finite_evidence_rule():
    X = np.zeros((12, 2), dtype=np.float64)
    unweighted = affinity._validate_weighted_cv_selection_evidence(
        X,
        _details_with_partial_failure(),
        kwargs={"sample_weight": None, "cv_folds": 3},
    )
    weighted = affinity._validate_weighted_cv_selection_evidence(
        X,
        _details_with_partial_failure(),
        kwargs={"sample_weight": np.ones(12), "cv_folds": 3},
    )

    assert weighted["alpha"] == unweighted["alpha"] == pytest.approx(0.05)
    np.testing.assert_array_equal(
        np.isnan(weighted["mean_mse"]),
        np.isnan(unweighted["mean_mse"]),
    )
    np.testing.assert_allclose(
        weighted["mean_mse"][1:],
        unweighted["mean_mse"][1:],
    )


def test_unweighted_scalar_selector_request_is_validated_via_details(monkeypatch):
    X = np.zeros((12, 2), dtype=np.float64)
    y = np.zeros(12, dtype=np.float64)
    observed = {}

    def synthetic_selector(X_arg, y_arg, *args, **kwargs):
        observed["return_details"] = kwargs.get("return_details")
        assert X_arg is X
        assert y_arg is y
        return _details_with_partial_failure()

    monkeypatch.setattr(affinity, "_ORIGINAL_SELECT", synthetic_selector)
    selected = affinity._call_selector_with_evidence(
        X,
        y,
        (),
        {"sample_weight": None, "cv_folds": 3, "return_details": False},
    )

    assert observed["return_details"] is True
    assert selected == pytest.approx(0.05)
