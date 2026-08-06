"""Regression tests for remaining public smoothing, spline, GAM and metric APIs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu import GAM, KernelDensityEstimator, KernelRegression, SplineTransformer
from statgpu.metrics import evaluate_binary_classification
from statgpu.nonparametric.splines import bspline_basis


def test_bspline_rejects_invalid_knots_degree_and_nonfinite_values():
    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="strictly increasing"):
        bspline_basis(x, [0.3, 0.3, 0.7])
    with pytest.raises(ValueError, match="degree"):
        bspline_basis(x, [0.3, 0.7], degree=-1)
    with pytest.raises(ValueError, match="finite"):
        bspline_basis(np.array([0.0, np.nan, 1.0]), [0.3, 0.7])


def test_spline_transformer_constant_extrapolation_clamps_to_boundary():
    X = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    model = SplineTransformer(device='cpu', n_knots=5, degree=3, extrapolation="constant").fit(X)
    boundary = np.asarray(model.transform(np.array([[0.0], [1.0]])))
    outside = np.asarray(model.transform(np.array([[-2.0], [3.0]])))
    assert_allclose(outside, boundary, rtol=1e-12, atol=1e-12)


def test_spline_transformer_linear_extrapolation_is_linear_at_boundaries():
    X = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    model = SplineTransformer(device='cpu', n_knots=5, degree=3, extrapolation="linear").fit(X)
    left = np.asarray(model.transform(np.array([[-1.0], [-0.5], [0.0]])))
    right = np.asarray(model.transform(np.array([[1.0], [1.5], [2.0]])))
    assert_allclose(left[0] - left[1], left[1] - left[2], rtol=1e-9, atol=1e-9)
    assert_allclose(right[2] - right[1], right[1] - right[0], rtol=1e-9, atol=1e-9)


def test_spline_transformer_continue_matches_scipy_bspline_extrapolation():
    from scipy.interpolate import BSpline

    X = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    model = SplineTransformer(device='cpu', n_knots=5, degree=3, extrapolation="continue").fit(X)
    points = np.array([[-0.4], [0.2], [1.4]])
    actual = np.asarray(model.transform(points))

    kts = model.knots_[0]
    augmented = np.r_[
        np.repeat(kts[0], model.degree + 1),
        kts[1:-1],
        np.repeat(kts[-1], model.degree + 1),
    ]
    n_basis = len(augmented) - model.degree - 1
    expected = BSpline(augmented, np.eye(n_basis), model.degree, extrapolate=True)(points[:, 0])
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_spline_transformer_quantile_ties_do_not_corrupt_output_dimension():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 30),
        np.repeat([0.0, 1.0, 2.0], 10),
    ])
    with pytest.raises(ValueError, match="distinct"):
        SplineTransformer(device='cpu', n_knots=5, knots="quantile").fit(X)


def test_spline_transformer_custom_knots_validate_shape_and_boundaries():
    X = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 2, 20)])
    with pytest.raises(ValueError, match="shape"):
        SplineTransformer(device='cpu', knots=np.array([[0.0, 1.0, 2.0]])).fit(X)
    with pytest.raises(ValueError, match="strictly increasing"):
        SplineTransformer(device='cpu', 
            n_knots=3,
            knots=np.array([[0.0, 1.0], [0.0, 1.5], [1.0, 2.0]]),
        ).fit(X)


def test_spline_transformer_declared_dimension_matches_transform():
    X = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 3, 20)])
    model = SplineTransformer(device='cpu', n_knots=6, degree=2, include_bias=False).fit(X)
    transformed = np.asarray(model.transform(X))
    assert transformed.shape[1] == model.n_features_out_
    assert len(model.get_feature_names_out()) == model.n_features_out_


def test_shared_kernel_smoothing_rejects_nonfinite_samples_points_and_weights():
    X = np.linspace(0.0, 1.0, 20)
    y = np.sin(X)
    bad_X = X.copy()
    bad_X[3] = np.nan
    bad_weights = np.ones(20)
    bad_weights[4] = np.nan

    with pytest.raises(ValueError, match="finite"):
        KernelRegression().fit(bad_X, y)
    with pytest.raises(ValueError, match="finite"):
        KernelRegression(weights=bad_weights).fit(X, y)
    model = KernelRegression().fit(X, y)
    with pytest.raises(ValueError, match="finite"):
        model.predict(np.array([0.1, np.nan]))

    with pytest.raises(ValueError, match="finite"):
        KernelDensityEstimator().fit(bad_X)
    kde = KernelDensityEstimator().fit(X)
    with pytest.raises(ValueError, match="finite"):
        kde.pdf(np.array([0.1, np.nan]))


def test_gam_validates_parameters_and_data():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    y = np.sin(X[:, 0])
    invalid = [
        (dict(n_splines=4, degree=3), "n_splines"),
        (dict(degree=-1), "degree"),
        (dict(lam=-1.0), "lam"),
        (dict(penalty_order=0), "penalty_order"),
        (dict(knot_method="bad"), "knot_method"),
        (dict(gamma=0.0), "gamma"),
    ]
    for kwargs, match in invalid:
        with pytest.raises(ValueError, match=match):
            GAM(**kwargs).fit(X, y)

    bad_X = X.copy()
    bad_X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        GAM(n_splines=6, lam=1.0).fit(bad_X, y)
    with pytest.raises(ValueError, match="same number"):
        GAM(n_splines=6, lam=1.0).fit(X, y[:-1])
    with pytest.raises(ValueError, match="constant"):
        GAM(n_splines=6, lam=1.0).fit(np.ones((30, 1)), y)


def test_gam_single_feature_predict_accepts_one_dimensional_points():
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    y = np.sin(2 * np.pi * X[:, 0])
    model = GAM(n_splines=7, degree=3, lam=0.5).fit(X, y)
    pred_1d = model.predict(np.array([0.1, 0.4, 0.8]))
    pred_2d = model.predict(np.array([[0.1], [0.4], [0.8]]))
    assert_allclose(pred_1d, pred_2d, rtol=1e-12, atol=1e-12)


def test_binary_evaluation_rejects_nonfinite_threshold():
    with pytest.raises(ValueError, match="threshold"):
        evaluate_binary_classification(
            np.array([0, 1]), np.array([0.2, 0.8]), threshold=np.nan
        )
