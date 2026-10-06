"""Analytic and documentation checks for PR168's linked smoothing user journeys.

The strict xfails identify separately tracked numerical work; this documentation
change deliberately leaves all executable production code unchanged.
"""
import re
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline

from statgpu.nonparametric.kernel_methods import (
    KernelRidge,
    KernelRidgeCV,
    Nystroem,
    pairwise_kernels,
)
from statgpu.nonparametric.splines import (
    SplineTransformer,
    bspline_basis,
    cyclic_cubic_spline_basis,
    natural_cubic_spline_basis,
    thin_plate_spline_basis,
)

ROOT = Path(__file__).resolve().parents[2]


class _IgnoredKernelRidgeWeights(Exception):
    """Zero-weight observations still change the kernel-ridge prediction."""


class _NonfiniteKernelRidgeCVSelection(Exception):
    """The singular zero-alpha candidate was selected with a nonfinite score."""


class _NonperiodicCyclicBasis(Exception):
    """The cyclic basis has unequal one-sided boundary values or derivatives."""


def test_kernel_ridge_matches_rkhs_objective_not_coefficient_ridge():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 8.0, 3.0])
    alpha = 0.7
    model = KernelRidge(alpha=alpha, device="cpu").fit(X, y)
    K = pairwise_kernels(X, metric="rbf", xp=np)
    c = np.asarray(model.dual_coef_).ravel()
    assert_allclose((K + alpha * np.eye(len(X))) @ c, y, atol=1e-12)
    assert_allclose(K @ (K @ c - y) + alpha * K @ c, 0.0, atol=1e-12)
    assert np.linalg.norm(K @ (K @ c - y) + alpha * c) > 0.1


@pytest.mark.parametrize("lang", ["en", "cn"])
def test_linked_kernel_docs_state_objective_and_current_boundaries(lang):
    text = (ROOT / f"docs/{lang}/models/kernel-methods.md").read_text()
    assert r"\alpha c^\top Kc" in text
    assert r"\alpha\lVert c\rVert_2^2" not in text
    assert "sample_weight" in text
    assert "mean_mse" in text and "best_score_" in text
    assert "NumPy SVD" in text
    assert r"Z=K_{nm}V\Lambda^{-1/2}V^\top" in text


def test_nystroem_normalization_and_rank_deficient_output_width():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    model = Nystroem(n_components=4, random_state=42, device="cpu").fit(X)
    Kmm = pairwise_kernels(model.components_, metric="rbf", xp=np)
    U, s, Vt = np.linalg.svd(Kmm, full_matrices=False)
    expected = (U / np.sqrt(np.maximum(s, 1e-12))[None, :]) @ Vt
    assert_allclose(model.normalization_, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(model.eigenvalues_, np.maximum(s, 1e-12))
    features = model.transform(X)
    assert features.shape == (4, 4)
    assert_allclose(features, pairwise_kernels(X, model.components_, metric="rbf", xp=np) @ expected)


def test_natural_basis_and_transformer_dimensions_match_linked_examples():
    x = np.linspace(0.0, 1.0, 500)
    knots = np.linspace(0.1, 0.9, 10)
    natural = natural_cubic_spline_basis(x, knots, xp=np)
    assert natural.shape == (500, 12)
    assert np.ptp(natural[:, 0]) > 0.1  # No dedicated first-column intercept.
    X = np.random.default_rng(42).normal(size=(500, 3))
    model = SplineTransformer(n_knots=10, degree=3, knots="quantile", device="cpu")
    assert model.fit_transform(X).shape == (500, 36)
    assert model.n_features_out_ == 36


@pytest.mark.parametrize("lang", ["en", "cn"])
def test_linked_spline_docs_disclose_boundary_and_inference_limits(lang):
    text = (ROOT / f"docs/{lang}/models/splines.md").read_text()
    assert "n_knots + 2" in text
    assert "n_knots + 4" in text
    assert "boundary_lo" in text and "boundary_hi" in text
    assert "n_jobs" in text
    assert "NumPy" in text
    if lang == "en":
        assert "Do not rely on this function for periodic continuity" in text
        assert "does not provide coefficient inference or confidence bands" in text
        assert "inferred from `x` if `None`" not in text
    else:
        assert "需要周期连续性时，请勿依赖此函数" in text
        assert "不提供系数推断或置信带" in text
        assert "从 `x` 推断" not in text


def test_thin_plate_polynomial_block_is_linear_even_for_order_three():
    x = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 0.5]])
    knots = np.array([[-0.3, 0.2], [0.4, -0.1]])
    B = thin_plate_spline_basis(x, knots, penalty_order=3, xp=np)
    assert B.shape == (3, 5)
    assert_allclose(B[:, -3:], np.column_stack([np.ones(3), x]))


@pytest.mark.xfail(strict=True, raises=_IgnoredKernelRidgeWeights, reason="KernelRidge.fit currently ignores sample_weight")
def test_kernel_ridge_zero_weight_should_equal_excluding_that_observation():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 8.0, 3.0])
    weights = np.array([1.0, 1.0, 0.0, 1.0])
    try:
        weighted = KernelRidge(alpha=0.7, device="cpu").fit(X, y, sample_weight=weights)
    except (ValueError, NotImplementedError) as error:
        # The linked weight issue accepts an explicit unsupported request.
        # An unrelated error must fail rather than count as the known defect.
        message = str(error)
        if "sample_weight" in message and re.search(
            r"unsupported|not supported|not implemented", message, re.IGNORECASE,
        ):
            return  # Strict XPASS prompts removal of the resolved-defect marker.
        raise
    omitted = KernelRidge(alpha=0.7, device="cpu").fit(X[weights > 0], y[weights > 0])
    actual = weighted.predict(X)
    expected = omitted.predict(X)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise _IgnoredKernelRidgeWeights("Zero-weight prediction differs from deleting that row")


@pytest.mark.xfail(strict=True, raises=_NonfiniteKernelRidgeCVSelection, reason="KernelRidgeCV may select zero alpha with nonfinite CV evidence")
def test_kernel_ridge_cv_should_not_select_nan_candidate():
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            model = KernelRidgeCV(
                alphas=[0.0, 1.0], cv=2, kernel="linear", random_state=0, device="cpu",
            ).fit(np.zeros((8, 1)), np.arange(8.0))
    except (ValueError, NotImplementedError) as error:
        # Rejecting the zero-alpha request is an accepted singular-solve policy.
        message = str(error)
        if re.search(r"alphas?", message, re.IGNORECASE) and re.search(
            r"strictly positive|greater than (?:zero|0)|zero.*(?:unsupported|not supported)|"
            r"(?:unsupported|not supported).*zero", message, re.IGNORECASE,
        ):
            return
        raise
    if model.alpha_ == 0.0 and not np.isfinite(model.best_score_):
        raise _NonfiniteKernelRidgeCVSelection("Selected zero alpha with nonfinite CV evidence")
    assert model.alpha_ == 1.0
    assert np.isfinite(model.best_score_)


@pytest.mark.xfail(strict=True, raises=_NonperiodicCyclicBasis, reason="Cyclic projection does not enforce true one-sided boundary derivatives")
def test_cyclic_basis_should_satisfy_analytic_periodicity():
    x = np.linspace(0.0, 1.0, 500)
    knots = np.linspace(0.1, 0.9, 10)
    ordinary = bspline_basis(x, knots, xp=np)
    cyclic = cyclic_cubic_spline_basis(x, knots, xp=np)
    projection = np.linalg.lstsq(ordinary, cyclic, rcond=None)[0]
    assert_allclose(ordinary @ projection, cyclic, atol=1e-12)
    augmented = np.r_[np.zeros(4), knots, np.ones(4)]
    spline = BSpline(augmented, projection, 3)
    for order in (0, 1, 2):
        derivative = spline.derivative(order)
        left, right = derivative(0.0), derivative(1.0)
        assert np.isfinite(left).all() and np.isfinite(right).all()
        if not np.allclose(left, right, rtol=1e-8, atol=1e-8):
            raise _NonperiodicCyclicBasis(f"Boundary derivative order {order} is not periodic")
