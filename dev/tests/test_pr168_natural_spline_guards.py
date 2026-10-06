"""Mutation/repair checks for the two narrowly classified issue #224 xfails.

All replacement bases are built independently with SciPy. These tests exercise
the actual marked test bodies without requiring the production defect to remain.
They are NumPy CPU checks, not GPU evidence or a production spline repair.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from dev.tests import test_pr168_survival_smoothing_cycle4 as contracts

SCALES = [1e6, 1e-8, 1e-7]
DTYPES = [np.float32, np.float64]


def _ordinary(x, knots):
    augmented = np.r_[np.full(4, x[0]), knots, np.full(4, x[-1])]
    return BSpline(augmented, np.eye(len(knots) + 4), 3)


def _analytic_constraints(basis):
    lo, hi = basis.t[0], basis.t[-1]
    return basis.derivative(2)([lo, hi]) * (hi - lo)**2


def _defective_constraints(x, knots):
    """Synthetic known-defect fixtures, independent of production output."""
    if float(x[-1]) > 1e5:
        constraints = np.zeros((2, len(knots) + 4))
        constraints[0, 1:3] = [-4., 1.]
        constraints[1, -2:] = [-1., 1.]
        return constraints
    step = 1e-6
    lo, hi = float(x[0]), float(x[-1])
    # A quadratic forward/backward stencil sampled in the incorrectly enlarged
    # knot interval is the small-scale bug, not the analytic endpoint derivative.
    samples = np.array([[lo, lo + step, lo + 2*step],
                        [hi, hi - step, hi - 2*step]])
    augmented = np.r_[np.full(4, samples.min()), knots,
                      np.full(4, samples.max())]
    values = BSpline(augmented, np.eye(len(knots) + 4), 3)(samples)
    return np.einsum("ijk,j->ik", values, [1., -2., 1.])


def _transform(coefficients, kind):
    n_columns = coefficients.shape[1]
    if kind == "identity":
        return coefficients
    if kind == "signed_permutation":
        return coefficients[:, ::-1] * (-1.)**np.arange(n_columns)
    if kind == "orthogonal":
        rotation, _ = np.linalg.qr(np.random.default_rng(224).normal(
            size=(n_columns, n_columns)))
        return coefficients @ rotation
    if kind == "mixed":
        change = np.diag(np.linspace(.5, 2., n_columns))
        change += .1 * np.triu(np.ones((n_columns, n_columns)), 1)
        return coefficients @ change
    assert kind == "tiny_scale"
    return coefficients * 1e-14


def _run_guard(scale, dtype):
    if scale == 1e6:
        contracts.test_natural_basis_should_represent_constants_independent_of_units(dtype)
    else:
        contracts.test_natural_endpoint_curvature_should_remain_small_after_unit_change(dtype, scale)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fault", ["zero", "rows", "columns", "nan", "inf",
                                  "rank", "ordinary", "wrong_subspace", "off_spline",
                                  "tiny_off_spline"])
def test_natural_guards_reject_unrelated_basis_corruption(monkeypatch, scale, dtype, fault):
    def corrupted(x, knots, xp=None):
        basis = _ordinary(x, knots)
        values = basis(x)
        good = values @ null_space(_analytic_constraints(basis))
        if fault == "zero":
            return np.zeros_like(good)
        if fault == "rows":
            return good[:-1]
        if fault == "columns":
            return good[:, :-1]
        if fault in {"nan", "inf"}:
            good[0, 0] = np.nan if fault == "nan" else np.inf
            return good
        if fault == "rank":
            good[:, -1] = good[:, 0]
            return good
        if fault == "ordinary":
            # The exact original small-range counterexample remains finite,
            # full-rank and representable by ordinary cubics.
            return values[:, :good.shape[1]]
        if fault == "wrong_subspace":
            coefficients, _ = np.linalg.qr(np.random.default_rng(225).normal(
                size=(values.shape[1], good.shape[1])))
            return values @ coefficients
        if fault == "tiny_off_spline":
            # Preserve the inferred analytic coefficient space and constants,
            # but add a non-cubic component. An absolute reconstruction tolerance
            # would hide it after shrinking the columns.
            noise = np.sin(20 * np.asarray(x, dtype=float) / x[-1])
            noise -= values @ np.linalg.lstsq(values, noise, rcond=None)[0]
            constant = np.linalg.lstsq(good, np.ones(len(x)), rcond=None)[0]
            direction = np.r_[-constant[1], constant[0], np.zeros(good.shape[1] - 2)]
            return (good + .1 * noise[:, None] * direction) * 1e-14
        assert fault == "off_spline"
        good[:, 0] += .1 * np.sin(20 * np.asarray(x, dtype=float) / x[-1])
        return good

    monkeypatch.setattr(contracts, "natural_cubic_spline_basis", corrupted)
    # The custom expected-failure exceptions do not inherit AssertionError.
    with pytest.raises(AssertionError):
        _run_guard(scale, dtype)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_natural_guards_propagate_unrelated_exceptions(monkeypatch, scale, dtype):
    def broken(*args, **kwargs):
        raise RuntimeError("unrelated spline failure")

    monkeypatch.setattr(contracts, "natural_cubic_spline_basis", broken)
    with pytest.raises(RuntimeError, match="unrelated spline failure"):
        _run_guard(scale, dtype)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transform", ["identity", "signed_permutation", "orthogonal",
                                      "mixed", "tiny_scale"])
def test_known_defect_classification_is_basis_invariant(monkeypatch, scale, dtype, transform):
    def known_defect(x, knots, xp=None):
        coefficients = null_space(_defective_constraints(x, knots))
        return _ordinary(x, knots)(x) @ _transform(coefficients, transform)

    monkeypatch.setattr(contracts, "natural_cubic_spline_basis", known_defect)
    sentinel = (contracts._NaturalConstantMismatch if scale == 1e6
                else contracts._NaturalCurvatureMismatch)
    with pytest.raises(sentinel):
        _run_guard(scale, dtype)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boundary", [0, 1], ids=["left_only", "right_only"])
def test_natural_guards_reject_one_boundary_repairs(monkeypatch, scale, dtype, boundary):
    def partial_repair(x, knots, xp=None):
        basis = _ordinary(x, knots)
        analytic = _analytic_constraints(basis)
        constraints = _defective_constraints(x, knots)
        constraints[boundary] = analytic[boundary]
        coefficients = null_space(constraints)
        assert coefficients.shape == (len(knots) + 4, len(knots) + 2)
        assert_allclose(analytic[boundary] @ coefficients, 0., atol=1e-10)
        assert np.linalg.norm(analytic[1 - boundary] @ coefficients) > 1.
        return basis(x) @ coefficients

    monkeypatch.setattr(contracts, "natural_cubic_spline_basis", partial_repair)
    with pytest.raises(AssertionError):
        _run_guard(scale, dtype)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transform", ["identity", "signed_permutation", "orthogonal",
                                      "mixed", "tiny_scale"])
def test_analytic_repair_succeeds_in_actual_strict_xfail_body(monkeypatch, scale, dtype, transform):
    def analytic_repair(x, knots, xp=None):
        basis = _ordinary(x, knots)
        constraints = _analytic_constraints(basis)
        coefficients = _transform(null_space(constraints), transform)
        values = basis(x) @ coefficients
        assert_allclose(constraints @ coefficients, 0., atol=1e-10)
        assert_allclose(values @ np.linalg.lstsq(values, np.ones(len(x)), rcond=None)[0],
                        1., atol=1e-12)
        return values

    monkeypatch.setattr(contracts, "natural_cubic_spline_basis", analytic_repair)
    # Ordinary return becomes strict XPASS when pytest runs the marked test.
    _run_guard(scale, dtype)


@pytest.mark.parametrize("function,sentinel", [
    (contracts.test_natural_basis_should_represent_constants_independent_of_units,
     contracts._NaturalConstantMismatch),
    (contracts.test_natural_endpoint_curvature_should_remain_small_after_unit_change,
     contracts._NaturalCurvatureMismatch),
])
def test_natural_expected_failure_markers_remain_strict_and_narrow(function, sentinel):
    markers = [mark for mark in function.pytestmark if mark.name == "xfail"]
    assert len(markers) == 1
    assert markers[0].kwargs["strict"] is True
    assert markers[0].kwargs["raises"] is sentinel
