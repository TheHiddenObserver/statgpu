"""Mutation and repair controls for the cyclic periodicity xfail.

Independent SciPy fixtures exercise the actual marked test body. The known
stencil space must retain its narrow exception; malformed or unrelated spaces
must fail normally, and a complete analytic repair must return successfully.
These are NumPy CPU checks, not a production spline repair or GPU evidence.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from dev.tests import test_pr168_survival_smoothing_cycle3 as contracts


def _fixture(space):
    x = np.linspace(0., 1., 500)
    knots = np.linspace(.1, .9, 10)
    basis = BSpline(np.r_[np.zeros(4), knots, np.ones(4)], np.eye(14), 3)
    if space == "analytic":
        constraints = np.vstack([
            basis.derivative(order)(0.) - basis.derivative(order)(1.)
            for order in (0, 1, 2)
        ])
    else:
        assert space == "known_defect"
        # Independent SciPy evaluation of the known zero-exterior stencil.
        h = 1e-6
        values = basis(np.array([[0., h], [1., 1. - h]]))
        constraints = np.vstack([
            values[0, 0] - values[1, 0],
            (values[0, 1] + values[1, 1]) / (2*h),
            ((values[0, 1] - 2*values[0, 0])
             - (values[1, 1] - 2*values[1, 0])) / h**2,
        ])
    coefficients = null_space(constraints)
    assert coefficients.shape == (14, 11 if space == "analytic" else 12)
    return basis(x), coefficients


def _transform(values, kind):
    n_columns = values.shape[1]
    if kind == "identity":
        return values
    if kind == "signed_permutation":
        return values[:, ::-1] * (-1.)**np.arange(n_columns)
    if kind == "orthogonal":
        rotation, _ = np.linalg.qr(np.random.default_rng(168).normal(
            size=(n_columns, n_columns)))
        return values @ rotation
    if kind in {"mixed", "mixed_disparate_scale"}:
        change = np.diag(np.linspace(.5, 2., n_columns))
        change += .1 * np.triu(np.ones((n_columns, n_columns)), 1)
        values = values @ change
        if kind == "mixed":
            return values
    if kind in {"disparate_scale", "mixed_disparate_scale"}:
        return values * np.geomspace(1e-150, 1e150, n_columns)
    assert kind == "tiny_scale"
    return values * 1e-14


def _run_with(monkeypatch, values):
    def replacement(x, knots, xp=None):
        assert x.shape == (500,) and knots.shape == (10,)
        assert xp is np
        return values.copy()

    monkeypatch.setattr(contracts, "cyclic_cubic_spline_basis", replacement)
    contracts.test_cyclic_basis_should_satisfy_analytic_periodicity()


@pytest.mark.parametrize("space", ["known_defect", "analytic"])
@pytest.mark.parametrize("transform", ["identity", "signed_permutation", "orthogonal",
                                       "mixed", "tiny_scale", "disparate_scale",
                                       "mixed_disparate_scale"])
def test_cyclic_guard_preserves_known_defect_and_accepts_analytic_repair(
        monkeypatch, space, transform):
    ordinary, coefficients = _fixture(space)
    values = _transform(ordinary @ coefficients, transform)
    if space == "known_defect":
        with pytest.raises(contracts._NonperiodicCyclicBasis):
            _run_with(monkeypatch, values)
    else:
        _run_with(monkeypatch, values)


@pytest.mark.parametrize("space", ["known_defect", "analytic"])
@pytest.mark.parametrize("fault, message", [
    ("duplicate", "rank deficient"),
    ("linear_combination", "rank deficient"),
    ("scaled_duplicate", "rank deficient"),
    ("zero", "zero column"),
    ("zero_column", "zero column"),
    ("rows", "shape"),
    ("vector", "shape"),
    ("extra_columns", "width"),
    ("no_columns", "width"),
    ("nan", "nonfinite"),
    ("inf", "nonfinite"),
])
def test_cyclic_guard_rejects_degenerate_bases(monkeypatch, space, fault, message):
    ordinary, coefficients = _fixture(space)
    values = ordinary @ coefficients
    if fault in {"duplicate", "scaled_duplicate"}:
        values[:, -1] = values[:, 0]
        if fault == "scaled_duplicate":
            values = _transform(values, "disparate_scale")
    elif fault == "linear_combination":
        values[:, -1] = values[:, 0] + 2*values[:, 1]
    elif fault == "zero":
        values[:] = 0.
    elif fault == "zero_column":
        values[:, -1] = 0.
    elif fault == "rows":
        values = values[:-1]
    elif fault == "vector":
        values = values[:, 0]
    elif fault == "extra_columns":
        values = ordinary
    elif fault == "no_columns":
        values = values[:, :0]
    else:
        assert fault in {"nan", "inf"}
        values[0, 0] = np.nan if fault == "nan" else np.inf

    # The known-defect exception does not inherit AssertionError.
    with pytest.raises(AssertionError, match=message):
        _run_with(monkeypatch, values)


@pytest.mark.parametrize("space", ["known_defect", "analytic"])
@pytest.mark.parametrize("transform", ["identity", "tiny_scale", "disparate_scale",
                                       "mixed_disparate_scale"])
def test_cyclic_guard_rejects_columnwise_off_spline_values(monkeypatch, space, transform):
    ordinary, coefficients = _fixture(space)
    values = ordinary @ coefficients
    if transform == "mixed_disparate_scale":
        values = _transform(values, "mixed")
    noise = np.sin(np.linspace(0., 20., len(values)))
    noise -= ordinary @ np.linalg.lstsq(ordinary, noise, rcond=None)[0]
    noise *= .2 * np.linalg.norm(values[:, 1]) / np.linalg.norm(noise)
    assert_allclose(ordinary.T @ noise, 0., atol=1e-12)
    assert_allclose(np.linalg.norm(noise) / np.linalg.norm(values[:, 1]), .2)
    values[:, 1] += noise
    scaling = "disparate_scale" if transform == "mixed_disparate_scale" else transform
    values = _transform(values, scaling)
    balanced = values / np.max(np.abs(values), axis=0)
    assert np.linalg.matrix_rank(balanced) == values.shape[1]
    with pytest.raises(AssertionError, match="off-spline values"):
        _run_with(monkeypatch, values)


@pytest.mark.parametrize("n_columns", [11, 12])
@pytest.mark.parametrize("kind", ["ordinary_subset", "random_subspace"])
def test_cyclic_guard_rejects_unrelated_full_rank_spline_spaces(monkeypatch, n_columns, kind):
    ordinary, _ = _fixture("known_defect")
    if kind == "ordinary_subset":
        values = ordinary[:, :n_columns]
    else:
        coefficients, _ = np.linalg.qr(np.random.default_rng(216).normal(
            size=(ordinary.shape[1], n_columns)))
        values = ordinary @ coefficients
    assert np.linalg.matrix_rank(values) == n_columns
    with pytest.raises(AssertionError, match="Unexpected cyclic"):
        _run_with(monkeypatch, values)


def test_cyclic_guard_rejects_incomplete_periodic_subspace(monkeypatch):
    ordinary, coefficients = _fixture("analytic")
    with pytest.raises(AssertionError, match="width"):
        _run_with(monkeypatch, ordinary @ coefficients[:, :-1])


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_cyclic_guard_propagates_unrelated_runtime_errors(monkeypatch, error_type):
    def broken(*args, **kwargs):
        raise error_type("unrelated cyclic failure")

    monkeypatch.setattr(contracts, "cyclic_cubic_spline_basis", broken)
    with pytest.raises(error_type, match="unrelated cyclic failure"):
        contracts.test_cyclic_basis_should_satisfy_analytic_periodicity()


def test_cyclic_marker_is_strict_and_uses_only_the_known_defect_exception():
    marker, = [mark for mark in
               contracts.test_cyclic_basis_should_satisfy_analytic_periodicity.pytestmark
               if mark.name == "xfail"]
    assert marker.kwargs["strict"] is True
    assert marker.kwargs["raises"] is contracts._NonperiodicCyclicBasis
