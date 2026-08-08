"""Regression contracts for Issue #120 CuPy inverse-LUT ordering.

These tests exercise the CuPy special-function implementation without requiring
CUDA by supplying NumPy/SciPy as API-compatible stand-ins.  Physical CuPy CUDA
validation remains a separate acceptance gate.
"""

from __future__ import annotations

import numpy as np
import scipy.special as sc
from numpy.testing import assert_allclose
from scipy import stats

from statgpu.inference._distributions_backend import (
    CuPySpecialFunctions,
    TDistributionBase,
)


def _hosted_cupy_special_functions() -> CuPySpecialFunctions:
    """Construct the CuPy implementation with NumPy/SciPy test doubles."""
    sf = CuPySpecialFunctions.__new__(CuPySpecialFunctions)
    sf._cp = np
    sf._csp = sc
    sf.use_lut = True
    sf._betaincinv_lut = {}
    sf._gammaincinv_lut = {}
    return sf


def test_cupy_betaincinv_lut_preserves_x_y_cache_order_for_panel_t_tail():
    sf = _hosted_cupy_special_functions()
    probability = np.asarray([0.01, 0.05, 0.25, 0.50, 0.90], dtype=np.float64)

    actual = sf.betaincinv(22.5, 0.5, probability)
    expected = sc.betaincinv(22.5, 0.5, probability)
    assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)

    # Exercise the cached path again: the regression was specifically caused by
    # storing (x_grid, y_grid) and then unpacking that tuple in reverse order.
    cached = sf.betaincinv(22.5, 0.5, probability)
    assert len(sf._betaincinv_lut) == 1
    assert_allclose(cached, expected, rtol=2e-9, atol=2e-11)


def test_cupy_t_critical_value_matches_scipy_for_pr119_failure_case():
    sf = _hosted_cupy_special_functions()
    dist = TDistributionBase(sf)

    actual = float(np.asarray(dist.isf(0.025, 45)))
    expected = float(stats.t.isf(0.025, 45))

    assert actual > 1.9  # A swapped LUT collapsed this to approximately zero.
    assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)


def test_cupy_panel_failure_case_confidence_intervals_have_correct_width():
    sf = _hosted_cupy_special_functions()
    critical = float(np.asarray(TDistributionBase(sf).isf(0.025, 45)))

    coef = np.asarray([0.36067077, 1.06308319, -0.61715928])
    bse = np.asarray([0.06958899, 0.08102734, 0.08449355])
    actual = np.column_stack([coef - critical * bse, coef + critical * bse])

    expected_critical = stats.t.isf(0.025, 45)
    expected = np.column_stack(
        [coef - expected_critical * bse, coef + expected_critical * bse]
    )
    assert np.all(actual[:, 1] > actual[:, 0])
    assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)


def test_cupy_gammaincinv_lut_preserves_x_y_cache_order():
    sf = _hosted_cupy_special_functions()
    probability = np.asarray([0.01, 0.20, 0.50, 0.95], dtype=np.float64)

    actual = sf.gammaincinv(4.0, probability)
    expected = sc.gammaincinv(4.0, probability)
    assert_allclose(actual, expected, rtol=1e-8, atol=5e-9)

    cached = sf.gammaincinv(4.0, probability)
    assert len(sf._gammaincinv_lut) == 1
    assert_allclose(cached, expected, rtol=1e-8, atol=5e-9)
