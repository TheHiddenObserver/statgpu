"""Regression contracts for Issue #120 CuPy inverse-LUT ordering.

These tests exercise the CuPy special-function implementation without requiring
CUDA by supplying NumPy/SciPy as API-compatible stand-ins. Physical CuPy CUDA
validation remains a separate acceptance gate.

The regression file deliberately covers three layers:

1. raw inverse-special-function cache semantics;
2. public distribution PPF/ISF surfaces that consume those primitives;
3. representative panel-inference consumers that request the CuPy t
   distribution.

This keeps the blast-radius coverage focused on paths that actually use the
CuPy inverse-LUT implementation. CPU-only SciPy consumers are intentionally not
included here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.special as sc
from numpy.testing import assert_allclose
from scipy import stats

import statgpu.inference._distributions_backend as dist_backend
from statgpu.inference._distributions_backend import (
    CuPySpecialFunctions,
    TDistributionBase,
)


# benchmark_distributions.py defines 1e-6 as the PASS threshold for PPF/ISF;
# inverse special functions are documented there as approximately 1e-7 accurate.
_INVERSE_ABS_ACCURACY = 1e-6


def _hosted_cupy_special_functions() -> CuPySpecialFunctions:
    """Construct the CuPy implementation with NumPy/SciPy test doubles."""
    sf = CuPySpecialFunctions.__new__(CuPySpecialFunctions)
    sf._cp = np
    sf._csp = sc
    sf.use_lut = True
    sf._betaincinv_lut = {}
    sf._gammaincinv_lut = {}
    return sf


def _patch_hosted_cupy_factory(monkeypatch):
    """Route ``backend='cupy'`` through the exact CuPy implementation on CPU."""
    original = dist_backend._make_sf

    def _make_sf(backend, device=None, *, use_lut=True):
        if backend == "cupy":
            sf = _hosted_cupy_special_functions()
            sf.use_lut = bool(use_lut)
            return sf
        return original(backend, device, use_lut=use_lut)

    monkeypatch.setattr(dist_backend, "_make_sf", _make_sf)


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


@pytest.mark.parametrize(
    ("name", "kwargs", "scipy_dist"),
    [
        ("t", {"df": 45}, stats.t),
        ("beta", {"a": 2.5, "b": 5.5}, stats.beta),
        ("f", {"dfn": 5, "dfd": 24}, stats.f),
        ("gamma", {"a": 4.0}, stats.gamma),
        ("chi2", {"df": 8}, stats.chi2),
    ],
)
def test_cupy_public_inverse_distribution_surface_matches_scipy(
    monkeypatch, name, kwargs, scipy_dist
):
    """PPF/ISF and inverse round-trips stay correct through the public factory."""
    _patch_hosted_cupy_factory(monkeypatch)
    dist = dist_backend.get_distribution(name, backend="cupy", use_lut=True)
    # Avoid the exact Student-t symmetry point q=0.5 here: the LUT contract is
    # approximate (1e-6 absolute PPF/ISF threshold). A dedicated test below
    # checks that the median residual remains inside that documented contract.
    probability = np.asarray(
        [0.01, 0.025, 0.20, 0.40, 0.60, 0.95, 0.975], dtype=np.float64
    )

    actual_ppf = np.asarray(dist.ppf(probability, **kwargs))
    expected_ppf = np.asarray(scipy_dist.ppf(probability, **kwargs))
    actual_isf = np.asarray(dist.isf(probability, **kwargs))
    expected_isf = np.asarray(scipy_dist.isf(probability, **kwargs))

    assert_allclose(actual_ppf, expected_ppf, rtol=5e-8, atol=5e-9)
    assert_allclose(actual_isf, expected_isf, rtol=5e-8, atol=5e-9)
    assert_allclose(
        np.asarray(dist.cdf(actual_ppf, **kwargs)),
        probability,
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        np.asarray(dist.sf(actual_isf, **kwargs)),
        probability,
        rtol=5e-8,
        atol=5e-9,
    )


@pytest.mark.parametrize("df", [1.0, 10.0, 45.0, 60.0, 80.0])
def test_cupy_t_inverse_quantiles_cover_lut_and_native_fallback(monkeypatch, df):
    """Exercise t quantiles across the inverse-beta LUT eligibility boundary."""
    _patch_hosted_cupy_factory(monkeypatch)
    dist = dist_backend.get_distribution("t", backend="cupy", use_lut=True)
    probability = np.asarray([0.025, 0.25, 0.75, 0.975], dtype=np.float64)

    assert_allclose(
        np.asarray(dist.ppf(probability, df=df)),
        stats.t.ppf(probability, df=df),
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        np.asarray(dist.isf(probability, df=df)),
        stats.t.isf(probability, df=df),
        rtol=5e-8,
        atol=5e-9,
    )


@pytest.mark.parametrize("df", [1.0, 10.0, 45.0, 60.0, 80.0])
def test_cupy_t_median_and_symmetry_stay_within_inverse_accuracy_contract(
    monkeypatch, df
):
    """Track the existing LUT median residual separately from Issue #120."""
    _patch_hosted_cupy_factory(monkeypatch)
    dist = dist_backend.get_distribution("t", backend="cupy", use_lut=True)

    median = float(np.asarray(dist.ppf(0.5, df=df)))
    assert abs(median) < _INVERSE_ABS_ACCURACY

    tails = np.asarray(dist.ppf(np.asarray([0.025, 0.975]), df=df))
    assert abs(float(tails[0] + tails[1])) < _INVERSE_ABS_ACCURACY


def test_cupy_distribution_proxies_route_inverse_quantiles_through_fixed_factory(
    monkeypatch,
):
    """Module-level public proxies must hit the same corrected CuPy primitives."""
    _patch_hosted_cupy_factory(monkeypatch)
    probability = np.asarray([0.025, 0.25, 0.75, 0.975], dtype=np.float64)

    assert_allclose(
        np.asarray(dist_backend.t.ppf(probability, df=45, backend="cupy")),
        stats.t.ppf(probability, df=45),
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        np.asarray(
            dist_backend.beta.ppf(
                probability, a=2.5, b=5.5, backend="cupy"
            )
        ),
        stats.beta.ppf(probability, a=2.5, b=5.5),
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        np.asarray(dist_backend.gamma.ppf(probability, a=4.0, backend="cupy")),
        stats.gamma.ppf(probability, a=4.0),
        rtol=5e-8,
        atol=5e-9,
    )


def test_shared_panel_inference_uses_correct_cupy_t_quantile(monkeypatch):
    """Exercise the retained shared panel utility with the corrected CuPy t path."""
    _patch_hosted_cupy_factory(monkeypatch)
    from statgpu.panel._utils import compute_panel_inference

    rng = np.random.default_rng(120)
    n, k = 48, 3
    X = rng.normal(size=(n, k))
    params = np.asarray([0.35, 1.05, -0.60], dtype=np.float64)
    resid = rng.normal(scale=0.25, size=n)
    scale = float(np.sum(resid ** 2) / 45.0)
    model = SimpleNamespace()

    compute_panel_inference(
        model,
        X,
        resid,
        params,
        scale,
        n,
        k,
        np,
        "cupy",
        "nonrobust",
        0.05,
        dist_df=45,
    )

    critical = stats.t.isf(0.025, 45)
    expected = np.column_stack(
        [
            model.coef_ - critical * model.bse_,
            model.coef_ + critical * model.bse_,
        ]
    )
    assert np.all(model.conf_int_[:, 1] > model.conf_int_[:, 0])
    assert_allclose(model.conf_int_, expected, rtol=5e-8, atol=5e-9)


@pytest.mark.parametrize("model_name", ["PanelOLS", "RandomEffects"])
def test_stage_a_panel_estimators_use_shared_cupy_t_inference(monkeypatch, model_name):
    """Stage-A estimators inherit the shared inference path that uses CuPy t quantiles."""
    _patch_hosted_cupy_factory(monkeypatch)
    from statgpu.panel._base import BasePanelModel
    from statgpu.panel._fixed_effects import PanelOLS
    from statgpu.panel._random_effects import RandomEffects

    model_cls = {"PanelOLS": PanelOLS, "RandomEffects": RandomEffects}[model_name]
    assert issubclass(model_cls, BasePanelModel)

    rng = np.random.default_rng(121 if model_name == "PanelOLS" else 122)
    n, k = 48, 3
    X = rng.normal(size=(n, k))
    coef = np.asarray([0.35, 1.05, -0.60], dtype=np.float64)
    resid = rng.normal(scale=0.25, size=n)
    scale = float(np.sum(resid ** 2) / 45.0)
    holder = SimpleNamespace(alpha=0.05)
    backend = SimpleNamespace(xp=np, name="cupy")

    model_cls._panel_store_ols_inference(
        holder,
        X,
        resid,
        coef,
        scale=scale,
        df_resid=45,
        backend=backend,
        cov_type="nonrobust",
        distribution_df=45,
    )

    critical = stats.t.isf(0.025, 45)
    expected = np.column_stack(
        [
            holder.coef_ - critical * holder.bse_,
            holder.coef_ + critical * holder.bse_,
        ]
    )
    assert np.all(holder.conf_int_[:, 1] > holder.conf_int_[:, 0])
    assert_allclose(holder.conf_int_, expected, rtol=5e-8, atol=5e-9)
