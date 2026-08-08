"""Hosted compatibility coverage for Issue #120 legacy inverse-quantile aliases."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.special as sc
from numpy.testing import assert_allclose
from scipy import stats

import statgpu.backends as backends
import statgpu.inference._distributions_backend as dist_backend
from statgpu.inference._distributions_backend import CuPySpecialFunctions
from statgpu.linear_model.legacy import _distributions_legacy_gpu as legacy


def _hosted_cupy_special_functions() -> CuPySpecialFunctions:
    sf = CuPySpecialFunctions.__new__(CuPySpecialFunctions)
    sf._cp = np
    sf._csp = sc
    sf.use_lut = True
    sf._betaincinv_lut = {}
    sf._gammaincinv_lut = {}
    return sf


def test_legacy_inverse_aliases_route_through_corrected_cupy_quantiles(monkeypatch):
    """R-style aliases preserve the corrected CuPy PPF path and keyword shapes."""
    original_make_sf = dist_backend._make_sf

    def _make_sf(backend, device=None, *, use_lut=True):
        if backend == "cupy":
            sf = _hosted_cupy_special_functions()
            sf.use_lut = bool(use_lut)
            return sf
        return original_make_sf(backend, device, use_lut=use_lut)

    # DistributionProxy resolves the backend at call time. Force that public
    # resolver to CuPy while retaining NumPy/SciPy as API-compatible stand-ins.
    monkeypatch.setattr(
        backends,
        "_resolve_backend",
        lambda *arrays, backend="auto": SimpleNamespace(name="cupy"),
    )
    monkeypatch.setattr(dist_backend, "_make_sf", _make_sf)

    q = np.asarray([0.025, 0.25, 0.75, 0.975], dtype=np.float64)

    assert_allclose(
        legacy.qt_gpu(q, 45),
        stats.t.ppf(q, df=45),
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        legacy.qbeta_gpu(q, a=2.5, b=5.5),
        stats.beta.ppf(q, a=2.5, b=5.5),
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        legacy.qf_gpu(q, dfn=5, dfd=24),
        stats.f.ppf(q, dfn=5, dfd=24),
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        legacy.qgamma_gpu(q, a=4.0),
        stats.gamma.ppf(q, a=4.0),
        rtol=5e-8,
        atol=5e-9,
    )
    assert_allclose(
        legacy.qchisq_gpu(q, df=8),
        stats.chi2.ppf(q, df=8),
        rtol=5e-8,
        atol=5e-9,
    )
