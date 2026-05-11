"""Tests for KDE inference utilities."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu._config import cuda_available
from statgpu.inference import (
    KDE,
    KDEBootstrapResult,
    fit_kde,
    kde_pdf,
    kde_bootstrap_confidence_interval,
)
from statgpu.nonparametric import KernelDensityEstimator, select_bandwidth


class TestKDE:
    def test_kde_1d_matches_scipy_scott(self):
        scipy_stats = pytest.importorskip("scipy.stats")

        rng = np.random.default_rng(20260414)
        x = np.concatenate(
            [
                rng.normal(loc=-1.0, scale=0.8, size=220),
                rng.normal(loc=1.3, scale=0.5, size=180),
            ]
        )
        points = np.linspace(-4.0, 4.0, 257)

        kde = fit_kde(x, bandwidth="scott", backend="numpy")
        dens = kde.pdf(points)

        ref = scipy_stats.gaussian_kde(x, bw_method="scott")
        dens_ref = ref(points)

        assert dens.shape == points.shape
        assert np.max(np.abs(dens - dens_ref)) < 5e-6
        assert np.mean(np.abs(dens - dens_ref)) < 2e-6

    def test_kde_2d_weighted_matches_scipy_silverman(self):
        scipy_stats = pytest.importorskip("scipy.stats")

        rng = np.random.default_rng(20260414)
        x = rng.normal(size=(280, 2))
        x[:, 1] = 0.7 * x[:, 0] + 0.6 * x[:, 1]
        w = rng.uniform(0.1, 2.0, size=x.shape[0])
        points = rng.normal(size=(90, 2))

        kde = fit_kde(x, bandwidth="silverman", weights=w, backend="numpy")
        dens = kde(points)

        ref = scipy_stats.gaussian_kde(x.T, bw_method="silverman", weights=w)
        dens_ref = ref(points.T)

        assert dens.shape == (points.shape[0],)
        assert np.max(np.abs(dens - dens_ref)) < 2e-5
        assert np.mean(np.abs(dens - dens_ref)) < 5e-6

    def test_kde_pdf_return_log_matches_pdf(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=200)
        p = np.linspace(-2.5, 2.5, 101)

        pdf = kde_pdf(x, p, bandwidth="scott", backend="numpy")
        log_pdf = kde_pdf(
            x,
            p,
            bandwidth="scott",
            backend="numpy",
            return_log=True,
        )

        assert np.all(pdf > 0.0)
        assert np.allclose(np.log(pdf), log_pdf, atol=1e-12, rtol=1e-12)

    def test_explicit_bandwidth_and_metadata(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=(120, 3))

        kde = fit_kde(x, bandwidth=0.8, backend="numpy")
        meta = kde.to_numpy_metadata()

        assert isinstance(kde, KDE)
        assert np.isclose(kde.bandwidth_factor_, 0.8)
        assert meta["n_samples"] == 120
        assert meta["n_features"] == 3
        assert meta["backend"] == "numpy"
        assert meta["kernel"] == "gaussian"
        assert meta["covariance"].shape == (3, 3)
        assert meta["inv_covariance"].shape == (3, 3)

    def test_supported_kernels_integrate_to_one_approximately(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=800)
        grid = np.linspace(-6.0, 6.0, 2401)

        for kernel in (
            "gaussian",
            "rectangular",
            "triangular",
            "epanechnikov",
            "biweight",
            "triweight",
            "cosine",
            "optcosine",
        ):
            kde = fit_kde(x, bandwidth="scott", kernel=kernel, backend="numpy")
            dens = np.asarray(kde(grid), dtype=np.float64)
            if hasattr(np, "trapezoid"):
                integral = float(np.trapezoid(dens, grid))
            else:
                integral = float(np.trapz(dens, grid))

            assert np.min(dens) >= -1e-12
            assert abs(integral - 1.0) < 0.08

    def test_compact_support_kde_preserves_zero_density_regions(self):
        x_1d = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
        kde_1d = fit_kde(x_1d, bandwidth=0.1, kernel="epanechnikov", backend="numpy")

        pdf_1d = np.asarray(kde_1d.pdf(np.array([5.0])), dtype=np.float64)
        logpdf_1d = np.asarray(kde_1d.logpdf(np.array([5.0])), dtype=np.float64)

        assert pdf_1d.shape == (1,)
        assert pdf_1d[0] == 0.0
        assert logpdf_1d.shape == (1,)
        assert np.isneginf(logpdf_1d[0])

        rng = np.random.default_rng(20260511)
        x_hd = rng.normal(scale=0.1, size=(32, 8))
        points_hd = np.full((2, 8), 10.0, dtype=np.float64)
        kde_hd = fit_kde(x_hd, bandwidth=0.2, kernel="epanechnikov", backend="numpy")

        pdf_hd = np.asarray(kde_hd.pdf(points_hd), dtype=np.float64)
        logpdf_hd = np.asarray(kde_hd.logpdf(points_hd), dtype=np.float64)

        assert np.array_equal(pdf_hd, np.zeros(2, dtype=np.float64))
        assert np.all(np.isneginf(logpdf_hd))

    def test_gaussian_kde_logpdf_stays_finite_in_tail(self):
        x_1d = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        kde_1d = fit_kde(x_1d, bandwidth=0.1, kernel="gaussian", backend="numpy")

        logpdf = np.asarray(kde_1d.logpdf(np.array([50.0])), dtype=np.float64)

        assert logpdf.shape == (1,)
        assert np.isfinite(logpdf[0])
        assert logpdf[0] < 0.0

    def test_nrd_bandwidth_rules_for_1d(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(loc=0.3, scale=1.8, size=600)

        kde_nrd0 = fit_kde(x, bandwidth="nrd0", backend="numpy")
        kde_nrd = fit_kde(x, bandwidth="nrd", backend="numpy")

        bw_nrd0 = float(np.sqrt(np.asarray(kde_nrd0.covariance_)[0, 0]))
        bw_nrd = float(np.sqrt(np.asarray(kde_nrd.covariance_)[0, 0]))

        assert bw_nrd0 > 0.0
        assert bw_nrd > 0.0
        assert bw_nrd > bw_nrd0

    def test_r_style_bandwidth_selectors_for_1d(self):
        rng = np.random.default_rng(20260414)
        x = np.concatenate(
            [
                rng.normal(loc=-1.0, scale=0.7, size=260),
                rng.normal(loc=1.1, scale=0.5, size=240),
            ]
        )
        points = np.linspace(-4.0, 4.0, 121)

        selected = {}
        for bw in ("ucv", "bcv", "sj", "sj-ste", "sj-dpi"):
            kde = fit_kde(x, bandwidth=bw, backend="numpy")
            dens = np.asarray(kde(points), dtype=np.float64)

            bw_abs = float(np.sqrt(np.asarray(kde.covariance_)[0, 0]))
            assert np.isfinite(bw_abs)
            assert bw_abs > 0.0
            assert np.all(dens >= -1e-12)
            selected[bw] = bw_abs

        assert np.isclose(selected["sj"], selected["sj-ste"], rtol=1e-10, atol=1e-12)

    def test_r_style_bandwidth_selectors_support_weighted_samples(self):
        rng = np.random.default_rng(20260414)
        x = np.concatenate(
            [
                rng.normal(loc=-0.9, scale=0.6, size=180),
                rng.normal(loc=1.4, scale=0.7, size=160),
            ]
        )
        w = rng.uniform(0.2, 3.0, size=x.size)
        points = np.linspace(-4.0, 4.0, 111)

        for bw in ("ucv", "bcv", "sj", "sj-ste", "sj-dpi"):
            kde = fit_kde(x, bandwidth=bw, weights=w, backend="numpy")
            dens = np.asarray(kde(points), dtype=np.float64)

            assert np.isfinite(kde.bandwidth_factor_)
            assert kde.bandwidth_factor_ > 0.0
            assert np.all(dens >= -1e-12)
            assert kde.bandwidth_info_.weighted is True

    def test_multivariate_selector_support_for_nrd_and_r_style(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=(260, 3))
        x[:, 1] = 0.55 * x[:, 0] + 0.8 * x[:, 1]
        x[:, 2] = 0.3 * x[:, 0] - 0.2 * x[:, 1] + 0.7 * x[:, 2]
        w = rng.uniform(0.1, 1.8, size=x.shape[0])
        points = rng.normal(size=(40, 3))

        for bw in ("nrd0", "nrd", "sj"):
            kde = fit_kde(x, bandwidth=bw, weights=w, backend="numpy")
            dens = np.asarray(kde(points), dtype=np.float64)

            assert np.isfinite(kde.bandwidth_factor_)
            assert kde.bandwidth_factor_ > 0.0
            assert np.all(np.isfinite(dens))
            assert kde.bandwidth_info_.selector_dimension == 1
            assert kde.bandwidth_info_.multivariate_strategy in ("projection_pca_1d", "none")

    def test_bandwidth_selection_diagnostics_present(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=(140, 2))
        w = rng.uniform(0.5, 2.0, size=x.shape[0])

        kde = fit_kde(x, bandwidth="scott", weights=w, backend="numpy")
        meta = kde.to_numpy_metadata()

        assert "bandwidth_selection" in meta
        assert isinstance(meta["bandwidth_selection"], dict)
        assert meta["bandwidth_selection"]["method"] == "scott"
        assert meta["bandwidth_selection"]["n_features"] == 2

        bw_info = select_bandwidth(
            "silverman",
            n_eff=float(kde.bandwidth_info_.n_eff),
            n_features=2,
            samples_2d=np.asarray(x, dtype=np.float64),
            weights_1d=np.asarray(w / np.sum(w), dtype=np.float64),
            data_cov=np.asarray(meta["covariance"], dtype=np.float64),
            xp=np,
        )
        assert bw_info.factor > 0.0
        assert bw_info.method == "silverman"

    def test_kde_bootstrap_confidence_interval_shapes_and_order(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=240)
        points = np.linspace(-2.5, 2.5, 31)

        result = kde_bootstrap_confidence_interval(
            x,
            points,
            bandwidth="scott",
            kernel="epanechnikov",
            backend="numpy",
            n_resamples=30,
            confidence_level=0.9,
            random_state=7,
            return_bootstrap_samples=True,
        )

        assert isinstance(result, KDEBootstrapResult)
        assert result.estimate.shape == points.shape
        assert result.lower.shape == points.shape
        assert result.upper.shape == points.shape
        assert result.bootstrap_samples is not None
        assert result.bootstrap_samples.shape == (30, points.size)
        assert np.all(result.lower <= result.upper)
        assert np.all(result.lower <= result.estimate + 1e-12)
        assert np.all(result.estimate <= result.upper + 1e-12)

        payload = result.to_dict()
        assert payload["kernel"] == "epanechnikov"
        assert payload["n_resamples"] == 30
        assert "bootstrap_samples" in payload

    def test_invalid_kernel_and_ci_arguments_raise(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=120)
        points = np.linspace(-2.0, 2.0, 13)
        x2 = rng.normal(size=(120, 2))

        with pytest.raises(ValueError):
            fit_kde(x, kernel="invalid-kernel", backend="numpy")

        with pytest.raises(ValueError):
            fit_kde(x2, kernel="cosine", backend="numpy")

        kde_nrd0 = fit_kde(x2, bandwidth="nrd0", backend="numpy")
        kde_ucv = fit_kde(x2, bandwidth="ucv", backend="numpy")
        assert kde_nrd0.bandwidth_factor_ > 0.0
        assert kde_ucv.bandwidth_factor_ > 0.0

        with pytest.raises(ValueError):
            kde_bootstrap_confidence_interval(x, points, n_resamples=0)

        with pytest.raises(ValueError):
            kde_bootstrap_confidence_interval(x, points, confidence_level=1.0)

        with pytest.raises(ValueError):
            kde_bootstrap_confidence_interval(x, points, method="basic")

    def test_invalid_inputs_raise(self):
        rng = np.random.default_rng(20260414)
        x = rng.normal(size=100)

        with pytest.raises(ValueError):
            fit_kde(np.array([1.0]), backend="numpy")

        with pytest.raises(ValueError):
            fit_kde(x, bandwidth=0.0, backend="numpy")

        with pytest.raises(ValueError):
            fit_kde(x, weights=np.ones(10), backend="numpy")

        kde = fit_kde(x, backend="numpy")
        with pytest.raises(ValueError):
            kde.pdf(np.ones((5, 2)))

    def test_kernel_density_estimator_fit_predict_and_score_samples(self):
        rng = np.random.default_rng(20260415)
        x = rng.normal(size=220)
        grid = np.linspace(-3.0, 3.0, 91)

        est = KernelDensityEstimator(bandwidth="scott", kernel="gaussian", backend="numpy")
        est.fit(x)

        pdf = np.asarray(est.predict(grid), dtype=np.float64)
        log_pdf = np.asarray(est.score_samples(grid), dtype=np.float64)

        assert np.all(pdf > 0.0)
        assert np.all(np.isfinite(log_pdf))
        assert np.allclose(np.log(pdf), log_pdf, atol=1e-12, rtol=1e-12)
        assert np.isfinite(est.score(grid))

    @pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
    def test_kde_cupy_matches_numpy(self):
        import cupy as cp

        rng = np.random.default_rng(20260414)
        x_np = rng.normal(size=(260, 2))
        x_np[:, 1] = 0.3 * x_np[:, 0] + x_np[:, 1]
        p_np = rng.normal(size=(80, 2))

        kde_np = fit_kde(x_np, bandwidth="scott", backend="numpy")
        dens_np = kde_np(p_np)

        x_cp = cp.asarray(x_np)
        p_cp = cp.asarray(p_np)
        kde_cp = fit_kde(x_cp, bandwidth="scott", backend="cupy")
        dens_cp = kde_cp(p_cp)
        np.testing.assert_allclose(dens_np, cp.asnumpy(dens_cp), atol=5e-6)
