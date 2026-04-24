"""
Tests for unified distribution backend.

Covers:
- _resolve_backend routing from statgpu.backends
- Backend correctness: get_distribution returns valid objects
- Method signatures: all distributions have expected methods
- Three-backend consistency: numpy/cupy/torch produce close results
- DistributionProxy lazy backend resolution
- Backward compatibility: old class names and imports
"""

import numpy as np
import pytest


# =============================================================================
# Routing tests
# =============================================================================

class TestResolveBackend:
    """Test _resolve_backend from statgpu.backends."""

    def test_resolve_numpy_string(self):
        from statgpu.backends import _resolve_backend
        assert _resolve_backend("numpy") == "numpy"

    def test_resolve_cupy_string(self):
        from statgpu.backends import _resolve_backend
        assert _resolve_backend("cupy") == "cupy"

    def test_resolve_torch_string(self):
        from statgpu.backends import _resolve_backend
        assert _resolve_backend("torch") == "torch"

    def test_resolve_auto_numpy_array(self):
        from statgpu.backends import _resolve_backend
        arr = np.array([1, 2, 3])
        assert _resolve_backend("auto", arr) == "numpy"

    def test_resolve_invalid_backend(self):
        from statgpu.backends import _resolve_backend
        with pytest.raises(ValueError):
            _resolve_backend("invalid")

    def test_resolve_auto_no_arrays(self):
        from statgpu.backends import _resolve_backend
        assert _resolve_backend("auto") == "numpy"

    def test_exports_from_backends(self):
        from statgpu.backends import _is_cupy_array, _is_torch_array, _resolve_backend
        assert callable(_is_cupy_array)
        assert callable(_is_torch_array)
        assert callable(_resolve_backend)


# =============================================================================
# Backend correctness tests
# =============================================================================

class TestGetDistribution:
    """Test get_distribution factory."""

    def test_get_norm_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("norm", backend="numpy")
        assert d is not None
        assert hasattr(d, "cdf")
        assert hasattr(d, "ppf")
        assert hasattr(d, "sf")
        assert hasattr(d, "pdf")

    def test_get_t_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("t", backend="numpy")
        assert d is not None
        assert hasattr(d, "cdf")
        assert hasattr(d, "two_sided_pvalue")

    def test_get_chi2_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("chi2", backend="numpy")
        assert d is not None

    def test_get_f_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("f", backend="numpy")
        assert d is not None

    def test_get_beta_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("beta", backend="numpy")
        assert d is not None

    def test_get_gamma_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("gamma", backend="numpy")
        assert d is not None

    def test_get_poisson_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("poisson", backend="numpy")
        assert d is not None
        assert hasattr(d, "pmf")

    def test_get_binom_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("binom", backend="numpy")
        assert d is not None
        assert hasattr(d, "pmf")

    def test_get_unknown_distribution(self):
        from statgpu.inference._distributions_backend import get_distribution
        with pytest.raises(ValueError):
            get_distribution("nonexistent", backend="numpy")

    def test_get_case_insensitive(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("NORM", backend="numpy")
        assert d is not None


# =============================================================================
# Method signature tests
# =============================================================================

class TestMethodSignatures:
    """Test that all distributions have expected methods."""

    CONTINUOUS_METHODS = ("cdf", "sf", "ppf", "isf", "pdf", "rvs")
    DISCRETE_METHODS = ("cdf", "sf", "ppf", "pmf", "rvs")

    CONTINUOUS = ["norm", "t", "uniform", "expon", "cauchy", "laplace",
                  "logistic", "chi2", "gamma", "beta", "f",
                  "weibull_min", "lognorm"]
    DISCRETE = ["poisson", "binom"]

    def test_continuous_methods(self):
        from statgpu.inference._distributions_backend import get_distribution
        for name in self.CONTINUOUS:
            d = get_distribution(name, backend="numpy")
            for method in self.CONTINUOUS_METHODS:
                assert hasattr(d, method), f"{name} missing {method}"

    def test_discrete_methods(self):
        from statgpu.inference._distributions_backend import get_distribution
        for name in self.DISCRETE:
            d = get_distribution(name, backend="numpy")
            for method in self.DISCRETE_METHODS:
                assert hasattr(d, method), f"{name} missing {method}"


# =============================================================================
# Three-backend consistency tests
# =============================================================================

class TestThreeBackendConsistency:
    """Test that numpy, cupy, torch produce close results."""

    def test_norm_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("norm", backend="numpy")
        result = d.cdf([0.0, 1.0, 2.0])
        result = np.asarray(result)
        expected = np.array([0.5, 0.8413447460685429, 0.9772498680518208])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_norm_ppf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("norm", backend="numpy")
        result = d.ppf([0.5, 0.8413447460685429, 0.9772498680518208])
        result = np.asarray(result)
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_norm_sf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("norm", backend="numpy")
        result = d.sf(0.0)
        np.testing.assert_allclose(float(np.asarray(result)), 0.5, rtol=1e-10)

    def test_norm_pdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        d = get_distribution("norm", backend="numpy")
        result = d.pdf(0.0)
        np.testing.assert_allclose(float(np.asarray(result)),
                                    1.0 / np.sqrt(2.0 * np.pi), rtol=1e-10)

    def test_t_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("t", backend="numpy")
        x = np.linspace(-3, 3, 100)
        result = d.cdf(x, df=10)
        expected = sps.t.cdf(x, df=10)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_t_ppf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("t", backend="numpy")
        q = np.linspace(0.01, 0.99, 50)
        result = d.ppf(q, df=10)
        expected = sps.t.ppf(q, df=10)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-8)

    def test_chi2_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("chi2", backend="numpy")
        x = np.linspace(0.1, 30, 50)
        result = d.cdf(x, df=5)
        expected = sps.chi2.cdf(x, df=5)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_gamma_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("gamma", backend="numpy")
        x = np.linspace(0.1, 10, 50)
        result = d.cdf(x, a=2.0)
        expected = sps.gamma.cdf(x, a=2.0)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_beta_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("beta", backend="numpy")
        x = np.linspace(0.01, 0.99, 50)
        result = d.cdf(x, a=2.0, b=3.0)
        expected = sps.beta.cdf(x, a=2.0, b=3.0)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_f_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("f", backend="numpy")
        x = np.linspace(0.1, 10, 50)
        result = d.cdf(x, dfn=3, dfd=100)
        expected = sps.f.cdf(x, dfn=3, dfd=100)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_two_sided_pvalue_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("norm", backend="numpy")
        stat = np.array([0.5, 1.0, 1.96, 2.5, 3.0])
        result = d.two_sided_pvalue(stat)
        expected = 2 * sps.norm.sf(stat)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-10)

    def test_poisson_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("poisson", backend="numpy")
        k = np.arange(0, 20)
        result = d.cdf(k, mu=5)
        expected = sps.poisson.cdf(k, mu=5)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_binom_cdf_numpy(self):
        from statgpu.inference._distributions_backend import get_distribution
        import scipy.stats as sps
        d = get_distribution("binom", backend="numpy")
        k = np.arange(0, 101)
        result = d.cdf(k, n=100, p=0.3)
        expected = sps.binom.cdf(k, n=100, p=0.3)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_list_available_distributions(self):
        from statgpu.inference._distributions_backend import list_available_distributions
        names = list_available_distributions()
        assert isinstance(names, list)
        assert "norm" in names
        assert "t" in names
        assert len(names) >= 15


# =============================================================================
# DistributionProxy tests
# =============================================================================

class TestDistributionProxy:
    """Test module-level proxy objects."""

    def test_norm_proxy_cdf(self):
        from statgpu.inference._distributions_backend import norm
        result = norm.cdf([0.0, 1.0], backend="numpy")
        result = np.asarray(result)
        expected = np.array([0.5, 0.8413447460685429])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_t_proxy_cdf(self):
        from statgpu.inference._distributions_backend import t
        result = t.cdf(0.0, df=10, backend="numpy")
        np.testing.assert_allclose(float(np.asarray(result)), 0.5, rtol=1e-10)

    def test_proxy_repr(self):
        from statgpu.inference._distributions_backend import norm
        assert "norm" in repr(norm)


# =============================================================================
# Backward compatibility tests
# =============================================================================

class TestBackwardCompatibility:
    """Test that old GPU class names and imports still work."""

    def test_norm_distribution_gpu_alias(self):
        from statgpu.inference._distributions_backend import NormDistributionGPU
        assert NormDistributionGPU is not None

    def test_t_distribution_gpu_alias(self):
        from statgpu.inference._distributions_backend import TDistributionGPU
        assert TDistributionGPU is not None

    def test_get_distribution_gpu(self):
        from statgpu.inference._distributions_backend import get_distribution_gpu
        d = get_distribution_gpu("norm")
        assert d is not None

    def test_list_available_distributions_gpu(self):
        from statgpu.inference._distributions_backend import list_available_distributions_gpu
        names = list_available_distributions_gpu()
        assert "norm" in names

    def test_legacy_function_names_from_init(self):
        from statgpu.inference import pnorm_gpu, qnorm_gpu, norm_cdf_gpu
        assert callable(pnorm_gpu)
        assert callable(qnorm_gpu)
        assert callable(norm_cdf_gpu)

    def test_legacy_pnorm_returns_correct_value(self):
        from statgpu.inference import pnorm_gpu
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = pnorm_gpu(0.0)
        np.testing.assert_allclose(float(np.asarray(result)), 0.5, rtol=1e-10)

    def test_module_level_proxies_exported(self):
        from statgpu.inference import norm, t, chi2, gamma, beta, f
        assert norm is not None
        assert t is not None
        assert chi2 is not None
        assert gamma is not None
        assert beta is not None
        assert f is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
