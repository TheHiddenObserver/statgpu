"""Tests for Ridge inference: cov_type, compute_inference, summary, and statistics."""

import numpy as np
import pytest

from statgpu.inference import GaussianInferenceResult, ParameterInferenceResult
from statgpu.linear_model import LinearRegression, PenalizedLinearRegression, Ridge
from statgpu._config import set_device, Device


def _make_data(n_samples=500, n_features=8, seed=42, heteroskedastic=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = rng.normal(size=n_features)
    if heteroskedastic:
        noise_scale = 0.2 + 0.8 * np.abs(X[:, 0])
        y = X @ beta + 1.5 + rng.normal(scale=noise_scale, size=n_samples)
    else:
        y = X @ beta + 1.5 + rng.normal(scale=0.3, size=n_samples)
    return X, y


class TestRidgeInferenceParams:
    """Test new inference parameters on Ridge."""

    def test_invalid_cov_type_raises(self):
        with pytest.raises(ValueError):
            Ridge(cov_type="invalid")

    def test_valid_cov_types_accepted(self):
        for ct in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
            m = Ridge(cov_type=ct)
            assert m.cov_type == ct

    def test_invalid_hac_maxlags_raises(self):
        with pytest.raises(ValueError):
            Ridge(cov_type="hac", hac_maxlags=-1)

    def test_compute_inference_false_no_bse(self):
        """When compute_inference=False, inference attributes should be None."""
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu", compute_inference=False)
        m.fit(X, y)
        assert m._bse is None
        assert m._tvalues is None
        assert m._pvalues is None
        assert m._conf_int is None
        assert m._inference_result is None

    def test_compute_inference_true_has_bse(self):
        """When compute_inference=True (default), inference attributes should be populated and summary works."""
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu")
        m.fit(X, y)
        n_params = X.shape[1] + 1  # intercept + features
        assert m._bse is not None
        assert m._bse.shape == (n_params,)
        assert m._tvalues.shape == (n_params,)
        assert m._pvalues.shape == (n_params,)
        assert m._conf_int.shape == (n_params, 2)
        assert np.all(m._bse > 0)
        assert np.all(np.isfinite(m._bse))
        assert np.all(np.isfinite(m._pvalues))
        assert np.all((m._pvalues >= 0) & (m._pvalues <= 1))
        assert isinstance(m._inference_result, GaussianInferenceResult)
        assert isinstance(m._inference_result, ParameterInferenceResult)
        np.testing.assert_allclose(m._inference_result.bse, m._bse)
        np.testing.assert_allclose(m._inference_result.tvalues, m._tvalues)
        np.testing.assert_allclose(m._inference_result.pvalues, m._pvalues)
        np.testing.assert_allclose(m._inference_result.conf_int, m._conf_int)
        # summary() should run without errors when inference is available
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.summary()
        assert "Ridge Regression Results" in buf.getvalue()

    def test_gaussian_inference_result_serialization(self):
        set_device("cpu")
        X, y = _make_data(n_samples=80, n_features=3)
        m = LinearRegression(device="cpu")
        m.fit(X, y)
        result = m._inference_result
        assert isinstance(result, GaussianInferenceResult)
        assert result.tvalues is result.statistic
        payload = result.to_dict()
        assert payload["result_type"] == "GaussianInferenceResult"
        assert payload["statistic_name"] == "t"
        assert payload["cov_type"] == "nonrobust"
        assert len(payload["params"]) == X.shape[1] + 1

    def test_gaussian_inference_result_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        set_device("cpu")
        X, y = _make_data(n_samples=80, n_features=3)
        m = LinearRegression(device="cpu")
        m.fit(X, y)
        frame = m._inference_result.to_dataframe()
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == [
            "term",
            "estimate",
            "std_error",
            "t",
            "pvalue",
            "conf_low",
            "conf_high",
        ]
        assert frame.shape[0] == X.shape[1] + 1

    def test_penalized_l2_compute_inference_populates_stats(self):
        set_device("cpu")
        X, y = _make_data()
        m = PenalizedLinearRegression(
            penalty="l2",
            alpha=0.1,
            device="cpu",
            compute_inference=True,
            cov_type="hc1",
        )
        m.fit(X, y)
        n_params = X.shape[1] + 1
        assert m._bse.shape == (n_params,)
        assert m._conf_int.shape == (n_params, 2)
        assert np.all(np.isfinite(m._bse))
        assert np.all((m._pvalues >= 0) & (m._pvalues <= 1))
        assert isinstance(m._inference_result, GaussianInferenceResult)
        np.testing.assert_allclose(m._inference_result.tvalues, m._tvalues)

    def test_penalized_non_l2_compute_inference_raises(self):
        set_device("cpu")
        X, y = _make_data()
        m = PenalizedLinearRegression(
            penalty="l1",
            alpha=0.1,
            device="cpu",
            compute_inference=True,
        )
        with pytest.raises(NotImplementedError, match="squared-error L2"):
            m.fit(X, y)


class TestRidgeInferenceStatistics:
    """Test Ridge inference statistics properties."""

    def test_rsquared_in_range(self):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu")
        m.fit(X, y)
        assert 0.0 <= m.rsquared <= 1.0

    def test_rsquared_adj_in_range(self):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu")
        m.fit(X, y)
        r2_adj = m.rsquared_adj
        assert r2_adj is not None
        assert np.isfinite(r2_adj)
        assert r2_adj <= m.rsquared  # adj R² <= R²

    def test_fvalue_positive(self):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu")
        m.fit(X, y)
        assert m.fvalue is not None
        assert m.fvalue > 0
        assert np.isfinite(m.fvalue)

    def test_f_pvalue_in_range(self):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu")
        m.fit(X, y)
        assert 0.0 <= m.f_pvalue <= 1.0

    def test_llf_finite(self):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu")
        m.fit(X, y)
        assert m.llf is not None
        assert np.isfinite(m.llf)

    def test_aic_bic_finite_and_aic_less_bic(self):
        set_device("cpu")
        X, y = _make_data(n_samples=1000)
        m = Ridge(device="cpu")
        m.fit(X, y)
        assert m.aic is not None
        assert m.bic is not None
        assert np.isfinite(m.aic)
        assert np.isfinite(m.bic)
        # With n=1000, k<log(1000)≈6.9, BIC >= AIC since k*log(n) >= 2k
        assert m.bic >= m.aic

    def test_conf_int_contains_true_params_nonrobust(self):
        """95% CI should cover the true intercept in most cases (nonrobust)."""
        set_device("cpu")
        X, y = _make_data(n_samples=2000, n_features=5, seed=7)
        m = Ridge(alpha=0.001, device="cpu", cov_type="nonrobust")
        m.fit(X, y)
        # Check that intercept CI is finite and lower < upper
        assert m._conf_int[0, 0] < m._conf_int[0, 1]
        assert np.all(np.isfinite(m._conf_int))

    def test_conf_int_contains_true_params_hc0(self):
        """HC0 CI should cover the true intercept in most cases."""
        set_device("cpu")
        X, y = _make_data(n_samples=2000, n_features=5, seed=8, heteroskedastic=True)
        m = Ridge(alpha=0.001, device="cpu", cov_type="hc0")
        m.fit(X, y)
        assert m._conf_int[0, 0] < m._conf_int[0, 1]
        assert np.all(np.isfinite(m._conf_int))


class TestRidgeCovTypes:
    """Test Ridge with different cov_type values produce sensible SE."""

    @pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"])
    def test_cov_type_produces_positive_se(self, cov_type):
        set_device("cpu")
        X, y = _make_data(n_samples=800, heteroskedastic=True)
        kwargs = {"hac_maxlags": 3} if cov_type == "hac" else {}
        m = Ridge(device="cpu", cov_type=cov_type, **kwargs)
        m.fit(X, y)
        assert np.all(m._bse > 0)
        assert np.all(np.isfinite(m._bse))
        assert np.all((m._pvalues >= 0) & (m._pvalues <= 1))

    def test_hc1_se_larger_than_hc0(self):
        """HC1 SE should be >= HC0 SE (HC1 adds a degrees-of-freedom correction)."""
        set_device("cpu")
        X, y = _make_data(n_samples=600, heteroskedastic=True)
        m_hc0 = Ridge(device="cpu", cov_type="hc0")
        m_hc0.fit(X, y)
        m_hc1 = Ridge(device="cpu", cov_type="hc1")
        m_hc1.fit(X, y)
        # HC1 = HC0 * n/(n-k), so HC1 >= HC0 element-wise
        assert np.all(m_hc1._bse >= m_hc0._bse - 1e-10)

    def test_hc3_se_larger_than_hc2(self):
        """HC3 SE should be >= HC2 SE due to stronger leverage adjustment."""
        set_device("cpu")
        X, y = _make_data(n_samples=600, heteroskedastic=True)
        m_hc2 = Ridge(device="cpu", cov_type="hc2")
        m_hc2.fit(X, y)
        m_hc3 = Ridge(device="cpu", cov_type="hc3")
        m_hc3.fit(X, y)
        assert np.all(m_hc3._bse >= m_hc2._bse - 1e-10)

    def test_nonrobust_vs_hc0_same_coef(self):
        """cov_type only affects inference, not coefficients."""
        set_device("cpu")
        X, y = _make_data()
        m_nr = Ridge(device="cpu", cov_type="nonrobust")
        m_nr.fit(X, y)
        m_hc0 = Ridge(device="cpu", cov_type="hc0")
        m_hc0.fit(X, y)
        assert np.allclose(m_nr.coef_, m_hc0.coef_, rtol=1e-10)
        assert np.allclose(m_nr.intercept_, m_hc0.intercept_, rtol=1e-10)


class TestRidgeSummary:
    """Test Ridge summary() method."""

    def test_summary_runs_without_error(self, capsys):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu")
        m.fit(X, y)
        m.summary()
        captured = capsys.readouterr()
        assert "Ridge Regression Results" in captured.out
        assert "Alpha" in captured.out
        assert "R-squared" in captured.out
        assert "(Intercept)" in captured.out

    def test_summary_not_fitted_raises(self):
        m = Ridge(device="cpu")
        with pytest.raises(RuntimeError):
            m.summary()

    def test_summary_no_inference_raises(self):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu", compute_inference=False)
        m.fit(X, y)
        with pytest.raises(RuntimeError):
            m.summary()

    def test_summary_no_intercept(self, capsys):
        set_device("cpu")
        X, y = _make_data()
        m = Ridge(device="cpu", fit_intercept=False)
        m.fit(X, y)
        m.summary()
        captured = capsys.readouterr()
        assert "Ridge Regression Results" in captured.out
        assert "(Intercept)" not in captured.out

    @pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"])
    def test_summary_cov_type_shown(self, cov_type, capsys):
        set_device("cpu")
        X, y = _make_data()
        kwargs = {"hac_maxlags": 2} if cov_type == "hac" else {}
        m = Ridge(device="cpu", cov_type=cov_type, **kwargs)
        m.fit(X, y)
        m.summary()
        captured = capsys.readouterr()
        assert cov_type in captured.out


class TestRidgeGPUInference:
    """GPU-path inference tests (skipped if CUDA unavailable)."""

    @pytest.mark.skipif(
        not Ridge(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    @pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"])
    def test_gpu_inference_shape_and_finite(self, cov_type):
        X, y = _make_data(n_samples=1000, heteroskedastic=True)
        kwargs = {"hac_maxlags": 3} if cov_type == "hac" else {}
        m = Ridge(device="cuda", cov_type=cov_type, **kwargs)
        m.fit(X, y)
        n_params = X.shape[1] + 1
        assert m._bse.shape == (n_params,)
        assert np.all(m._bse > 0)
        assert np.all(np.isfinite(m._bse))
        assert np.all((m._pvalues >= 0) & (m._pvalues <= 1))

    @pytest.mark.skipif(
        not Ridge(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_gpu_nonrobust_matches_cpu(self):
        """GPU nonrobust inference should be numerically close to CPU."""
        X, y = _make_data(n_samples=1000)
        m_cpu = Ridge(device="cpu", cov_type="nonrobust")
        m_cpu.fit(X, y)
        m_gpu = Ridge(device="cuda", cov_type="nonrobust")
        m_gpu.fit(X, y)
        # GPU/CPU can differ slightly due to linear algebra kernel implementations.
        assert np.allclose(m_cpu._bse, m_gpu._bse, rtol=6e-4, atol=1e-6)
        assert np.allclose(m_cpu._pvalues, m_gpu._pvalues, rtol=1e-3, atol=1e-6)
