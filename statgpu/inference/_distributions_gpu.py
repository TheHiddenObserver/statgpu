"""GPU distribution helpers with scipy.stats / R-style naming.

This module centralizes object-style probability-distribution utilities used by
GPU inference paths.

Function-style compatibility APIs are separated into
``_distributions_legacy_gpu``.
R-style names (for example ``pnorm_gpu``/``qnorm_gpu``) are retained for
compatibility, while non-R historical names (for example ``*_cdf_gpu``)
are soft-deprecated.

All native distribution paths return CuPy arrays/scalars and stay on GPU.
Optional SciPy fallback is exposed only via explicit APIs.
"""

import numpy as np
from functools import lru_cache

from ._distribution_utils_gpu import (
    gammainc_gpu,
    gammaincc_gpu,
    gammaincinv_gpu,
    gammaln_gpu,
    regularized_betainc_gpu,
    regularized_betaincinv_gpu,
    scipy_dist_call_gpu,
)

class NormDistributionGPU:
    """scipy.stats.norm-like GPU distribution object.

    Examples
    --------
    - norm.cdf(x)
    - norm.ppf(q)
    - norm.rvs(size=1000)
    """

    @staticmethod
    def _cdf_standard(x):
        """Standard normal CDF kernel used by class/object APIs."""
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        try:
            import cupyx.scipy.special as csp

            erf_x = csp.erf(x_gpu / cp.sqrt(cp.array(2.0, dtype=cp.float64)))
        except Exception as exc:
            raise RuntimeError("cupyx.scipy.special.erf is required for GPU backend") from exc
        return 0.5 * (1.0 + erf_x)

    @staticmethod
    def _sf_standard(x):
        """Standard normal SF kernel used by class/object APIs."""
        import cupy as cp

        return cp.clip(1.0 - NormDistributionGPU._cdf_standard(x), 0.0, 1.0)

    @staticmethod
    def _ppf_standard(q):
        """Standard normal quantile kernel used by class/object APIs."""
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        try:
            import cupyx.scipy.special as csp

            return -cp.sqrt(cp.array(2.0, dtype=cp.float64)) * csp.erfcinv(2.0 * q_gpu)
        except Exception as exc:
            raise RuntimeError("cupyx.scipy.special.erfcinv is required for GPU backend") from exc

    @staticmethod
    def _isf_standard(q):
        """Standard normal inverse survival kernel used by class/object APIs."""
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        return NormDistributionGPU._ppf_standard(1.0 - q_gpu)

    @staticmethod
    def _two_sided_pvalue_standard(stat_abs):
        """Standard normal two-sided p-value kernel."""
        import cupy as cp

        stat_gpu = cp.asarray(stat_abs, dtype=cp.float64)
        return cp.minimum(1.0, 2.0 * NormDistributionGPU._sf_standard(cp.abs(stat_gpu)))

    @staticmethod
    def _two_sided_critical_value_standard(alpha):
        """Positive two-sided critical value kernel for standard normal."""
        import cupy as cp

        a = float(alpha)
        if not (0.0 < a < 1.0):
            return cp.array(cp.nan, dtype=cp.float64)
        return NormDistributionGPU._ppf_standard(1.0 - a / 2.0)

    @staticmethod
    def cdf(x, *, loc=0.0, scale=1.0):
        """Normal CDF on GPU (supports location/scale)."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(x, dtype=cp.float64), cp.nan, dtype=cp.float64)
        x_std = (cp.asarray(x, dtype=cp.float64) - float(loc)) / scale_f
        return NormDistributionGPU._cdf_standard(x_std)

    @staticmethod
    def sf(x, *, loc=0.0, scale=1.0):
        """Normal survival function on GPU."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(x, dtype=cp.float64), cp.nan, dtype=cp.float64)
        x_std = (cp.asarray(x, dtype=cp.float64) - float(loc)) / scale_f
        return NormDistributionGPU._sf_standard(x_std)

    @staticmethod
    def ppf(q, *, loc=0.0, scale=1.0):
        """Normal quantile on GPU (supports location/scale)."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(q, dtype=cp.float64), cp.nan, dtype=cp.float64)
        z = NormDistributionGPU._ppf_standard(q)
        return float(loc) + scale_f * z

    @staticmethod
    def isf(q, *, loc=0.0, scale=1.0):
        """Normal inverse survival function on GPU."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(q, dtype=cp.float64), cp.nan, dtype=cp.float64)
        z = NormDistributionGPU._isf_standard(q)
        return float(loc) + scale_f * z

    @staticmethod
    def pdf(x, *, loc=0.0, scale=1.0):
        """Normal PDF on GPU (supports location/scale)."""
        import cupy as cp

        scale_f = float(scale)
        x_gpu = cp.asarray(x, dtype=cp.float64)
        if scale_f <= 0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        norm_const = cp.sqrt(cp.array(2.0 * cp.pi, dtype=cp.float64))
        return cp.exp(-0.5 * cp.square(z)) / (scale_f * norm_const)

    @staticmethod
    def two_sided_pvalue(stat_abs):
        """Two-sided p-value for standard normal test statistics."""
        return NormDistributionGPU._two_sided_pvalue_standard(stat_abs)

    @staticmethod
    def two_sided_critical_value(alpha):
        """Positive two-sided critical value for standard normal."""
        return NormDistributionGPU._two_sided_critical_value_standard(alpha)

    @staticmethod
    def rvs(*, size=None, loc=0.0, scale=1.0, dtype=None):
        """Draw random variates from Normal on GPU."""
        import cupy as cp

        out = cp.random.normal(loc=float(loc), scale=float(scale), size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class TDistributionGPU:
    """scipy.stats.t-like GPU distribution object.

    Examples
    --------
    - t.cdf(x, df=10)
    - t.ppf(q, df=10)
    - t.rvs(df=10, size=1000)
    """

    @staticmethod
    def _cdf_standard(x, df):
        """Standardized Student t CDF kernel used by class/object APIs."""
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        df_val = float(df)
        if df_val <= 0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)

        z = df_val / (df_val + cp.square(cp.abs(x_gpu)))
        ibeta = regularized_betainc_gpu(df_val / 2.0, 0.5, z)
        lower_tail = 0.5 * ibeta
        return cp.where(x_gpu >= 0.0, 1.0 - lower_tail, lower_tail)

    @staticmethod
    def _sf_standard(x, df):
        """Standardized Student t SF kernel used by class/object APIs."""
        import cupy as cp

        return cp.clip(1.0 - TDistributionGPU._cdf_standard(x, df), 0.0, 1.0)

    @staticmethod
    def _two_sided_pvalue_standard(stat_abs, df):
        """Standardized two-sided Student t p-value kernel."""
        import cupy as cp

        stat_gpu = cp.asarray(stat_abs, dtype=cp.float64)
        df_val = float(df)
        if df_val <= 0:
            return cp.full_like(stat_gpu, cp.nan, dtype=cp.float64)

        z = df_val / (df_val + cp.square(cp.abs(stat_gpu)))
        return regularized_betainc_gpu(df_val / 2.0, 0.5, z)

    @staticmethod
    def _ppf_standard(q, df, *, max_bisect_steps=60):
        """Standardized Student t quantile kernel used by class/object APIs."""
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        df_val = float(df)
        if df_val <= 0:
            return cp.full_like(q_gpu, cp.nan, dtype=cp.float64)

        out = cp.full(q_gpu.shape, cp.nan, dtype=cp.float64)
        out = cp.where(q_gpu == 0.0, -cp.inf, out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)

        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        if not bool(cp.any(valid).item()):
            return out

        try:
            tail = cp.minimum(q_gpu, 1.0 - q_gpu)
            y = 2.0 * tail
            y_inv = regularized_betaincinv_gpu(df_val / 2.0, 0.5, y)
            x_pos = cp.sqrt(df_val * (1.0 - y_inv) / y_inv)
            quant = cp.where(q_gpu >= 0.5, x_pos, -x_pos)
            return cp.where(valid, quant, out)
        except Exception as exc:
            raise RuntimeError("cupyx.scipy.special.betaincinv is required for GPU backend") from exc

    @staticmethod
    def cdf(x, df, *, loc=0.0, scale=1.0):
        """Student t CDF on GPU (supports location/scale)."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(x, dtype=cp.float64), cp.nan, dtype=cp.float64)
        x_std = (cp.asarray(x, dtype=cp.float64) - float(loc)) / scale_f
        return TDistributionGPU._cdf_standard(x_std, df)

    @staticmethod
    def sf(x, df, *, loc=0.0, scale=1.0):
        """Student t survival function on GPU."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(x, dtype=cp.float64), cp.nan, dtype=cp.float64)
        x_std = (cp.asarray(x, dtype=cp.float64) - float(loc)) / scale_f
        return TDistributionGPU._sf_standard(x_std, df)

    @staticmethod
    def ppf(q, df, *, loc=0.0, scale=1.0, max_bisect_steps=60):
        """Student t quantile on GPU (supports location/scale)."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(q, dtype=cp.float64), cp.nan, dtype=cp.float64)
        z = TDistributionGPU._ppf_standard(q, df, max_bisect_steps=max_bisect_steps)
        return float(loc) + scale_f * z

    @staticmethod
    def isf(q, df, *, loc=0.0, scale=1.0, max_bisect_steps=60):
        """Student t inverse survival function on GPU."""
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0:
            return cp.full_like(cp.asarray(q, dtype=cp.float64), cp.nan, dtype=cp.float64)
        q_gpu = cp.asarray(q, dtype=cp.float64)
        z = TDistributionGPU._ppf_standard(1.0 - q_gpu, df, max_bisect_steps=max_bisect_steps)
        return float(loc) + scale_f * z

    @staticmethod
    def pdf(x, df, *, loc=0.0, scale=1.0):
        """Student t PDF on GPU (supports location/scale)."""
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        df_val = float(df)
        scale_f = float(scale)
        if df_val <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)

        z = (x_gpu - float(loc)) / scale_f
        half_nu = df_val / 2.0
        log_coef = (
            gammaln_gpu((df_val + 1.0) / 2.0)
            - gammaln_gpu(half_nu)
            - 0.5 * (cp.log(cp.asarray(df_val, dtype=cp.float64)) + cp.log(cp.asarray(cp.pi, dtype=cp.float64)))
        )
        log_pdf = log_coef - ((df_val + 1.0) / 2.0) * cp.log1p(cp.square(z) / df_val) - cp.log(
            cp.asarray(scale_f, dtype=cp.float64)
        )
        return cp.exp(log_pdf)

    @staticmethod
    def two_sided_pvalue(stat_abs, df):
        """Two-sided p-value for Student t statistics."""
        return TDistributionGPU._two_sided_pvalue_standard(stat_abs, df)

    @staticmethod
    def two_sided_critical_value(alpha, df, *, max_bisect_steps=60):
        """Positive two-sided critical value for Student t."""
        import cupy as cp

        a = float(alpha)
        if not (0.0 < a < 1.0):
            return cp.array(cp.nan, dtype=cp.float64)
        return TDistributionGPU._ppf_standard(1.0 - a / 2.0, df, max_bisect_steps=max_bisect_steps)

    @staticmethod
    def rvs(df, *, size=None, loc=0.0, scale=1.0, dtype=None):
        """Draw random variates from Student t on GPU."""
        import cupy as cp

        base = cp.random.standard_t(df=float(df), size=size)
        if dtype is not None:
            base = base.astype(dtype)
        return float(loc) + float(scale) * cp.asarray(base)


class UniformDistributionGPU:
    """scipy.stats.uniform-like GPU distribution object."""

    @staticmethod
    def cdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        scale_f = float(scale)
        x_gpu = cp.asarray(x, dtype=cp.float64)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return cp.clip(z, 0.0, 1.0)

    @staticmethod
    def sf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        return cp.clip(1.0 - UniformDistributionGPU.cdf(x, loc=loc, scale=scale), 0.0, 1.0)

    @staticmethod
    def ppf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if scale_f <= 0.0:
            return out
        valid = (q_gpu >= 0.0) & (q_gpu <= 1.0)
        return cp.where(valid, float(loc) + scale_f * q_gpu, out)

    @staticmethod
    def isf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        return UniformDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), loc=loc, scale=scale)

    @staticmethod
    def pdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        in_support = (z >= 0.0) & (z <= 1.0)
        return cp.where(in_support, 1.0 / scale_f, 0.0)

    @staticmethod
    def rvs(*, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = cp.random.uniform(low=float(loc), high=float(loc) + scale_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class ExponDistributionGPU:
    """scipy.stats.expon-like GPU distribution object."""

    @staticmethod
    def cdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return cp.where(z <= 0.0, 0.0, 1.0 - cp.exp(-z))

    @staticmethod
    def sf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return cp.where(z <= 0.0, 1.0, cp.exp(-z))

    @staticmethod
    def ppf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, float(loc), out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        return cp.where(valid, float(loc) - scale_f * cp.log1p(-q_gpu), out)

    @staticmethod
    def isf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        return ExponDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), loc=loc, scale=scale)

    @staticmethod
    def pdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return cp.where(z >= 0.0, cp.exp(-z) / scale_f, 0.0)

    @staticmethod
    def rvs(*, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = float(loc) + cp.random.exponential(scale=scale_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class CauchyDistributionGPU:
    """scipy.stats.cauchy-like GPU distribution object."""

    @staticmethod
    def cdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return 0.5 + cp.arctan(z) / cp.pi

    @staticmethod
    def sf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        return cp.clip(1.0 - CauchyDistributionGPU.cdf(x, loc=loc, scale=scale), 0.0, 1.0)

    @staticmethod
    def ppf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, -cp.inf, out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        return cp.where(valid, float(loc) + scale_f * cp.tan(cp.pi * (q_gpu - 0.5)), out)

    @staticmethod
    def isf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        return CauchyDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), loc=loc, scale=scale)

    @staticmethod
    def pdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return 1.0 / (cp.pi * scale_f * (1.0 + cp.square(z)))

    @staticmethod
    def rvs(*, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        u = cp.random.random(size=size)
        out = float(loc) + scale_f * cp.tan(cp.pi * (u - 0.5))
        if dtype is not None:
            out = out.astype(dtype)
        return out


class LaplaceDistributionGPU:
    """scipy.stats.laplace-like GPU distribution object."""

    @staticmethod
    def cdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return cp.where(z < 0.0, 0.5 * cp.exp(z), 1.0 - 0.5 * cp.exp(-z))

    @staticmethod
    def sf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return cp.where(z < 0.0, 1.0 - 0.5 * cp.exp(z), 0.5 * cp.exp(-z))

    @staticmethod
    def ppf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, -cp.inf, out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        lower = (q_gpu > 0.0) & (q_gpu < 0.5)
        upper = (q_gpu >= 0.5) & (q_gpu < 1.0)
        out = cp.where(lower, float(loc) + scale_f * cp.log(2.0 * q_gpu), out)
        out = cp.where(upper, float(loc) - scale_f * cp.log(2.0 * (1.0 - q_gpu)), out)
        return out

    @staticmethod
    def isf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        return LaplaceDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), loc=loc, scale=scale)

    @staticmethod
    def pdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = cp.abs((x_gpu - float(loc)) / scale_f)
        return 0.5 * cp.exp(-z) / scale_f

    @staticmethod
    def rvs(*, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = cp.random.laplace(loc=float(loc), scale=scale_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class LogisticDistributionGPU:
    """scipy.stats.logistic-like GPU distribution object."""

    @staticmethod
    def cdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return 1.0 / (1.0 + cp.exp(-z))

    @staticmethod
    def sf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (x_gpu - float(loc)) / scale_f
        return 1.0 / (1.0 + cp.exp(z))

    @staticmethod
    def ppf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, -cp.inf, out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        return cp.where(valid, float(loc) + scale_f * cp.log(q_gpu / (1.0 - q_gpu)), out)

    @staticmethod
    def isf(q, *, loc=0.0, scale=1.0):
        import cupy as cp

        return LogisticDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), loc=loc, scale=scale)

    @staticmethod
    def pdf(x, *, loc=0.0, scale=1.0):
        import cupy as cp

        cdf_x = LogisticDistributionGPU.cdf(x, loc=loc, scale=scale)
        scale_f = float(scale)
        if scale_f <= 0.0:
            return cdf_x
        return cdf_x * (1.0 - cdf_x) / scale_f

    @staticmethod
    def rvs(*, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        scale_f = float(scale)
        if scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        u = cp.random.random(size=size)
        out = float(loc) + scale_f * cp.log(u / (1.0 - u))
        if dtype is not None:
            out = out.astype(dtype)
        return out


class Chi2DistributionGPU:
    """scipy.stats.chi2-like GPU distribution object."""

    @staticmethod
    def cdf(x, df):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        df_f = float(df)
        if df_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = x_gpu / 2.0
        return cp.where(x_gpu <= 0.0, 0.0, gammainc_gpu(df_f / 2.0, y))

    @staticmethod
    def sf(x, df):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        df_f = float(df)
        if df_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = x_gpu / 2.0
        return cp.where(x_gpu <= 0.0, 1.0, gammaincc_gpu(df_f / 2.0, y))

    @staticmethod
    def ppf(q, df):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        df_f = float(df)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if df_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, 0.0, out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        quant = 2.0 * gammaincinv_gpu(df_f / 2.0, q_gpu)
        return cp.where(valid, quant, out)

    @staticmethod
    def isf(q, df):
        import cupy as cp

        return Chi2DistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), df)

    @staticmethod
    def pdf(x, df):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        df_f = float(df)
        if df_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = cp.maximum(x_gpu, 1e-300)
        logpdf = ((df_f / 2.0) - 1.0) * cp.log(y) - y / 2.0 - (df_f / 2.0) * cp.log(2.0) - gammaln_gpu(df_f / 2.0)
        return cp.where(x_gpu > 0.0, cp.exp(logpdf), 0.0)

    @staticmethod
    def rvs(df, *, size=None, dtype=None):
        import cupy as cp

        df_f = float(df)
        if df_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = cp.random.chisquare(df=df_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class GammaDistributionGPU:
    """scipy.stats.gamma-like GPU distribution object."""

    @staticmethod
    def cdf(x, a, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        a_f = float(a)
        scale_f = float(scale)
        if a_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        return cp.where(y <= 0.0, 0.0, gammainc_gpu(a_f, y))

    @staticmethod
    def sf(x, a, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        a_f = float(a)
        scale_f = float(scale)
        if a_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        return cp.where(y <= 0.0, 1.0, gammaincc_gpu(a_f, y))

    @staticmethod
    def ppf(q, a, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        a_f = float(a)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if a_f <= 0.0 or scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, float(loc), out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        quant = float(loc) + scale_f * gammaincinv_gpu(a_f, q_gpu)
        return cp.where(valid, quant, out)

    @staticmethod
    def isf(q, a, *, loc=0.0, scale=1.0):
        import cupy as cp

        return GammaDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), a, loc=loc, scale=scale)

    @staticmethod
    def pdf(x, a, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        a_f = float(a)
        scale_f = float(scale)
        if a_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        y_safe = cp.maximum(y, 1e-300)
        logpdf = (a_f - 1.0) * cp.log(y_safe) - y_safe - gammaln_gpu(a_f) - cp.log(scale_f)
        return cp.where(y > 0.0, cp.exp(logpdf), 0.0)

    @staticmethod
    def rvs(a, *, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        a_f = float(a)
        scale_f = float(scale)
        if a_f <= 0.0 or scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = float(loc) + cp.random.gamma(shape=a_f, scale=scale_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class BetaDistributionGPU:
    """scipy.stats.beta-like GPU distribution object."""

    @staticmethod
    def cdf(x, a, b, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        a_f = float(a)
        b_f = float(b)
        scale_f = float(scale)
        if a_f <= 0.0 or b_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        core = regularized_betainc_gpu(a_f, b_f, cp.clip(y, 0.0, 1.0))
        out = cp.where(y <= 0.0, 0.0, core)
        return cp.where(y >= 1.0, 1.0, out)

    @staticmethod
    def sf(x, a, b, *, loc=0.0, scale=1.0):
        import cupy as cp

        return cp.clip(1.0 - BetaDistributionGPU.cdf(x, a, b, loc=loc, scale=scale), 0.0, 1.0)

    @staticmethod
    def ppf(q, a, b, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        a_f = float(a)
        b_f = float(b)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if a_f <= 0.0 or b_f <= 0.0 or scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, float(loc), out)
        out = cp.where(q_gpu == 1.0, float(loc) + scale_f, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        quant = float(loc) + scale_f * regularized_betaincinv_gpu(a_f, b_f, q_gpu)
        return cp.where(valid, quant, out)

    @staticmethod
    def isf(q, a, b, *, loc=0.0, scale=1.0):
        import cupy as cp

        return BetaDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), a, b, loc=loc, scale=scale)

    @staticmethod
    def pdf(x, a, b, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        a_f = float(a)
        b_f = float(b)
        scale_f = float(scale)
        if a_f <= 0.0 or b_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        y_safe = cp.clip(y, 1e-300, 1.0 - 1e-300)
        betaln = gammaln_gpu(a_f) + gammaln_gpu(b_f) - gammaln_gpu(a_f + b_f)
        logpdf = (a_f - 1.0) * cp.log(y_safe) + (b_f - 1.0) * cp.log1p(-y_safe) - betaln - cp.log(scale_f)
        in_support = (y > 0.0) & (y < 1.0)
        return cp.where(in_support, cp.exp(logpdf), 0.0)

    @staticmethod
    def rvs(a, b, *, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        a_f = float(a)
        b_f = float(b)
        scale_f = float(scale)
        if a_f <= 0.0 or b_f <= 0.0 or scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = float(loc) + scale_f * cp.random.beta(a_f, b_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FDistributionGPU:
    """scipy.stats.f-like GPU distribution object."""

    @staticmethod
    def cdf(x, dfn, dfd):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        dfn_f = float(dfn)
        dfd_f = float(dfd)
        if dfn_f <= 0.0 or dfd_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        z = (dfn_f * cp.maximum(x_gpu, 0.0)) / (dfn_f * cp.maximum(x_gpu, 0.0) + dfd_f)
        core = regularized_betainc_gpu(dfn_f / 2.0, dfd_f / 2.0, z)
        return cp.where(x_gpu <= 0.0, 0.0, core)

    @staticmethod
    def sf(x, dfn, dfd):
        import cupy as cp

        return cp.clip(1.0 - FDistributionGPU.cdf(x, dfn, dfd), 0.0, 1.0)

    @staticmethod
    def ppf(q, dfn, dfd):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        dfn_f = float(dfn)
        dfd_f = float(dfd)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if dfn_f <= 0.0 or dfd_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, 0.0, out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        z = regularized_betaincinv_gpu(dfn_f / 2.0, dfd_f / 2.0, q_gpu)
        x = (dfd_f * z) / (dfn_f * (1.0 - z))
        return cp.where(valid, x, out)

    @staticmethod
    def isf(q, dfn, dfd):
        import cupy as cp

        return FDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), dfn, dfd)

    @staticmethod
    def pdf(x, dfn, dfd):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        dfn_f = float(dfn)
        dfd_f = float(dfd)
        if dfn_f <= 0.0 or dfd_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        a = dfn_f / 2.0
        b = dfd_f / 2.0
        x_safe = cp.maximum(x_gpu, 1e-300)
        betaln = gammaln_gpu(a) + gammaln_gpu(b) - gammaln_gpu(a + b)
        logpdf = a * cp.log(dfn_f / dfd_f) + (a - 1.0) * cp.log(x_safe) - betaln - (a + b) * cp.log1p((dfn_f / dfd_f) * x_safe)
        return cp.where(x_gpu > 0.0, cp.exp(logpdf), 0.0)

    @staticmethod
    def rvs(dfn, dfd, *, size=None, dtype=None):
        import cupy as cp

        dfn_f = float(dfn)
        dfd_f = float(dfd)
        if dfn_f <= 0.0 or dfd_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = cp.random.f(dfn_f, dfd_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class WeibullMinDistributionGPU:
    """scipy.stats.weibull_min-like GPU distribution object."""

    @staticmethod
    def cdf(x, c, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        c_f = float(c)
        scale_f = float(scale)
        if c_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        yc = cp.power(cp.maximum(y, 0.0), c_f)
        return cp.where(y <= 0.0, 0.0, 1.0 - cp.exp(-yc))

    @staticmethod
    def sf(x, c, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        c_f = float(c)
        scale_f = float(scale)
        if c_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        yc = cp.power(cp.maximum(y, 0.0), c_f)
        return cp.where(y <= 0.0, 1.0, cp.exp(-yc))

    @staticmethod
    def ppf(q, c, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        c_f = float(c)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if c_f <= 0.0 or scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, float(loc), out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        quant = float(loc) + scale_f * cp.power(-cp.log1p(-q_gpu), 1.0 / c_f)
        return cp.where(valid, quant, out)

    @staticmethod
    def isf(q, c, *, loc=0.0, scale=1.0):
        import cupy as cp

        return WeibullMinDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), c, loc=loc, scale=scale)

    @staticmethod
    def pdf(x, c, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        c_f = float(c)
        scale_f = float(scale)
        if c_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        y_pos = cp.maximum(y, 1e-300)
        logpdf = cp.log(c_f / scale_f) + (c_f - 1.0) * cp.log(y_pos) - cp.power(y_pos, c_f)
        return cp.where(y > 0.0, cp.exp(logpdf), 0.0)

    @staticmethod
    def rvs(c, *, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        c_f = float(c)
        scale_f = float(scale)
        if c_f <= 0.0 or scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = float(loc) + scale_f * cp.random.weibull(c_f, size=size)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class LognormDistributionGPU:
    """scipy.stats.lognorm-like GPU distribution object."""

    @staticmethod
    def cdf(x, s, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        s_f = float(s)
        scale_f = float(scale)
        if s_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        z = cp.log(cp.maximum(y, 1e-300)) / s_f
        return cp.where(y <= 0.0, 0.0, NormDistributionGPU._cdf_standard(z))

    @staticmethod
    def sf(x, s, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        s_f = float(s)
        scale_f = float(scale)
        if s_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        z = cp.log(cp.maximum(y, 1e-300)) / s_f
        return cp.where(y <= 0.0, 1.0, NormDistributionGPU._sf_standard(z))

    @staticmethod
    def ppf(q, s, *, loc=0.0, scale=1.0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        s_f = float(s)
        scale_f = float(scale)
        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if s_f <= 0.0 or scale_f <= 0.0:
            return out
        out = cp.where(q_gpu == 0.0, float(loc), out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        quant = float(loc) + scale_f * cp.exp(s_f * NormDistributionGPU._ppf_standard(q_gpu))
        return cp.where(valid, quant, out)

    @staticmethod
    def isf(q, s, *, loc=0.0, scale=1.0):
        import cupy as cp

        return LognormDistributionGPU.ppf(1.0 - cp.asarray(q, dtype=cp.float64), s, loc=loc, scale=scale)

    @staticmethod
    def pdf(x, s, *, loc=0.0, scale=1.0):
        import cupy as cp

        x_gpu = cp.asarray(x, dtype=cp.float64)
        s_f = float(s)
        scale_f = float(scale)
        if s_f <= 0.0 or scale_f <= 0.0:
            return cp.full_like(x_gpu, cp.nan, dtype=cp.float64)
        y = (x_gpu - float(loc)) / scale_f
        y_pos = cp.maximum(y, 1e-300)
        z = cp.log(y_pos) / s_f
        logpdf = -0.5 * cp.square(z) - cp.log(y_pos * s_f * cp.sqrt(cp.array(2.0 * cp.pi, dtype=cp.float64))) - cp.log(scale_f)
        return cp.where(y > 0.0, cp.exp(logpdf), 0.0)

    @staticmethod
    def rvs(s, *, size=None, loc=0.0, scale=1.0, dtype=None):
        import cupy as cp

        s_f = float(s)
        scale_f = float(scale)
        if s_f <= 0.0 or scale_f <= 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = float(loc) + scale_f * cp.exp(s_f * cp.random.normal(size=size))
        if dtype is not None:
            out = out.astype(dtype)
        return out


class PoissonDistributionGPU:
    """scipy.stats.poisson-like GPU distribution object."""

    @staticmethod
    def _ppf_gpu_search(q, mu):
        """Vectorized Poisson quantile search on GPU for scalar ``mu``."""
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        mu_f = float(mu)

        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if mu_f < 0.0:
            return out

        out = cp.where(q_gpu == 0.0, -1.0, out)
        out = cp.where(q_gpu == 1.0, cp.inf, out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        if not bool(cp.any(valid).item()):
            return out

        # A conservative upper bracket that works well for small/medium mu.
        hi0 = float(max(1.0, np.ceil(mu_f + 10.0 * np.sqrt(mu_f + 1.0) + 10.0)))
        low = cp.full_like(q_gpu, -1.0, dtype=cp.float64)
        high = cp.full_like(q_gpu, hi0, dtype=cp.float64)

        # Expand only unresolved entries until CDF(high) >= q.
        for _ in range(16):
            cdf_high = cp.where(high < 0.0, 0.0, gammaincc_gpu(high + 1.0, mu_f))
            need_expand = valid & (cdf_high < q_gpu)
            high = cp.where(need_expand, cp.maximum(high * 2.0 + 1.0, 1.0), high)

        max_high = float(cp.asnumpy(cp.max(cp.where(valid, high, 0.0))))
        steps = int(np.ceil(np.log2(max(max_high + 2.0, 2.0)))) + 2

        for _ in range(max(1, steps)):
            mid = cp.floor((low + high) / 2.0)
            cdf_mid = cp.where(mid < 0.0, 0.0, gammaincc_gpu(mid + 1.0, mu_f))
            move_right = valid & (cdf_mid < q_gpu)
            low = cp.where(move_right, mid, low)
            high = cp.where(valid & (~move_right), mid, high)

        k = cp.floor(high)
        cdf_k = cp.where(k < 0.0, 0.0, gammaincc_gpu(k + 1.0, mu_f))
        k = cp.where(valid & (cdf_k < q_gpu), k + 1.0, k)

        # One-step correction for round-off around discontinuities.
        km1 = k - 1.0
        cdf_km1 = cp.where(km1 < 0.0, 0.0, gammaincc_gpu(k, mu_f))
        k = cp.where(valid & (km1 >= -1.0) & (cdf_km1 >= q_gpu), km1, k)

        return cp.where(valid, k, out)

    @staticmethod
    def pmf(k, mu, *, loc=0):
        import cupy as cp

        k_gpu = cp.asarray(k, dtype=cp.float64) - float(loc)
        mu_f = float(mu)
        if mu_f < 0.0:
            return cp.full_like(k_gpu, cp.nan, dtype=cp.float64)
        k_floor = cp.floor(k_gpu)
        is_int = (k_floor == k_gpu)
        valid = (k_gpu >= 0.0) & is_int
        k_safe = cp.maximum(k_floor, 0.0)
        logpmf = k_safe * cp.log(cp.maximum(mu_f, 1e-300)) - mu_f - gammaln_gpu(k_safe + 1.0)
        return cp.where(valid, cp.exp(logpmf), 0.0)

    @staticmethod
    def cdf(k, mu, *, loc=0):
        import cupy as cp

        k_gpu = cp.asarray(k, dtype=cp.float64) - float(loc)
        mu_f = float(mu)
        if mu_f < 0.0:
            return cp.full_like(k_gpu, cp.nan, dtype=cp.float64)
        k_floor = cp.floor(k_gpu)
        return cp.where(k_floor < 0.0, 0.0, gammaincc_gpu(k_floor + 1.0, mu_f))

    @staticmethod
    def sf(k, mu, *, loc=0):
        import cupy as cp

        return cp.clip(1.0 - PoissonDistributionGPU.cdf(k, mu, loc=loc), 0.0, 1.0)

    @staticmethod
    def ppf(q, mu, *, loc=0):
        import cupy as cp

        loc_f = float(loc)
        q_gpu = cp.asarray(q, dtype=cp.float64)
        base = PoissonDistributionGPU._ppf_gpu_search(q_gpu, mu)
        return base + loc_f

    @staticmethod
    def isf(q, mu, *, loc=0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        return PoissonDistributionGPU.ppf(1.0 - q_gpu, mu, loc=loc)

    @staticmethod
    def rvs(mu, *, size=None, loc=0, dtype=None):
        import cupy as cp

        mu_f = float(mu)
        if mu_f < 0.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = cp.random.poisson(lam=mu_f, size=size) + int(loc)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class BinomDistributionGPU:
    """scipy.stats.binom-like GPU distribution object."""

    @staticmethod
    def _ppf_gpu_search(q, n, p):
        """Vectorized Binomial quantile search on GPU."""
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        n_i = int(n)
        p_f = float(p)

        out = cp.full_like(q_gpu, cp.nan, dtype=cp.float64)
        if n_i < 0 or p_f < 0.0 or p_f > 1.0:
            return out

        out = cp.where(q_gpu == 0.0, -1.0, out)
        out = cp.where(q_gpu == 1.0, float(n_i), out)
        valid = (q_gpu > 0.0) & (q_gpu < 1.0)
        if not bool(cp.any(valid).item()):
            return out

        low = cp.full_like(q_gpu, -1.0, dtype=cp.float64)
        high = cp.full_like(q_gpu, float(n_i), dtype=cp.float64)
        steps = int(np.ceil(np.log2(max(n_i + 2, 2)))) + 2

        for _ in range(max(1, steps)):
            mid = cp.floor((low + high) / 2.0)
            cdf_mid = BinomDistributionGPU.cdf(mid, n_i, p_f, loc=0.0)
            move_right = valid & (cdf_mid < q_gpu)
            low = cp.where(move_right, mid, low)
            high = cp.where(valid & (~move_right), mid, high)

        k = cp.floor(high)
        cdf_k = BinomDistributionGPU.cdf(k, n_i, p_f, loc=0.0)
        k = cp.where(valid & (cdf_k < q_gpu), k + 1.0, k)

        km1 = k - 1.0
        cdf_km1 = BinomDistributionGPU.cdf(km1, n_i, p_f, loc=0.0)
        k = cp.where(valid & (km1 >= -1.0) & (cdf_km1 >= q_gpu), km1, k)

        return cp.where(valid, k, out)

    @staticmethod
    def pmf(k, n, p, *, loc=0):
        import cupy as cp

        n_i = int(n)
        p_f = float(p)
        k_gpu = cp.asarray(k, dtype=cp.float64) - float(loc)
        if n_i < 0 or p_f < 0.0 or p_f > 1.0:
            return cp.full_like(k_gpu, cp.nan, dtype=cp.float64)
        k_floor = cp.floor(k_gpu)
        is_int = (k_floor == k_gpu)
        valid = (k_floor >= 0.0) & (k_floor <= n_i) & is_int
        k_safe = cp.clip(k_floor, 0.0, float(n_i))
        logcoef = gammaln_gpu(n_i + 1.0) - gammaln_gpu(k_safe + 1.0) - gammaln_gpu(n_i - k_safe + 1.0)
        logpmf = logcoef + k_safe * cp.log(cp.maximum(p_f, 1e-300)) + (n_i - k_safe) * cp.log(cp.maximum(1.0 - p_f, 1e-300))
        return cp.where(valid, cp.exp(logpmf), 0.0)

    @staticmethod
    def cdf(k, n, p, *, loc=0):
        import cupy as cp

        n_i = int(n)
        p_f = float(p)
        k_gpu = cp.asarray(k, dtype=cp.float64) - float(loc)
        if n_i < 0 or p_f < 0.0 or p_f > 1.0:
            return cp.full_like(k_gpu, cp.nan, dtype=cp.float64)
        k_floor = cp.floor(k_gpu)
        out = cp.where(k_floor < 0.0, 0.0, regularized_betainc_gpu(n_i - k_floor, k_floor + 1.0, 1.0 - p_f))
        return cp.where(k_floor >= n_i, 1.0, out)

    @staticmethod
    def sf(k, n, p, *, loc=0):
        import cupy as cp

        return cp.clip(1.0 - BinomDistributionGPU.cdf(k, n, p, loc=loc), 0.0, 1.0)

    @staticmethod
    def ppf(q, n, p, *, loc=0):
        import cupy as cp

        loc_f = float(loc)
        q_gpu = cp.asarray(q, dtype=cp.float64)
        base = BinomDistributionGPU._ppf_gpu_search(q_gpu, n, p)
        return base + loc_f

    @staticmethod
    def isf(q, n, p, *, loc=0):
        import cupy as cp

        q_gpu = cp.asarray(q, dtype=cp.float64)
        return BinomDistributionGPU.ppf(1.0 - q_gpu, n, p, loc=loc)

    @staticmethod
    def rvs(n, p, *, size=None, loc=0, dtype=None):
        import cupy as cp

        n_i = int(n)
        p_f = float(p)
        if n_i < 0 or p_f < 0.0 or p_f > 1.0:
            if size is None:
                return cp.asarray(cp.nan, dtype=cp.float64)
            return cp.full(size, cp.nan, dtype=cp.float64)
        out = cp.random.binomial(n_i, p_f, size=size) + int(loc)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class ScipyFallbackDistributionGPU:
    """Dynamic scipy.stats distribution wrapper returning CuPy outputs.

    Notes
    -----
    This wrapper guarantees a unified GPU-friendly API surface. For long-tail
    distributions not yet natively implemented in this module, calls are
    delegated to SciPy and results are copied back to GPU.
    """

    def __init__(self, name: str):
        self.name = str(name)

    def __repr__(self):
        return f"ScipyFallbackDistributionGPU(name='{self.name}')"

    def cdf(self, x, *shape_args, **kwargs):
        return scipy_dist_call_gpu(self.name, "cdf", x, *shape_args, **kwargs)

    def sf(self, x, *shape_args, **kwargs):
        return scipy_dist_call_gpu(self.name, "sf", x, *shape_args, **kwargs)

    def ppf(self, q, *shape_args, **kwargs):
        return scipy_dist_call_gpu(self.name, "ppf", q, *shape_args, **kwargs)

    def isf(self, q, *shape_args, **kwargs):
        return scipy_dist_call_gpu(self.name, "isf", q, *shape_args, **kwargs)

    def pdf(self, x, *shape_args, **kwargs):
        return scipy_dist_call_gpu(self.name, "pdf", x, *shape_args, **kwargs)

    def pmf(self, x, *shape_args, **kwargs):
        return scipy_dist_call_gpu(self.name, "pmf", x, *shape_args, **kwargs)

    def rvs(self, *shape_args, size=None, dtype=None, **kwargs):
        out = scipy_dist_call_gpu(self.name, "rvs", *shape_args, size=size, **kwargs)
        if dtype is not None:
            return out.astype(dtype)
        return out


# scipy.stats-like distribution objects.
norm = NormDistributionGPU()
t = TDistributionGPU()
uniform = UniformDistributionGPU()
expon = ExponDistributionGPU()
cauchy = CauchyDistributionGPU()
laplace = LaplaceDistributionGPU()
logistic = LogisticDistributionGPU()
chi2 = Chi2DistributionGPU()
gamma = GammaDistributionGPU()
beta = BetaDistributionGPU()
f = FDistributionGPU()
weibull_min = WeibullMinDistributionGPU()
lognorm = LognormDistributionGPU()
poisson = PoissonDistributionGPU()
binom = BinomDistributionGPU()


_NATIVE_DISTRIBUTIONS = {
    "norm": norm,
    "t": t,
    "uniform": uniform,
    "expon": expon,
    "cauchy": cauchy,
    "laplace": laplace,
    "logistic": logistic,
    "chi2": chi2,
    "gamma": gamma,
    "beta": beta,
    "f": f,
    "weibull_min": weibull_min,
    "lognorm": lognorm,
    "poisson": poisson,
    "binom": binom,
}


@lru_cache(maxsize=256)
def _fallback_distribution_gpu(name: str):
    return ScipyFallbackDistributionGPU(name)


def get_distribution_gpu(name: str, *, allow_fallback: bool = False):
    """Get a GPU distribution object by scipy.stats distribution name.

    Native GPU implementations are returned when available.

    Parameters
    ----------
    name : str
        Distribution name following scipy.stats naming.
    allow_fallback : bool, default=False
        If True, allow explicit SciPy-backed fallback for unsupported names.
        If False, raise for non-native distributions.
    """
    import scipy.stats as sps

    key = str(name).strip()
    if key in _NATIVE_DISTRIBUTIONS:
        return _NATIVE_DISTRIBUTIONS[key]
    key_lower = key.lower()
    if key_lower in _NATIVE_DISTRIBUTIONS:
        return _NATIVE_DISTRIBUTIONS[key_lower]

    if allow_fallback and hasattr(sps, key_lower):
        return _fallback_distribution_gpu(key_lower)
    if allow_fallback and hasattr(sps, key):
        return _fallback_distribution_gpu(key)

    if hasattr(sps, key_lower) or hasattr(sps, key):
        raise ValueError(
            f"Distribution '{name}' has no native GPU implementation. "
            "Set allow_fallback=True for explicit SciPy fallback."
        )
    raise ValueError(f"Unknown scipy.stats distribution: {name}")


def list_available_distributions_gpu(include_scipy: bool = True):
    """List available distribution names.

    Parameters
    ----------
    include_scipy : bool, default=True
        If True, include all scipy.stats distribution names (native + fallback).
        If False, include only native GPU distributions from this module.
    """
    native = sorted(_NATIVE_DISTRIBUTIONS.keys())
    if not include_scipy:
        return native

    import scipy.stats as sps
    from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete

    scipy_names = []
    for n in dir(sps):
        if n.startswith("_"):
            continue
        try:
            obj = getattr(sps, n)
        except Exception:
            continue
        if isinstance(obj, (rv_continuous, rv_discrete)):
            scipy_names.append(n)

    return sorted(set(native + scipy_names))


_LEGACY_DISTRIBUTION_FUNCTION_NAMES = {
    "t_cdf_gpu",
    "t_sf_gpu",
    "t_ppf_gpu",
    "t_two_sided_pvalue_gpu",
    "t_two_sided_critical_value_gpu",
    "norm_cdf_gpu",
    "norm_sf_gpu",
    "norm_ppf_gpu",
    "norm_isf_gpu",
    "norm_two_sided_pvalue_gpu",
    "norm_two_sided_critical_value_gpu",
    "rnorm_gpu",
    "dnorm_gpu",
    "dt_gpu",
    "rt_gpu",
    "pnorm_gpu",
    "qnorm_gpu",
    "pt_gpu",
    "qt_gpu",
    "dchisq_gpu",
    "pchisq_gpu",
    "qchisq_gpu",
    "rchisq_gpu",
    "dgamma_gpu",
    "pgamma_gpu",
    "qgamma_gpu",
    "rgamma_gpu",
    "dbeta_gpu",
    "pbeta_gpu",
    "qbeta_gpu",
    "rbeta_gpu",
    "df_gpu",
    "pf_gpu",
    "qf_gpu",
    "rf_gpu",
    "dpois_gpu",
    "ppois_gpu",
    "qpois_gpu",
    "rpois_gpu",
    "dbinom_gpu",
    "pbinom_gpu",
    "qbinom_gpu",
    "rbinom_gpu",
}


def __getattr__(name):
    """Lazy access to scipy.stats-compatible distribution objects."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _LEGACY_DISTRIBUTION_FUNCTION_NAMES:
        from . import _distributions_legacy_gpu as legacy

        return getattr(legacy, name)

    try:
        return get_distribution_gpu(name)
    except Exception as exc:
        raise AttributeError(f"module {__name__} has no attribute {name}") from exc


__all__ = [
    "regularized_betainc_gpu",
    "regularized_betaincinv_gpu",
    "NormDistributionGPU",
    "TDistributionGPU",
    "UniformDistributionGPU",
    "ExponDistributionGPU",
    "CauchyDistributionGPU",
    "LaplaceDistributionGPU",
    "LogisticDistributionGPU",
    "Chi2DistributionGPU",
    "GammaDistributionGPU",
    "BetaDistributionGPU",
    "FDistributionGPU",
    "WeibullMinDistributionGPU",
    "LognormDistributionGPU",
    "PoissonDistributionGPU",
    "BinomDistributionGPU",
    "ScipyFallbackDistributionGPU",
    "get_distribution_gpu",
    "list_available_distributions_gpu",
    "norm",
    "t",
    "uniform",
    "expon",
    "cauchy",
    "laplace",
    "logistic",
    "chi2",
    "gamma",
    "beta",
    "f",
    "weibull_min",
    "lognorm",
    "poisson",
    "binom",
    "t_cdf_gpu",
    "t_sf_gpu",
    "t_ppf_gpu",
    "t_two_sided_pvalue_gpu",
    "t_two_sided_critical_value_gpu",
    "norm_cdf_gpu",
    "norm_sf_gpu",
    "norm_ppf_gpu",
    "norm_isf_gpu",
    "norm_two_sided_pvalue_gpu",
    "norm_two_sided_critical_value_gpu",
    "rnorm_gpu",
    "dnorm_gpu",
    "dt_gpu",
    "rt_gpu",
    "pnorm_gpu",
    "qnorm_gpu",
    "pt_gpu",
    "qt_gpu",
    "dchisq_gpu",
    "pchisq_gpu",
    "qchisq_gpu",
    "rchisq_gpu",
    "dgamma_gpu",
    "pgamma_gpu",
    "qgamma_gpu",
    "rgamma_gpu",
    "dbeta_gpu",
    "pbeta_gpu",
    "qbeta_gpu",
    "rbeta_gpu",
    "df_gpu",
    "pf_gpu",
    "qf_gpu",
    "rf_gpu",
    "dpois_gpu",
    "ppois_gpu",
    "qpois_gpu",
    "rpois_gpu",
    "dbinom_gpu",
    "pbinom_gpu",
    "qbinom_gpu",
    "rbinom_gpu",
]
