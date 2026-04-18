"""
Torch-native probability distribution utilities.

This module provides scipy.stats-like distribution objects implemented
using PyTorch operations, for use in GPU inference paths.

Module API mirrors _distributions_gpu.py but uses torch.* instead of cupy.*.
"""

import numpy as np
from functools import lru_cache

from ._distribution_utils_torch import (
    gammainc_torch,
    gammaincc_torch,
    gammaincinv_torch,
    gammaln_torch,
    regularized_betainc_torch,
    regularized_betaincinv_torch,
    erf_torch,
    erfc_torch,
    erfcinv_torch,
    to_numpy_for_scipy,
    scipy_dist_call_torch,
)

# Conservative finite bracket for t-quantile inversion fallback
_T_PPF_BISECT_LOWER = -64.0
_T_PPF_BISECT_UPPER = 64.0


def _import_torch():
    """Deferred torch import."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PyTorch (torch) is required for Torch backend") from exc


def _get_torch_device():
    """Get current Torch device."""
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class NormDistributionTorch:
    """
    scipy.stats.norm-like GPU distribution object using Torch.

    Examples
    --------
    >>> from statgpu.inference._distributions_torch import norm
    >>> norm.cdf(x)
    >>> norm.ppf(q)
    >>> norm.two_sided_pvalue(z_abs)
    """

    @staticmethod
    def _cdf_standard(x, device=None):
        """Standard normal CDF kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device))

        try:
            erf_val = erf_torch(x_tensor / sqrt2, device=device)
        except Exception as exc:
            raise RuntimeError("erf is required for Torch backend") from exc

        return 0.5 * (1.0 + erf_val)

    @staticmethod
    def _sf_standard(x, device=None):
        """Standard normal survival function kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        return torch.clamp(1.0 - NormDistributionTorch._cdf_standard(x, device), 0.0, 1.0)

    @staticmethod
    def _ppf_standard(q, device=None):
        """Standard normal quantile kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        q_tensor = torch.as_tensor(q, dtype=torch.float64, device=device)

        try:
            # Φ^{-1}(q) = -sqrt(2) * erfcinv(2q)
            return -torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device)) * erfcinv_torch(2.0 * q_tensor, device=device)
        except Exception as exc:
            raise RuntimeError("erfcinv is required for Torch backend") from exc

    @staticmethod
    def _isf_standard(q, device=None):
        """Standard normal inverse survival kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        q_tensor = torch.as_tensor(q, dtype=torch.float64, device=device)
        return NormDistributionTorch._ppf_standard(1.0 - q_tensor, device=device)

    @staticmethod
    def _two_sided_pvalue_standard(stat_abs, device=None):
        """Standard normal two-sided p-value kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        stat_tensor = torch.as_tensor(stat_abs, dtype=torch.float64, device=device)
        return torch.clamp(2.0 * NormDistributionTorch._sf_standard(torch.abs(stat_tensor), device), 0.0, 1.0)

    @staticmethod
    def _two_sided_critical_value_standard(alpha, device=None):
        """Positive two-sided critical value kernel for standard normal."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        a = float(alpha)
        if not (0.0 < a < 1.0):
            return torch.tensor(float('nan'), dtype=torch.float64, device=device)
        return NormDistributionTorch._ppf_standard(1.0 - a / 2.0, device=device)

    @staticmethod
    def cdf(x, *, loc=0.0, scale=1.0, device=None):
        """Normal CDF on Torch (supports location/scale)."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(x, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        x_std = (torch.as_tensor(x, dtype=torch.float64, device=device) - float(loc)) / scale_f
        return NormDistributionTorch._cdf_standard(x_std, device=device)

    @staticmethod
    def sf(x, *, loc=0.0, scale=1.0, device=None):
        """Normal survival function on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(x, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        x_std = (torch.as_tensor(x, dtype=torch.float64, device=device) - float(loc)) / scale_f
        return NormDistributionTorch._sf_standard(x_std, device=device)

    @staticmethod
    def ppf(q, *, loc=0.0, scale=1.0, device=None):
        """Normal quantile on Torch (supports location/scale)."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(q, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        z = NormDistributionTorch._ppf_standard(q, device=device)
        return float(loc) + scale_f * z

    @staticmethod
    def isf(q, *, loc=0.0, scale=1.0, device=None):
        """Normal inverse survival function on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(q, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        z = NormDistributionTorch._isf_standard(q, device=device)
        return float(loc) + scale_f * z

    @staticmethod
    def pdf(x, *, loc=0.0, scale=1.0, device=None):
        """Normal PDF on Torch (supports location/scale)."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)

        if scale_f <= 0:
            return torch.full_like(x_tensor, torch.nan, dtype=torch.float64)

        z = (x_tensor - float(loc)) / scale_f
        norm_const = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.float64, device=device))
        return torch.exp(-0.5 * torch.square(z)) / (scale_f * norm_const)

    @staticmethod
    def two_sided_pvalue(stat_abs, device=None):
        """Two-sided p-value for standard normal test statistics."""
        return NormDistributionTorch._two_sided_pvalue_standard(stat_abs, device=device)

    @staticmethod
    def two_sided_critical_value(alpha, device=None):
        """Positive two-sided critical value for standard normal."""
        return NormDistributionTorch._two_sided_critical_value_standard(alpha, device=device)

    @staticmethod
    def rvs(*, size=None, loc=0.0, scale=1.0, dtype=None, device=None):
        """Draw random variates from Normal on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        out = torch.randn(size, dtype=torch.float64, device=device) * float(scale) + float(loc)
        if dtype is not None:
            out = out.to(dtype)
        return out


class TDistributionTorch:
    """
    scipy.stats.t-like GPU distribution object using Torch.

    Examples
    --------
    >>> from statgpu.inference._distributions_torch import t
    >>> t.cdf(x, df=10)
    >>> t.ppf(q, df=10)
    >>> t.two_sided_pvalue(t_abs, df=10)
    """

    @staticmethod
    def _cdf_standard(x, df, device=None):
        """Standardized Student t CDF kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
        df_val = float(df)

        if df_val <= 0:
            return torch.full_like(x_tensor, torch.nan, dtype=torch.float64)

        z = df_val / (df_val + torch.square(torch.abs(x_tensor)))
        ibeta = regularized_betainc_torch(df_val / 2.0, 0.5, z, device=device)
        lower_tail = 0.5 * ibeta

        return torch.where(x_tensor >= 0.0, 1.0 - lower_tail, lower_tail)

    @staticmethod
    def _sf_standard(x, df, device=None):
        """Standardized Student t survival function kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        return torch.clamp(1.0 - TDistributionTorch._cdf_standard(x, df, device=device), 0.0, 1.0)

    @staticmethod
    def _two_sided_pvalue_standard(stat_abs, df, device=None):
        """Standardized two-sided Student t p-value kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        stat_tensor = torch.as_tensor(stat_abs, dtype=torch.float64, device=device)
        df_val = float(df)

        if df_val <= 0:
            return torch.full_like(stat_tensor, torch.nan, dtype=torch.float64)

        z = df_val / (df_val + torch.square(torch.abs(stat_tensor)))
        return regularized_betainc_torch(df_val / 2.0, 0.5, z, device=device)

    @staticmethod
    def _ppf_standard(q, df, *, max_bisect_steps=60, device=None):
        """Standardized Student t quantile kernel."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        q_tensor = torch.as_tensor(q, dtype=torch.float64, device=device)
        df_val = float(df)

        if df_val <= 0:
            return torch.full_like(q_tensor, torch.nan, dtype=torch.float64)

        out = torch.full_like(q_tensor, torch.nan, dtype=torch.float64)
        out = torch.where(q_tensor == 0.0, -torch.inf, out)
        out = torch.where(q_tensor == 1.0, torch.inf, out)

        valid = (q_tensor > 0.0) & (q_tensor < 1.0)

        if not bool(torch.any(valid).item()):
            return out

        try:
            # Use inverse beta approach
            tail = torch.minimum(q_tensor, 1.0 - q_tensor)
            y = 2.0 * tail
            y_inv = regularized_betaincinv_torch(df_val / 2.0, 0.5, y, device=device)
            x_pos = torch.sqrt(df_val * (1.0 - y_inv) / y_inv)
            quant = torch.where(q_tensor >= 0.5, x_pos, -x_pos)
            return torch.where(valid, quant, out)
        except Exception:
            # Monotone bisection fallback
            steps = max(int(max_bisect_steps), 1)
            lo = torch.full_like(q_tensor, _T_PPF_BISECT_LOWER, dtype=q_tensor.dtype)
            hi = torch.full_like(q_tensor, _T_PPF_BISECT_UPPER, dtype=q_tensor.dtype)

            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                cdf_mid = TDistributionTorch._cdf_standard(mid, df_val, device=device)
                go_right = cdf_mid < q_tensor
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)

            quant = 0.5 * (lo + hi)
            return torch.where(valid, quant, out)

    @staticmethod
    def cdf(x, df, *, loc=0.0, scale=1.0, device=None):
        """Student t CDF on Torch (supports location/scale)."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(x, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        x_std = (torch.as_tensor(x, dtype=torch.float64, device=device) - float(loc)) / scale_f
        return TDistributionTorch._cdf_standard(x_std, df, device=device)

    @staticmethod
    def sf(x, df, *, loc=0.0, scale=1.0, device=None):
        """Student t survival function on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(x, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        x_std = (torch.as_tensor(x, dtype=torch.float64, device=device) - float(loc)) / scale_f
        return TDistributionTorch._sf_standard(x_std, df, device=device)

    @staticmethod
    def ppf(q, df, *, loc=0.0, scale=1.0, max_bisect_steps=60, device=None):
        """Student t quantile on Torch (supports location/scale)."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(q, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        z = TDistributionTorch._ppf_standard(q, df, max_bisect_steps=max_bisect_steps, device=device)
        return float(loc) + scale_f * z

    @staticmethod
    def isf(q, df, *, loc=0.0, scale=1.0, max_bisect_steps=60, device=None):
        """Student t inverse survival function on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        scale_f = float(scale)
        if scale_f <= 0:
            return torch.full_like(torch.as_tensor(q, dtype=torch.float64, device=device), torch.nan, dtype=torch.float64)

        torch_lib = _import_torch()
        q_tensor = torch.as_tensor(q, dtype=torch.float64, device=device)
        z = TDistributionTorch._ppf_standard(1.0 - q_tensor, df, max_bisect_steps=max_bisect_steps, device=device)
        return float(loc) + scale_f * z

    @staticmethod
    def pdf(x, df, *, loc=0.0, scale=1.0, device=None):
        """Student t PDF on Torch (supports location/scale)."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
        df_val = float(df)
        scale_f = float(scale)

        if df_val <= 0.0 or scale_f <= 0.0:
            return torch.full_like(x_tensor, torch.nan, dtype=torch.float64)

        z = (x_tensor - float(loc)) / scale_f
        half_nu = df_val / 2.0

        log_coef = (
            gammaln_torch((df_val + 1.0) / 2.0, device=device)
            - gammaln_torch(half_nu, device=device)
            - 0.5 * (torch.log(torch.as_tensor(df_val, dtype=torch.float64, device=device)) + torch.log(torch.as_tensor(torch.pi, dtype=torch.float64, device=device)))
        )

        log_pdf = log_coef - ((df_val + 1.0) / 2.0) * torch.log1p(torch.square(z) / df_val) - torch.log(torch.as_tensor(scale_f, dtype=torch.float64, device=device))
        return torch.exp(log_pdf)

    @staticmethod
    def two_sided_pvalue(stat_abs, df, device=None):
        """Two-sided p-value for Student t statistics."""
        return TDistributionTorch._two_sided_pvalue_standard(stat_abs, df, device=device)

    @staticmethod
    def two_sided_critical_value(alpha, df, *, max_bisect_steps=60, device=None):
        """Positive two-sided critical value for Student t."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        a = float(alpha)
        if not (0.0 < a < 1.0):
            return torch.tensor(float('nan'), dtype=torch.float64, device=device)

        return TDistributionTorch._ppf_standard(1.0 - a / 2.0, df, max_bisect_steps=max_bisect_steps, device=device)

    @staticmethod
    def rvs(df, *, size=None, loc=0.0, scale=1.0, dtype=None, device=None):
        """Draw random variates from Student t on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        # Use inverse transform sampling or accept-reject
        # For simplicity, use SciPy fallback if needed
        try:
            # Torch doesn't have native Student-t random generator
            # Use standard normal / sqrt(chi2/df) ratio
            z = torch.randn(size, dtype=torch.float64, device=device)
            chi2 = torch.square(torch.randn(size, dtype=torch.float64, device=device))
            t_var = z / torch.sqrt(chi2 / float(df))
            base = t_var * float(scale) + float(loc)
        except Exception:
            # Fallback to SciPy
            base = scipy_dist_call_torch('t', 'rvs', df, size=size, loc=loc, scale=scale)

        if dtype is not None:
            base = base.to(dtype)
        return base


class FDistributionTorch:
    """
    scipy.stats.f-like GPU distribution object using Torch.

    Examples
    --------
    >>> from statgpu.inference._distributions_torch import f
    >>> f.cdf(x, dfn=5, dfd=100)
    >>> f.sf(x, dfn=5, dfd=100)
    """

    @staticmethod
    def cdf(x, dfn, dfd, device=None):
        """F-distribution CDF on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
        dfn_f = float(dfn)
        dfd_f = float(dfd)

        if dfn_f <= 0.0 or dfd_f <= 0.0:
            return torch.full_like(x_tensor, torch.nan, dtype=torch.float64)

        z = (dfn_f * torch.maximum(x_tensor, torch.tensor(0.0, dtype=torch.float64, device=device))) / (dfn_f * torch.maximum(x_tensor, torch.tensor(0.0, dtype=torch.float64, device=device)) + dfd_f)
        core = regularized_betainc_torch(dfn_f / 2.0, dfd_f / 2.0, z, device=device)
        return torch.where(x_tensor <= 0.0, torch.tensor(0.0, dtype=torch.float64, device=device), core)

    @staticmethod
    def sf(x, dfn, dfd, device=None):
        """F-distribution survival function on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        return torch.clamp(1.0 - FDistributionTorch.cdf(x, dfn, dfd, device=device), 0.0, 1.0)

    @staticmethod
    def ppf(q, dfn, dfd, device=None):
        """F-distribution quantile function on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        q_tensor = torch.as_tensor(q, dtype=torch.float64, device=device)
        dfn_f = float(dfn)
        dfd_f = float(dfd)

        out = torch.full_like(q_tensor, torch.nan, dtype=torch.float64)

        if dfn_f <= 0.0 or dfd_f <= 0.0:
            return out

        out = torch.where(q_tensor == 0.0, torch.tensor(0.0, dtype=torch.float64, device=device), out)
        out = torch.where(q_tensor == 1.0, torch.tensor(float('inf'), dtype=torch.float64, device=device), out)

        valid = (q_tensor > 0.0) & (q_tensor < 1.0)
        z = regularized_betaincinv_torch(dfn_f / 2.0, dfd_f / 2.0, q_tensor, device=device)
        x = (dfd_f * z) / (dfn_f * (1.0 - z))
        return torch.where(valid, x, out)

    @staticmethod
    def isf(q, dfn, dfd, device=None):
        """F-distribution inverse survival function on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        return FDistributionTorch.ppf(1.0 - torch.as_tensor(q, dtype=torch.float64, device=device), dfn, dfd, device=device)

    @staticmethod
    def pdf(x, dfn, dfd, device=None):
        """F-distribution PDF on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
        dfn_f = float(dfn)
        dfd_f = float(dfd)

        if dfn_f <= 0.0 or dfd_f <= 0.0:
            return torch.full_like(x_tensor, torch.nan, dtype=torch.float64)

        # F PDF: f(x) = (dfn/dfd)^(dfn/2) * x^(dfn/2 - 1) * (1 + dfn*x/dfd)^(-(dfn+dfd)/2) / B(dfn/2, dfd/2)
        log_coef = (
            (dfn_f / 2.0) * torch.log(torch.tensor(dfn_f / dfd_f, dtype=torch.float64, device=device))
            + gammaln_torch((dfn_f + dfd_f) / 2.0, device=device)
            - gammaln_torch(dfn_f / 2.0, device=device)
            - gammaln_torch(dfd_f / 2.0, device=device)
        )

        log_pdf = (
            log_coef
            + (dfn_f / 2.0 - 1.0) * torch.log(torch.clamp(x_tensor, 1e-30, float('inf')))
            - ((dfn_f + dfd_f) / 2.0) * torch.log1p((dfn_f / dfd_f) * x_tensor)
        )

        return torch.exp(log_pdf)

    @staticmethod
    def rvs(dfn, dfd, *, size=None, dtype=None, device=None):
        """Draw random variates from F-distribution on Torch."""
        torch = _import_torch()

        if device is None:
            device = _get_torch_device()

        # Use ratio of chi-squared variables
        # F ~ (chi2_dfn / dfn) / (chi2_dfd / dfd)
        chi2_dfn = torch.square(torch.randn(size, dtype=torch.float64, device=device))
        for _ in range(int(dfn) - 1):
            chi2_dfn += torch.square(torch.randn(size, dtype=torch.float64, device=device))

        chi2_dfd = torch.square(torch.randn(size, dtype=torch.float64, device=device))
        for _ in range(int(dfd) - 1):
            chi2_dfd += torch.square(torch.randn(size, dtype=torch.float64, device=device))

        f_var = (chi2_dfn / float(dfn)) / (chi2_dfd / float(dfd))

        if dtype is not None:
            f_var = f_var.to(dtype)
        return f_var


# Create module-level singleton objects for API compatibility
norm = NormDistributionTorch()
t = TDistributionTorch()
f = FDistributionTorch()

# Exported symbols
__all__ = [
    "NormDistributionTorch",
    "TDistributionTorch",
    "FDistributionTorch",
    "norm",
    "t",
    "f",
    "regularized_betainc_torch",
    "regularized_betaincinv_torch",
    "gammainc_torch",
    "gammaincc_torch",
    "gammaincinv_torch",
    "gammaln_torch",
]
