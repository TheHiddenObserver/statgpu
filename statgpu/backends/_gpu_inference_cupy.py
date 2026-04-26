"""
GPU utility functions for full GPU computation.
All statistical computations on GPU.
"""

import numpy as np

from statgpu.inference._distributions_backend import (
    norm,
    t,
    regularized_betainc_gpu,
)


def t_two_tail_pvalues_gpu(t_abs, df_resid):
    """Backward-compatible alias for two-sided t p-values on GPU."""
    return t.two_sided_pvalue(t_abs, df=df_resid)


def t_crit_gpu_two_tail(alpha, df_resid, *, max_bisect_steps: int = 60):
    """Backward-compatible alias for two-sided t critical value on GPU."""
    return t.two_sided_critical_value(
        alpha,
        df=df_resid,
        max_bisect_steps=max_bisect_steps,
    )


def norm_two_tail_pvalues_gpu(z_abs):
    """Backward-compatible alias for two-sided normal p-values on GPU."""
    return norm.two_sided_pvalue(z_abs)


def norm_crit_gpu_two_tail(alpha):
    """Backward-compatible alias for two-sided normal critical value on GPU."""
    return norm.two_sided_critical_value(alpha)


def compute_inference_gpu(X_design, resid, scale, df_resid, params_gpu):
    """
    Compute standard errors, t-values, p-values on GPU.
    
    Parameters
    ----------
    X_design : cupy.ndarray
        Design matrix on GPU.
    resid : cupy.ndarray
        Residuals on GPU.
    scale : float or cupy.ndarray
        Error variance estimate.
    df_resid : int
        Degrees of freedom.
    params_gpu : cupy.ndarray
        Parameters on GPU.
    
    Returns
    -------
    bse_gpu : cupy.ndarray
        Standard errors on GPU.
    tvalues_gpu : cupy.ndarray
        t-statistics on GPU.
    pvalues_gpu : cupy.ndarray
        p-values on GPU.
    conf_int_gpu : cupy.ndarray
        Confidence intervals on GPU.
    """
    import cupy as cp
    
    # Compute (X'X)^-1 on GPU
    XtX = cp.matmul(X_design.T, X_design)
    
    try:
        # Use Cholesky for inversion
        L = cp.linalg.cholesky(XtX)
        XtX_inv = cp.linalg.inv(XtX)  # Simpler but less stable
    except Exception:
        # Fallback to pseudo-inverse
        XtX_inv = cp.linalg.pinv(XtX)
    
    # Standard errors: sqrt(scale * diag((X'X)^-1))
    bse_gpu = cp.sqrt(scale * cp.diag(XtX_inv))
    
    # t-statistics
    tvalues_gpu = params_gpu / bse_gpu
    
    # p-values (two-tailed t-test), entirely on GPU.
    pvalues_gpu = t.two_sided_pvalue(tvalues_gpu, df=df_resid)
    
    # Confidence intervals (95%)
    alpha = 0.05  # two-tailed significance level for 95% CI
    t_crit_gpu = t.two_sided_critical_value(alpha, df=df_resid)
    
    margin = t_crit_gpu * bse_gpu
    conf_int_lower = params_gpu - margin
    conf_int_upper = params_gpu + margin
    conf_int_gpu = cp.stack([conf_int_lower, conf_int_upper], axis=1)
    
    return bse_gpu, tvalues_gpu, pvalues_gpu, conf_int_gpu


def compute_r2_gpu(y, resid):
    """
    Compute R-squared on GPU.
    
    Parameters
    ----------
    y : cupy.ndarray
        True values on GPU.
    resid : cupy.ndarray
        Residuals on GPU.
    
    Returns
    -------
    r2 : float
        R-squared value.
    """
    import cupy as cp
    
    y_mean = y.mean()
    ss_res = cp.sum(resid ** 2)
    ss_tot = cp.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    return float(cp.asnumpy(r2))


def compute_aic_bic_gpu(n, k, scale):
    """
    Compute AIC/BIC on GPU.
    
    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of parameters.
    scale : float or cupy.ndarray
        Error variance (MLE estimate: RSS/n).
    
    Returns
    -------
    aic : float
        AIC value.
    bic : float
        BIC value.
    """
    import cupy as cp
    
    # Convert to cupy if needed
    if not hasattr(scale, 'get'):
        scale = cp.array(scale)
    
    # AIC = n * log(scale) + 2*k
    # BIC = n * log(scale) + k * log(n)
    n_gpu = cp.array(float(n))
    k_gpu = cp.array(float(k))
    
    aic_gpu = n_gpu * cp.log(scale) + 2 * k_gpu
    bic_gpu = n_gpu * cp.log(scale) + k_gpu * cp.log(n_gpu)
    
    return float(cp.asnumpy(aic_gpu)), float(cp.asnumpy(bic_gpu))


def compute_f_stat_gpu(y, resid, X_design, df_resid):
    """
    Compute F-statistic on GPU.
    
    Parameters
    ----------
    y : cupy.ndarray
        True values on GPU.
    resid : cupy.ndarray
        Residuals on GPU.
    X_design : cupy.ndarray
        Design matrix on GPU.
    df_resid : int
        Residual degrees of freedom.
    
    Returns
    -------
    fvalue : float
        F-statistic.
    """
    import cupy as cp
    
    y_mean = y.mean()
    ss_tot = cp.sum((y - y_mean) ** 2)
    ss_res = cp.sum(resid ** 2)
    ss_reg = ss_tot - ss_res
    
    k = X_design.shape[1] - 1  # exclude intercept
    if k == 0 or ss_res <= 0:
        return np.inf
    
    fvalue_gpu = (ss_reg / k) / (ss_res / df_resid)
    fvalue = float(cp.asnumpy(fvalue_gpu))
    
    # p-value on GPU using F CDF expressed via regularized incomplete beta.
    #
    # For F ~ F(d1, d2):
    #   CDF(x) = I_{ d1 x / (d1 x + d2) }(d1/2, d2/2)
    #   pvalue = 1 - CDF
    d1 = float(k)
    d2 = float(df_resid)
    if d2 <= 0 or d1 <= 0:
        pvalue = 1.0
    else:
        z = (d1 * fvalue) / (d1 * fvalue + d2)
        cdf = regularized_betainc_gpu(d1 / 2.0, d2 / 2.0, cp.asarray(z))
        pvalue = float(1.0 - cp.asnumpy(cdf))
    
    return fvalue, pvalue
