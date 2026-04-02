"""
GPU utility functions for full GPU computation.
All statistical computations on GPU.
"""

import numpy as np


def _regularized_betainc_gpu(a, b, x):
    """
    Regularized incomplete beta I_x(a, b) evaluated on GPU.

    Tries to use `cupyx.scipy.special.betainc`; falls back to CPU SciPy if
    unavailable (this fallback may synchronize and reduce performance).
    """
    import cupy as cp

    try:
        import cupyx.scipy.special as csp

        return csp.betainc(a, b, x)
    except Exception:
        # Fallback: compute on CPU then move back.
        import scipy.special as sps

        x_cpu = cp.asnumpy(x)
        return cp.asarray(sps.betainc(a, b, x_cpu))


def _regularized_betaincinv_gpu(a, b, y):
    """
    Inverse of regularized incomplete beta:
    find x such that I_x(a,b) = y.
    """
    import cupy as cp

    try:
        import cupyx.scipy.special as csp

        return csp.betaincinv(a, b, y)
    except Exception:
        # Caller can implement a numeric inversion; this exists mainly
        # to centralize betaincinv availability.
        raise RuntimeError("cupyx.scipy.special.betaincinv is not available")


def t_two_tail_pvalues_gpu(t_abs, df_resid):
    """
    Two-tailed p-values for Student's t distribution on GPU.

    Uses the identity:
      p = P(|T| >= t) = I_{ df / (df + t^2) }(df/2, 1/2)
    where I is the regularized incomplete beta.
    """
    import cupy as cp

    df = float(df_resid)
    if df <= 0:
        return cp.full_like(t_abs, cp.nan, dtype=cp.float64)

    x = df / (df + cp.square(t_abs))
    a = df / 2.0
    b = 0.5
    return _regularized_betainc_gpu(a, b, x)


def t_crit_gpu_two_tail(alpha, df_resid, *, max_bisect_steps: int = 60):
    """
    Compute positive t critical value for two-tailed test:
      p_two_tail(t_crit) = alpha
    """
    import cupy as cp

    df = float(df_resid)
    if df <= 0:
        return cp.array(cp.nan, dtype=cp.float64)

    a = df / 2.0
    b = 0.5

    # Preferred: inverse regularized incomplete beta -> closed-form for t.
    try:
        # Find y = I^{-1}(a,b,alpha) where y = df/(df+t^2).
        y = _regularized_betaincinv_gpu(a, b, float(alpha))
        # t = sqrt(df*(1-y)/y)
        t = cp.sqrt(df * (1.0 - y) / y)
        return t
    except Exception:
        # Fallback: bisection on t using GPU betainc.
        low = 0.0
        high = 1.0

        # Increase high until p(high) <= alpha.
        p_high = float(t_two_tail_pvalues_gpu(cp.array(high, dtype=cp.float64), df_resid).get())
        it = 0
        while p_high > alpha and it < 50:
            high *= 2.0
            p_high = float(
                t_two_tail_pvalues_gpu(cp.array(high, dtype=cp.float64), df_resid).get()
            )
            it += 1

        for _ in range(max_bisect_steps):
            mid = 0.5 * (low + high)
            p_mid = float(t_two_tail_pvalues_gpu(cp.array(mid, dtype=cp.float64), df_resid).get())
            if p_mid > alpha:
                low = mid
            else:
                high = mid

        return cp.array(high, dtype=cp.float64)


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
    pvalues_gpu = t_two_tail_pvalues_gpu(cp.abs(tvalues_gpu), df_resid)
    
    # Confidence intervals (95%)
    alpha = 0.05  # two-tailed significance level for 95% CI
    t_crit_gpu = t_crit_gpu_two_tail(alpha, df_resid)
    
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
        cdf = _regularized_betainc_gpu(d1 / 2.0, d2 / 2.0, cp.asarray(z))
        pvalue = float(1.0 - cp.asnumpy(cdf))
    
    return fvalue, pvalue
