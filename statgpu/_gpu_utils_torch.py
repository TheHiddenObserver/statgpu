"""
Torch-specific GPU utility functions for full GPU computation.

This module mirrors _gpu_utils.py but uses PyTorch operations instead of CuPy.
All statistical computations run on GPU via Torch.
"""

import numpy as np


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


def t_two_tail_pvalues_torch(t_abs, df_resid, device=None):
    """
    Backward-compatible alias for two-sided t p-values on Torch GPU.

    Parameters
    ----------
    t_abs : torch.Tensor or array-like
        Absolute t-statistics.
    df_resid : int or float
        Residual degrees of freedom.
    device : str, optional
        Torch device string.

    Returns
    -------
    torch.Tensor
        Two-sided p-values.
    """
    from statgpu.inference._distributions_backend import get_distribution
    t_dist = get_distribution("t", backend="torch", device=device)
    return t_dist.two_sided_pvalue(t_abs, df=df_resid)


def t_crit_torch_two_tail_torch(alpha, df_resid, *, max_bisect_steps=60, device=None):
    """
    Backward-compatible alias for two-sided t critical value on Torch GPU.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).
    df_resid : int or float
        Residual degrees of freedom.
    max_bisect_steps : int, default=60
        Maximum bisection iterations for quantile computation.
    device : str, optional
        Torch device string.

    Returns
    -------
    torch.Tensor
        Critical t-value.
    """
    from statgpu.inference._distributions_backend import get_distribution
    t_dist = get_distribution("t", backend="torch", device=device)
    return t_dist.two_sided_critical_value(alpha, df=df_resid, max_bisect_steps=max_bisect_steps)


def norm_two_tail_pvalues_torch(z_abs, device=None):
    """
    Backward-compatible alias for two-sided normal p-values on Torch GPU.

    Parameters
    ----------
    z_abs : torch.Tensor or array-like
        Absolute z-statistics.
    device : str, optional
        Torch device string.

    Returns
    -------
    torch.Tensor
        Two-sided p-values.
    """
    from statgpu.inference._distributions_backend import norm
    return norm.two_sided_pvalue(z_abs, backend="torch", device=device)


def norm_crit_torch_two_tail_torch(alpha, device=None):
    """
    Backward-compatible alias for two-sided normal critical value on Torch GPU.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).
    device : str, optional
        Torch device string.

    Returns
    -------
    torch.Tensor
        Critical z-value.
    """
    from statgpu.inference._distributions_backend import norm
    return norm.two_sided_critical_value(alpha, backend="torch")


def compute_inference_torch(X_design, resid, scale, df_resid, params_torch, cov_type="nonrobust", device=None):
    """
    Compute standard errors, t-values, p-values, and confidence intervals on Torch GPU.

    Parameters
    ----------
    X_design : torch.Tensor
        Design matrix on GPU.
    resid : torch.Tensor
        Residuals on GPU.
    scale : float or torch.Tensor
        Error variance estimate (sigma^2).
    df_resid : int
        Degrees of freedom.
    params_torch : torch.Tensor
        Parameters on GPU.
    cov_type : str, default='nonrobust'
        Covariance type: 'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'.
    device : str, optional
        Torch device string.

    Returns
    -------
    bse_torch : torch.Tensor
        Standard errors on GPU.
    tvalues_torch : torch.Tensor
        t-statistics on GPU.
    pvalues_torch : torch.Tensor
        p-values on GPU.
    conf_int_torch : torch.Tensor
        Confidence intervals on GPU.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    from statgpu.inference._distributions_backend import get_distribution
    t_dist = get_distribution("t", backend="torch", device=device)

    # Compute (X'X)^-1 on GPU
    XtX = torch.matmul(X_design.T, X_design)

    try:
        # Use Cholesky for inversion (more stable for positive definite)
        L = torch.linalg.cholesky(XtX)
        # Solve L @ L.T @ x = b for each column
        XtX_inv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        XtX_inv = torch.linalg.pinv(XtX)

    # Handle HC2/HC3 leverage adjustment
    if cov_type in ("hc2", "hc3"):
        # Compute leverage values: diag(X @ (X'X)^-1 @ X.T)
        # More efficient: diag(X @ (X'X)^-1 @ X.T) = sum((X @ (X'X)^-1) * X, axis=1)
        XtX_inv_half = torch.linalg.cholesky(XtX_inv)
        X_white = torch.matmul(X_design, XtX_inv_half)
        leverage = torch.sum(X_white * X_design, dim=1)
        leverage = torch.clamp(leverage, 0.0, 1.0 - 1e-12)

        if cov_type == "hc2":
            # HC2: e2 / (1 - h_ii)
            e2 = torch.square(resid) / (1.0 - leverage)
        else:
            # HC3: e2 / (1 - h_ii)^2
            e2 = torch.square(resid) / torch.square(1.0 - leverage)

        # Sandwich: (X'X)^-1 @ (X' @ diag(e2) @ X) @ (X'X)^-1
        Xw = X_design * e2[:, None]
        meat = torch.matmul(X_design.T, Xw)
        cov_params = torch.matmul(XtX_inv, torch.matmul(meat, XtX_inv))
        bse_torch = torch.sqrt(torch.clamp(torch.diag(cov_params), 0.0))
    elif cov_type == "hc1":
        # HC1: scale adjustment
        n, k = X_design.shape
        if n > k:
            scale_factor = n / (n - k)
        else:
            scale_factor = 1.0
        bse_torch = torch.sqrt(scale_factor * scale * torch.clamp(torch.diag(XtX_inv), 0.0))
    else:
        # Nonrobust (HC0-style): scale * diag((X'X)^-1)
        bse_torch = torch.sqrt(scale * torch.clamp(torch.diag(XtX_inv), 0.0))

    # t-statistics
    tvalues_torch = params_torch / (bse_torch + 1e-30)

    # p-values (two-tailed t-test), entirely on GPU
    pvalues_torch = t_dist.two_sided_pvalue(tvalues_torch, df=df_resid)

    # Confidence intervals (95%)
    alpha = 0.05  # two-tailed significance level for 95% CI
    t_crit = t_dist.two_sided_critical_value(alpha, df=df_resid)

    margin = t_crit * bse_torch
    conf_int_lower = params_torch - margin
    conf_int_upper = params_torch + margin
    conf_int_torch = torch.stack([conf_int_lower, conf_int_upper], dim=1)

    return bse_torch, tvalues_torch, pvalues_torch, conf_int_torch


def compute_r2_torch(y, resid):
    """
    Compute R-squared on Torch GPU.

    Parameters
    ----------
    y : torch.Tensor
        True values on GPU.
    resid : torch.Tensor
        Residuals on GPU.

    Returns
    -------
    r2 : float
        R-squared value.
    """
    torch = _import_torch()

    y_mean = torch.mean(y)
    ss_res = torch.sum(resid ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    return float(r2.cpu().numpy())


def compute_aic_bic_torch(n, k, scale, device=None):
    """
    Compute AIC/BIC on Torch GPU.

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of parameters.
    scale : float or torch.Tensor
        Error variance (MLE estimate: RSS/n).
    device : str, optional
        Torch device string.

    Returns
    -------
    aic : float
        AIC value.
    bic : float
        BIC value.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    # Convert to torch if needed
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float64, device=device)

    # AIC = n * log(scale) + 2*k
    # BIC = n * log(scale) + k * log(n)
    n_tensor = torch.tensor(float(n), dtype=torch.float64, device=device)
    k_tensor = torch.tensor(float(k), dtype=torch.float64, device=device)

    aic_tensor = n_tensor * torch.log(scale) + 2 * k_tensor
    bic_tensor = n_tensor * torch.log(scale) + k_tensor * torch.log(n_tensor)

    return float(aic_tensor.cpu().numpy()), float(bic_tensor.cpu().numpy())


def compute_f_stat_torch(y, resid, X_design, df_resid, device=None):
    """
    Compute F-statistic and p-value on Torch GPU.

    Parameters
    ----------
    y : torch.Tensor
        True values on GPU.
    resid : torch.Tensor
        Residuals on GPU.
    X_design : torch.Tensor
        Design matrix on GPU.
    df_resid : int
        Residual degrees of freedom.
    device : str, optional
        Torch device string.

    Returns
    -------
    fvalue : float
        F-statistic.
    pvalue : float
        p-value for F-statistic.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    from statgpu.inference._distributions_backend import get_distribution
    f_dist = get_distribution("f", backend="torch", device=device)

    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    ss_res = torch.sum(resid ** 2)
    ss_reg = ss_tot - ss_res

    k = X_design.shape[1] - 1  # exclude intercept

    if k == 0 or ss_res <= 0:
        return float('inf'), 1.0

    fvalue_tensor = (ss_reg / k) / (ss_res / df_resid)
    fvalue = float(fvalue_tensor.cpu().numpy())

    # p-value using F CDF
    # For F ~ F(d1, d2): CDF(x) = I_{d1*x/(d1*x+d2)}(d1/2, d2/2)
    d1 = float(k)
    d2 = float(df_resid)

    if d2 <= 0 or d1 <= 0:
        pvalue = 1.0
    else:
        z = (d1 * fvalue) / (d1 * fvalue + d2)
        cdf = f_dist.cdf(fvalue, dfn=d1, dfd=d2)
        pvalue = 1.0 - float(cdf.cpu().numpy())

    return fvalue, pvalue


def torch_memory_cleanup():
    """
    Best-effort Torch memory cleanup.

    Empties CUDA cache if available.
    """
    torch = _import_torch()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def is_torch_tensor(x):
    """Check if input is a Torch tensor."""
    torch = _import_torch()
    return isinstance(x, torch.Tensor)


def to_numpy_from_torch(x):
    """
    Convert Torch tensor to NumPy array.

    Handles both CPU and CUDA tensors.
    """
    torch = _import_torch()

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        return x.detach().numpy()

    # Handle non-tensor inputs
    if hasattr(x, 'get'):  # CuPy array
        return x.get()
    return np.asarray(x)


def to_torch_from_numpy(x, device=None, dtype=None):
    """
    Convert NumPy array (or other types) to Torch tensor.

    Parameters
    ----------
    x : array-like
        Input data (NumPy, CuPy, or list).
    device : str, optional
        Target device ('cpu' or 'cuda').
    dtype : torch.dtype, optional
        Target dtype.

    Returns
    -------
    torch.Tensor
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    # Handle CuPy arrays
    if hasattr(x, 'get'):
        x = x.get()

    # Handle Torch tensors
    if isinstance(x, torch.Tensor):
        if x.device.type != device:
            x = x.to(device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        return x

    # Convert to numpy first, then to torch
    x_np = np.asarray(x)
    tensor = torch.from_numpy(x_np).to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor
