"""
Torch-specific distribution utilities.

This module provides Torch-native implementations of special functions
and distribution methods used by GPU inference paths.

Torch provides:
- torch.special: special functions (erf, erfc, gammaln, etc.)
- torch.distributions: probability distributions
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


# =============================================================================
# Special Functions (Torch native via torch.special)
# =============================================================================

def regularized_betainc_torch(a, b, x, device=None):
    """
    Regularized incomplete beta I_x(a, b) on Torch.

    Uses torch.special.betainc which computes the regularized incomplete beta function.
    Falls back to scipy.stats.beta.cdf if torch.special.betainc is not available.

    Parameters
    ----------
    a, b, x : array-like or floats
        Input parameters. a, b > 0, x in [0, 1].
    device : str, optional
        Torch device string ('cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        I_x(a, b) values on the specified device.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    a_tensor = torch.as_tensor(a, dtype=torch.float64, device=device)
    b_tensor = torch.as_tensor(b, dtype=torch.float64, device=device)

    try:
        return torch.special.betainc(a_tensor, b_tensor, x_tensor)
    except (AttributeError, RuntimeError):
        # Fallback: use scipy.stats.beta.cdf
        # I_x(a, b) = beta.cdf(x, a, b)
        return scipy_beta_cdf_torch(a_tensor, b_tensor, x_tensor, device=device)


def regularized_betaincinv_torch(a, b, y, device=None):
    """
    Inverse regularized incomplete beta on Torch.

    Uses torch.special.betaincinv (available in PyTorch 1.9+).

    Parameters
    ----------
    a, b, y : array-like or floats
        Input parameters. a, b > 0, y in [0, 1].
    device : str, optional
        Torch device string.

    Returns
    -------
    torch.Tensor
        I^{-1}_y(a, b) values.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    y_tensor = torch.as_tensor(y, dtype=torch.float64, device=device)
    a_tensor = torch.as_tensor(a, dtype=torch.float64, device=device)
    b_tensor = torch.as_tensor(b, dtype=torch.float64, device=device)

    try:
        return torch.special.betaincinv(a_tensor, b_tensor, y_tensor)
    except AttributeError:
        # Fallback for older PyTorch versions using bisection
        return _betaincinv_bisect_torch(a_tensor, b_tensor, y_tensor)
    except Exception as exc:
        raise RuntimeError("torch.special.betaincinv is required for Torch backend") from exc


def scipy_beta_cdf_torch(a, b, x, device=None):
    """
    Fallback: use scipy.stats.beta.cdf for regularized incomplete beta.

    I_x(a, b) = beta.cdf(x, a, b)

    Parameters
    ----------
    a, b, x : array-like or floats
        Input parameters. a, b > 0, x in [0, 1].
    device : str, optional
        Torch device string.

    Returns
    -------
    torch.Tensor
        I_x(a, b) values on the specified device.
    """
    torch = _import_torch()
    import scipy.stats as sps

    if device is None:
        device = _get_torch_device()

    # Convert to numpy for scipy
    a_np = a.cpu().numpy() if hasattr(a, 'cpu') else a
    b_np = b.cpu().numpy() if hasattr(b, 'cpu') else b
    x_np = x.cpu().numpy() if hasattr(x, 'cpu') else x

    result = sps.beta.cdf(x_np, a_np, b_np)

    return torch.as_tensor(result, dtype=torch.float64, device=device)


def _betaincinv_bisect_torch(a, b, y, max_iter=100, tol=1e-10):
    """Bisection-based inverse beta for older PyTorch versions."""
    torch = _import_torch()

    # Initialize search interval [0, 1]
    lo = torch.zeros_like(y)
    hi = torch.ones_like(y)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        cdf_mid = torch.special.betainc(a, b, mid)
        go_right = cdf_mid < y
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)

    return (lo + hi) / 2


def gammainc_torch(a, x, device=None):
    """
    Regularized lower incomplete gamma P(a, x) on Torch.

    Uses torch.special.gammainc.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    a_tensor = torch.as_tensor(a, dtype=torch.float64, device=device)

    try:
        return torch.special.gammainc(a_tensor, x_tensor)
    except Exception as exc:
        raise RuntimeError("torch.special.gammainc is required for Torch backend") from exc


def gammaincc_torch(a, x, device=None):
    """
    Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x) on Torch.

    Uses torch.special.gammaincc (complementary incomplete gamma).
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    a_tensor = torch.as_tensor(a, dtype=torch.float64, device=device)

    try:
        return torch.special.gammaincc(a_tensor, x_tensor)
    except Exception as exc:
        raise RuntimeError("torch.special.gammaincc is required for Torch backend") from exc


def gammaincinv_torch(a, q, device=None):
    """
    Inverse regularized lower incomplete gamma on Torch.

    Uses torch.special.gammaincinv (available in PyTorch 1.9+).
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    q_tensor = torch.as_tensor(q, dtype=torch.float64, device=device)
    a_tensor = torch.as_tensor(a, dtype=torch.float64, device=device)

    try:
        return torch.special.gammaincinv(a_tensor, q_tensor)
    except AttributeError:
        # Fallback for older PyTorch versions
        return _gammaincinv_bisect_torch(a_tensor, q_tensor)
    except Exception as exc:
        raise RuntimeError("torch.special.gammaincinv is required for Torch backend") from exc


def _gammaincinv_bisect_torch(a, q, max_iter=100, tol=1e-10):
    """Bisection-based inverse gamma for older PyTorch versions."""
    torch = _import_torch()

    # Use a reasonable upper bound
    lo = torch.zeros_like(q)
    hi = torch.maximum(q * 10, torch.ones_like(q) * 100)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        cdf_mid = torch.special.gammainc(a, mid)
        go_right = cdf_mid < q
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)

    return (lo + hi) / 2


def gammaln_torch(x, device=None):
    """
    Log-gamma function on Torch.

    Uses torch.lgamma (returns log of absolute value of gamma).
    For x > 0, lgamma(x) = ln(gamma(x)).
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)

    return torch.lgamma(x_tensor)


def erf_torch(x, device=None):
    """Error function on Torch."""
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    return torch.erf(x_tensor)


def erfc_torch(x, device=None):
    """Complementary error function on Torch."""
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    return torch.erfc(x_tensor)


def erfcinv_torch(y, device=None):
    """
    Inverse complementary error function on Torch.

    PyTorch 1.9+ has torch.special.erfcinv.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    y_tensor = torch.as_tensor(y, dtype=torch.float64, device=device)

    try:
        return torch.special.erfcinv(y_tensor)
    except AttributeError:
        # Fallback: use relationship erfcinv(y) = erfinv(1-y)
        return torch.special.erfinv(1 - y_tensor)
    except Exception as exc:
        raise RuntimeError("torch.special.erfcinv or torch.special.erfinv required") from exc


# =============================================================================
# Conversion Utilities
# =============================================================================

def to_numpy_for_scipy(value):
    """
    Convert Torch tensors to NumPy for SciPy fallback calls.

    Handles both CPU and GPU tensors.
    """
    torch = _import_torch()

    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return value.detach().cpu().numpy()
        return value.detach().numpy()
    return value


def scipy_dist_call_torch(dist_name, method_name, *args, device=None, **kwargs):
    """
    Call scipy.stats distribution method and return result on Torch.

    This is the Torch equivalent of scipy_dist_call_gpu.
    """
    torch = _import_torch()
    import scipy.stats as sps

    if device is None:
        device = _get_torch_device()

    dist = getattr(sps, dist_name)
    method = getattr(dist, method_name)

    # Convert Torch tensors to NumPy for SciPy
    np_args = [to_numpy_for_scipy(v) for v in args]
    np_kwargs = {k: to_numpy_for_scipy(v) for k, v in kwargs.items()}

    out = method(*np_args, **np_kwargs)

    # Convert back to Torch tensor
    if isinstance(out, tuple):
        return tuple(
            torch.as_tensor(v, dtype=torch.float64, device=device)
            if (np.isscalar(v) or isinstance(v, np.ndarray)) else v
            for v in out
        )
    if np.isscalar(out) or isinstance(out, np.ndarray):
        return torch.as_tensor(out, dtype=torch.float64, device=device)
    return out
