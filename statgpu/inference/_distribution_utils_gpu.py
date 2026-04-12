"""Internal GPU utility helpers shared by distribution implementations."""

import numpy as np


def regularized_betainc_gpu(a, b, x):
    """Regularized incomplete beta ``I_x(a, b)`` evaluated on GPU."""
    import cupy as cp

    x_gpu = cp.asarray(x, dtype=cp.float64)
    try:
        import cupyx.scipy.special as csp

        return csp.betainc(a, b, x_gpu)
    except Exception as exc:
        raise RuntimeError("cupyx.scipy.special.betainc is required for GPU backend") from exc


def regularized_betaincinv_gpu(a, b, y):
    """Inverse regularized incomplete beta on GPU."""
    import cupy as cp

    y_gpu = cp.asarray(y, dtype=cp.float64)
    try:
        import cupyx.scipy.special as csp

        return csp.betaincinv(a, b, y_gpu)
    except Exception as exc:
        raise RuntimeError("cupyx.scipy.special.betaincinv is required for GPU backend") from exc


def gammainc_gpu(a, x):
    """Regularized lower incomplete gamma on GPU."""
    import cupy as cp

    a_gpu = cp.asarray(a, dtype=cp.float64)
    x_gpu = cp.asarray(x, dtype=cp.float64)
    try:
        import cupyx.scipy.special as csp

        return csp.gammainc(a_gpu, x_gpu)
    except Exception as exc:
        raise RuntimeError("cupyx.scipy.special.gammainc is required for GPU backend") from exc


def gammaincc_gpu(a, x):
    """Regularized upper incomplete gamma on GPU."""
    import cupy as cp

    a_gpu = cp.asarray(a, dtype=cp.float64)
    x_gpu = cp.asarray(x, dtype=cp.float64)
    try:
        import cupyx.scipy.special as csp

        return csp.gammaincc(a_gpu, x_gpu)
    except Exception as exc:
        raise RuntimeError("cupyx.scipy.special.gammaincc is required for GPU backend") from exc


def gammaincinv_gpu(a, q):
    """Inverse regularized lower incomplete gamma on GPU."""
    import cupy as cp

    a_gpu = cp.asarray(a, dtype=cp.float64)
    q_gpu = cp.asarray(q, dtype=cp.float64)
    try:
        import cupyx.scipy.special as csp

        return csp.gammaincinv(a_gpu, q_gpu)
    except Exception as exc:
        raise RuntimeError("cupyx.scipy.special.gammaincinv is required for GPU backend") from exc


def gammaln_gpu(x):
    """Log-gamma on GPU."""
    import cupy as cp

    x_gpu = cp.asarray(x, dtype=cp.float64)
    try:
        import cupyx.scipy.special as csp

        return csp.gammaln(x_gpu)
    except Exception as exc:
        raise RuntimeError("cupyx.scipy.special.gammaln is required for GPU backend") from exc


def to_numpy_for_scipy(value):
    """Convert CuPy arrays to NumPy for SciPy fallback calls."""
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except Exception:
        pass
    return value


def scipy_dist_call_gpu(dist_name, method_name, *args, **kwargs):
    """Call scipy.stats distribution method and return result on GPU."""
    import cupy as cp
    import scipy.stats as sps

    dist = getattr(sps, dist_name)
    method = getattr(dist, method_name)
    np_args = [to_numpy_for_scipy(v) for v in args]
    np_kwargs = {k: to_numpy_for_scipy(v) for k, v in kwargs.items()}
    out = method(*np_args, **np_kwargs)

    if isinstance(out, tuple):
        return tuple(cp.asarray(v) if (np.isscalar(v) or isinstance(v, np.ndarray)) else v for v in out)
    if np.isscalar(out) or isinstance(out, np.ndarray):
        return cp.asarray(out)
    return out
