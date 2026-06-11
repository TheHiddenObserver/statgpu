"""
Backend utilities for GLM loss functions.

Provides wrapper functions that dispatch to numpy/cupy/torch
based on the input array type, so GLM loss functions can use
a single code path for all backends.
"""

import numpy as np

from statgpu.backends._base import _resolve_backend


def _xp(arr):
    """Get the array module (numpy/cupy/torch) from array type."""
    mod = type(arr).__module__
    if mod.startswith("cupy"):
        import cupy
        return cupy
    if mod.startswith("torch"):
        import torch
        return torch
    import numpy
    return numpy


def _clip(arr, lo, hi):
    """Clip array values."""
    xp = _xp(arr)
    if xp.__name__ == "torch":
        if lo is not None and hi is not None:
            return xp.clamp(arr, min=lo, max=hi)
        if lo is not None:
            return xp.clamp(arr, min=lo)
        if hi is not None:
            return xp.clamp(arr, max=hi)
        return arr
    return xp.clip(arr, lo, hi)


def _exp(arr):
    """Element-wise exponential."""
    xp = _xp(arr)
    return xp.exp(arr)


def _log(arr):
    """Element-wise natural log."""
    xp = _xp(arr)
    return xp.log(arr)


def _log1p(arr):
    """Element-wise log(1+x)."""
    xp = _xp(arr)
    return xp.log1p(arr)


def _sigmoid(arr):
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    xp = _xp(arr)
    # float32 overflows exp() at ~89; float64 at ~709
    dtype = getattr(arr, 'dtype', None)
    max_val = 88.0 if dtype is not None and '32' in str(dtype) else 500.0
    z = _clip(arr, -max_val, max_val)
    if xp.__name__ == "torch":
        return xp.sigmoid(z)
    return 1.0 / (1.0 + xp.exp(-z))


def _softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    xp = _xp(x)
    return xp.log1p(xp.exp(-xp.abs(x))) + _clip(x, 0.0, None)


def _sum(arr):
    """Sum of all elements."""
    xp = _xp(arr)
    return xp.sum(arr)


def _eigvalsh(arr):
    """Eigenvalues of a symmetric matrix (sorted ascending)."""
    xp = _xp(arr)
    return xp.linalg.eigvalsh(arr)


def _zeros_like(arr):
    """Create zeros array with same shape/type as arr."""
    xp = _xp(arr)
    return xp.zeros_like(arr)


def _zeros(n, backend, ref_tensor=None, dtype=None):
    """Create a 1-D zeros vector on the requested backend."""
    backend = _resolve_backend(backend, ref_tensor)
    if backend == "numpy":
        return np.zeros(n, dtype=dtype)
    if backend == "cupy":
        import cupy as cp
        out_dtype = (
            dtype if dtype is not None else getattr(ref_tensor, "dtype", cp.float64)
        )
        return cp.zeros(n, dtype=out_dtype)
    import torch
    device = getattr(ref_tensor, "device", "cpu") if ref_tensor is not None else "cpu"
    out_dtype = dtype or (
        getattr(ref_tensor, "dtype", torch.float64)
        if ref_tensor is not None
        else torch.float64
    )
    return torch.zeros(n, device=device, dtype=out_dtype)


def _copy_arr(arr):
    """Copy array: .clone() for torch, .copy() for numpy/cupy."""
    if hasattr(arr, "clone"):
        return arr.clone()
    return arr.copy()


def _diag(reg, backend="auto", ref_tensor=None, dtype=None):
    """Create a diagonal matrix on the requested backend."""
    backend = _resolve_backend(backend, ref_tensor, reg)
    if backend == "cupy":
        import cupy as cp
        out_dtype = dtype if dtype is not None else getattr(reg, "dtype", cp.float64)
        return cp.diag(cp.asarray(reg, dtype=out_dtype))
    if backend == "torch":
        import torch
        device = (
            ref_tensor.device
            if ref_tensor is not None
            else getattr(reg, "device", "cpu")
        )
        out_dtype = dtype or (
            ref_tensor.dtype
            if ref_tensor is not None
            and getattr(ref_tensor, "is_floating_point", lambda: False)()
            else reg.dtype
            if hasattr(reg, "is_floating_point")
            and reg.is_floating_point()
            else torch.float64
        )
        return torch.diag(torch.as_tensor(reg, dtype=out_dtype, device=device))
    arr = np.asarray(reg, dtype=dtype) if dtype is not None else reg
    return np.diag(arr)


def _to_backend(arr, backend="auto", ref_tensor=None, dtype=None):
    """Convert an array to the requested backend, matching ref_tensor when needed."""
    backend = _resolve_backend(backend, ref_tensor, arr)
    if backend == "cupy":
        import cupy as cp
        out_dtype = dtype
        if out_dtype is None:
            ref_dtype = getattr(ref_tensor, "dtype", None)
            try:
                if ref_dtype is not None and np.issubdtype(ref_dtype, np.floating):
                    out_dtype = ref_dtype
                else:
                    out_dtype = cp.float64
            except TypeError:
                out_dtype = cp.float64
        return cp.asarray(arr, dtype=out_dtype)
    if backend == "torch":
        import torch
        device = (
            ref_tensor.device
            if ref_tensor is not None
            else getattr(arr, "device", "cpu")
        )
        out_dtype = dtype or (
            ref_tensor.dtype
            if ref_tensor is not None
            and getattr(ref_tensor, "is_floating_point", lambda: False)()
            else arr.dtype
            if hasattr(arr, "is_floating_point")
            and arr.is_floating_point()
            else torch.float64
        )
        return torch.as_tensor(arr, dtype=out_dtype, device=device)
    return np.asarray(arr, dtype=dtype or float)


def _solve_linear_system(A, b, backend="auto"):
    """Solve a linear system, falling back to least squares if singular."""
    backend = _resolve_backend(backend, A)
    try:
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.solve(A, b_col)
            return sol.squeeze(1) if b.ndim == 1 else sol
        if backend == "cupy":
            import cupy as cp
            return cp.linalg.solve(A, b)
        return np.linalg.solve(A, b)
    except (np.linalg.LinAlgError, Exception):
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.lstsq(A, b_col).solution
            return sol.squeeze(1) if b.ndim == 1 else sol
        if backend == "cupy":
            import cupy as cp
            return cp.linalg.lstsq(A, b)[0]
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _eye_like(n, ref):
    """Create an identity matrix on the same backend/device as ref."""
    backend = _resolve_backend("auto", ref)
    if backend == "cupy":
        import cupy as cp
        return cp.eye(n, dtype=ref.dtype)
    if backend == "torch":
        import torch
        return torch.eye(n, dtype=ref.dtype, device=ref.device)
    return np.eye(n, dtype=getattr(ref, "dtype", np.float64))


def _sync_scalars(*dev_vals, backend):
    """Batch device scalars into Python floats with one backend sync point."""
    backend = _resolve_backend(backend, *dev_vals)
    if backend == "numpy":
        return tuple(float(v) for v in dev_vals)
    if backend == "torch":
        import torch
        ref = next(
            (
                v
                for v in dev_vals
                if type(v).__module__.startswith("torch")
            ),
            None,
        )
        device = getattr(ref, "device", None)
        dtype = getattr(ref, "dtype", torch.float64)
        stacked = torch.stack(
            [torch.as_tensor(v, device=device, dtype=dtype) for v in dev_vals]
        )
        return tuple(stacked[i].item() for i in range(len(dev_vals)))
    import cupy as cp
    stacked = cp.stack([cp.asarray(v) for v in dev_vals])
    return tuple(float(stacked[i]) for i in range(len(dev_vals)))


def _abs_sum(x):
    """Sum of absolute values, returned as a Python scalar."""
    xp = _xp(x)
    if xp.__name__ == "torch":
        return float(xp.sum(xp.abs(x)).item())
    return float(xp.sum(xp.abs(x)))


def _abs_max(x):
    """Max absolute value, returned as a Python scalar."""
    xp = _xp(x)
    if xp.__name__ == "torch":
        return float(xp.max(xp.abs(x)).item())
    return float(xp.max(xp.abs(x)))


def _norm2(x):
    """L2 norm, returned as a Python scalar."""
    xp = _xp(x)
    if xp.__name__ == "torch":
        return float(xp.linalg.norm(x).item())
    return float(xp.linalg.norm(x))


def _dot(a, b):
    """Dot product, returned as a Python scalar."""
    val = a.dot(b)
    return float(val.item() if hasattr(val, "item") else val)


def _dot_dev(a, b):
    """Dot product staying on device for GPU backends."""
    if isinstance(a, np.ndarray):
        return float(a.dot(b))
    return a.dot(b)


def _sum_sq(x):
    """Sum of squares, returned as a Python scalar."""
    xp = _xp(x)
    val = xp.sum(x ** 2)
    return float(val.item() if hasattr(val, "item") else val)


def _sum_sq_dev(x):
    """Sum of squares staying on device for GPU backends."""
    xp = _xp(x)
    val = xp.sum(x ** 2)
    if xp.__name__ == "numpy":
        return float(val)
    return val


def _norm2_dev(x):
    """L2 norm staying on device for GPU backends."""
    xp = _xp(x)
    val = xp.linalg.norm(x)
    if xp.__name__ == "numpy":
        return float(val)
    return val


def _abs_sum_dev(x):
    """Sum of absolute values staying on device for GPU backends."""
    xp = _xp(x)
    val = xp.sum(xp.abs(x))
    if xp.__name__ == "numpy":
        return float(val)
    return val


def _device_leq(a, b):
    """Device-side a <= b comparison, returned as a Python bool."""
    backend = _resolve_backend("auto", a, b)
    if backend == "torch":
        return bool((a <= b).item())
    if backend == "cupy":
        return bool(a <= b)
    return a <= b


def _device_gt(a, b):
    """Device-side a > b comparison, returned as a Python bool."""
    backend = _resolve_backend("auto", a, b)
    if backend == "torch":
        return bool((a > b).item())
    if backend == "cupy":
        return bool(a > b)
    return a > b


def _clip_grad_on_device(grad, coef_old, backend):
    """Clip gradient entirely on the selected backend."""
    if backend == "numpy":
        gn = float(np.linalg.norm(grad))
        ca = float(np.sum(np.abs(coef_old)))
        gmax = max(ca * 10.0 + 1e3, 1e4)
        if gn > gmax:
            return grad * (gmax / gn)
        return grad
    if backend == "torch":
        import torch
        gn_sq = torch.sum(grad ** 2)
        coef_abs = torch.sum(torch.abs(coef_old))
        gmax = coef_abs * 10.0 + 1e3
        gmax = torch.clamp(gmax, min=1e4)
        scale = torch.where(
            gn_sq > gmax * gmax,
            gmax / torch.sqrt(gn_sq + 1e-30),
            torch.ones(1, device=grad.device, dtype=grad.dtype),
        )
        return grad * scale
    import cupy as cp
    gn_sq = cp.sum(grad ** 2)
    coef_abs = cp.sum(cp.abs(coef_old))
    gmax = cp.maximum(coef_abs * 10.0 + 1e3, 1e4)
    scale = cp.where(
        gn_sq > gmax * gmax,
        gmax / cp.sqrt(gn_sq + 1e-30),
        cp.ones(1, dtype=grad.dtype),
    )
    return grad * scale


def _max_eigval_power(mat, n_iter=20, tol=1e-8):
    """Largest eigenvalue of a symmetric matrix via power iteration.

    Much faster than full eigendecomposition, especially on GPU
    where cuSOLVER eigvalsh has high kernel compilation overhead.
    O(p^2) vs O(p^3). Accuracy within 1% for 20 iterations.

    Parameters
    ----------
    mat : 2-d array (p, p), symmetric positive semi-definite.
    n_iter : int
        Max power iterations.
    tol : float
        Early stopping tolerance on eigenvalue change.

    Returns
    -------
    float : max eigenvalue estimate.
    """
    xp = _xp(mat)
    p = mat.shape[0]
    dtype = getattr(mat, 'dtype', None)
    # Build a deterministic but non-constant seed vector to avoid the
    # pathological case where an all-ones vector is orthogonal to the top
    # eigenspace (e.g., [[1,-1],[-1,1]]).
    if xp.__name__ == "torch":
        v = xp.arange(1, p + 1, dtype=dtype, device=mat.device)
    else:
        v = xp.arange(1, p + 1)
        if dtype is not None and hasattr(v, 'astype'):
            v = v.astype(dtype)

    v_norm = xp.sqrt(xp.dot(v, v))
    v_norm_val = float(v_norm)
    if v_norm_val < 1e-15:
        return 1.0
    v = v / v_norm

    if xp.__name__ == "numpy":
        lambda_old = 0.0
        lambda_new = 0.0
        for _ in range(n_iter):
            v_new = mat @ v
            # Cache dot(v_new, v_new) to avoid recomputing mat @ v.
            nv2 = xp.dot(v_new, v_new)
            v_norm_sq = float(nv2)
            if v_norm_sq < 1e-30:
                return 1.0
            v_norm = v_norm_sq ** 0.5
            v = v_new / v_norm
            # lambda = v^T A v = v^T v_new (v_new = A v, already computed)
            lambda_new = float(xp.dot(v, v_new))
            if abs(lambda_new - lambda_old) < tol * abs(lambda_new):
                break
            lambda_old = lambda_new
        return lambda_new

    lambda_old = 0.0
    lambda_val = 0.0
    for i in range(n_iter):
        v_new = mat @ v
        dot_vn_vn = xp.dot(v_new, v_new)
        v_norm_sq = float(dot_vn_vn.item() if hasattr(dot_vn_vn, "item") else dot_vn_vn)
        if v_norm_sq < 1e-30:
            return 1.0  # Zero matrix — same fallback as numpy path
        v_norm = v_norm_sq ** 0.5
        v = v_new / v_norm
        lambda_new = xp.dot(v, v_new)
        lambda_val = float(lambda_new.item() if hasattr(lambda_new, "item") else lambda_new)
        if i > 0 and abs(lambda_val - lambda_old) < tol * abs(lambda_val):
            return lambda_val
        lambda_old = lambda_val
    return lambda_val


def _soft_threshold(w, thresh):
    """Soft-thresholding operator: sign(w) * max(|w| - thresh, 0).

    Works across numpy/cupy/torch.  ``thresh`` may be a scalar or an
    array with the same shape as ``w`` (adaptive weights).

    Uses ``xp.where`` for fewer intermediate arrays (2 vs 4 with
    sign*clip formulation).
    """
    xp = _xp(w)
    abs_w = xp.abs(w)
    return xp.where(abs_w > thresh, abs_w - thresh, 0.0) * xp.sign(w)


def _scalar_tensor(val, ref_arr):
    """Create a scalar value compatible with *ref_arr*'s backend/device.

    For torch, returns a 0-d tensor on the same device and dtype.
    For cupy/numpy, returns a plain Python float (scalars work directly).
    """
    xp = _xp(ref_arr)
    if xp.__name__ == "torch":
        import torch
        return torch.tensor(val, dtype=ref_arr.dtype, device=ref_arr.device)
    return float(val)


def _xp_copy(arr):
    """Copy array on the same backend.  `.clone()` for torch, `.copy()` for others."""
    xp = _xp(arr)
    if xp.__name__ == "torch":
        return arr.clone()
    return arr.copy()


def _xp_zeros(shape, dtype, ref_arr):
    """Create zeros array on the same device/dtype as *ref_arr*."""
    xp = _xp(ref_arr)
    if xp.__name__ == "torch":
        import torch
        return torch.zeros(shape, dtype=dtype or ref_arr.dtype, device=ref_arr.device)
    return xp.zeros(shape, dtype=dtype or getattr(ref_arr, 'dtype', None))


def _xp_asarray(arr, dtype, ref_arr):
    """Convert array to the same backend/device as *ref_arr*.

    Handles numpy→cupy, numpy→torch, and same-backend dtype casts.
    """
    xp = _xp(ref_arr)
    if xp.__name__ == "torch":
        import torch
        if isinstance(arr, torch.Tensor):
            out = arr.to(dtype=dtype, device=ref_arr.device)
        else:
            out = torch.as_tensor(np.asarray(arr, dtype=np.float64),
                                  dtype=dtype, device=ref_arr.device)
        return out
    if xp.__name__ == "cupy":
        return xp.asarray(arr, dtype=dtype)
    return np.asarray(arr, dtype=dtype)


def _xp_eye(n, dtype, ref_arr):
    """Create identity matrix on the same device/dtype as *ref_arr*."""
    xp = _xp(ref_arr)
    if xp.__name__ == "torch":
        import torch
        return torch.eye(n, dtype=dtype or ref_arr.dtype, device=ref_arr.device)
    return xp.eye(n, dtype=dtype or getattr(ref_arr, 'dtype', None))
