"""
Unified IRLS solver for GLM.

Extracted from the duplicated IRLS loops in _logistic.py across CPU/GPU/Torch.
Single implementation works on numpy/cupy/torch backends via auto detection.
"""

from typing import Optional

import numpy as np


def _infer_backend(X):
    """Detect backend from array type."""
    mod = type(X).__module__
    if mod.startswith("cupy"):
        return "cupy"
    if mod.startswith("torch"):
        return "torch"
    return "numpy"


def _solve(A, b, backend="auto"):
    """Solve linear system, fallback to lstsq if singular."""
    if backend == "auto":
        backend = _infer_backend(A)

    try:
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.solve(A, b_col)
            return sol.squeeze(1) if b.ndim == 1 else sol
        elif backend == "cupy":
            import cupy as cp
            return cp.linalg.solve(A, b)
        else:
            return np.linalg.solve(A, b)
    except (np.linalg.LinAlgError, Exception):
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.lstsq(A, b_col).solution
            return sol.squeeze(1) if b.ndim == 1 else sol
        elif backend == "cupy":
            import cupy as cp
            return cp.linalg.lstsq(A, b)[0]
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _clip(x, lo, hi, backend):
    if backend == "torch":
        import torch

        if lo is not None:
            x = torch.clamp(x, min=lo)
        if hi is not None:
            x = torch.clamp(x, max=hi)
        return x
    return np.clip(x, lo, hi)


def _norm(x, backend):
    if backend == "torch":
        import torch

        return float(torch.linalg.norm(x).item())
    return float(np.linalg.norm(x))


def _zeros(n, backend, ref_tensor=None, dtype=np.float64):
    if backend == "cupy":
        import cupy as cp
        return cp.zeros(n, dtype=cp.float64)
    if backend == "torch":
        import torch
        device = ref_tensor.device if ref_tensor is not None else "cpu"
        return torch.zeros(n, dtype=torch.float64, device=device)
    return np.zeros(n, dtype=dtype)


def _diag(reg, backend, ref_tensor=None):
    """Create diagonal matrix from 1D array."""
    if backend == "cupy":
        import cupy as cp
        return cp.diag(cp.asarray(reg, dtype=cp.float64))
    if backend == "torch":
        import torch
        return torch.diag(
            torch.tensor(reg, dtype=torch.float64, device=ref_tensor.device if ref_tensor is not None else "cpu")
        )
    return np.diag(reg)


def _to_backend(arr, backend, ref_tensor):
    """Convert numpy array to the target backend."""
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(arr, dtype=cp.float64)
    if backend == "torch":
        import torch
        return torch.tensor(arr, dtype=torch.float64, device=ref_tensor.device if ref_tensor is not None else "cpu")
    return np.asarray(arr, dtype=float)


def _copy_arr(arr):
    """Copy array: .clone() for torch, .copy() for numpy/cupy."""
    if hasattr(arr, 'clone'):
        return arr.clone()
    return arr.copy()


# =============================================================================
# Torch.compile for IRLS elementwise chain fusion
# =============================================================================
# When backend is torch on CUDA, the per-iteration elementwise ops
# (link inverse, weight computation, working response, weighted matmul)
# can be fused via torch.compile to reduce kernel launch overhead.

_IRLS_STEP_COMPILED = None


def _torch_compile_supported():
    """Check if torch.compile is safe (CUDA Capability >= 7.0)."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 7
    except Exception:
        pass
    return True


def _get_irls_step_compiled():
    """Lazily create a torch.compile'd IRLS step function."""
    global _IRLS_STEP_COMPILED
    if _IRLS_STEP_COMPILED is not None:
        return _IRLS_STEP_COMPILED

    import torch

    def _irls_weighted_gemm(X, W, z):
        """Weighted X'WX and X'Wz — elementwise ops fused by torch.compile."""
        W_col = W.unsqueeze(1)
        XtWX = X.T @ (X * W_col)
        Xtz = X.T @ (W * z)
        return XtWX, Xtz

    if _torch_compile_supported():
        try:
            _IRLS_STEP_COMPILED = torch.compile(_irls_weighted_gemm, dynamic=True, fullgraph=False)
        except Exception:
            _IRLS_STEP_COMPILED = _irls_weighted_gemm
    else:
        _IRLS_STEP_COMPILED = _irls_weighted_gemm

    return _IRLS_STEP_COMPILED


def _irls_step_call(compiled_fn, *args):
    """Call compiled IRLS step, falling back to eager on GPU arch mismatch."""
    try:
        return compiled_fn(*args)
    except Exception:
        def _irls_gemm_eager(X, W, z):
            W_col = W.unsqueeze(1)
            XtWX = X.T @ (X * W_col)
            Xtz = X.T @ (W * z)
            return XtWX, Xtz
        return _irls_gemm_eager(*args)


def irls_solver(
    family,
    X,
    y,
    max_iter=100,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
    ridge_alpha=0.0,
    ridge_penalize_intercept=False,
    backend="auto",
):
    """IRLS: solve GLM by iteratively weighted least squares.

    Parameters
    ----------
    family : Family
        GLM family with link/variance/irls_* methods.
    X : array
        Design matrix (n_samples, n_features).
    y : array
        Target (n_samples,).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on parameter change.
    init_coef : array, optional
        Initial coefficient vector.
    sample_weight : array, optional
        Sample weights.
    ridge_alpha : float
        L2 regularization (lambda = 1/(2*C) format).
    ridge_penalize_intercept : bool
        Whether to penalize the intercept.
    backend : str
        'numpy', 'cupy', 'torch', or 'auto'.

    Returns
    -------
    params : array
        Fitted parameters.
    n_iter : int
        Number of iterations.
    """
    if backend == "auto":
        backend = _infer_backend(X)

    if init_coef is None:
        n_features = X.shape[1]
        params = _zeros(n_features, backend, ref_tensor=X)
    else:
        params = init_coef

    for iteration in range(max_iter):
        params_old = _copy_arr(params)

        # Step 1: linear predictor
        eta = X @ params

        # Step 2: inverse link -> mean
        mu = family.link.inverse(eta)

        # Step 3: IRLS weights
        W = family.irls_weights(mu, y)
        W = _clip(W, 1e-10, None, backend)

        if sample_weight is not None:
            sw = _to_backend(sample_weight, backend, X)
            W = W * sw

        # Step 4: working response
        z = family.irls_working_response(mu, y, eta)

        # Step 5: weighted least squares (X'WX + lambda*I) params = X'Wz
        # Broadcasting: W is (n_samples,), X is (n_samples, n_features)
        if backend == "torch":
            import torch
            # Use torch.compile'd weighted GEMM for elementwise fusion
            W_col = W.unsqueeze(1)
            _compiled_step = _get_irls_step_compiled()
            XtWX, Xtz = _irls_step_call(_compiled_step, X, W, z)
        else:
            if backend == "cupy":
                import cupy as cp
                W_col = W[:, cp.newaxis]
            else:
                W_col = W[:, np.newaxis]
            XtWX = X.T @ (X * W_col)
            Xtz = X.T @ (W * z)

        if ridge_alpha > 0:
            reg = np.full(XtWX.shape[0], ridge_alpha)
            if not ridge_penalize_intercept:
                reg[0] = 0.0
            XtWX = XtWX + _diag(reg, backend, ref_tensor=X)

        params = _solve(XtWX, Xtz, backend)

        if _norm(params - params_old, backend) < tol:
            break

    return params, iteration + 1


class IRLSSolver:
    """Unified IRLS solver: each iteration solves weighted least squares.

    Supports numpy / cupy / torch backends (auto-detect X type).
    """

    def __init__(self, family, max_iter=100, tol=1e-4):
        self.family = family
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X,
        y,
        init_coef=None,
        sample_weight=None,
        ridge_alpha=0.0,
        ridge_penalize_intercept=False,
        backend="auto",
    ):
        """Run IRLS loop.

        Parameters
        ----------
        ridge_alpha : float
            L2 regularization (lambda = 1/(2*C) format).
        ridge_penalize_intercept : bool
            Whether to penalize the intercept.
        """
        return irls_solver(
            self.family,
            X,
            y,
            max_iter=self.max_iter,
            tol=self.tol,
            init_coef=init_coef,
            sample_weight=sample_weight,
            ridge_alpha=ridge_alpha,
            ridge_penalize_intercept=ridge_penalize_intercept,
            backend=backend,
        )
