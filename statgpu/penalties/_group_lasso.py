"""
Group Lasso penalty.

Yuan & Lin, JRSSB 2006. Convex penalty that selects groups of features.

The penalty is:
    P(w) = alpha * sum_g sqrt(p_g) * ||w_g||_2

where w_g is the subvector of w for group g, and p_g is the size of group g.
"""
from typing import Optional, List, Union
import numpy as np
from ._base import Penalty

# ---- torch.compile lazy-loader for vectorized proximal on GPU ---------
_GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL = None


def _get_group_lasso_torch_compiled_equal():
    """torch.compile'd equal-size group lasso proximal (G,gs)→norms→scale→flat."""
    global _GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL
    if _GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL is not None:
        return _GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL
    from statgpu.penalties import _torch_compile_ok
    if not _torch_compile_ok():
        _GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL = None
        return None
    try:
        import torch
        def _prox(w_mat, sqrt_pg, alpha, step):
            thresh = alpha * sqrt_pg * step
            norms = torch.linalg.norm(w_mat, dim=1)
            scale = torch.clamp(1.0 - thresh / (norms + 1e-12), min=0.0)
            return (w_mat * scale[:, None]).reshape(-1)
        _GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL = torch.compile(
            _prox, dynamic=True, mode='reduce-overhead'
        )
    except Exception:
        _GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL = None
    return _GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL


def _vector_norm(x, xp, dim=None):
    """Backend-aware L2 norm along a dimension."""
    if xp.__name__ == "torch":
        return xp.linalg.norm(x, dim=dim) if dim is not None else xp.linalg.norm(x)
    return xp.linalg.norm(x, axis=dim) if dim is not None else xp.linalg.norm(x)


def _to_backend_array(arr, xp, ref_arr=None):
    """Convert numpy array to backend array type."""
    if xp.__name__ == "torch":
        import torch
        t = torch.from_numpy(np.asarray(arr, dtype=np.float64))
        if ref_arr is not None:
            t = t.to(device=ref_arr.device)
        return t
    return xp.asarray(arr)


def _backend_zeros(shape, xp, dtype=None, ref_arr=None):
    """Create zeros array on the correct backend."""
    if xp.__name__ == "torch":
        import torch
        t = torch.zeros(shape, dtype=dtype if dtype is not None else torch.float64)
        if ref_arr is not None:
            t = t.to(device=ref_arr.device)
        return t
    return xp.zeros(shape, dtype=dtype)


class GroupLassoPenalty(Penalty):
    """Group Lasso penalty.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    groups : list of lists, or 1D array-like
        Group membership specification. Two forms accepted:
        - List of lists of feature indices, e.g. [[0,1], [2,3,4]]
        - 1D array of length n_features where each entry is the group ID
    """

    name = "group_lasso"
    is_convex = True
    supports_group = True

    def __init__(
        self,
        alpha: float = 1.0,
        groups=None,
    ):
        self.alpha = alpha
        self._group_indices = None
        self._group_sizes = None
        self._all_equal_size = False
        self._is_contiguous = False
        self._group_size_uniform = None
        self._flat_indices = None

        if groups is not None:
            self._init_groups(groups)

    def _init_groups(self, groups):
        """Parse group specification into internal format."""
        if isinstance(groups, np.ndarray) and groups.ndim == 1:
            group_ids = np.asarray(groups, dtype=int)
            n_groups = int(group_ids.max() + 1)
            self._group_indices = [
                np.where(group_ids == g)[0] for g in range(n_groups)
            ]
        elif isinstance(groups, (list, tuple)):
            if len(groups) == 0:
                raise ValueError("groups must not be empty")
            if isinstance(groups[0], (list, tuple, np.ndarray)):
                self._group_indices = [
                    np.asarray(g, dtype=int) for g in groups
                ]
            else:
                group_ids = np.asarray(groups, dtype=int)
                n_groups = int(group_ids.max() + 1)
                self._group_indices = [
                    np.where(group_ids == g)[0] for g in range(n_groups)
                ]
        else:
            raise TypeError(
                f"groups must be list or array, got {type(groups).__name__}"
            )

        self._group_sizes = np.array(
            [len(g) for g in self._group_indices], dtype=int
        )
        self._sqrt_pg = np.sqrt(self._group_sizes.astype(float))
        self._n_groups = len(self._group_indices)

        # Detect equal-size groups for fast vectorized path
        sizes = self._group_sizes
        if len(sizes) > 0:
            unique_sizes = np.unique(sizes)
            self._all_equal_size = len(unique_sizes) == 1
            if self._all_equal_size:
                self._group_size_uniform = int(sizes[0])

        # Check if groups are contiguous [0..p1-1], [p1..p1+p2-1], ...
        self._is_contiguous = True
        pos = 0
        for g in range(self._n_groups):
            sz = sizes[g]
            if not np.array_equal(self._group_indices[g], np.arange(pos, pos + sz)):
                self._is_contiguous = False
                break
            pos += sz

        # Precompute flat indices for gather/scatter (only needed if non-contiguous)
        if not self._is_contiguous:
            self._flat_indices = np.concatenate(
                [np.asarray(g, dtype=int) for g in self._group_indices]
            )

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef: np.ndarray) -> float:
        if self._group_indices is None:
            raise ValueError("groups must be set before calling value()")

        total = 0.0
        for g, idx in enumerate(self._group_indices):
            w_g = coef[idx]
            total += self.alpha * self._sqrt_pg[g] * float(np.linalg.norm(w_g))
        return total

    # ----------------------------------------------------------------
    # Gradient
    # ----------------------------------------------------------------

    def gradient(self, coef: np.ndarray) -> np.ndarray:
        if self._group_indices is None:
            raise ValueError("groups must be set before calling gradient()")

        grad = np.zeros_like(coef, dtype=float)
        for g, idx in enumerate(self._group_indices):
            w_g = coef[idx]
            norm = float(np.linalg.norm(w_g))
            if norm > 1e-15:
                grad[idx] = self.alpha * self._sqrt_pg[g] * w_g / norm
        return grad

    # ----------------------------------------------------------------
    # Proximal operator (block soft-thresholding)
    # ----------------------------------------------------------------

    def proximal(self, w, step: float, backend: str = "numpy"):
        """Group soft-thresholding: each group is shrunk toward zero.

        GPU backends use vectorized reshape + axis-norm instead of a per-group
        serial loop, eliminating G× kernel-launch + D2H-sync overhead.
        """
        if self._group_indices is None:
            raise ValueError("groups must be set before calling proximal()")

        if backend == "cupy":
            import cupy as cp
            return self._proximal_vectorized(w, step, cp)
        elif backend == "torch":
            import torch
            return self._proximal_vectorized(w, step, torch)
        else:
            return self._proximal_loop(w, step, np)

    def _proximal_loop(self, w, step, xp):
        """Per-group serial loop (numpy CPU path)."""
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        for g, idx in enumerate(self._group_indices):
            w_g = w[idx]
            norm = float(xp.linalg.norm(w_g))
            thresh = self.alpha * self._sqrt_pg[g] * step
            if norm > thresh:
                result[idx] = w_g * (1.0 - thresh / norm)
            else:
                result[idx] = 0.0
        return result

    def _proximal_vectorized(self, w, step, xp):
        """Vectorized proximal: reshape groups into (G, gs) matrix, compute
        norms in one kernel, scale in one broadcast — O(1) kernel launches.

        For non-contiguous group layouts, a gather/scatter pass is added.
        """
        G = self._n_groups

        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            return self._proximal_equal(w, step, xp, G, gs)

        # Unequal groups: pad to max size
        max_sz = int(self._group_sizes.max())
        return self._proximal_padded(w, step, xp, G, max_sz)

    def _gather(self, w, xp):
        """Permute w so groups are contiguous. Identity if already contiguous."""
        if self._is_contiguous:
            return w.reshape(self._n_groups, self._group_size_uniform)
        return w[self._flat_indices].reshape(self._n_groups, self._group_size_uniform)

    def _scatter(self, w_mat_flat, result, xp):
        """Scatter vectorized result back. No-op if already contiguous."""
        if self._is_contiguous:
            result[:] = w_mat_flat
        else:
            result[self._flat_indices] = w_mat_flat
        return result

    def _proximal_equal(self, w, step, xp, G, gs):
        """Fast path: all groups equal size, vectorized norm + scale."""
        # Gather into (G, gs) matrix
        if self._is_contiguous:
            w_mat = w.reshape(G, gs)
        else:
            w_mat = w[self._flat_indices].reshape(G, gs)

        sqrt_pg_arr = _to_backend_array(self._sqrt_pg, xp, w)

        # Torch compiled fast path
        if xp.__name__ == "torch":
            compiled_fn = _get_group_lasso_torch_compiled_equal()
            if compiled_fn is not None:
                scaled_flat = compiled_fn(w_mat, sqrt_pg_arr, self.alpha, step)
                result = w.clone()
                if self._is_contiguous:
                    result[:] = scaled_flat
                else:
                    result[self._flat_indices] = scaled_flat
                return result

        # Generic vectorized path
        norms = _vector_norm(w_mat, xp, dim=1)
        thresh = self.alpha * sqrt_pg_arr * step
        scale = xp.clip(1.0 - thresh / (norms + 1e-12), 0.0, None)
        scaled_flat = (w_mat * scale[:, None]).reshape(-1)

        result = w.copy() if hasattr(w, 'copy') else w.clone()
        if self._is_contiguous:
            result[:] = scaled_flat
        else:
            result[self._flat_indices] = scaled_flat
        return result

    def _proximal_padded(self, w, step, xp, G, max_sz):
        """General path: pad unequal groups, compute norms vectorized."""
        sizes = self._group_sizes

        # Build padded matrix (G, max_sz)
        padded = _backend_zeros((G, max_sz), xp, dtype=w.dtype, ref_arr=w)
        pos = 0
        for g in range(G):
            sz = int(sizes[g])
            if sz > 0:
                if self._is_contiguous:
                    padded[g, :sz] = w[pos:pos + sz]
                else:
                    idx = self._flat_indices[pos:pos + sz]
                    padded[g, :sz] = w[idx]
            pos += sz

        # Vectorized norms
        norms = _vector_norm(padded, xp, dim=1)

        sqrt_pg_arr = _to_backend_array(self._sqrt_pg, xp, w)
        thresh = self.alpha * sqrt_pg_arr * step
        scale = xp.clip(1.0 - thresh / (norms + 1e-12), 0.0, None)

        # Apply scaling
        padded_scaled = padded * scale[:, None]

        # Scatter back
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        pos = 0
        for g in range(G):
            sz = int(sizes[g])
            if sz > 0:
                if self._is_contiguous:
                    result[pos:pos + sz] = padded_scaled[g, :sz]
                else:
                    idx = self._flat_indices[pos:pos + sz]
                    result[idx] = padded_scaled[g, :sz]
            pos += sz
        return result

    # ----------------------------------------------------------------

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "alpha": self.alpha,
            "n_groups": self._n_groups if self._group_indices else 0,
        })
        return params
