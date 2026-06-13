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
        arr_np = np.asarray(arr)
        # Preserve int types (needed for indexing), convert others to float64
        if arr_np.dtype.kind in ('i', 'u'):
            t = torch.from_numpy(arr_np)
        else:
            t = torch.from_numpy(arr_np.astype(np.float64))
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


def _batched_group_norms(coef, group_indices, xp):
    """Compute L2 norms for each group, all on device. Returns (G,) array."""
    norms_list = []
    for idx in group_indices:
        if len(idx) > 0:
            norms_list.append(_vector_norm(coef[idx], xp))
        else:
            if xp.__name__ == "torch":
                norms_list.append(xp.zeros(1, device=coef.device, dtype=coef.dtype)[0])
            elif xp.__name__ == "cupy":
                norms_list.append(xp.zeros(1, dtype=coef.dtype)[0])
            else:
                norms_list.append(0.0)
    if xp.__name__ == "torch":
        return xp.stack(norms_list)
    elif xp.__name__ == "cupy":
        return xp.array(norms_list)
    return np.array(norms_list)


def _get_xp(coef):
    """Return the correct module (numpy/torch/cupy) for a given array."""
    mod = type(coef).__module__
    if mod.startswith("torch"):
        import torch
        return torch
    elif mod.startswith("cupy"):
        import cupy
        return cupy
    return np


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
                [np.asarray(g, dtype=np.int64) for g in self._group_indices]
            )

        # Invalidate cached device tensors for _sqrt_pg
        self._sqrt_pg_torch = None
        self._sqrt_pg_cupy = None

        # Precompute padded gather/scatter index arrays (for unequal groups)
        if not self._all_equal_size:
            self._padded_row_idx = np.repeat(np.arange(self._n_groups), self._group_sizes).astype(np.int64)
            self._padded_col_idx = np.concatenate([np.arange(sz) for sz in self._group_sizes]).astype(np.int64)

        # Precompute feature→group mapping (for gradient/lla_weights vectorization)
        flat_indices = np.concatenate(
            [np.asarray(g, dtype=np.int64) for g in self._group_indices]
        )
        if flat_indices.size == 0:
            raise ValueError("groups must contain at least one feature index")
        max_idx = int(flat_indices.max())
        expected = max_idx + 1
        unique_idx = np.unique(flat_indices)
        if unique_idx.size != flat_indices.size:
            raise ValueError("groups contain duplicate feature indices")
        if unique_idx.size != expected:
            raise ValueError(
                "groups must cover a dense range of feature indices [0..max_index]"
            )
        self._group_feat_idx = np.empty(expected, dtype=np.int64)
        for g, idx in enumerate(self._group_indices):
            self._group_feat_idx[idx] = g

        # Invalidate all cached device tensors
        self._padded_row_idx_torch = None
        self._padded_row_idx_cupy = None
        self._padded_col_idx_torch = None
        self._padded_col_idx_cupy = None
        self._flat_indices_torch = None
        self._flat_indices_cupy = None
        self._group_feat_idx_torch = None
        self._group_feat_idx_cupy = None

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef) -> float:
        if self._group_indices is None:
            raise ValueError("groups must be set before calling value()")

        xp = _get_xp(coef)
        is_torch = xp.__name__ == "torch"
        is_cupy = xp.__name__ == "cupy"

        p_total = int(self._group_sizes.sum())
        coef_feat = coef[:p_total]  # handle augmented intercept

        # Compute all group norms in one batch (stays on device)
        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            if self._is_contiguous:
                w_mat = coef_feat.reshape(self._n_groups, gs)
            else:
                w_mat = coef_feat[self._flat_indices].reshape(self._n_groups, gs)
            norms = _vector_norm(w_mat, xp, dim=1)
        else:
            norms = self._batched_group_norms_vec(coef_feat, xp, coef)

        sqrt_pg = self._get_sqrt_pg(xp, coef)

        if is_torch:
            return xp.sum(self.alpha * sqrt_pg * norms).item()
        elif is_cupy:
            return float(xp.sum(self.alpha * sqrt_pg * norms))
        else:
            return float(np.sum(self.alpha * sqrt_pg * norms))

    # ----------------------------------------------------------------
    # Gradient
    # ----------------------------------------------------------------

    def gradient(self, coef) -> np.ndarray:
        if self._group_indices is None:
            raise ValueError("groups must be set before calling gradient()")

        xp = _get_xp(coef)
        is_torch = xp.__name__ == "torch"
        is_cupy = xp.__name__ == "cupy"

        p_total = int(self._group_sizes.sum())
        coef_feat = coef[:p_total]  # handle augmented intercept

        # Equal-size groups: fully vectorized path
        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            G = self._n_groups
            if self._is_contiguous:
                w_mat = coef_feat.reshape(G, gs)
            else:
                w_mat = coef_feat[self._flat_indices].reshape(G, gs)

            norms = _vector_norm(w_mat, xp, dim=1)
            sqrt_pg = self._get_sqrt_pg(xp, coef)

            # Unified path for all backends
            safe_norms = xp.clamp(norms, min=1e-15) if is_torch else xp.maximum(norms, 1e-15)
            scale = xp.where(norms > 1e-15,
                             self.alpha * sqrt_pg / safe_norms,
                             0.0)
            grad_mat = w_mat * scale[:, None]
            if is_torch or is_cupy:
                grad = xp.zeros_like(coef)
            else:
                grad = np.zeros_like(coef, dtype=float)
            if self._is_contiguous:
                grad[:p_total] = grad_mat.reshape(-1)
            else:
                grad[self._flat_indices] = grad_mat.reshape(-1)
            return grad

        # Unequal groups: vectorized scale + scatter via _group_feat_idx
        norms = self._batched_group_norms_vec(coef_feat, xp, coef)
        sqrt_pg = self._get_sqrt_pg(xp, coef)

        # Fused: single scale_g (eliminates separate safe_norms + where)
        safe_norms = xp.clamp(norms, min=1e-15) if is_torch else xp.maximum(norms, 1e-15)
        scale_g = xp.where(norms > 1e-15,
                           self.alpha * sqrt_pg / safe_norms,
                           0.0)

        feat_idx = self._get_cached('_group_feat_idx', xp, coef)
        grad = xp.zeros_like(coef)
        grad[:p_total] = scale_g[feat_idx] * coef_feat
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

    def _get_sqrt_pg(self, xp, w):
        """Cached device tensor for _sqrt_pg."""
        if xp.__name__ == "torch":
            if self._sqrt_pg_torch is None:
                self._sqrt_pg_torch = _to_backend_array(self._sqrt_pg, xp, w)
            return self._sqrt_pg_torch
        else:
            if self._sqrt_pg_cupy is None:
                self._sqrt_pg_cupy = _to_backend_array(self._sqrt_pg, xp, w)
            return self._sqrt_pg_cupy

    def _get_cached(self, attr_name, xp, w):
        """Get or create cached device tensor for a numpy attribute."""
        backend = "torch" if xp.__name__ == "torch" else "cupy"
        cache_attr = f"_{attr_name}_{backend}"
        cached = getattr(self, cache_attr, None)
        if cached is None:
            cached = _to_backend_array(getattr(self, attr_name), xp, w)
            setattr(self, cache_attr, cached)
        return cached

    def _get_flat_indices(self, xp, w):
        """Cached device tensor for _flat_indices."""
        if not hasattr(self, '_flat_indices') or self._flat_indices is None:
            return None
        return self._get_cached('_flat_indices', xp, w)

    def _batched_group_norms_vec(self, coef_feat, xp, w_ref):
        """Vectorized batched group norms using padded fancy indexing.

        Replaces _batched_group_norms() Python loop with 3 kernels:
        1. zeros allocation
        2. fancy index scatter
        3. vectorized norm along dim=1
        """
        G = self._n_groups
        max_sz = int(self._group_sizes.max())
        padded = _backend_zeros((G, max_sz), xp, dtype=coef_feat.dtype, ref_arr=w_ref)
        row_idx_dev = self._get_cached('_padded_row_idx', xp, w_ref)
        col_idx_dev = self._get_cached('_padded_col_idx', xp, w_ref)
        if self._is_contiguous:
            padded[row_idx_dev, col_idx_dev] = coef_feat
        else:
            flat_idx_dev = self._get_flat_indices(xp, w_ref)
            padded[row_idx_dev, col_idx_dev] = coef_feat[flat_idx_dev]
        return _vector_norm(padded, xp, dim=1)

    def _proximal_equal(self, w, step, xp, G, gs):
        """Fast path: all groups equal size, vectorized norm + scale."""
        p_total = G * gs
        w_feat = w[:p_total]  # handle augmented intercept

        # Gather into (G, gs) matrix
        if self._is_contiguous:
            w_mat = w_feat.reshape(G, gs)
        else:
            w_mat = w_feat[self._flat_indices].reshape(G, gs)

        sqrt_pg_arr = self._get_sqrt_pg(xp, w)

        # Torch compiled fast path
        if xp.__name__ == "torch":
            compiled_fn = _get_group_lasso_torch_compiled_equal()
            if compiled_fn is not None:
                scaled_flat = compiled_fn(w_mat, sqrt_pg_arr, self.alpha, step)
                result = w.clone()
                if self._is_contiguous:
                    result[:p_total] = scaled_flat
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
            result[:p_total] = scaled_flat
        else:
            result[self._flat_indices] = scaled_flat
        return result

    def _proximal_padded(self, w, step, xp, G, max_sz):
        """General path: pad unequal groups, compute norms vectorized."""
        p_total = int(self._group_sizes.sum())
        w_feat = w[:p_total]  # handle augmented intercept

        # Build padded matrix (G, max_sz) via fancy indexing — 1 kernel launch
        padded = _backend_zeros((G, max_sz), xp, dtype=w.dtype, ref_arr=w)
        row_idx_dev = self._get_cached('_padded_row_idx', xp, w)
        col_idx_dev = self._get_cached('_padded_col_idx', xp, w)
        if self._is_contiguous:
            padded[row_idx_dev, col_idx_dev] = w_feat
        else:
            flat_idx_dev = self._get_flat_indices(xp, w)
            padded[row_idx_dev, col_idx_dev] = w_feat[flat_idx_dev]

        # Vectorized norms
        norms = _vector_norm(padded, xp, dim=1)

        sqrt_pg_arr = self._get_sqrt_pg(xp, w)
        thresh = self.alpha * sqrt_pg_arr * step
        scale = xp.clip(1.0 - thresh / (norms + 1e-12), 0.0, None)

        # Apply scaling
        padded_scaled = padded * scale[:, None]

        # Scatter back via fancy indexing — 1 kernel launch
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        if self._is_contiguous:
            result[:p_total] = padded_scaled[row_idx_dev, col_idx_dev]
        else:
            result[flat_idx_dev] = padded_scaled[row_idx_dev, col_idx_dev]
        return result

    # ----------------------------------------------------------------

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "alpha": self.alpha,
            "n_groups": self._n_groups if self._group_indices else 0,
        })
        return params


class AdaptiveGroupLassoPenalty(GroupLassoPenalty):
    """Group Lasso with per-group weights for LLA linearization of group SCAD/MCP.

    The penalty is:
        P(w) = alpha * sum_g weights_g * sqrt(p_g) * ||w_g||_2

    where weights_g are per-group LLA weights.
    """

    name = "adaptive_group_lasso"

    def __init__(self, groups, alpha=1.0, weights=None):
        super().__init__(alpha=alpha, groups=groups)
        # weights: per-group weight array, shape (n_groups,)
        # None = uniform (same as GroupLasso)
        self._group_weights = weights

    def set_weights(self, weights):
        """Update per-group weights (numpy array, shape (n_groups,))."""
        self._group_weights = weights
        # Invalidate cached device tensors
        self._group_weights_torch = None
        self._group_weights_cupy = None

    def _get_group_weights(self, xp, w):
        """Cached device tensor for _group_weights."""
        if self._group_weights is None:
            return None
        if xp.__name__ == "torch":
            if not hasattr(self, '_group_weights_torch') or self._group_weights_torch is None:
                self._group_weights_torch = _to_backend_array(self._group_weights, xp, w)
            return self._group_weights_torch
        else:
            if not hasattr(self, '_group_weights_cupy') or self._group_weights_cupy is None:
                self._group_weights_cupy = _to_backend_array(self._group_weights, xp, w)
            return self._group_weights_cupy

    def _proximal_loop(self, w, step, xp):
        """Per-group serial loop with per-group weights."""
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        for g, idx in enumerate(self._group_indices):
            w_g = w[idx]
            norm = float(xp.linalg.norm(w_g))
            wg = float(self._group_weights[g]) if self._group_weights is not None else 1.0
            thresh = self.alpha * wg * self._sqrt_pg[g] * step
            if norm > thresh:
                result[idx] = w_g * (1.0 - thresh / norm)
            else:
                result[idx] = 0.0
        return result

    def _proximal_equal(self, w, step, xp, G, gs):
        """Fast path: all groups equal size, vectorized norm + scale with weights."""
        p_total = G * gs
        w_feat = w[:p_total]  # handle augmented intercept

        if self._is_contiguous:
            w_mat = w_feat.reshape(G, gs)
        else:
            w_mat = w_feat[self._flat_indices].reshape(G, gs)

        sqrt_pg_arr = self._get_sqrt_pg(xp, w)
        weights_arr = self._get_group_weights(xp, w)
        if weights_arr is None:
            weights_arr = xp.ones(G, dtype=w.dtype)
            if hasattr(w, 'device'):
                weights_arr = weights_arr.to(device=w.device)

        norms = _vector_norm(w_mat, xp, dim=1)
        thresh = self.alpha * weights_arr * sqrt_pg_arr * step
        scale = xp.clamp(1.0 - thresh / (norms + 1e-12), 0.0, None) if xp.__name__ == "torch" else xp.clip(1.0 - thresh / (norms + 1e-12), 0.0, None)
        scaled_flat = (w_mat * scale[:, None]).reshape(-1)

        result = w.clone() if hasattr(w, 'clone') else w.copy()
        if self._is_contiguous:
            result[:p_total] = scaled_flat
        else:
            result[self._flat_indices] = scaled_flat
        return result

    def _proximal_padded(self, w, step, xp, G, max_sz):
        """General path: pad unequal groups with per-group weights (fancy indexing)."""
        p_total = int(self._group_sizes.sum())
        w_feat = w[:p_total]  # handle augmented intercept

        padded = _backend_zeros((G, max_sz), xp, dtype=w.dtype, ref_arr=w)
        row_idx_dev = self._get_cached('_padded_row_idx', xp, w)
        col_idx_dev = self._get_cached('_padded_col_idx', xp, w)
        if self._is_contiguous:
            padded[row_idx_dev, col_idx_dev] = w_feat
        else:
            flat_idx_dev = self._get_flat_indices(xp, w)
            padded[row_idx_dev, col_idx_dev] = w_feat[flat_idx_dev]

        norms = _vector_norm(padded, xp, dim=1)
        sqrt_pg_arr = self._get_sqrt_pg(xp, w)
        weights_arr = self._get_group_weights(xp, w)
        if weights_arr is None:
            weights_arr = xp.ones(G, dtype=w.dtype)
            if hasattr(w, 'device'):
                weights_arr = weights_arr.to(device=w.device)

        thresh = self.alpha * weights_arr * sqrt_pg_arr * step
        if xp.__name__ == "torch":
            scale = xp.clamp(1.0 - thresh / (norms + 1e-12), min=0.0)
        else:
            scale = xp.clip(1.0 - thresh / (norms + 1e-12), 0.0, None)
        padded_scaled = padded * scale[:, None]

        result = w.copy() if hasattr(w, 'copy') else w.clone()
        if self._is_contiguous:
            result[:p_total] = padded_scaled[row_idx_dev, col_idx_dev]
        else:
            result[flat_idx_dev] = padded_scaled[row_idx_dev, col_idx_dev]
        return result

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "weights": self._group_weights,
        })
        return params
