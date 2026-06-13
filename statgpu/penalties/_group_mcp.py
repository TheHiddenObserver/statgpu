"""
Group MCP penalty.

Breheny & Huang 2009 (grpreg). Non-convex group penalty: applies MCP
concavity to the L2 norm of each feature group.

Penalty:
    P(w) = sum_g MCP(||w_g||_2; alpha * sqrt(p_g), gamma)

where MCP(t; lambda, gamma) is the element-wise MCP penalty.
"""
from typing import Optional, List, Union
import numpy as np
from ._base import Penalty
from ._group_lasso import _vector_norm, _to_backend_array, _backend_zeros, _batched_group_norms, _get_xp

# ---- torch.compile lazy-loader for vectorized MCP proximal ---------
_GROUP_MCP_PROXIMAL_TORCH_COMPILED = None


def _get_group_mcp_torch_compiled():
    global _GROUP_MCP_PROXIMAL_TORCH_COMPILED
    if _GROUP_MCP_PROXIMAL_TORCH_COMPILED is not None:
        return _GROUP_MCP_PROXIMAL_TORCH_COMPILED
    from statgpu.penalties import _torch_compile_ok
    if not _torch_compile_ok():
        _GROUP_MCP_PROXIMAL_TORCH_COMPILED = None
        return None
    try:
        import torch
        def _prox(w_mat, sqrt_pg, alpha, step, gamma):
            t_g = alpha * sqrt_pg * step
            gamma_alpha_g = gamma * alpha * sqrt_pg
            norms = torch.linalg.norm(w_mat, dim=1)
            mask_zero = norms <= t_g
            mask_shrink = (norms > t_g) & (norms <= gamma_alpha_g)
            denom = norms * (1.0 - step / gamma)
            denom = torch.where(mask_shrink, denom, torch.ones_like(denom))
            scale_shrink = (norms - t_g) / denom
            scale = torch.where(mask_shrink, scale_shrink, 1.0)
            scale = torch.where(mask_zero, 0.0, scale)
            return (w_mat * scale[:, None]).reshape(-1)
        _GROUP_MCP_PROXIMAL_TORCH_COMPILED = torch.compile(
            _prox, dynamic=True, mode='reduce-overhead'
        )
    except Exception:
        _GROUP_MCP_PROXIMAL_TORCH_COMPILED = None
    return _GROUP_MCP_PROXIMAL_TORCH_COMPILED


class GroupMCPPenalty(Penalty):
    """Group MCP penalty.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    gamma : float, default=3.0
        MCP concavity parameter. Larger gamma gives less bias (closer to
        group lasso). Must be > 1.
    groups : list of lists, or 1D array-like
        Group membership specification.

    Notes
    -----
    Group MCP is **non-convex** (``is_convex=False``), optimized via LLA
    (Local Linear Approximation). The objective function may contain multiple
    local minima. Different solvers or different initializations can converge
    to different local minima with comparable objective values — a coefficient
    ``max|diff|`` up to ~1e-2 across runs is expected and does not indicate a
    bug.
    """

    name = "group_mcp"
    is_convex = False
    supports_group = True

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 3.0,
        groups=None,
    ):
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar for group MCP penalty")
        if not np.isfinite(gamma) or gamma <= 1.0:
            raise ValueError("gamma must be a finite scalar greater than 1 for group MCP penalty")
        self.alpha = alpha
        self.gamma = gamma
        self._group_indices = None
        self._sqrt_pg = None
        self._n_groups = 0
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

        sizes = self._group_sizes
        if len(sizes) > 0:
            unique_sizes = np.unique(sizes)
            self._all_equal_size = len(unique_sizes) == 1
            if self._all_equal_size:
                self._group_size_uniform = int(sizes[0])

        self._is_contiguous = True
        pos = 0
        for g in range(self._n_groups):
            sz = sizes[g]
            if not np.array_equal(self._group_indices[g], np.arange(pos, pos + sz)):
                self._is_contiguous = False
                break
            pos += sz

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
        """Vectorized batched group norms using padded fancy indexing."""
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

    def _reshape_to_matrix(self, w, xp, G, gs):
        """Reshape w into (G, gs) matrix, handling non-contiguous layouts."""
        p_total = G * gs
        w_feat = w[:p_total]  # handle augmented intercept
        if self._is_contiguous:
            return w_feat.reshape(G, gs)
        return w_feat[self._flat_indices].reshape(G, gs)

    def _scatter_from_flat(self, flat_vals, result, xp):
        """Scatter flat values back, handling non-contiguous layouts."""
        p_total = len(flat_vals)
        if self._is_contiguous:
            result[:p_total] = flat_vals
        else:
            flat_idx = self._get_flat_indices(xp, result)
            result[flat_idx] = flat_vals

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
        alpha_g = self.alpha * sqrt_pg
        gamma_alpha_g = self.gamma * alpha_g

        if is_torch:
            import torch
            mask_small = norms <= gamma_alpha_g
            total = torch.sum(alpha_g[mask_small] * norms[mask_small]
                              - norms[mask_small] ** 2 / (2.0 * self.gamma))
            total += torch.sum(0.5 * self.gamma * alpha_g[~mask_small] ** 2)
            return total.item()
        elif is_cupy:
            import cupy as cp
            mask_small = norms <= gamma_alpha_g
            total = cp.sum(alpha_g[mask_small] * norms[mask_small]
                           - norms[mask_small] ** 2 / (2.0 * self.gamma))
            total += cp.sum(0.5 * self.gamma * alpha_g[~mask_small] ** 2)
            return float(total)
        else:
            mask_small = norms <= gamma_alpha_g
            total = np.sum(alpha_g[mask_small] * norms[mask_small]
                           - norms[mask_small] ** 2 / (2.0 * self.gamma))
            total += np.sum(0.5 * self.gamma * alpha_g[~mask_small] ** 2)
            return float(total)

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

        # Compute all group norms in one batch
        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            G = self._n_groups
            if self._is_contiguous:
                w_mat = coef_feat.reshape(G, gs)
            else:
                w_mat = coef_feat[self._flat_indices].reshape(G, gs)
            norms = _vector_norm(w_mat, xp, dim=1)
        else:
            norms = self._batched_group_norms_vec(coef_feat, xp, coef)

        sqrt_pg = self._get_sqrt_pg(xp, coef)
        alpha_g = self.alpha * sqrt_pg
        gamma_alpha_g = self.gamma * alpha_g

        # Fused: single scale_g per group (eliminates intermediate deriv_g + inv_norms_g)
        mask_active = (norms > 0) & (norms <= gamma_alpha_g)
        safe_norms = xp.clamp(norms, min=1e-15) if is_torch else xp.maximum(norms, 1e-15)
        scale_g = xp.where(mask_active,
                           (alpha_g - norms / self.gamma) / safe_norms,
                           0.0)

        feat_idx = self._get_cached('_group_feat_idx', xp, coef)
        grad = xp.zeros_like(coef)
        grad[:p_total] = scale_g[feat_idx] * coef_feat
        return grad

    # ----------------------------------------------------------------
    # Proximal operator (group MCP)
    # ----------------------------------------------------------------

    def proximal(self, w, step: float, backend: str = "numpy"):
        """Per-group MCP proximal — vectorized on GPU."""
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
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        for g, idx in enumerate(self._group_indices):
            w_g = w[idx]
            ng = float(xp.linalg.norm(w_g))
            t_g = self.alpha * self._sqrt_pg[g] * step
            gamma_alpha_g = self.gamma * self.alpha * self._sqrt_pg[g]

            if ng <= t_g:
                result[idx] = 0.0
            elif t_g < ng <= gamma_alpha_g:
                scale = (ng - t_g) / (ng * (1.0 - step / self.gamma))
                result[idx] = w_g * scale
            else:
                result[idx] = w_g
        return result

    def _proximal_vectorized(self, w, step, xp):
        """Vectorized group MCP proximal."""
        G = self._n_groups

        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            return self._proximal_equal(w, step, xp, G, gs)

        max_sz = int(self._group_sizes.max())
        return self._proximal_padded(w, step, xp, G, max_sz)

    def _proximal_equal(self, w, step, xp, G, gs):
        """Equal-size groups: vectorized MCP proximal."""
        # Clamp step to prevent division by zero in denom = norms*(1 - step/gamma)
        step = min(float(step), 0.9 * self.gamma)
        w_mat = self._reshape_to_matrix(w, xp, G, gs)
        sqrt_pg_arr = self._get_sqrt_pg(xp, w)

        # Torch compiled fast path
        if xp.__name__ == "torch":
            compiled_fn = _get_group_mcp_torch_compiled()
            if compiled_fn is not None:
                scaled_flat = compiled_fn(w_mat, sqrt_pg_arr, self.alpha, step, self.gamma)
                result = w.clone()
                self._scatter_from_flat(scaled_flat, result, xp)
                return result

        # Generic vectorized path
        norms = _vector_norm(w_mat, xp, dim=1)
        t_g = self.alpha * sqrt_pg_arr * step                    # (G,)
        gamma_alpha_g = self.gamma * self.alpha * sqrt_pg_arr    # (G,)

        # Region 1: norm <= t_g  → zero
        mask_zero = norms <= t_g
        # Region 2: t_g < norm <= gamma_alpha_g  → MCP shrinkage
        mask_shrink = (norms > t_g) & (norms <= gamma_alpha_g)
        # Region 3: norm > gamma_alpha_g  → no shrinkage (identity)

        denom = norms * (1.0 - step / self.gamma)
        denom = xp.where(mask_shrink, denom, xp.ones_like(denom))
        scale_shrink = (norms - t_g) / denom                    # (G,)
        scale = xp.where(mask_shrink, scale_shrink, 1.0)        # (G,)
        scale = xp.where(mask_zero, 0.0, scale)

        scaled_flat = (w_mat * scale[:, None]).reshape(-1)
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        self._scatter_from_flat(scaled_flat, result, xp)
        return result

    def _proximal_padded(self, w, step, xp, G, max_sz):
        """Unequal groups: pad, vectorize, unpack."""
        step = min(float(step), 0.9 * self.gamma)
        p_total = int(self._group_sizes.sum())
        w_feat = w[:p_total]  # handle augmented intercept

        # Build padded matrix via fancy indexing — 1 kernel launch
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
        t_g = self.alpha * sqrt_pg_arr * step
        gamma_alpha_g = self.gamma * self.alpha * sqrt_pg_arr

        mask_zero = norms <= t_g
        mask_shrink = (norms > t_g) & (norms <= gamma_alpha_g)
        denom = norms * (1.0 - step / self.gamma)
        denom = xp.where(mask_shrink, denom, xp.ones_like(denom))
        scale_shrink = (norms - t_g) / denom
        scale = xp.where(mask_shrink, scale_shrink, 1.0)
        scale = xp.where(mask_zero, 0.0, scale)

        padded_scaled = padded * scale[:, None]

        # Scatter back via fancy indexing — 1 kernel launch
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        if self._is_contiguous:
            result[:p_total] = padded_scaled[row_idx_dev, col_idx_dev]
        else:
            result[flat_idx_dev] = padded_scaled[row_idx_dev, col_idx_dev]
        return result

    # ----------------------------------------------------------------
    # LLA weights (for LLA outer loop optimization)
    # ----------------------------------------------------------------

    def lla_weights(self, coef):
        if self._group_indices is None:
            raise ValueError("groups must be set before calling lla_weights()")

        xp = _get_xp(coef)
        is_torch = xp.__name__ == "torch"
        is_cupy = xp.__name__ == "cupy"

        p_total = int(self._group_sizes.sum())
        coef_feat = coef[:p_total]  # handle augmented intercept

        # Compute all group norms in one batch
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
        alpha_g = self.alpha * sqrt_pg
        gamma_alpha_g = self.gamma * alpha_g

        # Per-group derivative weight
        if is_torch:
            import torch
            weight_g = torch.where(
                norms <= gamma_alpha_g,
                torch.clamp(alpha_g - norms / self.gamma, min=0.0),
                torch.zeros_like(norms),
            )
            # Broadcast to per-coordinate
            if self._all_equal_size and self._group_size_uniform is not None:
                gs = self._group_size_uniform
                weights = weight_g.repeat_interleave(gs)
            else:
                feat_idx = self._get_cached('_group_feat_idx', xp, coef)
                weights = weight_g[feat_idx]
            return weights
        elif is_cupy:
            import cupy as cp
            weight_g = cp.where(
                norms <= gamma_alpha_g,
                cp.maximum(alpha_g - norms / self.gamma, 0.0),
                0.0,
            )
            if self._all_equal_size and self._group_size_uniform is not None:
                gs = self._group_size_uniform
                weights = cp.repeat(weight_g, gs)
            else:
                feat_idx = self._get_cached('_group_feat_idx', xp, coef)
                weights = weight_g[feat_idx]
            return weights
        else:
            weight_g = np.where(
                norms <= gamma_alpha_g,
                np.maximum(alpha_g - norms / self.gamma, 0.0),
                0.0,
            )
            if self._all_equal_size and self._group_size_uniform is not None:
                gs = self._group_size_uniform
                weights = np.repeat(weight_g, gs)
            else:
                weights = weight_g[self._group_feat_idx]
            return weights

    # ----------------------------------------------------------------

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "n_groups": self._n_groups,
        })
        return params
