"""
Group SCAD penalty.

Breheny & Huang 2009 (grpreg). Non-convex group penalty: applies SCAD
concavity to the L2 norm of each feature group.

Penalty:
    P(w) = sum_g SCAD(||w_g||_2; alpha * sqrt(p_g), a)

where SCAD(t; lambda, a) is the element-wise SCAD penalty.
"""

__all__ = ["GroupSCADPenalty"]

from typing import Optional, List, Union
import numpy as np
from statgpu.penalties._base import Penalty
from statgpu.penalties._group_lasso import _vector_norm, _to_backend_array, _backend_zeros, _batched_group_norms, _get_xp

# ---- torch.compile lazy-loader for vectorized SCAD proximal ---------
_GROUP_SCAD_PROXIMAL_TORCH_COMPILED = None


def _get_group_scad_torch_compiled():
    global _GROUP_SCAD_PROXIMAL_TORCH_COMPILED
    if _GROUP_SCAD_PROXIMAL_TORCH_COMPILED is not None:
        return _GROUP_SCAD_PROXIMAL_TORCH_COMPILED
    from statgpu.penalties import _torch_compile_ok
    if not _torch_compile_ok():
        _GROUP_SCAD_PROXIMAL_TORCH_COMPILED = None
        return None
    try:
        import torch
        def _prox(w_mat, sqrt_pg, alpha, step, a):
            alpha_g = alpha * sqrt_pg
            t_g = alpha_g * step
            a_alpha_g = a * alpha_g
            norms = torch.linalg.norm(w_mat, dim=1)
            mask_r1 = norms <= alpha_g + t_g
            safe_norms = torch.where(norms > 0.0, norms, torch.ones_like(norms))
            scale_r1 = torch.clamp((norms - t_g) / safe_norms, min=0.0)
            mask_r2 = (norms > alpha_g + t_g) & (norms <= a_alpha_g)
            denom = norms * (a - 1.0 - step)
            denom = torch.where(mask_r2, denom, torch.ones_like(denom))
            scale_r2 = ((a - 1.0) * norms - a * t_g) / denom
            scale = torch.where(mask_r1, scale_r1, 1.0)
            scale = torch.where(mask_r2, scale_r2, scale)
            return (w_mat * scale[:, None]).reshape(-1)
        _GROUP_SCAD_PROXIMAL_TORCH_COMPILED = torch.compile(
            _prox, dynamic=True, mode='reduce-overhead'
        )
    except Exception:
        _GROUP_SCAD_PROXIMAL_TORCH_COMPILED = None
    return _GROUP_SCAD_PROXIMAL_TORCH_COMPILED


class GroupSCADPenalty(Penalty):
    """Group SCAD penalty.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    a : float, default=3.7
        SCAD concavity parameter. Must be > 2.
    groups : list of lists, or 1D array-like
        Group membership specification.

    Notes
    -----
    Group SCAD is **non-convex** (``is_convex=False``), optimized via LLA
    (Local Linear Approximation). The objective function may contain multiple
    local minima. Different solvers or different initializations can converge
    to different local minima with comparable objective values — a coefficient
    ``max|diff|`` up to ~1e-2 across runs is expected and does not indicate a
    bug.
    """

    name = "group_scad"
    is_convex = False
    supports_group = True

    def __init__(
        self,
        alpha: float = 1.0,
        a: float = 3.7,
        groups=None,
    ):
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar for group SCAD penalty")
        if not np.isfinite(a) or a <= 2.0:
            raise ValueError("a must be a finite scalar greater than 2 for group SCAD penalty")
        self.alpha = alpha
        self.a = a
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
            if self._sqrt_pg_torch is None or self._sqrt_pg_torch.device != w.device:
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
        elif xp.__name__ == "torch" and hasattr(cached, 'device') and cached.device != w.device:
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
        p_total = G * gs
        w_feat = w[:p_total]  # handle augmented intercept
        if self._is_contiguous:
            return w_feat.reshape(G, gs)
        return w_feat[self._flat_indices].reshape(G, gs)

    def _scatter_from_flat(self, flat_vals, result, xp):
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
        a = self.a

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
        a_alpha_g = a * alpha_g

        if is_torch:
            import torch
            mask_r1 = norms <= alpha_g
            mask_r2 = (norms > alpha_g) & (norms <= a_alpha_g)
            # Region 1: alpha_g * ng
            total = torch.sum(alpha_g[mask_r1] * norms[mask_r1])
            # Region 2: -(ng^2 - 2*a*alpha_g*ng + alpha_g^2) / (2*(a-1))
            ng2 = norms[mask_r2]
            ag2 = alpha_g[mask_r2]
            total += torch.sum(-(ng2 * ng2 - 2.0 * a * ag2 * ng2 + ag2 * ag2) / (2.0 * (a - 1.0)))
            # Region 3: (a+1)*alpha_g^2 / 2
            mask_r3 = norms > a_alpha_g
            total += torch.sum((a + 1.0) * alpha_g[mask_r3] ** 2 / 2.0)
            return total.item()
        elif is_cupy:
            import cupy as cp
            mask_r1 = norms <= alpha_g
            mask_r2 = (norms > alpha_g) & (norms <= a_alpha_g)
            total = cp.sum(alpha_g[mask_r1] * norms[mask_r1])
            ng2 = norms[mask_r2]
            ag2 = alpha_g[mask_r2]
            total += cp.sum(-(ng2 * ng2 - 2.0 * a * ag2 * ng2 + ag2 * ag2) / (2.0 * (a - 1.0)))
            mask_r3 = norms > a_alpha_g
            total += cp.sum((a + 1.0) * alpha_g[mask_r3] ** 2 / 2.0)
            return float(total)
        else:
            mask_r1 = norms <= alpha_g
            mask_r2 = (norms > alpha_g) & (norms <= a_alpha_g)
            total = np.sum(alpha_g[mask_r1] * norms[mask_r1])
            ng2 = norms[mask_r2]
            ag2 = alpha_g[mask_r2]
            total += np.sum(-(ng2 * ng2 - 2.0 * a * ag2 * ng2 + ag2 * ag2) / (2.0 * (a - 1.0)))
            mask_r3 = norms > a_alpha_g
            total += np.sum((a + 1.0) * alpha_g[mask_r3] ** 2 / 2.0)
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
        a = self.a

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
        a_alpha_g = a * alpha_g

        # Fused: single scale_g per group (eliminates intermediate deriv_g + inv_norms_g)
        # Region 1 (|w|<=alpha): deriv=alpha_g, scale=alpha_g/norm
        # Region 2 (alpha<|w|<=a*alpha): deriv=(a*alpha_g-norm)/(a-1), scale=deriv/norm
        # Region 3 (|w|>a*alpha): scale=0
        safe_norms = xp.clamp(norms, min=1e-15) if is_torch else xp.maximum(norms, 1e-15)
        scale_g = xp.where(norms <= alpha_g,
                           alpha_g / safe_norms,
                           xp.where((norms > alpha_g) & (norms <= a_alpha_g),
                                    (a * alpha_g - norms) / ((a - 1.0) * safe_norms),
                                    0.0))

        feat_idx = self._get_cached('_group_feat_idx', xp, coef)
        grad = xp.zeros_like(coef)
        grad[:p_total] = scale_g[feat_idx] * coef_feat
        return grad

    # ----------------------------------------------------------------
    # Proximal operator (group SCAD)
    # ----------------------------------------------------------------

    def proximal(self, w, step: float, backend: str = "numpy"):
        """Per-group SCAD proximal — vectorized on GPU."""
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
        step = min(float(step), 0.9 * (self.a - 1.0))  # defense-in-depth clamping
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        for g, idx in enumerate(self._group_indices):
            w_g = w[idx]
            ng = float(xp.linalg.norm(w_g))
            alpha_g = self.alpha * self._sqrt_pg[g]
            t_g = alpha_g * step
            a_alpha_g = self.a * alpha_g

            if ng <= alpha_g + t_g:
                if ng > 0:
                    result[idx] = w_g * max(0.0, ng - t_g) / ng
                else:
                    result[idx] = 0.0
            elif ng > alpha_g + t_g and ng <= a_alpha_g:
                if ng > 1e-15:
                    scale = ((self.a - 1.0) * ng - self.a * t_g) / (ng * (self.a - 1.0 - step))
                    result[idx] = w_g * scale
                else:
                    result[idx] = 0.0
            else:
                result[idx] = w_g
        return result

    def _proximal_vectorized(self, w, step, xp):
        """Vectorized group SCAD proximal."""
        G = self._n_groups

        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            return self._proximal_equal(w, step, xp, G, gs)

        max_sz = int(self._group_sizes.max())
        return self._proximal_padded(w, step, xp, G, max_sz)

    def _proximal_equal(self, w, step, xp, G, gs):
        """Equal-size groups: vectorized SCAD proximal (3 regions)."""
        # Clamp step to prevent division by zero in denom = norms*(a-1-step)
        step = min(float(step), 0.9 * (self.a - 1.0))
        w_mat = self._reshape_to_matrix(w, xp, G, gs)
        sqrt_pg_arr = self._get_sqrt_pg(xp, w)

        # Torch compiled fast path
        if xp.__name__ == "torch":
            compiled_fn = _get_group_scad_torch_compiled()
            if compiled_fn is not None:
                scaled_flat = compiled_fn(w_mat, sqrt_pg_arr, self.alpha, step, self.a)
                result = w.clone()
                self._scatter_from_flat(scaled_flat, result, xp)
                return result

        # Generic vectorized path
        norms = _vector_norm(w_mat, xp, dim=1)
        alpha_g = self.alpha * sqrt_pg_arr               # (G,)
        t_g = alpha_g * step                              # (G,)
        a_alpha_g = self.a * alpha_g                      # (G,)

        # Region 1: soft-threshold  ng <= alpha_g + t_g
        mask_r1 = norms <= alpha_g + t_g
        safe_norms = xp.where(norms > 0.0, norms, xp.ones_like(norms))
        scale_r1 = xp.clip((norms - t_g) / safe_norms, 0.0, None)

        # Region 2: SCAD intermediate  alpha_g + t_g < ng <= a_alpha_g
        mask_r2 = (norms > alpha_g + t_g) & (norms <= a_alpha_g)
        denom = norms * (self.a - 1.0 - step)
        denom = xp.where(mask_r2, denom, xp.ones_like(denom))
        scale_r2 = ((self.a - 1.0) * norms - self.a * t_g) / denom

        # Region 3: no shrinkage  ng > a_alpha_g
        # scale = 1.0 (default)

        scale = xp.where(mask_r1, scale_r1, 1.0)
        scale = xp.where(mask_r2, scale_r2, scale)

        scaled_flat = (w_mat * scale[:, None]).reshape(-1)
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        self._scatter_from_flat(scaled_flat, result, xp)
        return result

    def _proximal_padded(self, w, step, xp, G, max_sz):
        """Unequal groups: pad, vectorize, unpack."""
        step = min(float(step), 0.9 * (self.a - 1.0))
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
        alpha_g = self.alpha * sqrt_pg_arr
        t_g = alpha_g * step
        a_alpha_g = self.a * alpha_g

        mask_r1 = norms <= alpha_g + t_g
        safe_norms = xp.where(norms > 0.0, norms, xp.ones_like(norms))
        scale_r1 = xp.clip((norms - t_g) / safe_norms, 0.0, None)

        mask_r2 = (norms > alpha_g + t_g) & (norms <= a_alpha_g)
        denom = norms * (self.a - 1.0 - step)
        denom = xp.where(mask_r2, denom, xp.ones_like(denom))
        scale_r2 = ((self.a - 1.0) * norms - self.a * t_g) / denom

        scale = xp.where(mask_r1, scale_r1, 1.0)
        scale = xp.where(mask_r2, scale_r2, scale)

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
        a = self.a

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
        a_alpha_g = a * alpha_g

        # Per-group derivative weight
        if is_torch:
            import torch
            weight_g = torch.where(
                norms <= alpha_g,
                alpha_g,
                torch.where(
                    norms <= a_alpha_g,
                    (a * alpha_g - norms) / (a - 1.0),
                    torch.zeros_like(norms),
                ),
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
                norms <= alpha_g,
                alpha_g,
                cp.where(
                    norms <= a_alpha_g,
                    (a * alpha_g - norms) / (a - 1.0),
                    0.0,
                ),
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
                norms <= alpha_g,
                alpha_g,
                np.where(
                    norms <= a_alpha_g,
                    (a * alpha_g - norms) / (a - 1.0),
                    0.0,
                ),
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
            "a": self.a,
            "n_groups": self._n_groups,
        })
        return params
