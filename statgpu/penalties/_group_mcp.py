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
from ._group_lasso import _vector_norm, _to_backend_array, _backend_zeros

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
            denom = norms * (1.0 - step / gamma) + 1e-12
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
                [np.asarray(g, dtype=int) for g in self._group_indices]
            )

    def _reshape_to_matrix(self, w, xp, G, gs):
        """Reshape w into (G, gs) matrix, handling non-contiguous layouts."""
        if self._is_contiguous:
            return w.reshape(G, gs)
        return w[self._flat_indices].reshape(G, gs)

    def _scatter_from_flat(self, flat_vals, result, xp):
        """Scatter flat values back, handling non-contiguous layouts."""
        if self._is_contiguous:
            result[:] = flat_vals
        else:
            result[self._flat_indices] = flat_vals

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef: np.ndarray) -> float:
        if self._group_indices is None:
            raise ValueError("groups must be set before calling value()")

        total = 0.0
        for g, idx in enumerate(self._group_indices):
            w_g = coef[idx]
            ng = float(np.linalg.norm(w_g))
            alpha_g = self.alpha * self._sqrt_pg[g]
            gamma_alpha_g = self.gamma * alpha_g

            if ng <= gamma_alpha_g:
                total += alpha_g * ng - (ng * ng) / (2.0 * self.gamma)
            else:
                total += 0.5 * self.gamma * alpha_g * alpha_g
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
            ng = float(np.linalg.norm(w_g))
            alpha_g = self.alpha * self._sqrt_pg[g]

            if ng > 0 and ng <= self.gamma * alpha_g:
                deriv = alpha_g - ng / self.gamma
                grad[idx] = deriv * w_g / ng
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
        w_mat = self._reshape_to_matrix(w, xp, G, gs)
        sqrt_pg_arr = _to_backend_array(self._sqrt_pg, xp, w)

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

        denom = norms * (1.0 - step / self.gamma) + 1e-12
        scale_shrink = (norms - t_g) / denom                    # (G,)
        scale = xp.where(mask_shrink, scale_shrink, 1.0)        # (G,)
        scale = xp.where(mask_zero, 0.0, scale)

        scaled_flat = (w_mat * scale[:, None]).reshape(-1)
        result = w.copy() if hasattr(w, 'copy') else w.clone()
        self._scatter_from_flat(scaled_flat, result, xp)
        return result

    def _proximal_padded(self, w, step, xp, G, max_sz):
        """Unequal groups: pad, vectorize, unpack."""
        sizes = self._group_sizes
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

        norms = _vector_norm(padded, xp, dim=1)
        sqrt_pg_arr = _to_backend_array(self._sqrt_pg, xp, w)
        t_g = self.alpha * sqrt_pg_arr * step
        gamma_alpha_g = self.gamma * self.alpha * sqrt_pg_arr

        mask_zero = norms <= t_g
        mask_shrink = (norms > t_g) & (norms <= gamma_alpha_g)
        denom = norms * (1.0 - step / self.gamma) + 1e-12
        scale_shrink = (norms - t_g) / denom
        scale = xp.where(mask_shrink, scale_shrink, 1.0)
        scale = xp.where(mask_zero, 0.0, scale)

        padded_scaled = padded * scale[:, None]

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
    # LLA weights (for LLA outer loop optimization)
    # ----------------------------------------------------------------

    def lla_weights(self, coef: np.ndarray) -> np.ndarray:
        if self._group_indices is None:
            raise ValueError("groups must be set before calling lla_weights()")

        weights = np.zeros(len(coef), dtype=float)
        for g, idx in enumerate(self._group_indices):
            w_g = coef[idx]
            ng = float(np.linalg.norm(w_g))
            alpha_g = self.alpha * self._sqrt_pg[g]
            gamma_alpha_g = self.gamma * alpha_g

            if ng <= gamma_alpha_g:
                weight_g = alpha_g - ng / self.gamma
                weights[idx] = max(weight_g, 0.0)
            else:
                weights[idx] = 0.0
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
