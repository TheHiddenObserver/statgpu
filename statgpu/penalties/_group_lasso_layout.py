"""Canonical public Group Lasso penalty boundary.

Explicit nested group specifications may list members within a group in any
order. Group penalties are invariant to these within-group permutations, but
optimized solver paths rely on truthful contiguous-layout metadata. This module
keeps the historical public import and pickle path while ensuring that new
objects, legacy serialized state, sklearn reconstruction, and adaptive weighted
objectives all share one canonical layout contract.
"""

from __future__ import annotations

import numpy as np

from . import _group_lasso as _group_lasso_impl


_BaseGroupLassoPenalty = _group_lasso_impl.GroupLassoPenalty
_BaseAdaptiveGroupLassoPenalty = _group_lasso_impl.AdaptiveGroupLassoPenalty


def _normalize_groups_parameter(groups):
    """Create an immutable clone-safe snapshot of a public groups argument."""
    if groups is None:
        return None
    if isinstance(groups, np.ndarray):
        if groups.ndim != 1:
            return groups
        return tuple(int(value) for value in groups.tolist())
    if not isinstance(groups, (list, tuple)):
        return groups
    if len(groups) == 0:
        return groups if isinstance(groups, tuple) else tuple()

    first = groups[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        already_normalized = isinstance(groups, tuple) and all(
            isinstance(group, tuple)
            and all(type(index) is int for index in group)
            and tuple(sorted(group)) == group
            for group in groups
        )
        if already_normalized:
            return groups
        return tuple(
            tuple(sorted(int(index) for index in group)) for group in groups
        )

    if isinstance(groups, tuple) and all(type(value) is int for value in groups):
        return groups
    return tuple(int(value) for value in groups)


def _canonicalize_nested_groups(groups):
    """Convert immutable explicit groups to the base implementation format."""
    if not isinstance(groups, (list, tuple)) or not groups:
        return groups
    first = groups[0]
    if not isinstance(first, (list, tuple, np.ndarray)):
        return groups
    return [np.asarray(group, dtype=int) for group in groups]


def _weights_to_numpy(weights):
    """Convert supported host/device weight arrays for validation only."""
    if weights is None:
        return None
    module = type(weights).__module__
    if module.startswith("torch"):
        return weights.detach().cpu().numpy()
    if module.startswith("cupy"):
        return weights.get()
    return np.asarray(weights)


def _normalize_weights_parameter(weights, n_groups):
    """Validate and snapshot adaptive weights as an immutable float tuple."""
    if weights is None:
        return None
    try:
        values = np.asarray(_weights_to_numpy(weights), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("group weights must be a one-dimensional numeric array") from exc
    if values.ndim != 1 or values.shape[0] != n_groups:
        raise ValueError(
            f"group weights must have shape ({n_groups},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("group weights must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("group weights must be non-negative")
    if isinstance(weights, tuple) and all(type(value) is float for value in weights):
        return weights
    return tuple(float(value) for value in values)


class GroupLassoPenalty(_BaseGroupLassoPenalty):
    """Group Lasso with canonical layout and clone-safe constructor state.

    ``groups`` is stored as an immutable normalized tuple. This prevents later
    mutation of a caller-owned list/array from changing clone or pickle state
    without changing the already-built numerical layout. A normalized tuple
    received from sklearn reconstruction is retained by identity for the
    sklearn <=1.2 constructor-identity gate.

    ``__setstate__`` intentionally rebuilds all derived layout metadata. This
    migrates objects serialized by versions that preserved unsorted nested
    groups or stale contiguity flags.
    """

    def __init__(self, alpha: float = 1.0, groups=None):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super().__init__(alpha=alpha, groups=normalized_groups)

    def _init_groups(self, groups):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super()._init_groups(_canonicalize_nested_groups(normalized_groups))

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise TypeError("GroupLassoPenalty pickle state must be a dict")
        self.__dict__.update(state)
        groups = state.get("groups", state.get("_group_indices"))
        self.groups = _normalize_groups_parameter(groups)
        if groups is not None:
            # Re-parse rather than trusting serialized derived fields such as
            # _is_contiguous, _flat_indices, padded indices, or device caches.
            self._init_groups(groups)

    def get_params(self, deep: bool = True) -> dict:
        """Return descriptive state or constructor-only clone parameters."""
        if not deep:
            return {"alpha": self.alpha, "groups": self.groups}
        # Preserve the historical descriptive serialization contract.
        return _BaseGroupLassoPenalty.get_params(self)


class AdaptiveGroupLassoPenalty(
    _BaseAdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
):
    """Weighted Group Lasso preserving the public Group Lasso hierarchy."""

    def __init__(self, groups, alpha=1.0, weights=None):
        # Let the original adaptive implementation establish the cooperative
        # MRO and canonical group layout, then validate/invalidate weight state.
        super().__init__(groups=groups, alpha=alpha, weights=None)
        self.set_weights(weights)

    def set_weights(self, weights):
        """Update validated per-group weights and invalidate device caches."""
        self._group_weights = _normalize_weights_parameter(
            weights, self._n_groups
        )
        self._group_weights_torch = None
        self._group_weights_cupy = None

    def __setstate__(self, state):
        weights = state.get("_group_weights", state.get("weights"))
        super().__setstate__(state)
        # Never retain serialized device tensors from another process/device.
        self.set_weights(weights)

    def _get_group_weights(self, xp, w):
        """Return weights on the requested backend without cross-backend cache reuse."""
        if self._group_weights is None:
            return None
        if xp.__name__ == "numpy":
            return np.asarray(self._group_weights, dtype=w.dtype)
        if xp.__name__ == "torch":
            cached = getattr(self, "_group_weights_torch", None)
            if cached is None or cached.device != w.device or cached.dtype != w.dtype:
                cached = _group_lasso_impl._to_backend_array(
                    self._group_weights, xp, w
                ).to(dtype=w.dtype)
                self._group_weights_torch = cached
            return cached

        cached = getattr(self, "_group_weights_cupy", None)
        same_device = (
            cached is not None
            and getattr(cached, "device", None) is not None
            and getattr(w, "device", None) is not None
            and int(cached.device.id) == int(w.device.id)
        )
        if cached is None or not same_device or cached.dtype != w.dtype:
            cached = _group_lasso_impl._to_backend_array(
                self._group_weights, xp, w
            ).astype(w.dtype, copy=False)
            self._group_weights_cupy = cached
        return cached

    def _weighted_group_components(self, coef):
        """Return backend module, feature view, norms, sqrt sizes, and weights."""
        if self._group_indices is None:
            raise ValueError("groups must be set before evaluating the penalty")
        xp = _group_lasso_impl._get_xp(coef)
        p_total = int(self._group_sizes.sum())
        coef_feat = coef[:p_total]
        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            if self._is_contiguous:
                grouped = coef_feat.reshape(self._n_groups, gs)
            else:
                grouped = coef_feat[self._flat_indices].reshape(
                    self._n_groups, gs
                )
            norms = _group_lasso_impl._vector_norm(grouped, xp, dim=1)
        else:
            norms = self._batched_group_norms_vec(coef_feat, xp, coef)
        sqrt_pg = self._get_sqrt_pg(xp, coef)
        weights = self._get_group_weights(xp, coef)
        if weights is None:
            weights = xp.ones(self._n_groups, dtype=coef.dtype)
            if xp.__name__ == "torch":
                weights = weights.to(device=coef.device)
        return xp, coef_feat, norms, sqrt_pg, weights

    def value(self, coef) -> float:
        """Evaluate the weighted Group Lasso objective consistently with prox."""
        xp, _, norms, sqrt_pg, weights = self._weighted_group_components(coef)
        total = xp.sum(self.alpha * weights * sqrt_pg * norms)
        if xp.__name__ == "torch":
            return total.item()
        return float(total)

    def gradient(self, coef):
        """Return a weighted group subgradient, with zero at zero-norm groups."""
        xp, coef_feat, norms, sqrt_pg, weights = self._weighted_group_components(
            coef
        )
        if xp.__name__ == "torch":
            safe_norms = xp.clamp(norms, min=1e-15)
        else:
            safe_norms = xp.maximum(norms, 1e-15)
        scale_g = xp.where(
            norms > 1e-15,
            self.alpha * weights * sqrt_pg / safe_norms,
            0.0,
        )
        grad = xp.zeros_like(coef)
        if self._all_equal_size and self._group_size_uniform is not None:
            gs = self._group_size_uniform
            if self._is_contiguous:
                grouped = coef_feat.reshape(self._n_groups, gs)
            else:
                grouped = coef_feat[self._flat_indices].reshape(
                    self._n_groups, gs
                )
            grad_grouped = grouped * scale_g[:, None]
            if self._is_contiguous:
                grad[: coef_feat.shape[0]] = grad_grouped.reshape(-1)
            else:
                grad[self._flat_indices] = grad_grouped.reshape(-1)
            return grad

        feat_idx = self._get_cached("_group_feat_idx", xp, coef)
        grad[: coef_feat.shape[0]] = scale_g[feat_idx] * coef_feat
        return grad

    def get_params(self, deep: bool = True) -> dict:
        """Return descriptive state or constructor-only clone parameters."""
        if not deep:
            return {
                "groups": self.groups,
                "alpha": self.alpha,
                "weights": self._group_weights,
            }
        return _BaseAdaptiveGroupLassoPenalty.get_params(self)


# Preserve historical import/pickle paths and ensure direct imports from
# ``statgpu.penalties._group_lasso`` resolve to the same public classes after
# package initialization. Rebinding both classes keeps
# ``issubclass(AdaptiveGroupLassoPenalty, GroupLassoPenalty)`` true.
GroupLassoPenalty.__module__ = _group_lasso_impl.__name__
AdaptiveGroupLassoPenalty.__module__ = _group_lasso_impl.__name__
_group_lasso_impl.GroupLassoPenalty = GroupLassoPenalty
_group_lasso_impl.AdaptiveGroupLassoPenalty = AdaptiveGroupLassoPenalty
