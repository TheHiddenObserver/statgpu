"""Canonical public Group Lasso penalty boundary.

Explicit nested group specifications may list members within a group in any
order. Group penalties are invariant to these within-group permutations, but
optimized solver paths rely on truthful contiguous-layout metadata. This module
keeps the historical public import and pickle path while ensuring that new
objects, legacy serialized state, sklearn reconstruction, strict public input
validation, feature coverage, and adaptive weighted objectives share one
canonical contract.
"""

from __future__ import annotations

from numbers import Integral, Real
import warnings

import numpy as np

from . import _group_lasso as _group_lasso_impl


_BaseGroupLassoPenalty = _group_lasso_impl.GroupLassoPenalty
_BaseAdaptiveGroupLassoPenalty = _group_lasso_impl.AdaptiveGroupLassoPenalty


def _normalize_group_alpha(alpha):
    """Validate convex group-penalty strength without lossy coercion."""
    if isinstance(alpha, (bool, np.bool_)):
        raise TypeError("alpha must be a finite non-negative numeric scalar")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "alpha must be a finite non-negative numeric scalar"
        ) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("alpha must be a finite non-negative scalar")
    return value


def _coerce_group_integer(value, *, label):
    """Accept integer scalars and exact finite integer-valued reals only."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be integer-valued, not boolean")
    if isinstance(value, (Integral, np.integer)):
        return int(value)
    if isinstance(value, (Real, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{label} must be finite")
        if not numeric.is_integer():
            raise ValueError(f"{label} must be integer-valued")
        return int(numeric)
    raise TypeError(f"{label} must be an integer-valued numeric scalar")


def _normalize_groups_parameter(groups):
    """Validate and create an immutable clone-safe groups snapshot."""
    if groups is None:
        return None
    if isinstance(groups, np.ndarray):
        if groups.ndim != 1:
            raise ValueError(
                "groups arrays must be one-dimensional; use a list of lists "
                "for explicit feature-index groups"
            )
        groups = groups.tolist()
    if not isinstance(groups, (list, tuple)):
        raise TypeError(
            f"groups must be a one-dimensional array, list, or tuple, got "
            f"{type(groups).__name__}"
        )
    if len(groups) == 0:
        raise ValueError("groups must not be empty")

    nested_flags = [
        isinstance(group, (list, tuple, np.ndarray)) for group in groups
    ]
    if any(nested_flags) and not all(nested_flags):
        raise TypeError(
            "groups must be either a flat group-ID sequence or a nested "
            "sequence of feature-index groups"
        )

    if all(nested_flags):
        normalized = []
        all_indices = []
        already_normalized = isinstance(groups, tuple)
        for group_id, group in enumerate(groups):
            if isinstance(group, np.ndarray):
                if group.ndim != 1:
                    raise ValueError(
                        f"groups[{group_id}] must be one-dimensional"
                    )
                group = group.tolist()
            if len(group) == 0:
                raise ValueError("explicit groups must not contain empty groups")
            indices = tuple(
                sorted(
                    _coerce_group_integer(
                        index, label=f"groups[{group_id}] index"
                    )
                    for index in group
                )
            )
            if any(index < 0 for index in indices):
                raise ValueError("feature indices in groups must be non-negative")
            normalized.append(indices)
            all_indices.extend(indices)
            already_normalized = already_normalized and isinstance(group, tuple)
            already_normalized = already_normalized and group == indices
            already_normalized = already_normalized and all(
                type(index) is int for index in group
            )
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("groups contain duplicate feature indices")
        if already_normalized:
            return groups
        return tuple(normalized)

    group_ids = tuple(
        _coerce_group_integer(value, label="group ID") for value in groups
    )
    if any(value < 0 for value in group_ids):
        raise ValueError("group IDs must be non-negative")
    observed = sorted(set(group_ids))
    expected = list(range(observed[-1] + 1))
    if observed != expected:
        raise ValueError(
            "group IDs must be contiguous and start at zero; "
            f"observed {observed}"
        )
    if isinstance(groups, tuple) and all(type(value) is int for value in groups):
        return groups
    return group_ids


def _canonicalize_nested_groups(groups):
    """Convert immutable explicit groups to the base implementation format."""
    if not isinstance(groups, (list, tuple)) or not groups:
        return groups
    first = groups[0]
    if not isinstance(first, (list, tuple, np.ndarray)):
        return groups
    return [np.asarray(group, dtype=int) for group in groups]


def _canonical_internal_groups(penalty):
    return tuple(
        tuple(int(index) for index in np.asarray(group, dtype=np.int64))
        for group in penalty._group_indices
    )


def _sync_groups_snapshot_after_base_init(penalty, normalized_groups):
    """Retain clone identity unless base auto-fill changed explicit groups."""
    if normalized_groups is None:
        penalty.groups = None
        return
    is_explicit = isinstance(normalized_groups[0], tuple)
    if is_explicit:
        internal = _canonical_internal_groups(penalty)
        penalty.groups = (
            normalized_groups if internal == normalized_groups else internal
        )
    else:
        penalty.groups = normalized_groups


def _validate_group_feature_coverage(penalty, n_features):
    """Make group coverage solver-independent once the design width is known."""
    if isinstance(n_features, (bool, np.bool_)):
        raise TypeError("n_features must be a positive integer")
    try:
        n_features = int(n_features)
    except (TypeError, ValueError) as exc:
        raise TypeError("n_features must be a positive integer") from exc
    if n_features < 1:
        raise ValueError("n_features must be a positive integer")
    if penalty._group_indices is None:
        raise ValueError("groups must be set before fitting a group penalty")

    flat = np.concatenate(
        [np.asarray(group, dtype=np.int64) for group in penalty._group_indices]
    )
    if flat.size == 0:
        raise ValueError("groups must contain at least one feature index")
    if int(flat.max()) >= n_features:
        raise ValueError(
            "groups contain a feature index outside the design matrix: "
            f"max index {int(flat.max())}, n_features={n_features}"
        )
    missing = sorted(set(range(n_features)) - set(flat.tolist()))
    if not missing:
        return penalty

    existing_weights = getattr(penalty, "_group_weights", None)
    if existing_weights is not None:
        raise ValueError(
            "adaptive group weights require groups to cover every design "
            f"feature; missing indices {missing}"
        )

    warnings.warn(
        f"Groups do not cover design features {missing}. Auto-adding "
        f"{len(missing)} single-feature groups.",
        UserWarning,
        stacklevel=3,
    )
    completed = list(_canonical_internal_groups(penalty))
    completed.extend((index,) for index in missing)
    penalty._init_groups(tuple(completed))
    return penalty


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
    raw = np.asarray(_weights_to_numpy(weights))
    if raw.dtype.kind in ("b", "S", "U"):
        raise TypeError("group weights must be a one-dimensional numeric array")
    if raw.dtype.kind == "O":
        for value in raw.ravel():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (Real, np.number)
            ):
                raise TypeError(
                    "group weights must be a one-dimensional numeric array"
                )
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "group weights must be a one-dimensional numeric array"
        ) from exc
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
    """Group Lasso with canonical layout and clone-safe constructor state."""

    def __init__(self, alpha: float = 1.0, groups=None):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super().__init__(
            alpha=_normalize_group_alpha(alpha), groups=normalized_groups
        )

    def _init_groups(self, groups):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super()._init_groups(_canonicalize_nested_groups(normalized_groups))
        _sync_groups_snapshot_after_base_init(self, normalized_groups)

    def validate_n_features(self, n_features):
        return _validate_group_feature_coverage(self, n_features)

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise TypeError("GroupLassoPenalty pickle state must be a dict")
        self.__dict__.update(state)
        self.alpha = _normalize_group_alpha(state.get("alpha", self.alpha))
        groups = state.get("groups", state.get("_group_indices"))
        self.groups = _normalize_groups_parameter(groups)
        if groups is not None:
            self._init_groups(groups)

    def get_params(self, deep: bool = True) -> dict:
        if not deep:
            return {"alpha": self.alpha, "groups": self.groups}
        return _BaseGroupLassoPenalty.get_params(self)


class AdaptiveGroupLassoPenalty(
    _BaseAdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
):
    """Weighted Group Lasso preserving the public Group Lasso hierarchy."""

    def __init__(self, groups, alpha=1.0, weights=None):
        super().__init__(groups=groups, alpha=alpha, weights=None)
        self.set_weights(weights)

    def set_weights(self, weights):
        self._group_weights = _normalize_weights_parameter(
            weights, self._n_groups
        )
        self._group_weights_torch = None
        self._group_weights_cupy = None

    def __setstate__(self, state):
        weights = state.get("_group_weights", state.get("weights"))
        super().__setstate__(state)
        self.set_weights(weights)

    def _get_group_weights(self, xp, w):
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
        xp, _, norms, sqrt_pg, weights = self._weighted_group_components(coef)
        total = xp.sum(self.alpha * weights * sqrt_pg * norms)
        if xp.__name__ == "torch":
            return total.item()
        return float(total)

    def gradient(self, coef):
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
        if not deep:
            return {
                "groups": self.groups,
                "alpha": self.alpha,
                "weights": self._group_weights,
            }
        return _BaseAdaptiveGroupLassoPenalty.get_params(self)


GroupLassoPenalty.__module__ = _group_lasso_impl.__name__
AdaptiveGroupLassoPenalty.__module__ = _group_lasso_impl.__name__
_group_lasso_impl.GroupLassoPenalty = GroupLassoPenalty
_group_lasso_impl.AdaptiveGroupLassoPenalty = AdaptiveGroupLassoPenalty
