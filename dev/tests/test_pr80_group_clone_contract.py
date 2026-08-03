"""Clone and reconstruction contracts for PR #80 group penalties."""

from __future__ import annotations

import io
import pickle

import joblib
import numpy as np
from sklearn.base import clone

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import AdaptiveGroupLassoPenalty, GroupLassoPenalty


_GROUPS = [[3, 0], [2, 1]]


def _assert_legacy_clone_reconstruction(penalty):
    """Emulate sklearn <=1.2 constructor reconstruction and identity gate."""
    params = penalty.get_params(deep=False)
    rebuilt = type(penalty)(**params)
    rebuilt_params = rebuilt.get_params(deep=False)

    assert set(rebuilt_params) == set(params)
    for name, value in params.items():
        assert rebuilt_params[name] is value
    return rebuilt


def _legacy_state_penalty():
    """Create a pickle equivalent to state emitted before the compatibility layer."""
    current = GroupLassoPenalty(alpha=0.07, groups=[[0, 3], [2, 1]])
    state = dict(current.__dict__)
    state.pop("groups", None)
    state["_group_indices"] = [
        np.array([0, 3], dtype=np.int64),
        np.array([2, 1], dtype=np.int64),
    ]
    state["_is_contiguous"] = True
    state["_flat_indices"] = None

    legacy = object.__new__(GroupLassoPenalty)
    legacy.__dict__.update(state)
    return legacy


def test_group_lasso_shallow_params_reconstruct_constructor_exactly():
    penalty = GroupLassoPenalty(alpha=0.07, groups=_GROUPS)
    rebuilt = _assert_legacy_clone_reconstruction(penalty)

    assert type(rebuilt) is GroupLassoPenalty
    assert penalty.groups == ((0, 3), (1, 2))
    np.testing.assert_array_equal(rebuilt._group_indices[0], np.array([0, 3]))
    np.testing.assert_array_equal(rebuilt._group_indices[1], np.array([1, 2]))
    assert rebuilt._is_contiguous is False
    assert "n_groups" not in penalty.get_params(deep=False)
    assert penalty.get_params()["n_groups"] == 2


def test_adaptive_group_lasso_shallow_params_preserve_groups_and_weights():
    weights = np.array([1.0, 1.5])
    penalty = AdaptiveGroupLassoPenalty(
        groups=_GROUPS,
        alpha=0.07,
        weights=weights,
    )
    rebuilt = _assert_legacy_clone_reconstruction(penalty)

    assert isinstance(rebuilt, GroupLassoPenalty)
    assert rebuilt._group_weights == (1.0, 1.5)
    assert rebuilt.get_params(deep=False)["weights"] is penalty.get_params(
        deep=False
    )["weights"]
    np.testing.assert_array_equal(rebuilt._group_indices[0], np.array([0, 3]))
    np.testing.assert_array_equal(rebuilt._group_indices[1], np.array([1, 2]))
    assert rebuilt._is_contiguous is False
    assert "n_groups" not in penalty.get_params(deep=False)
    assert penalty.get_params()["n_groups"] == 2


def test_constructor_snapshots_mutable_groups_and_weights():
    groups = [[3, 0], [2, 1]]
    weights = np.array([1.0, 1.5])
    penalty = AdaptiveGroupLassoPenalty(
        groups=groups,
        alpha=0.07,
        weights=weights,
    )

    groups[0][:] = [1, 2]
    weights[:] = [9.0, 9.0]

    assert penalty.groups == ((0, 3), (1, 2))
    assert penalty._group_weights == (1.0, 1.5)
    restored = pickle.loads(pickle.dumps(penalty))
    assert restored.groups == penalty.groups
    assert restored._group_weights == penalty._group_weights
    np.testing.assert_array_equal(restored._flat_indices, np.array([0, 3, 1, 2]))


def test_modern_sklearn_clone_preserves_public_penalty_types_and_layout():
    group = GroupLassoPenalty(alpha=0.07, groups=_GROUPS)
    adaptive = AdaptiveGroupLassoPenalty(
        groups=_GROUPS,
        alpha=0.07,
        weights=np.array([1.0, 1.5]),
    )

    group_clone = clone(group)
    adaptive_clone = clone(adaptive)

    assert type(group_clone) is GroupLassoPenalty
    assert type(adaptive_clone) is AdaptiveGroupLassoPenalty
    assert group_clone is not group
    assert adaptive_clone is not adaptive
    assert isinstance(adaptive_clone, GroupLassoPenalty)
    np.testing.assert_array_equal(group_clone._flat_indices, np.array([0, 3, 1, 2]))
    np.testing.assert_array_equal(adaptive_clone._flat_indices, np.array([0, 3, 1, 2]))
    np.testing.assert_allclose(adaptive_clone._group_weights, np.array([1.0, 1.5]))


def test_estimator_clone_deep_copies_group_penalty_object():
    penalty = GroupLassoPenalty(alpha=0.07, groups=_GROUPS)
    estimator = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=penalty,
        alpha=0.07,
        solver="auto",
        fit_intercept=True,
        compute_inference=False,
    )

    cloned = clone(estimator)

    assert type(cloned.penalty) is GroupLassoPenalty
    assert cloned.penalty is not penalty
    np.testing.assert_array_equal(cloned.penalty._flat_indices, np.array([0, 3, 1, 2]))
    assert cloned.penalty._is_contiguous is False


def test_legacy_pickle_and_joblib_state_is_cloneable_after_migration():
    legacy = _legacy_state_penalty()
    restored_pickle = pickle.loads(pickle.dumps(legacy))

    buffer = io.BytesIO()
    joblib.dump(legacy, buffer)
    buffer.seek(0)
    restored_joblib = joblib.load(buffer)

    for restored in (restored_pickle, restored_joblib):
        assert type(restored) is GroupLassoPenalty
        assert restored.groups == ((0, 3), (1, 2))
        np.testing.assert_array_equal(restored._group_indices[0], np.array([0, 3]))
        np.testing.assert_array_equal(restored._group_indices[1], np.array([1, 2]))
        np.testing.assert_array_equal(restored._flat_indices, np.array([0, 3, 1, 2]))
        assert restored._is_contiguous is False

        rebuilt = _assert_legacy_clone_reconstruction(restored)
        cloned = clone(restored)
        assert type(rebuilt) is GroupLassoPenalty
        assert type(cloned) is GroupLassoPenalty
        np.testing.assert_array_equal(cloned._flat_indices, np.array([0, 3, 1, 2]))
