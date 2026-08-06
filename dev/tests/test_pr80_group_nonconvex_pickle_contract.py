"""Legacy pickle/joblib migration for Group MCP and Group SCAD."""

from __future__ import annotations

import io
import pickle

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from statgpu.penalties import GroupMCPPenalty, GroupSCADPenalty


_RAW_GROUPS = [[0, 3], [2, 1]]
_EXPECTED_FLAT = np.array([0, 3, 1, 2], dtype=np.int64)


def _legacy_object(kind):
    if kind == "group_mcp":
        current = GroupMCPPenalty(
            alpha=0.18, gamma=3.0, groups=_RAW_GROUPS
        )
    else:
        current = GroupSCADPenalty(
            alpha=0.18, a=3.7, groups=_RAW_GROUPS
        )
    state = dict(current.__dict__)
    state.pop("groups", None)
    state["_group_indices"] = [
        np.array([0, 3], dtype=np.int64),
        np.array([2, 1], dtype=np.int64),
    ]
    state["_is_contiguous"] = True
    state["_flat_indices"] = None
    state["_all_equal_size"] = True
    state["_group_size_uniform"] = 2

    legacy = object.__new__(type(current))
    legacy.__dict__.update(state)
    return legacy


def _restore_joblib(value):
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    buffer.seek(0)
    return joblib.load(buffer)


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_legacy_group_nonconvex_pickle_and_joblib_rebuild_layout(kind):
    legacy = _legacy_object(kind)
    restored_values = (
        pickle.loads(pickle.dumps(legacy)),
        _restore_joblib(legacy),
    )

    for restored in restored_values:
        assert restored.groups == ((0, 3), (1, 2))
        np.testing.assert_array_equal(
            restored._group_indices[0], np.array([0, 3])
        )
        np.testing.assert_array_equal(
            restored._group_indices[1], np.array([1, 2])
        )
        np.testing.assert_array_equal(restored._flat_indices, _EXPECTED_FLAT)
        assert restored._is_contiguous is False

        cloned = clone(restored)
        assert type(cloned) is type(restored)
        np.testing.assert_array_equal(cloned._flat_indices, _EXPECTED_FLAT)
        coef = np.array([0.15, 0.9, -0.7, 0.05])
        weights = restored.lla_weights(coef)
        cloned_weights = cloned.lla_weights(coef)
        np.testing.assert_allclose(weights, cloned_weights, rtol=0.0, atol=0.0)
        assert weights[0] == pytest.approx(weights[3])
        assert weights[1] == pytest.approx(weights[2])


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_group_nonconvex_constructor_snapshots_mutable_groups(kind):
    groups = [[3, 0], [2, 1]]
    if kind == "group_mcp":
        penalty = GroupMCPPenalty(alpha=0.18, gamma=3.0, groups=groups)
    else:
        penalty = GroupSCADPenalty(alpha=0.18, a=3.7, groups=groups)

    groups[0][:] = [1, 2]
    groups[1][:] = [0, 3]

    assert penalty.groups == ((0, 3), (1, 2))
    np.testing.assert_array_equal(penalty._flat_indices, _EXPECTED_FLAT)
    restored = pickle.loads(pickle.dumps(penalty))
    assert restored.groups == penalty.groups
    np.testing.assert_array_equal(restored._flat_indices, _EXPECTED_FLAT)
