"""Library ``Penalty.clone()`` contracts for all public group penalties."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.penalties import (
    AdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
)


_PENALTIES = [
    pytest.param(
        GroupLassoPenalty(alpha=0.1, groups=[[3, 0], [2, 1]]),
        id="group-lasso",
    ),
    pytest.param(
        AdaptiveGroupLassoPenalty(
            alpha=0.1,
            groups=[[3, 0], [2, 1]],
            weights=[0.5, 1.5],
        ),
        id="adaptive-group-lasso",
    ),
    pytest.param(
        GroupMCPPenalty(
            alpha=0.1,
            gamma=3.0,
            groups=[[3, 0], [2, 1]],
        ),
        id="group-mcp",
    ),
    pytest.param(
        GroupSCADPenalty(
            alpha=0.1,
            a=3.7,
            groups=[[3, 0], [2, 1]],
        ),
        id="group-scad",
    ),
]


@pytest.mark.parametrize("penalty", _PENALTIES)
def test_library_clone_reconstructs_group_penalty_from_constructor_params(penalty):
    cloned = penalty.clone()

    assert type(cloned) is type(penalty)
    assert cloned is not penalty
    assert cloned.groups == penalty.groups == ((0, 3), (1, 2))
    np.testing.assert_array_equal(cloned._flat_indices, penalty._flat_indices)
    assert cloned._is_contiguous is False
    assert cloned.get_params(deep=False) == penalty.get_params(deep=False)
    if isinstance(penalty, AdaptiveGroupLassoPenalty):
        assert cloned._group_weights == penalty._group_weights == (0.5, 1.5)


def test_library_clone_does_not_share_mutable_device_caches():
    penalty = AdaptiveGroupLassoPenalty(
        alpha=0.1,
        groups=[[0, 3], [1, 2]],
        weights=[0.5, 1.5],
    )
    penalty._group_weights_torch = object()
    penalty._group_weights_cupy = object()

    cloned = penalty.clone()

    assert cloned._group_weights_torch is None
    assert cloned._group_weights_cupy is None
