"""Strict public hyperparameters for Group MCP and Group SCAD."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone

from statgpu.penalties import GroupMCPPenalty, GroupSCADPenalty


GROUPS = [[0, 1], [2, 3]]


@pytest.mark.parametrize(
    "kwargs,error_type,match",
    [
        ({"alpha": True, "gamma": 3.0}, TypeError, "alpha.*numeric"),
        ({"alpha": 0.1, "gamma": True}, TypeError, "gamma.*numeric"),
        ({"alpha": "0.1", "gamma": 3.0}, TypeError, "alpha.*numeric"),
        ({"alpha": 0.1, "gamma": "3"}, TypeError, "gamma.*numeric"),
        ({"alpha": np.nan, "gamma": 3.0}, ValueError, "alpha.*finite"),
        ({"alpha": 0.1, "gamma": np.inf}, ValueError, "gamma.*finite"),
        ({"alpha": 0.0, "gamma": 3.0}, ValueError, "alpha.*positive"),
        ({"alpha": 0.1, "gamma": 1.0}, ValueError, "gamma.*greater"),
    ],
)
def test_group_mcp_rejects_invalid_hyperparameters(kwargs, error_type, match):
    with pytest.raises(error_type, match=match):
        GroupMCPPenalty(groups=GROUPS, **kwargs)


@pytest.mark.parametrize(
    "kwargs,error_type,match",
    [
        ({"alpha": True, "a": 3.7}, TypeError, "alpha.*numeric"),
        ({"alpha": 0.1, "a": True}, TypeError, "a.*numeric"),
        ({"alpha": "0.1", "a": 3.7}, TypeError, "alpha.*numeric"),
        ({"alpha": 0.1, "a": "3.7"}, TypeError, "a.*numeric"),
        ({"alpha": np.nan, "a": 3.7}, ValueError, "alpha.*finite"),
        ({"alpha": 0.1, "a": np.inf}, ValueError, "a.*finite"),
        ({"alpha": 0.0, "a": 3.7}, ValueError, "alpha.*positive"),
        ({"alpha": 0.1, "a": 2.0}, ValueError, "a.*greater"),
    ],
)
def test_group_scad_rejects_invalid_hyperparameters(kwargs, error_type, match):
    with pytest.raises(error_type, match=match):
        GroupSCADPenalty(groups=GROUPS, **kwargs)


@pytest.mark.parametrize(
    "penalty",
    [
        GroupMCPPenalty(alpha=0.1, gamma=3.0, groups=GROUPS),
        GroupSCADPenalty(alpha=0.1, a=3.7, groups=GROUPS),
    ],
)
def test_valid_group_nonconvex_hyperparameters_remain_clone_and_pickle_safe(penalty):
    cloned = clone(penalty)
    restored = pickle.loads(pickle.dumps(penalty))

    assert type(cloned) is type(penalty)
    assert type(restored) is type(penalty)
    assert cloned.get_params(deep=False) == penalty.get_params(deep=False)
    assert restored.get_params(deep=False) == penalty.get_params(deep=False)
