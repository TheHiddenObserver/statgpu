"""Exact Group MCP/SCAD LLA surrogate scaling contracts for PR #80."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.penalties import GroupMCPPenalty, GroupSCADPenalty
from statgpu.solvers import fista_lla_path
from statgpu.solvers import _fista_lla_group_contract as group_contract
from statgpu.solvers._fista_lla_group_contract import _group_surrogate_factory


_GROUPS = [[0, 3], [1, 2]]


def _expected_surrogate_value(coef, derivatives):
    total = 0.0
    for group in _GROUPS:
        idx = np.asarray(group, dtype=np.int64)
        total += float(derivatives[idx[0]]) * np.linalg.norm(coef[idx])
    return total


@pytest.mark.parametrize(
    "penalty",
    [
        pytest.param(
            GroupMCPPenalty(alpha=0.18, gamma=3.0, groups=_GROUPS),
            id="group-mcp",
        ),
        pytest.param(
            GroupSCADPenalty(alpha=0.18, a=3.7, groups=_GROUPS),
            id="group-scad",
        ),
    ],
)
def test_group_surrogate_factory_matches_exact_linearized_penalty(penalty):
    derivatives = np.array([0.4, 1.2, 1.2, 0.4])
    coef = np.array([0.8, -0.3, 0.5, 0.6])
    inner = _group_surrogate_factory(penalty)(derivatives)

    assert inner.alpha == pytest.approx(1.0)
    np.testing.assert_allclose(
        inner._group_weights,
        np.array([0.4, 1.2]) / np.sqrt(2.0),
        rtol=0.0,
        atol=1e-15,
    )
    assert inner.value(coef) == pytest.approx(
        _expected_surrogate_value(coef, derivatives),
        rel=0.0,
        abs=1e-14,
    )


@pytest.mark.parametrize("target_alpha", [0.03, 0.18, 0.7])
def test_group_surrogate_scaling_does_not_multiply_target_alpha_again(target_alpha):
    penalty = GroupMCPPenalty(
        alpha=target_alpha,
        gamma=3.0,
        groups=_GROUPS,
    )
    derivatives = np.array([0.25, 0.9, 0.9, 0.25])
    coef = np.array([0.4, -0.2, 0.7, 0.1])

    inner = _group_surrogate_factory(penalty)(derivatives)

    assert inner.alpha == pytest.approx(1.0)
    assert inner.value(coef) == pytest.approx(
        _expected_surrogate_value(coef, derivatives),
        rel=0.0,
        abs=1e-14,
    )


def test_group_surrogate_factory_rejects_mixed_derivatives_within_group():
    penalty = GroupSCADPenalty(alpha=0.18, a=3.7, groups=_GROUPS)
    factory = _group_surrogate_factory(penalty)

    with pytest.raises(ValueError, match="constant within each group"):
        factory(np.array([0.4, 1.2, 1.2, 0.5]))


def test_group_surrogate_factory_rejects_negative_or_nonfinite_derivatives():
    penalty = GroupMCPPenalty(alpha=0.18, gamma=3.0, groups=_GROUPS)
    factory = _group_surrogate_factory(penalty)

    with pytest.raises(ValueError, match="non-negative"):
        factory(np.array([-0.1, 1.2, 1.2, -0.1]))
    with pytest.raises(FloatingPointError, match="finite"):
        factory(np.array([0.4, np.nan, np.nan, 0.4]))


def test_direct_group_solver_call_installs_group_surrogate_without_factory(monkeypatch):
    penalty = GroupSCADPenalty(alpha=0.18, a=3.7, groups=_GROUPS)
    derivatives = np.array([0.4, 1.2, 1.2, 0.4])
    coef = np.array([0.8, -0.3, 0.5, 0.6])
    captured = {}

    def fake_base(*args, **kwargs):
        factory = kwargs["lla_penalty_factory"]
        inner = factory(derivatives)
        captured["inner"] = inner
        return "sentinel"

    monkeypatch.setattr(group_contract, "_base_fista_lla_path", fake_base)
    result = group_contract.fista_lla_path(
        loss=object(),
        scad_penalty=penalty,
        X=np.zeros((2, 4)),
        y=np.zeros(2),
        alpha_path=[0.18],
        lla_penalty_factory=None,
    )

    assert result == "sentinel"
    assert captured["inner"].value(coef) == pytest.approx(
        _expected_surrogate_value(coef, derivatives),
        rel=0.0,
        abs=1e-14,
    )


def test_non_group_solver_call_preserves_caller_factory(monkeypatch):
    class DummyPenalty:
        name = "mcp"

    sentinel_factory = object()
    captured = {}

    def fake_base(*args, **kwargs):
        captured["factory"] = kwargs["lla_penalty_factory"]
        return "sentinel"

    monkeypatch.setattr(group_contract, "_base_fista_lla_path", fake_base)
    result = group_contract.fista_lla_path(
        loss=object(),
        scad_penalty=DummyPenalty(),
        X=np.zeros((2, 1)),
        y=np.zeros(2),
        alpha_path=[0.18],
        lla_penalty_factory=sentinel_factory,
    )

    assert result == "sentinel"
    assert captured["factory"] is sentinel_factory


def test_public_solver_export_uses_group_contract_wrapper():
    assert fista_lla_path.__module__ == "statgpu.solvers._fista_lla_group_contract"
