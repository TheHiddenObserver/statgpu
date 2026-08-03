"""Adaptive Group Lasso contributes no smooth Lipschitz curvature."""

from __future__ import annotations

from statgpu.penalties import AdaptiveGroupLassoPenalty
from statgpu.solvers import _fista, _fista_bb, _utils


def test_adaptive_group_lasso_has_zero_smooth_lipschitz_in_all_fista_bindings():
    penalty = AdaptiveGroupLassoPenalty(
        groups=[[0, 1], [2, 3]],
        alpha=7.5,
        weights=[0.6, 1.4],
    )

    assert _utils._smooth_penalty_lipschitz(penalty) == 0.0
    assert _fista._smooth_penalty_lipschitz(penalty) == 0.0
    assert _fista_bb._smooth_penalty_lipschitz(penalty) == 0.0
    assert getattr(
        _utils._smooth_penalty_lipschitz,
        "_statgpu_adaptive_group_contract",
        False,
    )


def test_l2_and_elasticnet_smooth_curvature_remain_unchanged():
    from statgpu.penalties import ElasticNetPenalty, L2Penalty

    l2 = L2Penalty(alpha=0.8)
    en = ElasticNetPenalty(alpha=0.8, l1_ratio=0.25)

    assert _utils._smooth_penalty_lipschitz(l2) == 0.8
    assert _utils._smooth_penalty_lipschitz(en) == 0.8 * 0.75
