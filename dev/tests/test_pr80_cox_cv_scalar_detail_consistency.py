"""Scalar and detailed Cox CV selector calls must choose identically."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.survival import _cox_cv as cox_cv
from statgpu.survival import _cox_cv_penalty_order_contract as contract


def test_scalar_selector_uses_the_same_stronger_near_tie_policy(monkeypatch):
    calls = []

    def fake_selector(*args, **kwargs):
        calls.append(dict(kwargs))
        np.testing.assert_array_equal(kwargs["penalties"], [1.0, 0.5, 0.1])
        assert kwargs["return_details"] is True
        return 0.5, {
            "penalty": 0.5,
            "best_pl": 5.0 + 1e-11,
            "mean_pl": np.array([5.0, 5.0 + 1e-11, 4.0]),
            "candidate_complete": np.array([True, True, True]),
        }

    monkeypatch.setattr(
        contract,
        "_ORIGINAL_SELECT_COXPH_PENALTY_CV",
        fake_selector,
    )
    selected = cox_cv._select_coxph_penalty_cv(
        np.zeros((4, 1)),
        np.arange(1.0, 5.0),
        np.array([1.0, 0.0, 1.0, 0.0]),
        penalties=np.array([0.1, 1.0, 0.5]),
        return_details=False,
    )

    assert len(calls) == 1
    assert selected == pytest.approx(1.0)
