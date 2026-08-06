"""Public documentation contract for CoxPHCV custom penalty grids."""

from __future__ import annotations

import inspect

from statgpu.survival import CoxPHCV


def test_coxphcv_docstring_exposes_custom_grid_order_semantics():
    documentation = inspect.getdoc(CoxPHCV)
    assert documentation is not None
    assert "Custom penalty-grid contract" in documentation
    assert "strongest to weakest" in documentation
    assert "penalty_evaluation_order" in documentation
    assert "caller's original order" in documentation
