"""
Penalized GLM estimators.

This module re-exports from the ``penalized`` package (split via mixin pattern).
The original monolithic implementation has been refactored into:
- ``penalized/_base.py``: SelectivePenalty + PenalizedGeneralizedLinearModel core
- ``penalized/_fit_mixin.py``: fit/solver methods
- ``penalized/_inference_mixin.py``: inference methods
- ``penalized/_predict_mixin.py``: predict/score methods
"""

from __future__ import annotations

__all__ = ["PenalizedGeneralizedLinearModel", "PenalizedLinearRegression", "PenalizedLogisticRegression", "PenalizedPoissonRegression"]

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel, SelectivePenalty

# Re-export for backward compatibility
from statgpu.linear_model.penalized._base import _get_selective_penalty_singleton
