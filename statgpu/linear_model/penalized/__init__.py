"""Penalized GLM models (split via mixin pattern)."""

from ._base import PenalizedGeneralizedLinearModel, SelectivePenalty

__all__ = ["PenalizedGeneralizedLinearModel", "SelectivePenalty"]
