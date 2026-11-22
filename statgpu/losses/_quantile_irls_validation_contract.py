"""Compatibility imports for the class-owned Quantile IRLS validation boundary."""

from __future__ import annotations

from ._quantile import QuantileLoss, _validate_quantile_xy_shapes


_MARKER = "_statgpu_quantile_irls_weight_validation_contract"


def install_quantile_irls_validation_contract() -> None:
    """No-op: ``QuantileLoss.irls`` owns validation directly."""


__all__ = ["install_quantile_irls_validation_contract"]
