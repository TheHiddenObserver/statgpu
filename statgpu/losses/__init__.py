"""
Generic loss functions for statgpu.

This package provides loss types beyond GLM families:
- QuantileLoss: quantile regression (pinball loss)
- HuberLoss: robust regression (Huber M-estimator)
- CoxPartialLikelihoodLoss: survival analysis (Cox PH)

All losses inherit from LossBase and plug into the existing
penalty/solver infrastructure (FISTA, Newton, L-BFGS, ADMM).

Usage:
    from statgpu.losses import QuantileLoss, HuberLoss, get_loss

    loss = QuantileLoss(quantile=0.5)
    loss = get_loss('huber', delta=1.345)
"""

from ._base import LossBase
from ._registry import register_loss, get_loss, list_losses
from ._quantile import QuantileLoss
from ._huber import HuberLoss
from ._bisquare import BisquareLoss
from ._fair import FairLoss
from ._cox_ph import CoxPartialLikelihoodLoss

__all__ = [
    "LossBase",
    "QuantileLoss",
    "HuberLoss",
    "BisquareLoss",
    "FairLoss",
    "CoxPartialLikelihoodLoss",
    "register_loss",
    "get_loss",
    "list_losses",
]
