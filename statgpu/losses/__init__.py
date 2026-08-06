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
from ._robust_base import RobustLossBase
from ._registry import register_loss, get_loss, list_losses
from ._quantile import QuantileLoss
from ._huber import HuberLoss
from ._bisquare import BisquareLoss
from ._fair import FairLoss


def __getattr__(name):
    """Load survival losses only when their public symbol is requested.

    ``glm_core._base`` inherits from ``losses._base``. Eagerly importing the
    Cox loss while that base module is still initializing enters the
    ``survival`` package, which imports model code that depends on GLM losses.
    Keeping the Cox export lazy removes that package-initialization cycle while
    preserving ``from statgpu.losses import CoxPartialLikelihoodLoss``.
    """
    if name == "CoxPartialLikelihoodLoss":
        from ._cox_ph import CoxPartialLikelihoodLoss

        globals()[name] = CoxPartialLikelihoodLoss
        return CoxPartialLikelihoodLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "LossBase",
    "RobustLossBase",
    "QuantileLoss",
    "HuberLoss",
    "BisquareLoss",
    "FairLoss",
    "CoxPartialLikelihoodLoss",
    "register_loss",
    "get_loss",
    "list_losses",
]
