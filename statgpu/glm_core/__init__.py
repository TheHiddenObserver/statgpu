"""
GLM core utilities for statgpu.

Usage:
    from statgpu.glm_core import get_glm_loss, register_glm_loss

    # Built-in
    loss = get_glm_loss('squared_error')

    # Custom
    @register_glm_loss('huber')
    class HuberLoss(GLMLoss):
        ...
"""

from ._base import (
    GLMLoss,
    get_glm_loss,
    register_glm_loss,
    list_glm_losses,
)
from ._squared import SquaredErrorLoss
from ._logistic import LogisticLoss
from ._poisson import PoissonLoss
from ._gamma import GammaLoss
from ._inverse_gaussian import InverseGaussianLoss
from ._negative_binomial import NegativeBinomialLoss
from ._tweedie import TweedieLoss
from ._family import (
    GLMFamily,
    Link,
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
    NegativeBinomial,
    Tweedie,
)
from ._irls import IRLSSolver

# Solvers: re-export from solvers/ (generic)
from statgpu.solvers import (
    fista_solver,
    fista_bb_solver,
    fista_lla_path,
    newton_solver,
    lbfgs_solver,
    admm_solver,
    ConvergenceWarning,
)

__all__ = [
    "GLMLoss",
    "SquaredErrorLoss",
    "LogisticLoss",
    "PoissonLoss",
    "GammaLoss",
    "InverseGaussianLoss",
    "NegativeBinomialLoss",
    "TweedieLoss",
    "GLMFamily",
    "Link",
    "Gaussian",
    "Binomial",
    "Poisson",
    "Gamma",
    "InverseGaussian",
    "NegativeBinomial",
    "Tweedie",
    "IRLSSolver",
    "fista_solver",
    "fista_bb_solver",
    "admm_solver",
    "newton_solver",
    "lbfgs_solver",
    "ConvergenceWarning",
    "get_glm_loss",
    "register_glm_loss",
    "list_glm_losses",
]
