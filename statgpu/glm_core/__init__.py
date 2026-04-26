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
from ._family import GLMFamily, Link, Gaussian, Binomial, Poisson
from ._irls import IRLSSolver
from ._solver import fista_solver, lbfgs_solver, newton_solver

__all__ = [
    "GLMLoss",
    "SquaredErrorLoss",
    "LogisticLoss",
    "PoissonLoss",
    "GLMFamily",
    "Link",
    "Gaussian",
    "Binomial",
    "Poisson",
    "IRLSSolver",
    "fista_solver",
    "newton_solver",
    "lbfgs_solver",
    "get_glm_loss",
    "register_glm_loss",
    "list_glm_losses",
]
