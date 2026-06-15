"""Generic optimization solvers for penalized loss functions.

These solvers work with any loss that implements the GLMLoss interface
(value, gradient, fused_value_and_gradient, lipschitz, hessian, preprocess)
and any penalty with a proximal operator.
"""

__all__ = [
    "fista_solver",
    "fista_bb_solver",
    "fista_lla_path",
    "newton_solver",
    "lbfgs_solver",
    "admm_solver",
    "ConvergenceWarning",
]

from ._convergence import ConvergenceWarning
from ._fista import fista_solver
from ._fista_bb import fista_bb_solver
from ._fista_lla import fista_lla_path
from ._newton import newton_solver
from ._lbfgs import lbfgs_solver
from ._admm import admm_solver
