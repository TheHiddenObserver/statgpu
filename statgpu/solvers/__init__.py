"""Generic optimization solvers for penalized loss functions.

These solvers consume loss objects that implement the required numerical
interface (for example value/gradient/Hessian/Lipschitz primitives) together
with compatible penalty objects. Support is solver-specific: implementing the
shared loss interface does not imply that every loss × solver combination is
supported, and unsupported public calls raise an error.
"""

__all__ = [
    "fista_solver",
    "fista_bb_solver",
    "fista_lla_path",
    "newton_solver",
    "proximal_newton_solver",
    "proximal_irls_quantile_solver",
    "quantile_cd_solver",
    "lbfgs_solver",
    "lbfgs_b_solver",
    "admm_solver",
    "ConvergenceWarning",
]

# Install utility compatibility contracts before solver modules bind helper
# functions from ``._utils`` at import time.
from . import _adaptive_group_lipschitz_contract as _adaptive_group_lipschitz_contract

from ._convergence import ConvergenceWarning
from ._fista import fista_solver
from ._fista_lla_group_contract import fista_lla_path
from ._newton import newton_solver
from ._proximal_newton import proximal_newton_solver
from ._quantile_proximal_public_contract import proximal_irls_quantile_solver
from ._quantile_cd import quantile_cd_solver
from ._lbfgs import lbfgs_solver
from ._lbfgs_b import lbfgs_b_solver
from ._quantile_solver_guard import admm_solver, fista_bb_solver
