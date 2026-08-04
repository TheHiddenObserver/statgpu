"""
Survival analysis models.

.. rubric:: Naming conventions

- ``_cuda`` — CUDA RawKernel (pre-compiled CUDA C kernels).
- ``_cupy`` — CuPy array operations (GPU via CuPy).
- ``_triton`` — Triton kernel (GPU via OpenAI Triton).
"""

from ._cox import CoxPH
from ._cox_cv import CoxPHCV
from ._cox_errors import CoxFitNumericalError

# Install the public custom-grid boundary only after CoxPHCV and its selector
# are fully defined. The wrapper keeps user-facing result arrays in input order
# while continuation and staged screening run by numerical penalty rank.
from . import _cox_cv_penalty_order_contract as _cox_cv_penalty_order_contract

# Experimental two-stage/successive-halving screening currently has no complete
# three-backend correctness proof. Convert any request into an explicit single
# exhaustive full-precision run rather than allowing silent candidate removal.
from . import _cox_cv_staged_safety_contract as _cox_cv_staged_safety_contract

# Preserve one-shot custom split generators across repeated fit, clone, and
# serialization without rewriting the public constructor parameter.
from . import (
    _cox_cv_split_lifecycle_contract as _cox_cv_split_lifecycle_contract,
)

__all__ = ['CoxPH', 'CoxPHCV', 'CoxFitNumericalError']
