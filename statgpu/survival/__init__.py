"""
Survival analysis models.

.. rubric:: Naming conventions

- ``_cuda`` — CUDA RawKernel (pre-compiled CUDA C kernels).
- ``_cupy`` — CuPy array operations (GPU via CuPy).
- ``_triton`` — Triton kernel (GPU via OpenAI Triton).
"""

from ._cox import CoxPH
from ._cox_cv import CoxPHCV
from ._cox_errors import CoxCandidateNumericalError

__all__ = ['CoxPH', 'CoxPHCV', 'CoxCandidateNumericalError']
