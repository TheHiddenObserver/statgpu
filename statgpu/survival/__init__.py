"""
Survival analysis models.

.. rubric:: Naming conventions

- ``_cuda`` — CUDA RawKernel (pre-compiled CUDA C kernels).
- ``_cupy`` — CuPy array operations (GPU via CuPy).
- ``_triton`` — Triton kernel (GPU via OpenAI Triton).
"""

from ._cox import CoxPH
from ._cox_fit_adapter import install_coxph_fit_adapter
from ._cox_cv import CoxPHCV

install_coxph_fit_adapter(CoxPH)
del install_coxph_fit_adapter

__all__ = ['CoxPH', 'CoxPHCV']
