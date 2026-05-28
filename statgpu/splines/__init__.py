"""
Spline basis functions and GAM (Generalized Additive Model) with GPU support.
"""

from ._bspline_basis import bspline_basis, natural_cubic_spline_basis
from ._gam import GAM

__all__ = ['bspline_basis', 'natural_cubic_spline_basis', 'GAM']
