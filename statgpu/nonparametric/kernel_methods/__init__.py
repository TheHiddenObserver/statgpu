"""Kernel methods with GPU acceleration."""

from ._kernels import (
    rbf_kernel,
    polynomial_kernel,
    linear_kernel,
    laplacian_kernel,
    sigmoid_kernel,
    cosine_kernel,
    chi2_kernel,
    pairwise_kernels,
)
from ._krr import KernelRidge
from ._krr_cv import KernelRidgeCV
from ._nystroem import Nystroem
from ._kpca import KernelPCA

__all__ = [
    "rbf_kernel",
    "polynomial_kernel",
    "linear_kernel",
    "laplacian_kernel",
    "sigmoid_kernel",
    "cosine_kernel",
    "chi2_kernel",
    "pairwise_kernels",
    "KernelRidge",
    "KernelRidgeCV",
    "Nystroem",
    "KernelPCA",
]
