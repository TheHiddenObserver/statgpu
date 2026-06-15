"""Nonparametric estimators."""

# Kernel smoothing (KDE + Nadaraya-Watson kernel regression)
from .kernel_smoothing import (
    BandwidthSelectionResult,
    select_bandwidth,
    select_bandwidth_factor,
    KernelDensityEstimator,
    KDE,
    KDEBootstrapResult,
    fit_kde,
    kde_pdf,
    kde_confidence_interval,
    kde_bootstrap_confidence_interval,
    KernelRegression,
    KernelRegressionRegressor,
    fit_kernel_regression,
    kernel_regression_predict,
)

# Kernel ridge regression
from .kernel_methods import KernelRidge, KernelRidgeCV, pairwise_kernels

# Spline basis functions
from .splines import bspline_basis, natural_cubic_spline_basis

__all__ = [
    # Kernel smoothing
    "BandwidthSelectionResult",
    "select_bandwidth",
    "select_bandwidth_factor",
    "KernelDensityEstimator",
    "KDE",
    "KDEBootstrapResult",
    "fit_kde",
    "kde_pdf",
    "kde_confidence_interval",
    "kde_bootstrap_confidence_interval",
    "KernelRegression",
    "KernelRegressionRegressor",
    "fit_kernel_regression",
    "kernel_regression_predict",
    # Kernel methods
    "KernelRidge",
    "KernelRidgeCV",
    "pairwise_kernels",
    # Splines
    "bspline_basis",
    "natural_cubic_spline_basis",
]
