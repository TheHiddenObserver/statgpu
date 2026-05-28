"""
statgpu: GPU-accelerated statistical methods

A sklearn-compatible library for statistical computing with GPU support.
"""

__version__ = "0.1.0"

from ._config import get_device, set_device, Device
from ._base import BaseEstimator
from .linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    PoissonRegression,
    GammaRegression,
    InverseGaussianRegression,
    NegativeBinomialRegression,
    TweedieRegression,
    GeneralizedLinearModel,
    OrderedGeneralizedLinearModel,
    OrderedLogitRegression,
    OrderedProbitRegression,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
)
from .survival import CoxPH, CoxPHCV
from .panel import PanelOLS, RandomEffects
from .backends import get_backend, NumpyBackend, CuPyBackend, TorchBackend
from .metrics import evaluate_binary_classification
from .feature_selection import (
    FixedXKnockoffSelector,
    KnockoffSelector,
    fixed_x_knockoff_filter,
    knockoff_filter,
    model_x_knockoff_filter,
)
from .inference import adjust_pvalues, combine_pvalues, multipletests
from .inference import bootstrap_statistic, permutation_test
from .anova import f_oneway
from .kernel_methods import KernelRidge, KernelRidgeCV
from .nonparametric import (
    BandwidthSelectionResult,
    KernelDensityEstimator,
    KDE,
    KDEBootstrapResult,
    KernelRegression,
    KernelRegressionRegressor,
    fit_kde,
    fit_kernel_regression,
    kde_pdf,
    kde_confidence_interval,
    kernel_regression_predict,
    kde_bootstrap_confidence_interval,
    select_bandwidth,
    select_bandwidth_factor,
)
from .splines import GAM, bspline_basis, natural_cubic_spline_basis
from .covariance import EmpiricalCovariance, LedoitWolf, OAS

__all__ = [
    "get_device",
    "set_device",
    "Device",
    "BaseEstimator",
    "LinearRegression",
    "LogisticRegression",
    "LogisticRegressionCV",
    "PoissonRegression",
    "GammaRegression",
    "InverseGaussianRegression",
    "NegativeBinomialRegression",
    "TweedieRegression",
    "GeneralizedLinearModel",
    "OrderedGeneralizedLinearModel",
    "OrderedLogitRegression",
    "OrderedProbitRegression",
    "PenalizedGeneralizedLinearModel",
    "PenalizedLinearRegression",
    "PenalizedLogisticRegression",
    "PenalizedPoissonRegression",
    "Ridge",
    "RidgeCV",
    "Lasso",
    "LassoCV",
    "ElasticNet",
    "ElasticNetCV",
    "CoxPH",
    "CoxPHCV",
    "PanelOLS",
    "RandomEffects",
    "KernelRidge",
    "KernelRidgeCV",
    "get_backend",
    "NumpyBackend",
    "CuPyBackend",
    "TorchBackend",
    "evaluate_binary_classification",
    "knockoff_filter",
    "fixed_x_knockoff_filter",
    "model_x_knockoff_filter",
    "KnockoffSelector",
    "FixedXKnockoffSelector",
    "adjust_pvalues",
    "combine_pvalues",
    "multipletests",
    "bootstrap_statistic",
    "permutation_test",
    "f_oneway",
    "BandwidthSelectionResult",
    "KernelDensityEstimator",
    "KDE",
    "KDEBootstrapResult",
    "KernelRegression",
    "KernelRegressionRegressor",
    "fit_kde",
    "fit_kernel_regression",
    "kde_pdf",
    "kde_confidence_interval",
    "kernel_regression_predict",
    "kde_bootstrap_confidence_interval",
    "select_bandwidth",
    "select_bandwidth_factor",
    "GAM",
    "bspline_basis",
    "natural_cubic_spline_basis",
    "EmpiricalCovariance",
    "LedoitWolf",
    "OAS",
]
