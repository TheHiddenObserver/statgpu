"""
Panel data models with GPU acceleration.

Provides fixed effects, random effects, pooled, between, first-difference,
and Fama-MacBeth estimators for panel/longitudinal data, shared structured
fit/test results, and panel covariance estimators.
"""

from ._fixed_effects import PanelOLS, FixedEffects
from ._random_effects import RandomEffects, RandomEffectsOLS
from ._covariance import (
    clustered_covariance,
    two_way_clustered_covariance,
    hac_covariance,
    driscoll_kraay_covariance,
)
from ._utils import PanelSummary
from ._results import PanelTestResult, PanelFitStatistics
from ._diagnostics import hausman_test, pooling_f_test, breusch_pagan_lm_test
from ._pooled import PooledOLS
from ._between import BetweenOLS
from ._first_diff import FirstDifferenceOLS
from ._fama_macbeth import FamaMacBeth

__all__ = [
    'PanelOLS',
    'FixedEffects',
    'RandomEffects',
    'RandomEffectsOLS',
    'PooledOLS',
    'BetweenOLS',
    'FirstDifferenceOLS',
    'FamaMacBeth',
    'PanelSummary',
    'PanelTestResult',
    'PanelFitStatistics',
    'hausman_test',
    'pooling_f_test',
    'breusch_pagan_lm_test',
    'clustered_covariance',
    'two_way_clustered_covariance',
    'hac_covariance',
    'driscoll_kraay_covariance',
]
