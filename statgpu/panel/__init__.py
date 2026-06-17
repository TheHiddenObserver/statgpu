"""
Panel data models with GPU acceleration.

Provides fixed effects, random effects, pooled, between, first-difference,
and Fama-MacBeth estimators for panel/longitudinal data, along with
clustered and HAC covariance estimators.
"""

from ._fixed_effects import PanelOLS
from ._random_effects import RandomEffects, RandomEffectsOLS
from ._covariance import clustered_covariance, two_way_clustered_covariance, hac_covariance
from ._utils import PanelSummary
from ._pooled import PooledOLS
from ._between import BetweenOLS
from ._first_diff import FirstDifferenceOLS
from ._fama_macbeth import FamaMacBeth

__all__ = [
    'PanelOLS',
    'RandomEffects',
    'RandomEffectsOLS',
    'PooledOLS',
    'BetweenOLS',
    'FirstDifferenceOLS',
    'FamaMacBeth',
    'PanelSummary',
    'clustered_covariance',
    'two_way_clustered_covariance',
    'hac_covariance',
]
