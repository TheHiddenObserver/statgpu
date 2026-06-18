"""ANOVA with GPU acceleration."""

from ._oneway import f_oneway, AnovaResult
from ._twoway import f_twoway, TwoWayAnovaResult
from ._welch import f_welch
from ._posthoc import tukey_hsd, bonferroni, TukeyResult, PosthocResult, PairwiseComparison
from ._effect_size import cohens_f, partial_eta_squared

__all__ = [
    'f_oneway',
    'AnovaResult',
    'f_twoway',
    'TwoWayAnovaResult',
    'f_welch',
    'tukey_hsd',
    'bonferroni',
    'TukeyResult',
    'PosthocResult',
    'PairwiseComparison',
    'cohens_f',
    'partial_eta_squared',
]
