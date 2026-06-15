"""
statgpu.core.formula – R-style formula interface for statgpu models.

This module provides formula-based model fitting similar to statsmodels/patsy::

    >>> import statgpu as sg
    >>> model = sg.LinearRegression()
    >>> model.fit(formula="y ~ x1 + x2 + C(cat)", data=df)
    >>> model.summary()

The formula syntax is parsed by `patsy` (optional dependency). Install with::

    pip install statgpu[formula]

Public API
----------
FormulaParser
    Main class for parsing R-style formulas and building design matrices.
parse_formula
    Convenience function for one-shot formula evaluation.
"""

from ._parser import FormulaParser
from ._design import parse_formula, parse_formula_safe
from ._terms import make_surv_env, _surv

__all__ = [
    "FormulaParser",
    "parse_formula",
    "parse_formula_safe",
    "make_surv_env",
    "_surv",
]
