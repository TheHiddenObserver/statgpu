"""Backward-compatibility shim. Use statgpu.linear_model.cv instead.

.. deprecated::
    Will be removed in v0.3.0.
"""
import warnings
warnings.warn(
    "statgpu.linear_model._logistic_cv.py is deprecated. Use statgpu.linear_model.cv._logistic_cv instead.",
    DeprecationWarning, stacklevel=2,
)
from statgpu.linear_model.cv._logistic_cv import *  # noqa: F401,F403
from statgpu.linear_model.cv._logistic_cv import LogisticRegressionCV  # noqa: F401
