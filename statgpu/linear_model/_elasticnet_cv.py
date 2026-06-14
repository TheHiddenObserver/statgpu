"""Backward-compatibility shim. Use statgpu.linear_model.cv instead.

.. deprecated::
    Will be removed in v0.3.0.
"""
import warnings
warnings.warn(
    "statgpu.linear_model._elasticnet_cv.py is deprecated. Use statgpu.linear_model.cv._elasticnet_cv instead.",
    DeprecationWarning, stacklevel=2,
)
from statgpu.linear_model.cv._elasticnet_cv import *  # noqa: F401,F403
from statgpu.linear_model.cv._elasticnet_cv import ElasticNetCV  # noqa: F401
