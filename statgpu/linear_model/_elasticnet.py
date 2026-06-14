"""Backward-compatibility shim. Use statgpu.linear_model.wrappers instead.

.. deprecated::
    This module is deprecated. Import from statgpu.linear_model.wrappers instead.
    Will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "statgpu.linear_model._elasticnet.py is deprecated. "
    "Use statgpu.linear_model.wrappers._elasticnet instead.",
    DeprecationWarning,
    stacklevel=2,
)

from statgpu.linear_model.wrappers._elasticnet import *  # noqa: F401,F403
from statgpu.linear_model.wrappers._elasticnet import ElasticNet  # noqa: F401
