"""Backward-compatibility shim. Use statgpu.linear_model.wrappers instead.

.. deprecated::
    This module is deprecated. Import from statgpu.linear_model.wrappers instead.
    Will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "statgpu.linear_model._gamma_glm.py is deprecated. "
    "Use statgpu.linear_model.wrappers._gamma instead.",
    DeprecationWarning,
    stacklevel=2,
)

from statgpu.linear_model.wrappers._gamma import *  # noqa: F401,F403
from statgpu.linear_model.wrappers._gamma import GammaRegression  # noqa: F401
