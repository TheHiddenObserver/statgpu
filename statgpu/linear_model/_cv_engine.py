"""Backward-compatibility shim. Use statgpu.cross_validation instead.

.. deprecated::
    This module is deprecated. Import from ``statgpu.cross_validation`` instead.
    Will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "statgpu.linear_model._cv_engine is deprecated. "
    "Use statgpu.cross_validation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from statgpu.cross_validation._engine import *  # noqa: F401,F403
from statgpu.cross_validation._engine import __all__  # noqa: F401
