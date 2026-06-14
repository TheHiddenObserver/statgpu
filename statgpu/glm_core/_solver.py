"""Backward-compatibility shim. Use statgpu.solvers instead.

.. deprecated::
    This module is deprecated. Import from ``statgpu.solvers`` instead.
    Will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "statgpu.glm_core._solver is deprecated. Use statgpu.solvers instead.",
    DeprecationWarning,
    stacklevel=2,
)

from statgpu.solvers import *  # noqa: F401,F403
from statgpu.solvers import __all__  # noqa: F401

# Also re-export fista_lla_path which is in solvers/
from statgpu.solvers import fista_lla_path  # noqa: F401
