"""Backward compatibility shim. Use statgpu.nonparametric.kernel_methods instead.

.. deprecated:: Will be removed in v0.3.0.
"""
import warnings
warnings.warn(
    "statgpu.kernel_methods is deprecated. Use statgpu.nonparametric.kernel_methods instead.",
    DeprecationWarning, stacklevel=2,
)
from statgpu.nonparametric.kernel_methods import *  # noqa: F401,F403
from statgpu.nonparametric.kernel_methods import __all__  # noqa: F401
