"""Backward-compatibility shim. Use statgpu.cross_validation instead.

.. deprecated::
    This module is deprecated. Import from ``statgpu.cross_validation`` instead.
    Will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "statgpu.linear_model._cv_base is deprecated. "
    "Use statgpu.cross_validation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from statgpu.cross_validation._base import *  # noqa: F401,F403
from statgpu.cross_validation._base import __all__  # noqa: F401

# Re-export names not in __all__ but used by downstream code
from statgpu.cross_validation._base import (  # noqa: F401
    CVEstimatorBase,
    CVCache,
    INTERCEPT_CLIP_BOUND,
    kfold_indices,
    folds_are_complete,
    hash_cv_data,
    validate_cv_sample_weight,
    batch_mse,
    detect_gpu_input,
    _torch_cuda_available,
)
