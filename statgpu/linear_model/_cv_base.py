"""Backward compatibility shim."""
from statgpu.cross_validation._base import *  # noqa: F401,F403
from statgpu.cross_validation._base import CVEstimatorBase, CVCache, INTERCEPT_CLIP_BOUND, kfold_indices, folds_are_complete, hash_cv_data, validate_cv_sample_weight, batch_mse, detect_gpu_input, _torch_cuda_available  # noqa: F401
