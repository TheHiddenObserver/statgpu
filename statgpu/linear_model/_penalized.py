"""Backward compatibility shim. Use statgpu.linear_model.penalized instead."""
from statgpu.linear_model.penalized import *  # noqa: F401,F403
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel, SelectivePenalty  # noqa: F401
