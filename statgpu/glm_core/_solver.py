"""Backward compatibility shim. Use statgpu.solvers instead."""
from statgpu.solvers import *  # noqa: F401,F403
from statgpu.solvers import __all__  # noqa: F401
from statgpu.solvers import fista_lla_path  # noqa: F401
