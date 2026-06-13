"""Backward compatibility shim. Use statgpu.nonparametric.splines and statgpu.semiparametric instead."""
from statgpu.nonparametric.splines import *  # noqa: F401,F403
from statgpu.nonparametric.splines import __all__ as _orig_all  # noqa: F401
# GAM moved to semiparametric but re-exported here for backward compat
from statgpu.semiparametric import GAM  # noqa: F401
__all__ = list(_orig_all) + ["GAM"]
