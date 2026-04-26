"""
statgpu.core — Shared core utilities for statgpu.

Provides common infrastructure used across all statgpu model modules:
- ``core.formula``: R-style formula interface (``y ~ x1 + x2``).
"""

from . import formula

__all__ = ["formula"]
