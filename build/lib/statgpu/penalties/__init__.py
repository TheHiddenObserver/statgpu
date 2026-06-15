"""
Penalty function registry for statgpu.

Usage:
    from statgpu.penalties import get_penalty, register_penalty

    # Built-in
    pen = get_penalty('l1', alpha=0.1)

    # Custom
    @register_penalty('custom')
    class CustomPenalty(Penalty):
        ...
"""

from ._base import Penalty, CompositePenalty
from ._l1 import L1Penalty
from ._l2 import L2Penalty
from ._elasticnet import ElasticNetPenalty
from ._scad import SCADPenalty
from ._mcp import MCPPenalty
from ._adaptive_l1 import AdaptiveL1Penalty
from ._group_lasso import GroupLassoPenalty, AdaptiveGroupLassoPenalty
from ._group_mcp import GroupMCPPenalty
from ._group_scad import GroupSCADPenalty


def _torch_compile_ok():
    """Check if torch.compile is usable (CUDA capability >= 7.0 required)."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 7
        return True  # CPU-only torch can compile
    except Exception:
        return False


__all__ = [
    "Penalty",
    "CompositePenalty",
    "L1Penalty",
    "L2Penalty",
    "ElasticNetPenalty",
    "SCADPenalty",
    "MCPPenalty",
    "AdaptiveL1Penalty",
    "GroupLassoPenalty",
    "AdaptiveGroupLassoPenalty",
    "GroupMCPPenalty",
    "GroupSCADPenalty",
    "get_penalty",
    "register_penalty",
    "list_penalties",
]

_PENALTY_REGISTRY = {
    "l1": L1Penalty,
    "l2": L2Penalty,
    "l2_squared": L2Penalty,
    "ridge": L2Penalty,
    "elasticnet": ElasticNetPenalty,
    "en": ElasticNetPenalty,
    "scad": SCADPenalty,
    "mcp": MCPPenalty,
    "adaptive_l1": AdaptiveL1Penalty,
    "adaptive_lasso": AdaptiveL1Penalty,
    "group_lasso": GroupLassoPenalty,
    "gl": GroupLassoPenalty,
    "group_mcp": GroupMCPPenalty,
    "gmcp": GroupMCPPenalty,
    "group_scad": GroupSCADPenalty,
    "gscad": GroupSCADPenalty,
}


def get_penalty(name: str, **kwargs) -> Penalty:
    """
    Get a penalty by name from the registry.

    Parameters
    ----------
    name : str
        Penalty name: 'l1', 'l2', 'ridge', 'elasticnet', 'en'.
    **kwargs
        Arguments passed to the penalty constructor.

    Returns
    -------
    Penalty
        Instantiated penalty object.

    Raises
    ------
    ValueError
        If penalty name is not in the registry.
    """
    if name not in _PENALTY_REGISTRY:
        available = list(_PENALTY_REGISTRY.keys())
        raise ValueError(
            f"Unknown penalty: {name}. Available penalties: {available}"
        )
    return _PENALTY_REGISTRY[name](**kwargs)


def register_penalty(name: str):
    """
    Decorator to register a custom penalty class.

    Parameters
    ----------
    name : str
        Name to register the penalty under.

    Returns
    -------
    callable
        Decorator function that registers the penalty class.

    Example
    -------
    >>> @register_penalty('huber')
    ... class HuberPenalty(Penalty):
    ...     ...
    """
    def decorator(cls):
        if not issubclass(cls, Penalty):
            raise TypeError(
                f"Penalty class must inherit from Penalty, got {cls.__bases__}"
            )
        _PENALTY_REGISTRY[name] = cls
        return cls
    return decorator


def list_penalties() -> list:
    """List all registered penalty names."""
    return list(_PENALTY_REGISTRY.keys())
