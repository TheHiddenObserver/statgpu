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

from ._base import Penalty
from ._l1 import L1Penalty
from ._l2 import L2Penalty
from ._elasticnet import ElasticNetPenalty

# Lazy loading for non-convex penalties to avoid circular imports
__all__ = [
    "Penalty",
    "L1Penalty",
    "L2Penalty",
    "ElasticNetPenalty",
    "get_penalty",
    "register_penalty",
]

_PENALTY_REGISTRY = {
    "l1": L1Penalty,
    "l2": L2Penalty,
    "l2_squared": L2Penalty,
    "ridge": L2Penalty,
    "elasticnet": ElasticNetPenalty,
    "en": ElasticNetPenalty,
    # Non-convex penalties (loaded lazily)
    # "scad": SCADPenalty,
    # "mcp": MCPPenalty,
    # Adaptive and group penalties
    # "adaptive_l1": AdaptiveL1Penalty,
    # "adaptive_lasso": AdaptiveL1Penalty,
    # "group_lasso": GroupLassoPenalty,
    # "gl": GroupLassoPenalty,
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
