"""
Generic loss function registry.

Provides ``register_loss()``, ``get_loss()``, and ``list_losses()`` for all
loss types (GLM, quantile, robust, survival). The GLM-specific registry
in ``statgpu.glm_core._base`` is preserved for backward compatibility.
"""

__all__ = ["register_loss", "get_loss", "list_losses"]

from ._base import LossBase

# Global registry for all loss types
_LOSS_REGISTRY: dict = {}


def get_loss(name: str, **kwargs) -> LossBase:
    """
    Get a loss by name from the registry.

    Parameters
    ----------
    name : str
        Loss name: 'squared_error', 'logistic', 'quantile', 'huber', etc.
    **kwargs
        Arguments passed to the loss constructor.

    Returns
    -------
    LossBase
        Instantiated loss object.

    Raises
    ------
    ValueError
        If loss name is not in the registry.
    """
    if name not in _LOSS_REGISTRY:
        available = list(_LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss: {name}. Available losses: {available}"
        )
    return _LOSS_REGISTRY[name](**kwargs)


def register_loss(name: str):
    """
    Decorator to register a loss class.

    Parameters
    ----------
    name : str
        Name to register the loss under.

    Returns
    -------
    callable
        Decorator function that registers the loss class.
    """
    def decorator(cls):
        if not issubclass(cls, LossBase):
            raise TypeError(
                f"Loss class must inherit from LossBase, got {cls.__bases__}"
            )
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def list_losses() -> list:
    """List all registered loss names."""
    return list(_LOSS_REGISTRY.keys())
