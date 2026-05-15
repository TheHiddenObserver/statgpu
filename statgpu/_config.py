"""
Device configuration and GPU detection.
"""

import os
import warnings
from enum import Enum
from typing import Optional, Union


class Device(Enum):
    """Device types for computation."""
    CPU = "cpu"
    CUDA = "cuda"
    TORCH = "torch"
    AUTO = "auto"


class _DeviceManager:
    """Internal device state manager."""

    def __init__(self):
        self._current_device = Device.AUTO
        self._cupy_available = None
        self._torch_available = None
        self._cuda_available = None

    def _check_cupy(self) -> bool:
        """Check if CuPy is available and working."""
        if self._cupy_available is None:
            try:
                import cupy as cp
                # Test actual CUDA functionality
                cp.cuda.Device(0).use()
                self._cupy_available = True
            except Exception:
                self._cupy_available = False
        return self._cupy_available

    def _check_torch(self) -> bool:
        """Check if PyTorch CUDA is available."""
        if self._torch_available is None:
            try:
                import torch
                self._torch_available = torch.cuda.is_available()
            except Exception:
                self._torch_available = False
        return self._torch_available

    def cuda_available(self) -> bool:
        """Check if any CUDA backend is available (CuPy or Torch)."""
        if self._cuda_available is None:
            self._cuda_available = self._check_cupy() or self._check_torch()
        return self._cuda_available

    def get_device(self) -> Device:
        """Get current device setting."""
        if self._current_device == Device.AUTO:
            if self.cuda_available():
                # Prefer CuPy if both are available for backward compatibility
                return Device.CUDA if self._check_cupy() else Device.TORCH
            return Device.CPU
        return self._current_device

    def set_device(self, device: Union[str, Device]) -> None:
        """Set device for computation."""
        if isinstance(device, str):
            device = Device(device.lower())

        if device in (Device.CUDA, Device.TORCH) and not self.cuda_available():
            warnings.warn(
                "CUDA requested but not available. statgpu keeps the explicit "
                "device setting and model execution will raise unless a matching "
                "GPU backend is installed; use device='auto' for automatic CPU selection.",
                RuntimeWarning
            )

        self._current_device = device


# Global device manager instance
_device_manager = _DeviceManager()


def get_device() -> Device:
    """
    Get the current computation device.
    
    Returns
    -------
    Device
        Current resolved device. If the configured device is ``'auto'``, this
        resolves to CuPy CUDA when available, then Torch CUDA, then CPU.
    """
    return _device_manager.get_device()


def set_device(device: Union[str, Device]) -> None:
    """
    Set the computation device.
    
    Parameters
    ----------
    device : str or Device
        Device to use: ``'cpu'``, ``'cuda'``, ``'torch'``, or ``'auto'``.
        ``'cuda'`` and ``'torch'`` are explicit GPU requests and are kept as
        configured; model execution raises if the matching backend is not
        available. ``'auto'`` chooses CuPy CUDA when available, then Torch CUDA,
        then CPU.
    
    Examples
    --------
    >>> import statgpu as sg
    >>> sg.set_device('cuda')  # Force CuPy CUDA
    >>> sg.set_device('torch') # Force Torch CUDA
    >>> sg.set_device('auto')  # Auto-detect
    """
    _device_manager.set_device(device)


def cuda_available() -> bool:
    """Check if CUDA is available."""
    return _device_manager.cuda_available()


# Convenience imports
__all__ = ['Device', 'get_device', 'set_device', 'cuda_available']
