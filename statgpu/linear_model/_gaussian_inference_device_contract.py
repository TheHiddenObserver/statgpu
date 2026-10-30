"""Compatibility import for the former Gaussian device-selection installer.

Device selection now lives directly in ``_gaussian_inference._as_backend_array``.
Keep this historical private module and callable importable without rebinding
the helper or adding an import-order dependency.
"""

from __future__ import annotations

def _install_host_torch_device_contract() -> None:
    """Compatibility no-op; the shared helper already owns device selection."""
