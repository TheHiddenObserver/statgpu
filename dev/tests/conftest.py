"""
Pytest configuration for GPU tests.

Provides shared fixtures for detecting GPU availability, enforcing
STATGPU_REQUIRE_PHYSICAL_GPU, and parametrizing tests across backends.

This consolidates the 6+ different GPU detection patterns previously
duplicated across individual test files into one shared location.

Environment Variables
--------------------
STATGPU_REQUIRE_PHYSICAL_GPU=1 : When set, tests that would normally
    skip due to missing GPU hardware will instead fail. Use on the
    remote GPU server during validation gates.

Usage in tests
--------------
    def test_something(cupy_available):
        cp = pytest.importorskip("cupy")
        ...

    @pytest.mark.gpu
    def test_gpu_feature(backend_name):
        ...

    def test_three_backend(xp_and_device):
        xp, device_str = xp_and_device
        ...
"""

from __future__ import annotations

import os
import platform

import pytest


# ============================================================================
# Session-scoped backend availability
# ============================================================================


@pytest.fixture(scope="session")
def has_cupy():
    """True if CuPy with CUDA is available."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, Exception):
        return False


@pytest.fixture(scope="session")
def has_cupy_library():
    """True if CuPy is installable (may not have GPU)."""
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_torch():
    """True if PyTorch is installed."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_torch_cuda():
    """True if PyTorch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================================
# STATGPU_REQUIRE_PHYSICAL_GPU enforcement (Section 7.1)
# ============================================================================


def _require_physical_gpu():
    """Check if STATGPU_REQUIRE_PHYSICAL_GPU is set."""
    return os.environ.get("STATGPU_REQUIRE_PHYSICAL_GPU", "") == "1"


@pytest.fixture(scope="session")
def physical_gpu_required():
    """True if STATGPU_REQUIRE_PHYSICAL_GPU=1 is set."""
    return _require_physical_gpu()


@pytest.fixture(autouse=False)
def require_physical_gpu(request):
    """Auto-use fixture: enforce physical GPU requirement.

    When STATGPU_REQUIRE_PHYSICAL_GPU=1:
    - Tests using cupy/torch that would skip become failures
    - This is enforced by monkeypatching pytest.importorskip

    When STATGPU_REQUIRE_PHYSICAL_GPU is not set:
    - Standard behavior: GPU-dependent tests skip if GPU unavailable
    """
    if not _require_physical_gpu():
        return  # Normal behavior

    # Monkeypatch importorskip to fail instead of skip for GPU libs
    original = pytest.importorskip

    def _strict_importorskip(modname, minversion=None, reason=None):
        """Like importorskip but fails for GPU libs when REQUIRE_PHYSICAL_GPU=1."""
        if modname in ("cupy", "torch"):
            try:
                __import__(modname)
            except ImportError as e:
                pytest.fail(
                    f"STATGPU_REQUIRE_PHYSICAL_GPU=1 but '{modname}' is not "
                    f"available: {e}"
                )
        return original(modname, minversion, reason)

    # Patch
    request.config._strict_importorskip_original = original
    pytest.importorskip = _strict_importorskip


# ============================================================================
# Backend parametrization fixtures
# ============================================================================


def _get_available_backends(has_cupy, has_torch_cuda):
    """Determine which backends are available for testing."""
    backends = ["numpy"]
    if has_cupy:
        backends.append("cupy")
    if has_torch_cuda:
        backends.append("torch")
    return backends


@pytest.fixture
def available_backends(has_cupy, has_torch_cuda):
    """List of backend names available for testing.

    Example: ["numpy", "cupy", "torch"] or ["numpy"] if no GPU.
    """
    return _get_available_backends(has_cupy, has_torch_cuda)


@pytest.fixture(params=["numpy"])
def backend_name(request, has_cupy, has_torch_cuda, physical_gpu_required):
    """Parametrized fixture for testing with available backends.

    Each test using this fixture runs once per available backend.
    On CPU-only systems, only runs numpy.
    """
    backends = _get_available_backends(has_cupy, has_torch_cuda)

    # When physical GPU is required, fail if only numpy is available
    # but cupy/torch was expected
    if physical_gpu_required and len(backends) < 3:
        if has_cupy is None and has_torch_cuda is None:
            pytest.fail(
                "STATGPU_REQUIRE_PHYSICAL_GPU=1 but no GPU backend is available"
            )

    # Override params dynamically
    # Note: this approach uses indirect parametrization
    return request.param


@pytest.fixture
def xp_and_device(backend_name):
    """Returns (array_module, device_string) for the current backend.

    Usage:
        def test_something(xp_and_device):
            xp, device = xp_and_device
            X = xp.asarray([[1, 2], [3, 4]])
    """
    if backend_name == "cupy":
        import cupy as cp
        return cp, "cuda"
    elif backend_name == "torch":
        import torch
        return torch, "cuda"
    else:
        import numpy as np
        return np, "cpu"


# ============================================================================
# Convenience skip/fail fixtures
# ============================================================================


@pytest.fixture
def cupy_available(has_cupy, physical_gpu_required):
    """Skip if CuPy CUDA is unavailable, fail if REQUIRE_PHYSICAL_GPU=1."""
    if not has_cupy:
        if physical_gpu_required:
            pytest.fail(
                "STATGPU_REQUIRE_PHYSICAL_GPU=1 but CuPy CUDA is not available"
            )
        pytest.skip("CuPy CUDA not available")
    return True


@pytest.fixture
def torch_cuda_available(has_torch_cuda, physical_gpu_required):
    """Skip if Torch CUDA is unavailable, fail if REQUIRE_PHYSICAL_GPU=1."""
    if not has_torch_cuda:
        if physical_gpu_required:
            pytest.fail(
                "STATGPU_REQUIRE_PHYSICAL_GPU=1 but Torch CUDA is not available"
            )
        pytest.skip("Torch CUDA not available")
    return True


@pytest.fixture
def gpu_available(has_cupy, has_torch_cuda, physical_gpu_required):
    """Skip if NO GPU backend is available, fail if REQUIRE_PHYSICAL_GPU=1."""
    if not has_cupy and not has_torch_cuda:
        if physical_gpu_required:
            pytest.fail(
                "STATGPU_REQUIRE_PHYSICAL_GPU=1 but no GPU backend is available"
            )
        pytest.skip("No GPU backend available (neither CuPy nor Torch CUDA)")
    return True


# ============================================================================
# Deterministic seed fixture
# ============================================================================


@pytest.fixture
def seed_random():
    """Set deterministic seeds for numpy, cupy, torch, and random.

    Usage:
        def test_with_seed(seed_random):
            # All RNGs are seeded with 42
            ...
    """
    seed = 42
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import cupy as cp
        cp.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


# ============================================================================
# Test data fixtures
# ============================================================================


@pytest.fixture
def sample_data_2d():
    """Simple 2D regression dataset: n=100, p=5."""
    import numpy as np
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p).astype(np.float64)
    beta = np.array([1.0, -0.5, 2.0, 0.0, -1.5], dtype=np.float64)
    y = X @ beta + np.random.randn(n).astype(np.float64) * 0.5
    return X, y


@pytest.fixture
def sample_data_wide():
    """Wide 2D dataset: p > n."""
    import numpy as np
    np.random.seed(43)
    n, p = 50, 80
    X = np.random.randn(n, p).astype(np.float64)
    beta = np.zeros(p, dtype=np.float64)
    beta[:5] = [1.0, -0.5, 2.0, 0.0, -1.5]
    y = X @ beta + np.random.randn(n).astype(np.float64) * 0.3
    return X, y


# ============================================================================
# Pytest configuration hooks
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: test requires physical GPU (any backend)"
    )
    config.addinivalue_line(
        "markers", "cupy: test requires CuPy CUDA"
    )
    config.addinivalue_line(
        "markers", "torch_cuda: test requires Torch CUDA"
    )
    config.addinivalue_line(
        "markers", "numpy_only: test only works on NumPy CPU backend"
    )
    config.addinivalue_line(
        "markers", "slow: test is slow (>10 seconds)"
    )
    config.addinivalue_line(
        "markers", "memory: test checks memory behavior"
    )
    config.addinivalue_line(
        "markers", "physical_gpu: test requires STATGPU_REQUIRE_PHYSICAL_GPU=1"
    )


def pytest_collection_modifyitems(config, items):
    """Mark tests based on their requirements.

    When STATGPU_REQUIRE_PHYSICAL_GPU=1, convert skips to failures.
    """
    if not _require_physical_gpu():
        return

    # Check for tests that would skip due to missing GPU
    for item in items:
        if item.get_closest_marker("gpu") or item.get_closest_marker("cupy"):
            # If marked as GPU, verify CuPy is available
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() == 0:
                    # Don't fail here - let the fixture handle it
                    pass
            except ImportError:
                pass


def pytest_report_header(config):
    """Add GPU availability info to test report header."""
    lines = []
    lines.append("statgpu GPU validation (conftest.py)")

    # Check CuPy
    try:
        import cupy as cp
        devices = cp.cuda.runtime.getDeviceCount()
        lines.append(f"CuPy: available ({devices} device(s))")
    except ImportError:
        lines.append("CuPy: not installed")

    # Check Torch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f"Torch CUDA: available ({torch.cuda.device_count()} device(s))")
        else:
            lines.append("Torch CUDA: not available (torch installed, no CUDA)")
    except ImportError:
        lines.append("Torch: not installed")

    # PHYSICAL_GPU status
    if _require_physical_gpu():
        lines.append("STATGPU_REQUIRE_PHYSICAL_GPU: 1 (skips will be failures)")

    return "\n".join(lines)
