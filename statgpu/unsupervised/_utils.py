"""Shared utilities for unsupervised estimators."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def check_2d_array(X, name: str = "X") -> None:
    """Validate that *X* is a non-empty 2D array-like object."""
    if getattr(X, "ndim", None) != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one sample and one feature")


def reject_sparse(X, estimator_name: str) -> None:
    """Raise a consistent error for unsupported sparse inputs."""
    if sparse.issparse(X):
        raise NotImplementedError(f"sparse input is not supported in {estimator_name} v1")


def scalar_to_float(x) -> float:
    """Convert a NumPy/CuPy/Torch scalar to Python float."""
    if hasattr(x, "detach"):
        return float(x.detach().cpu().item())
    if hasattr(x, "get"):
        return float(x.get())
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def scalar_to_int(x) -> int:
    """Convert a NumPy/CuPy/Torch scalar to Python int."""
    if hasattr(x, "detach"):
        return int(x.detach().cpu().item())
    if hasattr(x, "get"):
        return int(x.get())
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def draw_random_seed(random_state) -> int:
    """Draw an integer seed from int/None/RandomState/Generator inputs."""
    if random_state is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, np.iinfo(np.int64).max))
    return int(random_state)


def backend_random_normal(backend, random_state, size, scale: float = 1.0):
    """Generate deterministic normal variates directly on the target backend.

    This avoids allocating a NumPy random matrix and then transferring it to a
    GPU backend.  The lightweight Box-Muller generator is used only for
    estimator initialization and randomized projections, where deterministic
    seeded behavior is more important than cryptographic-quality randomness.
    """
    total = int(np.prod(size))
    seed = draw_random_seed(random_state)
    idx = backend.arange(total, dtype=backend.float64)
    xp = backend.xp

    def uniform(offset):
        values = xp.sin((idx + 1.0 + float(offset)) * (12.9898 + 0.001 * float(seed))) * 43758.5453
        return values - xp.floor(values)

    u1 = backend.maximum(uniform(0), 1e-12)
    u2 = uniform(total + 17)
    z = backend.sqrt(-2.0 * backend.log(u1)) * xp.cos(2.0 * np.pi * u2)
    return backend.reshape(z * float(scale), size)


def squared_euclidean_distances(backend, X, Y=None):
    """Compute dense squared Euclidean distances with backend arrays."""
    Y = X if Y is None else Y
    x_norm = backend.sum(X * X, axis=1, keepdims=True)
    y_norm = backend.sum(Y * Y, axis=1, keepdims=True)
    distances = x_norm + y_norm.T - 2.0 * backend.matmul(X, Y.T)
    return backend.maximum(distances, 0.0)


def topk_smallest(backend, distances, k: int):
    """Return the k smallest values and indices along axis 1."""
    order = backend.argsort(distances, axis=1)
    idx = order[:, :k]
    values = backend.take_along_axis(distances, idx, axis=1)
    return values, idx


def svd_flip_components(backend, components):
    """Apply a deterministic sign convention to right singular vectors."""
    max_abs_cols = backend.argmax(backend.abs(components), axis=1)
    rows = backend.arange(components.shape[0])
    signs = backend.where(components[rows, max_abs_cols] < 0.0, -1.0, 1.0)
    return components * backend.reshape(signs, (components.shape[0], 1))


def randomized_svd(
    backend,
    X,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state=None,
):
    """Backend randomized SVD for dense matrices."""
    n_samples, n_features = X.shape
    n_random = min(min(n_samples, n_features), int(n_components) + int(n_oversamples))
    omega = backend_random_normal(backend, random_state, size=(n_features, n_random))

    # Re-orthogonalize each power iteration. This is slightly more work than
    # the raw power method, but greatly improves randomized SVD stability when
    # singular values are close or the matrix is moderately ill-conditioned.
    Q, _ = backend.qr(backend.matmul(X, omega))
    for _ in range(int(n_iter)):
        Q, _ = backend.qr(backend.matmul(X.T, Q))
        Q, _ = backend.qr(backend.matmul(X, Q))

    B = backend.matmul(Q.T, X)
    _, singular_values, vh = backend.svd(B, full_matrices=False)
    return singular_values[:n_components], svd_flip_components(backend, vh[:n_components])


def eye(backend, n: int, dtype=None):
    """Create an identity matrix on the requested backend."""
    if hasattr(backend, "eye"):
        return backend.eye(n, dtype=dtype)
    return backend.asarray(np.eye(n), dtype=dtype or backend.float64)
