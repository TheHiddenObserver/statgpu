"""Panel data generators for PooledOLS, FE, RE, Between, FD, FamaMacBeth."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple


def generate_pooled_balanced(
    n_entities: int = 30,
    n_periods: int = 5,
    n_features: int = 3,
    seed: int = 42,
    noise_std: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Balanced panel with numeric entity/time labels."""
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    X = rng.normal(size=(n, n_features)).astype(np.float64)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_entities)
    beta = np.array([1.0, -0.5, 0.3], dtype=np.float64)[:n_features]
    y = (X @ beta + rng.normal(scale=noise_std, size=n)).astype(np.float64)
    return X, y, entity, time_idx, beta


def generate_unbalanced_panel(
    n_entities: int = 40,
    max_periods: int = 6,
    n_features: int = 3,
    seed: int = 43,
    noise_std: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unbalanced panel with random missing periods."""
    rng = np.random.default_rng(seed)
    rows = []
    entity_ids = []
    time_ids = []
    for i in range(n_entities):
        n_periods_i = rng.integers(2, max_periods + 1)
        X_i = rng.normal(size=(n_periods_i, n_features)).astype(np.float64)
        rows.append(X_i)
        entity_ids.append(np.full(n_periods_i, i, dtype=np.int64))
        time_ids.append(np.arange(n_periods_i, dtype=np.int64))
    X = np.vstack(rows)
    entity = np.concatenate(entity_ids)
    time_idx = np.concatenate(time_ids)
    beta = np.array([0.8, -0.4, 0.5], dtype=np.float64)[:n_features]
    y = (X @ beta + rng.normal(scale=noise_std, size=len(entity))).astype(np.float64)
    return X, y, entity, time_idx, beta


def generate_pooled_cluster(
    n_entities: int = 20,
    n_periods: int = 8,
    n_features: int = 2,
    seed: int = 44,
    noise_std: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Panel with string cluster labels for clustered covariance."""
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    X = rng.normal(size=(n, n_features)).astype(np.float64)
    entity_int = np.repeat(np.arange(n_entities), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_entities)
    cluster = np.array([f"firm_{i}" for i in entity_int])
    beta = np.array([1.2, -0.6], dtype=np.float64)[:n_features]
    y = (X @ beta + rng.normal(scale=noise_std, size=n)).astype(np.float64)
    return X, y, entity_int, time_idx, cluster


def generate_pooled_rank_def(
    n_entities: int = 20,
    n_periods: int = 5,
    n_features: int = 4,
    seed: int = 45,
    noise_std: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Panel with collinear design (rank-deficient)."""
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    X = rng.normal(size=(n, n_features)).astype(np.float64)
    X[:, 3] = X[:, 2]  # exact collinearity
    entity = np.repeat(np.arange(n_entities), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_entities)
    beta = np.array([1.0, -0.5, 0.3, 0.0], dtype=np.float64)[:n_features]
    y = (X @ beta + rng.normal(scale=noise_std, size=n)).astype(np.float64)
    return X, y, entity, time_idx


def case_params_pooled() -> Dict[str, Any]:
    return {"domain": "panel", "n_entities": 30, "n_periods": 5,
            "n_features": 3, "seed": 42, "balanced": True}


def case_params_pooled_rank_def() -> Dict[str, Any]:
    return {"domain": "panel", "n_entities": 20, "n_periods": 5,
            "n_features": 4, "seed": 45, "rank_deficient": True}
