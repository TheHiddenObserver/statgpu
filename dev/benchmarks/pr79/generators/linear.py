"""Data generators for LinearRegression, Ridge, and Panel benchmark cases."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple


def generate_linear_full_rank(
    n_samples: int = 1000,
    n_features: int = 10,
    seed: int = 42,
    noise_std: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate well-conditioned linear regression data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = rng.normal(size=n_features).astype(np.float64)
    y = (X @ beta + rng.normal(scale=noise_std, size=n_samples)).astype(np.float64)
    return X, y, beta


def generate_linear_rank_deficient(
    n_samples: int = 200,
    n_features: int = 6,
    seed: int = 42,
    noise_std: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate rank-deficient design (one collinear column)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    X[:, n_features - 1] = X[:, n_features - 2]  # exact collinearity
    beta = rng.normal(size=n_features).astype(np.float64)
    y = (X @ beta + rng.normal(scale=noise_std, size=n_samples)).astype(np.float64)
    return X, y, beta


def generate_linear_weighted(
    n_samples: int = 500,
    n_features: int = 8,
    seed: int = 42,
    noise_std: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate weighted linear regression data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = rng.normal(size=n_features).astype(np.float64)
    y = (X @ beta + rng.normal(scale=noise_std, size=n_samples)).astype(np.float64)
    weights = rng.uniform(0.5, 2.0, size=n_samples).astype(np.float64)
    return X, y, beta, weights


def generate_coxph_simple(
    n_samples: int = 200,
    n_features: int = 4,
    seed: int = 42,
    event_rate: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate simple CoxPH data with no ties."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.array([0.5, -0.3, 0.2, 0.0], dtype=np.float64)[:n_features]
    eta = X @ beta
    # Generate survival times from exponential with hazard = exp(eta)
    baseline = rng.exponential(scale=1.0, size=n_samples).astype(np.float64)
    time = baseline / np.exp(eta)
    # Random censoring
    censor_time = rng.exponential(scale=np.percentile(time, int(event_rate * 100)),
                                  size=n_samples).astype(np.float64)
    event = (time <= censor_time).astype(np.int32)
    time = np.minimum(time, censor_time)
    return X, time, event, beta


def generate_coxph_ties(
    n_samples: int = 300,
    n_features: int = 4,
    seed: int = 42,
    tie_prob: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate CoxPH data with small tied failure times."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.array([0.5, -0.3, 0.2, 0.0], dtype=np.float64)[:n_features]
    eta = X @ beta
    baseline = rng.exponential(scale=1.0, size=n_samples).astype(np.float64)
    time_raw = baseline / np.exp(eta)
    # Create small tie groups by rounding some times
    mask = rng.random(n_samples) < tie_prob
    time_raw[mask] = np.round(time_raw[mask] * 4) / 4
    # Censor some
    censor_time = rng.exponential(scale=1.5, size=n_samples).astype(np.float64)
    event = (time_raw <= censor_time).astype(np.int32)
    time = np.minimum(time_raw, censor_time)
    return X, time, event, beta


def generate_panel_balanced(
    n_entities: int = 30,
    n_periods: int = 5,
    n_features: int = 3,
    seed: int = 42,
    noise_std: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate balanced panel data."""
    rng = np.random.default_rng(seed)
    n_total = n_entities * n_periods
    X = rng.normal(size=(n_total, n_features)).astype(np.float64)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_entities)
    beta = np.array([1.0, -0.5, 0.3], dtype=np.float64)[:n_features]
    y = (X @ beta + rng.normal(scale=noise_std, size=n_total)).astype(np.float64)
    return X, y, entity, time_idx, beta


def case_params_linear() -> Dict[str, Any]:
    """Canonical case parameters for LinearRegression accuracy."""
    return {
        "domain": "linear",
        "n_samples": 1000,
        "n_features": 10,
        "seed": 42,
        "noise_std": 0.3,
        "rank_regime": "full_rank",
        "weighted": False,
    }


def case_params_linear_weighted() -> Dict[str, Any]:
    return {
        "domain": "linear",
        "n_samples": 500,
        "n_features": 8,
        "seed": 42,
        "noise_std": 0.5,
        "rank_regime": "full_rank",
        "weighted": True,
    }


def case_params_linear_rank_def() -> Dict[str, Any]:
    return {
        "domain": "linear",
        "n_samples": 200,
        "n_features": 6,
        "seed": 42,
        "noise_std": 0.3,
        "rank_regime": "rank_deficient",
        "weighted": False,
    }
