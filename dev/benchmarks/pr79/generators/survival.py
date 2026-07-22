"""Survival data generators for CoxPH benchmark cases."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, Optional


def generate_coxph_no_ties(
    n_samples: int = 200,
    n_features: int = 4,
    seed: int = 42,
    event_rate: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Continuous times with no tied failures (Efron = Breslow)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.array([0.5, -0.3, 0.2, 0.0], dtype=np.float64)[:n_features]
    eta = X @ beta
    baseline = rng.exponential(scale=1.0, size=n_samples).astype(np.float64)
    time_raw = baseline / np.exp(eta)
    censor_time = rng.exponential(scale=np.percentile(time_raw, int(event_rate * 100)),
                                  size=n_samples).astype(np.float64)
    event = (time_raw <= censor_time).astype(np.int32)
    time = np.minimum(time_raw, censor_time)
    return X, time, event, beta


def generate_coxph_small_ties(
    n_samples: int = 300,
    n_features: int = 4,
    seed: int = 42,
    tie_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small tie groups (size 2-4) for Efron exactness testing."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.array([0.5, -0.3, 0.2, 0.0], dtype=np.float64)[:n_features]
    eta = X @ beta
    baseline = rng.exponential(scale=1.0, size=n_samples).astype(np.float64)
    time_raw = baseline / np.exp(eta)
    # Create small tie groups by discretizing some times
    n_groups = n_samples // tie_size
    for i in range(0, n_groups * tie_size, tie_size):
        time_raw[i:i + tie_size] = np.median(time_raw[i:i + tie_size])
    censor_time = rng.exponential(scale=2.0, size=n_samples).astype(np.float64)
    event = (time_raw <= censor_time).astype(np.int32)
    time = np.minimum(time_raw, censor_time)
    return X, time, event, beta


def generate_coxph_entry(
    n_samples: int = 200,
    n_features: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delayed entry (left truncation) data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.array([0.5, -0.3, 0.2, 0.0], dtype=np.float64)[:n_features]
    eta = X @ beta
    baseline = rng.exponential(scale=1.0, size=n_samples).astype(np.float64)
    time_raw = baseline / np.exp(eta)
    entry = rng.exponential(scale=0.5, size=n_samples).astype(np.float64)
    # Only keep observations where entry < time (truncation)
    valid = entry < time_raw
    time_raw = time_raw[valid]
    entry = entry[valid]
    X = X[valid]
    censor_time = rng.exponential(scale=2.0, size=time_raw.shape[0]).astype(np.float64)
    event = (time_raw <= censor_time).astype(np.int32)
    time = np.minimum(time_raw, censor_time)
    return X, time, event, entry[:X.shape[0]], beta


def generate_coxph_penalized(
    n_samples: int = 100,
    n_features: int = 8,
    seed: int = 42,
    penalty: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Data for penalized Cox fit with moderate p relative to n."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.1, 0.0, 0.0], dtype=np.float64)[:n_features]
    eta = X @ beta
    baseline = rng.exponential(scale=1.0, size=n_samples).astype(np.float64)
    time_raw = baseline / np.exp(eta)
    censor_time = rng.exponential(scale=1.5, size=n_samples).astype(np.float64)
    event = (time_raw <= censor_time).astype(np.int32)
    time = np.minimum(time_raw, censor_time)
    return X, time, event, beta


def case_params_coxph_no_ties() -> Dict[str, Any]:
    return {"domain": "survival", "n_samples": 200, "n_features": 4,
            "seed": 42, "ties": "efron", "entry": False, "penalty": 0.0}


def case_params_coxph_small_ties() -> Dict[str, Any]:
    return {"domain": "survival", "n_samples": 300, "n_features": 4,
            "seed": 42, "ties": "efron", "tie_size": 3, "entry": False,
            "penalty": 0.0}


def case_params_coxph_entry() -> Dict[str, Any]:
    return {"domain": "survival", "n_samples": 200, "n_features": 4,
            "seed": 42, "ties": "efron", "entry": True, "penalty": 0.0}


def case_params_coxph_penalized() -> Dict[str, Any]:
    return {"domain": "survival", "n_samples": 100, "n_features": 8,
            "seed": 42, "ties": "efron", "entry": False, "penalty": 0.1}
