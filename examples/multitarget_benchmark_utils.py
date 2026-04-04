from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


@dataclass
class MetricRow:
    framework: str
    fit_ms: float
    pred_ms: float
    mean_r2: float
    rmse: float
    mean_abs_coef_err: float
    max_abs_coef_err: float
    mean_abs_intercept_err: float
    max_abs_intercept_err: float
    notes: str = ""

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def make_multitarget_data(
    seed: int,
    n_samples: int,
    n_features: int,
    n_targets: int,
    noise_std: float,
):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_coef = rng.normal(size=(n_targets, n_features))
    true_intercept = rng.normal(size=(n_targets,))
    Y = X @ true_coef.T + true_intercept + rng.normal(scale=noise_std, size=(n_samples, n_targets))
    return X, Y, true_coef, true_intercept


def split_train_test(X, Y, test_ratio: float, seed: int):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


def mean_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(np.mean(r2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def coef_error(pred_coef: np.ndarray, true_coef: np.ndarray):
    diff = np.abs(np.asarray(pred_coef) - np.asarray(true_coef))
    return float(np.mean(diff)), float(np.max(diff))


def intercept_error(pred_intercept: np.ndarray, true_intercept: np.ndarray):
    diff = np.abs(np.asarray(pred_intercept) - np.asarray(true_intercept))
    return float(np.mean(diff)), float(np.max(diff))
