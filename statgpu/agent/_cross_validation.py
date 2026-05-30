"""Cross-validation evaluation for the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np


@dataclass
class CVResult:
    """Cross-validation result."""

    n_folds: int
    metric_name: str
    fold_scores: List[float]
    mean: float
    std: float
    ci_low: float
    ci_high: float

    def to_dict(self):
        return {
            "n_folds": self.n_folds,
            "metric_name": self.metric_name,
            "fold_scores": self.fold_scores,
            "mean": self.mean,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def _concordance_index(time: np.ndarray, risk: np.ndarray, event: np.ndarray) -> float:
    """Compute concordance index (C-index) for survival data."""
    n = len(time)
    concordant = 0
    permissible = 0
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] != time[j]:
                if time[i] < time[j] and event[i] == 1:
                    permissible += 1
                    if risk[i] > risk[j]:
                        concordant += 1
                elif time[j] < time[i] and event[j] == 1:
                    permissible += 1
                    if risk[j] > risk[i]:
                        concordant += 1
    return concordant / permissible if permissible > 0 else np.nan


def _kfold_indices(n: int, n_folds: int, random_state: Optional[int] = None) -> List[tuple]:
    """Generate K-fold train/test indices."""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1
    folds = []
    current = 0
    for fold_size in fold_sizes:
        test_idx = indices[current : current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size :]])
        folds.append((train_idx, test_idx))
        current += fold_size
    return folds


class AgentCrossValidator:
    """Cross-validation evaluator for the agent pipeline."""

    def __init__(self, n_folds: int = 5, random_state: Optional[int] = 0):
        self.n_folds = n_folds
        self.random_state = random_state

    def evaluate_supervised(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
    ) -> CVResult:
        """Run K-fold CV and return aggregated metrics."""
        folds = _kfold_indices(X.shape[0], self.n_folds, self.random_state)
        scores = []
        metric_name = self._metric_name(task_type)

        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model = model_factory()
                model.fit(X_train, y_train)
                # Compute the metric that matches metric_name
                if metric_name == "roc_auc" and hasattr(model, "roc_auc_score"):
                    score = model.roc_auc_score(X_test, y_test)
                else:
                    score = model.score(X_test, y_test)
                scores.append(float(score))
            except Exception:
                scores.append(np.nan)

        scores_arr = np.array(scores)
        valid = scores_arr[np.isfinite(scores_arr)]
        if valid.size == 0:
            return CVResult(
                n_folds=self.n_folds,
                metric_name="score",
                fold_scores=scores,
                mean=np.nan,
                std=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
            )

        mean = float(np.mean(valid))
        std = float(np.std(valid))
        # Use t-distribution for small samples (more accurate than z=1.96)
        try:
            from scipy.stats import t as t_dist
            t_crit = float(t_dist.ppf(0.975, df=valid.size - 1))
        except ImportError:
            # Fallback: approximate t-critical for small df
            t_crit = 2.776 if valid.size <= 5 else 2.571 if valid.size <= 10 else 2.262 if valid.size <= 20 else 2.093 if valid.size <= 30 else 1.96
        ci_low = mean - t_crit * std / np.sqrt(valid.size)
        ci_high = mean + t_crit * std / np.sqrt(valid.size)

        return CVResult(
            n_folds=self.n_folds,
            metric_name=self._metric_name(task_type),
            fold_scores=scores,
            mean=mean,
            std=std,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    def evaluate_survival(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> CVResult:
        """Run K-fold CV for survival models, returning C-index on test folds."""
        folds = _kfold_indices(X.shape[0], self.n_folds, self.random_state)
        scores = []

        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            t_train, t_test = time[train_idx], time[test_idx]
            e_train, e_test = event[train_idx], event[test_idx]

            try:
                model = model_factory()
                model.fit(X_train, t_train, e_train)
                # Compute C-index on TEST fold, not training
                risk = np.dot(X_test, model.coef_)
                cindex = _concordance_index(t_test, risk, e_test)
                scores.append(float(cindex))
            except Exception:
                scores.append(np.nan)

        scores_arr = np.array(scores)
        valid = scores_arr[np.isfinite(scores_arr)]
        if valid.size == 0:
            return CVResult(
                n_folds=self.n_folds,
                metric_name="c_index",
                fold_scores=scores,
                mean=np.nan, std=np.nan, ci_low=np.nan, ci_high=np.nan,
            )

        mean = float(np.mean(valid))
        std = float(np.std(valid))
        try:
            from scipy.stats import t as t_dist
            t_crit = float(t_dist.ppf(0.975, df=valid.size - 1))
        except ImportError:
            t_crit = 2.776 if valid.size <= 5 else 2.571 if valid.size <= 10 else 2.262 if valid.size <= 20 else 2.093 if valid.size <= 30 else 1.96
        return CVResult(
            n_folds=self.n_folds,
            metric_name="c_index",
            fold_scores=scores,
            mean=mean,
            std=std,
            ci_low=mean - t_crit * std / np.sqrt(valid.size),
            ci_high=mean + t_crit * std / np.sqrt(valid.size),
        )

    @staticmethod
    def _metric_name(task_type: str) -> str:
        return {
            "regression": "r2",
            "binary_classification": "roc_auc",
            "poisson": "deviance",
            "survival": "c_index",
        }.get(task_type, "score")
