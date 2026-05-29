"""Model comparison and multi-candidate competition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ModelComparison:
    """Side-by-side comparison of fitted models."""

    ranking_metric: str
    ranking: List[Tuple[str, float]]
    best_model: str
    delta_table: List[Dict[str, Any]]

    def to_dict(self):
        return {
            "ranking_metric": self.ranking_metric,
            "ranking": [{"model": name, "score": score} for name, score in self.ranking],
            "best_model": self.best_model,
            "delta_table": self.delta_table,
        }


class ModelComparator:
    """Compare models fitted on the same dataset."""

    def compare(self, models: Sequence, task_type: str) -> Optional[ModelComparison]:
        """Rank models by appropriate metric and compute deltas."""
        successful = [m for m in models if m.error is None and m.metrics]
        if len(successful) < 2:
            return None

        metric_name, higher_is_better = self._ranking_metric(task_type)
        scores = []
        # Map metrics to their sort direction (higher_is_better)
        _direction = {
            "aic": False, "bic": False, "mean_poisson_deviance": False, "inertia": False,
            "roc_auc": True, "accuracy": True, "f1": True, "c_index": True,
            "log_likelihood": True, "score": True, "explained_variance_ratio_sum": True,
        }

        for model in successful:
            score = model.metrics.get(metric_name)
            used_metric = metric_name
            if score is None:
                # Try fallback metrics
                for fallback in self._fallback_metrics(task_type):
                    if fallback in model.metrics:
                        score = model.metrics[fallback]
                        used_metric = fallback
                        break
            if score is not None:
                scores.append((model.name, float(score), used_metric))

        if len(scores) < 2:
            return None

        # Use the direction of the actual metric used (primary or fallback)
        actual_metric = scores[0][2] if scores else metric_name
        actual_direction = _direction.get(actual_metric, higher_is_better)
        scores.sort(key=lambda x: x[1], reverse=actual_direction)
        # Strip metric name from scores for output
        scores = [(name, score) for name, score, _ in scores]
        best_name, best_score = scores[0]

        # Compute deltas
        delta_table = []
        for name, score in scores[1:]:
            delta_table.append({
                "model": name,
                "score": score,
                "delta": score - best_score,
                "delta_pct": ((score - best_score) / abs(best_score) * 100) if best_score != 0 else np.nan,
            })

        return ModelComparison(
            ranking_metric=actual_metric,
            ranking=scores,
            best_model=best_name,
            delta_table=delta_table,
        )

    @staticmethod
    def _ranking_metric(task_type: str) -> Tuple[str, bool]:
        """Return (metric_name, higher_is_better) for a task type."""
        return {
            "regression": ("aic", False),  # lower AIC is better
            "binary_classification": ("roc_auc", True),
            "poisson": ("mean_poisson_deviance", False),
            "survival": ("c_index", True),
            "unsupervised": ("explained_variance_ratio_sum", True),
        }.get(task_type, ("score", True))

    @staticmethod
    def _fallback_metrics(task_type: str) -> List[str]:
        """Fallback metrics when primary is not available."""
        return {
            "regression": ["score", "bic"],
            "binary_classification": ["accuracy", "f1"],
            "poisson": ["score"],
            "survival": ["log_likelihood"],
            "unsupervised": ["inertia"],
        }.get(task_type, ["score"])
