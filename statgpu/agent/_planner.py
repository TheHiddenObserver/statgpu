"""Task inference, method registry, and pruning rules."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._config import AgentConfig


# ---------------------------------------------------------------------------
# Method Registry
# ---------------------------------------------------------------------------

class MethodRegistry:
    """Pluggable registry of analysis methods."""

    _registry: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def register(
        cls,
        task_type: str,
        name: str,
        factory: Callable,
        *,
        priority: int = 0,
        prerequisites: Optional[List[str]] = None,
    ):
        """Register an analysis method for a task type."""
        cls._registry.setdefault(task_type, []).append({
            "name": name,
            "factory": factory,
            "priority": priority,
            "prerequisites": prerequisites or [],
        })

    @classmethod
    def get_methods(cls, task_type: str) -> List[Dict[str, Any]]:
        """Return registered methods for a task type, sorted by priority."""
        candidates = cls._registry.get(task_type, [])
        return sorted(candidates, key=lambda x: x["priority"])

    @classmethod
    def get_method_names(cls, task_type: str) -> List[str]:
        """Return method names for a task type."""
        return [m["name"] for m in cls.get_methods(task_type)]


class TaskRegistry:
    """Pluggable registry of task types."""

    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, inference_fn: Callable, description: str = ""):
        """Register a new task type."""
        cls._registry[name] = {
            "inference_fn": inference_fn,
            "description": description,
        }

    @classmethod
    def infer(cls, prepared) -> Optional[str]:
        """Try to infer task type from registered inference functions."""
        for name, entry in cls._registry.items():
            if entry["inference_fn"](prepared):
                return name
        return None


class PruningRuleRegistry:
    """Pluggable registry of pruning rules."""

    _rules: Dict[str, List[Tuple[Callable, str]]] = {}

    @classmethod
    def register(cls, method_name: str, rule: Callable, reason: str):
        """Register a pruning rule for a method."""
        cls._rules.setdefault(method_name, []).append((rule, reason))

    @classmethod
    def should_exclude(cls, method_name: str, prepared) -> Optional[str]:
        """Check if a method should be excluded. Returns reason or None."""
        for rule, reason in cls._rules.get(method_name, []):
            if rule(prepared):
                return reason
        return None


# ---------------------------------------------------------------------------
# Default method registrations
# ---------------------------------------------------------------------------

def _register_default_methods():
    """Register the default set of methods."""
    from statgpu.linear_model import LinearRegression, LogisticRegression, PoissonRegression, Ridge
    from statgpu.survival import CoxPH
    from statgpu.unsupervised import KMeans, PCA

    MethodRegistry.register("regression", "LinearRegression",
                            factory=lambda: LinearRegression(), priority=0)
    MethodRegistry.register("regression", "Ridge(alpha=1.0)",
                            factory=lambda: Ridge(alpha=1.0), priority=1)

    MethodRegistry.register("binary_classification", "LogisticRegression",
                            factory=lambda: LogisticRegression(), priority=0)

    MethodRegistry.register("poisson", "PoissonRegression",
                            factory=lambda: PoissonRegression(), priority=0)

    MethodRegistry.register("survival", "CoxPH",
                            factory=lambda: CoxPH(ties="efron"), priority=0)

    MethodRegistry.register("unsupervised", "PCA",
                            factory=lambda: PCA(), priority=0)
    MethodRegistry.register("unsupervised", "KMeans",
                            factory=lambda: KMeans(), priority=1)


# Initialize default methods on module load
_register_default_methods()


# ---------------------------------------------------------------------------
# Method Pruner
# ---------------------------------------------------------------------------

class MethodPruner:
    """Proactively exclude unsuitable methods based on data characteristics."""

    def prune(self, prepared, task_type: str, config: AgentConfig) -> List[str]:
        """Return list of method names that are suitable for the given data."""
        candidates = MethodRegistry.get_method_names(task_type)
        if not candidates:
            candidates = self._fallback_candidates(task_type)

        n, p = prepared.X.shape

        # High-dimensional: p > n
        if p > n:
            candidates = [c for c in candidates if c not in ("LinearRegression",)]
            for m in ("Lasso", "Ridge", "ElasticNet"):
                if m not in candidates:
                    candidates.insert(0, m)

        # Near-high-dimensional: p > n/2
        elif p > n / 2:
            candidates = [c for c in candidates if c != "LinearRegression"]
            if "Ridge" not in candidates:
                candidates.insert(0, "Ridge")

        # High condition number
        try:
            cond = float(np.linalg.cond(prepared.X))
            if np.isfinite(cond) and cond > 1e10:
                candidates = [c for c in candidates if c != "LinearRegression"]
                if "Ridge" not in candidates:
                    candidates.insert(0, "Ridge")
        except Exception:
            pass

        # Extreme class imbalance for binary classification
        if task_type == "binary_classification" and prepared.y is not None:
            y = prepared.y[np.isfinite(prepared.y)]
            pos_rate = float(np.mean(y > 0.5))
            if pos_rate < 0.05 or pos_rate > 0.95:
                # Keep LogisticRegression but flag it
                pass

        # Insufficient events for survival
        if task_type == "survival" and prepared.event is not None:
            events = int(np.sum(prepared.event == 1))
            if events < p:
                candidates = [c for c in candidates if c != "CoxPH"]
                if "CoxPH(penalized)" not in candidates:
                    candidates.insert(0, "CoxPH(penalized)")

        # Apply custom pruning rules
        pruned = []
        for name in candidates:
            reason = PruningRuleRegistry.should_exclude(name, prepared)
            if reason is None:
                pruned.append(name)

        return pruned if pruned else candidates[:1]  # fallback to first candidate

    @staticmethod
    def _fallback_candidates(task_type: str) -> List[str]:
        """Fallback candidates when registry is empty."""
        return {
            "regression": ["LinearRegression", "Ridge"],
            "binary_classification": ["LogisticRegression"],
            "poisson": ["PoissonRegression"],
            "survival": ["CoxPH"],
            "unsupervised": ["PCA", "KMeans"],
        }.get(task_type, [])


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------

_TASK_ALIASES = {
    "auto": "auto",
    "regression": "regression",
    "linear": "regression",
    "classification": "binary_classification",
    "binary": "binary_classification",
    "binary_classification": "binary_classification",
    "poisson": "poisson",
    "count": "poisson",
    "survival": "survival",
    "cox": "survival",
    "unsupervised": "unsupervised",
    "clustering": "unsupervised",
}


def infer_task(prepared, requested: str) -> str:
    """Infer task type from data characteristics or user request."""
    key = requested.lower()
    if key not in _TASK_ALIASES:
        raise ValueError(f"Unknown task='{requested}'.")
    task = _TASK_ALIASES[key]
    if task != "auto":
        return task

    # Auto-detect from registered task types first
    registered = TaskRegistry.infer(prepared)
    if registered is not None:
        return registered

    # Default auto-detection logic
    if prepared.time is not None and prepared.event is not None:
        return "survival"
    if prepared.y is None:
        return "unsupervised"

    y = prepared.y[np.isfinite(prepared.y)]
    unique = np.unique(y)
    if unique.size == 2:
        return "binary_classification"
    if (
        unique.size > 2
        and np.all(y >= 0)
        and np.allclose(y, np.round(y), atol=1e-8)
        and unique.size <= max(20, int(0.5 * y.size))
    ):
        return "poisson"
    return "regression"


def build_plan(task_type: str, prepared, available_methods: List[str],
               config: AgentConfig) -> "AnalysisPlan":
    """Build an analysis plan from available methods."""
    from ._analysis import AnalysisPlan

    agents = ["profiler", "planner", "statgpu_runner", "self_validator", "reporter"]
    methods = list(available_methods)

    rationale = [
        "The profiler selected the task from the target and survival fields.",
        "The runner uses statgpu estimators with the configured device policy.",
        "The validator checks missingness, rank, conditioning, class balance, and fit outputs.",
    ]
    if config.cov_type:
        rationale.append(f"Inference models use cov_type='{config.cov_type}' where supported.")

    return AnalysisPlan(task_type=task_type, agents=agents, methods=methods, rationale=rationale)
