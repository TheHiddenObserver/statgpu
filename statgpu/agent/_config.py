"""Centralized configuration for the statgpu agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


MISSING_STRINGS = frozenset({"", "na", "nan", "null", "none", "missing"})
TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "event", "case", "dead"})
FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "censored", "control", "alive"})


@dataclass
class AgentConfig:
    """Centralized configuration replacing hardcoded thresholds."""

    # Profiler
    max_categories: int = 20

    # Validator
    min_sample_size_warn: int = 30
    condition_number_threshold: float = 1e8
    imbalance_low_threshold: float = 0.1
    imbalance_high_threshold: float = 0.9
    min_events_per_feature: int = 10
    vif_warn_threshold: float = 5.0
    vif_high_threshold: float = 10.0

    # Cross-validation
    cv_folds: int = 5

    # Multiple testing
    multiple_testing_method: str = "none"  # "none", "bh", "by", "holm", "bonferroni", "hochberg"
    alpha: float = 0.05

    # Device and inference
    device: str = "auto"
    cov_type: str = "hc3"
    random_state: Optional[int] = 0

    # Agent behavior
    include_regularized: bool = True
    include_unsupervised_diagnostics: bool = True
    gpu_memory_cleanup: bool = False
    max_correction_rounds: int = 3
