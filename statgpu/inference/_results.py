"""Shared inference result containers.

These classes are intentionally lightweight.  They describe how inference
results are carried, serialized, and synchronized back to estimator attributes;
model-specific inference algorithms live in the model/helper modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _to_numpy_or_none(value):
    if value is None:
        return None
    return np.asarray(value)


def _serializable(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


@dataclass
class BaseInferenceResult:
    """Base class for model inference results."""

    method: str
    feature_names: Optional[Sequence[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply_to(self, estimator):
        """Attach this result object to an estimator."""
        estimator._inference_result = self
        return estimator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_type": self.__class__.__name__,
            "method": self.method,
            "feature_names": list(self.feature_names) if self.feature_names is not None else None,
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc
        return pd.DataFrame([self.to_dict()])


@dataclass
class ParameterInferenceResult(BaseInferenceResult):
    """Parameter-level inference result.

    This class only promises per-parameter estimates/statistics.  It does not
    imply a joint Wald test or a complete precision matrix.
    """

    params: Any = None
    bse: Any = None
    statistic: Any = None
    statistic_name: str = "statistic"
    pvalues: Any = None
    conf_int: Any = None
    cov_type: Optional[str] = None
    distribution: Optional[str] = None
    df: Optional[float] = None

    def __post_init__(self):
        self.params = _to_numpy_or_none(self.params)
        self.bse = _to_numpy_or_none(self.bse)
        self.statistic = _to_numpy_or_none(self.statistic)
        self.pvalues = _to_numpy_or_none(self.pvalues)
        self.conf_int = _to_numpy_or_none(self.conf_int)

    def apply_to(self, estimator):
        super().apply_to(estimator)
        estimator._params = None if self.params is None else np.asarray(self.params).copy()
        estimator._bse = None if self.bse is None else np.asarray(self.bse).copy()
        if self.statistic is not None:
            stat = np.asarray(self.statistic).copy()
            if self.statistic_name == "z":
                estimator._zvalues = stat
                estimator._tvalues = stat  # backward compat: z ≈ t for large n
            elif self.statistic_name == "t":
                estimator._tvalues = stat
                estimator._zvalues = stat  # backward compat
            else:
                estimator._statistic = stat
                if hasattr(estimator, "_zvalues"):
                    estimator._zvalues = None
        estimator._pvalues = None if self.pvalues is None else np.asarray(self.pvalues).copy()
        estimator._conf_int = None if self.conf_int is None else np.asarray(self.conf_int).copy()
        return estimator

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "params": _serializable(self.params),
                "bse": _serializable(self.bse),
                "statistic": _serializable(self.statistic),
                "statistic_name": self.statistic_name,
                "pvalues": _serializable(self.pvalues),
                "conf_int": _serializable(self.conf_int),
                "cov_type": self.cov_type,
                "distribution": self.distribution,
                "df": self.df,
            }
        )
        return data

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        params = np.asarray(self.params)
        if params.ndim != 1:
            raise ValueError("to_dataframe() is only supported for one-dimensional parameter results.")
        names = (
            list(self.feature_names)
            if self.feature_names is not None
            else [f"param_{i}" for i in range(params.shape[0])]
        )
        data = {
            "term": names,
            "estimate": params,
        }
        if self.bse is not None:
            data["std_error"] = np.asarray(self.bse)
        if self.statistic is not None:
            data[self.statistic_name] = np.asarray(self.statistic)
        if self.pvalues is not None:
            data["pvalue"] = np.asarray(self.pvalues)
        if self.conf_int is not None:
            ci = np.asarray(self.conf_int)
            data["conf_low"] = ci[:, 0]
            data["conf_high"] = ci[:, 1]
        return pd.DataFrame(data)


@dataclass
class GaussianInferenceResult(ParameterInferenceResult):
    """Gaussian linear-model parameter inference result."""

    method: str = "classical"
    statistic_name: str = "t"
    distribution: Optional[str] = "t"

    @property
    def tvalues(self):
        return self.statistic


@dataclass
class DebiasedInferenceResult(ParameterInferenceResult):
    """Placeholder result type for debiased parameter inference."""

    method: str = "debiased"
    statistic_name: str = "z"
    distribution: Optional[str] = "normal"
    precision_method: Optional[str] = None
    simultaneous_conf_int: Any = None
    simultaneous_method: Optional[str] = None
    simultaneous_alpha: Optional[float] = None
    simultaneous_n_bootstrap: Optional[int] = None
    simultaneous_critical_value: Optional[float] = None
    simultaneous_target_mask: Any = None

    def __post_init__(self):
        super().__post_init__()
        self.simultaneous_conf_int = _to_numpy_or_none(self.simultaneous_conf_int)
        self.simultaneous_target_mask = _to_numpy_or_none(self.simultaneous_target_mask)

    def apply_to(self, estimator):
        super().apply_to(estimator)
        if self.statistic is not None:
            stat = np.asarray(self.statistic).copy()
            estimator._zvalues = stat
            # Existing Lasso summary code displays debiased z-statistics through
            # the legacy _tvalues slot.
            estimator._tvalues = stat
        if self.simultaneous_conf_int is not None:
            estimator._conf_int_simultaneous = np.asarray(self.simultaneous_conf_int).copy()
            estimator._simultaneous_enabled = True
            estimator._simultaneous_method = self.simultaneous_method
            estimator._simultaneous_alpha = self.simultaneous_alpha
            estimator._simultaneous_n_bootstrap = self.simultaneous_n_bootstrap
            estimator._simultaneous_critical_value = self.simultaneous_critical_value
            estimator._simultaneous_target_mask = (
                None
                if self.simultaneous_target_mask is None
                else np.asarray(self.simultaneous_target_mask).copy()
            )
        return estimator

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["precision_method"] = self.precision_method
        data.update(
            {
                "simultaneous_conf_int": _serializable(self.simultaneous_conf_int),
                "simultaneous_method": self.simultaneous_method,
                "simultaneous_alpha": self.simultaneous_alpha,
                "simultaneous_n_bootstrap": self.simultaneous_n_bootstrap,
                "simultaneous_critical_value": self.simultaneous_critical_value,
                "simultaneous_target_mask": _serializable(self.simultaneous_target_mask),
            }
        )
        return data


@dataclass
class OracleActiveSetInferenceResult(ParameterInferenceResult):
    """Placeholder result type for active-set/oracle-style inference."""

    method: str = "active_set"
    statistic_name: str = "z"
    distribution: Optional[str] = "normal"
    active_set: Any = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["active_set"] = _serializable(self.active_set)
        return data


@dataclass
class ResamplingInferenceResult(BaseInferenceResult):
    """Placeholder result type for bootstrap/permutation-style inference."""

    samples: Any = None
    observed: Any = None
    confidence_interval: Any = None
    pvalue: Optional[float] = None

    def __post_init__(self):
        self.samples = _to_numpy_or_none(self.samples)
        self.observed = _to_numpy_or_none(self.observed)
        self.confidence_interval = _to_numpy_or_none(self.confidence_interval)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "samples": _serializable(self.samples),
                "observed": _serializable(self.observed),
                "confidence_interval": _serializable(self.confidence_interval),
                "pvalue": self.pvalue,
            }
        )
        return data
