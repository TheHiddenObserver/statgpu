"""Internal structured panel metadata and result substrate.

Stage A of issue #93 establishes these containers without exposing new public
diagnostic methods. Stage B can populate the optional statistic fields after
the corresponding econometric definitions and applicability rules are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from statgpu.backends import _to_numpy


@dataclass(frozen=True)
class PanelTestResult:
    """Structured representation for a panel diagnostic test result."""

    statistic: Optional[float] = None
    pvalue: Optional[float] = None
    distribution: Optional[str] = None
    df: Optional[Union[float, Tuple[float, ...]]] = None
    null: Optional[str] = None
    alternative: Optional[str] = None
    applicable: bool = False
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PanelFitStatistics:
    """Optional fit-statistics substrate for Stage-B panel diagnostics."""

    rsquared_within: Optional[float] = None
    rsquared_between: Optional[float] = None
    rsquared_overall: Optional[float] = None
    rsquared_adj: Optional[float] = None
    f_statistic: Optional[float] = None
    f_pvalue: Optional[float] = None
    f_df: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PanelIndexInfo:
    """CPU metadata describing entity/time structure without reordering rows."""

    nobs: int
    entity_codes: Optional[np.ndarray] = None
    entity_labels: Optional[np.ndarray] = None
    entity_counts: Optional[np.ndarray] = None
    time_codes: Optional[np.ndarray] = None
    time_labels: Optional[np.ndarray] = None
    time_counts: Optional[np.ndarray] = None
    n_entities: Optional[int] = None
    n_times: Optional[int] = None
    is_balanced: Optional[bool] = None
    has_duplicate_pairs: Optional[bool] = None
    original_order: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


def _factorize_metadata(values, name: str, nobs: int):
    if values is None:
        return None, None, None
    # Entity/time identifiers are metadata and may safely be factorized on the
    # host. Use the common backend conversion so CuPy/Torch CUDA labels do not
    # rely on an invalid implicit ``np.asarray`` device transfer.
    arr = np.asarray(_to_numpy(values))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if arr.shape[0] != nobs:
        raise ValueError(f"{name} must have {nobs} observations")
    try:
        labels, codes = np.unique(arr, return_inverse=True)
    except TypeError as exc:
        raise ValueError(f"{name} must contain mutually comparable labels") from exc
    counts = np.bincount(codes, minlength=len(labels)).astype(np.int64, copy=False)
    return codes.astype(np.int64, copy=False), labels, counts


def build_panel_index_info(
    nobs: int,
    *,
    entity_ids=None,
    time_ids=None,
) -> PanelIndexInfo:
    """Build shared panel-structure metadata without changing row order.

    Numerical design/response arrays are deliberately absent from this helper.
    Entity/time labels are observation metadata and may be factorized on CPU.
    """

    nobs = int(nobs)
    if nobs <= 0:
        raise ValueError("nobs must be positive")

    entity_codes, entity_labels, entity_counts = _factorize_metadata(
        entity_ids, "entity_ids", nobs
    )
    time_codes, time_labels, time_counts = _factorize_metadata(
        time_ids, "time_ids", nobs
    )

    n_entities = None if entity_labels is None else int(len(entity_labels))
    n_times = None if time_labels is None else int(len(time_labels))
    is_balanced = None
    has_duplicate_pairs = None

    if entity_codes is not None and time_codes is not None:
        pair_codes = entity_codes.astype(np.int64) * max(n_times, 1) + time_codes
        has_duplicate_pairs = bool(np.unique(pair_codes).size != nobs)
        if not has_duplicate_pairs and n_entities and n_times:
            is_balanced = bool(nobs == n_entities * n_times)
        else:
            is_balanced = False

    return PanelIndexInfo(
        nobs=nobs,
        entity_codes=entity_codes,
        entity_labels=entity_labels,
        entity_counts=entity_counts,
        time_codes=time_codes,
        time_labels=time_labels,
        time_counts=time_counts,
        n_entities=n_entities,
        n_times=n_times,
        is_balanced=is_balanced,
        has_duplicate_pairs=has_duplicate_pairs,
        original_order=np.arange(nobs, dtype=np.int64),
    )
