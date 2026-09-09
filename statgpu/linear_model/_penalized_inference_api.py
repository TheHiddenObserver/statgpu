"""Shared public-value normalization for penalized coefficient inference.

Statistical method identity is intentionally independent of execution hardware.
The canonical active-set diagnostic is ``post_selection_ols``; historical
hardware-bearing spellings are compatibility aliases only.
"""

from __future__ import annotations

POST_SELECTION_OLS = "post_selection_ols"

# Values accepted by the current unified penalized estimator surface.
POST_SELECTION_OLS_ALIASES = frozenset({"cpu_ols", "gpu_ols"})

# LassoCV still exposed the older legacy-Lasso spellings before the unified
# public surface was cleaned up. Keep them accepted only at that CV boundary.
LASSOCV_LEGACY_POST_SELECTION_OLS_ALIASES = frozenset(
    {"cpu_ols_inference", "gpu_ols_inference"}
)


def normalize_penalized_inference_method(
    value,
    *,
    allow_lassocv_legacy: bool = False,
) -> str:
    """Return the runtime inference-method name for a public spelling."""
    normalized = str(value).strip().lower()
    aliases = POST_SELECTION_OLS_ALIASES
    if allow_lassocv_legacy:
        aliases = aliases | LASSOCV_LEGACY_POST_SELECTION_OLS_ALIASES
    if normalized in aliases:
        return POST_SELECTION_OLS
    return normalized


def deprecated_post_selection_alias(
    value,
    *,
    allow_lassocv_legacy: bool = False,
):
    """Return the deprecated alias spelling, or ``None`` when canonical."""
    normalized = str(value).strip().lower()
    aliases = POST_SELECTION_OLS_ALIASES
    if allow_lassocv_legacy:
        aliases = aliases | LASSOCV_LEGACY_POST_SELECTION_OLS_ALIASES
    return normalized if normalized in aliases else None


__all__ = [
    "POST_SELECTION_OLS",
    "POST_SELECTION_OLS_ALIASES",
    "LASSOCV_LEGACY_POST_SELECTION_OLS_ALIASES",
    "normalize_penalized_inference_method",
    "deprecated_post_selection_alias",
]
