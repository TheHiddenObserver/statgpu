"""Fifth-review closure for post-selection OLS migration edge cases.

This contract keeps two late-discovered public boundaries aligned with #137:

* pre-fit migration detection must recognize public ``Penalty`` objects, not
  only string penalty names, so warning normalization and AUTO native-input
  routing are identical for both constructor forms;
* a no-intercept fit with an empty active set has no coefficient covariance to
  evaluate, but its reporting result must still preserve the caller's requested
  covariance/reference-distribution semantics instead of silently claiming
  ``nonrobust`` Student-t inference.

The project installs review contracts after the primary implementation modules;
patch both the API router's global helper and the implementation module export so
all subsequent public/direct imports observe one runtime contract.
"""

from __future__ import annotations

import functools

from statgpu.linear_model import _penalized_inference_api_contract as _api_contract
from statgpu.linear_model.penalized import _post_selection_ols as _post_selection


_FIFTH_REVIEW_MARKER = "__statgpu_pr138_fifth_review_contract__"
_SPARSE_GAUSSIAN_PENALTIES = frozenset({"l1", "elasticnet", "en"})
_ORIGINAL_POST_SELECTION = _post_selection.compute_post_selection_ols_inference


def _supports_sparse_gaussian_migration(self) -> bool:
    """Recognize sparse Gaussian scope before or after penalty resolution."""
    loss_name = str(getattr(self, "loss", "squared_error")).strip().lower()
    penalty_obj = getattr(self, "_penalty", None)
    if penalty_obj is None:
        penalty_obj = getattr(self, "penalty", "")
    penalty_name = str(getattr(penalty_obj, "name", penalty_obj)).strip().lower()
    return loss_name == "squared_error" and penalty_name in _SPARSE_GAUSSIAN_PENALTIES


@functools.wraps(_ORIGINAL_POST_SELECTION)
def _compute_post_selection_ols_inference(model, X, y, sample_weight=None):
    result = _ORIGINAL_POST_SELECTION(model, X, y, sample_weight=sample_weight)
    metadata = dict(getattr(result, "metadata", {}) or {})

    # With no intercept and no selected features, every public parameter slot is
    # an inactive compatibility placeholder. There is no covariance matrix to
    # compute, but the result must not rewrite an HC/HAC request to nonrobust/t.
    if int(metadata.get("n_selected", -1)) == 0 and not bool(model._effective_intercept):
        cov_type = str(getattr(model, "_cov_type", "nonrobust")).strip().lower()
        distribution = "t" if cov_type == "nonrobust" else "normal"
        result.cov_type = cov_type
        result.distribution = distribution
        result.statistic_name = "t" if distribution == "t" else "z"
        result.df = (
            float(metadata["refit_df_resid"])
            if distribution == "t" and "refit_df_resid" in metadata
            else None
        )
        result.apply_to(model)
    return result


def install_post_selection_ols_fifth_review_contract():
    """Install the fifth-review closure once."""
    current = _api_contract._supports_sparse_gaussian_migration
    if getattr(current, _FIFTH_REVIEW_MARKER, False):
        return

    setattr(_supports_sparse_gaussian_migration, _FIFTH_REVIEW_MARKER, True)
    _api_contract._supports_sparse_gaussian_migration = _supports_sparse_gaussian_migration

    setattr(_compute_post_selection_ols_inference, _FIFTH_REVIEW_MARKER, True)
    _post_selection.compute_post_selection_ols_inference = _compute_post_selection_ols_inference
    _api_contract.compute_post_selection_ols_inference = _compute_post_selection_ols_inference


__all__ = ["install_post_selection_ols_fifth_review_contract"]
