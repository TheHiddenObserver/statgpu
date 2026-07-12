from __future__ import annotations
"""Model metadata merge rules."""

# Central registry: canonical primary_category_id per model.
# Order-independent — merge uses this registry, not first-wins.
MODEL_PRIMARY_CATEGORY: dict[str, str] = {
    "CoxPH": "survival",
    "KnockoffFilter": "feature_selection",
    "Lasso": "linear_models",
    "LassoCV": "linear_models",
    "LassoSelector": "feature_selection",
    "MarginalCorrelationSelector": "feature_selection",
    "PenalizedGammaRegression": "penalized_glm",
    "PenalizedInverseGaussianRegression": "penalized_glm",
    "PenalizedLinearRegression": "penalized_glm",
    "PenalizedLogisticRegression": "penalized_glm",
    "PenalizedNegativeBinomialRegression": "penalized_glm",
    "PenalizedPoissonRegression": "penalized_glm",
    "PenalizedTweedieRegression": "penalized_glm",
    "QuantileRegression": "robust_quantile",
    "RobustRegression": "robust_quantile",
    "OrderedLogit": "ordered",
    "OrderedProbit": "ordered",
    "PCA": "unsupervised",
    "KMeans": "unsupervised",
    "GaussianMixture": "unsupervised",
    "NMF": "unsupervised",
    "TruncatedSVD": "unsupervised",
    "IncrementalPCA": "unsupervised",
    "AgglomerativeClustering": "unsupervised",
    "DBSCAN": "unsupervised",
    "UMAP": "unsupervised",
    "TSNE": "unsupervised",
    "MiniBatchKMeans": "unsupervised",
    "MiniBatchNMF": "unsupervised",
    "PanelOLS": "panel",
    "RandomEffects": "panel",
    "GAM": "nonparametric",
    "Nystroem": "nonparametric",
    "RBFKernel": "nonparametric",
    "BSplineBasis": "nonparametric",
    "EmpiricalCovariance": "covariance",
}


def merge_model_entries(existing: dict, incoming: dict) -> dict:
    """Merge model entries from multiple parsers. Order-independent."""
    mid = incoming["model_id"]
    result = dict(existing) if existing else {
        "model_id": mid,
        "primary_category_id": "",
        "category_ids": [],
        "supports_penalty": False,
        "supports_inference": False,
    }

    # category_ids: union
    cat_set = set(result.get("category_ids", []))
    cat_set.update(incoming.get("category_ids", []))
    result["category_ids"] = sorted(cat_set)

    # primary_category_id: from central registry (order-independent)
    if mid in MODEL_PRIMARY_CATEGORY:
        result["primary_category_id"] = MODEL_PRIMARY_CATEGORY[mid]
    elif incoming.get("primary_category_id"):
        result["primary_category_id"] = incoming["primary_category_id"]

    # supports_*: logical OR
    if incoming.get("supports_penalty", False):
        result["supports_penalty"] = True
    if incoming.get("supports_inference", False):
        result["supports_inference"] = True

    return result
