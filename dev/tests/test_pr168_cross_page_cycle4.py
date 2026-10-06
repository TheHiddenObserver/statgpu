"""Cross-page public claims corrected by the independent fourth review."""

from pathlib import Path

import numpy as np
import pytest

import statgpu
from statgpu import unsupervised
from statgpu.inference import combine_pvalues
from statgpu.unsupervised import PCA

ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "agglomerative-clustering", "dbscan", "gaussian-mixture", "incremental-pca",
    "kmeans", "minibatch-kmeans", "minibatch-nmf", "nmf", "pca",
    "truncated-svd", "tsne", "umap",
)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_public_inventory_distinguishes_internal_neighbor_search(language):
    text = (ROOT / f"docs/{language}/guides/implemented-methods.md").read_text()
    assert not hasattr(statgpu, "NNDescent")
    assert not hasattr(unsupervised, "NNDescent")
    assert 'nn_method="nndescent"' in text
    assert "../unsupervised/umap.md" in text
    assert "`UMAP`, `TSNE`, `NNDescent`" not in text
    assert "`UMAP`、`TSNE`、`NNDescent`" not in text
    assert ("not a separate public estimator" in text if language == "en"
            else "不是单独公开导出的估计器" in text)


@pytest.mark.parametrize("solver", ["full", "randomized"])
def test_whitening_uses_fitted_scales_without_promising_exact_randomized_variance(solver):
    X = np.random.default_rng(919).normal(size=(80, 30))
    model = PCA(n_components=2, svd_solver=solver, whiten=True,
                n_oversamples=0, iterated_power=0, random_state=3, device="cpu")
    scores = model.fit_transform(X)
    expected = (X - model.mean_) @ model.components_.T
    expected /= np.sqrt(model.explained_variance_)
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    if solver == "full":
        np.testing.assert_allclose(np.cov(scores, rowvar=False), np.eye(2), atol=1e-12)
    assert "randomized decomposition only approximates" in PCA.__doc__
    assert "positive retained variances" in PCA.__doc__
    for language in ("en", "cn"):
        text = (ROOT / f"docs/{language}/unsupervised/pca.md").read_text()
        assert ("randomized solver this is approximate" in text if language == "en"
                else "随机化求解只能近似" in text)


@pytest.mark.parametrize("name", METHODS)
def test_chinese_unsupervised_reference_headings_are_localized(name):
    text = (ROOT / f"docs/cn/unsupervised/{name}.md").read_text()
    assert "## 参考文献" in text
    assert "## References" not in text
    # Preserve external links to the former English heading.
    assert '<a id="references"></a>' in text


@pytest.mark.parametrize("name", ["minibatch-kmeans", "truncated-svd", "tsne", "umap"])
def test_bilingual_unsupervised_navigation_has_reciprocal_links(name):
    for language, other in (("en", "cn"), ("cn", "en")):
        text = (ROOT / f"docs/{language}/unsupervised/{name}.md").read_text()
        assert f"../../{other}/unsupervised/{name}.md" in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_glm_overview_links_actual_result_and_oracle_boundaries(language):
    text = (ROOT / f"docs/{language}/models/generalized-linear-model.md").read_text()
    for fragment in (
        "inference_method_", "inference_target_", "_inference_result.method",
        "../guides/inference-modes.md#post_selection_ols",
        "../guides/penalized-glm-inference.md#current-non-gaussian-oracle-limitation",
    ):
        assert fragment in text
    assert "productized" not in text and "产品化" not in text
    assert "fail-closed" not in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_shared_weight_warning_describes_normalization_sum(language):
    weights = np.array([1e308, 1e308])
    for method in ("cauchy", "stouffer"):
        expected = combine_pvalues([.01, .1], method=method, weights=[1., 1.])
        actual = combine_pvalues([.01, .1], method=method, weights=weights / weights.max())
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    text = (ROOT / f"docs/{language}/reference/estimator-api.md").read_text()
    assert "Stouffer squared norm" not in text and "Stouffer 平方范数" not in text
    assert ("normalization sum" in text if language == "en" else "归一化求和" in text)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("solver,C", [
    ("auto", 1.), ("irls", 2.), ("irls", 0.),
    ("newton", .1), ("newton", 10.), ("lbfgs", .1), ("lbfgs", 10.),
])
def test_ordinary_glm_objective_retains_solver_specific_c_semantics(weighted, solver, C):
    from statgpu.linear_model import GeneralizedLinearModel

    rng = np.random.default_rng(59)
    X = rng.normal(size=(100, 2))
    y = rng.poisson(np.exp(.2 + X @ np.array([.3, -.2])))
    weights = np.linspace(.5, 2., len(X)) if weighted else np.ones(len(X))
    model = GeneralizedLinearModel(
        family="poisson", solver=solver, C=C, device="cpu", max_iter=1000, tol=1e-8,
    ).fit(X, y, sample_weight=weights if weighted else None)
    mean = np.exp(model.intercept_ + X @ model.coef_)
    score = weights * (mean - y) / weights.sum()
    ridge = 1 / (2 * C) if solver in {"auto", "irls"} and C > 0 else 0.
    np.testing.assert_allclose(X.T @ score + ridge * model.coef_, 0., atol=1e-7)
    np.testing.assert_allclose(score.sum(), 0., atol=1e-7)
    for language in ("en", "cn"):
        text = (ROOT / f"docs/{language}/models/generalized-linear-model.md").read_text()
        assert "\\frac{1}{4C}" in text
        assert "| `C` |" in text and "| `l1_ratio` |" in text
        assert "| `l1_ratio` | `None`" not in text
        assert "future unified result" not in text.lower()
        assert "未来预留接口" not in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_nonparametric_inventory_uses_resolvable_public_density_names(language):
    from statgpu import nonparametric

    text = (ROOT / f"docs/{language}/guides/implemented-methods.md").read_text()
    for name in ("KernelDensityEstimator", "KDE", "KernelRegression"):
        assert f"`{name}`" in text
        assert callable(getattr(statgpu, name))
        assert callable(getattr(nonparametric, name))
    assert "`KernelDensity`" not in text
    if language == "cn":
        assert "knockoff filter 与" not in text
        assert "有序 logit 回归" in text and "有序 probit 回归" in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_torch_entry_guide_links_current_device_exceptions(language):
    text = (ROOT / f"docs/{language}/guides/pytorch-backend.md").read_text()
    assert "device-and-memory.md#current-smoothing-and-spline-exceptions" in text
    assert text.count("#current-smoothing-and-spline-exceptions") >= 2
    readme = (ROOT / "README.md").read_text()
    assert "docs/en/guides/device-and-memory.md#current-smoothing-and-spline-exceptions" in readme


def test_missing_rejection_xfails_cannot_swallow_unrelated_exceptions():
    import importlib.util

    path = ROOT / "dev/tests/test_pr168_unsupervised_cycle4.py"
    spec = importlib.util.spec_from_file_location("pr168_known_gap_guards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "test_singular_gmm_rejects_fit_before_publishing_invalid_results",
        "test_minibatch_kmeans_rejects_nonfinite_initial_centers",
    ):
        test = getattr(module, name)
        marker = next(mark for mark in test.pytestmark if mark.name == "xfail")
        assert marker.kwargs["strict"] is True
        assert marker.kwargs["raises"] is pytest.fail.Exception
