"""Run self-contained learner examples and supported numerical workarounds."""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.unsupervised import PCA, TSNE, GaussianMixture, KMeans, MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]
GUIDES = {
    "pca": ("Z", (80, 3)),
    "kmeans": ("labels", (60,)),
    "dbscan": ("labels", (41,)),
    "gaussian-mixture": ("proba", (80, 2)),
    "nmf": ("W", (60, 2)),
    "agglomerative-clustering": ("labels", (30,)),
    "truncated-svd": ("Z", (60, 2)),
    "minibatch-kmeans": ("labels", (60,)),
    "incremental-pca": ("Z", (60, 2)),
    "minibatch-nmf": ("W", (60, 2)),
    "umap": ("embedding", (40, 2)),
    "tsne": ("embedding", (40, 2)),
}


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("slug", GUIDES)
def test_model_learner_example_runs_independently(language, slug):
    path = ROOT / f"docs/{language}/unsupervised/{slug}.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- learner-example: {slug} -->\s*```python\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    namespace = {}
    exec(compile(match.group(1), str(path), "exec"), namespace)  # noqa: S102
    name, shape = GUIDES[slug]
    result = namespace[name]
    assert result.shape == shape
    assert np.isfinite(result).all()
    assert namespace["model"].n_features_in_ == namespace["X"].shape[1]
    if slug == "dbscan":
        assert result[-1] == -1
        assert len(set(result[:-1])) == 2
    if slug == "gaussian-mixture":
        np.testing.assert_allclose(result.sum(axis=1), 1)
        assert namespace["model"].converged_
    if slug in {"nmf", "minibatch-nmf"}:
        assert (result >= 0).all()
    if slug == "minibatch-kmeans":
        assert namespace["model"].labels_.shape == (15,)
    if slug == "incremental-pca":
        assert namespace["model"].n_samples_seen_ == 60


def test_full_pca_preserves_centered_variance_with_large_offset():
    X = np.random.default_rng(43).normal(size=(100, 3))
    shifted = X + 1e8
    model = PCA(n_components=3, svd_solver="full", device="cpu").fit(shifted)
    expected = np.linalg.eigvalsh(np.cov(shifted, rowvar=False))[::-1]
    np.testing.assert_allclose(model.explained_variance_, expected, rtol=1e-10)
    assert model.explained_variance_ratio_.sum() == pytest.approx(1)
    np.testing.assert_allclose(model.inverse_transform(model.transform(shifted)), shifted)


@pytest.mark.parametrize("cls", (KMeans, MiniBatchKMeans))
def test_training_offset_workaround_preserves_cluster_distance_objective(cls):
    X = np.random.default_rng(43).normal(size=(100, 3)) + 1e8
    offset = X.mean(axis=0)
    centered = X - offset
    model = cls(n_clusters=2, random_state=0, device="cpu").fit(centered)
    distances = np.linalg.norm(centered[:, None] - model.cluster_centers_[None], axis=2)
    np.testing.assert_allclose(model.transform(centered), distances, atol=1e-12)
    expected = np.min(distances, axis=1) ** 2
    assert model.inertia_ == pytest.approx(expected.sum())
    assert model.score(centered) == pytest.approx(-expected.sum())
    np.testing.assert_array_equal(model.predict(centered), distances.argmin(axis=1))


@pytest.mark.parametrize("covariance_type", ("diag", "spherical"))
def test_training_offset_workaround_preserves_mixture_moments_and_density(covariance_type):
    X = np.random.default_rng(43).normal(size=(100, 3)) + 1e8
    offset = X.mean(axis=0)
    centered = X - offset
    model = GaussianMixture(
        n_components=1, covariance_type=covariance_type,
        random_state=0, device="cpu",
    ).fit(centered)
    variance = centered.var(axis=0)
    if covariance_type == "spherical":
        variance = np.repeat(variance.mean(), 3)
    np.testing.assert_allclose(model.covariances_, variance[:1] if covariance_type == "spherical" else variance[None], rtol=1e-12)
    expected = -0.5 * (3 * np.log(2 * np.pi) + np.log(variance).sum()
                       + np.sum((centered - model.means_[0]) ** 2 / variance, axis=1))
    np.testing.assert_allclose(model.score_samples(centered), expected, rtol=1e-12)
    assert model.converged_


def test_rescaled_tsne_has_finite_embedding_and_nonnegative_kl():
    X = np.arange(12.0).reshape(6, 2) * 1e12
    # Scaling is a caller-side preparation step, not a production change.
    scaled = X / np.max(np.abs(X))
    model = TSNE(
        perplexity=2, max_iter=250, init="random", random_state=1, device="cpu"
    ).fit(scaled)
    assert np.isfinite(model.embedding_).all()
    assert model.kl_divergence_ >= 0


@pytest.mark.parametrize("language", ("en", "cn"))
def test_guides_describe_current_behavior_without_historical_benchmark_claims(language):
    for slug in GUIDES:
        text = (ROOT / f"docs/{language}/unsupervised/{slug}.md").read_text()
        assert "Phase " not in text
        assert "results/" not in text
        assert "dev/tests/" not in text
        assert not any(ord(c) < 32 and c not in "\n\t" for c in text)
    dbscan = (ROOT / f"docs/{language}/unsupervised/dbscan.md").read_text()
    assert "float32" in dbscan
    assert "zero GPU" not in dbscan and "没有 GPU→CPU" not in dbscan
    reference = (ROOT / f"docs/{language}/unsupervised/api-reference.md").read_text()
    assert "n * negative_sample_rate" in reference
    assert "1 GiB" in reference
    assert "NumPy 2" in reference
