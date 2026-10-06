"""Exercise documented ownership boundaries and numerical workarounds on CPU.

These tests protect working caller-side preparation and public API guidance;
passing them does not establish that the uncentered or spectral defects are fixed.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from statgpu import unsupervised
from statgpu.unsupervised._utils import squared_euclidean_distances

ROOT = Path(__file__).resolve().parents[2]


class _MissingUMAPNeighbors(Exception):
    """The finite diagonal mask removed required distinct-neighbor edges."""


def _data():
    return np.random.default_rng(31).normal(size=(20, 3))


def _dense_graph(model):
    source, target, weight, n = model.graph_
    return coo_matrix((weight, (source, target)), shape=(n, n)).toarray()


def test_float64_centering_preserves_pairwise_geometry_before_distance_expansion():
    shifted = _data() + 1e8
    centered = shifted - shifted.mean(axis=0)
    backend = unsupervised.TSNE(device="cpu")._get_backend()
    expected = np.sum((shifted[:, None] - shifted[None, :]) ** 2, axis=2)
    actual = squared_euclidean_distances(backend, centered)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_tsne_centering_workaround_preserves_affinities_after_translation():
    X = _data()
    shifted = X + 1e8
    centered = X - X.mean(axis=0)
    centered_shifted = shifted - shifted.mean(axis=0)
    model = unsupervised.TSNE(perplexity=3, device="cpu")
    backend = model._get_backend()
    expected = model._joint_probabilities(backend, centered)
    actual = model._joint_probabilities(backend, centered_shifted)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(actual.sum(), 1.0)


def test_umap_centering_workaround_preserves_graph_after_translation():
    X = _data()
    shifted = X + 1e8
    options = {"n_neighbors": 3, "n_epochs": 1, "init": "random", "random_state": 7, "device": "cpu"}
    expected = unsupervised.UMAP(**options).fit(X - X.mean(axis=0))
    actual = unsupervised.UMAP(**options).fit(shifted - shifted.mean(axis=0))
    np.testing.assert_allclose(_dense_graph(actual), _dense_graph(expected), rtol=1e-5, atol=1e-6)


def test_umap_random_initialization_is_seeded_and_avoids_spectral_solver(monkeypatch):
    def reject_spectral(*args, **kwargs):
        raise AssertionError("random initialization must not call eigsh")

    monkeypatch.setattr("scipy.sparse.linalg.eigsh", reject_spectral)
    X = _data()
    options = {"n_neighbors": 3, "n_components": 2, "n_epochs": 5,
               "init": "random", "nn_method": "exact", "random_state": 7, "device": "cpu"}
    first = unsupervised.UMAP(**options).fit_transform(X)
    second = unsupervised.UMAP(**options).fit_transform(X)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.linalg.matrix_rank(first) == 2


@pytest.mark.parametrize("name, options", [
    ("KMeans", {"n_clusters": 2, "random_state": 7}),
    ("MiniBatchKMeans", {"n_clusters": 2, "random_state": 7}),
    ("DBSCAN", {"eps": 2.0, "min_samples": 2}),
    ("AgglomerativeClustering", {"n_clusters": 2}),
])
def test_fit_predict_returns_fitted_labels_without_a_defensive_copy(name, options):
    model = getattr(unsupervised, name)(device="cpu", **options)
    labels = model.fit_predict(_data())
    assert labels is model.labels_
    independent = np.asarray(labels).copy()
    independent[:] = -123
    assert not np.any(model.labels_ == -123)


@pytest.mark.parametrize("name, options", [
    ("UMAP", {"n_neighbors": 3, "n_epochs": 2, "init": "random"}),
    ("TSNE", {"perplexity": 3, "max_iter": 250, "init": "random"}),
])
def test_fit_transform_returns_stored_embedding_without_a_defensive_copy(name, options):
    model = getattr(unsupervised, name)(device="cpu", random_state=7, **options)
    embedding = model.fit_transform(_data())
    assert embedding is model.embedding_
    independent = np.asarray(embedding).copy()
    independent[:] = np.nan
    assert np.isfinite(model.embedding_).all()


def test_nmf_fit_transform_returns_stored_training_factors():
    model = unsupervised.NMF(n_components=2, max_iter=10, random_state=7, device="cpu")
    factors = model.fit_transform(np.abs(_data()) + 0.1)
    assert factors is model._fit_W


@pytest.mark.parametrize("cls", (unsupervised.PCA, unsupervised.IncrementalPCA))
def test_copy_false_does_not_modify_input(cls):
    X = _data()
    original = X.copy()
    model = cls(n_components=2, copy=False, device="cpu")
    model.fit_transform(X)
    np.testing.assert_array_equal(X, original)


def test_new_fit_resets_default_incremental_rank():
    X = _data()
    model = unsupervised.IncrementalPCA(device="cpu")
    model.partial_fit(X[:2]).partial_fit(X[2:])
    assert model.n_components_ == 2
    model.fit(X)
    assert model.n_components_ == 3
    assert model.n_samples_seen_ == len(X)


@pytest.mark.parametrize("covariance_type, shape", [
    ("diag", (2, 3)), ("spherical", (2,)), ("tied", (3, 3)), ("full", (2, 3, 3)),
])
def test_mixture_precision_factor_shapes_and_orientation(covariance_type, shape):
    model = unsupervised.GaussianMixture(
        n_components=2, covariance_type=covariance_type,
        random_state=7, device="cpu",
    ).fit(_data())
    covariance = model.covariances_
    factor = model.precisions_cholesky_
    assert covariance.shape == factor.shape == shape
    if covariance_type in {"diag", "spherical"}:
        np.testing.assert_allclose(factor, 1.0 / np.sqrt(covariance))
    else:
        np.testing.assert_array_equal(factor, np.tril(factor))
        np.testing.assert_allclose(factor @ np.swapaxes(factor, -1, -2), np.linalg.inv(covariance))


@pytest.mark.parametrize("language", ("en", "cn"))
def test_bilingual_reference_keeps_ownership_and_workaround_guidance(language):
    directory = ROOT / f"docs/{language}/unsupervised"
    reference = (directory / "api-reference.md").read_text()
    assert "L @ L.T = inv(covariance)" in reference
    assert "a.copy()" in reference and "a.clone()" in reference
    assert "init=\"random\"" in reference
    for name in ("umap", "tsne", "agglomerative-clustering"):
        text = (directory / f"{name}.md").read_text()
        assert "float64" in text
    assert reference.count("Array-returning methods do not universally") <= 1
    assert reference.count("返回数组的方法并不统一") <= 1


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_public_pca_inverse_transform_rejects_nonfinite_coordinates(nonfinite):
    model = unsupervised.PCA(n_components=2, device="cpu").fit(_data())
    scores = np.array([[0.0, nonfinite]])
    with pytest.raises(ValueError, match="finite"):
        model.inverse_transform(scores)


def _centered_wide_umap_data():
    data = np.array([[-3., -1.], [0., 2.], [4., -1.], [5., 4.]])
    return (data - data.mean(axis=0)) * 1e13


def _fit_small_exact_umap(data):
    return unsupervised.UMAP(
        n_neighbors=3, n_epochs=1, init="random", nn_method="exact",
        random_state=4, device="cpu",
    ).fit(data)


@pytest.mark.xfail(
    strict=True,
    raises=_MissingUMAPNeighbors,
    reason="UMAP's finite diagonal mask admits self-neighbors at large distances",
)
def test_exact_umap_keeps_all_distinct_neighbors_at_large_scale():
    # Four rows and k=3 require all three OTHER rows in every neighborhood.
    # The current finite diagonal mask admits self-neighbors, which are then
    # removed from the graph. This oracle must pass after a numerical repair.
    model = _fit_small_exact_umap(_centered_wide_umap_data())
    graph = _dense_graph(model)
    assert np.isfinite(model.embedding_).all()
    neighbor_count = np.count_nonzero(graph)
    if neighbor_count < 4 * 3:
        raise _MissingUMAPNeighbors(f"Expected all 12 distinct-neighbor edges, observed {neighbor_count}")
    assert neighbor_count == 4 * 3


def test_common_training_scale_preserves_umap_distinct_neighbor_graph():
    data = _centered_wide_umap_data()
    offset = data.mean(axis=0)
    centered = data - offset
    scale = np.max(np.abs(centered))
    assert np.isfinite(scale) and scale > 0
    prepared = centered / scale
    model = _fit_small_exact_umap(prepared)
    graph = _dense_graph(model)
    assert np.count_nonzero(graph) == 4 * 3
    assert np.isfinite(model.embedding_).all()
    np.testing.assert_array_equal(np.diag(graph), np.zeros(4))
    reference = _fit_small_exact_umap(centered / 1e13)
    np.testing.assert_allclose(graph, _dense_graph(reference), rtol=1e-5, atol=1e-6)
    original_distances = np.sum((data[:, None] - data[None, :]) ** 2, axis=2)
    prepared_distances = np.sum((prepared[:, None] - prepared[None, :]) ** 2, axis=2)
    np.testing.assert_allclose(prepared_distances, original_distances / scale ** 2)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_umap_guidance_requires_common_scaling_beyond_centering(language):
    directory = ROOT / f"docs/{language}/unsupervised"
    guide = (directory / "umap.md").read_text()
    reference = (directory / "api-reference.md").read_text()
    if language == "en":
        for text in (guide, reference):
            assert "one common positive" in text
            assert "neighbor ordering" in text
        assert "Centering alone does not address this restriction" in guide
    else:
        for text in (guide, reference):
            assert "同一个正尺度" in text
            assert "欧氏近邻顺序" in text
        assert "仅中心化不能解决这一限制" in guide
    assert "self-neighbors" in unsupervised.UMAP.__doc__
