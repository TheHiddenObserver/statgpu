"""Fifth-pass documentation contracts for graph weights and inverse factors.

These are NumPy CPU API/analytic checks, not physical GPU or benchmark evidence.
The graph tests describe the published mean-distance variant; they do not claim
that it implements reference UMAP bandwidth calibration or optimization.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from statgpu.unsupervised import NMF, UMAP, MiniBatchNMF

ROOT = Path(__file__).resolve().parents[2]


def _dense_graph(model):
    source, target, weight, size = model.graph_
    return coo_matrix((weight, (source, target)), shape=(size, size)).toarray()


def _documented_graph(X, k):
    """Independent small-data implementation of the documented equations."""
    distances = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    selected = np.take_along_axis(distances, neighbors, axis=1)
    excess = np.maximum(selected - selected[:, :1], 0)
    scale = np.maximum(excess.mean(axis=1, keepdims=True), 1e-12)
    directed = np.zeros((len(X), len(X)))
    directed[np.arange(len(X))[:, None], neighbors] = np.exp(-excess / scale)
    return directed + directed.T - directed * directed.T


@pytest.mark.parametrize("duplicate", (False, True))
def test_public_umap_graph_matches_documented_bandwidth_and_fuzzy_union(duplicate):
    # Moderate exactly represented coordinates avoid known float32-offset and
    # finite-self-sentinel defects, so this isolates the graph-weight equation.
    X = np.array([[0., 0.], [1., 2.], [4., 1.], [7., 5.], [11., 3.], [16., 9.]])
    if duplicate:
        X[1] = X[0]
    model = UMAP(n_neighbors=3, init="random", nn_method="exact", n_epochs=1,
                 random_state=12, device="cpu").fit(X)
    actual = _dense_graph(model)
    np.testing.assert_allclose(actual, _documented_graph(X, 3), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual, actual.T)
    np.testing.assert_array_equal(np.diag(actual), np.zeros(len(X)))
    assert np.all((actual >= 0) & (actual <= 1))
    if duplicate:
        assert actual[0, 1] == pytest.approx(1.0)


def test_umap_layout_parameters_do_not_reweight_high_dimensional_graph():
    X = np.array([[0., 0.], [1., 2.], [4., 1.], [7., 5.], [11., 3.], [16., 9.]])
    options = {
        "n_neighbors": 3, "init": "random", "nn_method": "exact", "n_epochs": 1,
        "random_state": 12, "device": "cpu",
    }
    first = UMAP(min_dist=0.1, spread=1.0, **options).fit(X)
    second = UMAP(min_dist=0.6, spread=2.0, **options).fit(X)
    np.testing.assert_array_equal(_dense_graph(first), _dense_graph(second))
    assert not np.allclose(first._attraction_curve_params(), second._attraction_curve_params())


@pytest.mark.parametrize("estimator", (NMF, MiniBatchNMF))
def test_signed_inverse_coordinates_are_linear_and_observation_encoding_rejects_negatives(estimator):
    X = np.random.default_rng(53).uniform(0.2, 2.0, size=(12, 3))
    model = estimator(n_components=2, max_iter=3, random_state=53, device="cpu").fit(X)
    coordinates = np.array([[-1., -2.], [0., 1.]])
    original = coordinates.copy()
    reconstructed = model.inverse_transform(coordinates)
    np.testing.assert_allclose(reconstructed, coordinates @ model.components_)
    np.testing.assert_array_equal(coordinates, original)
    assert reconstructed.shape == (2, 3)
    assert np.all(reconstructed[0] < 0)
    assert np.all(reconstructed[1] >= 0)
    with pytest.raises(ValueError, match="non-negative"):
        model.transform(-X)
    with pytest.raises(ValueError, match="non-negative"):
        estimator(n_components=2, device="cpu").fit(-X)
    if estimator is MiniBatchNMF:
        with pytest.raises(ValueError, match="non-negative"):
            model.partial_fit(-X)


@pytest.mark.parametrize("estimator", (NMF, MiniBatchNMF))
@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_inverse_coordinates_still_reject_nonfinite_values(estimator, bad):
    X = np.random.default_rng(54).uniform(0.2, 2.0, size=(12, 3))
    model = estimator(n_components=2, max_iter=3, random_state=54, device="cpu").fit(X)
    with pytest.raises(ValueError, match="finite"):
        model.inverse_transform(np.array([[bad, 1.0]]))


@pytest.mark.parametrize("language", ("en", "cn"))
def test_bilingual_guides_state_graph_variant_and_signed_inverse_boundary(language):
    directory = ROOT / "docs" / language / "unsupervised"
    graph_guide = (directory / "umap.md").read_text(encoding="utf-8")
    reference = (directory / "api-reference.md").read_text(encoding="utf-8")
    for token in (r"\rho_i=d_{i1}", r"\sigma_i=\max", r"w_{ij}=v_{ij}+v_{ji}-v_{ij}v_{ji}",
                  "smooth_knn_dist", "min_dist", "spread"):
        assert token in graph_guide
    assert "umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html#smooth_knn_dist" in graph_guide
    assert "smooth_knn_dist" in UMAP.__doc__ or "membership-sum" in UMAP.__doc__
    for filename in ("nmf.md", "minibatch-nmf.md"):
        guide = (directory / filename).read_text(encoding="utf-8")
        assert "inverse_transform" in guide
        assert ("negative coordinates" if language == "en" else "负坐标") in guide
    assert reference.count("Negative coordinates are accepted" if language == "en" else "允许负坐标") == 2
    assert ("umap.md#graph-weights" if language == "en" else "umap.md#图权重") in reference
    for estimator in (NMF, MiniBatchNMF):
        assert "does\n        not reject negative coordinates" in estimator.inverse_transform.__doc__
