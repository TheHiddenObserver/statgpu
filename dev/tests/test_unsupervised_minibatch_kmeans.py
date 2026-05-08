import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import MiniBatchKMeans


def _blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, -3.0], [0.0, 3.0], [3.0, -2.0]])
    return np.vstack([center + 0.2 * rng.normal(size=(20, 2)) for center in centers])


def test_minibatch_kmeans_fit_predict_shapes_and_reproducibility():
    X = _blobs()
    model = MiniBatchKMeans(
        n_clusters=3,
        batch_size=12,
        max_iter=8,
        max_no_improvement=20,
        random_state=0,
        device="cpu",
    ).fit(X)
    again = MiniBatchKMeans(
        n_clusters=3,
        batch_size=12,
        max_iter=8,
        max_no_improvement=20,
        random_state=0,
        device="cpu",
    ).fit(X)

    assert model.cluster_centers_.shape == (3, 2)
    assert model.labels_.shape == (60,)
    assert np.isfinite(model.inertia_)
    assert model.n_steps_ >= model.n_iter_ >= 1
    np.testing.assert_allclose(model.cluster_centers_, again.cluster_centers_)
    np.testing.assert_array_equal(model.predict(X), model.labels_)


def test_minibatch_kmeans_partial_fit_and_score():
    X = _blobs()
    model = MiniBatchKMeans(n_clusters=3, init=X[:3], random_state=0, device="cpu")
    model.partial_fit(X[:30]).partial_fit(X[30:])

    assert model.cluster_centers_.shape == (3, 2)
    assert model.counts_.shape == (3,)
    assert model.n_steps_ == 2
    assert model.transform(X[:5]).shape == (5, 3)
    assert model.score(X) <= 0.0


def test_minibatch_kmeans_partial_fit_initial_batch_validates_cluster_count():
    X = _blobs()
    model = MiniBatchKMeans(n_clusters=4, init="random", random_state=0, device="cpu")
    with pytest.raises(ValueError, match="n_clusters"):
        model.partial_fit(X[:3])


def test_minibatch_kmeans_partial_fit_allows_small_batch_with_explicit_init():
    X = _blobs()
    model = MiniBatchKMeans(n_clusters=4, init=X[:4], random_state=0, device="cpu")
    model.partial_fit(X[:2])

    assert model.cluster_centers_.shape == (4, 2)
    assert model.n_steps_ == 1


def test_minibatch_kmeans_score_validates_input_shape():
    X = _blobs()
    model = MiniBatchKMeans(n_clusters=3, init=X[:3], random_state=0, device="cpu").fit(X)
    with pytest.raises(ValueError, match="2D"):
        model.score(X[0])
    with pytest.raises(ValueError, match="features"):
        model.score(np.ones((5, 3)))
    with pytest.raises(NotImplementedError, match="sparse"):
        model.score(sparse.csr_matrix(X))


def test_minibatch_kmeans_rejects_unsupported_inputs():
    X = _blobs()
    with pytest.raises(NotImplementedError, match="sparse"):
        MiniBatchKMeans(device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="sample_weight"):
        MiniBatchKMeans(device="cpu").fit(X, sample_weight=np.ones(X.shape[0]))
    with pytest.raises(ValueError, match="init array"):
        MiniBatchKMeans(n_clusters=3, init=np.zeros((2, 2)), device="cpu").fit(X)
