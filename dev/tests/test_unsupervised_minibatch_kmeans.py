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


def test_minibatch_kmeans_rejects_unsupported_inputs():
    X = _blobs()
    with pytest.raises(NotImplementedError, match="sparse"):
        MiniBatchKMeans(device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="sample_weight"):
        MiniBatchKMeans(device="cpu").fit(X, sample_weight=np.ones(X.shape[0]))
    with pytest.raises(ValueError, match="init array"):
        MiniBatchKMeans(n_clusters=3, init=np.zeros((2, 2)), device="cpu").fit(X)
