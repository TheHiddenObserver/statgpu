"""Tests for statgpu.unsupervised.AgglomerativeClustering."""

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from statgpu.unsupervised import AgglomerativeClustering


def _make_data(seed=20260501):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(-3.0, 0.0), scale=0.2, size=(20, 2))
    b = rng.normal(loc=(3.0, 0.0), scale=0.2, size=(20, 2))
    c = rng.normal(loc=(0.0, 3.0), scale=0.2, size=(20, 2))
    return np.vstack([a, b, c])


def test_agglomerative_single_matches_sklearn_labels():
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X = _make_data()

    sg = AgglomerativeClustering(n_clusters=3, linkage="single", device="cpu").fit(X)
    sk = sk_cluster.AgglomerativeClustering(n_clusters=3, linkage="single", metric="euclidean").fit(X)

    assert sg.labels_.shape == (X.shape[0],)
    assert sg.children_.shape == (X.shape[0] - 1, 2)
    assert sg.distances_.shape == (X.shape[0] - 1,)
    assert adjusted_rand_score(sk.labels_, sg.labels_) == pytest.approx(1.0)


def test_agglomerative_fit_predict_and_validation_errors():
    X = _make_data(seed=3)
    labels = AgglomerativeClustering(n_clusters=3, device="cpu").fit_predict(X)

    assert labels.shape == (X.shape[0],)
    with pytest.raises(ValueError, match="n_clusters"):
        AgglomerativeClustering(n_clusters=0, device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="linkage"):
        AgglomerativeClustering(linkage="average", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="metric"):
        AgglomerativeClustering(metric="manhattan", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        AgglomerativeClustering(device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="predict"):
        AgglomerativeClustering(device="cpu").fit(X).predict(X)


def test_agglomerative_explicit_gpu_raises_without_fallback():
    X = _make_data(seed=5)
    with pytest.raises(NotImplementedError, match="CPU execution"):
        AgglomerativeClustering(n_clusters=3, device="cuda").fit(X)
    with pytest.raises(NotImplementedError, match="CPU execution"):
        AgglomerativeClustering(n_clusters=3, device="torch").fit(X)
