"""Tests for statgpu.unsupervised.DBSCAN."""

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from statgpu.unsupervised import DBSCAN


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _make_blobs(seed=20260501):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(-2.0, 0.0), scale=0.12, size=(40, 2))
    b = rng.normal(loc=(2.0, 0.0), scale=0.12, size=(40, 2))
    noise = np.array([[0.0, 3.0], [0.0, -3.0]])
    return np.vstack([a, b, noise])


def test_dbscan_matches_sklearn_labels_up_to_permutation():
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X = _make_blobs()

    sg = DBSCAN(eps=0.35, min_samples=4, device="cpu").fit(X)
    sk = sk_cluster.DBSCAN(eps=0.35, min_samples=4, metric="euclidean").fit(X)

    sg_labels = _to_numpy(sg.labels_)
    assert sg_labels.shape == (X.shape[0],)
    assert sg.core_sample_indices_.shape[0] == len(sk.core_sample_indices_)
    assert np.array_equal(sg_labels == -1, sk.labels_ == -1)
    assert adjusted_rand_score(sk.labels_, sg_labels) == pytest.approx(1.0)


def test_dbscan_all_noise_and_fit_predict():
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    labels = _to_numpy(DBSCAN(eps=0.1, min_samples=2, device="cpu").fit_predict(X))

    assert np.array_equal(labels, np.full(3, -1))


def test_dbscan_validation_errors():
    X = _make_blobs()
    with pytest.raises(ValueError, match="eps"):
        DBSCAN(eps=0.0, device="cpu").fit(X)
    with pytest.raises(ValueError, match="min_samples"):
        DBSCAN(min_samples=0, device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="metric"):
        DBSCAN(metric="manhattan", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        DBSCAN(device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="predict"):
        DBSCAN(device="cpu").fit(X).predict(X)


def test_dbscan_cuda_matches_cpu_when_available():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")
    X = _make_blobs(seed=11)
    cpu = _to_numpy(DBSCAN(eps=0.35, min_samples=4, device="cpu").fit(X).labels_)
    gpu = _to_numpy(DBSCAN(eps=0.35, min_samples=4, device="cuda").fit(X).labels_)

    assert adjusted_rand_score(cpu, gpu) == pytest.approx(1.0)


def test_dbscan_torch_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")
    X = _make_blobs(seed=13)
    cpu = _to_numpy(DBSCAN(eps=0.35, min_samples=4, device="cpu").fit(X).labels_)
    gpu = _to_numpy(DBSCAN(eps=0.35, min_samples=4, device="torch").fit(X).labels_)

    assert adjusted_rand_score(cpu, gpu) == pytest.approx(1.0)
