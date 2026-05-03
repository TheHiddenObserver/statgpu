"""Tests for statgpu.unsupervised.KMeans."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from statgpu.unsupervised import KMeans


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _make_cluster_data(seed=20260430):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-4.0, 0.0, 0.5],
            [0.5, 3.5, -1.0],
            [4.0, -2.0, 1.5],
        ]
    )
    parts = [center + rng.normal(scale=0.25, size=(70, 3)) for center in centers]
    return np.vstack(parts), centers


def _match_centers(a, b):
    distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    rows, cols = linear_sum_assignment(distances)
    return a[rows], b[cols]


def test_kmeans_cpu_basic_interface_and_reproducibility():
    X, _ = _make_cluster_data()

    m1 = KMeans(n_clusters=3, random_state=17, n_init=3, device="cpu").fit(X)
    m2 = KMeans(n_clusters=3, random_state=17, n_init=3, device="cpu").fit(X)

    labels = _to_numpy(m1.predict(X))
    distances = _to_numpy(m1.transform(X))

    assert m1.cluster_centers_.shape == (3, 3)
    assert m1.labels_.shape == (X.shape[0],)
    assert labels.shape == (X.shape[0],)
    assert distances.shape == (X.shape[0], 3)
    assert np.isfinite(m1.inertia_)
    assert m1.inertia_ > 0.0
    assert m1.n_iter_ >= 1
    assert m1.n_features_in_ == 3
    assert np.array_equal(_to_numpy(m1.labels_), _to_numpy(m2.labels_))
    assert np.allclose(_to_numpy(m1.cluster_centers_), _to_numpy(m2.cluster_centers_))
    assert np.isclose(m1.score(X), -m1.inertia_)


def test_kmeans_fit_predict_returns_labels():
    X, _ = _make_cluster_data(seed=11)

    model = KMeans(n_clusters=3, random_state=5, n_init=2, device="cpu")
    labels = model.fit_predict(X)

    assert np.array_equal(_to_numpy(labels), _to_numpy(model.labels_))


def test_kmeans_cpu_matches_sklearn_centers_and_inertia():
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X, _ = _make_cluster_data(seed=13)

    sg = KMeans(n_clusters=3, init="k-means++", n_init=5, random_state=23, device="cpu").fit(X)
    sk = sk_cluster.KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=5,
        random_state=23,
        algorithm="lloyd",
        tol=1e-4,
    ).fit(X)

    sg_centers, sk_centers = _match_centers(_to_numpy(sg.cluster_centers_), sk.cluster_centers_)

    assert np.allclose(sg_centers, sk_centers, rtol=1e-4, atol=1e-4)
    assert np.isclose(sg.inertia_, sk.inertia_, rtol=1e-4, atol=1e-4)


def test_kmeans_random_init_runs_multiple_initializations():
    X, _ = _make_cluster_data(seed=19)

    model = KMeans(n_clusters=3, init="random", n_init="auto", random_state=29, device="cpu").fit(X)

    assert model.cluster_centers_.shape == (3, 3)
    assert np.isfinite(model.inertia_)


def test_kmeans_empty_cluster_reinitializes_to_finite_center():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])

    model = KMeans(n_clusters=3, init="random", n_init=1, random_state=3, device="cpu").fit(X)

    assert np.all(np.isfinite(_to_numpy(model.cluster_centers_)))
    assert np.isfinite(model.inertia_)


@pytest.mark.parametrize(
    "kwargs,match,exc",
    [
        ({"n_clusters": 0}, "n_clusters", ValueError),
        ({"n_clusters": 999}, "n_clusters", ValueError),
        ({"init": "bad"}, "init", ValueError),
        ({"n_init": 0}, "n_init", ValueError),
        ({"max_iter": 0}, "max_iter", ValueError),
        ({"tol": -1.0}, "tol", ValueError),
    ],
)
def test_kmeans_validation_errors(kwargs, match, exc):
    X, _ = _make_cluster_data(seed=31)
    with pytest.raises(exc, match=match):
        KMeans(device="cpu", **kwargs).fit(X)


def test_kmeans_unsupported_inputs_raise_clear_errors():
    X, _ = _make_cluster_data(seed=37)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        KMeans(n_clusters=3, device="cpu").fit(X, sample_weight=np.ones(X.shape[0]))
    with pytest.raises(NotImplementedError, match="sparse"):
        KMeans(n_clusters=2, device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="callable init"):
        KMeans(n_clusters=3, init=lambda X, k: X[:k], device="cpu").fit(X)


def test_kmeans_cuda_matches_cpu_inertia():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")

    X, _ = _make_cluster_data(seed=41)

    cpu = KMeans(n_clusters=3, n_init=2, random_state=43, device="cpu").fit(X)
    gpu = KMeans(n_clusters=3, n_init=2, random_state=43, device="cuda").fit(X)

    assert np.isclose(gpu.inertia_, cpu.inertia_, rtol=1e-4, atol=1e-4)


def test_kmeans_explicit_torch_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")

    X, _ = _make_cluster_data(seed=47)

    cpu = KMeans(n_clusters=3, n_init=2, random_state=53, device="cpu").fit(X)
    torch_model = KMeans(n_clusters=3, n_init=2, random_state=53, device="torch").fit(X)

    assert np.isclose(torch_model.inertia_, cpu.inertia_, rtol=1e-4, atol=1e-4)
