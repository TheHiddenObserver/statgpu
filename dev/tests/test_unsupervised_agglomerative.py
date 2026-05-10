"""Tests for statgpu.unsupervised.AgglomerativeClustering."""

import importlib

import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import AgglomerativeClustering


def _make_data(seed=20260501):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(-3.0, 0.0), scale=0.2, size=(20, 2))
    b = rng.normal(loc=(3.0, 0.0), scale=0.2, size=(20, 2))
    c = rng.normal(loc=(0.0, 3.0), scale=0.2, size=(20, 2))
    return np.vstack([a, b, c])


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_agglomerative_linkages_match_sklearn_labels(linkage):
    sk_cluster = pytest.importorskip("sklearn.cluster")
    sk_metrics = pytest.importorskip("sklearn.metrics")
    X = _make_data()

    sg = AgglomerativeClustering(n_clusters=3, linkage=linkage, device="cpu").fit(X)
    sk = sk_cluster.AgglomerativeClustering(n_clusters=3, linkage=linkage, metric="euclidean").fit(X)

    assert sg.labels_.shape == (X.shape[0],)
    assert sg.children_.shape == (X.shape[0] - 1, 2)
    assert sg.distances_.shape == (X.shape[0] - 1,)
    assert sk_metrics.adjusted_rand_score(sk.labels_, sg.labels_) == pytest.approx(1.0)


def test_agglomerative_fit_predict_and_validation_errors():
    X = _make_data(seed=3)
    labels = AgglomerativeClustering(n_clusters=3, device="cpu").fit_predict(X)

    assert labels.shape == (X.shape[0],)
    with pytest.raises(ValueError, match="n_clusters"):
        AgglomerativeClustering(n_clusters=0, device="cpu").fit(X)
    with pytest.raises(ValueError, match="linkage"):
        AgglomerativeClustering(linkage="bad", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="metric"):
        AgglomerativeClustering(metric="manhattan", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        AgglomerativeClustering(device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="predict"):
        AgglomerativeClustering(device="cpu").fit(X).predict(X)


def test_gpu_memory_env_invalid_value_fallback(monkeypatch):
    import statgpu.unsupervised._agglomerative as agglomerative_module

    monkeypatch.setenv("STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES", "not_an_int")
    with pytest.warns(RuntimeWarning, match="Invalid STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES"):
        importlib.reload(agglomerative_module)
    assert agglomerative_module.AgglomerativeClustering._GPU_DISTANCE_LIMIT_BYTES == 1 << 30

    monkeypatch.delenv("STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES", raising=False)
    importlib.reload(agglomerative_module)
    assert agglomerative_module.AgglomerativeClustering._GPU_DISTANCE_LIMIT_BYTES == 1 << 30


def test_agglomerative_explicit_cuda_runs_without_fallback():
    X = _make_data(seed=5)
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")

    gpu = AgglomerativeClustering(n_clusters=3, linkage="single", device="cuda").fit(X)
    cpu = AgglomerativeClustering(n_clusters=3, linkage="single", device="cpu").fit(X)

    assert gpu.labels_.shape == cpu.labels_.shape
    assert gpu.children_.shape == cpu.children_.shape
    assert gpu.distances_.shape == cpu.distances_.shape
    assert np.all(np.diff(gpu.distances_) >= -1e-10)


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_agglomerative_cupy_matches_cpu_labels(linkage):
    sk_metrics = pytest.importorskip("sklearn.metrics")
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")

    X = _make_data(seed=11)
    cpu = AgglomerativeClustering(n_clusters=3, linkage=linkage, device="cpu").fit(X)
    gpu = AgglomerativeClustering(n_clusters=3, linkage=linkage, device="cuda").fit(cp.asarray(X))

    assert sk_metrics.adjusted_rand_score(cpu.labels_, gpu.labels_) == pytest.approx(1.0)
    assert gpu.children_.shape == (X.shape[0] - 1, 2)
    assert gpu.distances_.shape == (X.shape[0] - 1,)
    assert np.all(np.diff(gpu.distances_) >= -1e-10)


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_agglomerative_torch_matches_cpu_labels(linkage):
    sk_metrics = pytest.importorskip("sklearn.metrics")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")

    X = _make_data(seed=13)
    cpu = AgglomerativeClustering(n_clusters=3, linkage=linkage, device="cpu").fit(X)
    X_torch = torch.as_tensor(X, dtype=torch.float64, device="cuda")
    gpu = AgglomerativeClustering(n_clusters=3, linkage=linkage, device="torch").fit(X_torch)

    assert sk_metrics.adjusted_rand_score(cpu.labels_, gpu.labels_) == pytest.approx(1.0)
    assert gpu.children_.shape == (X.shape[0] - 1, 2)
    assert gpu.distances_.shape == (X.shape[0] - 1,)
    assert np.all(np.diff(gpu.distances_) >= -1e-10)
