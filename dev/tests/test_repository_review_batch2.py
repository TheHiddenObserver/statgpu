
import numpy as np
import pytest
from scipy import sparse

from statgpu.cross_validation import (
    CVCache,
    batch_mse,
    detect_gpu_input,
    validate_cv_sample_weight,
)
from statgpu.unsupervised import KMeans, UMAP


def test_cv_cache_rejects_invalid_size_and_zero_disables_storage():
    with pytest.raises(ValueError, match="maxsize"):
        CVCache(-1)
    cache = CVCache(0)
    cache.put("key", 1)
    assert cache.get("key") is None


def test_sample_weight_and_batch_mse_validation():
    with pytest.raises(ValueError, match="positive sum"):
        validate_cv_sample_weight(np.zeros(3), 3)
    X = np.eye(2)
    y = np.ones(2)
    coefs = np.ones((2, 2))
    with pytest.raises(ValueError, match="chunk_size"):
        batch_mse(X, y, coefs, chunk_size=0)
    with pytest.raises(ValueError, match="intercepts length"):
        batch_mse(X, y, coefs, intercepts=np.zeros(1))
    with pytest.raises(ValueError, match="sample_weight length"):
        batch_mse(X, y, coefs, sample_weight=np.ones(1))
    with pytest.raises(ValueError, match="positive sum"):
        batch_mse(X, y, coefs, sample_weight=np.zeros(2))


def test_detect_gpu_input_numpy_pair_is_unchanged():
    X, y = np.ones((3, 2)), np.ones(3)
    backend, X_out, y_out = detect_gpu_input(X, y)
    assert backend == "numpy"
    assert X_out is X and y_out is y


def test_kmeans_score_validates_input_shape_and_sparsity():
    X = np.arange(24.0).reshape(8, 3)
    model = KMeans(n_clusters=2, random_state=0, device="cpu").fit(X)
    with pytest.raises(ValueError, match="2D"):
        model.score(X[:, 0])
    with pytest.raises(ValueError, match="features"):
        model.score(np.ones((2, 4)))
    with pytest.raises(NotImplementedError, match="sparse"):
        model.score(sparse.csr_matrix(X))


def test_umap_small_spectral_initialization_has_requested_dimension():
    X = np.array([[0.0], [1.0], [2.0]])
    embedding = UMAP(
        n_neighbors=2,
        n_components=2,
        n_epochs=1,
        init="spectral",
        random_state=0,
        device="cpu",
    ).fit_transform(X)
    assert embedding.shape == (3, 2)
    assert np.all(np.isfinite(embedding))
