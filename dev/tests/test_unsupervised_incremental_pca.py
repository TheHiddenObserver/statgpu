"""Tests for statgpu.unsupervised.IncrementalPCA."""

import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import IncrementalPCA


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _make_data(seed=20260507):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(120, 3))
    loadings = rng.normal(size=(3, 8))
    return latent @ loadings + 0.05 * rng.normal(size=(120, 8))


def test_incremental_pca_basic_interface_and_sklearn_alignment():
    sk_decomposition = pytest.importorskip("sklearn.decomposition")
    X = _make_data()
    model = IncrementalPCA(n_components=3, batch_size=30, device="cpu").fit(X)
    sk = sk_decomposition.IncrementalPCA(n_components=3, batch_size=30).fit(X)

    Xt = _to_numpy(model.transform(X))
    X_inv = _to_numpy(model.inverse_transform(model.transform(X)))

    assert model.components_.shape == (3, X.shape[1])
    assert model.mean_.shape == (X.shape[1],)
    assert model.var_.shape == (X.shape[1],)
    assert model.explained_variance_.shape == (3,)
    assert model.explained_variance_ratio_.shape == (3,)
    assert model.singular_values_.shape == (3,)
    assert model.n_samples_seen_ == X.shape[0]
    assert Xt.shape == (X.shape[0], 3)
    assert X_inv.shape == X.shape
    assert np.allclose(_to_numpy(model.mean_), sk.mean_, atol=1e-10)
    assert np.allclose(_to_numpy(model.explained_variance_), sk.explained_variance_, rtol=1e-5, atol=1e-5)


def test_incremental_pca_partial_fit_and_whiten():
    X = _make_data(seed=5)
    model = IncrementalPCA(n_components=2, whiten=True, device="cpu")
    model.partial_fit(X[:60]).partial_fit(X[60:])

    Xt = _to_numpy(model.predict(X))
    assert Xt.shape == (X.shape[0], 2)
    assert model.n_samples_seen_ == X.shape[0]
    assert np.all(np.isfinite(Xt))


def test_incremental_pca_whiten_constant_data_stays_finite():
    X = np.ones((8, 4), dtype=np.float64)
    model = IncrementalPCA(n_components=2, whiten=True, device="cpu").fit(X)
    Xt = _to_numpy(model.transform(X))
    X_back = _to_numpy(model.inverse_transform(Xt))

    assert np.all(np.isfinite(Xt))
    assert np.all(np.isfinite(X_back))


def test_incremental_pca_default_components_supports_wide_inputs():
    X = np.arange(30, dtype=np.float64).reshape(3, 10)
    model = IncrementalPCA(device="cpu").fit(X)
    Xt = _to_numpy(model.transform(X))

    assert model.n_components_ == min(X.shape)
    assert Xt.shape == (X.shape[0], min(X.shape))


def test_incremental_pca_fit_handles_small_batch_size_with_larger_components():
    X = _make_data(seed=12)
    model = IncrementalPCA(n_components=6, batch_size=3, device="cpu").fit(X)
    Xt = _to_numpy(model.transform(X))

    assert model.n_components_ == 6
    assert Xt.shape == (X.shape[0], 6)


def test_incremental_pca_validation_errors():
    X = _make_data(seed=9)
    with pytest.raises(ValueError, match="n_components"):
        IncrementalPCA(n_components=0, device="cpu").fit(X)
    with pytest.raises(ValueError, match="first partial_fit batch"):
        IncrementalPCA(n_components=6, device="cpu").partial_fit(X[:5])
    with pytest.raises(NotImplementedError, match="sparse"):
        IncrementalPCA(n_components=2, device="cpu").fit(sparse.csr_matrix(X))


def test_incremental_pca_cuda_torch_matches_cpu_when_available():
    X = _make_data(seed=13)
    cpu = IncrementalPCA(n_components=3, batch_size=40, device="cpu").fit(X)

    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")
    gpu = IncrementalPCA(n_components=3, batch_size=40, device="cuda").fit(X)
    assert np.allclose(_to_numpy(gpu.explained_variance_), _to_numpy(cpu.explained_variance_), rtol=1e-6, atol=1e-6)

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")
    tgpu = IncrementalPCA(n_components=3, batch_size=40, device="torch").fit(X)
    assert np.allclose(_to_numpy(tgpu.explained_variance_), _to_numpy(cpu.explained_variance_), rtol=1e-6, atol=1e-6)
