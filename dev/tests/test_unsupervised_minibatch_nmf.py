"""Tests for statgpu.unsupervised.MiniBatchNMF."""

import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import MiniBatchNMF


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _make_nonnegative(seed=20260507):
    rng = np.random.default_rng(seed)
    W = rng.random((100, 3))
    H = rng.random((3, 8))
    return W @ H + 0.02 * rng.random((100, 8))


def test_minibatch_nmf_basic_interface_and_sklearn_smoke():
    sk_decomposition = pytest.importorskip("sklearn.decomposition")
    X = _make_nonnegative()
    model = MiniBatchNMF(n_components=3, batch_size=25, max_iter=10, random_state=0, device="cpu").fit(X)
    sk = sk_decomposition.MiniBatchNMF(
        n_components=3,
        batch_size=25,
        max_iter=10,
        random_state=0,
        init="random",
        beta_loss="frobenius",
    ).fit(X)

    W = _to_numpy(model.transform(X))
    X_inv = _to_numpy(model.inverse_transform(W))
    assert model.components_.shape == (3, X.shape[1])
    assert model.n_components_ == 3
    assert model.n_features_in_ == X.shape[1]
    assert np.isfinite(model.reconstruction_err_)
    assert W.shape == (X.shape[0], 3)
    assert X_inv.shape == X.shape
    assert np.all(_to_numpy(model.components_) >= 0.0)
    assert np.all(W >= 0.0)
    assert model.reconstruction_err_ <= np.linalg.norm(X)
    assert np.isfinite(sk.reconstruction_err_)


def test_minibatch_nmf_partial_fit():
    X = _make_nonnegative(seed=5)
    model = MiniBatchNMF(n_components=3, batch_size=20, max_iter=5, random_state=5, device="cpu")
    model.partial_fit(X[:50]).partial_fit(X[50:])

    W = _to_numpy(model.predict(X))
    assert W.shape == (X.shape[0], 3)
    assert model.n_iter_ == 2
    assert np.all(W >= 0.0)


def test_minibatch_nmf_validation_errors():
    X = _make_nonnegative(seed=9)
    with pytest.raises(ValueError, match="n_components"):
        MiniBatchNMF(n_components=0, device="cpu").fit(X)
    with pytest.raises(ValueError, match="non-negative"):
        MiniBatchNMF(n_components=2, device="cpu").fit(X - 2.0)
    with pytest.raises(NotImplementedError, match="init"):
        MiniBatchNMF(n_components=2, init="nndsvd", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        MiniBatchNMF(n_components=2, device="cpu").fit(sparse.csr_matrix(X))


def test_minibatch_nmf_cuda_torch_matches_cpu_when_available():
    X = _make_nonnegative(seed=13)
    cpu = MiniBatchNMF(n_components=3, batch_size=25, max_iter=5, random_state=3, device="cpu").fit(X)

    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")
    gpu = MiniBatchNMF(n_components=3, batch_size=25, max_iter=5, random_state=3, device="cuda").fit(X)
    assert np.isclose(gpu.reconstruction_err_, cpu.reconstruction_err_, rtol=1e-6, atol=1e-6)

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")
    tgpu = MiniBatchNMF(n_components=3, batch_size=25, max_iter=5, random_state=3, device="torch").fit(X)
    assert np.isclose(tgpu.reconstruction_err_, cpu.reconstruction_err_, rtol=1e-6, atol=1e-6)
