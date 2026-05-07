"""Tests for statgpu.unsupervised.NMF."""

import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import NMF


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _make_nmf_data(seed=20260501):
    rng = np.random.default_rng(seed)
    W = rng.random((50, 3))
    H = rng.random((3, 12))
    return W @ H + 0.01 * rng.random((50, 12))


def test_nmf_basic_interface_and_sklearn_reconstruction_scale():
    sk_decomposition = pytest.importorskip("sklearn.decomposition")
    X = _make_nmf_data()

    sg = NMF(n_components=3, max_iter=120, random_state=3, device="cpu").fit(X)
    sk = sk_decomposition.NMF(
        n_components=3,
        init="random",
        solver="mu",
        beta_loss="frobenius",
        max_iter=120,
        random_state=3,
        tol=1e-4,
    ).fit(X)

    assert sg.components_.shape == (3, X.shape[1])
    assert sg.n_components_ == 3
    assert sg.n_features_in_ == X.shape[1]
    assert sg.reconstruction_err_ <= sk.reconstruction_err_ * 1.5
    assert np.all(_to_numpy(sg.components_) >= 0.0)


def test_nmf_transform_inverse_shapes():
    X = _make_nmf_data(seed=7)
    model = NMF(n_components=4, max_iter=40, random_state=7, device="cpu").fit(X)

    W = model.transform(X)
    X_hat = model.inverse_transform(W)

    assert _to_numpy(W).shape == (X.shape[0], 4)
    assert _to_numpy(X_hat).shape == X.shape
    assert np.all(_to_numpy(W) >= 0.0)


def test_nmf_random_init_uses_random_state_when_sampling_rows():
    X = _make_nmf_data(seed=11)
    m0 = NMF(n_components=4, max_iter=1, random_state=0, device="cpu").fit(X)
    m1 = NMF(n_components=4, max_iter=1, random_state=1, device="cpu").fit(X)
    m0_repeat = NMF(n_components=4, max_iter=1, random_state=0, device="cpu").fit(X)

    assert np.allclose(_to_numpy(m0.components_), _to_numpy(m0_repeat.components_))
    assert not np.allclose(_to_numpy(m0.components_), _to_numpy(m1.components_))


def test_nmf_random_state_none_is_not_forced_to_fixed_seed():
    X = _make_nmf_data(seed=19)
    first = NMF(n_components=3, max_iter=1, random_state=None, device="cpu").fit(X)
    second = NMF(n_components=3, max_iter=1, random_state=None, device="cpu").fit(X)

    assert not np.allclose(_to_numpy(first.components_), _to_numpy(second.components_))


def test_nmf_validation_errors():
    X = _make_nmf_data()
    with pytest.raises(ValueError, match="non-negative"):
        NMF(n_components=2, device="cpu").fit(X - 10.0)
    with pytest.raises(NotImplementedError, match="sparse"):
        NMF(n_components=2, device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="solver"):
        NMF(n_components=2, solver="cd", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="beta_loss"):
        NMF(n_components=2, beta_loss="kullback-leibler", device="cpu").fit(X)


def test_nmf_cuda_torch_reconstruction_matches_cpu_when_available():
    X = _make_nmf_data(seed=9)
    cpu = NMF(n_components=3, max_iter=30, random_state=9, device="cpu").fit(X)

    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")
    gpu = NMF(n_components=3, max_iter=30, random_state=9, device="cuda").fit(X)
    assert np.isclose(gpu.reconstruction_err_, cpu.reconstruction_err_, rtol=1e-8, atol=1e-8)

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")
    tgpu = NMF(n_components=3, max_iter=30, random_state=9, device="torch").fit(X)
    assert np.isclose(tgpu.reconstruction_err_, cpu.reconstruction_err_, rtol=1e-8, atol=1e-8)
