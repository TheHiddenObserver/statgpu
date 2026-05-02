"""Tests for statgpu.unsupervised.PCA."""

import numpy as np
import pytest

from statgpu.unsupervised import PCA


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _orthogonal_projector(components):
    return components.T @ components


def test_pca_cpu_basic_interface_and_inverse_transform():
    rng = np.random.default_rng(20260430)
    X = rng.normal(size=(120, 8))
    X[:, 1] = 0.4 * X[:, 0] + 0.6 * X[:, 1]

    model = PCA(n_components=4, device="cpu")
    Xt = model.fit_transform(X)
    Xr = model.inverse_transform(Xt)

    assert Xt.shape == (120, 4)
    assert Xr.shape == X.shape
    assert model.components_.shape == (4, 8)
    assert model.mean_.shape == (8,)
    assert model.explained_variance_.shape == (4,)
    assert model.explained_variance_ratio_.shape == (4,)
    assert model.singular_values_.shape == (4,)
    assert model.n_components_ == 4
    assert model.n_features_in_ == 8
    assert np.all(np.isfinite(_to_numpy(Xt)))
    assert np.all(np.isfinite(_to_numpy(Xr)))


@pytest.mark.parametrize("solver", ["full", "covariance", "auto"])
def test_pca_cpu_matches_sklearn_variance_and_subspace(solver):
    sk_decomposition = pytest.importorskip("sklearn.decomposition")

    rng = np.random.default_rng(20260430)
    X = rng.normal(size=(160, 6))
    X[:, 3] = 0.8 * X[:, 0] - 0.2 * X[:, 2] + 0.1 * X[:, 3]

    sg = PCA(n_components=3, svd_solver=solver, device="cpu").fit(X)
    sk = sk_decomposition.PCA(n_components=3, svd_solver="full").fit(X)

    assert np.allclose(
        _to_numpy(sg.explained_variance_),
        sk.explained_variance_,
        rtol=1e-6,
        atol=1e-8,
    )
    assert np.allclose(
        _to_numpy(sg.explained_variance_ratio_),
        sk.explained_variance_ratio_,
        rtol=1e-6,
        atol=1e-8,
    )
    assert np.allclose(
        _orthogonal_projector(_to_numpy(sg.components_)),
        _orthogonal_projector(sk.components_),
        rtol=1e-6,
        atol=1e-8,
    )


def test_pca_whiten_transform_has_unit_sample_variance():
    rng = np.random.default_rng(20260430)
    X = rng.normal(size=(180, 5))

    Xt = _to_numpy(PCA(n_components=3, whiten=True, device="cpu").fit_transform(X))

    assert np.allclose(np.var(Xt, axis=0, ddof=1), np.ones(3), rtol=1e-6, atol=1e-8)


def test_pca_predict_aliases_transform():
    rng = np.random.default_rng(20260430)
    X = rng.normal(size=(80, 4))
    model = PCA(n_components=2, device="cpu").fit(X)

    assert np.allclose(_to_numpy(model.predict(X)), _to_numpy(model.transform(X)))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_components": 0}, "n_components"),
        ({"n_components": 99}, "n_components"),
        ({"svd_solver": "bad"}, "svd_solver"),
        ({"n_oversamples": -1}, "n_oversamples"),
        ({"iterated_power": -1}, "iterated_power"),
    ],
)
def test_pca_validation_errors(kwargs, match):
    X = np.ones((10, 3))
    with pytest.raises(ValueError, match=match):
        PCA(device="cpu", **kwargs).fit(X)


def test_pca_randomized_solver_matches_low_rank_subspace():
    sk_decomposition = pytest.importorskip("sklearn.decomposition")

    rng = np.random.default_rng(20260430)
    latent = rng.normal(size=(240, 3))
    loadings = rng.normal(size=(3, 20))
    X = latent @ loadings + rng.normal(scale=0.01, size=(240, 20))

    sg = PCA(
        n_components=3,
        svd_solver="randomized",
        random_state=7,
        n_oversamples=8,
        iterated_power=3,
        device="cpu",
    ).fit(X)
    sk = sk_decomposition.PCA(n_components=3, svd_solver="full").fit(X)

    assert np.allclose(
        _to_numpy(sg.explained_variance_),
        sk.explained_variance_,
        rtol=1e-3,
        atol=1e-5,
    )
    assert np.allclose(
        _orthogonal_projector(_to_numpy(sg.components_)),
        _orthogonal_projector(sk.components_),
        rtol=1e-3,
        atol=1e-5,
    )


def test_pca_cuda_matches_cpu_transform():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")

    rng = np.random.default_rng(20260430)
    X = rng.normal(size=(256, 10))

    cpu = PCA(n_components=5, device="cpu").fit(X)
    gpu = PCA(n_components=5, device="cuda").fit(X)

    assert np.allclose(
        np.abs(_to_numpy(gpu.transform(X))),
        np.abs(_to_numpy(cpu.transform(X))),
        rtol=1e-4,
        atol=1e-5,
    )


def test_pca_explicit_torch_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")

    rng = np.random.default_rng(20260430)
    X = rng.normal(size=(180, 7))

    cpu = PCA(n_components=4, device="cpu").fit(X)
    torch_model = PCA(n_components=4, device="torch").fit(X)

    assert np.allclose(
        np.abs(_to_numpy(torch_model.transform(X))),
        np.abs(_to_numpy(cpu.transform(X))),
        rtol=1e-4,
        atol=1e-5,
    )
