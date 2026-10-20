import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import UMAP


def _data():
    rng = np.random.default_rng(1)
    return rng.normal(size=(25, 5))


def test_umap_fit_transform_shape_and_attributes():
    X = _data()
    model = UMAP(
        n_neighbors=5,
        n_components=2,
        n_epochs=8,
        init="random",
        random_state=0,
        device="cpu",
    )
    embedding = model.fit_transform(X)

    assert embedding.shape == (25, 2)
    # graph_ is now sparse COO representation: (src, dst, weights, n_samples)
    assert isinstance(model.graph_, tuple) and len(model.graph_) == 4
    assert model.graph_[3] == 25  # n_samples
    assert hasattr(model, 'n_epochs_')  # epochs may vary based on sparse init
    assert model.n_features_in_ == 5
    assert np.all(np.isfinite(embedding))


def test_umap_seeded_random_init_is_reproducible():
    X = _data()
    params = dict(n_neighbors=5, n_epochs=5, init="random", random_state=42, device="cpu")
    emb1 = UMAP(**params).fit_transform(X)
    emb2 = UMAP(**params).fit_transform(X)
    # With fixed RNG (created once before epoch loop), seeded runs must be identical
    np.testing.assert_allclose(emb1, emb2, rtol=1e-10, atol=1e-10)


def test_umap_rejects_unsupported_modes():
    X = _data()
    with pytest.raises(NotImplementedError, match="metric"):
        UMAP(metric="cosine", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        UMAP(device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="transforming new data"):
        UMAP(n_neighbors=5, n_epochs=2, init="random", device="cpu").fit(X).transform(X)


def test_umap_min_dist_and_spread_change_embedding():
    X = _data()
    base = UMAP(
        n_neighbors=5,
        n_components=2,
        n_epochs=8,
        init="random",
        random_state=7,
        min_dist=0.05,
        spread=1.0,
        device="cpu",
    ).fit_transform(X)
    changed = UMAP(
        n_neighbors=5,
        n_components=2,
        n_epochs=8,
        init="random",
        random_state=7,
        min_dist=0.8,
        spread=2.0,
        device="cpu",
    ).fit_transform(X)
    assert not np.allclose(base, changed)


def test_umap_attraction_curve_params_are_positive():
    model = UMAP(min_dist=0.2, spread=1.5, device="cpu")
    a, b = model._attraction_curve_params()
    assert np.isfinite(a) and a > 0.0
    assert np.isfinite(b) and b > 0.0


@pytest.mark.parametrize("nn_method", ["exact", "nndescent"])
@pytest.mark.parametrize("n_components", [2, 3])
def test_umap_cpu_dispatch_and_rng_without_gpu_libraries(monkeypatch, nn_method, n_components):
    """NumPy 2 ndarray.device must not select Torch NNDescent or RNG."""
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "cupy"}:
            raise AssertionError(f"CPU UMAP attempted to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    params = dict(n_neighbors=5, n_components=n_components, n_epochs=4,
                  init="random", random_state=19, device="cpu", nn_method=nn_method)
    model = UMAP(**params).fit(_data())
    repeat = UMAP(**params).fit(_data())
    assert isinstance(model.embedding_, np.ndarray)
    assert model.embedding_.shape == (25, n_components)
    assert np.isfinite(model.embedding_).all()
    np.testing.assert_array_equal(model.embedding_, repeat.embedding_)
    for actual, expected in zip(model.graph_[:3], repeat.graph_[:3]):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    changed = UMAP(**dict(params, random_state=20)).fit(_data())
    assert not np.array_equal(model.embedding_, changed.embedding_)


@pytest.mark.parametrize("nn_method", ["exact", "nndescent"])
@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_umap_explicit_gpu_dispatch_when_available(device, nn_method):
    from statgpu.backends._utils import _to_numpy

    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            available = cp.cuda.runtime.getDeviceCount() > 0
        except cp.cuda.runtime.CUDARuntimeError:
            available = False
        if not available:
            pytest.skip("CuPy CUDA hardware unavailable")
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA hardware unavailable")
    params = dict(n_neighbors=5, n_epochs=3, init="random", random_state=19,
                  device=device, nn_method=nn_method)
    model = UMAP(**params).fit(_data())
    repeat = UMAP(**params).fit(_data())
    if device == "cuda":
        assert isinstance(model.embedding_, cp.ndarray)
        assert all(isinstance(x, cp.ndarray) for x in model.graph_[:3])
    else:
        assert model.embedding_.is_cuda
        assert all(x.is_cuda for x in model.graph_[:3])
    assert model.embedding_.shape == (25, 2)
    assert np.isfinite(_to_numpy(model.embedding_)).all()
    np.testing.assert_allclose(_to_numpy(model.embedding_), _to_numpy(repeat.embedding_),
                               rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("nn_method", ["exact", "nndescent"])
def test_umap_seeded_cpu_fit_preserves_global_numpy_rng(nn_method):
    before = np.random.get_state()
    UMAP(n_neighbors=5, n_epochs=3, init="random", random_state=19,
         device="cpu", nn_method=nn_method).fit(_data())
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
