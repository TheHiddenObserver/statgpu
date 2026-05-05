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
    assert model.graph_.shape == (25, 25)
    assert model.n_epochs_ == 8
    assert model.n_features_in_ == 5
    assert np.all(np.isfinite(embedding))


def test_umap_seeded_random_init_is_reproducible():
    X = _data()
    params = dict(n_neighbors=5, n_epochs=5, init="random", random_state=42, device="cpu")
    emb1 = UMAP(**params).fit_transform(X)
    emb2 = UMAP(**params).fit_transform(X)
    np.testing.assert_allclose(emb1, emb2)


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
