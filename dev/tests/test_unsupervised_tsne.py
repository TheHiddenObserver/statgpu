import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import TSNE


def _data():
    rng = np.random.default_rng(2)
    return rng.normal(size=(28, 6))


def test_tsne_fit_transform_shape_and_attributes():
    X = _data()
    model = TSNE(
        n_components=2,
        perplexity=5,
        max_iter=250,
        init="random",
        random_state=0,
        device="cpu",
    )
    embedding = model.fit_transform(X)

    assert embedding.shape == (28, 2)
    assert model.n_iter_ == 250
    assert model.n_features_in_ == 6
    assert np.isfinite(model.kl_divergence_)
    assert np.all(np.isfinite(embedding))


def test_tsne_seeded_random_init_is_reproducible():
    X = _data()
    params = dict(perplexity=5, max_iter=250, init="random", random_state=42, device="cpu")
    emb1 = TSNE(**params).fit_transform(X)
    emb2 = TSNE(**params).fit_transform(X)
    np.testing.assert_allclose(emb1, emb2)


def test_tsne_rejects_unsupported_modes():
    X = _data()
    with pytest.raises(NotImplementedError, match="metric"):
        TSNE(metric="cosine", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        TSNE(device="cpu").fit(sparse.csr_matrix(X))
    with pytest.raises(NotImplementedError, match="transforming new data"):
        TSNE(perplexity=5, max_iter=250, init="random", device="cpu").fit(X).transform(X)
