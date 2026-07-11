
import numpy as np
import pytest

from statgpu.penalties import AdaptiveL1Penalty
from statgpu.unsupervised import UMAP
from statgpu.unsupervised._nndescent import nndescent_numpy
import statgpu.unsupervised._umap as umap_module


def test_adaptive_l1_gradient_numpy():
    penalty = AdaptiveL1Penalty(
        alpha=2.0, weights=np.array([1.0, 3.0]), normalize=False
    )
    np.testing.assert_array_equal(
        penalty.gradient(np.array([-4.0, 5.0])), np.array([-2.0, 6.0])
    )


def test_nndescent_numpy_unique_and_validated():
    X = np.random.default_rng(123).normal(size=(24, 4))
    indices, distances = nndescent_numpy(X, k=5, max_iter=3, seed=7)
    assert indices.shape == distances.shape == (24, 5)
    assert np.all(np.isfinite(distances))
    for i, row in enumerate(indices):
        assert i not in row
        assert len(np.unique(row)) == 5
    with pytest.raises(ValueError, match="k must"):
        nndescent_numpy(X, k=24)


def test_umap_none_seed_draws_once_per_fit(monkeypatch):
    seeds = iter([101, 202])
    monkeypatch.setattr(umap_module, "draw_random_seed", lambda state: next(seeds))
    X = np.arange(60.0).reshape(20, 3)
    params = dict(
        n_neighbors=4,
        n_components=2,
        n_epochs=1,
        init="random",
        random_state=None,
        device="cpu",
    )
    first, second = UMAP(**params).fit(X), UMAP(**params).fit(X)
    assert (first._fit_random_seed_, second._fit_random_seed_) == (101, 202)
    assert not np.allclose(first.embedding_, second.embedding_)
