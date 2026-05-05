import numpy as np
import pytest
from scipy import sparse

from statgpu.unsupervised import TruncatedSVD


def _data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 8))


def test_truncated_svd_full_matches_numpy_singular_values():
    X = _data()
    model = TruncatedSVD(n_components=3, algorithm="full", device="cpu").fit(X)
    _, s, vh = np.linalg.svd(X, full_matrices=False)

    assert model.components_.shape == (3, 8)
    assert model.singular_values_.shape == (3,)
    np.testing.assert_allclose(model.singular_values_, s[:3], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.abs(model.components_), np.abs(vh[:3]), rtol=1e-7, atol=1e-7)


def test_truncated_svd_transform_inverse_shapes():
    X = _data()
    model = TruncatedSVD(n_components=4, algorithm="randomized", random_state=0, device="cpu")
    Z = model.fit_transform(X)
    X_hat = model.inverse_transform(Z)

    assert Z.shape == (30, 4)
    assert X_hat.shape == X.shape
    assert model.explained_variance_ratio_.shape == (4,)
    assert model.n_features_in_ == 8


def test_truncated_svd_rejects_invalid_inputs():
    X = _data()
    with pytest.raises(ValueError, match="algorithm"):
        TruncatedSVD(algorithm="arpack", device="cpu").fit(X)
    with pytest.raises(ValueError, match="n_components"):
        TruncatedSVD(n_components=20, device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        TruncatedSVD(device="cpu").fit(sparse.csr_matrix(X))
