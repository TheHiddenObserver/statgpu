"""Tests for statgpu.unsupervised.GaussianMixture."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from statgpu.unsupervised import GaussianMixture


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _make_gmm_data(seed=20260501):
    rng = np.random.default_rng(seed)
    means = np.array([[-3.0, 0.0], [2.5, 1.0], [0.5, -3.0]])
    scales = np.array([[0.35, 0.6], [0.5, 0.3], [0.4, 0.45]])
    parts = [m + rng.normal(scale=s, size=(80, 2)) for m, s in zip(means, scales)]
    return np.vstack(parts)


def _match_rows(a, b):
    distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    rows, cols = linear_sum_assignment(distances)
    return a[rows], b[cols]


@pytest.mark.parametrize(
    ("covariance_type", "covariance_shape"),
    [
        ("diag", (3, 2)),
        ("spherical", (3,)),
        ("tied", (2, 2)),
        ("full", (3, 2, 2)),
    ],
)
def test_gmm_covariance_types_basic_interface_and_sklearn_score_alignment(covariance_type, covariance_shape):
    sk_mixture = pytest.importorskip("sklearn.mixture")
    X = _make_gmm_data()

    sg = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        random_state=7,
        max_iter=100,
        tol=1e-5,
        device="cpu",
    ).fit(X)
    sk = sk_mixture.GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        random_state=7,
        max_iter=100,
        tol=1e-5,
        reg_covar=1e-6,
        n_init=1,
    ).fit(X)

    assert sg.weights_.shape == (3,)
    assert sg.means_.shape == (3, 2)
    assert sg.covariances_.shape == covariance_shape
    assert sg.precisions_cholesky_.shape == covariance_shape
    assert sg.n_features_in_ == 2
    assert np.isfinite(sg.lower_bound_)
    assert np.isfinite(sg.score(X))
    assert np.isfinite(sg.bic(X))
    assert np.isfinite(sg.aic(X))

    sg_means, sk_means = _match_rows(_to_numpy(sg.means_), sk.means_)
    assert np.allclose(sg_means, sk_means, atol=0.25)
    assert np.isclose(sg.score(X), sk.score(X), rtol=0.08, atol=0.15)


def test_gmm_predict_proba_shapes_and_normalization():
    X = _make_gmm_data(seed=5)
    model = GaussianMixture(n_components=3, random_state=5, max_iter=30, device="cpu").fit(X)

    proba = _to_numpy(model.predict_proba(X))
    labels = _to_numpy(model.predict(X))
    samples = _to_numpy(model.score_samples(X))

    assert proba.shape == (X.shape[0], 3)
    assert labels.shape == (X.shape[0],)
    assert samples.shape == (X.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_gmm_validation_errors():
    X = _make_gmm_data()
    with pytest.raises(ValueError, match="n_components"):
        GaussianMixture(n_components=0, device="cpu").fit(X)
    with pytest.raises(ValueError, match="covariance_type"):
        GaussianMixture(n_components=2, covariance_type="bad", device="cpu").fit(X)
    with pytest.raises(ValueError, match="init_params"):
        GaussianMixture(n_components=2, init_params="bad", device="cpu").fit(X)
    with pytest.raises(NotImplementedError, match="sparse"):
        GaussianMixture(n_components=2, device="cpu").fit(sparse.csr_matrix(X))


@pytest.mark.parametrize("covariance_type", ["diag", "spherical", "tied", "full"])
def test_gmm_cuda_torch_score_matches_cpu_when_available(covariance_type):
    X = _make_gmm_data(seed=17)
    cpu = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        random_state=11,
        max_iter=30,
        device="cpu",
    ).fit(X)

    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("CuPy CUDA not available")
    gpu = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        random_state=11,
        max_iter=30,
        device="cuda",
    ).fit(X)
    assert np.isclose(gpu.score(X), cpu.score(X), rtol=1e-5, atol=1e-5)

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")
    tgpu = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        random_state=11,
        max_iter=30,
        device="torch",
    ).fit(X)
    assert np.isclose(tgpu.score(X), cpu.score(X), rtol=1e-5, atol=1e-5)
