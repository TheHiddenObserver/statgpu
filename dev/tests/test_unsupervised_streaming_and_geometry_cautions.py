"""Exercise documented streaming preparation and covariance/geometry contracts."""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.unsupervised import DBSCAN, GaussianMixture, MiniBatchNMF

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("language", ("en", "cn"))
def test_buffered_minibatch_nmf_example_initializes_active_features(language):
    path = ROOT / f"docs/{language}/unsupervised/minibatch-nmf.md"
    match = re.search(
        r"<!-- learner-example: minibatch-nmf-warmup -->\s*```python\n(.*?)```",
        path.read_text(), re.DOTALL,
    )
    assert match is not None
    namespace = {}
    exec(compile(match.group(1), str(path), "exec"), namespace)  # noqa: S102
    model = namespace["model"]
    assert np.all(np.any(model.components_ > 0, axis=0))
    reconstructed = namespace["reconstructed"]
    assert reconstructed.shape == namespace["second"].shape
    assert np.isfinite(reconstructed).all()
    assert np.all(reconstructed[:, 1] > 0)


def test_new_fit_reinitializes_minibatch_nmf_after_feature_support_changes():
    # Verify the documented restart, without requiring the old stream to fail.
    first = np.array([[1., 0.], [2., 0.], [3., 0.]])
    later = np.array([[1., 1.], [2., 2.], [3., 3.]])
    model = MiniBatchNMF(n_components=1, random_state=0, device="cpu")
    model.partial_fit(first)
    model.fit(later)
    fresh = MiniBatchNMF(n_components=1, random_state=0, device="cpu").fit(later)
    np.testing.assert_allclose(model.components_, fresh.components_)
    reconstructed = model.inverse_transform(model.transform(later))
    np.testing.assert_allclose(reconstructed, later, atol=1e-10)
    assert np.all(model.components_ > 0)


@pytest.mark.parametrize("batch_size", (1, 3, None))
def test_centered_dbscan_distance_helper_preserves_eps_neighborhoods(batch_size):
    # Execute the production backend helper with NumPy. This is not evidence
    # that a physical CuPy/Torch GPU path was executed.
    X = np.array([[0., 0.], [.25, 0.], [4., 0.], [4.25, 0.]]) + 1e8
    centered = X - X.mean(axis=0)
    model = DBSCAN(eps=.5, min_samples=2, batch_size=batch_size, device="cpu")
    rows, cols = model._neighbor_graph_sparse(model._get_backend(), centered)
    actual = np.zeros((len(X), len(X)), dtype=bool)
    actual[rows, cols] = True
    expected = np.linalg.norm(X[:, None] - X[None, :], axis=2) <= model.eps
    np.testing.assert_array_equal(actual, expected)
    cpu_fit = model.fit(centered)
    assert len(np.unique(cpu_fit.labels_)) == 2
    assert np.all(cpu_fit.labels_ >= 0)


@pytest.mark.parametrize("covariance_type", ("diag", "spherical", "tied", "full"))
def test_one_component_covariance_structure_and_density(covariance_type):
    X = np.array([[-4., -1.], [-2., 1.], [-1., -2.],
                  [1., 2.], [2., -1.], [4., 1.]])
    model = GaussianMixture(
        n_components=1, covariance_type=covariance_type, reg_covar=0.,
        init_params="random", random_state=0, device="cpu",
    ).fit(X)
    centered = X - X.mean(axis=0)
    empirical = centered.T @ centered / len(X)
    if covariance_type == "diag":
        covariance = np.diag(np.diag(empirical))
        np.testing.assert_allclose(model.covariances_[0], np.diag(covariance))
    elif covariance_type == "spherical":
        covariance = np.eye(X.shape[1]) * np.trace(empirical) / X.shape[1]
        assert model.covariances_[0] == pytest.approx(covariance[0, 0])
    elif covariance_type == "tied":
        covariance = empirical
        np.testing.assert_allclose(model.covariances_, covariance)
    else:
        covariance = empirical
        np.testing.assert_allclose(model.covariances_[0], covariance)
    expected = -.5 * (
        X.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]
        + np.einsum("ij,jk,ik->i", centered, np.linalg.inv(covariance), centered)
    )
    np.testing.assert_allclose(model.score_samples(X), expected, rtol=1e-12)
    assert model.converged_


@pytest.mark.parametrize("language", ("en", "cn"))
def test_streaming_and_gpu_geometry_cautions_are_reachable(language):
    directory = ROOT / f"docs/{language}/unsupervised"
    reference = (directory / "api-reference.md").read_text()
    nmf = (directory / "minibatch-nmf.md").read_text()
    dbscan = (directory / "dbscan.md").read_text()
    assert "minibatch-nmf.md#" in reference
    assert "float64" in dbscan and "float32" in dbscan
    assert "eps" in dbscan and "components_" in dbscan
    if language == "en":
        assert "zero-column initialization limit" in (directory / "README.md").read_text()
        assert "Later positive values cannot revive" in nmf
        assert "translation preserves Euclidean distances" in dbscan
    else:
        assert "MiniBatchNMF 首批全零特征的限制" in (directory / "README.md").read_text()
        assert "后续正值重新激活" in nmf
        assert "平移保持欧氏距离不变" in dbscan
    assert "zero" in MiniBatchNMF.__doc__
    assert "float32" in DBSCAN.__doc__
