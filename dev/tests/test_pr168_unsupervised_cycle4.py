"""Fresh PR168 unsupervised doc contracts and bounded numerical precautions.

Only NumPy CPU behavior is exercised. Strict xfails express the desired runtime
rejection of independently reproduced gaps; they do not bless invalid output.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.unsupervised import NMF, GaussianMixture, MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]


class _SingularGMMNotRejected(Exception):
    """The unregularized singular GMM fit returned instead of rejecting it."""


class _NonfiniteInitialCentersNotRejected(Exception):
    """The MiniBatchKMeans call accepted nonfinite initial centers."""


@pytest.mark.parametrize("covariance_type", ("diag", "spherical", "tied", "full"))
def test_positive_gmm_regularization_keeps_constant_data_outputs_finite(covariance_type):
    X = np.ones((4, 2))
    model = GaussianMixture(
        n_components=1, covariance_type=covariance_type,
        reg_covar=1e-6, random_state=0, device="cpu",
    ).fit(X)
    assert model.converged_
    assert np.isfinite(model.covariances_).all()
    assert np.isfinite(model.precisions_cholesky_).all()
    assert np.isfinite(model.score_samples(X)).all()
    np.testing.assert_allclose(model.predict_proba(X), np.ones((4, 1)))
    if covariance_type in {"diag", "spherical"}:
        assert np.all(model.covariances_ > 0)
    else:
        assert np.all(np.linalg.eigvalsh(model.covariances_) > 0)
    # The floor/ridge yields a valid positive covariance for this example.
    expected_density = -np.log(2 * np.pi * 1e-6)
    np.testing.assert_allclose(model.score_samples(X), expected_density, rtol=1e-8)


@pytest.mark.parametrize("covariance_type", ("diag", "spherical"))
@pytest.mark.xfail(
    strict=True,
    raises=_SingularGMMNotRejected,
    reason="Issue #225: Unregularized singular GMM currently publishes NaN fitted outputs",
)
def test_singular_gmm_rejects_fit_before_publishing_invalid_results(covariance_type):
    model = GaussianMixture(
        n_components=1, covariance_type=covariance_type,
        reg_covar=0, random_state=0, device="cpu",
    )
    X = np.ones((4, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            model.fit(X)
        except (ValueError, np.linalg.LinAlgError) as error:
            assert re.search("covariance|singular|positive|Singular", str(error))
        else:
            raise _SingularGMMNotRejected("Singular GMM fit did not reject its covariance")
    assert not model._fitted


@pytest.mark.parametrize("method", ("fit", "partial_fit"))
@pytest.mark.parametrize("nonfinite", (np.nan, np.inf, -np.inf))
@pytest.mark.xfail(
    strict=True,
    raises=_NonfiniteInitialCentersNotRejected,
    reason="Issue #226: Constructor initial-center arrays bypass public finite-input checks",
)
def test_minibatch_kmeans_rejects_nonfinite_initial_centers(method, nonfinite):
    X = np.array([[0., 0.], [1., 1.], [2., 2.]])
    centers = np.array([[nonfinite, 0.], [2., 2.]])
    model = None
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            model = MiniBatchKMeans(
                n_clusters=2, init=centers, random_state=0, device="cpu",
            )
            getattr(model, method)(X)
        except ValueError as error:
            assert re.search("finite", str(error))
        else:
            raise _NonfiniteInitialCentersNotRejected("Nonfinite initial centers were accepted")
    if model is not None:  # Constructor-time rejection is also valid.
        assert not model._fitted


@pytest.mark.parametrize("method", ("fit", "partial_fit"))
def test_finite_explicit_minibatch_centers_support_both_fit_apis(method):
    X = np.array([[0., 0.], [1., 1.], [2., 2.]])
    initial_centers = X[[0, 2]].copy()
    assert np.isfinite(initial_centers).all()
    original = initial_centers.copy()
    model = MiniBatchKMeans(
        n_clusters=2, init=initial_centers, random_state=0, device="cpu",
    )
    assert getattr(model, method)(X) is model
    assert np.isfinite(model.cluster_centers_).all()
    assert np.isfinite(model.inertia_)
    assert np.isfinite(model.transform(X)).all()
    np.testing.assert_array_equal(initial_centers, original)


@pytest.mark.parametrize("max_iter", (1, 4))
@pytest.mark.parametrize("tol", (0., 1e6))
def test_nmf_transform_runs_fixed_budget_independent_of_fit_tolerance(monkeypatch, max_iter, tol):
    X = np.array([[1., 2.], [2., 1.], [3., 4.]])
    model = NMF(
        n_components=1, max_iter=max_iter, tol=tol,
        random_state=0, device="cpu",
    ).fit(X)
    update = model._update_w
    calls = []

    def count_update(*args, **kwargs):
        calls.append(None)
        return update(*args, **kwargs)

    monkeypatch.setattr(model, "_update_w", count_update)
    result = model.transform(X)
    assert len(calls) == max_iter
    assert result.shape == (len(X), 1)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("language", ("en", "cn"))
def test_bilingual_pages_explain_constructor_validation_and_solver_boundaries(language):
    directory = ROOT / f"docs/{language}/unsupervised"
    reference = (directory / "api-reference.md").read_text()
    mixture = (directory / "gaussian-mixture.md").read_text()
    minibatch = (directory / "minibatch-kmeans.md").read_text()
    nmf = (directory / "nmf.md").read_text()
    overview = (ROOT / f"docs/{language}/models/unsupervised.md").read_text()
    assert "`reg_covar=0`" in mixture
    assert "NaN" in mixture and "NaN" in reference
    assert "`score_samples(X)`" in mixture and "`predict_proba(X)`" in mixture
    assert "np.isfinite(initial_centers).all()" in minibatch
    assert "partial_fit" in minibatch
    assert "minibatch-kmeans.md#" in overview
    assert "max_iter" in nmf and "tol" in nmf
    if language == "en":
        assert "without early stopping by `tol`" in reference
        assert "does not stop the transform solve early" in nmf
        assert "initial centers also need to be finite" in overview
    else:
        assert "不会按 `tol` 提前停止" in reference
        assert "不会让转换求解提前结束" in nmf
        assert "显式初始中心也必须有限" in overview
    assert "reg_covar=0" in GaussianMixture.__doc__
    assert "NaN/Inf" in MiniBatchKMeans.__doc__
    assert "does not stop early" in NMF.__doc__
