"""Small-unit NMF documentation contracts, using NumPy CPU only.

Strict xfails identify only the reproduced near-zero-factor collapse. Runtime
errors, invalid arrays, and other reconstruction failures remain real failures.
These tests do not establish physical GPU or benchmark evidence.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu import unsupervised
from statgpu.unsupervised import NMF, MiniBatchNMF

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ((NMF, "fit_transform"), (MiniBatchNMF, "fit_transform"),
          (MiniBatchNMF, "partial_fit"))


class _CollapsedSmallPositiveFactors(Exception):
    """Finite positive rank-one data collapsed to essentially zero factors."""


def _rank_one():
    return np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])


def _fit_coordinates(estimator, operation, X, tol):
    model = estimator(n_components=1, max_iter=300, tol=tol,
                      random_state=5, device="cpu")
    if operation == "partial_fit":
        model.partial_fit(X)
        W = model.transform(X)
    else:
        W = model.fit_transform(X)
    reconstruction = model.inverse_transform(W)
    assert model._fitted
    assert W.shape == (len(X), 1)
    assert model.components_.shape == (1, X.shape[1])
    assert reconstruction.shape == X.shape
    for array in (W, model.components_, reconstruction):
        assert np.isfinite(array).all()
        assert np.all(array >= 0)
    np.testing.assert_allclose(reconstruction, W @ model.components_, rtol=1e-12, atol=0)
    return model, W, reconstruction


@pytest.mark.parametrize("estimator, operation", ROUTES)
@pytest.mark.parametrize("tol", (1e-4, 0.))
@pytest.mark.xfail(
    strict=True,
    raises=_CollapsedSmallPositiveFactors,
    reason="Issue #240: Fixed absolute NMF stabilizers collapse strictly positive small-unit factors",
)
def test_positive_rank_one_reconstruction_does_not_collapse_when_units_shrink(
    estimator, operation, tol,
):
    unit = _rank_one()
    _, _, ordinary = _fit_coordinates(estimator, operation, unit, tol)
    np.testing.assert_allclose(ordinary, unit, rtol=1e-10, atol=0)
    X = unit * 1e-12
    assert np.all(X > 0)  # Distinct from the zero-feature initialization gap.
    original = X.copy()
    _, _, reconstruction = _fit_coordinates(estimator, operation, X, tol)
    np.testing.assert_array_equal(X, original)
    relative_error = np.linalg.norm(X - reconstruction) / np.linalg.norm(X)
    relative_output = np.linalg.norm(reconstruction) / np.linalg.norm(X)
    if relative_output < 1e-12 and abs(relative_error - 1.) < 1e-12:
        raise _CollapsedSmallPositiveFactors("Positive rank-one data reconstructed as zero")
    # A different regression must not be hidden by the known-collapse xfail.
    assert relative_error < 1e-8


@pytest.mark.parametrize("estimator, operation", ROUTES)
@pytest.mark.parametrize("tol", (1e-4, 0.))
def test_fixed_common_scale_restores_original_units_for_simple_positive_factors(
    estimator, operation, tol,
):
    X = _rank_one() * 1e-12
    original = X.copy()
    scale = float(X.max())
    assert np.isfinite(scale) and scale > 0
    model, W, reconstructed_scaled = _fit_coordinates(estimator, operation, X / scale, tol)
    restored = reconstructed_scaled * scale
    np.testing.assert_allclose(restored / scale, X / scale, rtol=1e-10, atol=0)
    np.testing.assert_allclose(W @ (model.components_ * scale), restored,
                               rtol=1e-12, atol=0)
    components = model.components_.copy()
    new_rows = 1e-12 * np.array([[5., 10.], [6., 12.]])
    new_factors = model.transform(new_rows / scale)
    new_reconstruction = model.inverse_transform(new_factors) * scale
    np.testing.assert_allclose(new_reconstruction / scale, new_rows / scale,
                               rtol=1e-10, atol=0)
    np.testing.assert_array_equal(model.components_, components)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("tol", (1e-4, 0.))
def test_streaming_scale_stays_fixed_for_all_partial_fit_batches_and_transform(tol):
    X = _rank_one() * 1e-12
    scale = float(X[:2].max())  # Select once using buffered training rows.
    model = MiniBatchNMF(n_components=1, max_iter=300, tol=tol,
                         random_state=5, device="cpu")
    for batch in (X[:2], X[2:]):
        model.partial_fit(batch / scale)
    factors = model.transform(X / scale)
    restored = model.inverse_transform(factors) * scale
    np.testing.assert_allclose(restored / scale, X / scale, rtol=1e-10, atol=0)
    # The same positive scalar weights every entry of the squared-error loss.
    perturbed = restored + np.array([[1., -1.], [2., 1.], [-1., 2.], [1., 1.]]) * 1e-13
    original_loss = np.sum((X - perturbed) ** 2)
    scaled_loss = np.sum((X / scale - perturbed / scale) ** 2)
    np.testing.assert_allclose(scaled_loss * scale ** 2, original_loss, rtol=1e-12)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("slug", ("nmf", "minibatch-nmf"))
def test_small_unit_examples_run_independently_and_report_original_units(language, slug):
    path = ROOT / f"docs/{language}/unsupervised/{slug}.md"
    match = re.search(rf"<!-- learner-example: {slug}-units -->\s*```python\n(.*?)```",
                      path.read_text(), re.DOTALL)
    assert match is not None
    namespace = {}
    exec(compile(match.group(1), str(path), "exec"), namespace)  # noqa: S102
    scale = namespace["scale"]
    np.testing.assert_allclose(namespace["X_hat"] / scale, namespace["X"] / scale,
                               rtol=1e-10, atol=0)
    if slug == "nmf":
        np.testing.assert_allclose(namespace["new_hat"] / scale,
                                   namespace["new_rows"] / scale, rtol=1e-10, atol=0)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_bilingual_small_unit_cautions_cover_fit_streaming_and_inverse_mapping(language):
    directory = ROOT / f"docs/{language}/unsupervised"
    reference = (directory / "api-reference.md").read_text()
    anchor = "very-small-input-units" if language == "en" else "极小数值单位"
    for slug, estimator in (("nmf", NMF), ("minibatch-nmf", MiniBatchNMF)):
        guide = (directory / f"{slug}.md").read_text()
        assert "1e-12" in guide and "tol=0" in guide
        assert "inverse_transform" in guide and "scale" in guide
        assert f"{slug}.md#{anchor}" in reference
        assert "fixed" in estimator.__doc__ and "original" in estimator.__doc__
    assert ("same scale" if language == "en" else "同一个尺度") in (directory / "minibatch-nmf.md").read_text()


RANDOM_OPTIONS = {
    "PCA": {"n_components": 2, "svd_solver": "randomized"},
    "TruncatedSVD": {"n_components": 2, "algorithm": "randomized"},
    "NMF": {"n_components": 2, "max_iter": 3},
    "MiniBatchNMF": {"n_components": 2, "max_iter": 3},
    "UMAP": {"n_neighbors": 3, "n_epochs": 1, "init": "random"},
    "TSNE": {"perplexity": 3, "max_iter": 250, "init": "random"},
    "KMeans": {"n_clusters": 2},
    "MiniBatchKMeans": {"n_clusters": 2},
    "GaussianMixture": {"n_components": 2, "max_iter": 3},
}
NUMPY_SEED_CONSUMERS = {"KMeans", "MiniBatchKMeans", "GaussianMixture"}


@pytest.mark.parametrize("name", RANDOM_OPTIONS)
@pytest.mark.parametrize("factory", (np.random.default_rng, np.random.RandomState))
def test_documented_numpy_random_objects_are_stateful_fit_inputs(name, factory):
    if name in NUMPY_SEED_CONSUMERS and factory is np.random.RandomState:
        # These consumers deliberately defer accepted forms to NumPy. Older
        # NumPy releases do not coerce a legacy RandomState in default_rng.
        try:
            np.random.default_rng(factory(71))
        except TypeError:
            pytest.skip("Installed NumPy does not accept RandomState in default_rng")
    X = np.random.default_rng(70).uniform(0.1, 2., size=(12, 3))
    rng = factory(71)
    model = getattr(unsupervised, name)(random_state=rng, device="cpu", **RANDOM_OPTIONS[name])
    before = pickle.dumps(rng)
    assert model.get_params(deep=False)["random_state"] is rng
    assert model.fit(X) is model
    assert model._fitted and model.n_features_in_ == 3
    assert pickle.dumps(rng) != before


@pytest.mark.parametrize("name, options", (
    ("PCA", {"svd_solver": "full"}),
    ("PCA", {"svd_solver": "covariance"}),
    ("TruncatedSVD", {"algorithm": "full"}),
))
def test_deterministic_decomposition_does_not_consume_generator(name, options):
    X = np.random.default_rng(70).uniform(0.1, 2., size=(12, 3))
    rng = np.random.default_rng(71)
    before = pickle.dumps(rng)
    getattr(unsupervised, name)(n_components=2, random_state=rng,
                              device="cpu", **options).fit(X)
    assert pickle.dumps(rng) == before


@pytest.mark.parametrize("language", ("en", "cn"))
def test_randomness_reference_distinguishes_portable_and_numpy_dependent_forms(language):
    text = (ROOT / f"docs/{language}/unsupervised/api-reference.md").read_text()
    for name in RANDOM_OPTIONS:
        section = text.split(f"\n## {name}\n", 1)[1].split("\n## ", 1)[0]
        row = next(line for line in section.splitlines() if line.startswith("| `random_state` |"))
        assert "Generator" in row
        if name in NUMPY_SEED_CONSUMERS:
            assert "default_rng" in row
        else:
            assert "RandomState" in row and "2**32-1" in row
    assert "numpy.org/doc/stable/reference/random/generator.html" in text
    assert ("stateful" if language == "en" else "可变状态") in text
