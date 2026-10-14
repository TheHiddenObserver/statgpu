"""Exercise the bilingual unsupervised API reference against public runtime APIs."""

import inspect
import re
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from statgpu import unsupervised

ROOT = Path(__file__).resolve().parents[2]
GUIDES = {
    "PCA": "pca",
    "KMeans": "kmeans",
    "DBSCAN": "dbscan",
    "GaussianMixture": "gaussian-mixture",
    "NMF": "nmf",
    "AgglomerativeClustering": "agglomerative-clustering",
    "TruncatedSVD": "truncated-svd",
    "MiniBatchKMeans": "minibatch-kmeans",
    "IncrementalPCA": "incremental-pca",
    "MiniBatchNMF": "minibatch-nmf",
    "UMAP": "umap",
    "TSNE": "tsne",
}
SHARED_METHODS = {
    "get_params", "set_params", "adjust_pvalues", "combine_pvalues",
    "bootstrap_statistic", "permutation_test",
}
OPTIONS = {
    "PCA": {"n_components": 2, "svd_solver": "full"},
    "KMeans": {"n_clusters": 2, "random_state": 12},
    "DBSCAN": {"eps": 2.0, "min_samples": 2},
    "GaussianMixture": {"n_components": 2, "random_state": 12},
    "NMF": {"n_components": 2, "max_iter": 20, "random_state": 12},
    "AgglomerativeClustering": {"n_clusters": 2},
    "TruncatedSVD": {"n_components": 2, "algorithm": "full"},
    "MiniBatchKMeans": {"n_clusters": 2, "batch_size": 8, "random_state": 12},
    "IncrementalPCA": {"n_components": 2, "batch_size": 8},
    "MiniBatchNMF": {"n_components": 2, "max_iter": 3, "random_state": 12},
    "UMAP": {"n_neighbors": 3, "n_components": 2, "n_epochs": 2,
             "init": "random", "nn_method": "exact", "random_state": 12},
    "TSNE": {"n_components": 2, "perplexity": 3, "max_iter": 250,
             "init": "random", "random_state": 12},
}


def _page(language):
    return (ROOT / f"docs/{language}/unsupervised/api-reference.md").read_text()


def _section(text, name):
    return text.split(f"\n## {name}\n", 1)[1].split("\n## ", 1)[0]


def _signature(name, callable_object, omit_self=False):
    parameters = []
    for parameter in inspect.signature(callable_object).parameters.values():
        if omit_self and parameter.name == "self":
            continue
        default = parameter.default
        if isinstance(default, Enum):
            default = default.value
        parameters.append(parameter.replace(annotation=inspect.Parameter.empty, default=default))
    return name + str(inspect.Signature(parameters))


@pytest.mark.parametrize("language", ("en", "cn"))
def test_reference_covers_runtime_exports_constructor_defaults_and_methods(language):
    text = _page(language)
    assert set(GUIDES) == set(unsupervised.__all__)
    for name in GUIDES:
        cls = getattr(unsupervised, name)
        section = _section(text, name)
        assert _signature(name, cls) in section
        constructor_rows = re.findall(r"^\| `([a-z_]+)` \| `([^`]+)` \|", section, re.MULTILINE)
        expected = {}
        for key, parameter in inspect.signature(cls).parameters.items():
            default = parameter.default
            expected[key] = repr(default.value if isinstance(default, Enum) else default)
        assert dict(constructor_rows) == expected
        documented_methods = re.findall(r"^\| `([a-z_]+\([^`]*\))` \|", section, re.MULTILINE)
        runtime_methods = {
            key: member for key, member in inspect.getmembers(cls, callable)
            if not key.startswith("_") and key not in SHARED_METHODS
        }
        assert set(documented_methods) == {
            _signature(key, method, omit_self=True) for key, method in runtime_methods.items()
        }
    for method in SHARED_METHODS:
        assert method in text
    assert "../reference/estimator-api.md#parameter-management" in text
    assert "../reference/estimator-api.md#inference-helpers" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_every_guide_routes_to_its_complete_model_reference(language):
    directory = ROOT / f"docs/{language}/unsupervised"
    text = _page(language)
    for name, filename in GUIDES.items():
        assert f"](api-reference.md#{name.lower()})" in (directory / f"{filename}.md").read_text()
        assert f"\n## {name}\n" in text
        assert f"]({filename}.md)" in _section(text, name)
    assert "](api-reference.md)" in (directory / "README.md").read_text()
    assert not any(ord(character) < 32 and character not in "\n\t" for character in text)
    # Protect the mathematical bridge from Python/string escaping accidents.
    assert r"(X-\bar X)W^\top" in text
    assert r"\mathrm{AIC}=2d-2L" in text
    assert r"\mathrm{BIC}=d\log m-2L" in text


@pytest.fixture(scope="module")
def fitted_models():
    X = np.random.default_rng(12).normal(size=(24, 4))
    models = {}
    for name, options in OPTIONS.items():
        data = np.abs(X) + 0.1 if name in {"NMF", "MiniBatchNMF"} else X
        model = getattr(unsupervised, name)(device="cpu", **options)
        assert model.fit(data) is model
        models[name] = (model, data)
    return models


@pytest.mark.parametrize("name", GUIDES)
def test_reference_lists_observed_fitted_outputs(name, fitted_models):
    model, data = fitted_models[name]
    for language in ("en", "cn"):
        section = _section(_page(language), name)
        outputs = {key for key in vars(model) if key.endswith("_") and not key.startswith("_")}
        assert outputs
        assert all(f"`{key}`" in section for key in outputs)
    assert model.n_features_in_ == data.shape[1]


@pytest.mark.parametrize("name", ("PCA", "TruncatedSVD", "NMF", "IncrementalPCA", "MiniBatchNMF"))
def test_projection_factor_and_reconstruction_shapes(name, fitted_models):
    model, data = fitted_models[name]
    reduced = model.transform(data[:5])
    assert reduced.shape == (5, 2)
    assert model.inverse_transform(reduced).shape == (5, 4)
    np.testing.assert_allclose(model.predict(data[:5]), reduced)
    with pytest.raises(ValueError, match="features"):
        model.transform(data[:5, :3])
    with pytest.raises(ValueError, match="components"):
        model.inverse_transform(reduced[:, :1])


@pytest.mark.parametrize("name", ("KMeans", "MiniBatchKMeans"))
def test_cluster_distances_and_score_sign(name, fitted_models):
    model, data = fitted_models[name]
    distances = model.transform(data[:5])
    assert distances.shape == (5, 2)
    np.testing.assert_array_equal(model.predict(data[:5]), np.argmin(distances, axis=1))
    assert model.score(data[:5]) == pytest.approx(-np.sum(np.min(distances, axis=1) ** 2))
    with pytest.raises(NotImplementedError, match="sample_weight"):
        getattr(unsupervised, name)(device="cpu", n_clusters=2).fit(data, sample_weight=np.ones(len(data)))


def test_mixture_probabilities_scores_and_information_criteria(fitted_models):
    model, data = fitted_models["GaussianMixture"]
    responsibilities = model.predict_proba(data)
    assert responsibilities.shape == (24, 2)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1.0)
    log_density = model.score_samples(data)
    assert log_density.shape == (24,)
    assert model.score(data) == pytest.approx(log_density.mean())
    # Two-component, four-feature diagonal mixture: means + variances + weights.
    n_parameters = 2 * 4 + 2 * 4 + 1
    assert model.aic(data) == pytest.approx(2 * n_parameters - 2 * log_density.sum())
    assert model.bic(data) == pytest.approx(n_parameters * np.log(len(data)) - 2 * log_density.sum())


@pytest.mark.parametrize("name", ("DBSCAN", "AgglomerativeClustering", "UMAP", "TSNE"))
def test_unsupported_prediction_stubs_are_explicit(name, fitted_models):
    model, data = fitted_models[name]
    with pytest.raises(NotImplementedError):
        model.predict(data)
    if name in {"UMAP", "TSNE"}:
        assert model.embedding_.shape == (24, 2)
        with pytest.raises(NotImplementedError):
            model.transform(data)
    else:
        assert model.labels_.shape == (24,)


def test_umap_graph_is_an_edge_tuple(fitted_models):
    model, _ = fitted_models["UMAP"]
    source, target, weights, n_samples = model.graph_
    assert source.shape == target.shape == weights.shape
    assert n_samples == 24
    assert np.all((weights >= 0) & (weights <= 1))


@pytest.mark.parametrize("language", ("en", "cn"))
def test_documented_incremental_example_runs_independently(language):
    blocks = re.findall(r"```python\n(.*?)```", _page(language), flags=re.DOTALL)
    assert len(blocks) == 1
    namespace = {}
    exec(compile(blocks[0], f"{language}/unsupervised/api-reference.md", "exec"), namespace)  # noqa: S102
    assert namespace["Z"].shape == (5, 2)
    assert namespace["reconstructed"].shape == (5, 4)
    assert namespace["ipca"].n_samples_seen_ == 36
    assert namespace["km"].labels_.shape == (12,)
    assert namespace["labels"].shape == (5,)
    assert namespace["factors"].shape == (5, 2)


def test_incremental_first_batch_width_and_default_rank_rules():
    data = np.random.default_rng(18).normal(size=(9, 4))
    # fit chooses the default rank from the full array, and enlarges its first
    # internal batch; direct partial_fit chooses from its first supplied batch.
    full_rank = unsupervised.IncrementalPCA(batch_size=2, device="cpu").fit(data)
    assert full_rank.n_components_ == 4
    assert full_rank.n_samples_seen_ == 9
    with pytest.raises(ValueError, match="at least n_components"):
        unsupervised.IncrementalPCA(n_components=3, device="cpu").partial_fit(data[:2])
    with pytest.raises(ValueError, match="n_clusters"):
        unsupervised.MiniBatchKMeans(n_clusters=3, device="cpu").partial_fit(data[:2])
    explicit = unsupervised.MiniBatchKMeans(n_clusters=3, init=data[:3].copy(), device="cpu")
    assert explicit.partial_fit(data[:1]) is explicit
    for cls in (unsupervised.IncrementalPCA, unsupervised.MiniBatchNMF):
        model = cls(device="cpu")
        positive = np.abs(data) + 0.1
        assert model.partial_fit(positive[:2]) is model
        assert model.n_components_ == 2
        model.partial_fit(positive[2:])
        assert model.n_components_ == 2
        with pytest.raises(ValueError, match="features"):
            model.partial_fit(positive[:, :3])
    model = unsupervised.MiniBatchKMeans(n_clusters=3, device="cpu", random_state=18)
    model.partial_fit(data[:4]).partial_fit(data[4:5])
    assert model.labels_.shape == (1,)
    with pytest.raises(ValueError, match="features"):
        model.partial_fit(data[:, :3])


@pytest.mark.parametrize("name", GUIDES)
def test_parameter_updates_invalidate_unsupervised_fits(name, fitted_models):
    model, _ = fitted_models[name]
    # A new fit keeps this check independent of other tests sharing the fixture.
    clone = type(model)(**model.get_params(deep=False))
    data = np.abs(np.random.default_rng(20).normal(size=(24, 4))) + 0.1
    clone.fit(data)
    assert clone.set_params(device="cpu") is clone
    assert not clone._fitted
    assert clone.get_params(deep=False)["device"] == "cpu"
