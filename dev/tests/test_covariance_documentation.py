"""Bounded learner/API contracts for the EN/CN covariance documentation split.

Run the real CPU examples without injected setup and check their statistical
meaning. Physical GPU support is covered separately by the covariance suites.
"""

import ast
import inspect
from pathlib import Path
import re

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import statgpu
import statgpu.covariance as covariance
from doc_examples import example_code, parse_examples, run_example


ROOT = Path(__file__).resolve().parents[2]
CLASSES = (
    "EmpiricalCovariance", "ShrunkCovariance", "LedoitWolf", "OAS",
    "MinCovDet", "GraphicalLasso", "GraphicalLassoCV",
)
LESSONS = (
    "covariance-empirical", "covariance-shrunk", "covariance-ledoit-wolf",
    "covariance-oas", "covariance-min-cov-det", "covariance-graphical-lasso",
    "covariance-graphical-lasso-cv",
)
METHODS = {
    "fit", "score", "predict", "mahalanobis", "get_params", "set_params",
    "adjust_pvalues", "combine_pvalues", "bootstrap_statistic", "permutation_test",
}
CONCEPTS = (
    "covariance-correlation-precision", "units-and-standardization",
    "sample-size-and-stability", "choose-an-estimator", "joint-distance",
    "inference-boundaries",
)
ENTRY_POINTS = ("README", "usage", "models/README", "guides/implemented-methods")


def _read(language, page):
    return (ROOT / f"docs/{language}/{page}.md").read_text(encoding="utf-8")


def _run(language, page):
    name = "covariance-geometry" if page == "covariance" else page
    return run_example(_read(language, f"models/{page}"), name,
                       f"docs/{language}/models/{page}.md")


def _assert_gaussian_query(model, X):
    distances = np.einsum("ij,jk,ik->i", X - model.location_, model.precision_,
                          X - model.location_)
    np.testing.assert_allclose(model.mahalanobis(X), distances, rtol=1e-10)
    np.testing.assert_allclose(model.predict(X), distances, rtol=1e-10)
    expected = multivariate_normal.logpdf(
        X, mean=model.location_, cov=model.covariance_).mean()
    assert model.score(X) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("page", ("covariance", *LESSONS))
def test_bilingual_examples_are_identical_and_self_contained(page):
    expected = "covariance-geometry" if page == "covariance" else page
    en, cn = [_read(language, f"models/{page}") for language in ("en", "cn")]
    for text in (en, cn):
        examples = parse_examples(text)
        assert tuple(examples) == (expected,)
        assert examples[expected].requires == ()
    assert example_code(en, expected) == example_code(cn, expected)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ("covariance", *LESSONS))
def test_every_printed_tutorial_output_matches_execution(language, page, capsys):
    text = _read(language, f"models/{page}")
    # Every text fence on these learner pages is a published output, not code.
    expected = "\n".join(re.findall(r"```text\n(.*?)```", text, re.DOTALL))
    assert expected.strip(), page
    capsys.readouterr()
    with np.printoptions():
        _run(language, page)
    actual = capsys.readouterr().out
    assert actual.split() == expected.split(), page


@pytest.mark.parametrize("language", ("en", "cn"))
def test_shared_geometry_example_matches_covariance_and_gaussian_definitions(language):
    scope = _run(language, "covariance")
    model, X, queries = scope["model"], scope["X"], scope["queries"]
    np.testing.assert_allclose(model.covariance_, np.cov(X, rowvar=False, ddof=0))
    np.testing.assert_allclose(model.precision_, np.linalg.inv(model.covariance_))
    np.testing.assert_allclose(np.diag(scope["correlation"]), 1)
    assert scope["distance_squared"][1] > 10 * scope["distance_squared"][0]
    _assert_gaussian_query(model, queries)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_empirical_example_preserves_divisor_n_and_squared_distances(language):
    scope = _run(language, "covariance-empirical")
    model, X = scope["model"], scope["X"]
    np.testing.assert_allclose(model.location_, X.mean(axis=0))
    np.testing.assert_allclose(model.covariance_, np.cov(X, rowvar=False, ddof=0))
    assert not np.allclose(model.covariance_, np.cov(X, rowvar=False, ddof=1))
    _assert_gaussian_query(model, scope["queries"])


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ("covariance-shrunk", "covariance-ledoit-wolf", "covariance-oas"))
def test_shrinkage_examples_match_their_distinct_formulas(language, page):
    scope = _run(language, page)
    model = scope["oas"] if page == "covariance-oas" else scope["model"]
    X = scope.get("X_train", scope.get("X"))
    Z = X - X.mean(axis=0)
    n, p = Z.shape
    S = Z.T @ Z / n
    mu = np.trace(S) / p
    if page == "covariance-shrunk":
        alpha = 0.2
    elif page == "covariance-ledoit-wolf":
        beta = (np.sum(np.sum(Z * Z, axis=1) ** 2) - n * np.sum(S * S)) / n ** 2
        delta = np.sum(S * S) - np.trace(S) ** 2 / p
        alpha = 1.0 if delta <= 0 else np.clip(beta / delta, 0, 1)
    else:
        q = np.mean(S * S)
        denominator = (n + 1) * (q - mu * mu / p)
        alpha = 1.0 if denominator <= 0 else np.clip((q + mu * mu) / denominator, 0, 1)
    assert model.shrinkage_ == pytest.approx(alpha)
    np.testing.assert_allclose(model.covariance_, (1 - alpha) * S + alpha * mu * np.eye(p))
    assert np.linalg.cond(model.covariance_) < np.linalg.cond(S)
    _assert_gaussian_query(model, scope.get("X_test", X))


@pytest.mark.parametrize("language", ("en", "cn"))
def test_sparse_example_satisfies_penalized_precision_conditions(language):
    scope = _run(language, "covariance-graphical-lasso")
    model, X = scope["model"], scope["X_train"]
    S = np.cov(X, rowvar=False, ddof=0)
    precision = model.precision_
    fitted_covariance = np.linalg.inv(precision)
    np.testing.assert_allclose(model.covariance_, fitted_covariance, atol=1e-8)
    residual = fitted_covariance - S
    np.testing.assert_allclose(np.diag(residual), 0, atol=2e-4)
    off = ~np.eye(len(precision), dtype=bool)
    active = off & (np.abs(precision) > 1e-4)
    np.testing.assert_allclose(residual[active],
                               model.alpha * np.sign(precision[active]), atol=2e-4)
    assert np.max(np.abs(residual[off & ~active])) <= model.alpha + 2e-4
    assert model.n_iter_ <= model.max_iter
    _assert_gaussian_query(model, scope["X_test"])


@pytest.mark.parametrize("language", ("en", "cn"))
def test_cv_example_selects_from_actual_fold_means_and_refits_same_model(language):
    scope = _run(language, "covariance-graphical-lasso-cv")
    model, X = scope["model"], scope["X_train"]
    results = model.cv_results_
    assert isinstance(results, list)
    assert [r["alpha"] for r in results] == list(model.alphas)
    for result in results:
        assert set(result) == {"alpha", "mean_score", "scores"}
        assert len(result["scores"]) == model.cv
        assert result["mean_score"] == pytest.approx(np.mean(result["scores"]))
    assert model.alpha_ == max(results, key=lambda row: row["mean_score"])["alpha"]
    assert not hasattr(model, "n_iter_")
    refit = covariance.GraphicalLasso(alpha=model.alpha_, max_iter=model.max_iter,
                                     tol=model.tol, device="cpu").fit(X)
    np.testing.assert_allclose(model.covariance_, refit.covariance_)
    np.testing.assert_allclose(model.precision_, refit.precision_)
    assert model.n_samples_ == len(X)
    _assert_gaussian_query(model, scope["X_test"])


@pytest.mark.parametrize("language", ("en", "cn"))
def test_mcd_example_distinguishes_robust_support_from_known_contamination(language):
    scope = _run(language, "covariance-min-cov-det")
    model, X = scope["robust"], scope["X"]
    support = model.support_
    assert support.dtype == np.bool_ and support.shape == (len(X),)
    assert 0 < support[:180].sum() < 180  # Some known-clean rows are excluded.
    assert support[180:].sum() == 0
    np.testing.assert_allclose(model.dist_, model.mahalanobis(X))
    assert np.linalg.norm(model.location_) < np.linalg.norm(scope["ordinary"].location_)
    _assert_gaussian_query(model, X)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_complete_reference_matches_runtime_constructor_defaults_and_public_inventory(language):
    text = _read(language, "reference/covariance-api")
    assert set(covariance.__all__) == set(CLASSES)
    calls = {}
    for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        for node in ast.parse(block).body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id in CLASSES:
                    calls[call.func.id] = call
    assert set(calls) == set(CLASSES)
    for name in CLASSES:
        cls = getattr(covariance, name)
        assert getattr(statgpu, name) is cls
        call = calls[name]
        signature = inspect.signature(cls)
        assert not call.args
        assert [k.arg for k in call.keywords] == list(signature.parameters)
        for keyword in call.keywords:
            value = keyword.value
            if isinstance(value, ast.Attribute):
                assert ast.unparse(value) == "Device.AUTO"
                default = statgpu.Device.AUTO
            else:
                default = ast.literal_eval(value)
            assert default == signature.parameters[keyword.arg].default
        public = {m for m, method in inspect.getmembers(cls, callable) if not m.startswith("_")}
        assert public == METHODS
        for method in public:
            assert re.search(r"\b" + method + r"\b", text), method
        assert set(cls().get_params()) == set(signature.parameters)
    for attr in ("covariance_", "precision_", "location_", "n_samples_", "n_features_",
                 "shrinkage_", "raw_location_", "raw_covariance_", "support_", "dist_",
                 "n_iter_", "alpha_", "cv_results_"):
        assert f"`{attr}`" in text
    for destination in ("estimator-api.md#adjust_pvalues", "estimator-api.md#combine_pvalues",
                        "estimator-api.md#bootstrap_statistic", "estimator-api.md#permutation_test"):
        assert f"]({destination})" in text
    for token in ("float8_e5m2", "bfloat16", "float64", "MPS", "set_params", "NaN",
                  "mean_score", "scores", "n_jobs", "gpu_memory_cleanup"):
        assert token in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_common_overview_keeps_distinct_concepts_and_routes_to_complete_layers(language):
    text = _read(language, "models/covariance")
    for anchor in CONCEPTS:
        assert text.count(f'<a id="{anchor}"></a>') == 1
        tail = text.split(f'<a id="{anchor}"></a>', 1)[1]
        heading = re.search(r"\n## [^\n]+\n", tail)
        assert heading
        body = tail[heading.end():].split("\n## ", 1)[0]
        assert len(re.sub(r"<[^>]+>", "", body).strip()) > 100
    for page in LESSONS:
        assert f"]({page}.md)" in text
        tutorial = _read(language, f"models/{page}")
        assert "](covariance.md" in tutorial
        assert "](../reference/covariance-api.md" in tutorial
        assert "$$" in tutorial
    assert "](../reference/covariance-api.md)" in text
    reference = _read(language, "reference/covariance-api")
    for page in ("covariance", *LESSONS):
        assert f"](../models/{page}.md)" in reference
    for legacy in ("covariance-basic", "covariance-robust", "covariance-sparse", "cpu-and-gpu-examples"):
        assert f'<a id="{legacy}"></a>' in text


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ENTRY_POINTS)
def test_entry_points_expose_common_concepts_all_methods_and_complete_api(language, page):
    targets = re.findall(r"\]\(([^)]+)\)", _read(language, page))
    for destination in ("covariance", *LESSONS, "covariance-api"):
        assert any(target.split("#")[0].endswith(destination + ".md") for target in targets)


def test_root_readme_exposes_all_covariance_layers():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for page in ("covariance", *LESSONS):
        assert f"](docs/en/models/{page}.md)" in text
    assert "](docs/en/reference/covariance-api.md)" in text


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ("models/covariance", *(f"models/{x}" for x in LESSONS),
                                  "reference/covariance-api"))
def test_public_covariance_pages_do_not_expose_internal_review_status(language, page):
    text = _read(language, page)
    for phrase in ("contract closure", "exact-head", "maintained route", "fail closed",
                   "review-clean", "维护中的路径", "证据闭环", "PR #168"):
        assert phrase not in text
