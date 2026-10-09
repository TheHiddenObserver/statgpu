"""Execute named EN/CN smoothing examples and check independent references."""

import re
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from doc_examples import parse_examples, run_example
from scipy.interpolate import BSpline
from scipy.stats import gaussian_kde

from statgpu.nonparametric import (
    KernelDensityEstimator,
    KernelRegressionRegressor,
    fit_kde,
    fit_kernel_regression,
    kde_bootstrap_confidence_interval,
    kde_confidence_interval,
)
from statgpu.semiparametric import GAM

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "semiparametric": {"gam-cpu", "gam-gpu"},
    "nonparametric": {
        "kde-cpu", "kernel-regression-cpu", "kde-bootstrap-cpu", "kde-gpu",
    },
}


def _blocks(language, page):
    text = (ROOT / f"docs/{language}/models/{page}.md").read_text(encoding="utf-8")
    examples = parse_examples(text, f"{language}/{page}.md")
    assert set(examples) == EXAMPLES[page]
    assert sum(len(example.blocks) for example in examples.values()) == len(
        re.findall(r"```python\n", text)
    ), "unnamed example"
    return examples


@pytest.fixture(scope="module", params=("en", "cn"))
def examples(request):
    """Run CPU tutorials with only their explicitly declared prerequisites."""
    namespaces = {}
    for page in EXAMPLES:
        text = (ROOT / f"docs/{request.param}/models/{page}.md").read_text(encoding="utf-8")
        for name, example in _blocks(request.param, page).items():
            for block in example.blocks:
                compile(block, f"docs/{request.param}/models/{page}.md:{name}", "exec")
            if name.endswith("-cpu"):
                namespaces[name] = run_example(text, name, f"{request.param}/{page}.md")
    return namespaces


def _gam_design(model, training, queries):
    """Independent SciPy basis construction, matching training centering."""
    blocks = []
    for j, knots in enumerate(model.knots_):
        knot_vector = np.r_[
            np.repeat(training[:, j].min(), model.degree + 1),
            knots,
            np.repeat(training[:, j].max(), model.degree + 1),
        ]
        n_basis = len(knot_vector) - model.degree - 1
        basis = BSpline(knot_vector, np.eye(n_basis), model.degree, extrapolate=False)
        train_basis = basis(training[:, j])
        query_basis = basis(queries[:, j])
        blocks.append(query_basis - train_basis.mean(axis=0))
    return np.column_stack([np.ones(queries.shape[0]), *blocks])


def test_gam_example_predictions_gcv_and_reference_system(examples):
    ns = examples["gam-cpu"]
    model, fixed = ns["gam"], ns["fixed"]
    assert isinstance(ns["prediction"], np.ndarray)
    assert ns["prediction"].shape == (80,)
    assert ns["mse"] < 0.04
    assert ns["mse"] < ns["baseline_mse"] / 10
    np.testing.assert_allclose(fixed.predict(ns["X_test"]), ns["prediction"], atol=1e-12)
    assert fixed.gcv_score_ is None
    assert "gcv_score" not in fixed.summary()
    assert model.summary()["gcv_score"] == model.gcv_score_
    np.testing.assert_allclose(model.intercept_, ns["y_train"].mean(), atol=1e-8)

    B = _gam_design(model, ns["X_train"], ns["X_train"])
    B_test = _gam_design(model, ns["X_train"], ns["X_test"])
    penalty = np.zeros((B.shape[1], B.shape[1]))
    offset = 1
    for knots in model.knots_:
        size = len(knots) + model.degree + 1
        difference = np.diff(np.eye(size), n=model.penalty_order, axis=0)
        penalty[offset:offset + size, offset:offset + size] = difference.T @ difference
        offset += size
    gram = B.T @ B
    system = gram + model.lam_ * penalty
    # The documented solver adds this diagonal stabilization before Cholesky.
    system += np.eye(len(system)) * (1e-10 * np.trace(system) / len(system))
    coefficients = np.linalg.solve(system, B.T @ ns["y_train"])
    reference_prediction = B_test @ coefficients
    np.testing.assert_allclose(ns["prediction"], reference_prediction, atol=2e-8)
    reference_edf = np.trace(np.linalg.solve(system, gram))
    np.testing.assert_allclose(model.edf_, reference_edf, atol=2e-7)
    n = len(ns["y_train"])
    rss = np.sum((ns["y_train"] - B @ coefficients) ** 2)
    reference_gcv = n * rss / (n - model.gamma * reference_edf) ** 2
    np.testing.assert_allclose(model.gcv_score_, reference_gcv, rtol=1e-7)
    assert np.any(np.isclose(model.lam_, np.logspace(-10, 10, 100), rtol=1e-12, atol=0))


def test_kde_example_scipy_reference_density_and_scores(examples):
    ns = examples["kde-cpu"]
    model = ns["kde"]
    reference = gaussian_kde(ns["x_train"], bw_method=0.35)
    np.testing.assert_allclose(ns["density"], reference(ns["grid"]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ns["one_shot"], ns["density"], atol=1e-14)
    np.testing.assert_allclose(ns["log_density"], np.log(ns["density"]), atol=1e-12)
    np.testing.assert_allclose(model.score(ns["x_test"]), reference.logpdf(ns["x_test"]).mean(), atol=1e-10)
    assert ns["density"].shape == (201,)
    assert abs(ns["mass_on_grid"] - 1) < 1e-4
    assert model.to_numpy_metadata()["bandwidth_factor"] == 0.35


def test_kernel_regression_example_local_weighted_ls_reference(examples):
    ns = examples["kernel-regression-cpu"]
    model = ns["regressor"]
    assert ns["prediction"].shape == (61,)
    assert ns["mse"] < 0.025
    assert ns["mse"] < ns["baseline_mse"] / 10
    assert model.score(ns["x_test"], ns["y_test"]) > 0.95
    assert np.mean((ns["prediction"] - ns["mean_test"]) ** 2) < 0.005
    np.testing.assert_allclose(ns["prediction"], ns["one_shot"], atol=1e-14)
    reference = []
    for query in ns["x_test"]:
        distance = ns["x_train"] - query
        design = np.column_stack([np.ones(distance.size), distance])
        weight = np.exp(-0.5 * (distance / 0.18) ** 2)
        root_weight = np.sqrt(weight)
        coef = np.linalg.lstsq(design * root_weight[:, None], ns["y_train"] * root_weight, rcond=None)[0]
        reference.append(coef[0])
    np.testing.assert_allclose(ns["prediction"], reference, atol=1e-12)
    np.testing.assert_allclose(model.covariance_, [[0.18 ** 2]], atol=1e-14)


def test_bootstrap_example_quantiles_determinism_and_scipy_replicates(examples):
    ns = examples["kde-bootstrap-cpu"]
    result = ns["ci"]
    assert result.estimate.shape == result.lower.shape == result.upper.shape == (3,)
    assert result.bootstrap_samples.shape == (100, 3)
    assert np.all(np.isfinite(result.bootstrap_samples))
    assert np.all(result.lower <= result.upper)
    np.testing.assert_allclose(result.lower, np.quantile(result.bootstrap_samples, 0.025, axis=0))
    np.testing.assert_allclose(result.upper, np.quantile(result.bootstrap_samples, 0.975, axis=0))
    np.testing.assert_allclose(result.estimate, gaussian_kde(ns["samples"], bw_method=0.4)(ns["points"]), atol=1e-10)
    replay = kde_bootstrap_confidence_interval(
        ns["samples"], ns["points"], bandwidth=0.4, backend="numpy",
        n_resamples=100, random_state=17, return_bootstrap_samples=True,
    )
    np.testing.assert_array_equal(result.bootstrap_samples, replay.bootstrap_samples)
    rng = np.random.default_rng(17)
    for i in range(3):
        indices = rng.choice(120, size=120, replace=True, p=np.full(120, 1 / 120))
        reference = gaussian_kde(ns["samples"][indices], bw_method=0.4)(ns["points"])
        np.testing.assert_allclose(result.bootstrap_samples[i], reference, atol=1e-10)
    assert result.metadata["method"] == "bootstrap"
    assert result.to_dict()["n_resamples"] == 100


@pytest.mark.parametrize("language", ("en", "cn"))
def test_gam_constructor_inventory_matches_current_signature(language):
    text = (ROOT / f"docs/{language}/models/semiparametric.md").read_text(encoding="utf-8")
    for parameter in signature(GAM).parameters:
        assert f"| `{parameter}` |" in text


@pytest.mark.parametrize("kwargs", [
    {"n_splines": 4}, {"degree": -1}, {"penalty_order": 20},
    {"knot_method": "unknown"}, {"lam": -1}, {"gamma": 0},
])
def test_gam_invalid_design_controls_fail(kwargs):
    x = np.linspace(-1, 1, 40)
    with pytest.raises(ValueError):
        GAM(device="cpu", **kwargs).fit(x, np.sin(x))


def test_gam_input_output_and_unsupported_likelihood_boundaries():
    x = np.linspace(-1, 1, 40)
    with pytest.raises(ValueError, match="constant"):
        GAM(device="cpu").fit(np.column_stack([x, np.ones(40)]), x)
    with pytest.raises(ValueError, match="same number"):
        GAM(device="cpu").fit(x, x[:-1])
    with pytest.raises(ValueError):
        GAM(device="cpu").fit(x, np.full(40, np.nan))
    with pytest.raises(TypeError):
        GAM(family="poisson", device="cpu")
    model = GAM(lam=1, device="cpu").fit(x, x[:, None])
    assert model.predict(np.array([0, 0.5])).shape == (2,)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        model.predict([np.inf])
    with pytest.raises(ValueError):
        GAM(device="cpu").fit(np.empty((0, 1)), np.empty(0))


def test_kernel_shapes_factor_widths_and_analytic_nw():
    x = np.linspace(-2, 2, 30)
    targets = np.column_stack([x, x ** 2])
    model = fit_kernel_regression(x, targets, bandwidth=0.5, backend="numpy")
    query = np.array([-0.5, 0, 0.5])
    pred = model.predict(query)
    weights = np.exp(-0.5 * (query[:, None] - x[None, :]) ** 2 / model.covariance_[0, 0])
    reference = (weights @ targets) / weights.sum(axis=1)[:, None]
    np.testing.assert_allclose(pred, reference, atol=1e-12)
    assert pred.shape == (3, 2)
    single_column = fit_kernel_regression(x, x[:, None], backend="numpy")
    assert single_column.predict(query).shape == (3, 1)
    np.testing.assert_allclose(model.covariance_[0, 0], 0.5 ** 2 * np.var(x, ddof=1), atol=1e-11)
    multivariate = fit_kde(targets, bandwidth=0.5, backend="numpy")
    assert multivariate.pdf(np.array([0.0, 0.1])).shape == (1,)
    with pytest.raises(ValueError, match="dimension"):
        multivariate.pdf(np.zeros((2, 3)))
    absolute = fit_kernel_regression(
        targets, x, bandwidth_per_feature=[0.2, 0.4],
        kernel_metric="diagonal", backend="numpy",
    )
    np.testing.assert_allclose(absolute.covariance_, np.diag([0.2 ** 2, 0.4 ** 2]), atol=1e-14)


def test_compact_support_zero_density_and_regression_fallback():
    x = np.linspace(-1, 1, 30)
    kde = fit_kde(x, bandwidth=0.2, kernel="epanechnikov", backend="numpy")
    np.testing.assert_array_equal(kde.pdf([100]), [0])
    assert np.isneginf(kde.logpdf([100])[0])
    for regression in ("nw", "local_linear"):
        model = fit_kernel_regression(x, x ** 2, bandwidth=0.2, kernel="epanechnikov", regression=regression, backend="numpy")
        np.testing.assert_allclose(model.predict([100]), [(x ** 2).mean()])


@pytest.mark.parametrize("kwargs", [
    {"bandwidth": 0}, {"kernel": "unknown"}, {"bandwidth": "cv-ll"},
    {"weights": [-1, 1, 1]}, {"weights": [0, 0, 0]}, {"weights": [1, 0, 0]},
    {"backend": "invalid"},
])
def test_kde_failure_boundaries(kwargs):
    options = {"backend": "numpy", **kwargs}
    with pytest.raises(ValueError):
        fit_kde([-1, 0, 1], **options)


def test_kernel_and_interval_unsupported_combinations():
    x = np.linspace(-1, 1, 30)
    with pytest.raises(RuntimeError):
        KernelDensityEstimator(backend="numpy").pdf([0])
    with pytest.raises(RuntimeError):
        KernelRegressionRegressor(backend="numpy").predict([0])
    with pytest.raises(ValueError):
        fit_kde([0], backend="numpy")
    with pytest.raises(ValueError):
        fit_kde([0, np.nan], backend="numpy")
    with pytest.raises(ValueError, match="1D"):
        fit_kde(np.column_stack([x, x ** 2]), kernel="cosine", backend="numpy")
    with pytest.raises(ValueError, match="diagonal"):
        fit_kernel_regression(x, x, bandwidth_per_feature=[0.2], backend="numpy")
    with pytest.raises(ValueError, match="positive"):
        fit_kernel_regression(x, x, kernel_metric="diagonal", bandwidth_per_feature=[0], backend="numpy")
    with pytest.raises(ValueError):
        fit_kernel_regression(x, x[:-1], backend="numpy")
    with pytest.raises(ValueError):
        fit_kernel_regression(x, np.full(30, np.nan), backend="numpy")
    with pytest.raises(ValueError, match="percentile"):
        kde_bootstrap_confidence_interval(x, [0], method="bca", backend="numpy")
    with pytest.raises(ValueError, match="1D Gaussian"):
        kde_confidence_interval(x, [0], kernel="epanechnikov", backend="numpy")
    with pytest.raises(ValueError, match="confidence_level"):
        kde_bootstrap_confidence_interval(x, [0], confidence_level=1, backend="numpy")
    with pytest.raises(ValueError, match="batch_size"):
        fit_kde(x, backend="numpy").pdf([0], batch_size=0)
