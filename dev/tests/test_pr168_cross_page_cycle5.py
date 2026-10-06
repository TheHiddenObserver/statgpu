"""Execute corrected entry-point docs and guard shared help contracts.

CPU execution only. GPU example inspection is static and makes no accelerator
execution claim. Numerical production behavior is not changed by these tests.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu import LinearRegression
from statgpu.inference import bootstrap_statistic, combine_pvalues

ROOT = Path(__file__).resolve().parents[2]


def _marked_example(path, marker):
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(
        r"<!-- api-example: " + re.escape(marker) + r" -->\s*```python\n(.*?)```",
        text, re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _execute(source, filename):
    namespace = {"__name__": "__doc_example__"}
    state = np.random.get_state()
    try:
        exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    finally:
        np.random.set_state(state)
    return namespace


def test_readme_quick_start_executes_from_an_empty_namespace():
    scope = _execute(_marked_example("README.md", "readme-quick-start"), "README.md")
    assert scope["X"].shape == (300, 8)
    assert scope["model"].score(scope["X"], scope["y"]) > 0.9
    assert scope["cv_model"].alpha_ in scope["cv_model"].alpha_grid
    predicted = scope["cv_model"].predict(scope["X"][:, :5])
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    np.testing.assert_array_equal(scope["reject"], [True, True, False])
    assert np.isfinite(scope["p_global"]) and 0 <= scope["p_global"] <= 1


@pytest.mark.parametrize("language", ["en", "cn"])
def test_inventory_poisson_example_executes_and_interprets_counts(language):
    path = f"docs/{language}/guides/implemented-methods.md"
    scope = _execute(_marked_example(path, "inventory-poisson"), path)
    model = scope["model"]
    np.testing.assert_allclose(
        model.predict(scope["X"][:3]), [1.45875781, 1.54732825, .77567214],
        atol=1e-7, rtol=1e-7,
    )
    assert model.get_params()["compute_inference"] is False
    assert model.get_params()["device"] == "cpu"


@pytest.mark.parametrize("language", ["en", "cn"])
def test_gpu_cleanup_example_has_inputs_without_claiming_execution(language):
    path = f"docs/{language}/guides/device-and-memory.md"
    source = _marked_example(path, "cuda-cleanup")
    tree = ast.parse(source)
    assigned = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert {"X", "y", "rng", "model"} <= assigned
    assert 'device="cuda"' in source and "gpu_memory_cleanup=True" in source
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "CuPy/CUDA" in text


def test_parameter_help_and_lifecycle_match_runtime():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 2))
    y = .2 + x @ np.array([.4, -.5])
    model = LinearRegression(device="cpu", compute_inference=False).fit(x, y)
    before = model.predict(x).copy()
    assert model.set_params() is model
    np.testing.assert_array_equal(model.predict(x), before)
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(unknown_setting=1)
    np.testing.assert_array_equal(model.predict(x), before)
    assert model.set_params(fit_intercept=False) is model
    with pytest.raises((RuntimeError, ValueError)):
        model.predict(x)
    assert "coef_" not in model.get_params()
    for method in (model.get_params, model.set_params):
        text = inspect.getdoc(method)
        assert "Parameters" in text and "Returns" in text


def test_estimator_weight_help_explains_shared_raw_sum_overflow():
    text = " ".join(inspect.getdoc(LinearRegression.combine_pvalues).split())
    assert "raw-weight normalization sum" in text
    assert "squared norms can overflow" not in text
    weights = np.array([1e308, 1e308])
    scaled = weights / weights.max()
    model = LinearRegression(device="cpu", compute_inference=False)
    for method in ("cauchy", "stouffer"):
        result = model.combine_pvalues([.01, .1], method=method, weights=scaled)
        expected = combine_pvalues([.01, .1], method=method, weights=[1, 1])
        np.testing.assert_allclose((result["statistic"], result["pvalue"]), expected)


def test_readme_benchmark_claims_are_bound_to_the_cited_historical_reports():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    archive = (ROOT / "dev/references/readme-historical-benchmarks.md").read_text(encoding="utf-8")
    assert "Benchmark Results (RTX 4090)" not in readme
    assert "196.9x" not in readme and "97.9x" not in readme
    assert "Both reports identify Tesla P100" in readme
    assert "not\nmeasurements of the current source" in readme
    assert "678K-by-42" in archive and "NumPy/FISTA" in archive
    assert "196.9x" in archive and "97.9x" in archive
    assert "historical transcription" in archive
    for path in ("results/glm_solver_benchmark_2026-06-23.md", "results/unsupervised_bench_2026-06-27.md"):
        assert path in readme
        report = (ROOT / path).read_text(encoding="utf-8")
        assert "Hardware: Tesla P100-SXM2-16GB" in report


@pytest.mark.parametrize("block_size", [1, 3, 10, 15])
def test_moving_block_contract_matches_independent_index_replay(block_size):
    data = np.arange(10.0)
    result = bootstrap_statistic(
        np.mean, data, strategy="block", block_size=block_size,
        n_resamples=23, random_state=7, backend="numpy",
    )
    b = min(block_size, len(data))
    count = (len(data) + b - 1) // b
    starts = np.random.default_rng(7).integers(0, len(data) - b + 1, size=(23, count))
    rows = (starts[:, :, None] + np.arange(b)).reshape(23, -1)[:, :len(data)]
    np.testing.assert_allclose(result.samples, data[rows].mean(axis=1))
    if block_size >= len(data):
        assert result.confidence_interval == (data.mean(), data.mean())


@pytest.mark.parametrize("language", ["en", "cn"])
def test_block_definition_and_degenerate_interval_warning_are_documented(language):
    page = (ROOT / f"docs/{language}/reference/estimator-api.md").read_text(encoding="utf-8")
    for token in ("ceil(n / b)", "n-b", "block_size >= n"):
        assert token in page
    assert ("do not wrap" in page if language == "en" else "不会从末尾绕回" in page)
    assert ("zero-width interval" in page if language == "en" else "零宽区间" in page)
    for function in (bootstrap_statistic, LinearRegression.bootstrap_statistic):
        help_text = " ".join(inspect.getdoc(function).split())
        assert "without circular wrapping" in help_text
        assert "zero-width interval" in help_text
