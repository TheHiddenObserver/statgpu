"""Seventh-review feature-selection RNG and sampler documentation boundaries.

Torch checks run on CPU and do not provide physical CUDA evidence. A narrow
strict xfail records the intended construction RNG contract until repaired.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from statgpu.feature_selection import (
    KnockoffSelector,
    fixed_x_knockoff_filter,
    knockoff_filter,
    model_x_knockoff_filter,
)

ROOT = Path(__file__).resolve().parents[2]


class _IgnoredModelXTorchSeed(AssertionError):
    """The seeded model-X construction draws from the global Torch RNG."""


def _problem():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(80, 5))
    return X, X[:, 0] + rng.normal(size=80)


def _modelx_result(entrypoint, X, y, **kwargs):
    if entrypoint == "function":
        return model_x_knockoff_filter(X, y, **kwargs)
    if entrypoint == "unified":
        return knockoff_filter(X, y, knockoff_type="model_x", **kwargs)
    return KnockoffSelector(knockoff_type="model_x", **kwargs).fit(X, y).result_


@pytest.mark.parametrize("entrypoint", ["function", "unified", "selector"])
def test_documented_numpy_modelx_seed_option_is_repeatable(entrypoint):
    X, y = _problem()
    opts = {"backend": "numpy", "modelx_draws": 3, "method": "corr_diff"}
    first = _modelx_result(entrypoint, X, y, random_state=123, **opts)
    second = _modelx_result(entrypoint, X, y, random_state=123, **opts)
    changed = _modelx_result(entrypoint, X, y, random_state=456, **opts)
    np.testing.assert_array_equal(first.W, second.W)
    np.testing.assert_array_equal(first.selected_features, second.selected_features)
    assert np.isfinite(first.W).all()
    assert not np.allclose(first.W, changed.W)


@pytest.fixture
def torch_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from statgpu.feature_selection import _knockoff_utils

    # Select CPU allocation only. The RNG source/seed/draw implementation is
    # unmodified; the same test can run on a host that also has CUDA available.
    monkeypatch.setattr(_knockoff_utils, "_get_torch_device_str", lambda: "cpu")
    with torch.random.fork_rng(devices=[]):
        yield torch


def _reference_modelx_w(torch, draws, *, global_generator=None, seed=None):
    """Independent NumPy construction algebra with explicitly chosen Torch draws."""
    X, y = _problem()
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    n, p = X.shape
    sigma = X.T @ X / (n - 1)
    sigma = (sigma + sigma.T) / 2
    sigma = .8 * sigma + .2 * np.trace(sigma) / p * np.eye(p)
    min_eigenvalue = np.linalg.eigvalsh(sigma).min()
    assert min_eigenvalue > 1e-6  # This fixture requires no ridge stabilization.
    S = min(2 * min_eigenvalue * .999, 1.) * np.eye(p)
    inverse_sigma_S = np.linalg.solve(sigma, S)
    residual_covariance = 2 * S - S @ inverse_sigma_S
    eigenvalues, eigenvectors = np.linalg.eigh((residual_covariance + residual_covariance.T) / 2)
    C = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    response = y - y.mean()
    original_score = np.abs(X.T @ response)
    values = []
    for draw in range(draws):
        generator = global_generator
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(seed + 104729 * draw)
        Z = torch.randn(n, p, dtype=torch.float64, device="cpu", generator=generator).numpy()
        Xk = X - X @ inverse_sigma_S + Z @ C
        values.append(original_score - np.abs(Xk.T @ response))
    return np.mean(values, axis=0)


def _assert_untied_selection(result):
    """Check the full knockoff+ rule on this fixture's distinct nonzero W."""
    absolute = np.abs(result.W)
    assert np.all(absolute > 0) and len(np.unique(absolute)) == len(absolute)
    threshold, ratio = np.inf, 0.
    for candidate in np.sort(absolute):
        candidate_ratio = ((1 + np.count_nonzero(result.W <= -candidate))
                           / max(1, np.count_nonzero(result.W >= candidate)))
        if candidate_ratio <= result.q:
            threshold, ratio = candidate, candidate_ratio
            break
    expected = np.flatnonzero(result.W >= threshold)
    np.testing.assert_array_equal(result.selected_features, expected)
    assert result.threshold == pytest.approx(threshold)
    assert result.estimated_fdr == pytest.approx(ratio)


def _assert_torch_modelx_local_seed(torch, entrypoint, draws):
    X, y = (torch.as_tensor(a) for a in _problem())
    opts = {"backend": "torch", "modelx_draws": draws, "method": "corr_diff", "q": .4}
    torch.random.default_generator.manual_seed(888)
    global_before = torch.get_rng_state().clone()
    first = _modelx_result(entrypoint, X, y, random_state=123, **opts)
    global_after = torch.get_rng_state().clone()
    second = _modelx_result(entrypoint, X, y, random_state=123, **opts)
    torch.random.default_generator.manual_seed(888)
    changed = _modelx_result(entrypoint, X, y, random_state=456, **opts)
    assert first.W.shape == second.W.shape == changed.W.shape == (5,)
    assert all(np.isfinite(result.W).all() for result in (first, second, changed))
    for result in (first, second, changed):
        _assert_untied_selection(result)

    known_joint_symptom = (
        not np.array_equal(first.W, second.W)
        and np.array_equal(first.W, changed.W)
        and not torch.equal(global_before, global_after)
    )
    if known_joint_symptom:
        generator = torch.Generator(device="cpu").manual_seed(888)
        expected_first = _reference_modelx_w(torch, draws, global_generator=generator)
        expected_state = generator.get_state().clone()
        expected_second = _reference_modelx_w(torch, draws, global_generator=generator)
        # A different finite-output corruption, draw shape/count, or matrix
        # computation must fail normally before the narrowly identified xfail.
        np.testing.assert_allclose(first.W, expected_first, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(second.W, expected_second, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(changed.W, expected_first, atol=1e-10, rtol=1e-12)
        assert torch.equal(global_after, expected_state)
        raise _IgnoredModelXTorchSeed("construction exactly followed the global RNG instead of random_state")

    np.testing.assert_array_equal(first.W, second.W)
    np.testing.assert_array_equal(first.selected_features, second.selected_features)
    assert not np.array_equal(first.W, changed.W)
    assert torch.equal(global_before, global_after)
    np.testing.assert_allclose(first.W, _reference_modelx_w(torch, draws, seed=123),
                               atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(changed.W, _reference_modelx_w(torch, draws, seed=456),
                               atol=1e-10, rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    raises=_IgnoredModelXTorchSeed,
    reason="Issue #242: native Torch model-X construction ignores random_state",
)
@pytest.mark.parametrize("entrypoint", ["function", "unified", "selector"])
@pytest.mark.parametrize("draws", [1, 3])
def test_torch_modelx_construction_should_use_its_local_seed(torch_cpu, entrypoint, draws):
    _assert_torch_modelx_local_seed(torch_cpu, entrypoint, draws)


@pytest.mark.parametrize("fault", ["runtime_error", "finite_W_corruption", "wrong_selection"])
def test_seed_probe_does_not_classify_unrelated_failures_as_known_bug(torch_cpu, monkeypatch, fault):
    import sys

    original = _modelx_result

    def broken(*args, **kwargs):
        if fault == "runtime_error":
            raise RuntimeError("unrelated construction failure")
        result = original(*args, **kwargs)
        if fault == "finite_W_corruption":
            result.W = result.W + 1.
        else:
            result.selected_features = np.array([4, 3, 2, 1, 0])
        return result

    monkeypatch.setattr(sys.modules[__name__], "_modelx_result", broken)
    expected_exception = RuntimeError if fault == "runtime_error" else AssertionError
    with pytest.raises(expected_exception) as error:
        _assert_torch_modelx_local_seed(torch_cpu, "function", 1)
    assert not isinstance(error.value, _IgnoredModelXTorchSeed)


def test_torch_fixedx_and_supplied_modelx_do_not_use_the_faulty_construction(torch_cpu):
    torch = torch_cpu
    X, y = (torch.as_tensor(a) for a in _problem())
    opts = {"backend": "torch", "random_state": 123, "method": "corr_diff"}
    fixed = [fixed_x_knockoff_filter(X, y, **opts) for _ in range(2)]
    np.testing.assert_array_equal(fixed[0].W, fixed[1].W)
    # A centered orthogonal supplied pair avoids automatic construction. It
    # tests only repeatability, not nominal FDR control for arbitrary data.
    rng = np.random.default_rng(22)
    basis, _ = np.linalg.qr(np.column_stack([np.ones(80), rng.normal(size=(80, 10))]))
    supplied_X = torch.as_tensor(basis[:, 1:6].copy())
    supplied_Xk = torch.as_tensor(basis[:, 6:11].copy())
    supplied_y = 3 * supplied_X[:, 0]
    supplied = [model_x_knockoff_filter(supplied_X, supplied_y, Xk=supplied_Xk, **opts)
                for _ in range(2)]
    np.testing.assert_array_equal(supplied[0].W, supplied[1].W)
    assert supplied[0].metadata["xk_source"] == "provided"


@pytest.mark.parametrize("language", ["en", "cn"])
def test_seed_warning_names_affected_route_and_reproducible_alternative(language):
    model = (ROOT / f"docs/{language}/models/knockoff.md").read_text()
    reference = (ROOT / f"docs/{language}/reference/feature-selection-api.md").read_text()
    for text in (model, reference):
        assert 'compat_mode="statgpu"' in text
        assert "random_state" in text and "Xk" in text
        assert ("global" in text if language == "en" else "全局" in text)
        assert "NumPy" in text and "Fixed-X" in text
    assert "reproducibility-of-generated-torch-model-x" in reference
    for consumer in (model_x_knockoff_filter, knockoff_filter, KnockoffSelector):
        doc = inspect.getdoc(consumer)
        assert "global Torch RNG" in doc and "random_state" in doc


def test_sampler_dispatch_boundaries_match_the_bilingual_reference():
    X, y = _problem()
    for options in (
        {"compat_mode": "statgpu"},
        {"compat_mode": "knockpy", "Xk": X.copy()},
    ):
        opts = dict(backend="numpy", random_state=123, modelx_draws=1, **options)
        requested = model_x_knockoff_filter(X, y, knockpy_sampler="gaussian", **opts)
        ordinary = model_x_knockoff_filter(X, y, **opts)
        np.testing.assert_array_equal(requested.W, ordinary.W)
    with pytest.raises(NotImplementedError):
        model_x_knockoff_filter(X, y, compat_mode="knockpy", backend="numpy",
                               knockpy_sampler="gaussian", random_state=123)
    for language in ("en", "cn"):
        reference = (ROOT / f"docs/{language}/reference/feature-selection-api.md").read_text()
        assert ('ignored otherwise' in reference if language == 'en' else '其他路径忽略' in reference)
        assert 'compat_mode="knockpy"' in reference
