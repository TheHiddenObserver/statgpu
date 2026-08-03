"""Weighted Adaptive Group Lasso objective/backend contracts for PR #80."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.penalties import AdaptiveGroupLassoPenalty
from statgpu.solvers._utils import _tracking_penalty_value


_LAYOUTS = [
    pytest.param([[3, 0], [2, 1]], id="equal-noncontiguous"),
    pytest.param([[4, 3, 0], [2, 1]], id="unequal-noncontiguous"),
]


def _as_numpy(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _backend_vector(backend_name, values):
    if backend_name == "numpy":
        return "numpy", np.asarray(values, dtype=np.float64)
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device unavailable")
        except Exception:
            pytest.skip("CuPy CUDA runtime unavailable")
        return "cupy", cp.asarray(values, dtype=cp.float64)

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(values, dtype=torch.float64, device="cuda"),
    )


def _expected_value_gradient_prox(coef, groups, weights, alpha, step):
    coef = np.asarray(coef, dtype=np.float64)
    value = 0.0
    gradient = np.zeros_like(coef)
    prox = coef.copy()
    for group, weight in zip(groups, weights):
        idx = np.asarray(sorted(group), dtype=np.int64)
        group_coef = coef[idx]
        norm = np.linalg.norm(group_coef)
        scale = alpha * float(weight) * np.sqrt(idx.size)
        value += scale * norm
        if norm > 1e-15:
            gradient[idx] = scale * group_coef / norm
        prox_scale = max(1.0 - step * scale / max(norm, 1e-300), 0.0)
        prox[idx] = group_coef * prox_scale
    return value, gradient, prox


@pytest.mark.parametrize("groups", _LAYOUTS)
def test_adaptive_group_value_gradient_and_prox_match_weighted_definition(groups):
    p = max(max(group) for group in groups) + 1
    coef = np.linspace(1.2, -0.7, p)
    weights = np.array([0.4, 1.7])
    alpha = 0.23
    step = 0.31
    expected_value, expected_gradient, expected_prox = (
        _expected_value_gradient_prox(coef, groups, weights, alpha, step)
    )

    penalty = AdaptiveGroupLassoPenalty(
        groups=groups,
        alpha=alpha,
        weights=weights,
    )

    assert penalty.value(coef) == pytest.approx(expected_value, rel=0.0, abs=1e-14)
    np.testing.assert_allclose(
        penalty.gradient(coef), expected_gradient, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        penalty.proximal(coef, step, backend="numpy"),
        expected_prox,
        rtol=0.0,
        atol=1e-14,
    )
    assert _tracking_penalty_value(penalty, coef) == pytest.approx(
        expected_value, rel=0.0, abs=1e-14
    )


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
@pytest.mark.parametrize("groups", _LAYOUTS)
def test_adaptive_group_cpu_then_gpu_reuse_does_not_poison_weight_cache(
    backend_name,
    groups,
):
    p = max(max(group) for group in groups) + 1
    coef = np.linspace(1.2, -0.7, p)
    weights = np.array([0.4, 1.7])
    alpha = 0.23
    step = 0.31
    expected_value, expected_gradient, expected_prox = (
        _expected_value_gradient_prox(coef, groups, weights, alpha, step)
    )
    penalty = AdaptiveGroupLassoPenalty(
        groups=groups,
        alpha=alpha,
        weights=weights,
    )

    # This CPU evaluation previously populated the field also used as the
    # CuPy cache, allowing a NumPy array to leak into a later CUDA operation.
    assert penalty.value(coef) == pytest.approx(expected_value)
    assert penalty._group_weights_cupy is None

    backend, coef_backend = _backend_vector(backend_name, coef)
    actual_value = penalty.value(coef_backend)
    actual_gradient = penalty.gradient(coef_backend)
    actual_prox = penalty.proximal(coef_backend, step, backend=backend)

    assert actual_value == pytest.approx(expected_value, rel=2e-12, abs=2e-12)
    np.testing.assert_allclose(
        _as_numpy(actual_gradient), expected_gradient, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        _as_numpy(actual_prox), expected_prox, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize(
    "weights, error_type, match",
    [
        ([1.0], ValueError, "shape"),
        ([1.0, 2.0, 3.0], ValueError, "shape"),
        ([1.0, np.nan], ValueError, "finite"),
        ([1.0, np.inf], ValueError, "finite"),
        ([1.0, -0.1], ValueError, "non-negative"),
        (["bad", 1.0], TypeError, "numeric"),
    ],
)
def test_adaptive_group_weights_fail_before_numerical_use(
    weights,
    error_type,
    match,
):
    with pytest.raises(error_type, match=match):
        AdaptiveGroupLassoPenalty(
            groups=[[0, 3], [2, 1]],
            alpha=0.2,
            weights=weights,
        )


def test_adaptive_group_set_weights_invalidates_backend_caches():
    penalty = AdaptiveGroupLassoPenalty(
        groups=[[0, 3], [2, 1]],
        alpha=0.2,
        weights=np.array([1.0, 1.0]),
    )
    penalty._group_weights_torch = object()
    penalty._group_weights_cupy = object()

    replacement = np.array([0.5, 1.5])
    penalty.set_weights(replacement)

    assert penalty._group_weights is replacement
    assert penalty._group_weights_torch is None
    assert penalty._group_weights_cupy is None
