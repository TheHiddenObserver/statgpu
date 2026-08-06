from __future__ import annotations

import numpy as np
import pytest


def test_binomial_probit_irls_matches_direct_bernoulli_optimum():
    scipy = pytest.importorskip("scipy")
    from scipy.optimize import minimize
    from scipy.special import ndtr

    from statgpu.glm_core._family import Binomial, ProbitLink
    from statgpu.glm_core._irls import IRLSSolver

    X = np.column_stack([np.ones(10), np.linspace(-1.8, 1.8, 10)])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=float)
    family = Binomial(link=ProbitLink())
    params, _ = IRLSSolver(family, max_iter=100, tol=1e-9).fit(
        X, y, backend="numpy"
    )

    def objective(beta):
        mu = np.clip(ndtr(X @ beta), 1e-10, 1 - 1e-10)
        return float(np.sum(-y * np.log(mu) - (1 - y) * np.log(1 - mu)))

    reference = minimize(objective, np.zeros(X.shape[1]), method="BFGS")
    assert reference.success
    np.testing.assert_allclose(params, reference.x, rtol=2e-4, atol=2e-4)
    assert objective(params) <= objective(np.zeros(X.shape[1]))


def test_direct_logistic_rejects_soft_and_out_of_range_labels():
    from statgpu.linear_model import LogisticRegression

    X = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError, match="binary y"):
        LogisticRegression(device="cpu").fit(
            X, np.array([0.0, 1.0, 0.5, 0.0, 1.0, 0.0])
        )
    with pytest.raises(ValueError, match="binary y"):
        LogisticRegression(device="cpu").fit(
            X, np.array([0.0, 1.0, 2.0, 0.0, 1.0, 0.0])
        )


def test_irls_numpy_warm_start_is_normalized_to_torch_backend():
    torch = pytest.importorskip("torch")
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    params, _ = IRLSSolver(Gaussian(), max_iter=5, tol=1e-12).fit(
        X, y, init_coef=np.zeros(2), backend="torch"
    )
    assert torch.is_tensor(params)
    assert params.device == X.device
    assert params.dtype == X.dtype
    np.testing.assert_allclose(params.detach().cpu().numpy(), [0.0, 1.0], atol=1e-10)


def test_irls_rejects_wrong_length_warm_start():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="init_coef"):
        IRLSSolver(Gaussian()).fit(
            np.ones((4, 2)), np.arange(4.0), init_coef=np.zeros(3)
        )


@pytest.mark.parametrize(
    "exc",
    [
        IndexError("index 5 is out of bounds for axis 0 with size 5"),
        TypeError("bad call signature"),
        AttributeError("missing coefficient state"),
        KeyError("scores"),
        AssertionError("unexpected path state"),
    ],
)
def test_cv_candidate_programming_errors_are_never_converted_to_nan(exc):
    from statgpu.linear_model.penalized._penalized_cv import (
        _raise_unless_recoverable_cv_candidate_failure,
    )

    with pytest.raises(type(exc)):
        _raise_unless_recoverable_cv_candidate_failure(exc)


def test_cv_candidate_numeric_failure_remains_explicitly_recoverable():
    from statgpu.linear_model.penalized._penalized_cv import (
        _raise_unless_recoverable_cv_candidate_failure,
    )

    _raise_unless_recoverable_cv_candidate_failure(
        np.linalg.LinAlgError("singular matrix")
    )
    _raise_unless_recoverable_cv_candidate_failure(
        FloatingPointError("non-finite iterate")
    )


def test_integer_weight_sum_uses_float64_accumulator():
    from statgpu.glm_core._validation import _safe_weight_sum

    weights = np.full(5, 2**62, dtype=np.int64)
    expected = float(5 * (2**62))
    assert _safe_weight_sum(weights) == pytest.approx(expected, rel=1e-15)


def test_reduce_overhead_repeated_cuda_calls_are_correct_or_visible_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a physical Torch CUDA backend")
    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("torch.compile requires CUDA capability >= 7")

    from statgpu.backends._torch_compile import compile_torch

    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "reduce-overhead")

    def update(x):
        return x.square() + 1.0

    guarded = compile_torch(update, workload="iterative")
    x = torch.linspace(-2.0, 2.0, 128, device="cuda", dtype=torch.float64)
    for _ in range(8):
        result = guarded(x)
        torch.cuda.synchronize()
        assert torch.allclose(result, update(x))
    assert guarded.__statgpu_compile_status__ in {
        "compiled", "runtime-fallback"
    }


def test_integral_glm_weights_are_promoted_before_downstream_normalization():
    from statgpu.glm_core._validation import validate_glm_sample_weight

    raw = np.full(5, 2**62, dtype=np.int64)
    validated = validate_glm_sample_weight(raw, raw.size)
    assert validated.dtype == np.float64
    assert np.isfinite(validated.sum())
    assert validated.sum() == pytest.approx(float(5 * 2**62), rel=1e-15)


def test_integral_solver_weights_are_promoted_before_uniform_checks():
    from statgpu.solvers._utils import _validated_sample_weight

    raw = np.full(5, 2**62, dtype=np.int64)
    backend, _, validated = _validated_sample_weight(raw, raw.size)
    assert backend == "numpy"
    assert validated.dtype == np.float64
    assert np.isfinite(validated.sum())


def test_weighted_glm_objective_with_integral_weights_stays_finite():
    from statgpu.glm_core._logistic import LogisticLoss
    from statgpu.glm_core._validation import validate_glm_sample_weight

    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    coef = np.array([-1.0, 0.5])
    weights = validate_glm_sample_weight(
        np.full(4, 2**62, dtype=np.int64), 4
    )
    value, gradient = LogisticLoss().fused_value_and_gradient(
        X, y, coef, sample_weight=weights
    )
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()


def test_torch_tensor_warm_start_does_not_use_copy_constructor_warning():
    import warnings

    torch = pytest.importorskip("torch")
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    init = torch.zeros(2, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        params, _ = IRLSSolver(Gaussian(), max_iter=5, tol=1e-12).fit(
            X, y, init_coef=init, backend="torch"
        )
    assert not any("copy construct from a tensor" in str(w.message) for w in captured)
    assert params.dtype == torch.float64
