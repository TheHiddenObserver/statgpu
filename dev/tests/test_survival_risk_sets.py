"""Mathematical and backend gates for Cox counting-process risk sets."""

import numpy as np
import pytest
from itertools import combinations

from statgpu.survival import _risk_sets as risk_sets_module
from statgpu.survival._risk_sets import (
    cox_baseline_hazard,
    cox_counting_process_objective,
    step_evaluate,
)
from statgpu.survival._cox_counting import fit_counting_process_cox


def test_hand_calculated_breslow_tied_risk_set():
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 1.0, 2.0])
    event = np.array([1, 1, 0])

    result = cox_counting_process_objective(
        np.zeros(1), X, stop, event, ties="breslow", score_residuals=True
    )

    assert np.allclose(result["log_likelihood"], -2.0 * np.log(3.0))
    assert np.allclose(result["score"], [-1.0])
    assert np.allclose(result["information"], [[4.0 / 3.0]])
    assert np.allclose(result["score_residuals"].sum(axis=0), result["score"])


def test_hand_calculated_efron_tied_risk_set():
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 1.0, 2.0])
    event = np.array([1, 1, 0])

    result = cox_counting_process_objective(
        np.zeros(1), X, stop, event, ties="efron", score_residuals=True
    )

    assert np.allclose(result["log_likelihood"], -np.log(3.0) - np.log(2.0))
    assert np.allclose(result["score"], [-1.25])
    assert np.allclose(result["information"], [[2.0 / 3.0 + 11.0 / 16.0]])
    # Robust inference follows statsmodels' conventional Breslow martingale
    # residual even when the likelihood bread uses Efron ties.
    assert np.allclose(result["score_residuals"].ravel(), [-1.0 / 3.0, 0.0, -2.0 / 3.0])


def test_hand_calculated_exact_tied_risk_set():
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 1.0, 2.0])
    event = np.array([1, 1, 0])
    result = cox_counting_process_objective(
        np.zeros(1), X, stop, event, ties="exact", score_residuals=True
    )
    assert np.allclose(result["log_likelihood"], -np.log(3.0))
    assert np.allclose(result["score"], [-1.0])
    assert np.allclose(result["information"], [[2.0 / 3.0]])
    assert np.allclose(result["score_residuals"].sum(axis=0), result["score"])


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
def test_loglik_only_objective_matches_full_objective_without_derivative_outputs(
    ties, monkeypatch
):
    rng = np.random.default_rng(7129)
    n_samples, n_features = 18, 7
    X = rng.normal(size=(n_samples, n_features))
    stop = np.tile(np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    start = np.zeros(n_samples)
    start[stop > 2.0] = 0.5
    event = np.tile(np.array([1, 1, 1, 0, 1, 0]), 3)
    strata = np.repeat(np.arange(3), 6)
    beta = rng.normal(scale=0.15, size=n_features)
    full = cox_counting_process_objective(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
    )

    allocated_shapes = []
    original_zeros = risk_sets_module._zeros

    def recording_zeros(backend, xp, shape, like):
        allocated_shapes.append(tuple(shape))
        return original_zeros(backend, xp, shape, like)

    monkeypatch.setattr(risk_sets_module, "_zeros", recording_zeros)
    loglik_only = cox_counting_process_objective(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
        compute_derivatives=False,
    )

    assert set(loglik_only) == {"log_likelihood"}
    assert np.allclose(
        loglik_only["log_likelihood"],
        full["log_likelihood"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert (n_features, n_features) not in allocated_shapes
    assert not any(len(shape) == 3 for shape in allocated_shapes)


def test_loglik_only_objective_rejects_score_residuals():
    with pytest.raises(ValueError, match="score_residuals requires"):
        cox_counting_process_objective(
            np.zeros(1),
            np.zeros((2, 1)),
            np.array([1.0, 2.0]),
            np.array([1, 0]),
            score_residuals=True,
            compute_derivatives=False,
        )


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
def test_loglik_only_torch_path_matches_full_objective(ties):
    torch = pytest.importorskip("torch")
    X = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6], [0.1, -0.2]],
        dtype=torch.float64,
    )
    stop = torch.tensor([1.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    event = torch.tensor([1, 1, 1, 0], dtype=torch.int64)
    beta = torch.tensor([0.15, -0.2], dtype=torch.float64)
    full = cox_counting_process_objective(beta, X, stop, event, ties=ties)
    loglik_only = cox_counting_process_objective(
        beta, X, stop, event, ties=ties, compute_derivatives=False
    )

    assert set(loglik_only) == {"log_likelihood"}
    assert torch.allclose(
        loglik_only["log_likelihood"],
        full["log_likelihood"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_exact_tie_partition_matches_brute_force():
    X = np.array([[0.2, -0.4], [1.1, 0.3], [-0.7, 0.8], [0.5, -0.2]])
    stop = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1, 1, 0, 0])
    beta = np.array([0.3, -0.15])
    result = cox_counting_process_objective(beta, X, stop, event, ties="exact")
    weights = np.exp(X @ beta)
    denominator = sum(
        np.prod(weights[list(index_set)])
        for index_set in combinations(range(X.shape[0]), 2)
    )
    expected = np.sum(X[:2] @ beta) - np.log(denominator)
    assert np.allclose(result["log_likelihood"], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
def test_failure_local_shift_ignores_extreme_rows_that_left_risk_set(ties):
    X = np.array([[1000.0], [0.0]])
    stop = np.array([1.0, 2.0])
    event = np.array([0, 1])
    result = cox_counting_process_objective(np.array([1.0]), X, stop, event, ties=ties)
    assert np.isfinite(result["log_likelihood"])
    assert np.allclose(result["log_likelihood"], 0.0, atol=0.0)
    assert np.allclose(result["score"], 0.0, atol=0.0)


def test_exact_partition_stays_finite_beyond_float64_combination_range():
    scipy_special = pytest.importorskip("scipy.special")
    n, d = 1100, 550
    X = np.zeros((n, 1), dtype=np.float64)
    stop = np.r_[np.ones(d), np.full(n - d, 2.0)]
    event = np.r_[np.ones(d, dtype=np.int64), np.zeros(n - d, dtype=np.int64)]
    result = cox_counting_process_objective(np.zeros(1), X, stop, event, ties="exact")
    expected = -(
        scipy_special.gammaln(n + 1)
        - scipy_special.gammaln(d + 1)
        - scipy_special.gammaln(n - d + 1)
    )
    assert np.isfinite(result["log_likelihood"])
    assert np.allclose(result["log_likelihood"], expected, rtol=0, atol=2e-11)
    assert np.all(np.isfinite(result["information"]))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("event", np.array([0.5, 1.0]), "event"),
        ("event", np.array([np.nan, 1.0]), "event"),
        ("X", np.array([[np.inf], [0.0]]), "X"),
        ("stop", np.array([1.0, np.nan]), "stop"),
    ],
)
def test_invalid_counting_inputs_are_rejected_before_integer_cast(field, value, match):
    inputs = {
        "X": np.array([[0.0], [1.0]]),
        "stop": np.array([1.0, 2.0]),
        "event": np.array([0.0, 1.0]),
    }
    inputs[field] = value
    with pytest.raises(ValueError, match=match):
        cox_counting_process_objective(
            np.zeros(1), inputs["X"], inputs["stop"], inputs["event"]
        )


def test_counting_process_interval_is_open_left_closed_right():
    X = np.array([[0.0], [1.0], [2.0]])
    start = np.array([0.0, 1.0, 0.0])
    stop = np.array([1.0, 2.0, 2.0])
    event = np.array([1, 0, 0])

    result = cox_counting_process_objective(
        np.zeros(1), X, stop, event, start=start, ties="breslow"
    )

    # At t=1, row 1 has start==t and is not yet at risk. Rows 0 and 2 are.
    assert np.allclose(result["log_likelihood"], -np.log(2.0))
    assert np.allclose(result["score"], [-1.0])


def test_strata_use_independent_risk_sets():
    X = np.array([[0.0], [2.0], [10.0], [14.0]])
    stop = np.array([1.0, 2.0, 1.0, 2.0])
    event = np.array([1, 0, 1, 0])
    strata = np.array([0, 0, 1, 1])

    result = cox_counting_process_objective(
        np.zeros(1), X, stop, event, strata=strata, ties="breslow"
    )

    assert np.allclose(result["log_likelihood"], -2.0 * np.log(2.0))
    assert np.allclose(result["score"], [-3.0])


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
def test_score_and_information_match_finite_differences_with_heavy_ties(ties):
    rng = np.random.default_rng(20260712)
    X = rng.normal(size=(40, 3))
    stop = rng.integers(1, 7, size=40).astype(float)
    event = rng.binomial(1, 0.7, size=40)
    event[0] = 1
    beta = np.array([0.2, -0.15, 0.08])
    eps = 2e-5

    result = cox_counting_process_objective(beta, X, stop, event, ties=ties)
    numeric_score = np.empty_like(beta)
    numeric_hessian = np.empty((beta.size, beta.size))
    for j in range(beta.size):
        step = np.zeros_like(beta)
        step[j] = eps
        plus = cox_counting_process_objective(beta + step, X, stop, event, ties=ties)
        minus = cox_counting_process_objective(beta - step, X, stop, event, ties=ties)
        numeric_score[j] = (plus["log_likelihood"] - minus["log_likelihood"]) / (
            2.0 * eps
        )
        numeric_hessian[:, j] = (plus["score"] - minus["score"]) / (2.0 * eps)

    assert np.allclose(result["score"], numeric_score, rtol=2e-6, atol=2e-6)
    assert np.allclose(result["information"], -numeric_hessian, rtol=2e-5, atol=2e-5)
    assert np.linalg.eigvalsh(result["information"]).min() >= -1e-10


def test_baseline_hazard_respects_entry_and_step_evaluation():
    X = np.array([[0.0], [1.0], [2.0]])
    start = np.array([0.0, 1.0, 0.0])
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])
    baseline = cox_baseline_hazard(
        np.zeros(1), X, stop, event, start=start, ties="breslow"
    )[0]

    # t=1 risk set: rows 0 and 2; t=2 risk set: rows 1 and 2.
    assert np.allclose(baseline["hazard"], [0.5, 0.5])
    evaluated = step_evaluate(
        np.array([0.5, 1.0, 1.5, 2.0, 4.0]),
        baseline["time"],
        baseline["cumulative_hazard"],
    )
    assert np.allclose(evaluated, [0.0, 0.5, 0.5, 1.0, 1.0])


def test_efron_uses_conventional_breslow_baseline_after_coefficient_fit():
    X = np.array([[0.0], [1.0], [2.0], [-0.5]])
    stop = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1, 1, 1, 0])
    beta = np.array([0.2])
    breslow = cox_baseline_hazard(beta, X, stop, event, ties="breslow")[0]
    efron = cox_baseline_hazard(beta, X, stop, event, ties="efron")[0]
    exact = cox_baseline_hazard(beta, X, stop, event, ties="exact")[0]
    assert np.allclose(efron["hazard"], breslow["hazard"], rtol=0, atol=0)
    assert np.allclose(exact["hazard"], breslow["hazard"], rtol=0, atol=0)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_counting_process_solver_matches_statsmodels_entry_and_strata(ties):
    smd = pytest.importorskip("statsmodels.duration.api")
    rng = np.random.default_rng(712)
    n, p = 180, 3
    X = rng.normal(size=(n, p))
    beta = np.array([0.35, -0.2, 0.12])
    raw_time = rng.exponential(scale=np.exp(-X @ beta))
    censor = rng.exponential(scale=np.median(raw_time) * 1.5, size=n)
    stop = np.minimum(raw_time, censor) + 0.2
    event = (raw_time <= censor).astype(int)
    start = rng.uniform(0.0, 0.15, size=n)
    strata = rng.integers(0, 3, size=n)

    result = fit_counting_process_cox(
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
        tol=1e-9,
        max_iter=80,
    )
    reference = smd.PHReg(
        stop, X, status=event, entry=start, strata=strata, ties=ties
    ).fit(disp=0)

    assert result["converged"]
    assert np.allclose(result["coef"], reference.params, rtol=2e-5, atol=2e-6)
    history = np.asarray(result["objective_history"], dtype=float)
    assert np.all(np.diff(history) >= -1e-10)
    assert np.linalg.norm(result["penalized_score"]) < 1e-6


def _backend_objective(backend, beta, X, stop, event, ties):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA device unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA unavailable: {exc}")
        result = cox_counting_process_objective(
            cp.asarray(beta),
            cp.asarray(X),
            cp.asarray(stop),
            cp.asarray(event),
            ties=ties,
        )
        return (
            float(cp.asnumpy(result["log_likelihood"])),
            cp.asnumpy(result["score"]),
            cp.asnumpy(result["information"]),
        )
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")
    result = cox_counting_process_objective(
        torch.as_tensor(beta, dtype=torch.float64, device="cuda"),
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(stop, dtype=torch.float64, device="cuda"),
        torch.as_tensor(event, dtype=torch.int64, device="cuda"),
        ties=ties,
    )
    return (
        float(result["log_likelihood"].detach().cpu()),
        result["score"].detach().cpu().numpy(),
        result["information"].detach().cpu().numpy(),
    )


@pytest.mark.parametrize("backend", ["cupy", "torch"])
@pytest.mark.parametrize("ties", ["efron", "exact"])
def test_counting_process_backend_parity(backend, ties):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(36, 4))
    stop = rng.integers(1, 6, size=36).astype(float)
    event = rng.binomial(1, 0.65, size=36)
    event[0] = 1
    beta = rng.normal(scale=0.1, size=4)
    expected = cox_counting_process_objective(beta, X, stop, event, ties=ties)
    actual = _backend_objective(backend, beta, X, stop, event, ties)
    assert np.allclose(actual[0], expected["log_likelihood"], rtol=1e-10, atol=1e-10)
    assert np.allclose(actual[1], expected["score"], rtol=1e-9, atol=1e-9)
    assert np.allclose(actual[2], expected["information"], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
def test_stratified_objective_is_invariant_to_per_stratum_constant_shifts(ties):
    rng = np.random.default_rng(19)
    rows_per_stratum = 8
    X = rng.normal(size=(2 * rows_per_stratum, 2))
    stop_one = np.array([1, 1, 2, 2, 3, 3, 4, 5], dtype=np.float64)
    event_one = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int64)
    stop = np.tile(stop_one, 2)
    event = np.tile(event_one, 2)
    strata = np.repeat([0, 1], rows_per_stratum)
    beta = np.array([0.15, -0.08])
    shifted_X = X.copy()
    shifted_X[strata == 0] += 1e10
    shifted_X[strata == 1] -= 1e10

    reference = cox_counting_process_objective(
        beta, X, stop, event, strata=strata, ties=ties
    )
    shifted = cox_counting_process_objective(
        beta, shifted_X, stop, event, strata=strata, ties=ties
    )
    assert np.allclose(
        shifted["log_likelihood"], reference["log_likelihood"], atol=2e-6
    )
    assert np.allclose(shifted["score"], reference["score"], rtol=2e-5, atol=2e-6)
    assert np.allclose(
        shifted["information"], reference["information"], rtol=2e-5, atol=2e-6
    )

    baseline_reference = cox_baseline_hazard(
        beta, X, stop, event, strata=strata, ties=ties
    )
    baseline_shifted = cox_baseline_hazard(
        beta, shifted_X, stop, event, strata=strata, ties=ties
    )
    for code in (0, 1):
        assert np.allclose(
            baseline_shifted[code]["log_cumulative_hazard_centered"],
            baseline_reference[code]["log_cumulative_hazard_centered"],
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"penalty": np.nan}, "penalty"),
        ({"penalty": np.inf}, "penalty"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 2.5}, "max_iter"),
        ({"tol": np.nan}, "tol"),
        ({"tol": np.inf}, "tol"),
    ],
)
def test_counting_process_solver_rejects_invalid_controls(kwargs, match):
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])
    with pytest.raises(ValueError, match=match):
        fit_counting_process_cox(X, stop, event, **kwargs)


def test_counting_process_solver_rejects_nonfinite_initial_coefficients():
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])
    with pytest.raises(ValueError, match="init_coef"):
        fit_counting_process_cox(X, stop, event, init_coef=[np.nan])
