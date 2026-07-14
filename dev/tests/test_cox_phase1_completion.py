"""End-to-end contract tests for Phase-1 public ``CoxPH`` capabilities."""

from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from statgpu.survival import CoxPH


def _right_censored_subjects(n=140, p=3, seed=3101):
    """Generate right-censored subject-level data with continuous times."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.45, -0.25, p)
    event_time = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor_time = rng.exponential(scale=1.8, size=n) + 0.05
    stop = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.int64)
    return X.astype(np.float64), stop.astype(np.float64), event


def _split_into_counting_rows(X, stop, event, strata=None):
    """Split every subject into two equivalent ``(start, stop]`` rows."""
    n = X.shape[0]
    cut = 0.5 * stop
    X_rows = np.repeat(X, 2, axis=0)
    start_rows = np.column_stack([np.zeros(n), cut]).reshape(-1)
    stop_rows = np.column_stack([cut, stop]).reshape(-1)
    event_rows = np.column_stack([np.zeros(n, dtype=np.int64), event]).reshape(-1)
    subject_rows = np.repeat(np.arange(n), 2)
    strata_rows = None if strata is None else np.repeat(strata, 2)
    return X_rows, start_rows, stop_rows, event_rows, subject_rows, strata_rows


def _stratified_subjects(n=240, p=3, seed=3102):
    """Generate two strata with shared coefficients and distinct baselines."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    labels = np.where(np.arange(n) < n // 2, "clinic-a", "clinic-b")
    beta = np.array([0.5, -0.3, 0.2])[:p]
    baseline_rate = np.where(labels == "clinic-a", 0.45, 1.6)
    event_time = rng.exponential(scale=1.0 / (baseline_rate * np.exp(X @ beta))) + 0.02
    censor_time = rng.exponential(scale=3.0, size=n) + 0.02
    stop = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.int64)
    return X.astype(np.float64), stop.astype(np.float64), event, labels


def _require_gpu_backend(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device is unavailable")
        except Exception as exc:  # pragma: no cover - driver-specific
            pytest.skip(f"CuPy CUDA backend is unavailable: {exc}")
        return

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")


def _manual_exact_loglik(beta, X, stop, event):
    """Brute-force exact tied-event partial log likelihood for one feature."""
    beta = float(beta)
    x = np.asarray(X, dtype=np.float64).reshape(-1)
    total = 0.0
    for failure_time in np.unique(stop[event == 1]):
        fail = np.flatnonzero((stop == failure_time) & (event == 1))
        risk = np.flatnonzero(stop >= failure_time)
        d = fail.size
        log_weights = [beta * np.sum(x[list(group)]) for group in combinations(risk, d)]
        largest = max(log_weights)
        log_partition = largest + np.log(
            np.sum(np.exp(np.asarray(log_weights) - largest))
        )
        total += beta * np.sum(x[fail]) - log_partition
    return float(total)


def _exact_tied_data():
    X = np.array([-1.4, -0.8, 0.1, 0.7, 1.3, -0.5, 0.4, 1.1, -1.0, 0.9])[:, None]
    stop = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 5], dtype=np.float64)
    event = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=np.int64)
    return X, stop, event


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_start_stop_rows_match_equivalent_subject_level_fit(ties):
    X, stop, event = _right_censored_subjects(seed=3110)
    X_rows, start_rows, stop_rows, event_rows, subject_rows, _ = (
        _split_into_counting_rows(X, stop, event)
    )

    reference = CoxPH(
        ties=ties,
        device="cpu",
        compute_inference=True,
        compute_cindex=True,
        max_iter=100,
        tol=1e-9,
    ).fit(X, stop, event)
    counting = CoxPH(
        ties=ties,
        device="cpu",
        compute_inference=True,
        compute_cindex=True,
        max_iter=100,
        tol=1e-9,
    ).fit(
        X_rows,
        stop_rows,
        event_rows,
        start=start_rows,
        subject_id=subject_rows,
    )

    assert counting._is_counting_process
    assert counting._converged
    assert np.all(np.diff(counting._objective_history) >= -1e-10)
    assert_allclose(counting.coef_, reference.coef_, rtol=2e-6, atol=2e-7)
    assert_allclose(
        counting._log_likelihood, reference._log_likelihood, rtol=2e-8, atol=2e-8
    )
    assert_allclose(counting._bse, reference._bse, rtol=2e-5, atol=2e-6)
    assert_allclose(counting._cindex, reference._cindex, rtol=0, atol=1e-12)


def test_stratified_fit_matches_statsmodels_and_has_independent_baselines():
    smd = pytest.importorskip("statsmodels.duration.api")
    X, stop, event, strata = _stratified_subjects()
    model = CoxPH(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    ).fit(X, stop, event, strata=strata)
    reference = smd.PHReg(stop, X, status=event, strata=strata, ties="efron").fit(
        disp=0
    )

    assert model._strata_labels.tolist() == ["clinic-a", "clinic-b"]
    assert set(model._baseline_by_stratum) == {0, 1}
    assert_allclose(model.coef_, reference.params, rtol=2e-5, atol=2e-6)
    assert_allclose(model._bse, reference.bse, rtol=3e-2, atol=3e-3)
    assert not np.array_equal(
        model._baseline_by_stratum[0]["cumulative_hazard"],
        model._baseline_by_stratum[1]["cumulative_hazard"],
    )


def test_surv_start_stop_formula_matches_direct_counting_api():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, stop, event = _right_censored_subjects(n=100, p=2, seed=3111)
    X_rows, start_rows, stop_rows, event_rows, subject_rows, _ = (
        _split_into_counting_rows(X, stop, event)
    )
    frame = pd.DataFrame(
        {
            "start": start_rows,
            "stop": stop_rows,
            "event": event_rows,
            "x1": X_rows[:, 0],
            "x2": X_rows[:, 1],
        }
    )

    direct = CoxPH(ties="efron", device="cpu", compute_cindex=False, tol=1e-9).fit(
        X_rows,
        stop_rows,
        event_rows,
        start=start_rows,
        subject_id=subject_rows,
    )
    formula = CoxPH(ties="efron", device="cpu", compute_cindex=False, tol=1e-9).fit(
        formula="Surv(start, stop, event) ~ x1 + x2",
        data=frame,
        subject_id=subject_rows,
    )

    assert formula._feature_names == ["x1", "x2"]
    assert formula._design_info is not None
    assert_allclose(formula.coef_, direct.coef_, rtol=1e-10, atol=1e-11)
    assert_allclose(formula._bse, direct._bse, rtol=1e-10, atol=1e-11)
    assert_allclose(
        formula._baseline_cumulative_hazard,
        direct._baseline_cumulative_hazard,
        rtol=1e-10,
        atol=1e-11,
    )
    assert np.isfinite(
        formula.score(
            frame,
            stop_rows,
            event_rows,
            start=start_rows,
            subject_id=subject_rows,
        )
    )


def test_counting_formula_supports_categorical_interaction_transform_and_na_drop():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, stop, event = _right_censored_subjects(n=90, p=2, seed=3118)
    X_rows, start_rows, stop_rows, event_rows, _, _ = _split_into_counting_rows(
        X, stop, event
    )
    frame = pd.DataFrame(
        {
            "start": start_rows,
            "stop": stop_rows,
            "event": event_rows,
            "x1": X_rows[:, 0],
            "positive_x2": np.exp(X_rows[:, 1]),
            "group": pd.Categorical(
                np.repeat(np.where(np.arange(X.shape[0]) % 2, "b", "a"), 2)
            ),
        }
    )
    frame.loc[7, "x1"] = np.nan
    model = CoxPH(ties="breslow", device="cpu", compute_cindex=False, tol=1e-9).fit(
        formula=("Surv(start, stop, event) ~ " "x1 * C(group) + np.log(positive_x2)"),
        data=frame,
    )
    assert model._nobs == frame.shape[0] - 1
    assert model._feature_names == [
        "C(group)[T.b]",
        "x1",
        "x1:C(group)[T.b]",
        "np.log(positive_x2)",
    ]
    assert model.coef_.shape == (4,)
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model._bse))


def test_formula_dataframe_prediction_reuses_fitted_design_matrix():
    pd = pytest.importorskip("pandas")
    patsy = pytest.importorskip("patsy")
    X, stop, event = _right_censored_subjects(n=90, p=2, seed=3139)
    frame = pd.DataFrame(
        {
            "time": stop,
            "event": event,
            "x1": X[:, 0],
            "grp": np.where(np.arange(X.shape[0]) % 2, "B", "A"),
        }
    )
    model = CoxPH(device="cpu", compute_inference=False, compute_cindex=False).fit(
        formula="Surv(time, event) ~ x1 + C(grp)", data=frame
    )

    prediction_frame = frame.iloc[:8].copy()
    design = np.asarray(
        patsy.build_design_matrices([model._design_info], prediction_frame)[0]
    )
    intercept_index = list(model._design_info.column_names).index("Intercept")
    design = np.delete(design, intercept_index, axis=1)
    expected_risk = design @ model.coef_

    assert_allclose(
        model.predict_risk_score(prediction_frame), expected_risk, rtol=0, atol=1e-12
    )
    assert_allclose(
        model.predict(prediction_frame),
        np.exp(expected_risk),
        rtol=0,
        atol=1e-12,
    )
    assert np.isfinite(model.score(prediction_frame, stop[:8], event[:8]))
    invalid_design = design.copy()
    invalid_design[0, 0] = np.nan
    with pytest.raises(ValueError, match="X must contain only finite values"):
        model.predict(invalid_design)


def test_exact_ties_fit_matches_bruteforce_partial_likelihood():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    X, stop, event = _exact_tied_data()

    model = CoxPH(
        ties="exact",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-11,
    ).fit(X, stop, event)
    optimum = scipy_optimize.minimize_scalar(
        lambda value: -_manual_exact_loglik(value, X, stop, event),
        bounds=(-6.0, 6.0),
        method="bounded",
        options={"xatol": 1e-12},
    )

    assert optimum.success
    assert model._converged
    assert_allclose(model.coef_[0], optimum.x, rtol=2e-7, atol=2e-8)
    assert_allclose(
        model._log_likelihood,
        _manual_exact_loglik(model.coef_[0], X, stop, event),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isfinite(model._bse[0]) and model._bse[0] > 0
    with pytest.raises(NotImplementedError, match="robust covariance"):
        CoxPH(ties="exact", cov_type="hc0", device="cpu").fit(X, stop, event)


def test_custom_time_stratified_survival_uses_each_baseline_step_function():
    X, stop, event, strata = _stratified_subjects(seed=3112)
    model = CoxPH(ties="efron", device="cpu", compute_cindex=False, tol=1e-9).fit(
        X, stop, event, strata=strata
    )
    first_a = model._baseline_by_stratum[0]["time"][0]
    first_b = model._baseline_by_stratum[1]["time"][0]
    final_time = max(
        model._baseline_by_stratum[0]["time"][-1],
        model._baseline_by_stratum[1]["time"][-1],
    )
    times = np.array([min(first_a, first_b) - 0.01, first_a, first_b, final_time + 1.0])
    X_new = np.vstack([X[0], X[0]])
    prediction_labels = np.array(["clinic-a", "clinic-b"])

    survival, returned_times = model.predict_survival(
        X_new, times=times, strata=prediction_labels
    )
    assert survival.shape == (2, times.size)
    assert_array_equal(returned_times, times)
    assert_allclose(survival[:, 0], 1.0, rtol=0, atol=0)

    for row, stratum_code in enumerate((0, 1)):
        baseline = model._baseline_by_stratum[stratum_code]
        indices = np.searchsorted(baseline["time"], times, side="right") - 1
        h0 = np.zeros(times.size)
        valid = indices >= 0
        h0[valid] = baseline["cumulative_hazard"][indices[valid]]
        expected = np.exp(-h0 * np.exp(X_new[row] @ model.coef_))
        assert_allclose(survival[row], expected, rtol=1e-12, atol=1e-12)

    with pytest.raises(ValueError, match="strata is required"):
        model.predict_survival(X_new, times=times)
    with pytest.raises(ValueError, match="unknown prediction stratum"):
        model.predict_survival(X_new[:1], times=times, strata=["not-trained"])


def test_counting_compute_inference_false_clears_all_inference_outputs():
    X, stop, event = _right_censored_subjects(n=90, seed=3113)
    X_rows, start_rows, stop_rows, event_rows, subject_rows, _ = (
        _split_into_counting_rows(X, stop, event)
    )
    model = CoxPH(
        ties="efron",
        device="cpu",
        compute_inference=False,
        compute_cindex=False,
    ).fit(
        X_rows,
        stop_rows,
        event_rows,
        start=start_rows,
        subject_id=subject_rows,
    )

    for value in (
        model._var_matrix,
        model._bse,
        model._zvalues,
        model._pvalues,
        model._conf_int,
        model._lr_test_stat,
        model._wald_test_stat,
        model._score_test_stat,
    ):
        assert value is None
    assert model._baseline_by_stratum is None
    assert model._baseline_cumulative_hazard is None
    with pytest.raises(RuntimeError, match="compute_inference=True"):
        model.predict_survival(X[:2], times=[0.2, 0.8])


def test_stratified_score_contract_does_not_depend_on_baseline_storage():
    X, stop, event, strata = _stratified_subjects(n=120, seed=3119)
    model = CoxPH(
        ties="breslow",
        device="cpu",
        compute_inference=False,
        compute_cindex=False,
    ).fit(X, stop, event, strata=strata)
    assert model._baseline_by_stratum is None
    with pytest.raises(ValueError, match="strata is required"):
        model.score(X, stop, event)
    score = model.score(X, stop, event, strata=strata)
    assert 0.0 <= score <= 1.0


def test_counting_hc0_hc1_and_cluster_covariance_contracts():
    X, stop, event = _right_censored_subjects(n=120, seed=3114)
    X_rows, start_rows, stop_rows, event_rows, subject_rows, _ = (
        _split_into_counting_rows(X, stop, event)
    )
    common = dict(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    )
    fits = {}
    for cov_type in ("hc0", "hc1", "cluster"):
        fits[cov_type] = CoxPH(cov_type=cov_type, **common).fit(
            X_rows,
            stop_rows,
            event_rows,
            start=start_rows,
            subject_id=subject_rows,
            cluster=subject_rows if cov_type == "cluster" else None,
        )
        covariance = fits[cov_type]._var_matrix
        assert covariance.shape == (X.shape[1], X.shape[1])
        assert np.all(np.isfinite(covariance))
        assert_allclose(covariance, covariance.T, rtol=0, atol=1e-12)
        assert_allclose(
            np.diag(covariance), fits[cov_type]._bse ** 2, rtol=1e-12, atol=1e-14
        )

    n_subjects = np.unique(subject_rows).size
    correction = n_subjects / (n_subjects - X_rows.shape[1])
    assert_allclose(
        fits["hc1"]._var_matrix,
        correction * fits["hc0"]._var_matrix,
        rtol=1e-10,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="cluster ids are required"):
        CoxPH(cov_type="cluster", **common).fit(
            X_rows, stop_rows, event_rows, start=start_rows
        )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_robust_bse_matches_statsmodels_martingale_residual_contract(ties):
    smd = pytest.importorskip("statsmodels.duration.api")
    rng = np.random.default_rng(3120)
    X = rng.normal(size=(180, 3))
    stop = rng.integers(1, 12, size=180).astype(np.float64)
    event = rng.binomial(1, 0.65, size=180)
    event[0] = 1
    model = CoxPH(
        ties=ties,
        cov_type="hc0",
        device="cpu",
        compute_cindex=False,
        tol=1e-10,
    ).fit(X, stop, event)
    reference = smd.PHReg(stop, X, status=event, ties=ties).fit(disp=0)
    score_residuals = np.nan_to_num(reference.model.score_residuals(reference.params))
    information = -reference.model.hessian(reference.params)
    bread = np.linalg.inv(information)
    reference_covariance = bread @ score_residuals.T @ score_residuals @ bread
    assert_allclose(model.coef_, reference.params, rtol=1e-8, atol=1e-9)
    assert_allclose(model._var_matrix, reference_covariance, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("cov_type", ["hc0", "hc1", "cluster"])
def test_repeated_subject_residuals_are_aggregated_before_sandwich(cov_type):
    X, stop, event = _right_censored_subjects(n=120, seed=3121)
    X_rows, start_rows, stop_rows, event_rows, subject_rows, _ = (
        _split_into_counting_rows(X, stop, event)
    )
    kwargs = dict(
        ties="breslow",
        cov_type=cov_type,
        device="cpu",
        compute_cindex=False,
        tol=1e-9,
    )
    reference = CoxPH(**kwargs).fit(
        X,
        stop,
        event,
        subject_id=np.arange(X.shape[0]),
        cluster=np.arange(X.shape[0]) if cov_type == "cluster" else None,
    )
    counting = CoxPH(**kwargs).fit(
        X_rows,
        stop_rows,
        event_rows,
        start=start_rows,
        subject_id=subject_rows,
        cluster=subject_rows if cov_type == "cluster" else None,
    )
    assert_allclose(counting.coef_, reference.coef_, rtol=2e-7, atol=2e-8)
    assert_allclose(counting._var_matrix, reference._var_matrix, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("penalty", [0.0, 10.0])
def test_zero_start_preserves_legacy_efron_inference_and_survival(penalty):
    X, stop, event = _right_censored_subjects(n=150, seed=3122)
    # Force real ties while keeping all stop times strictly positive.
    stop = np.maximum(np.ceil(stop * 8.0) / 8.0, 0.125)
    common = dict(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        tol=1e-9,
        penalty=penalty,
    )
    legacy = CoxPH(**common).fit(X, stop, event)
    counting = CoxPH(**common).fit(X, stop, event, start=np.zeros_like(stop))
    assert_allclose(counting.coef_, legacy.coef_, rtol=2e-8, atol=2e-9)
    assert_allclose(counting._bse, legacy._bse, rtol=2e-7, atol=2e-9)
    assert_allclose(
        counting._baseline_cumulative_hazard,
        legacy._baseline_cumulative_hazard,
        rtol=2e-8,
        atol=2e-10,
    )
    times = np.quantile(stop, [0.2, 0.5, 0.8])
    expected, _ = legacy.predict_survival(X[:3], times=times)
    actual, _ = counting.predict_survival(X[:3], times=times)
    assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)


def test_public_fit_rejects_fractional_events_before_cast_on_both_paths():
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([0.5, 1.0, 0.0])
    with pytest.raises(ValueError, match="event"):
        CoxPH(device="cpu").fit(X, stop, event)
    with pytest.raises(ValueError, match="event"):
        CoxPH(device="cpu").fit(X, stop, event, start=np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        CoxPH(device="cpu").fit(X, np.array([0.0, 2.0, 3.0]), [0, 1, 0])
    with pytest.raises(ValueError, match="at least one observed event"):
        CoxPH(device="cpu").fit(X, stop, np.zeros(3))


def test_public_score_rejects_fractional_events_before_cast():
    X, stop, event = _right_censored_subjects(n=80, p=2, seed=3138)
    model = CoxPH(device="cpu", compute_inference=False, compute_cindex=False).fit(
        X, stop, event
    )
    invalid_event = event.astype(np.float64)
    invalid_event[0] = 0.5
    with pytest.raises(ValueError, match="event must contain only 0/1"):
        model.score(X, stop, invalid_event)


def test_one_dimensional_multifeature_prediction_is_one_row():
    X, stop, event = _right_censored_subjects(n=100, p=3, seed=3123)
    model = CoxPH(device="cpu", compute_cindex=False).fit(X, stop, event)
    expected_risk = np.array([X[0] @ model.coef_])
    assert_allclose(model.predict_risk_score(X[0]), expected_risk)
    assert_allclose(model.predict_hazard_ratio(X[0]), np.exp(expected_risk))


def test_refit_from_counting_to_standard_resets_state_and_score_contract():
    X, stop, event, strata = _stratified_subjects(n=160, seed=3115)
    model = CoxPH(ties="breslow", device="cpu", compute_cindex=True, tol=1e-9).fit(
        X, stop, event, strata=strata
    )
    stratified_score = model.score(X, stop, event, strata=strata)
    assert 0.0 <= stratified_score <= 1.0

    X2, stop2, event2 = _right_censored_subjects(n=130, seed=3116)
    fresh = CoxPH(ties="breslow", device="cpu", compute_cindex=True, tol=1e-9).fit(
        X2, stop2, event2
    )
    model.fit(X2, stop2, event2)

    assert model._baseline_by_stratum is None
    assert model._strata is None
    assert model._strata_labels is None
    assert model._is_counting_process is False
    assert_allclose(model.coef_, fresh.coef_, rtol=1e-12, atol=1e-12)
    assert_allclose(model.score(X2, stop2, event2), fresh.score(X2, stop2, event2))


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_counting_strata_numpy_cupy_torch_parity(device):
    _require_gpu_backend(device)
    X, stop, event, strata = _stratified_subjects(n=140, seed=3117)
    X_rows, start_rows, stop_rows, event_rows, subject_rows, strata_rows = (
        _split_into_counting_rows(X, stop, event, strata=strata)
    )
    common = dict(
        ties="efron",
        compute_inference=True,
        compute_cindex=True,
        max_iter=100,
        tol=1e-9,
    )
    cpu = CoxPH(device="cpu", **common).fit(
        X_rows,
        stop_rows,
        event_rows,
        start=start_rows,
        strata=strata_rows,
        subject_id=subject_rows,
    )
    gpu = CoxPH(device=device, **common).fit(
        X_rows,
        stop_rows,
        event_rows,
        start=start_rows,
        strata=strata_rows,
        subject_id=subject_rows,
    )

    assert_allclose(gpu.coef_, cpu.coef_, rtol=2e-7, atol=2e-8)
    assert_allclose(gpu._bse, cpu._bse, rtol=2e-7, atol=2e-8)
    assert_allclose(gpu._log_likelihood, cpu._log_likelihood, rtol=2e-10, atol=2e-10)
    assert_allclose(gpu._cindex, cpu._cindex, rtol=0, atol=1e-12)
    for code in cpu._baseline_by_stratum:
        assert_allclose(
            gpu._baseline_by_stratum[code]["time"],
            cpu._baseline_by_stratum[code]["time"],
            rtol=0,
            atol=0,
        )
        assert_allclose(
            gpu._baseline_by_stratum[code]["cumulative_hazard"],
            cpu._baseline_by_stratum[code]["cumulative_hazard"],
            rtol=2e-7,
            atol=2e-9,
        )

    times = np.quantile(stop, [0.1, 0.5, 0.9])
    labels = np.array(["clinic-a", "clinic-b"])
    pred_cpu, _ = cpu.predict_survival(X[:2], times=times, strata=labels)
    pred_gpu, _ = gpu.predict_survival(X[:2], times=times, strata=labels)
    assert_allclose(pred_gpu, pred_cpu, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_exact_ties_public_api_numpy_cupy_torch_parity(device):
    _require_gpu_backend(device)
    X, stop, event = _exact_tied_data()
    common = dict(
        ties="exact",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-11,
    )
    cpu = CoxPH(device="cpu", **common).fit(X, stop, event)
    gpu = CoxPH(device=device, **common).fit(X, stop, event)

    assert_allclose(gpu.coef_, cpu.coef_, rtol=2e-8, atol=2e-9)
    assert_allclose(gpu._bse, cpu._bse, rtol=2e-8, atol=2e-9)
    assert_allclose(gpu._log_likelihood, cpu._log_likelihood, rtol=2e-10, atol=2e-10)
    assert_allclose(
        gpu._baseline_cumulative_hazard,
        cpu._baseline_cumulative_hazard,
        rtol=2e-8,
        atol=2e-10,
    )


def test_formula_na_drop_aligns_all_row_level_group_inputs():
    pd = pytest.importorskip("pandas")
    X, stop, event = _right_censored_subjects(n=90, p=2, seed=3130)
    data = pd.DataFrame({"stop": stop, "event": event, "x1": X[:, 0], "x2": X[:, 1]})
    data.index = np.repeat(np.arange(45), 2)  # duplicate labels are intentional
    data.iloc[7, data.columns.get_loc("x1")] = np.nan
    keep = np.arange(len(data)) != 7
    cluster = np.repeat(np.arange(30), 3)
    strata = np.where(np.arange(len(data)) % 2, 0.2, 0.8)
    subject_id = np.arange(len(data))

    formula = CoxPH(
        ties="efron",
        cov_type="cluster",
        device="cpu",
        compute_cindex=False,
    ).fit(
        formula="Surv(stop, event) ~ x1 + x2",
        data=data,
        cluster=cluster,
        strata=strata,
        subject_id=subject_id,
    )
    direct = CoxPH(
        ties="efron",
        cov_type="cluster",
        device="cpu",
        compute_cindex=False,
    ).fit(
        X[keep],
        stop[keep],
        event[keep],
        cluster=cluster[keep],
        strata=strata[keep],
        subject_id=subject_id[keep],
    )
    assert_allclose(formula.coef_, direct.coef_, rtol=2e-9, atol=2e-10)
    assert_allclose(formula._var_matrix, direct._var_matrix, rtol=2e-8, atol=2e-10)


def test_formula_start_stop_rejects_duplicate_explicit_entry_definition():
    pd = pytest.importorskip("pandas")
    X, stop, event = _right_censored_subjects(n=30, p=1, seed=3131)
    start = 0.2 * stop
    data = pd.DataFrame({"start": start, "stop": stop, "event": event, "x": X[:, 0]})
    with pytest.raises(ValueError, match="already defines entry times"):
        CoxPH(device="cpu").fit(
            formula="Surv(start, stop, event) ~ x",
            data=data,
            entry=np.zeros_like(stop),
        )


@pytest.mark.parametrize("ties", ["efron", "exact"])
def test_counting_objective_and_inference_are_invariant_to_large_feature_shift(ties):
    if ties == "exact":
        X, stop, event = _exact_tied_data()
    else:
        X, stop, event = _right_censored_subjects(n=100, p=2, seed=3132)
        stop = np.maximum(np.ceil(stop * 5.0) / 5.0, 0.2)
    start = np.zeros_like(stop)
    common = dict(
        ties=ties,
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        tol=1e-10,
    )
    reference = CoxPH(**common).fit(X, stop, event, start=start)
    shifted = CoxPH(**common).fit(X + 1e8, stop, event, start=start)
    assert_allclose(shifted.coef_, reference.coef_, rtol=2e-6, atol=2e-7)
    assert_allclose(shifted._bse, reference._bse, rtol=2e-6, atol=2e-7)
    assert_allclose(
        shifted.log_likelihood, reference.log_likelihood, rtol=1e-10, atol=2e-7
    )


def test_shifted_counting_survival_uses_stable_log_baseline_product():
    X, stop, event = _right_censored_subjects(n=100, p=2, seed=3133)
    start = np.zeros_like(stop)
    common = dict(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        tol=1e-10,
    )
    reference = CoxPH(**common).fit(X, stop, event, start=start)
    shifted = CoxPH(**common).fit(X + 1000.0, stop, event, start=start)
    times = np.quantile(stop[event == 1], [0.2, 0.5, 0.8])
    expected, _ = reference.predict_survival(X[:8], times=times)
    actual, _ = shifted.predict_survival(X[:8] + 1000.0, times=times)
    assert np.all(np.isfinite(actual))
    assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)


def test_standard_api_automatically_uses_stable_path_for_large_common_offset():
    X, stop, event = _right_censored_subjects(n=100, p=2, seed=3136)
    stop = np.maximum(np.ceil(stop * 5.0) / 5.0, 0.2)
    common = dict(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        tol=1e-10,
    )
    reference = CoxPH(**common).fit(X, stop, event)
    shifted = CoxPH(**common).fit(X + 1e8, stop, event)
    assert_allclose(shifted.coef_, reference.coef_, rtol=2e-6, atol=2e-7)
    assert_allclose(shifted._bse, reference._bse, rtol=2e-6, atol=2e-7)


def test_large_offset_detection_is_per_feature_not_masked_by_another_scale():
    X, stop, event = _right_censored_subjects(n=100, p=2, seed=3137)
    transformed = X.copy()
    transformed[:, 0] += 1e10
    # This scale was large enough to hide the first column from the former
    # max(location)-versus-max(scale) detector without making inference itself
    # numerically unidentified.
    transformed[:, 1] *= 1e5

    assert CoxPH._has_large_common_feature_offset(transformed)
    model = CoxPH(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        tol=1e-10,
    ).fit(transformed, stop, event)
    assert model._fitted
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.log_likelihood)


def test_exact_singular_information_is_not_reported_as_zero_variance():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    stop = np.ones(4)
    event = np.ones(4, dtype=np.int64)
    with pytest.raises(RuntimeError, match="information is singular"):
        CoxPH(ties="exact", device="cpu", compute_inference=True).fit(X, stop, event)
    estimation_only = CoxPH(
        ties="exact",
        device="cpu",
        compute_inference=False,
        compute_cindex=False,
    ).fit(X, stop, event)
    assert estimation_only._fitted
    assert estimation_only._bse is None


def test_public_likelihood_information_criteria_and_concordance_properties():
    X, stop, event = _right_censored_subjects(n=80, p=2, seed=3134)
    model = CoxPH(device="cpu", compute_cindex=True).fit(X, stop, event)
    assert model.log_likelihood == pytest.approx(model._log_likelihood)
    assert model.concordance_index == pytest.approx(model._cindex)
    assert model.aic == pytest.approx(-2 * model._log_likelihood + 4)
    assert model.bic == pytest.approx(
        -2 * model._log_likelihood + 2 * np.log(event.sum())
    )
    penalized = CoxPH(
        device="cpu",
        penalty=0.1,
        compute_inference=False,
        compute_cindex=False,
    ).fit(X, stop, event)
    assert penalized._lr_test_stat is None
    with pytest.raises(RuntimeError, match="unpenalized"):
        _ = penalized.aic


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_device_fractional_strata_are_encoded_without_integer_collapse(device):
    _require_gpu_backend(device)
    X, stop, event = _right_censored_subjects(n=100, p=2, seed=3135)
    strata = np.where(np.arange(len(stop)) % 2, 0.2, 0.8)
    common = dict(
        ties="efron",
        compute_inference=True,
        compute_cindex=False,
        tol=1e-10,
    )
    cpu = CoxPH(device="cpu", **common).fit(X, stop, event, strata=strata)
    if device == "cuda":
        import cupy as cp

        arrays = tuple(cp.asarray(value) for value in (X, stop, event, strata))
    else:
        import torch

        arrays = tuple(
            torch.as_tensor(value, device="cuda") for value in (X, stop, event, strata)
        )
    gpu = CoxPH(device=device, **common).fit(
        arrays[0], arrays[1], arrays[2], strata=arrays[3]
    )
    assert set(gpu._baseline_by_stratum) == {0, 1}
    assert_array_equal(gpu._strata_labels, np.array([0.2, 0.8]))
    assert_allclose(gpu.coef_, cpu.coef_, rtol=2e-7, atol=2e-8)


def test_formula_survival_response_na_is_removed_and_auxiliary_rows_align():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, stop, event = _right_censored_subjects(n=100, p=2, seed=3140)
    frame = pd.DataFrame(
        {"time": stop, "event": event.astype(float), "x1": X[:, 0], "x2": X[:, 1]}
    )
    frame.loc[11, "event"] = np.nan
    strata = np.where(np.arange(len(frame)) % 2, "a", "b")
    keep = np.arange(len(frame)) != 11

    formula = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False, tol=1e-10
    ).fit(
        formula="Surv(time, event) ~ x1 + x2",
        data=frame,
        strata=strata,
    )
    direct = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False, tol=1e-10
    ).fit(X[keep], stop[keep], event[keep], strata=strata[keep])

    assert formula._nobs == int(np.sum(keep))
    assert_allclose(formula.coef_, direct.coef_, rtol=1e-10, atol=1e-11)


def test_formula_prediction_missing_values_never_silently_drop_rows():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, stop, event = _right_censored_subjects(n=80, p=2, seed=3141)
    frame = pd.DataFrame(
        {"time": stop, "event": event, "x1": X[:, 0], "x2": X[:, 1]}
    )
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(formula="Surv(time, event) ~ x1 + x2", data=frame)
    prediction = frame.iloc[:8].copy()
    prediction.loc[prediction.index[3], "x1"] = np.nan

    with pytest.raises(ValueError, match="cannot be dropped silently"):
        model.predict(prediction)
    with pytest.raises(ValueError, match="cannot be dropped silently"):
        model.score(prediction, prediction[["time", "event"]].to_numpy())


def test_score_honors_subject_id_even_after_subject_level_fit():
    from statgpu.survival._risk_sets import counting_process_concordance

    X = np.array([[3.0], [0.0], [2.0], [1.0]])
    stop = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 0])
    subject_id = np.array([0, 0, 1, 2])
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, stop, event)
    # Fix a deterministic risk ordering so excluding the within-subject pair
    # changes the concordance value and catches accidental argument ignoring.
    model.coef_ = np.array([-1.0])

    expected = float(
        counting_process_concordance(
            model.coef_, X, stop, event, subject_id=subject_id
        )
    )
    assert expected != pytest.approx(model.score(X, stop, event))
    assert model.score(X, stop, event, subject_id=subject_id) == pytest.approx(expected)


def test_score_encodes_ad_hoc_string_strata_for_unstratified_fit():
    from statgpu.survival._risk_sets import counting_process_concordance

    X = np.array([[3.0], [0.0], [2.0], [1.0]])
    stop = np.array([1.0, 2.0, 1.0, 2.0])
    event = np.array([1, 0, 1, 0])
    labels = np.array(["a", "a", "b", "b"])
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, stop, event)
    model.coef_ = np.array([-1.0])
    expected = float(
        counting_process_concordance(
            model.coef_, X, stop, event, strata=np.array([0, 0, 1, 1])
        )
    )
    assert model.score(X, stop, event, strata=labels) == pytest.approx(expected)


def test_score_rejects_covariate_response_row_mismatch_explicitly():
    X, stop, event = _right_censored_subjects(n=50, p=2, seed=3142)
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, stop, event)
    with pytest.raises(ValueError, match="same number of rows"):
        model.score(X[:-1], stop, event)


def test_exact_robust_cov_type_is_ignored_when_inference_is_disabled():
    X = np.array([[-1.0], [-0.2], [0.4], [1.2], [0.8]])
    stop = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 0, 0])
    model = CoxPH(
        ties="exact",
        cov_type="hc0",
        compute_inference=False,
        compute_cindex=False,
        device="cpu",
    ).fit(X, stop, event)
    assert model._fitted
    assert model._bse is None
    with pytest.raises(NotImplementedError, match="robust covariance"):
        CoxPH(
            ties="exact",
            cov_type="hc0",
            compute_inference=True,
            compute_cindex=False,
            device="cpu",
        ).fit(X, stop, event)
