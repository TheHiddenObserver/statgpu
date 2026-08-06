"""Regression gates from the final complete PR80 review cycle."""

import numpy as np
import pytest

from statgpu.linear_model import PenalizedCoxPHModel
from statgpu.survival import CoxPH, CoxPHCV
from statgpu.survival import _cox_score as cox_score_module
from statgpu.survival import _risk_sets as risk_sets
from statgpu.survival._cox_score import (
    _MAX_CONCORDANCE_PAIR_ENTRIES,
    _concordance_tile_shape,
)
from statgpu.survival._risk_sets import counting_process_concordance


def _fit_sample(seed=2401, n=36, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    stop = np.arange(1, n + 1, dtype=np.float64)
    event = np.ones(n, dtype=np.float64)
    event[::5] = 0.0
    event[0] = 1.0
    return X, stop, event


def _backend_arrays(backend, *values):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA unavailable: {exc}")
        return tuple(cp.asarray(value) for value in values)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        return tuple(
            torch.as_tensor(value, dtype=torch.float64, device="cuda")
            for value in values
        )
    return tuple(np.asarray(value) for value in values)


@pytest.mark.parametrize(
    ("n_events", "n_samples"),
    [
        (100_000, 1_000),
        (1, _MAX_CONCORDANCE_PAIR_ENTRIES + 1),
        (10, 10 * _MAX_CONCORDANCE_PAIR_ENTRIES),
        (0, 1_000),
    ],
)
def test_concordance_pair_tiles_obey_hard_entry_bound(n_events, n_samples):
    event_tile, sample_tile = _concordance_tile_shape(n_events, n_samples)
    assert event_tile >= 1
    assert sample_tile >= 1
    assert event_tile * sample_tile <= _MAX_CONCORDANCE_PAIR_ENTRIES
    assert sample_tile <= max(n_samples, 1)


def test_ordinary_concordance_two_axis_tiling_matches_default(monkeypatch):
    X, stop, event = _fit_sample(seed=2404)
    fitted = CoxPH(
        compute_inference=False,
        compute_cindex=False,
        max_iter=80,
        tol=1e-7,
    ).fit(X, stop, event)
    expected = fitted.score(X, stop, event)
    monkeypatch.setattr(
        cox_score_module,
        "_concordance_tile_shape",
        lambda _n_events, _n_samples: (1, 2),
    )
    assert fitted.score(X, stop, event) == pytest.approx(expected, abs=1e-15)


def test_counting_concordance_two_axis_tiling_matches_default(monkeypatch):
    X, stop, event = _fit_sample(seed=2405)
    beta = np.array([0.2, -0.1])
    start = np.zeros(stop.shape[0], dtype=np.float64)
    strata = np.arange(stop.shape[0], dtype=np.int64) % 2
    expected = counting_process_concordance(
        beta, X, stop, event, start=start, strata=strata
    )
    monkeypatch.setattr(
        risk_sets,
        "concordance_tile_shape",
        lambda _n_events, _n_samples: (1, 2),
    )
    actual = counting_process_concordance(
        beta, X, stop, event, start=start, strata=strata
    )
    assert float(actual) == pytest.approx(float(expected), abs=1e-15)


def test_all_censored_concordance_is_neutral_across_public_paths():
    X, stop, event = _fit_sample(p=1)
    fitted = CoxPH(
        compute_inference=False,
        compute_cindex=False,
        max_iter=80,
        tol=1e-7,
    ).fit(X, stop, event)
    X_score = X[:6]
    stop_score = np.arange(1, 7, dtype=np.float64)
    censored = np.zeros(6, dtype=np.float64)

    assert fitted.score(X_score, stop_score, censored) == 0.5
    assert fitted.score(
        X_score,
        stop_score,
        censored,
        start=np.zeros(6),
        strata=np.array([0, 0, 0, 1, 1, 1]),
    ) == 0.5


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_all_censored_counting_concordance_is_neutral_on_backend(backend):
    X = np.arange(12, dtype=np.float64).reshape(6, 2) / 10.0
    beta = np.array([0.2, -0.1])
    stop = np.arange(1, 7, dtype=np.float64)
    event = np.zeros(6, dtype=np.float64)
    start = np.zeros(6, dtype=np.float64)
    strata = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    beta_b, X_b, stop_b, event_b, start_b, strata_b = _backend_arrays(
        backend, beta, X, stop, event, start, strata
    )
    value = counting_process_concordance(
        beta_b,
        X_b,
        stop_b,
        event_b,
        start=start_b,
        strata=strata_b,
    )
    if backend == "torch":
        value = value.detach().cpu().item()
    elif backend == "cupy":
        value = value.item()
    assert float(value) == 0.5


def test_penalized_cox_all_censored_score_is_neutral():
    X, stop, event = _fit_sample(seed=2402, p=1)
    model = PenalizedCoxPHModel(
        penalty="l2",
        alpha=0.2,
        max_iter=80,
        tol=1e-6,
        compute_inference=False,
    ).fit(X, np.column_stack((stop, event)))
    target = np.column_stack((stop[:5], np.zeros(5)))
    assert model.score(X[:5], target) == 0.5


def test_coxphcv_final_refit_skips_hidden_training_concordance():
    X, stop, event = _fit_sample(seed=2403)
    model = CoxPHCV(
        penalties=np.array([1.0]),
        cv=2,
        random_state=0,
        compute_inference=False,
        max_iter=100,
        tol=1e-6,
        device="cpu",
    ).fit(X, stop, event)
    assert model.estimator_.compute_cindex is False
    assert model.estimator_.concordance_ is None
    assert np.isfinite(model.score(X, stop, event))


@pytest.mark.parametrize(
    "name",
    ["fit_intercept", "gpu_memory_cleanup", "compute_inference", "lla"],
)
def test_penalized_cox_rejects_truthy_string_boolean_controls(name):
    with pytest.raises(ValueError, match=rf"{name} must be a boolean"):
        PenalizedCoxPHModel(**{name: "False"})

    model = PenalizedCoxPHModel()
    with pytest.raises(ValueError, match=rf"{name} must be a boolean"):
        model.set_params(**{name: "False"})


def test_penalized_cox_accepts_integer_boolean_controls_and_clones():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = PenalizedCoxPHModel(
        fit_intercept=0,
        gpu_memory_cleanup=0,
        compute_inference=0,
        lla=1,
    )
    cloned = clone(model)
    assert cloned.fit_intercept == 0
    assert cloned.gpu_memory_cleanup == 0
    assert cloned.compute_inference == 0
    assert cloned.lla == 1
