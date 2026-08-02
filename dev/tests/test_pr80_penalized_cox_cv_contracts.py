import types
import warnings

import numpy as np
import pytest

from statgpu import set_device
from statgpu.backends import _to_float_scalar, get_backend
from statgpu.cross_validation._base import _coerce_cv_indices
from statgpu.cross_validation import _base as cv_base_module
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import _penalized_cox_cv as cox_cv_module
from statgpu.linear_model.penalized import _penalized_cv as penalized_cv_module
from statgpu.losses import CoxPartialLikelihoodLoss
from statgpu.linear_model.penalized import (
    PenalizedCoxPHModel,
    PenalizedGeneralizedLinearModel,
)
from statgpu.penalties import (
    CompositePenalty,
    ElasticNetPenalty,
    L1Penalty,
    L2Penalty,
)
from statgpu.survival import CoxPH


def _survival_sample(seed=8101, n=24, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.65, -0.35, p)
    time = -np.log(rng.uniform(0.05, 0.95, size=n)) / np.exp(X @ beta)
    event = np.ones(n, dtype=np.float64)
    return X, np.column_stack([time, event])


def _backend_inputs(backend_name, X, y):
    if backend_name == "numpy":
        return "cpu", X, y
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device unavailable")
        except Exception:
            pytest.skip("CuPy CUDA runtime unavailable")
        return "cuda", cp.asarray(X), cp.asarray(y)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(y, dtype=torch.float64, device="cuda"),
    )


def _as_numpy(value):
    if type(value).__module__.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize("penalty", ["l1", "l2", "elasticnet", "scad", "mcp"])
def test_penalized_cox_cv_selects_supported_alpha_and_refits_matching_model(
    backend_name, penalty
):
    X, y = _survival_sample(seed=8102)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    alpha_grid = np.array([0.15, 0.03])
    cv_model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty=penalty,
        alpha_grid=alpha_grid,
        l1_ratio=0.4,
        cv=2,
        random_state=12,
        device=device,
        solver="auto",
        max_iter=400,
        tol=1e-6,
        loss_kwargs={"ties": "efron"},
    ).fit(Xb, yb)

    assert cv_model.alpha_ in alpha_grid
    assert np.isfinite(cv_model.best_score_)
    assert np.all(np.isfinite(cv_model.cv_results_["mean_score"]))
    np.testing.assert_array_equal(
        cv_model.cv_results_["valid_score_counts"], np.array([2, 2])
    )
    assert cv_model.cv_results_["required_valid_score_count"] == 2
    assert cv_model.cv_results_["scoring"] == (
        "negative_partial_log_likelihood_per_row"
    )
    assert cv_model.cv_results_["fit_intercept"] is False
    assert isinstance(cv_model.estimator_, PenalizedCoxPHModel)
    assert cv_model.estimator_.fit_intercept is False
    assert cv_model.intercept_ == 0.0
    expected_index = int(np.argmin(cv_model.cv_results_["mean_score"]))
    assert cv_model.alpha_ == pytest.approx(alpha_grid[expected_index])

    direct = PenalizedCoxPHModel(
        penalty=penalty,
        alpha=cv_model.alpha_,
        l1_ratio=0.4,
        ties="efron",
        device=device,
        solver="auto",
        max_iter=400,
        tol=1e-6,
        compute_inference=False,
    ).fit(Xb, yb)
    np.testing.assert_allclose(
        cv_model.coef_, direct.coef_, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        cv_model.predict(Xb), direct.predict(Xb), rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_penalized_cox_cv_preserves_two_column_target(backend_name, monkeypatch):
    X, y = _survival_sample(seed=8103, n=20)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    observed_shapes = []
    original_fit = PenalizedCoxPHModel.fit

    def recording_fit(self, X, y, *args, **kwargs):
        observed_shapes.append(tuple(int(value) for value in y.shape))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(PenalizedCoxPHModel, "fit", recording_fit)
    PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=2,
        random_state=3,
        device=device,
        max_iter=100,
        tol=1e-7,
    ).fit(Xb, yb)

    assert observed_shapes
    assert all(shape[1] == 2 for shape in observed_shapes)
    assert all(shape[0] <= X.shape[0] for shape in observed_shapes)


def test_penalized_cox_cv_accepts_array_like_two_column_target():
    X, y = _survival_sample(seed=8114, n=18)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=2,
        random_state=4,
        device="auto",
        max_iter=100,
        tol=1e-7,
    ).fit(X.tolist(), y.tolist())
    assert model.alpha_ == pytest.approx(0.1)
    assert isinstance(model.estimator_, PenalizedCoxPHModel)

def test_penalized_cox_cv_supports_penalty_object_and_auto_grid():
    X, y = _survival_sample(seed=8110, n=20)
    penalty = L2Penalty(alpha=0.4)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty=penalty,
        n_alphas=3,
        cv=2,
        random_state=9,
        device="cpu",
        max_iter=200,
        tol=1e-7,
    ).fit(X, y)

    assert model.alpha_grid_.shape == (3,)
    assert np.all(np.isfinite(model.alpha_grid_))
    assert model.alpha_ in model.alpha_grid_
    assert isinstance(model.estimator_.penalty, L2Penalty)
    assert model.estimator_.penalty.alpha == pytest.approx(model.alpha_)
    assert penalty.alpha == pytest.approx(0.4)


_BAD_CUSTOM_FOLD_INDICES = [
    np.array([0.2, 1.2]),
    np.array([True, False]),
    np.array([0.0, np.nan]),
    np.array([0.0, np.inf]),
    np.array(["0", "1"]),
    np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64),
    np.array([[0, 1]]),
]


@pytest.mark.parametrize(
    "bad_indices",
    _BAD_CUSTOM_FOLD_INDICES,
    ids=[
        "fractional",
        "boolean",
        "nan",
        "infinity",
        "numeric-string",
        "uint64-overflow",
        "two-dimensional",
    ],
)
def test_penalized_cox_cv_rejects_malformed_folds_before_candidate_fit(
    bad_indices, monkeypatch
):
    X, y = _survival_sample(seed=8120, n=18)
    candidate_calls = []

    def unexpected_candidate_fit(self, *args, **kwargs):
        candidate_calls.append(True)
        raise AssertionError("candidate fit must not run for malformed folds")

    monkeypatch.setattr(PenalizedCoxPHModel, "fit", unexpected_candidate_fit)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=2,
        cv_splits=[(bad_indices, np.arange(9, 18, dtype=np.int64))],
        device="cpu",
    )
    with pytest.raises(ValueError, match="indices"):
        model.fit(X, y)
    assert candidate_calls == []
    assert model._fitted is False
    assert model.alpha_ is None


def test_shared_cv_index_coercion_preserves_only_exact_integer_values():
    converted = _coerce_cv_indices(
        np.array([0.0, 2.0]), fold_idx=0, name="train"
    )
    np.testing.assert_array_equal(converted, np.array([0, 2], dtype=np.int64))
    assert converted.dtype == np.int64
    with pytest.raises(ValueError, match="integers"):
        _coerce_cv_indices(["0", "2"], fold_idx=0, name="train")


def test_penalized_cox_cv_accepts_general_disjoint_time_series_splits():
    sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
    X, y = _survival_sample(seed=8121, n=24)
    folds = list(
        sklearn_model_selection.TimeSeriesSplit(n_splits=3).split(X)
    )
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=3,
        cv_splits=folds,
        device="cpu",
        max_iter=200,
        tol=1e-7,
    ).fit(X, y)

    assert model.alpha_ == pytest.approx(0.1)
    assert len(model.cv_results_["fold_indices"]) == len(folds)
    for (actual_train, actual_validation), (train, validation) in zip(
        model.cv_results_["fold_indices"], folds
    ):
        np.testing.assert_array_equal(actual_train, train)
        np.testing.assert_array_equal(actual_validation, validation)


@pytest.mark.parametrize("alias", ["none", "null", ""])
def test_penalized_cox_cv_rejects_non_tunable_no_penalty_aliases(
    alias, monkeypatch
):
    X, y = _survival_sample(seed=8122, n=18)
    candidate_calls = []

    def unexpected_candidate_fit(self, *args, **kwargs):
        candidate_calls.append(True)
        raise AssertionError("no-penalty CV must fail before candidate fit")

    monkeypatch.setattr(PenalizedCoxPHModel, "fit", unexpected_candidate_fit)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty=alias,
        alpha_grid=[1.0, 0.1],
        cv=2,
        device="cpu",
    )
    with pytest.raises(ValueError, match="non-tunable"):
        model.fit(X, y)
    assert candidate_calls == []
    assert model.alpha_ is None
    assert model._fitted is False


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize("penalty_case", ["string", "object", "pure-l2"])
def test_elasticnet_auto_grid_starts_at_independent_zero_model_kkt(
    backend_name, penalty_case
):
    X, y = _survival_sample(seed=8123, n=22)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    if penalty_case == "string":
        penalty = "elasticnet"
        estimator_l1_ratio = 0.4
        expected_l1_ratio = 0.4
    elif penalty_case == "object":
        penalty = ElasticNetPenalty(alpha=7.0, l1_ratio=0.25)
        estimator_l1_ratio = 0.8
        expected_l1_ratio = 0.25
    else:
        penalty = "elasticnet"
        estimator_l1_ratio = 0.0
        expected_l1_ratio = 0.0

    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty=penalty,
        n_alphas=3,
        l1_ratio=estimator_l1_ratio,
        cv=2,
        random_state=4,
        device=device,
        max_iter=400,
        tol=1e-7,
        loss_kwargs={"ties": "efron"},
    ).fit(Xb, yb)

    resolved_backend = {
        "numpy": ("numpy", "cpu"),
        "cupy": ("cupy", "cuda"),
        "torch": ("torch", "cuda"),
    }[backend_name]
    backend = get_backend(
        backend=resolved_backend[0], device=resolved_backend[1]
    )
    zero = backend.zeros((X.shape[1],), dtype=backend.float64)
    reference_loss = CoxPartialLikelihoodLoss(ties="efron")
    try:
        gradient = reference_loss.gradient(Xb, yb, zero)
    finally:
        reference_loss.release_fit_cache()
    raw_zero_score = _to_float_scalar(
        backend.xp.max(backend.xp.abs(gradient))
    )
    expected_alpha_max = (
        raw_zero_score / expected_l1_ratio
        if expected_l1_ratio > 0.0
        else raw_zero_score
    )

    assert model.alpha_grid_[0] == pytest.approx(
        expected_alpha_max, rel=1e-11, abs=1e-12
    )
    assert model.cv_results_["alpha_grid_l1_ratio"] == pytest.approx(
        expected_l1_ratio
    )
    expected_rule = (
        "elasticnet_zero_score_kkt"
        if expected_l1_ratio > 0.0
        else "zero_score_l2_heuristic"
    )
    assert model.cv_results_["alpha_grid_rule"] == expected_rule
    if expected_l1_ratio > 0.0:
        assert (
            model.alpha_grid_[0] * expected_l1_ratio
            >= raw_zero_score - 1e-12
        )


@pytest.mark.parametrize("fold_count", [1, 4])
def test_penalized_cox_cv_passes_normalized_custom_fold_count(
    fold_count, monkeypatch
):
    X, y = _survival_sample(seed=8126, n=24)
    folds = [
        (
            np.arange(0, 12, dtype=np.int64),
            np.array([12 + fold_index], dtype=np.int64),
        )
        for fold_index in range(fold_count)
    ]
    observed_fold_counts = []

    def capture_device(self, X_value, penalty_name, n_alphas, *, n_folds=None):
        observed_fold_counts.append(n_folds)
        return "cpu"

    monkeypatch.setattr(
        PenalizedGLM_CV, "_effective_cv_device", capture_device
    )
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=9,
        cv_splits=folds,
        device="auto",
        max_iter=200,
        tol=1e-7,
    ).fit(X, y)

    assert observed_fold_counts == [fold_count]
    assert len(model.cv_results_["fold_indices"]) == fold_count


def test_effective_cv_device_uses_actual_fold_count_at_break_even(monkeypatch):
    model = PenalizedGLM_CV(
        loss="cox_ph", penalty="l2", n_alphas=100, cv=99, device="auto"
    )
    availability_calls = []

    def availability(name):
        availability_calls.append(name)
        return name == "torch"

    monkeypatch.setattr(
        penalized_cv_module, "_cuda_backend_available", availability
    )
    monkeypatch.setattr(
        penalized_cv_module, "_SMALL_PROBLEM_THRESHOLD", 200_000
    )
    monkeypatch.setattr(
        penalized_cv_module, "_GPU_BREAK_EVEN_THRESHOLD", 100_000_000
    )
    X = np.empty((2000, 100), dtype=np.float64)

    assert (
        model._effective_cv_device(
            X, "l2", 100, n_folds=1
        )
        == "cpu"
    )
    assert availability_calls == []
    assert (
        model._effective_cv_device(
            X, "l2", 100, n_folds=5
        )
        == "torch"
    )
    assert availability_calls == ["torch"]


def test_large_auto_cox_cv_falls_back_when_cuda_backends_are_unavailable(
    monkeypatch
):
    X, y = _survival_sample(seed=8124, n=20)
    availability_calls = []

    class ImportableButUnavailableBackend:
        def is_available(self):
            return False

    def unavailable_backend(backend, device):
        availability_calls.append((backend, device))
        return ImportableButUnavailableBackend()

    monkeypatch.setattr(
        cv_base_module, "get_backend", unavailable_backend
    )
    monkeypatch.setattr(penalized_cv_module, "_SMALL_PROBLEM_THRESHOLD", 0)
    monkeypatch.setattr(penalized_cv_module, "_GPU_BREAK_EVEN_THRESHOLD", 0)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=2,
        device="auto",
        max_iter=200,
        tol=1e-7,
    ).fit(X, y)

    assert availability_calls == [("torch", "cuda"), ("cupy", "cuda")]
    assert model.cv_selected_device_ == "cpu"
    assert model.cv_results_["cv_selected_device_"] == "cpu"


def test_cuda_backend_health_check_rejects_importable_unavailable_cupy(
    monkeypatch,
):
    class ImportableButUnavailableBackend:
        def is_available(self):
            return False

    calls = []

    def backend_factory(backend, device):
        calls.append((backend, device))
        return ImportableButUnavailableBackend()

    monkeypatch.setattr(cv_base_module, "get_backend", backend_factory)
    assert cv_base_module._cuda_backend_available("cupy") is False
    assert calls == [("cupy", "cuda")]


def test_large_auto_cox_cv_selects_only_operational_cupy(monkeypatch):
    model = PenalizedGLM_CV(
        loss="cox_ph", penalty="l2", n_alphas=300, cv=2, device="auto"
    )
    calls = []

    def availability(name):
        calls.append(name)
        return name == "cupy"

    monkeypatch.setattr(
        penalized_cv_module, "_cuda_backend_available", availability
    )
    X = np.empty((2000, 100), dtype=np.float64)
    assert model._effective_cv_device(X, "l2", 300) == "cuda"
    assert calls == ["torch", "cupy"]


def test_explicit_cuda_cox_cv_propagates_unavailable_backend(monkeypatch):
    X, y = _survival_sample(seed=8125, n=18)

    class UnavailableCupyBackend:
        float64 = np.float64

        def asarray(self, *args, **kwargs):
            raise RuntimeError("explicit CuPy backend unavailable sentinel")

    def unavailable_backend(backend, device):
        assert backend == "cupy"
        assert device == "cuda"
        return UnavailableCupyBackend()

    monkeypatch.setattr(cox_cv_module, "get_backend", unavailable_backend)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=2,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="explicit CuPy backend unavailable"):
        model.fit(X, y)
    assert model.cv_selected_device_ is None
    assert model._fitted is False

def test_penalized_cox_cv_rejects_dictionary_target_before_candidate_fit():
    X, y = _survival_sample(seed=8112, n=18)
    with pytest.raises(ValueError, match="dictionary targets are not supported"):
        PenalizedGLM_CV(
            loss="cox_ph",
            penalty="l2",
            alpha_grid=[0.1],
            cv=2,
            device="cpu",
        ).fit(X, {"time": y[:, 0], "event": y[:, 1]})


@pytest.mark.parametrize("n_alphas", [True, 1.5, 0])
def test_penalized_cox_cv_requires_positive_integer_n_alphas(n_alphas):
    X, y = _survival_sample(seed=8111, n=18)
    with pytest.raises(ValueError, match="n_alphas must be a positive integer"):
        PenalizedGLM_CV(
            loss="cox_ph",
            penalty="l2",
            n_alphas=n_alphas,
            cv=2,
            device="cpu",
        ).fit(X, y)


def test_penalized_cox_cv_all_candidate_failures_are_transactional(monkeypatch):
    X, y = _survival_sample(seed=8104, n=18)

    def numerical_failure(self, *args, **kwargs):
        raise FloatingPointError("candidate sentinel")

    monkeypatch.setattr(PenalizedCoxPHModel, "fit", numerical_failure)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l1",
        alpha_grid=[0.2, 0.05],
        cv=2,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="no alpha with finite evidence"):
        model.fit(X, y)
    assert model.alpha_ is None
    assert model.best_score_ is None
    assert model.estimator_ is None
    assert model.coef_ is None
    assert model._fitted is False


def test_penalized_cox_cv_excludes_nonconverged_candidates(monkeypatch):
    from statgpu.solvers import ConvergenceWarning

    X, y = _survival_sample(seed=8115, n=18)

    def nonconverged(self, *args, **kwargs):
        warnings.warn("solver sentinel", ConvergenceWarning)
        return self

    monkeypatch.setattr(PenalizedCoxPHModel, "fit", nonconverged)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l1",
        alpha_grid=[0.2, 0.05],
        cv=2,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="no alpha with finite evidence"):
        model.fit(X, y)
    assert model.alpha_ is None
    assert model.estimator_ is None
    assert model._fitted is False

def test_penalized_cox_cv_propagates_unexpected_candidate_errors(monkeypatch):
    X, y = _survival_sample(seed=8113, n=18)

    def runtime_failure(self, *args, **kwargs):
        raise RuntimeError("backend sentinel")

    monkeypatch.setattr(PenalizedCoxPHModel, "fit", runtime_failure)
    model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=2,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="backend sentinel"):
        model.fit(X, y)
    assert model.alpha_ is None
    assert model.estimator_ is None
    assert model._fitted is False

def test_scalar_glm_cv_all_nonfinite_scores_hard_fail(monkeypatch):
    X = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = np.linspace(-1.0, 1.0, 12)
    model = PenalizedGLM_CV(
        loss="squared_error",
        penalty="l1",
        alpha_grid=[0.2, 0.05],
        cv=2,
        device="cpu",
    )

    def all_nan(self, X, y, alpha_grid, cv_device, folds, **kwargs):
        return np.full((len(folds), len(alpha_grid)), np.nan)

    monkeypatch.setattr(
        model, "_compute_cv_scores", types.MethodType(all_nan, model)
    )
    with pytest.raises(RuntimeError, match="no finite candidate score"):
        model.fit(X, y)
    assert model.alpha_ is None
    assert model.estimator_ is None
    assert model._fitted is False


def test_scalar_glm_cv_does_not_ignore_infinite_fold_score(monkeypatch):
    X = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = np.linspace(-1.0, 1.0, 12)
    alpha_grid = np.array([0.2, 0.05])
    model = PenalizedGLM_CV(
        loss="squared_error",
        penalty="l1",
        alpha_grid=alpha_grid,
        cv=2,
        device="cpu",
    )

    def mixed_scores(self, X, y, alpha_grid, cv_device, folds, **kwargs):
        return np.array([[0.1, 0.2], [np.inf, 0.3]])

    monkeypatch.setattr(
        model, "_compute_cv_scores", types.MethodType(mixed_scores, model)
    )
    model.fit(X, y)
    assert model.alpha_ == pytest.approx(0.05)
    assert np.isinf(model.cv_results_["mean_score"][0])
    assert np.isfinite(model.cv_results_["mean_score"][1])

def test_penalized_cox_cv_rejects_scalar_response_controls():
    X, y = _survival_sample(seed=8105, n=18)
    with pytest.raises(NotImplementedError, match="cv_strategy='strict'"):
        PenalizedGLM_CV(
            loss="cox_ph",
            penalty="l2",
            alpha_grid=[0.1],
            cv=2,
            cv_strategy="two_stage",
            acknowledge_approx=True,
            device="cpu",
        ).fit(X, y)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        PenalizedGLM_CV(
            loss="cox_ph",
            penalty="l2",
            alpha_grid=[0.1],
            cv=2,
            device="cpu",
        ).fit(X, y, sample_weight=np.ones(X.shape[0]))


def test_composite_penalty_clone_contract_for_penalty_and_container():
    sklearn = pytest.importorskip("sklearn.base")
    penalty = CompositePenalty(
        [L1Penalty(alpha=0.1), L2Penalty(alpha=0.2)],
        weights=[0.25, 0.75],
    )
    shallow = penalty.get_params(deep=False)
    reconstructed = CompositePenalty(**shallow)
    # sklearn <=1.2 recursively clones shallow params and then requires the
    # constructor to retain those exact objects. Exercise that identity gate
    # even when the local sklearn uses the newer __sklearn_clone__ hook.
    legacy_penalties = tuple(component for component in shallow["penalties"])
    legacy_weights = tuple(float(weight) for weight in shallow["weights"])
    legacy_reconstructed = CompositePenalty(
        penalties=legacy_penalties,
        weights=legacy_weights,
    )
    cloned = sklearn.clone(penalty)
    container = PenalizedGeneralizedLinearModel(
        penalty=penalty, device="cpu", compute_inference=False
    )
    cloned_container = sklearn.clone(container)

    assert set(shallow) == {"penalties", "weights"}
    assert all(not isinstance(item, str) for item in shallow["penalties"])
    assert legacy_reconstructed.penalties is legacy_penalties
    assert legacy_reconstructed.weights is legacy_weights
    assert reconstructed.get_params() == penalty.get_params()
    assert cloned.get_params() == penalty.get_params()
    assert cloned_container.penalty.get_params() == penalty.get_params()


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_auto_cox_predictions_remain_on_fitted_backend(backend_name):
    X, y = _survival_sample(seed=8106, n=22)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    configured = "cpu" if backend_name == "numpy" else device
    expected_effective = configured
    expected_backend = {
        "numpy": "numpy",
        "cupy": "cupy",
        "torch": "torch",
    }[backend_name]
    set_device(configured)
    try:
        model = CoxPH(
            device="auto", compute_inference=False, compute_cindex=False
        ).fit(Xb, yb)
        assert model.effective_device_ == expected_effective
        assert model._fitted_backend_name == expected_backend

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_device("torch" if backend_name == "numpy" else "cpu")
        risk = model.predict_risk_score(X[:3])
        assert type(risk).__module__.startswith(expected_backend)
        assert np.all(np.isfinite(_as_numpy(risk)))
        assert np.isfinite(model.score(X[:8], y[:8]))
    finally:
        set_device("auto")


def test_failed_auto_cox_refit_clears_fitted_backend(monkeypatch):
    X, y = _survival_sample(seed=8107, n=20)
    set_device("cpu")
    try:
        model = CoxPH(
            device="auto", compute_inference=False, compute_cindex=False
        ).fit(X, y)
        assert model._fitted_backend_name == "numpy"

        def fail(*args, **kwargs):
            raise RuntimeError("refit sentinel")

        monkeypatch.setattr(model, "_fit_counting_process_dispatch", fail)
        with pytest.raises(RuntimeError, match="refit sentinel"):
            model.fit(X, y)
        assert model._fitted_backend_name is None
        assert model.effective_device_ is None
        assert model.coef_ is None
    finally:
        set_device("auto")


@pytest.mark.parametrize(
    "name,value_factory",
    [
        ("time", lambda n, time, event: time[:, None]),
        ("event", lambda n, time, event: event[None, :]),
        ("entry", lambda n, time, event: np.zeros((n, 1))),
        ("cluster", lambda n, time, event: np.zeros((n, 1))),
        ("strata", lambda n, time, event: np.zeros((1, n))),
        ("subject_id", lambda n, time, event: np.arange(n)[:, None]),
    ],
)
def test_coxphcv_side_array_shape_rejected_before_candidate_fit(
    name, value_factory, monkeypatch
):
    from statgpu.survival import _cox_cv as cox_cv

    X, y = _survival_sample(seed=8108, n=16)
    calls = []

    def forbidden_fit(self, *args, **kwargs):
        calls.append(1)
        raise AssertionError("candidate fit must not run")

    monkeypatch.setattr(cox_cv.CoxPH, "fit", forbidden_fit)
    kwargs = {
        "X": X,
        "time": y[:, 0],
        "event": y[:, 1],
        "penalties": [0.1],
        "cv_folds": 2,
        "device": "cpu",
    }
    kwargs[name] = value_factory(X.shape[0], y[:, 0], y[:, 1])
    with pytest.raises(ValueError, match=rf"{name} must have shape"):
        cox_cv._select_coxph_penalty_cv(**kwargs)
    assert calls == []