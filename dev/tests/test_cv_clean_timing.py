"""CPU/synthetic tests for separate diagnostic and uninstrumented CV timing.

Accelerator doubles validate provenance and synchronization only. These tests
never initialize CUDA, run GPU work, or claim measured performance evidence.
"""
from contextlib import nullcontext
import copy
import sys
import threading
from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dev.benchmarks import benchmark_cv_coverage as coverage
from dev.benchmarks import benchmark_cv_models as runner

SPEC = runner.CASE_SPECS[0]
SOURCE = {"git_commit": "fixture-head", "content_sha256": "fixture-content"}


def sample():
    return {"seed": 17, "workload_identity": {"sha256": "fixture-workload"},
            "selected_parameters": {"alpha": 0.1}, "validation_score": 0.25,
            "final_score": 0.0, "coefficients": [1.0, 1.0],
            "coefficient_dtype": "float64", "n_iter": 1}


@pytest.fixture
def clean_fixture(monkeypatch):
    events = []
    data = (np.ones((6, 2)), np.full(6, 2.0), np.ones((4, 2)), np.full(4, 2.0))
    diagnostic = sample()

    class Model:
        coef_ = np.ones(2)
        alpha_ = 0.1
        best_score_ = 0.25
        n_iter_ = 1
        def fit(self, X, y):
            events.append("fit")
            assert sys.getprofile() is None
            return self
        def predict(self, X):
            events.append("score")
            return X @ self.coef_

    model = Model()
    def prepare(*args):
        events.append("prepare-and-clear-cache")
        return model, data, copy.deepcopy(diagnostic["workload_identity"])
    def synchronize(backend, device):
        events.append(("resolve-sync", backend, device))
        return lambda: events.append(("sync", device))
    timestamps = iter([10.0, 10.125])
    def clock():
        events.append("clock")
        return next(timestamps)
    def source():
        events.append("source")
        return SOURCE.copy()
    monkeypatch.setattr(runner, "_prepare_fit", prepare)
    monkeypatch.setattr(runner, "_synchronizer", synchronize)
    monkeypatch.setattr(runner.time, "perf_counter", clock)
    return SimpleNamespace(events=events, diagnostic=diagnostic, model=model,
                           data=data, source=source)


def execute(fixture, **overrides):
    kwargs = {"diagnostic": fixture.diagnostic, "executed_device": "cpu", "check_source": fixture.source}
    kwargs.update(overrides)
    return runner._run_clean_once(SPEC, "numpy", 17, 6, 2, **kwargs)


def test_clean_clock_contains_only_public_fit_and_post_sync(clean_fixture, monkeypatch):
    def forbidden_profiler(*args, **kwargs):
        raise AssertionError("RegionProfiler must never be constructed for clean timing")
    monkeypatch.setattr(runner, "RegionProfiler", forbidden_profiler)
    result = execute(clean_fixture)
    assert result["total_fit_ms"] == 125.0
    assert result["instrumentation"] == "none"
    assert result["synchronized_device"] is None
    assert "cv_evaluation_ms" not in result and "final_refit_ms" not in result
    events = clean_fixture.events
    first = events.index("clock")
    last = events.index("clock", first + 1)
    assert events[first:last + 1] == ["clock", "fit", ("sync", "cpu"), "clock"]
    assert events[first - 1] == ("sync", "cpu")
    assert events.index("prepare-and-clear-cache") < first
    assert events.index("score") > last
    assert events[-1] == "source"


@pytest.mark.parametrize("module,attribute", [
    (sys, "getprofile"), (sys, "gettrace"),
    (threading, "getprofile"), (threading, "gettrace"),
])
def test_existing_profile_or_trace_fails_before_work(clean_fixture, monkeypatch, module, attribute):
    sentinel = object()
    monkeypatch.setattr(module, attribute, lambda: sentinel)
    with pytest.raises(RuntimeError, match="profiling/tracing"):
        execute(clean_fixture)
    assert clean_fixture.events == []
    assert getattr(module, attribute)() is sentinel


@pytest.mark.parametrize("monitoring_exists", [False, True])
def test_monitoring_tool_fails_closed(clean_fixture, monkeypatch, monitoring_exists):
    # Python <3.12 has no sys.monitoring; exercise both feature states without
    # requiring a real monitoring tool or altering the runner's fail-closed gate.
    if monitoring_exists:
        monkeypatch.setattr(sys, "monitoring", SimpleNamespace(get_tool=lambda i: None), raising=False)
    else:
        monkeypatch.delattr(sys, "monitoring", raising=False)
    monkeypatch.setattr(sys, "monitoring", SimpleNamespace(get_tool=lambda i: "coverage" if i == 2 else None), raising=False)
    with pytest.raises(RuntimeError, match="sys.monitoring"):
        execute(clean_fixture)
    assert clean_fixture.events == []


def test_missing_monitoring_api_allows_clean_uninstrumented_fit(clean_fixture, monkeypatch):
    monkeypatch.delattr(sys, "monitoring", raising=False)
    result = execute(clean_fixture)
    assert result["instrumentation"] == "none"
    assert result["total_fit_ms"] == 125.0
    assert not hasattr(sys, "monitoring")


def test_profiler_leaked_by_fit_rejects_sample(clean_fixture, monkeypatch):
    original = clean_fixture.model.fit
    def leaking_fit(*args):
        original(*args)
        monkeypatch.setattr(sys, "getprofile", lambda: object())
    clean_fixture.model.fit = leaking_fit
    with pytest.raises(RuntimeError, match="profiling/tracing"):
        execute(clean_fixture)


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_concrete_device_synchronization_not_default_stream(monkeypatch, backend):
    seen = []
    if backend == "cupy":
        class Device:
            def __init__(self, ordinal): self.ordinal = ordinal
            def synchronize(self): seen.append(self.ordinal)
        module = SimpleNamespace(cuda=SimpleNamespace(Device=Device,
            runtime=SimpleNamespace(getDevice=lambda: 9)))
    else:
        module = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 9,
            synchronize=lambda *, device: seen.append(device)))
    monkeypatch.setitem(sys.modules, backend, module)
    sync = runner._synchronizer(backend, "cuda:3")
    sync(); sync()
    assert seen == [3, 3]
    default_sync = runner._synchronizer(backend)
    default_sync()
    assert seen[-1] == 9


@pytest.mark.parametrize("backend,device", [("numpy", "cuda:1"), ("torch", "cuda"), ("cupy", "cpu")])
def test_synchronizer_rejects_unresolved_or_mismatched_device(backend, device):
    with pytest.raises(ValueError):
        runner._synchronizer(backend, device)


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_clean_fit_uses_observed_nonzero_device(monkeypatch, clean_fixture, backend):
    class Array:
        device = SimpleNamespace(id=3, index=3)
        is_cuda = True
    monkeypatch.setitem(sys.modules, backend, SimpleNamespace(ndarray=Array, Tensor=Array,
        cuda=SimpleNamespace(Device=lambda index: nullcontext(), device=lambda index: nullcontext())))
    converted = tuple(Array() for _ in range(4))
    monkeypatch.setattr(runner, "_prepare_fit", lambda *args: (
        clean_fixture.model, converted, clean_fixture.diagnostic["workload_identity"]))
    monkeypatch.setattr(runner, "_fit_result", lambda *args: sample())
    result = runner._run_clean_once(SPEC, backend, 17, 6, 2,
        diagnostic=clean_fixture.diagnostic, executed_device="cuda:3", check_source=clean_fixture.source)
    assert result["synchronized_device"] == "cuda:3"
    assert ("resolve-sync", backend, "cuda:3") in clean_fixture.events
    assert clean_fixture.events.count(("sync", "cuda:3")) == 2


def test_wrong_input_device_fails_before_clock(clean_fixture):
    with pytest.raises(RuntimeError, match="input device"):
        execute(clean_fixture, executed_device="cuda:0")
    assert "clock" not in clean_fixture.events


def test_changed_workload_fails_before_clock(clean_fixture):
    diagnostic = {**clean_fixture.diagnostic, "workload_identity": {"sha256": "different"}}
    with pytest.raises(RuntimeError, match="workload identity"):
        execute(clean_fixture, diagnostic=diagnostic)
    assert "clock" not in clean_fixture.events


@pytest.mark.parametrize("attribute,value,match", [
    ("alpha_", 0.5, "selection"),
    ("best_score_", 0.75, "score mismatch"),
    ("best_score_", float("nan"), "Nonfinite"),
    ("coef_", np.array([0.5, 1.5]), "coefficient mismatch"),
    ("coef_", np.ones(2, dtype=np.float32), "dtype mismatch"),
    ("coef_", np.array([float("inf"), 1.0]), "Nonfinite"),
])
def test_clean_correctness_failures_never_return_timing(clean_fixture, attribute, value, match):
    setattr(clean_fixture.model, attribute, value)
    with pytest.raises((RuntimeError, AssertionError), match=match):
        execute(clean_fixture)


def test_source_change_during_fit_invalidates_sample(clean_fixture):
    sources = iter([SOURCE, {**SOURCE, "content_sha256": "changed"}])
    with pytest.raises(RuntimeError, match="Source changed"):
        execute(clean_fixture, check_source=lambda: next(sources))


def test_fit_exception_retains_no_successful_duration(clean_fixture):
    def broken(*args): raise ValueError("factorization failed")
    clean_fixture.model.fit = broken
    with pytest.raises(ValueError, match="factorization failed"):
        execute(clean_fixture)
    assert clean_fixture.events.count("clock") == 1


def test_fixture_identity_binds_data_folds_dtype_grid_settings_and_seed():
    model = runner._construct(__import__(SPEC.import_module, fromlist=[SPEC.import_name]).RidgeCV,
                              runner._model_kwargs(SPEC, "numpy", 17))
    data = runner._regression_data(17, 30, 2)
    base = runner._workload_identity(SPEC, "numpy", 17, data, model)
    assert len(base["folds"]) == 3
    assert base["requested_settings"]["gpu_cv_mixed_precision"] is False
    assert base["effective_settings"]["gpu_cv_mixed_precision"] is False
    mutated = list(data)
    mutated[0] = mutated[0].astype(np.float32)
    assert runner._workload_identity(SPEC, "numpy", 17, mutated, model)["sha256"] != base["sha256"]
    altered = copy.deepcopy(SPEC)
    altered.grid_parameters["alphas"].append(3.0)
    assert runner._workload_identity(altered, "numpy", 17, data, model)["sha256"] != base["sha256"]
    assert base["grid"] == SPEC.grid_parameters  # snapshots do not alias mutable grids
    assert runner._workload_identity(SPEC, "numpy", 18, data, model)["folds"] != base["folds"]


def test_backend_conversion_cannot_silently_change_dtype(monkeypatch):
    monkeypatch.setattr(runner, "_to_backend", lambda value, backend: value.astype(np.float32))
    with pytest.raises(RuntimeError, match="conversion changed"):
        runner._prepare_fit(SPEC, "numpy", 17, 30, 2)


def test_strict_cache_clear_does_not_call_cache_class_methods(monkeypatch):
    class Cache:
        def __init__(self): self._cache = {"stale": object()}
        def clear(self): self._cache.clear()
    module = ModuleType("statgpu.synthetic_clean_cache")
    module.CVCache = Cache
    module.CV_CACHE = Cache()
    module.DICT_CACHE = {"stale": object()}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runner._clear_cv_caches(strict=True)
    assert module.CV_CACHE._cache == {} and module.DICT_CACHE == {}


@pytest.mark.parametrize("raises", [True, False])
def test_strict_cache_clear_failure_is_not_swallowed(monkeypatch, raises):
    class BrokenCache:
        _cache = {"stale": 1}
        def clear(self):
            if raises: raise ValueError("cannot clear")
    module = ModuleType("statgpu.synthetic_bad_cache")
    module.CV_CACHE = BrokenCache()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="Cannot clear CV cache"):
        runner._clear_cv_caches(strict=True)


def proof(backend="torch", device="cuda:3"):
    return {"backend": backend, "status": "success", "device": device,
            "synchronized_device": device, "invocation": {"kind": "repeat"},
            "workload_identity": sample()["workload_identity"],
            "stage_observed": {"selector": True, "final_refit": True},
            "numerical_observations": {stage: {"native.solve": {"device": device,
                "computed_array_dtypes": {"grad": "float64", "coef": "float64"}}}
                for stage in ("selector", "final_refit")}}


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(status="failed"), "does not match"),
    (lambda p: p.update(backend="cupy"), "does not match"),
    (lambda p: p.update(workload_identity={"sha256": "bad"}), "does not match"),
    (lambda p: p.update(synchronized_device="cuda:0"), "synchronized"),
    (lambda p: p["numerical_observations"].update(selector={}), "both CV stages"),
    (lambda p: p["stage_observed"].update(final_refit=False), "both CV stages"),
    (lambda p: p["numerical_observations"]["selector"]["native.solve"].update(device="cuda:2"), "dtype mismatch"),
    (lambda p: p["numerical_observations"]["selector"]["native.solve"].update(computed_array_dtypes={"grad": "float32"}), "dtype mismatch"),
])
def test_clean_series_rejects_unmatched_native_proof(mutation, match):
    record = proof(); mutation(record)
    with pytest.raises(RuntimeError, match=match):
        coverage._matched_execution_device("torch", [sample()], [record])


def test_missing_or_mixed_device_proof_rejected():
    with pytest.raises(RuntimeError, match="one native proof"):
        coverage._matched_execution_device("torch", [sample()], [])
    with pytest.raises(RuntimeError, match="changed executed device"):
        coverage._matched_execution_device("torch", [sample(), sample()], [proof(), proof(device="cuda:2")])
    assert coverage._matched_execution_device("torch", [sample()], [proof()]) == "cuda:3"


def series_args(**overrides):
    return SimpleNamespace(repeats=2, warmup=1, seed=17, features=2, **overrides)


def test_clean_series_keeps_raw_samples_and_never_blesses_performance(monkeypatch):
    calls = []
    monkeypatch.setattr(coverage, "source_identity", lambda: SOURCE.copy())
    def clean(*args, **kwargs):
        calls.append(kwargs)
        kwargs["check_source"]()
        return {**sample(), "total_fit_ms": float(len(calls)), "source": SOURCE}
    monkeypatch.setattr(runner, "_run_clean_once", clean)
    result = coverage.clean_public_timing(SPEC, "torch", series_args(), 6,
                                          [sample(), sample()], [proof(), proof()], SOURCE)
    assert result["status"] == "pass" and result["instrumented"] is False
    assert result["raw_seconds"] == [0.002, 0.003]
    assert result["timing"]["median_seconds"] == 0.0025
    assert len(calls) == 3 and {call["executed_device"] for call in calls} == {"cuda:3"}
    assert result["performance_eligible"] is False and result["comparisons"] == []
    assert result["performance_status"] == "UNVERIFIED_EXCLUSIVITY"


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_failed_warmup_or_repeat_discards_partial_timing(monkeypatch, failure_call):
    calls = []
    def clean(*args, **kwargs):
        calls.append(1)
        if len(calls) == failure_call: raise RuntimeError("selected proof failed")
        return {**sample(), "total_fit_ms": 1.0}
    monkeypatch.setattr(runner, "_run_clean_once", clean)
    result = coverage.clean_public_timing(SPEC, "numpy", series_args(), 6,
                                          [sample(), sample()], [], SOURCE)
    assert result["status"] == "failed" and "selected proof failed" in result["reason"]
    assert result["timing"] is None and result["raw_seconds"] == [] and result["repeat_samples"] == []


def test_source_changed_before_clean_series_blocks_fit(monkeypatch):
    monkeypatch.setattr(coverage, "source_identity", lambda: {**SOURCE, "git_commit": "changed"})
    def clean(*args, **kwargs):
        kwargs["check_source"]()
        raise AssertionError("must stop before fit")
    monkeypatch.setattr(runner, "_run_clean_once", clean)
    result = coverage.clean_public_timing(SPEC, "numpy", series_args(), 6,
                                          [sample(), sample()], [], SOURCE)
    assert result["status"] == "failed" and "Source changed" in result["reason"]
    assert result["raw_seconds"] == []


def test_prepare_reconstructs_and_clears_cache_each_time(monkeypatch):
    models, cleared = [], []
    original_construct = runner._construct
    def construct(*args):
        model = original_construct(*args)
        models.append(model)
        return model
    monkeypatch.setattr(runner, "_construct", construct)
    monkeypatch.setattr(runner, "_clear_cv_caches", lambda *, strict: cleared.append(strict))
    runner._prepare_fit(SPEC, "numpy", 17, 30, 2)
    runner._prepare_fit(SPEC, "numpy", 17, 30, 2)
    assert len(models) == 2 and models[0] is not models[1]
    assert cleared == [True, True]


@pytest.fixture
def synthetic_adapter(monkeypatch):
    """Exercise real orchestration with no numerical fits or GPU imports."""
    from contextlib import contextmanager
    active, events = [], []
    @contextmanager
    def observation(backend):
        active.append(backend)
        events.append("diagnostic-enter")
        try:
            yield []
        finally:
            active.pop()
            events.append("diagnostic-exit")
    def diagnostic(*args, **kwargs):
        assert active
        events.append("diagnostic-fit")
        return {**sample(), "cv_evaluation_ms": 2.0, "final_refit_ms": 1.0,
                "total_fit_ms": 5.0}
    def clean(*args, **kwargs):
        assert not active, "clean fitting must not run within observed_backend"
        events.append("clean-fit")
        kwargs["check_source"]()
        return {**sample(), "total_fit_ms": 0.5, "source": SOURCE}
    monkeypatch.setattr(coverage, "observed_backend", observation)
    monkeypatch.setattr(coverage, "source_identity", lambda: SOURCE.copy())
    monkeypatch.setattr(runner, "_backend_status", lambda backend: (True, backend, None))
    monkeypatch.setattr(runner, "_package_version", lambda name: "fixture")
    monkeypatch.setattr(runner, "_run_once", diagnostic)
    monkeypatch.setattr(runner, "_run_clean_once", clean)
    monkeypatch.setattr(runner, "build_sklearn_reference", lambda *args: None)
    monkeypatch.setattr(coverage, "direct_refit_check", lambda *args: {
        "status": "pass", "selected_parameters": sample()["selected_parameters"],
        "reference_final_score": 0.0, "reference_validation_score": 0.25})
    args = SimpleNamespace(sizes=[30], backends=["numpy"], seed=17, features=2,
                           repeats=1, warmup=0, smoke=False)
    return args, events


def test_adapter_emits_disjoint_diagnostic_and_clean_sections(synthetic_adapter):
    args, events = synthetic_adapter
    result = coverage.run(args)
    assert result["schema_version"] == "statgpu-scoped-cv-candidate-v2"
    assert result["validation"]["status"] == "complete"
    assert result["validation"]["performance_eligible"] is False
    assert events.index("diagnostic-exit") < events.index("clean-fit")
    assert events.count("clean-fit") == events.count("diagnostic-fit") == 6
    for row in result["results"]:
        assert row["status"] == "pass" and row["performance_valid"] is False
        assert row["diagnostic_timing"]["instrumented"] is True
        assert row["clean_public_fit"]["raw_seconds"] == [0.0005]
        assert row["clean_public_fit"]["instrumented"] is False
        native, = row["case"]["runs"]
        assert native["timing"]["total_fit_ms"] == 5.0
        # New workload/output evidence does not leak into immutable cv_source rows.
        assert "workload_identity" not in native["repeat_samples"][0]
        assert "coefficients" not in native["repeat_samples"][0]


def test_adapter_clean_failure_keeps_diagnostic_proof_but_not_durations(synthetic_adapter, monkeypatch):
    args, _ = synthetic_adapter
    def fail(*args, **kwargs): raise RuntimeError("clean reference mismatch")
    monkeypatch.setattr(runner, "_run_clean_once", fail)
    result = coverage.run(args)
    assert result["validation"]["status"] == "failed"
    for row in result["results"]:
        assert row["status"] == "failed"
        assert "clean reference mismatch" in row["reason"]
        assert row["clean_public_fit"]["raw_seconds"] == []
        assert row["clean_public_fit"]["timing"] is None
        assert row["case"]["runs"][0]["timing"] is None
        assert row["case"]["runs"][0]["repeat_samples"] == []


def test_final_source_drift_invalidates_all_previously_clean_samples(synthetic_adapter, monkeypatch):
    args, _ = synthetic_adapter
    identity_reads = []
    def identity():
        identity_reads.append(1)
        return {**SOURCE, "content_sha256": "changed" if len(identity_reads) > 1 else SOURCE["content_sha256"]}
    monkeypatch.setattr(coverage, "source_identity", identity)
    monkeypatch.setattr(coverage, "clean_public_timing", lambda *args: {
        "status": "pass", "timing": {"median_seconds": 0.5}, "raw_seconds": [0.5],
        "repeat_samples": [sample()], "performance_eligible": False})
    result = coverage.run(args)
    assert result["validation"]["source_changed"] is True
    for row in result["results"]:
        assert row["status"] == "failed"
        assert row["clean_public_fit"]["raw_seconds"] == []
        assert row["clean_public_fit"]["timing"] is None
        assert row["case"]["runs"][0]["timing"] is None


def test_reference_failure_blocks_clean_fit(synthetic_adapter, monkeypatch):
    args, events = synthetic_adapter
    monkeypatch.setattr(coverage, "direct_refit_check", lambda *args: (_ for _ in ()).throw(
        AssertionError("independent reference failed")))
    result = coverage.run(args)
    assert "clean-fit" not in events
    for row in result["results"]:
        assert row["status"] == "failed"
        assert row["clean_public_fit"]["status"] == "not_run"
        assert row["clean_public_fit"]["timing"] is None


def test_convergence_warning_is_failed_clean_series(monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    def nonconverged(*args, **kwargs):
        warnings.warn("iteration budget exhausted", ConvergenceWarning)
        return {**sample(), "total_fit_ms": 1.0}
    monkeypatch.setattr(runner, "_run_clean_once", nonconverged)
    result = coverage.clean_public_timing(SPEC, "numpy", series_args(), 6,
                                          [sample(), sample()], [], SOURCE)
    assert result["status"] == "failed"
    assert "iteration budget exhausted" in result["reason"]
    assert result["timing"] is None and result["raw_seconds"] == []


def test_clean_rejects_active_torch_native_profiler():
    torch = pytest.importorskip('torch')
    from dev.benchmarks import benchmark_cv_models as legacy
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with pytest.raises(RuntimeError, match='Torch native profiler'):
            legacy._require_uninstrumented()
    legacy._require_uninstrumented()


@pytest.mark.parametrize('field,value',[
    ('converged_',False),('converged_',None),('mse_path_',[1.,float('inf')]),
    ('cv_results_',{'converged_path':[[True,False]]}),
    ('cv_results_',{'fold_valid':[True,False]}),
    ('cv_results_',{'candidate_complete':[False]}),
    ('cv_results_',{'failure_path':[['failed']]}),
    ('cv_results_',{'all_scores':[float('nan')]}),
    ('cv_results_',{'mse_path':[[1.,float('nan')]]}),
    ('cv_results_',{'loss_path':[[float('inf')]]}),
])
def test_explicit_cv_nonconvergence_and_incomplete_paths_fail(field,value):
    from types import SimpleNamespace
    from dev.benchmarks import benchmark_cv_models as legacy
    with pytest.raises(RuntimeError,match='CV '):
        legacy._validate_convergence(SimpleNamespace(**{field:value}))


def test_final_refit_flag_dominates_matching_coefficients():
    from types import SimpleNamespace
    from dev.benchmarks import benchmark_cv_models as legacy
    with pytest.raises(RuntimeError,match='final_refit.converged_'):
        legacy._validate_convergence(SimpleNamespace(estimator_=SimpleNamespace(converged_=False)))


def test_real_cox_nonconverged_final_refit_is_rejected(monkeypatch):
    import warnings
    from dev.benchmarks import benchmark_cv_models as legacy
    from dev.benchmarks.benchmark_cv_coverage import convergence_errors, direct_refit_check
    from statgpu.survival import CoxPH
    spec=legacy.CASE_SPECS[-1]
    diagnostic=legacy._run_once(spec,'numpy',17,60,4)
    original=CoxPH.fit
    def underbudget(self,X,*args,**kwargs):
        if len(X)==60:
            self.set_params(max_iter=1)
        return original(self,X,*args,**kwargs)
    monkeypatch.setattr(CoxPH,'fit',underbudget)
    with warnings.catch_warnings(record=True) as caught:
        convergence_errors()
        with pytest.raises(RuntimeError,match='nonconvergence'):
            legacy._run_once(spec,'numpy',17,60,4)
        with pytest.raises(RuntimeError,match='nonconvergence'):
            direct_refit_check(spec,17,60,4)
        with pytest.raises(RuntimeError,match='nonconvergence'):
            legacy._run_clean_once(spec,'numpy',17,60,4,diagnostic=diagnostic,
                                   executed_device='cpu',check_source=lambda:{'fixed':'source'})
    assert any(issubclass(item.category,RuntimeWarning) for item in caught)


@pytest.mark.parametrize('backend',['cupy','torch'])
@pytest.mark.parametrize('model_id',['RidgeCV','LassoCV','ElasticNetCV','LogisticRegressionCV','PenalizedGLM_CV','CoxPHCV'])
def test_physical_cv_clean_lane_matches_observed_cuda_workload(backend,model_id):
    from types import SimpleNamespace
    from dev.benchmarks import benchmark_cv_models as legacy
    from dev.benchmarks import benchmark_cv_coverage as coverage
    available,_,reason=legacy._backend_status(backend)
    if not available:
        pytest.skip('Required physical CUDA backend unavailable: '+str(reason))
    spec=next(value for value in legacy.CASE_SPECS if value.model_id==model_id)
    source=coverage.source_identity()
    with coverage.observed_backend(backend) as proofs:
        diagnostic=legacy._run_once(spec,backend,17,60,4)
    args=SimpleNamespace(seed=17,features=4,warmup=0,repeats=1)
    clean=coverage.clean_public_timing(spec,backend,args,60,[diagnostic],proofs,source)
    assert clean['status']=='pass',clean['reason']
    assert clean['instrumented'] is False
    assert clean['synchronized_device']=='cuda:0'
    assert len(clean['raw_seconds'])==1
    assert clean['performance_eligible'] is False
