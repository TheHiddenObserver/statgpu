"""CPU-only regression tests for source-specific CV execution observation.

Array doubles exercise provenance validation, not physical CUDA computation.
The real-frame tests exercise Python profiling and pre-cleanup event ordering;
none of these tests runs a benchmark or imports an accelerator runtime.
"""
from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.benchmarks import benchmark_cv_coverage as coverage


# Deliberately independent of the production inventory: dropping a supported
# optimized numerical implementation must make its regression fail.
COMPUTATIONS = (
    ("statgpu.linear_model.cv._ridge_cv", "_solve_ridge_path_gpu_from_gram_eig",
     ("eigvals", "QTXty", "coefs")),
    ("statgpu.linear_model.cv._ridge_cv", "_solve_ridge_path_native_from_gram",
     ("system", "coef")),
    ("statgpu.linear_model.cv._logistic_cv", "_solve_logistic_path_gpu_from_batch",
     ("XtWX", "Xtz", "params")),
    ("statgpu.linear_model.wrappers._lasso", "_solve_lasso_path_gpu_fista_batched_from_gram",
     ("grad", "coef_new", "coefs")),
    ("statgpu.linear_model.wrappers._lasso", "_solve_lasso_path_gpu_fista_multi_fold_from_gram",
     ("grad", "coef_candidate", "coefs")),
    ("statgpu.linear_model.wrappers._lasso", "_solve_lasso_path_gpu_fista_multi_fold_from_gram_torch",
     ("grad", "coef_candidate", "coefs")),
    ("statgpu.linear_model.penalized._fit_mixin", "_fit_gpu_backend", ("xtx_y", "coef")),
    ("statgpu.linear_model.penalized._fit_mixin", "_fit_gpu_backend", ("grad", "coef")),
    ("statgpu.linear_model.penalized._fit_mixin", "_solve_exact_cupy", ("A", "_returned_solution")),
    ("statgpu.linear_model.penalized._fit_mixin", "_solve_exact_torch", ("A", "_returned_solution")),
    ("statgpu.linear_model.wrappers._logistic", "_fit_gpu", ("XtWX", "Xtz", "params")),
    ("statgpu.linear_model.wrappers._logistic", "_fit_torch", ("XtWX", "Xtz", "params")),
    ("statgpu.linear_model.penalized._penalized_cv", "_ridge_eig_single_backend",
     ("gram", "cross", "eigvals", "coef")),
    ("statgpu.solvers._newton", "newton_solver", ("grad", "hess", "params")),
    ("statgpu.solvers._fista", "fista_solver", ("grad", "coef")),
    ("statgpu.solvers._fista_bb", "fista_bb_solver", ("grad", "coef")),
    ("statgpu.losses._cox_ph", "_full_objective_from_eta_backend",
     ("score", "information", "weighted_second")),
)


@pytest.fixture(params=("cupy", "torch"))
def accelerator_double(monkeypatch, request):
    backend = request.param
    synchronizations = []

    class Array:
        def __init__(self, device=0, dtype="float64", is_cuda=True):
            self.device = SimpleNamespace(id=device, index=device)
            self.dtype = ("torch." if backend == "torch" else "") + dtype
            self.is_cuda = is_cuda
            self.shape = (4, 2)

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def synchronize(self):
            synchronizations.append(self.index)

    if backend == "cupy":
        module = SimpleNamespace(ndarray=Array, cuda=SimpleNamespace(Device=Device))
    else:
        module = SimpleNamespace(
            Tensor=Array,
            cuda=SimpleNamespace(
                device=Device,
                synchronize=lambda *, device: synchronizations.append(device),
            ),
        )
    monkeypatch.setitem(sys.modules, backend, module)
    return SimpleNamespace(backend=backend, Array=Array, synchronizations=synchronizations)


def frame_for(module, name, values=None, caller=None):
    return SimpleNamespace(
        f_globals={"__name__": module},
        f_code=SimpleNamespace(co_name=name),
        f_locals={} if values is None else values,
        f_back=caller,
    )


def profiler_for(outer=None):
    profiler = coverage.legacy.RegionProfiler(outer or object(), 4, lambda: None)
    # Fake-frame tests invoke __exit__ without __enter__; preserve any caller's
    # existing profiler rather than changing unrelated instrumentation.
    profiler._previous = sys.getprofile()
    return profiler


def numerical_frame(array_type, computation=COMPUTATIONS[0]):
    module, name, names = computation
    values = {key: array_type() for key in names}
    result = values.pop("_returned_solution", None)
    return frame_for(module, name, values), result


def observe_stage(profiler, frame, result, stage):
    profiler.selector_stack = [-1.0] if stage == "selector" else []
    profiler.refit_stack = [-1.0] if stage == "final_refit" else []
    profiler._profile(frame, "return", result)


@pytest.mark.parametrize("computation", COMPUTATIONS, ids=lambda item: item[1] + ":" + item[2][0])
def test_source_specific_computed_groups_cover_both_regions(accelerator_double, computation):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend) as proofs:
        profiler = profiler_for()
        profiler.invocation = {"kind": "measured", "seed": 17}
        profiler.input_dtypes = {"X": "float32", "target": "int32"}
        frame, result = numerical_frame(fake.Array, computation)
        for stage in ("selector", "final_refit"):
            observe_stage(profiler, frame, result, stage)
        profiler.__exit__(None, None, None)
    proof, = proofs
    assert proof["status"] == "success"
    assert proof["stage_observed"] == {"selector": True, "final_refit": True}
    assert proof["invocation"] == {"kind": "measured", "seed": 17}
    assert proof["input_dtypes"] == {"X": "float32", "target": "int32"}
    call = ".".join(computation[:2])
    for stage in ("selector", "final_refit"):
        observed = proof["numerical_observations"][stage][call]
        assert observed["device"] == "cuda:0"
        assert observed["computed_array_dtypes"] == {name: "float64" for name in computation[2]}


@pytest.mark.parametrize("backend", ("numpy", "cupy", "torch"))
@pytest.mark.parametrize("model_name", ("RidgeCV", "LassoCV", "LogisticRegressionCV"))
def test_accuracy_baseline_disables_mixed_precision_in_actual_constructors(backend, model_name):
    spec = next(spec for spec in coverage.legacy.CASE_SPECS if spec.model_id == model_name)
    kwargs = coverage.legacy._model_kwargs(spec, backend, 17)
    assert kwargs["gpu_cv_mixed_precision"] is False
    cls = getattr(importlib.import_module(spec.import_module), spec.import_name)
    model = coverage.legacy._construct(cls, kwargs)
    # Construction only: this neither resolves an accelerator nor fits data.
    assert model.gpu_cv_mixed_precision is False
    assert model.get_params(deep=False)["gpu_cv_mixed_precision"] is False


@pytest.mark.parametrize("computation", COMPUTATIONS, ids=lambda item: item[1] + ":" + item[2][0])
def test_call_input_and_backend_labels_never_count_as_computation(accelerator_double, computation):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend):
        profiler = profiler_for()
        profiler.selector_stack = [-1.0]
        frame, _ = numerical_frame(fake.Array, computation)
        frame.f_locals.update(X=fake.Array(), y=fake.Array(), backend=fake.backend, device="cuda:0")
        profiler._profile(frame, "call", None)
        assert not any(profiler.observed.values())
        frame.f_locals = {"X": fake.Array(), "y": fake.Array(), "backend": fake.backend}
        profiler._profile(frame, "return", fake.Array())
        assert not any(profiler.observed.values())


@pytest.mark.parametrize("computation", COMPUTATIONS, ids=lambda item: item[1] + ":" + item[2][0])
def test_incomplete_computed_group_cannot_establish_proof(accelerator_double, computation):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend):
        profiler = profiler_for()
        profiler.selector_stack = [-1.0]
        frame, result = numerical_frame(fake.Array, computation)
        # Remove a necessary computed local, leaving the native return and
        # inputs intact so that neither can substitute for the missing proof.
        del frame.f_locals[computation[2][0]]
        frame.f_locals.update(X=fake.Array(), y=fake.Array())
        profiler._profile(frame, "return", result)
        assert not any(profiler.observed.values())


@pytest.mark.parametrize("module,name", (
    ("statgpu.solvers._unknown", "fista_solver"),
    ("statgpu.solvers._fista", "unrecognized_helper"),
    ("statgpu.survival._cox", "_prepare_metadata"),
    ("statgpu.linear_model.penalized._fit_mixin", "_fit_torch"),
))
def test_unrecognized_or_dispatch_only_frames_do_not_count(accelerator_double, module, name):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend):
        profiler = profiler_for()
        profiler.selector_stack = [-1.0]
        values = {key: fake.Array() for key in ("X", "y", "grad", "coef", "hess", "params")}
        profiler._profile(frame_for(module, name, values), "return", fake.Array())
        assert not any(profiler.observed.values())


@pytest.mark.parametrize("mutation,match", (
    ("numpy", "CPU/non-native"),
    ("wrong_device", "unexpected CUDA device"),
    ("float32", "different compute dtype"),
    ("scalar", "CPU/non-native"),
))
def test_computed_arrays_must_all_be_native_device_zero_float64(accelerator_double, mutation, match):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend) as proofs:
        profiler = profiler_for()
        profiler.selector_stack = [-1.0]
        frame, result = numerical_frame(fake.Array)
        frame.f_locals["QTXty"] = {
            "numpy": np.ones(2), "wrong_device": fake.Array(device=1),
            "float32": fake.Array(dtype="float32"), "scalar": 1.0,
        }[mutation]
        with pytest.raises(RuntimeError, match=match) as caught:
            profiler._profile(frame, "return", result)
        profiler.__exit__(type(caught.value), caught.value, caught.value.__traceback__)
    assert proofs[0]["status"] == "failed"
    assert match in proofs[0]["reason"]
    assert proofs[0]["stage_active_at_failure"]["selector"]
    rejected, = proofs[0]["rejected_numerical_observations"]
    assert rejected["stage"] == "selector"
    assert rejected["devices"]["QTXty"] == {
        "numpy": None, "wrong_device": 1, "float32": 0, "scalar": None,
    }[mutation]
    assert rejected["computed_array_dtypes"]["QTXty"] == {
        "numpy": "float64", "wrong_device": "float64",
        "float32": "float32", "scalar": "unknown",
    }[mutation]


def test_torch_cpu_tensor_is_not_cuda_evidence(monkeypatch):
    class Tensor:
        is_cuda = False
        device = SimpleNamespace(index=None)
        dtype = "torch.float64"
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=Tensor, cuda=SimpleNamespace(device=lambda index: nullcontext(), synchronize=lambda **kwargs: None),
    ))
    with coverage.observed_backend("torch"):
        profiler = profiler_for()
        profiler.selector_stack = [-1.0]
        frame, result = numerical_frame(Tensor)
        with pytest.raises(RuntimeError, match="CPU/non-native"):
            profiler._profile(frame, "return", result)


@pytest.mark.parametrize("stage", ("selector", "final_refit"))
def test_missing_stage_keeps_partial_proof_and_fails(accelerator_double, stage):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend) as proofs:
        profiler = profiler_for()
        frame, result = numerical_frame(fake.Array)
        observe_stage(profiler, frame, result, stage)
        with pytest.raises(RuntimeError, match="both CV selector and final refit"):
            profiler.__exit__(None, None, None)
    assert proofs[0]["status"] == "failed"
    assert proofs[0]["stage_observed"][stage]
    assert sum(proofs[0]["stage_observed"].values()) == 1
    assert proofs[0]["observed_numerical_calls"][stage]


def test_nested_full_row_fit_is_not_final_refit_evidence(accelerator_double):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend):
        profiler = profiler_for()
        profiler.selector_stack = profiler.refit_stack = [-1.0]
        frame, result = numerical_frame(fake.Array)
        profiler._profile(frame, "return", result)
        assert profiler.observed["selector"]
        assert not profiler.observed["final_refit"]


@pytest.mark.parametrize("module,name", (
    ("statgpu.linear_model.wrappers._logistic", "_fit_cpu"),
    ("statgpu.linear_model.penalized._fit_mixin", "_solve_exact_numpy"),
    ("statgpu.linear_model.penalized._penalized_cv", "_ridge_eig_single"),
    ("statgpu.linear_model.wrappers._lasso", "_solve_lasso_path_cpu_from_gram"),
    ("numpy.linalg._linalg", "solve"),
    ("scipy.linalg._decomp", "eigh"),
))
def test_cpu_dispatch_rejected_only_in_active_region(accelerator_double, module, name):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend):
        profiler = profiler_for()
        frame = frame_for(module, name)
        profiler._profile(frame, "call", None)
        profiler.refit_stack = [-1.0]
        with pytest.raises(RuntimeError, match="CPU fit dispatch/numerical solve"):
            profiler._profile(frame, "call", None)


@pytest.mark.parametrize("event", ("c_call", "c_return", "child_call", "child_return"))
def test_logistic_temporaries_observed_before_cleanup(accelerator_double, event):
    fake = accelerator_double
    with coverage.observed_backend(fake.backend):
        profiler = profiler_for()
        profiler.refit_stack = [-1.0]
        name = "_fit_gpu" if fake.backend == "cupy" else "_fit_torch"
        logistic = frame_for("statgpu.linear_model.wrappers._logistic", name,
                             {key: fake.Array() for key in ("XtWX", "Xtz", "params")})
        if event.startswith("child_"):
            frame = frame_for("helper", "checkpoint", caller=logistic)
            profiler._profile(frame, event.removeprefix("child_"), None)
        else:
            profiler._profile(logistic, event, None)
        logistic.f_locals.clear()
        profiler._profile(logistic, "return", None)
        assert profiler.observed["final_refit"]


def test_real_python_frames_capture_pre_cleanup_locals_and_restore_profiler(accelerator_double):
    fake = accelerator_double
    name = "_fit_gpu" if fake.backend == "cupy" else "_fit_torch"
    namespace = {"__name__": "statgpu.linear_model.wrappers._logistic", "Array": fake.Array}
    exec(compile(
        "def checkpoint():\n    return None\n"
        f"def {name}():\n"
        "    XtWX, Xtz, params = Array(), Array(), Array()\n"
        "    checkpoint()\n"
        "    del XtWX, Xtz, params\n",
        "<observer-event-fixture>", "exec"), namespace)
    previous = sys.getprofile()
    original_profiler = coverage.legacy.RegionProfiler
    original_sync = coverage.legacy._synchronizer
    with coverage.observed_backend(fake.backend) as proofs:
        profiler = profiler_for()
        with profiler:
            profiler.selector_stack = [-1.0]
            namespace[name]()
            profiler.selector_stack = []
            profiler.refit_stack = [-1.0]
            namespace[name]()
        coverage.legacy._synchronizer(fake.backend)()
    assert proofs[0]["status"] == "success"
    assert fake.synchronizations == [0]
    assert sys.getprofile() is previous
    assert coverage.legacy.RegionProfiler is original_profiler
    assert coverage.legacy._synchronizer is original_sync


def test_failure_chain_retains_partial_stage_and_invocation_metadata(accelerator_double):
    fake = accelerator_double
    previous = sys.getprofile()
    original_profiler = coverage.legacy.RegionProfiler
    original_sync = coverage.legacy._synchronizer
    with coverage.observed_backend(fake.backend) as proofs:
        profiler = profiler_for()
        profiler.invocation = {"kind": "measured", "repeat": 0}
        profiler.input_dtypes = {"X": "float64", "y": "float64"}
        with pytest.raises(RuntimeError, match="final refit failed"):
            with profiler:
                frame, result = numerical_frame(fake.Array)
                observe_stage(profiler, frame, result, "selector")
                profiler.selector_stack = []
                profiler.refit_stack = [-1.0]
                try:
                    raise ValueError("native factorization rejected")
                except ValueError as cause:
                    raise RuntimeError("final refit failed") from cause
    proof, = proofs
    assert proof["status"] == "failed"
    assert proof["stage_observed"] == {"selector": True, "final_refit": False}
    assert proof["stage_active_at_failure"] == {"selector": False, "final_refit": True}
    assert proof["invocation"] == {"kind": "measured", "repeat": 0}
    assert proof["input_dtypes"] == {"X": "float64", "y": "float64"}
    assert "ValueError: native factorization rejected" in proof["exception_chain"]
    assert "RuntimeError: final refit failed" in proof["exception_chain"]
    assert sys.getprofile() is previous
    assert coverage.legacy.RegionProfiler is original_profiler
    assert coverage.legacy._synchronizer is original_sync


def test_runner_preserves_partial_evidence_when_source_construction_fails(accelerator_double, monkeypatch):
    fake = accelerator_double
    monkeypatch.setattr(coverage, "source_identity", lambda: {"git_commit": "fixture", "content_sha256": "unchanged"})
    monkeypatch.setattr(coverage.legacy, "_backend_status", lambda backend: (True, backend, None))

    def source_failure(options):
        # This replaces source construction completely; no fit or benchmark
        # work is performed by this regression.
        model = type("RidgeCV", (), {})()
        profiler = profiler_for(model)
        with profiler:
            frame, result = numerical_frame(fake.Array)
            observe_stage(profiler, frame, result, "selector")
            raise RuntimeError("fixture construction stopped")

    monkeypatch.setattr(coverage.legacy, "build_source", source_failure)
    args = SimpleNamespace(sizes=[60], backends=[fake.backend], seed=1,
                           repeats=1, warmup=0, features=4, smoke=False)
    report = coverage.run(args)
    assert report["validation"]["status"] == "failed"
    assert len(report["results"]) == 6
    for row in report["results"]:
        assert row["status"] == "failed"
        assert row["timing"] is None
        assert "fixture construction stopped" in row["reason"]
        if row["model_id"] == "RidgeCV":
            proof, = row["execution_provenance"]
            assert proof["stage_observed"] == {"selector": True, "final_refit": False}
            assert proof["status"] == "failed"
        else:
            assert row["execution_provenance"] == []
