"""Runner contracts; tiny CPU checks are not GPU/performance acceptance."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dev.benchmarks import benchmark_classification_metrics as bench


def options(**changes):
    args = dict(backends=["numpy"], sizes=[20], dtypes=["float32", "float64"],
        scenarios=list(bench.SCENARIOS), operations=list(bench.OPERATIONS),
        device=0, seed=42, warmup=0, repeats=1, require_backends=False, smoke=True)
    args.update(changes)
    return SimpleNamespace(**args)


@pytest.mark.parametrize("scenario", bench.SCENARIOS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reference_alignment_all_public_operations(scenario, dtype):
    pytest.importorskip("sklearn")
    labels, scores = bench.make_data(40, scenario, dtype, 123)
    references = bench.references(labels, scores)
    runtime = bench.Runtime("numpy", 0)
    for operation in bench.OPERATIONS:
        row = bench.benchmark_operation(runtime, operation, labels, scores,
            references[operation], 0, 1)
        assert row["status"] == "pass"
        assert row["timing"]["seconds"] >= 0
        assert len(row["timing"]["raw_seconds"]) == 1
        assert "speedup" not in row
    assert np.isposinf(references["binary_roc_curve"][2][0])
    assert np.isposinf(references["binary_precision_recall_curve"][2][0])


def test_float32_input_is_not_float32_computation():
    pytest.importorskip("sklearn")
    labels, scores = bench.make_data(20, "imbalanced-tied", "float32", 1)
    row = bench.benchmark_operation(bench.Runtime("numpy", 0), "binary_roc_curve",
        labels, scores, bench.references(labels, scores)["binary_roc_curve"], 0, 1)
    assert row["output_dtypes"] == ["float64", "float64", "float64"]


def test_failed_correctness_suppresses_timing(monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(bench.metrics, "binary_roc_auc_score", lambda *a, **kw: -5.0)
    result = bench.run(options(operations=["binary_roc_auc_score"], scenarios=["constant-score"], dtypes=["float64"]))
    assert result["validation"]["status"] == "failed"
    assert result["results"][0]["correctness"]["status"] == "failed"
    assert "timing" not in result["results"][0]


def test_unavailable_backend_never_falls_back(monkeypatch):
    original = bench.Runtime
    def runtime(name, device):
        if name != "numpy":
            raise bench.Unavailable("fixture GPU unavailable")
        return original(name, device)
    monkeypatch.setattr(bench, "Runtime", runtime)
    pytest.importorskip("sklearn")
    result = bench.run(options(backends=["numpy", "cupy"], operations=["binary_confusion_matrix"], scenarios=["constant-score"], dtypes=["float64"]))
    assert result["validation"]["status"] == "partial"
    assert result["results"][-1] == {"backend": "cupy", "status": "skipped", "reason": "fixture GPU unavailable"}
    required = bench.run(options(backends=["cupy"], require_backends=True))
    assert required["validation"]["status"] == "failed"


def test_runtime_error_is_failure_not_skip(monkeypatch):
    def broken(*args):
        raise ValueError("implementation failure")
    monkeypatch.setattr(bench, "Runtime", broken)
    result = bench.run(options())
    assert result["validation"]["status"] == "failed"
    assert result["results"][0]["status"] == "failed"


def test_gpu_host_scalar_rejected_except_metadata():
    runtime = object.__new__(bench.Runtime)
    runtime.backend = "cupy"
    runtime.ordinal = 0
    runtime.xp = SimpleNamespace(ndarray=type("FakeArray", (), {}))
    with pytest.raises(AssertionError, match="host array"):
        runtime.verify_device(0.9)
    runtime.verify_device({"threshold": 0.5})
    with pytest.raises(AssertionError, match="host array"):
        runtime.verify_device(np.ones(2))


def test_result_strict_json_and_provenance():
    pytest.importorskip("sklearn")
    result = bench.run(options(scenarios=["constant-score"], dtypes=["float64"]))
    json.dumps(result, allow_nan=False)
    assert result["source"]["git_commit"]
    assert len(result["source"]["script_sha256"]) == 64
    assert not result["validation"]["canonical_ready"]
    assert result["validation"]["tier"] == "local-minimal"


def test_main_smoke_writes_output(tmp_path):
    pytest.importorskip("sklearn")
    target = tmp_path / "smoke.json"
    assert bench.main(["--smoke", "--output", str(target)]) == 0
    result = json.loads(target.read_text())
    assert len(result["results"]) == 42
    assert all(row["status"] == "pass" for row in result["results"])


@pytest.mark.parametrize("values", [["--sizes", "1"], ["--repeats", "0"], ["--device", "-1"]])
def test_invalid_arguments(values, tmp_path):
    with pytest.raises(SystemExit) as error:
        bench.main([*values, "--output", str(tmp_path / "invalid.json")])
    assert error.value.code == 2


def test_check_rejects_bad_sentinels_and_values():
    with pytest.raises(AssertionError):
        bench.check(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(AssertionError):
        bench.check(np.array([2.0]), np.array([1.0]))
    with pytest.raises(AssertionError):
        bench.check(np.array([1.0]), np.array([np.inf]))


def test_wrong_import_origin_rejected(monkeypatch):
    monkeypatch.setattr(bench.statgpu, "__file__", "/tmp/other-site-packages/statgpu/__init__.py")
    with pytest.raises(RuntimeError, match="not from this checkout"):
        bench.source_identity()


def test_untracked_runner_is_overlay_not_clean():
    identity = bench.source_identity()
    if not identity["runner_tracked"]:
        assert identity["runner_overlay"]
        assert identity["working_tree"] == "dirty"


def test_midrun_source_change_fails(monkeypatch):
    pytest.importorskip("sklearn")
    count = [0]
    def changing_source():
        count[0] += 1
        return {"source_content_sha256": str(count[0])}
    monkeypatch.setattr(bench, "source_identity", changing_source)
    result = bench.run(options(scenarios=["constant-score"], dtypes=["float64"], operations=["binary_confusion_matrix"]))
    assert result["validation"]["status"] == "failed"
    assert "changed during execution" in result["results"][-1]["reason"]


def test_midrun_commit_change_fails(monkeypatch):
    pytest.importorskip("sklearn")
    count = [0]
    def changing_commit():
        count[0] += 1
        return {"source_content_sha256": "same-content", "git_commit": str(count[0])}
    monkeypatch.setattr(bench, "source_identity", changing_commit)
    result = bench.run(options(scenarios=["constant-score"], dtypes=["float64"], operations=["binary_confusion_matrix"]))
    assert result["validation"]["status"] == "failed"
    assert result["source_after"]["git_commit"] != result["source"]["git_commit"]
