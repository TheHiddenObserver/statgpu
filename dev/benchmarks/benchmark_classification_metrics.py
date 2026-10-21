#!/usr/bin/env python3
"""Correctness-gated binary-metric timings; not a canonical dashboard source.

Examples (from repository root)::

    python -m dev.benchmarks.benchmark_classification_metrics --smoke --output results/metrics-smoke.json
    python dev/benchmarks/benchmark_classification_metrics.py --backends numpy cupy torch --require-backends --sizes 1000 10000 100000 --output results/metrics.json

No sample-weight axis: these public APIs do not accept sample_weight. GPU input
transfer and report conversion are excluded; public validation/scalar syncs and
internal float64 promotion are included. This is not pure-kernel timing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# Ensure this checkout, not a globally installed statgpu, supplies the code.
sys.path.insert(0, str(ROOT))
import numpy as np
import statgpu
from statgpu import metrics

OPERATIONS = (
    "binary_confusion_matrix", "binary_classification_table", "binary_roc_curve",
    "binary_roc_auc_score", "binary_precision_recall_curve",
    "binary_average_precision_score", "evaluate_binary_classification",
)
SCENARIOS = ("balanced-continuous", "imbalanced-tied", "constant-score")
ATOL = 1e-12  # Rates are float64 even for float32 scores; counts/thresholds exact.


class Unavailable(RuntimeError):
    """An optional backend cannot be exercised, not a numerical failure."""


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_identity():
    if Path(statgpu.__file__).resolve() != (ROOT / "statgpu/__init__.py").resolve():
        raise RuntimeError("Imported statgpu is not from this checkout; start a fresh Python process")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)
    source_files = sorted((ROOT / "statgpu").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    content = json.dumps(hashes, sort_keys=True).encode()
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain"))
        tracked_paths = set(git("ls-files").decode().splitlines())
        runner_tracked = str(Path(__file__).resolve().relative_to(ROOT)) in tracked_paths
        dirty = dirty or not runner_tracked
        diff = git("diff", "HEAD", "--binary")
    except (OSError, subprocess.CalledProcessError):
        commit, dirty, diff, runner_tracked = None, True, b"git-unavailable", False
    return {
        "git_commit": commit, "working_tree": "dirty" if dirty else "clean",
        "statgpu_version": statgpu.__version__, "runner_tracked": runner_tracked,
        "runner_overlay": not runner_tracked,
        "statgpu_import_path": str(Path(statgpu.__file__).resolve()),
        "source_content_sha256": hashlib.sha256(content).hexdigest(),
        "worktree_fingerprint": hashlib.sha256(diff + content).hexdigest(),
        "fingerprint_scope": "git tracked binary diff plus all statgpu Python files and this runner (includes untracked runner)",
        "script_sha256": hashes[str(Path(__file__).resolve().relative_to(ROOT))],
    }


def base_environment():
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"), "os": platform.platform(),
            "cpu": cpu, "thread_limits": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


class Runtime:
    def __init__(self, backend, ordinal):
        self.backend = backend
        self.ordinal = ordinal
        self.environment = base_environment()
        self.environment.update({"backend": backend, "device": "cpu" if backend == "numpy" else f"cuda:{ordinal}"})
        if backend == "numpy":
            self.xp = np
        elif backend == "cupy":
            try:
                import cupy as cp
            except ImportError as exc:
                raise Unavailable(str(exc)) from exc
            try:
                count = cp.cuda.runtime.getDeviceCount()
            except cp.cuda.runtime.CUDARuntimeError as exc:
                raise Unavailable(str(exc)) from exc
            if count <= ordinal:
                raise Unavailable(f"CUDA device {ordinal} not available ({count} devices)")
            self.xp = cp
            cp.cuda.Device(ordinal).use()
            props = cp.cuda.runtime.getDeviceProperties(ordinal)
            name = props.get("name", "unknown")
            self.environment.update({"cupy": cp.__version__, "gpu": name.decode() if isinstance(name, bytes) else str(name),
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "driver": cp.cuda.runtime.driverGetVersion(),
                "gpu_uuid": str(props.get("uuid")) if props.get("uuid") is not None else None})
        else:
            try:
                import torch
            except ImportError as exc:
                raise Unavailable(str(exc)) from exc
            if not torch.cuda.is_available() or torch.cuda.device_count() <= ordinal:
                raise Unavailable(f"Torch CUDA device {ordinal} unavailable; CPU fallback forbidden")
            self.xp = torch
            torch.cuda.set_device(ordinal)
            props = torch.cuda.get_device_properties(ordinal)
            self.environment.update({"torch": torch.__version__, "gpu": props.name,
                "cuda_runtime": torch.version.cuda, "driver": None, "driver_note": "not exposed by Torch public API; record driver externally if required for canonical publication", "gpu_uuid": str(getattr(props, "uuid", "unavailable"))})
        self.environment["environment_id"] = hashlib.sha256(json.dumps(self.environment, sort_keys=True).encode()).hexdigest()[:20]

    def array(self, values):
        if self.backend == "torch":
            return self.xp.as_tensor(values, device=f"cuda:{self.ordinal}")
        return self.xp.asarray(values)

    def synchronize(self):
        if self.backend == "cupy":
            self.xp.cuda.Device(self.ordinal).synchronize()
        elif self.backend == "torch":
            self.xp.cuda.synchronize(device=self.ordinal)

    def to_host(self, value):
        if isinstance(value, dict):
            return {k: self.to_host(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self.to_host(v) for v in value)
        if self.backend == "cupy" and isinstance(value, self.xp.ndarray):
            return self.xp.asnumpy(value)
        if self.backend == "torch" and isinstance(value, self.xp.Tensor):
            return value.detach().cpu().numpy()
        return value

    def verify_device(self, value, allow_host_scalar=False):
        """Inspect actual outputs, not merely the requested backend name."""
        if isinstance(value, dict):
            for key, child in value.items():
                self.verify_device(child, allow_host_scalar=(key == "threshold"))
        elif isinstance(value, tuple):
            for child in value:
                self.verify_device(child)
        elif self.backend == "cupy":
            if isinstance(value, self.xp.ndarray):
                if value.device.id != self.ordinal:
                    raise AssertionError("CuPy output on the wrong device")
            elif not (allow_host_scalar and isinstance(value, (int, float))):
                raise AssertionError("CuPy operation returned a host array")
        elif self.backend == "torch":
            if isinstance(value, self.xp.Tensor):
                if value.device.type != "cuda" or value.device.index != self.ordinal:
                    raise AssertionError("Torch output not on the executing CUDA device")
            elif not (allow_host_scalar and isinstance(value, (int, float))):
                raise AssertionError("Torch operation returned a host array")


def make_data(n, scenario, dtype, seed):
    rng = np.random.default_rng(seed)
    positives = max(1, n // (20 if scenario == "imbalanced-tied" else 2))
    labels = np.zeros(n, dtype=np.int64)
    labels[:positives] = 1
    rng.shuffle(labels)
    scores = rng.uniform(size=n)
    if scenario == "imbalanced-tied":
        scores = np.round(scores * 8) / 8
    elif scenario == "constant-score":
        scores[:] = 0.5
    return labels, scores.astype(dtype)


def references(y, score):
    from sklearn import metrics as sk
    pred = (score >= 0.5).astype(np.int64)
    cm = sk.confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    table = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": sk.accuracy_score(y, pred), "precision": sk.precision_score(y, pred, zero_division=0),
        "recall": sk.recall_score(y, pred, zero_division=0), "f1": sk.f1_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp), "support_negative": tn + fp, "support_positive": fn + tp}
    fpr, tpr, thresholds = sk.roc_curve(y, score, drop_intermediate=False)
    # sklearn <1.3 used max(score)+1 for this sentinel. Public statgpu uses inf.
    thresholds = thresholds.astype(float)
    thresholds[0] = np.inf
    precision, recall, pr_thresholds = sk.precision_recall_curve(y, score)
    # statgpu goes from +inf down, with an explicit (precision=1, recall=0) start.
    pr = (precision[::-1], recall[::-1], np.r_[np.inf, pr_thresholds[::-1]])
    auc, ap = sk.roc_auc_score(y, score), sk.average_precision_score(y, score)
    batch = {"threshold": 0.5, "confusion_matrix": cm, "classification_table": table,
        "roc_auc": auc, "average_precision": ap,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
        "precision_recall_curve": dict(zip(("precision", "recall", "thresholds"), pr))}
    return dict(zip(OPERATIONS, (cm, table, (fpr, tpr, thresholds), auc, pr, ap, batch)))


def check(actual, expected):
    """Compare all fields, including threshold sentinels without inf subtraction."""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError("Output keys differ from reference")
        return max((check(actual[k], expected[k]) for k in expected), default=0.0)
    if isinstance(expected, tuple):
        if len(actual) != len(expected):
            raise AssertionError("Output tuple length differs")
        return max(check(a, b) for a, b in zip(actual, expected))
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(np.isposinf(a), np.isposinf(b)):
        raise AssertionError("Output shape or threshold sentinel differs")
    finite = np.isfinite(b)
    if not np.all(np.isfinite(a[finite])) or np.any(np.isnan(a)) or np.any(np.isneginf(a)):
        raise AssertionError("Non-finite metric")
    error = float(np.max(np.abs(a[finite] - b[finite]), initial=0.0))
    if error > ATOL:
        raise AssertionError(f"Reference mismatch: absolute error {error} > {ATOL}")
    return error


def output_dtypes(value):
    if isinstance(value, dict):
        return {key: output_dtypes(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [output_dtypes(child) for child in value]
    return str(np.asarray(value).dtype)


def benchmark_operation(runtime, operation, y, scores, expected, warmup, repeats):
    function = getattr(metrics, operation)
    second = (scores >= 0.5) if operation in OPERATIONS[:2] else scores
    call = lambda: function(y, second, backend=runtime.backend)
    # Correctness and actual CUDA ownership gate precede any measured sample.
    first = call()
    runtime.verify_device(first)
    error = check(runtime.to_host(first), expected)
    for _ in range(warmup):
        output = call()
        runtime.verify_device(output)
        error = max(error, check(runtime.to_host(output), expected))
    samples = []
    for _ in range(repeats):
        runtime.synchronize()
        start = time.perf_counter()
        output = call()
        runtime.synchronize()
        samples.append(time.perf_counter() - start)
        runtime.verify_device(output)
        error = max(error, check(runtime.to_host(output), expected))
    return {"status": "pass", "correctness": {"status": "pass", "max_abs_diff": error,
        "atol": ATOL, "threshold_source": "float64 empirical rates; exact integer counts and score thresholds", "convergence": "not-applicable"},
        "timing": {"scope": "public-operation-resident-input", "warmup": warmup, "repeats": repeats,
            "statistic": "median", "seconds": float(np.median(samples)), "raw_seconds": samples,
            "transfer_policy": "input allocation/transfer and report conversion excluded; internal casting and scalar synchronization included",
            "synchronized_device": runtime.environment["device"] if runtime.backend != "numpy" else None},
        "output_dtypes": output_dtypes(runtime.to_host(output)),
        "backend": {"executed": runtime.backend, "device": runtime.environment["device"], "output_device_verified": True}}


def run(args):
    source = source_identity()
    rows, environments = [], {}
    for backend in args.backends:
        try:
            runtime = Runtime(backend, args.device)
            environments[backend] = runtime.environment
        except Unavailable as exc:
            rows.append({"backend": backend, "status": "skipped", "reason": str(exc)})
            continue
        except Exception as exc:
            rows.append({"backend": backend, "status": "failed", "reason": f"initialization: {type(exc).__name__}: {exc}"})
            continue
        for n in args.sizes:
            for dtype in args.dtypes:
                for scenario in args.scenarios:
                    y, score = make_data(n, scenario, dtype, args.seed)
                    problem = {"shape": {"n": n}, "input_dtype": dtype, "internal_rate_dtype": "float64",
                        "seed": args.seed, "scenario": scenario, "positives": int(y.sum()),
                        "data_id": hashlib.sha256(y.tobytes() + score.tobytes()).hexdigest(),
                        "parameters": {"threshold": 0.5, "sample_weight": "unsupported by public API"}}
                    try:
                        expected = references(y, score)
                        dy, ds = runtime.array(y), runtime.array(score)
                        runtime.verify_device(dy)
                        runtime.verify_device(ds)
                    except Exception as exc:
                        rows.append({"backend": backend, "problem": problem, "status": "failed", "reason": f"setup/reference: {type(exc).__name__}: {exc}"})
                        continue
                    for operation in args.operations:
                        row = {"operation": operation, "requested_backend": backend, "problem": problem,
                            "environment_id": runtime.environment["environment_id"]}
                        try:
                            row.update(benchmark_operation(runtime, operation, dy, ds, expected[operation], args.warmup, args.repeats))
                        except Exception as exc:
                            row.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "correctness": {"status": "failed"}})
                        rows.append(row)
    source_after = source_identity()
    if (source_after["source_content_sha256"] != source["source_content_sha256"]
            or source_after.get("git_commit") != source.get("git_commit")):
        rows.append({"status": "failed", "reason": "Source commit, numerical source or runner changed during execution"})
    failed = any(r["status"] == "failed" for r in rows)
    skipped = any(r["status"] == "skipped" for r in rows)
    passed = any(r["status"] == "pass" for r in rows)
    status = "failed" if failed or (args.require_backends and skipped) or not passed else ("partial" if skipped else "complete")
    return {"schema_version": "statgpu-benchmark-v2", "benchmark": {"name": "binary-classification-metrics",
        "script": str(Path(__file__).relative_to(ROOT)), "command": [sys.executable, *sys.argv]},
        "source": source, "source_after": source_after, "environment": environments, "results": rows,
        "correctness": {"reference": {"name": "sklearn.metrics", "version": version("scikit-learn"), "reference_type": "external-python",
            "settings": {"roc_drop_intermediate": False, "roc_initial_threshold": "normalized to +inf", "pr_order": "reverse sklearn; prepend +inf threshold", "classification_zero_division": 0}}},
        "validation": {"status": status, "tier": "local-minimal" if args.smoke else "benchmark-candidate",
            "canonical_ready": False, "uncovered_reasons": [r["reason"] for r in rows if r["status"] == "skipped"],
            "notes": ["Not a speedup claim; no external timings, canonical parser or registration.",
                "CPU smoke validates runner behavior, not physical CUDA or representative performance.",
                "Curves/rates promote scores to float64; float32 input does not claim float32 computation."]}}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backends", nargs="+", choices=("numpy", "cupy", "torch"), default=["numpy", "cupy", "torch"])
    p.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    p.add_argument("--dtypes", nargs="+", choices=("float32", "float64"), default=["float32", "float64"])
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--require-backends", action="store_true")
    p.add_argument("--smoke", action="store_true", help="CPU-only, n=40, one warmup and two timed samples; not performance evidence")
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.smoke:
        args.backends, args.sizes, args.warmup, args.repeats = ["numpy"], [40], 1, 2
    if min(args.sizes) < 2 or args.device < 0 or args.warmup < 0 or args.repeats < 1:
        p.error("sizes >= 2, device/warmup >= 0, repeats >= 1 required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["validation"]["status"], "rows": len(result["results"]), "output": str(args.output)}))
    return 1 if result["validation"]["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
