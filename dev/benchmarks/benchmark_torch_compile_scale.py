"""Scale-crossover benchmark for statgpu's explicit ``torch.compile`` path.

The maintenance compile benchmark validates correctness at one small workload.
This benchmark varies ``n`` and ``p`` separately, isolates every case/mode/scale
combination in a fresh subprocess, and reports cold-start cost, post-compilation
latency, dispersion, and the estimated number of repeated fits required to
amortize compilation.

A physical CUDA GPU is required. Results are written to ``results/*.json`` and
an adjacent Markdown summary. No universal speedup claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


_PRECISION_RTOL = 1e-6
_PRECISION_ATOL = 1e-8
_BASELINE_SCALE = (1024, 64)
_CASE_NAMES = (
    "glm_irls",
    "lasso",
    "elasticnet",
    "scad",
    "mcp",
    "group_scad",
    "group_mcp",
)
_PRESETS = {
    "quick": {
        "scales": ((1024, 64),),
        "repeats": 3,
    },
    "standard": {
        "scales": (
            (1024, 64),
            (4096, 64),
            (16384, 64),
            (4096, 256),
            (4096, 1024),
            (16384, 1024),
        ),
        "repeats": 7,
    },
    "extended": {
        "scales": (
            (1024, 64),
            (4096, 64),
            (16384, 64),
            (65536, 64),
            (4096, 256),
            (4096, 1024),
            (16384, 256),
            (16384, 1024),
            (8192, 2048),
        ),
        "repeats": 11,
    },
}


def _parse_scales(value: str) -> tuple[tuple[int, int], ...]:
    """Parse a comma-separated ``n x p`` scale list with stable deduplication."""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("--scales must not be empty")

    scales = []
    seen = set()
    for raw_token in value.split(","):
        token = raw_token.strip().lower().replace("×", "x")
        parts = token.split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"invalid scale {raw_token!r}; expected NXP such as 1024x64"
            )
        try:
            n_samples, n_features = (int(part.strip()) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid scale {raw_token!r}; N and P must be integers"
            ) from exc
        if n_samples < 1 or n_features < 1:
            raise argparse.ArgumentTypeError(
                f"invalid scale {raw_token!r}; N and P must be positive"
            )
        scale = (n_samples, n_features)
        if scale not in seen:
            seen.add(scale)
            scales.append(scale)
    return tuple(scales)


def _parse_cases(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated benchmark case list."""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("--cases must not be empty")
    cases = []
    seen = set()
    for raw_case in value.split(","):
        case = raw_case.strip().lower()
        if case not in _CASE_NAMES:
            allowed = ", ".join(_CASE_NAMES)
            raise argparse.ArgumentTypeError(
                f"unknown case {raw_case!r}; expected one of {allowed}"
            )
        if case not in seen:
            seen.add(case)
            cases.append(case)
    return tuple(cases)


def _resolve_plan(
    preset: str,
    scales: Sequence[tuple[int, int]] | None,
    cases: Sequence[str] | None,
    repeats: int | None,
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...], int]:
    """Resolve CLI overrides against a named benchmark preset."""
    preset_config = _PRESETS[preset]
    resolved_scales = tuple(scales or preset_config["scales"])
    resolved_cases = tuple(cases or _CASE_NAMES)
    resolved_repeats = int(
        preset_config["repeats"] if repeats is None else repeats
    )
    if resolved_repeats < 3:
        raise ValueError("scale benchmark requires at least 3 repeats")
    return resolved_scales, resolved_cases, resolved_repeats


def _timing_stats(values: Iterable[float]) -> dict:
    """Return robust timing summaries, including median and IQR."""
    samples = np.asarray(tuple(float(value) for value in values), dtype=float)
    if samples.size == 0:
        raise ValueError("timing sample must not be empty")
    if not np.isfinite(samples).all() or np.any(samples <= 0):
        raise ValueError("timing samples must be finite and positive")
    q1, q3 = np.percentile(samples, [25.0, 75.0])
    return {
        "count": int(samples.size),
        "median": float(statistics.median(samples.tolist())),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _summarize_timings(
    eager_values: Sequence[float], compiled_values: Sequence[float]
) -> dict:
    """Summarize eager, cold compiled, warm compiled, and amortization metrics."""
    if len(compiled_values) < 2:
        raise ValueError("compiled timings require one cold and at least one warm run")

    eager = _timing_stats(eager_values)
    compiled_cold = float(compiled_values[0])
    compiled_warm = _timing_stats(compiled_values[1:])
    eager_median = eager["median"]
    warm_median = compiled_warm["median"]
    warm_speedup = eager_median / warm_median
    cold_overhead_ratio = compiled_cold / eager_median

    additional_fits = None
    total_fits = None
    if warm_median < eager_median:
        if compiled_cold <= eager_median:
            additional_fits = 0
        else:
            additional_fits = int(
                math.ceil(
                    (compiled_cold - eager_median)
                    / (eager_median - warm_median)
                )
            )
        total_fits = 1 + additional_fits

    return {
        "eager": eager,
        "compiled_cold": compiled_cold,
        "compiled_warm": compiled_warm,
        "warm_speedup": float(warm_speedup),
        "cold_overhead_ratio": float(cold_overhead_ratio),
        "first_fit_extra_seconds_vs_eager": float(compiled_cold - eager_median),
        "break_even_additional_warm_fits": additional_fits,
        "break_even_total_fits": total_fits,
    }


def _scale_axis(n_samples: int, n_features: int) -> str:
    if (n_samples, n_features) == _BASELINE_SCALE:
        return "baseline"
    if n_features == _BASELINE_SCALE[1]:
        return "n-scaling"
    if n_samples == 4096 and n_features > _BASELINE_SCALE[1]:
        return "p-scaling"
    return "joint-scaling"


def _make_data(n_samples: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    seed = 20260806 + 1009 * n_samples + 9176 * n_features
    rng = np.random.default_rng(seed % (2**32))
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.zeros(n_features, dtype=np.float64)
    signal = np.array([1.3, -1.0, 0.8, -0.6, 0.5, -0.4, 0.3, -0.2])
    beta[: min(n_features, signal.size)] = signal[: min(n_features, signal.size)]
    y = X @ beta + 0.05 * rng.normal(size=n_samples)
    return X, y


def _make_model(case: str, groups: list[list[int]]):
    from statgpu.linear_model import (
        ElasticNet,
        GeneralizedLinearModel,
        Lasso,
        PenalizedLinearRegression,
    )

    if case == "glm_irls":
        return GeneralizedLinearModel(
            family="gaussian",
            solver="irls",
            C=0.0,
            max_iter=50,
            tol=1e-7,
            device="torch",
            compute_inference=False,
        )
    if case == "lasso":
        return Lasso(alpha=0.01, max_iter=100, tol=1e-6, device="torch")
    if case == "elasticnet":
        return ElasticNet(
            alpha=0.01,
            l1_ratio=0.6,
            max_iter=100,
            tol=1e-6,
            device="torch",
        )

    penalty_kwargs = {"groups": groups} if case.startswith("group_") else {}
    return PenalizedLinearRegression(
        penalty=case,
        penalty_kwargs=penalty_kwargs,
        alpha=0.03,
        max_iter=60,
        max_lla_iters=3,
        tol=1e-6,
        lla_tol=1e-6,
        device="torch",
        compute_inference=False,
    )


def _validate_mode_evidence(mode: str, events: Sequence[dict], graph_delta: int) -> str:
    statuses = tuple(str(event.get("status", "")) for event in events)
    if any("fallback" in status for status in statuses):
        raise RuntimeError(f"compile mode {mode!r} entered fallback: {statuses!r}")

    if mode == "disable":
        if int(graph_delta) != 0:
            raise RuntimeError(
                f"disable mode unexpectedly created {graph_delta} Dynamo graph(s)"
            )
        if events and any(status != "disabled" for status in statuses):
            raise RuntimeError(
                f"disable mode emitted unexpected diagnostics {statuses!r}"
            )
        return "eager-no-dynamo-graph"

    if int(graph_delta) <= 0:
        raise RuntimeError("default mode did not create a Dynamo graph")
    if not any(status == "compiled" for status in statuses):
        raise RuntimeError(
            f"default mode has diagnostics {statuses!r} but no compiled event"
        )
    return "compiled-diagnostic-and-dynamo-graph"


def _run_child(
    *, mode: str, case: str, n_samples: int, n_features: int, repeats: int
) -> dict:
    os.environ["STATGPU_TORCH_COMPILE_MODE"] = mode

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A physical Torch CUDA GPU is required")

    from statgpu.backends import _to_numpy
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics

    X, y = _make_data(n_samples, n_features)
    groups = [
        list(range(start, min(start + 8, n_features)))
        for start in range(0, n_features, 8)
    ]

    get_torch_compile_diagnostics(clear=True)
    torch._dynamo.reset()
    before_graphs = int(
        torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
    )

    timings = []
    model = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model = _make_model(case, groups).fit(X, y)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    assert model is not None
    prediction = np.asarray(_to_numpy(model.predict(X)))
    coefficients = np.asarray(_to_numpy(model.coef_))
    events = get_torch_compile_diagnostics(clear=True)
    after_graphs = int(
        torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
    )
    graph_delta = after_graphs - before_graphs
    evidence = _validate_mode_evidence(mode, events, graph_delta)

    if not np.isfinite(prediction).all() or not np.isfinite(coefficients).all():
        raise RuntimeError(
            f"{case}:{mode}:n={n_samples}:p={n_features} produced non-finite output"
        )

    return {
        "mode": mode,
        "case": case,
        "n_samples": n_samples,
        "n_features": n_features,
        "fit_seconds": timings,
        "prediction": prediction.tolist(),
        "coefficients": coefficients.tolist(),
        "finite_prediction": True,
        "finite_coefficients": True,
        "n_iter": getattr(model, "n_iter_", None),
        "converged": getattr(model, "converged_", None),
        "compile_events": events,
        "compile_evidence": evidence,
        "unique_graphs_delta": graph_delta,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
    }


def _child_main(args: argparse.Namespace) -> None:
    result = _run_child(
        mode=args.mode,
        case=args.case,
        n_samples=args.n_samples,
        n_features=args.n_features,
        repeats=args.repeats,
    )
    Path(args.child_output).write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _run_subprocess_case(
    *,
    mode: str,
    case: str,
    n_samples: int,
    n_features: int,
    repeats: int,
    output_path: Path,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--mode",
        mode,
        "--case",
        case,
        "--n-samples",
        str(n_samples),
        "--n-features",
        str(n_features),
        "--repeats",
        str(repeats),
        "--child-output",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _format_optional_integer(value) -> str:
    return "never" if value is None else str(value)


def _render_markdown(report: dict, json_path: Path) -> str:
    environment = report["environment"]
    lines = [
        "# Torch Compile Scale-Crossover Benchmark",
        "",
        f"- JSON artifact: `{json_path.as_posix()}`",
        f"- Preset: `{report['preset']}`",
        f"- Repeats per mode/case/scale: `{report['repeats']}`",
        f"- GPU: `{environment['gpu']}`",
        f"- Torch: `{environment['torch']}`",
        f"- CUDA: `{environment['cuda']}`",
        "",
        "The first compiled repetition is reported as cold-start. Remaining compiled repetitions form the warm sample. Every case/mode/scale combination runs in a fresh subprocess.",
        "",
        "| axis | n | p | case | eager median (s) | compiled cold (s) | compiled warm median (s) | warm speedup | cold/eager | break-even total fits |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in report["results"]:
        summary = record["timing_summary"]
        lines.append(
            "| {axis} | {n} | {p} | `{case}` | {eager:.6f} | {cold:.6f} | "
            "{warm:.6f} | {speedup:.3f}x | {cold_ratio:.3f}x | {break_even} |".format(
                axis=record["axis"],
                n=record["n_samples"],
                p=record["n_features"],
                case=record["case"],
                eager=summary["eager"]["median"],
                cold=summary["compiled_cold"],
                warm=summary["compiled_warm"]["median"],
                speedup=summary["warm_speedup"],
                cold_ratio=summary["cold_overhead_ratio"],
                break_even=_format_optional_integer(
                    summary["break_even_total_fits"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "`never` means the measured compiled warm median was not faster than eager, so the observed cold cost cannot be amortized by repeating the same fit.",
            "",
        ]
    )
    return "\n".join(lines)


def _parent_main(args: argparse.Namespace) -> None:
    scales, cases, repeats = _resolve_plan(
        args.preset, args.scales, args.cases, args.repeats
    )

    raw_results = {}
    with tempfile.TemporaryDirectory() as directory:
        temp_dir = Path(directory)
        for n_samples, n_features in scales:
            for case in cases:
                key = (n_samples, n_features, case)
                raw_results[key] = {}
                for mode in ("disable", "default"):
                    child_path = temp_dir / (
                        f"n{n_samples}_p{n_features}_{case}_{mode}.json"
                    )
                    raw_results[key][mode] = _run_subprocess_case(
                        mode=mode,
                        case=case,
                        n_samples=n_samples,
                        n_features=n_features,
                        repeats=repeats,
                        output_path=child_path,
                    )

    results = []
    environment = None
    for (n_samples, n_features, case), modes in raw_results.items():
        eager = modes["disable"]
        compiled = modes["default"]
        environment = environment or compiled["environment"]

        eager_prediction = np.asarray(eager.pop("prediction"), dtype=float)
        compiled_prediction = np.asarray(compiled.pop("prediction"), dtype=float)
        eager_coef = np.asarray(eager.pop("coefficients"), dtype=float)
        compiled_coef = np.asarray(compiled.pop("coefficients"), dtype=float)
        np.testing.assert_allclose(
            compiled_prediction,
            eager_prediction,
            rtol=_PRECISION_RTOL,
            atol=_PRECISION_ATOL,
            err_msg=(
                f"{case}:n={n_samples}:p={n_features} compiled predictions differ"
            ),
        )
        np.testing.assert_allclose(
            compiled_coef,
            eager_coef,
            rtol=_PRECISION_RTOL,
            atol=_PRECISION_ATOL,
            err_msg=(
                f"{case}:n={n_samples}:p={n_features} compiled coefficients differ"
            ),
        )

        results.append(
            {
                "axis": _scale_axis(n_samples, n_features),
                "n_samples": n_samples,
                "n_features": n_features,
                "matrix_bytes_float64": n_samples * n_features * 8,
                "case": case,
                "timing_summary": _summarize_timings(
                    eager["fit_seconds"], compiled["fit_seconds"]
                ),
                "raw_fit_seconds": {
                    "disable": eager["fit_seconds"],
                    "default": compiled["fit_seconds"],
                },
                "precision": {
                    "prediction_max_abs_diff": float(
                        np.max(np.abs(compiled_prediction - eager_prediction))
                    ),
                    "coefficient_max_abs_diff": float(
                        np.max(np.abs(compiled_coef - eager_coef))
                    ),
                    "rtol": _PRECISION_RTOL,
                    "atol": _PRECISION_ATOL,
                    "status": "pass",
                },
                "compile": {
                    "disable": {
                        "evidence": eager["compile_evidence"],
                        "unique_graphs_delta": eager["unique_graphs_delta"],
                        "events": eager["compile_events"],
                    },
                    "default": {
                        "evidence": compiled["compile_evidence"],
                        "unique_graphs_delta": compiled["unique_graphs_delta"],
                        "events": compiled["compile_events"],
                    },
                },
                "convergence": {
                    "disable": {
                        "n_iter": eager["n_iter"],
                        "converged": eager["converged"],
                    },
                    "default": {
                        "n_iter": compiled["n_iter"],
                        "converged": compiled["converged"],
                    },
                },
            }
        )

    report = {
        "method": "torch_compile_scale_crossover",
        "benchmark_version": 1,
        "preset": args.preset,
        "scales": [
            {"n_samples": n_samples, "n_features": n_features}
            for n_samples, n_features in scales
        ],
        "cases": list(cases),
        "repeats": repeats,
        "environment": environment,
        "timing_scope": {
            "fit": "model construction plus fit; data generation and prediction excluded",
            "compiled_cold": "first default-mode fit, including first-use compilation",
            "compiled_warm": "default-mode fits after the first repetition",
        },
        "statistics": {
            "location": "median",
            "dispersion": "min, max, and linear-interpolation IQR",
            "warm_speedup": "eager median / compiled warm median",
            "break_even": (
                "minimum total identical-shape fits for cold + warm compiled time "
                "to be no greater than repeated eager median time"
            ),
        },
        "isolation": (
            "Each case/mode/scale combination runs in a fresh subprocess; "
            "module-level compiled-callable caches cannot leak between records."
        ),
        "results": results,
        "interpretation": (
            "This benchmark estimates workload-specific crossover behavior and "
            "does not establish a universal torch.compile speedup."
        ),
        "schema_status": "ok",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        _render_markdown(report, output_path) + "\n", encoding="utf-8"
    )
    print(output_path)
    print(summary_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/torch_compile_scale.json")
    parser.add_argument("--summary-output")
    parser.add_argument("--preset", choices=tuple(_PRESETS), default="standard")
    parser.add_argument("--scales", type=_parse_scales)
    parser.add_argument("--cases", type=_parse_cases)
    parser.add_argument("--repeats", type=int)

    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("disable", "default"), help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=_CASE_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--n-samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-features", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-output", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.child:
        required = {
            "mode": args.mode,
            "case": args.case,
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "repeats": args.repeats,
            "child_output": args.child_output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("--child requires " + ", ".join(missing))
        if args.n_samples < 1 or args.n_features < 1 or args.repeats < 3:
            parser.error("child dimensions must be positive and repeats >= 3")
        _child_main(args)
        return

    try:
        _resolve_plan(args.preset, args.scales, args.cases, args.repeats)
    except ValueError as exc:
        parser.error(str(exc))
    _parent_main(args)


if __name__ == "__main__":
    main()
