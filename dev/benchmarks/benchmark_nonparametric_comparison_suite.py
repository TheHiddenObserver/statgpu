"""Unified nonparametric comparison suite (Python + R).

Runs:
1) KDE vs SciPy
2) Kernel regression vs statsmodels
3) Nonparametric (KDE + kernel regression) vs R
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_py(command: list[str]) -> Dict[str, Any]:
    env = dict(os.environ)
    root_text = str(REPO_ROOT)
    cur_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root_text if not cur_py_path else (root_text + os.pathsep + cur_py_path)

    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env=env,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified nonparametric comparison suite")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=1200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--ci-resamples", type=int, default=120)
    parser.add_argument("--ci-confidence-level", type=float, default=0.95)
    parser.add_argument("--ci-repeats", type=int, default=1)
    parser.add_argument("--ci-warmup", type=int, default=0)
    parser.add_argument("--n-ci-eval", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--bandwidth-abs", type=float, default=0.45)
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    kde_out = out_dir / f"suite_kde_vs_scipy_{ts}.json"
    kr_out = out_dir / f"suite_kernel_reg_vs_statsmodels_{ts}.json"
    r_out = out_dir / f"suite_nonparametric_vs_r_{ts}.json"

    py = sys.executable

    kde_cmd = [
        py,
        "dev/benchmarks/benchmark_kde_vs_scipy.py",
        "--n-samples",
        str(int(args.n_samples)),
        "--n-eval",
        str(int(args.n_eval)),
        "--dim",
        "1",
        "--bandwidth",
        str(float(args.bandwidth_abs)),
        "--repeats",
        str(int(args.repeats)),
        "--warmup",
        str(int(args.warmup)),
        "--seed",
        str(int(args.seed)),
        "--json-out",
        str(kde_out),
    ]

    kr_cmd = [
        py,
        "dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py",
        "--n-samples",
        str(int(args.n_samples)),
        "--n-eval",
        str(int(args.n_eval)),
        "--dim",
        "1",
        "--bandwidth-abs",
        str(float(args.bandwidth_abs)),
        "--regression",
        "local_linear",
        "--kernel-metric",
        "diagonal",
        "--repeats",
        str(int(args.repeats)),
        "--warmup",
        str(int(args.warmup)),
        "--seed",
        str(int(args.seed)),
        "--json-out",
        str(kr_out),
    ]

    r_cmd = [
        py,
        "dev/benchmarks/benchmark_nonparametric_vs_r.py",
        "--n-samples",
        str(int(args.n_samples)),
        "--n-eval",
        str(int(args.n_eval)),
        "--bandwidth-abs",
        str(float(args.bandwidth_abs)),
        "--repeats",
        str(int(args.repeats)),
        "--warmup",
        str(int(args.warmup)),
        "--ci-resamples",
        str(int(args.ci_resamples)),
        "--ci-confidence-level",
        str(float(args.ci_confidence_level)),
        "--ci-repeats",
        str(int(args.ci_repeats)),
        "--ci-warmup",
        str(int(args.ci_warmup)),
        "--n-ci-eval",
        str(int(args.n_ci_eval)),
        "--seed",
        str(int(args.seed)),
        "--json-out",
        str(r_out),
    ]

    run_kde = _run_py(kde_cmd)
    run_kr = _run_py(kr_cmd)
    if args.skip_r:
        run_r = {
            "returncode": 0,
            "stdout": "",
            "stderr": "skipped by --skip-r",
        }
    elif shutil.which("Rscript") is None:
        run_r = {
            "returncode": 0,
            "stdout": "",
            "stderr": "skipped: Rscript not found on PATH",
        }
    else:
        run_r = _run_py(r_cmd)

    payload: Dict[str, Any] = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_samples": int(args.n_samples),
            "n_eval": int(args.n_eval),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "ci_resamples": int(args.ci_resamples),
            "ci_confidence_level": float(args.ci_confidence_level),
            "ci_repeats": int(args.ci_repeats),
            "ci_warmup": int(args.ci_warmup),
            "n_ci_eval": int(args.n_ci_eval),
            "seed": int(args.seed),
            "bandwidth_abs": float(args.bandwidth_abs),
        },
        "runs": {
            "kde_vs_scipy": run_kde,
            "kernel_regression_vs_statsmodels": run_kr,
            "nonparametric_vs_r": run_r,
        },
        "artifacts": {
            "kde_vs_scipy": str(kde_out),
            "kernel_regression_vs_statsmodels": str(kr_out),
            "nonparametric_vs_r": str(r_out),
        },
        "metrics": {},
    }

    if run_kde["returncode"] == 0 and kde_out.exists():
        payload["metrics"]["kde_vs_scipy"] = _read_json(kde_out).get("metrics", {})
    if run_kr["returncode"] == 0 and kr_out.exists():
        payload["metrics"]["kernel_regression_vs_statsmodels"] = _read_json(kr_out).get("metrics", {})
    if run_r["returncode"] == 0 and r_out.exists():
        payload["metrics"]["nonparametric_vs_r"] = _read_json(r_out).get("metrics", {})

    if args.json_out:
        out_path = Path(args.json_out)
    else:
        out_path = out_dir / f"nonparametric_comparison_suite_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out_path),
                "artifacts": payload["artifacts"],
                "returncodes": {
                    "kde_vs_scipy": run_kde["returncode"],
                    "kernel_regression_vs_statsmodels": run_kr["returncode"],
                    "nonparametric_vs_r": run_r["returncode"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
