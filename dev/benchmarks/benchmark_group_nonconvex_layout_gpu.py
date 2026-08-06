#!/usr/bin/env python3
"""Exact-source physical-GPU contract for Group MCP/SCAD layout semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import GroupMCPPenalty, GroupSCADPenalty
from statgpu.solvers import fista_lla_path
from statgpu.solvers._fista_lla_group_contract import _group_surrogate_factory


SOURCE_FILES = (
    "dev/benchmarks/benchmark_group_nonconvex_layout_gpu.py",
    "dev/tests/test_pr80_group_nonconvex_layout_contract.py",
    "dev/tests/test_pr80_group_nonconvex_pickle_contract.py",
    "dev/tests/test_pr80_group_nonconvex_capability_contract.py",
    "dev/tests/test_pr80_group_nonconvex_convergence_contract.py",
    "dev/tests/test_pr80_group_lla_surrogate_contract.py",
    "dev/tests/test_pr80_adaptive_group_penalty_contract.py",
    "dev/tests/test_pr80_group_clone_contract.py",
    "dev/tests/test_pr80_group_input_contract.py",
    "dev/tests/test_pr80_group_dimension_contract.py",
    "statgpu/linear_model/penalized/__init__.py",
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "statgpu/linear_model/penalized/_penalized_cv.py",
    "statgpu/linear_model/penalized/_group_penalty_model_contract.py",
    "statgpu/penalties/__init__.py",
    "statgpu/penalties/_group_lasso_layout.py",
    "statgpu/penalties/_group_nonconvex_layout.py",
    "statgpu/penalties/_group_dimension_contract.py",
    "statgpu/penalties/_group_mcp.py",
    "statgpu/penalties/_group_scad.py",
    "statgpu/solvers/__init__.py",
    "statgpu/solvers/_fista.py",
    "statgpu/solvers/_fista_lla.py",
    "statgpu/solvers/_fista_lla_group_contract.py",
    "statgpu/solvers/_utils.py",
)

INTERLEAVED = [[0, 3], [1, 2]]
GROUPED = [[0, 1], [2, 3]]
PERM = np.array([0, 3, 1, 2], dtype=np.int64)
INVERSE_PERM = np.argsort(PERM)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _as_numpy(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sample(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    beta = np.array([0.9, -0.55, 0.25, 0.7])
    y = 0.35 + X @ beta + rng.normal(scale=0.08, size=X.shape[0])
    return X, y


def _backend(name, X, y):
    if name == "cupy":
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy CUDA device unavailable")
        raw_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        device_name = (
            raw_name.decode("utf-8", errors="replace")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        return "cuda", cp.asarray(X), cp.asarray(y), device_name

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        torch.cuda.get_device_name(0),
    )


def _penalty(kind, groups, alpha=0.18):
    if kind == "group_mcp":
        return GroupMCPPenalty(alpha=alpha, gamma=3.0, groups=groups)
    return GroupSCADPenalty(alpha=alpha, a=3.7, groups=groups)


def _expected_lla(kind, coef, alpha=0.18):
    expected = np.zeros_like(coef, dtype=np.float64)
    for group in INTERLEAVED:
        idx = np.asarray(group, dtype=np.int64)
        norm = np.linalg.norm(coef[idx])
        alpha_g = alpha * np.sqrt(idx.size)
        if kind == "group_mcp":
            derivative = max(alpha_g - norm / 3.0, 0.0)
        elif norm <= alpha_g:
            derivative = alpha_g
        elif norm <= 3.7 * alpha_g:
            derivative = (3.7 * alpha_g - norm) / 2.7
        else:
            derivative = 0.0
        expected[idx] = derivative
    return expected


def _fit(kind, X, y, groups, device):
    kwargs = {"groups": groups}
    if kind == "group_mcp":
        kwargs["gamma"] = 3.0
    else:
        kwargs["a"] = 3.7
    return PenalizedGeneralizedLinearModel(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=kwargs,
        alpha=0.18,
        solver="auto",
        device=device,
        fit_intercept=True,
        compute_inference=False,
        max_iter=500,
        tol=1e-8,
        max_lla_iters=20,
        lla_tol=1e-8,
    ).fit(X, y)


def _cv(kind, X, y, groups, device):
    kwargs = {"groups": groups}
    if kind == "group_mcp":
        kwargs["gamma"] = 3.0
    else:
        kwargs["a"] = 3.7
    return PenalizedGLM_CV(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=kwargs,
        alpha_grid=[0.24, 0.12],
        cv=2,
        random_state=17,
        device=device,
        max_iter=400,
        tol=1e-7,
    ).fit(X, y)


def _surrogate_contract(penalty):
    derivatives = np.array([0.4, 1.2, 1.2, 0.4])
    coef = np.array([0.8, -0.3, 0.5, 0.6])
    inner = _group_surrogate_factory(penalty)(derivatives)
    expected = sum(
        float(derivatives[group[0]])
        * np.linalg.norm(coef[np.asarray(group, dtype=np.int64)])
        for group in INTERLEAVED
    )
    actual = inner.value(coef)
    error = abs(float(actual) - float(expected))
    return {
        "inner_alpha": float(inner.alpha),
        "inner_group_weights": [float(value) for value in inner._group_weights],
        "value_abs_error": float(error),
        "passed": bool(inner.alpha == 1.0 and error <= 1e-14),
    }


def _api_contract():
    results = {
        "solver_export_module": fista_lla_path.__module__,
        "solver_export_passed": (
            fista_lla_path.__module__
            == "statgpu.solvers._fista_lla_group_contract"
        ),
        "penalties": {},
    }
    for kind, cls in (
        ("group_mcp", GroupMCPPenalty),
        ("group_scad", GroupSCADPenalty),
    ):
        penalty = _penalty(kind, [[3, 0], [2, 1]])
        params = penalty.get_params(deep=False)
        rebuilt = cls(**params)
        surrogate = _surrogate_contract(penalty)
        identity_gate = all(
            rebuilt.get_params(deep=False)[name] is value
            for name, value in params.items()
        )
        results["penalties"][kind] = {
            "groups": [list(group) for group in penalty.groups],
            "flat_indices": penalty._flat_indices.tolist(),
            "clone_identity_gate": bool(identity_gate),
            "surrogate": surrogate,
            "passed": bool(
                penalty.groups == ((0, 3), (1, 2))
                and penalty._flat_indices.tolist() == [0, 3, 1, 2]
                and not penalty._is_contiguous
                and identity_gate
                and surrogate["passed"]
            ),
        }
    return results


def _backend_cases(name):
    X, y = _sample(9701)
    device, Xb, yb, device_name = _backend(name, X, y)
    cases = {}
    for kind in ("group_mcp", "group_scad"):
        coef_probe = np.array([0.15, 0.9, -0.7, 0.05])
        _, coef_backend, _, _ = _backend(name, coef_probe, coef_probe)
        penalty = _penalty(kind, INTERLEAVED)
        expected_lla = _expected_lla(kind, coef_probe)
        actual_lla = _as_numpy(penalty.lla_weights(coef_backend))
        lla_error = float(np.max(np.abs(actual_lla - expected_lla)))

        cpu = _fit(kind, X, y, INTERLEAVED, "cpu")
        gpu = _fit(kind, Xb, yb, INTERLEAVED, device)
        coef_error = float(
            np.max(np.abs(_as_numpy(gpu.coef_) - np.asarray(cpu.coef_)))
        )
        prediction_error = float(
            np.max(
                np.abs(
                    _as_numpy(gpu.predict(Xb)) - np.asarray(cpu.predict(X))
                )
            )
        )
        intercept_error = abs(float(gpu.intercept_) - float(cpu.intercept_))

        grouped = _fit(kind, X[:, PERM], y, GROUPED, "cpu")
        layout_error = float(
            np.max(
                np.abs(
                    np.asarray(cpu.coef_)
                    - np.asarray(grouped.coef_)[INVERSE_PERM]
                )
            )
        )

        cpu_cv = _cv(kind, X, y, INTERLEAVED, "cpu")
        gpu_cv = _cv(kind, Xb, yb, INTERLEAVED, device)
        score_error = float(
            np.max(
                np.abs(
                    np.asarray(gpu_cv.cv_results_["all_scores"])
                    - np.asarray(cpu_cv.cv_results_["all_scores"])
                )
            )
        )
        cv_coef_error = float(
            np.max(
                np.abs(
                    _as_numpy(gpu_cv.coef_) - np.asarray(cpu_cv.coef_)
                )
            )
        )
        selected_equal = bool(np.isclose(gpu_cv.alpha_, cpu_cv.alpha_))
        refit_equal = bool(np.isclose(gpu_cv.estimator_.alpha, gpu_cv.alpha_))

        passed = all(
            (
                lla_error <= 2e-12,
                coef_error <= 3e-5,
                prediction_error <= 3e-5,
                intercept_error <= 3e-5,
                layout_error <= 2e-6,
                score_error <= 3e-5,
                cv_coef_error <= 3e-5,
                selected_equal,
                refit_equal,
            )
        )
        cases[kind] = {
            "lla_max_abs_error": lla_error,
            "direct_coef_max_abs_error": coef_error,
            "direct_prediction_max_abs_error": prediction_error,
            "direct_intercept_abs_error": intercept_error,
            "cpu_interleaved_vs_grouped_coef_error": layout_error,
            "cv_score_max_abs_error": score_error,
            "cv_coef_max_abs_error": cv_coef_error,
            "selected_alpha": float(gpu_cv.alpha_),
            "cpu_selected_alpha": float(cpu_cv.alpha_),
            "final_refit_alpha": float(gpu_cv.estimator_.alpha),
            "passed": bool(passed),
        }
    return device_name, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    api_contract = _api_contract()
    report = {
        "schema_version": 3,
        "validation_tier": "remote-full",
        "source_commit": head,
        "source_clean": not dirty,
        "source_sha256": {path: _sha256(path) for path in SOURCE_FILES},
        "command": (
            "python dev/benchmarks/benchmark_group_nonconvex_layout_gpu.py "
            "--output <path>"
        ),
        "api_contract": api_contract,
        "backends": {},
        "gate_failures": [],
    }

    if not api_contract["solver_export_passed"]:
        report["gate_failures"].append("group LLA solver export")
    for kind, contract in api_contract["penalties"].items():
        if not contract["passed"]:
            report["gate_failures"].append(f"{kind}: public API or surrogate")

    for name in ("cupy", "torch"):
        try:
            device_name, cases = _backend_cases(name)
            passed = all(case["passed"] for case in cases.values())
            report["backends"][name] = {
                "device": device_name,
                "cases": cases,
                "passed": bool(passed),
            }
            if not passed:
                report["gate_failures"].append(
                    f"{name}: group MCP/SCAD layout parity"
                )
        except Exception as exc:
            report["backends"][name] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["gate_failures"].append(
                f"{name}: {type(exc).__name__}"
            )

    if dirty:
        report["gate_failures"].append("source tree is dirty")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
