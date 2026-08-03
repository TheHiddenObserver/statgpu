#!/usr/bin/env python3
"""Exact-source physical-GPU gate for group CV constructor semantics.

The runner verifies on both CuPy and Torch CUDA that:
- Group Lasso/MCP/SCAD penalty objects evaluate every requested CV alpha;
- object and string penalty forms produce the same fold scores, selection, and
  final refit;
- Adaptive Group Lasso object templates with different initial alpha produce
  the same requested CV path;
- the selected alpha reaches both resolved and public final penalty snapshots;
- fit-local group completion does not mutate public penalty objects or kwargs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.penalties import (
    AdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
)


SOURCE_FILES = (
    "dev/benchmarks/benchmark_group_cv_object_gpu.py",
    "dev/tests/test_pr80_adaptive_group_public_capability_contract.py",
    "dev/tests/test_pr80_group_cv_object_alpha_contract.py",
    "dev/tests/test_pr80_group_penalty_object_isolation_contract.py",
    "dev/tests/test_pr80_group_failed_refit_state_contract.py",
    "dev/tests/test_pr80_group_warm_start_transaction_contract.py",
    "dev/tests/test_pr80_group_input_contract.py",
    "dev/tests/test_pr80_group_nonconvex_hyperparameter_contract.py",
    "statgpu/linear_model/penalized/__init__.py",
    "statgpu/linear_model/penalized/_base.py",
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "statgpu/linear_model/penalized/_penalized_cv.py",
    "statgpu/linear_model/penalized/_group_penalty_model_contract.py",
    "statgpu/penalties/__init__.py",
    "statgpu/penalties/_categories.py",
    "statgpu/penalties/_group_clone_contract.py",
    "statgpu/penalties/_group_dimension_contract.py",
    "statgpu/penalties/_group_lasso.py",
    "statgpu/penalties/_group_lasso_layout.py",
    "statgpu/penalties/_group_mcp.py",
    "statgpu/penalties/_group_scad.py",
    "statgpu/penalties/_group_nonconvex_layout.py",
    "statgpu/solvers/__init__.py",
    "statgpu/solvers/_fista.py",
    "statgpu/solvers/_fista_lla.py",
    "statgpu/solvers/_fista_lla_group_contract.py",
    "statgpu/solvers/_utils.py",
)

GROUPS = [[0, 3], [1, 2]]
ALPHAS = [0.35, 0.025]
_CV_MARKER = "_statgpu_cv_alpha_from_estimator"


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


def _data(seed=11001):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=96)
    z1 = rng.normal(size=96)
    X = np.column_stack(
        [
            z0 + 0.08 * rng.normal(size=96),
            z1 + 0.08 * rng.normal(size=96),
            0.75 * z1 + 0.15 * rng.normal(size=96),
            0.85 * z0 + 0.12 * rng.normal(size=96),
        ]
    )
    y = 0.3 + X @ np.array([0.85, -0.55, 0.3, 0.7])
    y += rng.normal(scale=0.09, size=X.shape[0])
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


def _case_specs():
    return (
        (
            "group_lasso",
            lambda: GroupLassoPenalty(alpha=1.0, groups=GROUPS),
            {"groups": GROUPS},
            "squared_error",
            None,
        ),
        (
            "group_mcp",
            lambda: GroupMCPPenalty(alpha=1.0, gamma=3.0, groups=GROUPS),
            {"groups": GROUPS, "gamma": 3.0},
            "huber",
            {"delta": 1.0},
        ),
        (
            "group_scad",
            lambda: GroupSCADPenalty(alpha=1.0, a=3.7, groups=GROUPS),
            {"groups": GROUPS, "a": 3.7},
            "huber",
            {"delta": 1.0},
        ),
    )


def _fit_cv(penalty, X, y, device, loss, loss_kwargs, penalty_kwargs=None):
    return PenalizedGLM_CV(
        loss=loss,
        loss_kwargs=loss_kwargs,
        penalty=penalty,
        penalty_kwargs=penalty_kwargs,
        alpha_grid=ALPHAS,
        cv=2,
        random_state=47,
        device=device,
        max_iter=900,
        tol=1e-8,
    ).fit(X, y)


def _public_snapshot_ok(cv, original):
    fitted = cv.estimator_.penalty
    return bool(
        type(fitted) is type(original)
        and fitted is not original
        and np.isclose(fitted.alpha, cv.alpha_, rtol=0.0, atol=1e-14)
        and fitted.groups == cv.estimator_._penalty.groups
        and not hasattr(fitted, _CV_MARKER)
    )


def _object_cases(name):
    X, y = _data()
    device, Xb, yb, device_name = _backend(name, X, y)
    cases = {}

    for penalty_name, factory, kwargs, loss, loss_kwargs in _case_specs():
        penalty_object = factory()
        object_cv = _fit_cv(
            penalty_object, Xb, yb, device, loss, loss_kwargs
        )
        string_cv = _fit_cv(
            penalty_name,
            Xb,
            yb,
            device,
            loss,
            loss_kwargs,
            penalty_kwargs=kwargs,
        )

        object_scores = np.asarray(object_cv.cv_results_["all_scores"])
        string_scores = np.asarray(string_cv.cv_results_["all_scores"])
        score_error = float(np.max(np.abs(object_scores - string_scores)))
        coef_error = float(
            np.max(
                np.abs(
                    _as_numpy(object_cv.coef_)
                    - _as_numpy(string_cv.coef_)
                )
            )
        )
        columns_distinct = not np.allclose(
            object_scores[:, 0], object_scores[:, 1], rtol=1e-10, atol=1e-12
        )
        selected_equal = bool(
            np.isclose(object_cv.alpha_, string_cv.alpha_, rtol=0.0, atol=1e-14)
        )
        final_alpha_equal = bool(
            np.isclose(
                object_cv.estimator_._penalty.alpha,
                object_cv.alpha_,
                rtol=0.0,
                atol=1e-14,
            )
        )
        object_restored = bool(
            object_cv.penalty is penalty_object
            and np.isclose(penalty_object.alpha, 1.0)
            and penalty_object.groups == ((0, 3), (1, 2))
        )
        public_snapshot = _public_snapshot_ok(object_cv, penalty_object)
        passed = bool(
            score_error <= 8e-5
            and coef_error <= 8e-5
            and columns_distinct
            and selected_equal
            and final_alpha_equal
            and object_restored
            and public_snapshot
        )
        cases[penalty_name] = {
            "score_max_abs_error": score_error,
            "coef_max_abs_error": coef_error,
            "score_columns_distinct": bool(columns_distinct),
            "object_selected_alpha": float(object_cv.alpha_),
            "string_selected_alpha": float(string_cv.alpha_),
            "final_penalty_alpha": float(
                object_cv.estimator_._penalty.alpha
            ),
            "object_parameter_restored": object_restored,
            "public_final_snapshot": public_snapshot,
            "passed": passed,
        }

    # Adaptive Group Lasso is object-only. Different template alpha values must
    # not alter a CV path whose actual alpha grid is supplied by the estimator.
    first_parameter = AdaptiveGroupLassoPenalty(
        groups=GROUPS, alpha=1.0, weights=[0.7, 1.4]
    )
    second_parameter = AdaptiveGroupLassoPenalty(
        groups=GROUPS, alpha=2.0, weights=[0.7, 1.4]
    )
    first_cv = _fit_cv(
        first_parameter, Xb, yb, device, "squared_error", None
    )
    second_cv = _fit_cv(
        second_parameter, Xb, yb, device, "squared_error", None
    )
    adaptive_score_error = float(
        np.max(
            np.abs(
                np.asarray(first_cv.cv_results_["all_scores"])
                - np.asarray(second_cv.cv_results_["all_scores"])
            )
        )
    )
    adaptive_coef_error = float(
        np.max(
            np.abs(
                _as_numpy(first_cv.coef_) - _as_numpy(second_cv.coef_)
            )
        )
    )
    adaptive_selected_equal = bool(
        np.isclose(first_cv.alpha_, second_cv.alpha_, rtol=0.0, atol=1e-14)
    )
    adaptive_public = bool(
        _public_snapshot_ok(first_cv, first_parameter)
        and first_cv.estimator_.penalty._group_weights == (0.7, 1.4)
    )
    adaptive_restored = bool(
        first_cv.penalty is first_parameter
        and second_cv.penalty is second_parameter
        and np.isclose(first_parameter.alpha, 1.0)
        and np.isclose(second_parameter.alpha, 2.0)
    )
    adaptive_passed = bool(
        adaptive_score_error <= 8e-5
        and adaptive_coef_error <= 8e-5
        and adaptive_selected_equal
        and adaptive_public
        and adaptive_restored
        and not np.allclose(
            np.asarray(first_cv.cv_results_["all_scores"])[:, 0],
            np.asarray(first_cv.cv_results_["all_scores"])[:, 1],
            rtol=1e-10,
            atol=1e-12,
        )
    )
    cases["adaptive_group_lasso"] = {
        "template_score_max_abs_error": adaptive_score_error,
        "template_coef_max_abs_error": adaptive_coef_error,
        "selected_alpha": float(first_cv.alpha_),
        "templates_restored": adaptive_restored,
        "public_final_snapshot": adaptive_public,
        "passed": adaptive_passed,
    }

    # Fit-local completion for a string penalty: the estimator may use a full
    # group layout, but the constructor kwargs object must remain unchanged.
    incomplete_kwargs = {"groups": [[0, 1], [2]]}
    completion_cv = _fit_cv(
        "group_lasso",
        Xb,
        yb,
        device,
        "squared_error",
        None,
        penalty_kwargs=incomplete_kwargs,
    )
    completion_passed = bool(
        completion_cv._penalty_kwargs is incomplete_kwargs
        and incomplete_kwargs == {"groups": [[0, 1], [2]]}
        and completion_cv.estimator_._penalty.groups
        == ((0, 1), (2,), (3,))
    )
    cases["fit_local_completion"] = {
        "constructor_kwargs_unchanged": bool(
            incomplete_kwargs == {"groups": [[0, 1], [2]]}
        ),
        "final_groups": [
            list(group) for group in completion_cv.estimator_._penalty.groups
        ],
        "passed": completion_passed,
    }
    return device_name, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dirty = bool(_git("status", "--porcelain"))
    report = {
        "schema_version": 2,
        "validation_tier": "remote-full",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_clean": not dirty,
        "source_sha256": {path: _sha256(path) for path in SOURCE_FILES},
        "command": (
            "python dev/benchmarks/benchmark_group_cv_object_gpu.py "
            "--output <path>"
        ),
        "backends": {},
        "gate_failures": [],
    }

    for backend_name in ("cupy", "torch"):
        try:
            device_name, cases = _object_cases(backend_name)
            passed = all(case["passed"] for case in cases.values())
            report["backends"][backend_name] = {
                "device": device_name,
                "cases": cases,
                "passed": bool(passed),
            }
            if not passed:
                report["gate_failures"].append(
                    f"{backend_name}: group CV object/constructor contract"
                )
        except Exception as exc:
            report["backends"][backend_name] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["gate_failures"].append(
                f"{backend_name}: {type(exc).__name__}"
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
