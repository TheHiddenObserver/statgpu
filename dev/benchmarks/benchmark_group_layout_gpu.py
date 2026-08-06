#!/usr/bin/env python3
"""Exact-source physical-GPU contract for Group Lasso layout semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import AdaptiveGroupLassoPenalty, GroupLassoPenalty


SOURCE_FILES = (
    "dev/benchmarks/benchmark_group_layout_gpu.py",
    "dev/tests/test_pr80_group_layout_contract.py",
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "statgpu/linear_model/penalized/_penalized_cv.py",
    "statgpu/penalties/__init__.py",
    "statgpu/penalties/_group_lasso.py",
    "statgpu/penalties/_group_lasso_layout.py",
)

LAYOUTS = {
    "equal_noncontiguous": [[0, 2], [1, 3]],
    "misleading_first_index": [[3, 0], [2, 1]],
    "unequal_serial": [[0, 3, 4], [1, 2]],
}
CV_LAYOUTS = {
    "misleading_first_index": LAYOUTS["misleading_first_index"],
    "unequal_serial": LAYOUTS["unequal_serial"],
}
LEGACY_LAYOUT = [[0, 3], [2, 1]]


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


def _sample(seed, p):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, p))
    beta = np.linspace(0.8, -0.35, p)
    y = 0.4 + X @ beta + rng.normal(scale=0.04, size=X.shape[0])
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


def _fit(
    X,
    y,
    groups,
    *,
    device,
    fit_intercept,
    penalty=None,
):
    penalty_arg = penalty if penalty is not None else "group_lasso"
    penalty_kwargs = None if penalty is not None else {"groups": groups}
    alpha = float(penalty.alpha) if penalty is not None else 0.035
    return PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=penalty_arg,
        alpha=alpha,
        penalty_kwargs=penalty_kwargs,
        solver="auto",
        device=device,
        fit_intercept=fit_intercept,
        compute_inference=False,
        max_iter=3000,
        tol=1e-10,
    ).fit(X, y)


def _objective(model, X_input, y_np, groups):
    coef = _as_numpy(model.coef_).astype(np.float64, copy=False)
    pred = _as_numpy(model.predict(X_input)).astype(np.float64, copy=False)
    loss = 0.5 * float(np.mean((y_np - pred) ** 2))
    penalty = sum(
        np.sqrt(len(group)) * np.linalg.norm(coef[np.asarray(group, dtype=int)])
        for group in groups
    )
    return loss + 0.035 * float(penalty)


def _canonical_metadata(model):
    penalty = getattr(model, "_penalty", None)
    return {
        "groups": [
            np.asarray(group, dtype=int).tolist()
            for group in penalty._group_indices
        ],
        "all_equal_size": bool(penalty._all_equal_size),
        "is_contiguous": bool(penalty._is_contiguous),
        "flat_indices": None
        if penalty._flat_indices is None
        else np.asarray(penalty._flat_indices, dtype=int).tolist(),
    }


def _legacy_pickled_penalty(groups=LEGACY_LAYOUT):
    current = GroupLassoPenalty(alpha=0.035, groups=groups)
    state = dict(current.__dict__)
    state["_group_indices"] = [
        np.asarray(group, dtype=np.int64) for group in groups
    ]
    state["_is_contiguous"] = True
    state["_flat_indices"] = None
    state["_all_equal_size"] = len({len(group) for group in groups}) == 1
    state["_group_size_uniform"] = (
        len(groups[0]) if state["_all_equal_size"] else None
    )
    legacy = object.__new__(GroupLassoPenalty)
    legacy.__dict__.update(state)
    return pickle.loads(pickle.dumps(legacy))


def _api_contract():
    adaptive = AdaptiveGroupLassoPenalty(
        groups=[[3, 0], [2, 1]],
        alpha=0.035,
        weights=np.array([1.0, 1.5]),
    )
    restored = pickle.loads(pickle.dumps(adaptive))
    passed = all(
        (
            issubclass(AdaptiveGroupLassoPenalty, GroupLassoPenalty),
            isinstance(adaptive, GroupLassoPenalty),
            isinstance(restored, GroupLassoPenalty),
            not adaptive._is_contiguous,
            not restored._is_contiguous,
            np.array_equal(adaptive._flat_indices, np.array([0, 3, 1, 2])),
            np.array_equal(restored._flat_indices, np.array([0, 3, 1, 2])),
            np.allclose(restored._group_weights, np.array([1.0, 1.5])),
        )
    )
    return {
        "adaptive_is_group_lasso_subclass": bool(
            issubclass(AdaptiveGroupLassoPenalty, GroupLassoPenalty)
        ),
        "adaptive_groups": [
            np.asarray(group, dtype=int).tolist()
            for group in adaptive._group_indices
        ],
        "adaptive_pickle_groups": [
            np.asarray(group, dtype=int).tolist()
            for group in restored._group_indices
        ],
        "passed": bool(passed),
    }


def _direct_cases(name):
    results = {}
    device_name = None
    for layout_name, groups in LAYOUTS.items():
        p = max(max(group) for group in groups) + 1
        X, y = _sample(9300 + p, p)
        for fit_intercept in (False, True):
            key = f"{layout_name}__intercept_{int(fit_intercept)}"
            reference = _fit(
                X, y, groups, device="cpu", fit_intercept=fit_intercept
            )
            device, Xb, yb, device_name = _backend(name, X, y)
            actual = _fit(
                Xb, yb, groups, device=device, fit_intercept=fit_intercept
            )
            coef_error = float(
                np.max(
                    np.abs(
                        _as_numpy(actual.coef_)
                        - np.asarray(reference.coef_)
                    )
                )
            )
            prediction_error = float(
                np.max(
                    np.abs(
                        _as_numpy(actual.predict(Xb))
                        - np.asarray(reference.predict(X))
                    )
                )
            )
            intercept_error = abs(
                float(actual.intercept_) - float(reference.intercept_)
            )
            objective_error = abs(
                _objective(actual, Xb, y, groups)
                - _objective(reference, X, y, groups)
            )
            metadata = _canonical_metadata(actual)
            expected_groups = [
                sorted(int(index) for index in group) for group in groups
            ]
            passed = all(
                (
                    coef_error <= 2e-5,
                    prediction_error <= 2e-5,
                    intercept_error <= 2e-5,
                    objective_error <= 2e-6,
                    metadata["groups"] == expected_groups,
                    not metadata["is_contiguous"],
                )
            )
            results[key] = {
                "groups_input": groups,
                "metadata": metadata,
                "coef_max_abs_error": coef_error,
                "prediction_max_abs_error": prediction_error,
                "intercept_abs_error": intercept_error,
                "objective_abs_error": objective_error,
                "passed": bool(passed),
            }
    return device_name, results


def _legacy_cases(name):
    results = {}
    X, y = _sample(9350, 4)
    device, Xb, yb, device_name = _backend(name, X, y)
    for fit_intercept in (False, True):
        key = f"legacy_pickle__intercept_{int(fit_intercept)}"
        reference_penalty = _legacy_pickled_penalty()
        actual_penalty = _legacy_pickled_penalty()
        reference = _fit(
            X,
            y,
            LEGACY_LAYOUT,
            device="cpu",
            fit_intercept=fit_intercept,
            penalty=reference_penalty,
        )
        actual = _fit(
            Xb,
            yb,
            LEGACY_LAYOUT,
            device=device,
            fit_intercept=fit_intercept,
            penalty=actual_penalty,
        )
        coef_error = float(
            np.max(np.abs(_as_numpy(actual.coef_) - np.asarray(reference.coef_)))
        )
        prediction_error = float(
            np.max(
                np.abs(
                    _as_numpy(actual.predict(Xb))
                    - np.asarray(reference.predict(X))
                )
            )
        )
        intercept_error = abs(
            float(actual.intercept_) - float(reference.intercept_)
        )
        objective_error = abs(
            _objective(actual, Xb, y, LEGACY_LAYOUT)
            - _objective(reference, X, y, LEGACY_LAYOUT)
        )
        metadata = _canonical_metadata(actual)
        passed = all(
            (
                coef_error <= 2e-5,
                prediction_error <= 2e-5,
                intercept_error <= 2e-5,
                objective_error <= 2e-6,
                metadata["groups"] == [[0, 3], [1, 2]],
                metadata["flat_indices"] == [0, 3, 1, 2],
                not metadata["is_contiguous"],
            )
        )
        results[key] = {
            "serialized_groups_input": LEGACY_LAYOUT,
            "metadata_after_restore": metadata,
            "coef_max_abs_error": coef_error,
            "prediction_max_abs_error": prediction_error,
            "intercept_abs_error": intercept_error,
            "objective_abs_error": objective_error,
            "passed": bool(passed),
        }
    return device_name, results


def _cv_cases(name):
    results = {}
    for layout_name, groups in CV_LAYOUTS.items():
        p = max(max(group) for group in groups) + 1
        X, y = _sample(9400 + p, p)
        kwargs = dict(
            loss="squared_error",
            penalty="group_lasso",
            penalty_kwargs={"groups": groups},
            alpha_grid=[0.12, 0.035],
            cv=2,
            random_state=23,
            max_iter=2500,
            tol=1e-9,
        )
        reference = PenalizedGLM_CV(device="cpu", **kwargs).fit(X, y)
        device, Xb, yb, _ = _backend(name, X, y)
        actual = PenalizedGLM_CV(device=device, **kwargs).fit(Xb, yb)
        score_error = float(
            np.max(
                np.abs(
                    np.asarray(actual.cv_results_["all_scores"])
                    - np.asarray(reference.cv_results_["all_scores"])
                )
            )
        )
        coef_error = float(
            np.max(
                np.abs(
                    _as_numpy(actual.coef_)
                    - np.asarray(reference.coef_)
                )
            )
        )
        selected_equal = bool(np.isclose(actual.alpha_, reference.alpha_))
        refit_equal = bool(
            np.isclose(actual.estimator_.alpha, actual.alpha_)
        )
        metadata = _canonical_metadata(actual.estimator_)
        passed = all(
            (
                score_error <= 2e-5,
                coef_error <= 2e-5,
                selected_equal,
                refit_equal,
                not metadata["is_contiguous"],
            )
        )
        results[layout_name] = {
            "groups_input": groups,
            "metadata": metadata,
            "score_max_abs_error": score_error,
            "coef_max_abs_error": coef_error,
            "selected_alpha": float(actual.alpha_),
            "cpu_selected_alpha": float(reference.alpha_),
            "final_refit_alpha": float(actual.estimator_.alpha),
            "passed": bool(passed),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    api_contract = _api_contract()
    report = {
        "schema_version": 2,
        "validation_tier": "remote-full",
        "source_commit": head,
        "source_clean": not dirty,
        "source_sha256": {path: _sha256(path) for path in SOURCE_FILES},
        "command": (
            "python dev/benchmarks/benchmark_group_layout_gpu.py "
            "--output <path>"
        ),
        "api_contract": api_contract,
        "backends": {},
        "gate_failures": [],
    }

    if not api_contract["passed"]:
        report["gate_failures"].append("public group penalty class hierarchy")

    for name in ("cupy", "torch"):
        try:
            device_name, direct = _direct_cases(name)
            _, legacy = _legacy_cases(name)
            cv = _cv_cases(name)
            passed = (
                all(case["passed"] for case in direct.values())
                and all(case["passed"] for case in legacy.values())
                and all(case["passed"] for case in cv.values())
            )
            report["backends"][name] = {
                "device": device_name,
                "direct_fit": direct,
                "legacy_pickle": legacy,
                "cv": cv,
                "passed": bool(passed),
            }
            if not passed:
                report["gate_failures"].append(
                    f"{name}: layout or legacy-state parity"
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
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
