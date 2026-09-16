#!/usr/bin/env python3
"""Physical CUDA acceptance for PR #164 / Issue #163.

This validator is intentionally narrow. It validates the maintained Quantile
solver-provenance and CV-scoring contract changed by PR #164 on one physical
CUDA device through both CuPy and Torch:

* smooth L2 ``solver="auto"`` executes/reports IRLS;
* sparse L1 ``auto`` remains FISTA-family;
* SCAD ``auto`` executes/reports the dedicated Proximal IRLS-CD route;
* non-median Quantile CV scores use the requested ``q``;
* public ``cv_strategy="two_stage"`` L1 and SCAD CV remain executable after
  incomplete private fast paths are routed back to the maintained per-fold
  estimator implementation.

The gate requires a clean exact-source worktree and both CuPy CUDA and Torch
CUDA. It records no timing or speedup claim.

Example
-------
python dev/benchmarks/validate_quantile_solver_provenance_gpu.py \
  --output dev/reviews/pr164_quantile_solver_provenance_gpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedQuantileRegression
from statgpu.penalties import SCADPenalty


SCHEMA_VERSION = 1
Q = 0.20
SCAD_ALPHA = 0.025
ATOL_L2_COEF = 2e-5
ATOL_L2_SCORE = 2e-5
ATOL_SCAD_OBJECTIVE = 2e-4


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _require_clean_source() -> str:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "physical PR164 validation requires a clean exact-source worktree"
        )
    return sha


def _require_gpu_backends():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("CuPy is required for PR164 physical validation") from exc
    try:
        import torch
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("Torch is required for PR164 physical validation") from exc

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")

    cp.cuda.Device(0).use()
    torch.cuda.set_device(0)
    return cp, torch


def _data(seed=16401, n=128, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.78, -0.42, 0.24], dtype=np.float64)[:p]
    # Deliberately asymmetric noise: q=.20 and q=.50 validation objectives are
    # observably different, so the gate can detect a median-scoring regression.
    noise = rng.exponential(scale=0.28, size=n) - 0.28
    y = (0.22 + X @ beta + noise).astype(np.float64)
    weights = np.linspace(0.55, 1.65, n, dtype=np.float64)
    rng.shuffle(weights)
    half = n // 2
    folds = [
        (np.arange(0, half), np.arange(half, n)),
        (np.arange(half, n), np.arange(0, half)),
    ]
    return X, y, weights, folds


def _to_host(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _pinball(y, eta, q, sample_weight=None) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    eta = np.asarray(eta, dtype=np.float64).ravel()
    u = y - eta
    values = np.where(u >= 0.0, float(q) * u, (float(q) - 1.0) * u)
    if sample_weight is None:
        return float(np.mean(values))
    return float(
        np.average(values, weights=np.asarray(sample_weight, dtype=np.float64))
    )


def _scad_objective(X, y, weights, coef, intercept, *, alpha=SCAD_ALPHA) -> float:
    """Return the declared weighted Quantile + feature-only SCAD objective."""
    coef = np.asarray(coef, dtype=np.float64).ravel()
    data_fit = _pinball(y, X @ coef + float(intercept), Q, weights)
    penalty = SCADPenalty(alpha=float(alpha)).value(coef)
    return float(data_fit + penalty)


def _max_abs(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(aa - bb)))


def _native_inputs(backend, X, y, weights, cp, torch):
    if backend == "cupy":
        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            cp.asarray(weights, dtype=cp.float64),
            "cuda",
        )
    if backend == "torch":
        device = torch.device("cuda:0")
        return (
            torch.as_tensor(X, dtype=torch.float64, device=device),
            torch.as_tensor(y, dtype=torch.float64, device=device),
            torch.as_tensor(weights, dtype=torch.float64, device=device),
            "torch",
        )
    return X, y, weights, "cpu"


def _assert_provenance(name, model, expected_solver, expected_backend, expected_device):
    solver = str(getattr(model, "_selected_solver", "") or "")
    backend = str(getattr(model, "_selected_backend_name", "") or "")
    device = str(getattr(model, "_selected_backend_device", "") or "")
    if solver != expected_solver:
        raise AssertionError(
            f"{name}: executed solver mismatch: {solver!r} != {expected_solver!r}"
        )
    if backend != expected_backend:
        raise AssertionError(
            f"{name}: backend mismatch: {backend!r} != {expected_backend!r}"
        )
    if device != expected_device:
        raise AssertionError(
            f"{name}: device mismatch: {device!r} != {expected_device!r}"
        )
    return {"solver": solver, "backend": backend, "device": device}


def _direct_fit(X, y, weights, *, penalty, alpha, device, max_iter, tol):
    model = PenalizedQuantileRegression(
        quantile=Q,
        penalty=penalty,
        alpha=alpha,
        solver="auto",
        device=device,
        max_iter=max_iter,
        tol=tol,
    ).fit(X, y, sample_weight=weights)
    coef = _to_host(model.coef_).ravel()
    intercept = float(model.intercept_)
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        raise AssertionError(f"direct {penalty}: non-finite fitted parameters")
    return model, coef, intercept


def _cv_fit(
    X,
    y,
    weights,
    folds,
    *,
    penalty,
    alpha_grid,
    device,
    cv_strategy="strict",
    max_iter=500,
    tol=1e-8,
):
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=penalty,
        alpha_grid=np.asarray(alpha_grid, dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=164,
        solver="auto",
        device=device,
        cv_strategy=cv_strategy,
        acknowledge_approx=(cv_strategy == "two_stage"),
        refine_top_k=1,
        max_iter=max_iter,
        tol=tol,
    ).fit(X, y, sample_weight=weights)
    if str(model.cv_strategy_) != cv_strategy:
        raise AssertionError(
            f"{penalty} CV strategy drifted: {model.cv_strategy_!r} != {cv_strategy!r}"
        )
    return model


def _snapshot_cv(model):
    scores = np.asarray(model.cv_results_["all_scores"], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise AssertionError("CV produced non-finite strict candidate scores")
    stage1 = model.cv_results_.get("all_scores_stage1")
    if stage1 is not None:
        stage1 = np.asarray(stage1, dtype=np.float64)
        if not np.all(np.isfinite(stage1)):
            raise AssertionError("two-stage CV produced non-finite screening scores")
    refined_mask = model.cv_results_.get("refined_mask")
    if refined_mask is not None:
        refined_mask = np.asarray(refined_mask, dtype=bool)
    return {
        "alpha": float(model.alpha_),
        "scores": scores,
        "stage1_scores": stage1,
        "refined_mask": refined_mask,
        "coef": _to_host(model.coef_).ravel(),
        "intercept": float(model.intercept_),
    }


def _manual_fold_scores(X, y, weights, folds, alpha):
    scores = []
    median_scores = []
    for train_idx, val_idx in folds:
        fit = PenalizedQuantileRegression(
            quantile=Q,
            penalty="l2",
            alpha=float(alpha),
            solver="irls",
            device="cpu",
            max_iter=600,
            tol=1e-9,
        ).fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weights[train_idx],
        )
        eta = X[val_idx] @ np.asarray(fit.coef_, dtype=np.float64) + float(
            fit.intercept_
        )
        scores.append(_pinball(y[val_idx], eta, Q, weights[val_idx]))
        median_scores.append(_pinball(y[val_idx], eta, 0.5, weights[val_idx]))
    return np.asarray(scores), np.asarray(median_scores)


def _environment(cp, torch):
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props.get("name", b"") if isinstance(props, dict) else b""
    if isinstance(raw_name, bytes):
        cupy_name = raw_name.decode("utf-8", errors="replace")
    else:
        cupy_name = str(raw_name)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "device_ordinal": 0,
        "cupy_device_name": cupy_name,
        "torch_device_name": str(torch.cuda.get_device_name(0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="dev/reviews/pr164_quantile_solver_provenance_gpu.json",
    )
    args = parser.parse_args()

    source_sha = _require_clean_source()
    cp, torch = _require_gpu_backends()
    X, y, weights, folds = _data()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "source_clean": True,
        "status": "running",
        "quantile": Q,
        "environment": _environment(cp, torch),
        "tolerances": {
            "l2_coef_intercept": ATOL_L2_COEF,
            "l2_cv_score": ATOL_L2_SCORE,
            "scad_penalized_objective": ATOL_SCAD_OBJECTIVE,
        },
        "cases": [],
    }

    # CPU references are part of the exact-source gate, not external evidence.
    cpu_l2_model, cpu_l2_coef, cpu_l2_intercept = _direct_fit(
        X,
        y,
        weights,
        penalty="l2",
        alpha=0.03,
        device="cpu",
        max_iter=600,
        tol=1e-9,
    )
    _assert_provenance("cpu/direct/l2", cpu_l2_model, "irls", "numpy", "cpu")

    cpu_l1_model, _, _ = _direct_fit(
        X,
        y,
        weights,
        penalty="l1",
        alpha=0.025,
        device="cpu",
        max_iter=800,
        tol=1e-7,
    )
    _assert_provenance("cpu/direct/l1", cpu_l1_model, "fista", "numpy", "cpu")

    cpu_scad_model, cpu_scad_coef, cpu_scad_intercept = _direct_fit(
        X,
        y,
        weights,
        penalty="scad",
        alpha=SCAD_ALPHA,
        device="cpu",
        max_iter=220,
        tol=1e-6,
    )
    _assert_provenance(
        "cpu/direct/scad",
        cpu_scad_model,
        "proximal_irls_cd",
        "numpy",
        "cpu",
    )
    cpu_scad_objective = _scad_objective(
        X, y, weights, cpu_scad_coef, cpu_scad_intercept
    )

    l2_cv_alphas = np.asarray([0.045, 0.02], dtype=np.float64)
    cpu_l2_cv = _cv_fit(
        X,
        y,
        weights,
        folds,
        penalty="l2",
        alpha_grid=l2_cv_alphas,
        device="cpu",
        cv_strategy="strict",
        max_iter=600,
        tol=1e-9,
    )
    cpu_l2_cv_snap = _snapshot_cv(cpu_l2_cv)
    _assert_provenance(
        "cpu/cv/l2/final",
        cpu_l2_cv.estimator_,
        "irls",
        "numpy",
        "cpu",
    )

    for alpha_index, alpha in enumerate(l2_cv_alphas):
        expected, median = _manual_fold_scores(X, y, weights, folds, float(alpha))
        observed = cpu_l2_cv_snap["scores"][:, alpha_index]
        err = _max_abs(observed, expected)
        if err > ATOL_L2_SCORE:
            raise AssertionError(
                f"cpu/cv/l2 alpha={alpha}: manual q-score mismatch {err:.3e}"
            )
        if float(np.max(np.abs(observed - median))) <= 1e-4:
            raise AssertionError(
                "non-median fixture does not distinguish q=.20 from q=.50 scoring"
            )

    sparse_cv_alphas = np.asarray([0.04, 0.02], dtype=np.float64)
    cpu_l1_cv = _cv_fit(
        X,
        y,
        weights,
        folds,
        penalty="l1",
        alpha_grid=sparse_cv_alphas,
        device="cpu",
        cv_strategy="two_stage",
        max_iter=700,
        tol=1e-7,
    )
    cpu_l1_cv_snap = _snapshot_cv(cpu_l1_cv)
    _assert_provenance(
        "cpu/cv/l1/two_stage/final",
        cpu_l1_cv.estimator_,
        "fista",
        "numpy",
        "cpu",
    )
    if cpu_l1_cv_snap["stage1_scores"] is None:
        raise AssertionError("two-stage L1 CPU reference did not publish stage-1 scores")

    cpu_scad_cv = _cv_fit(
        X,
        y,
        weights,
        folds,
        penalty="scad",
        alpha_grid=np.asarray([SCAD_ALPHA], dtype=np.float64),
        device="cpu",
        cv_strategy="two_stage",
        max_iter=180,
        tol=1e-6,
    )
    cpu_scad_cv_snap = _snapshot_cv(cpu_scad_cv)
    _assert_provenance(
        "cpu/cv/scad/two_stage/final",
        cpu_scad_cv.estimator_,
        "proximal_irls_cd",
        "numpy",
        "cpu",
    )
    if cpu_scad_cv_snap["stage1_scores"] is None:
        raise AssertionError("two-stage SCAD CPU reference did not publish stage-1 scores")

    max_errors = {
        "l2_coef_intercept": 0.0,
        "l2_cv_score": 0.0,
        "scad_penalized_objective": 0.0,
    }

    for backend in ("cupy", "torch"):
        Xb, yb, wb, device = _native_inputs(backend, X, y, weights, cp, torch)
        expected_device = "cuda:0"

        l2_model, l2_coef, l2_intercept = _direct_fit(
            Xb,
            yb,
            wb,
            penalty="l2",
            alpha=0.03,
            device=device,
            max_iter=600,
            tol=1e-9,
        )
        l2_provenance = _assert_provenance(
            f"{backend}/direct/l2",
            l2_model,
            "irls",
            backend,
            expected_device,
        )
        coef_error = _max_abs(l2_coef, cpu_l2_coef)
        intercept_error = abs(l2_intercept - cpu_l2_intercept)
        l2_error = max(coef_error, intercept_error)
        max_errors["l2_coef_intercept"] = max(
            max_errors["l2_coef_intercept"], l2_error
        )
        if l2_error > ATOL_L2_COEF:
            raise AssertionError(
                f"{backend}/direct/l2 parity failed: {l2_error:.3e}"
            )
        payload["cases"].append(
            {
                "name": f"{backend}/direct/l2",
                "provenance": l2_provenance,
                "coef_error": coef_error,
                "intercept_error": intercept_error,
            }
        )

        l1_model, _, _ = _direct_fit(
            Xb,
            yb,
            wb,
            penalty="l1",
            alpha=0.025,
            device=device,
            max_iter=800,
            tol=1e-7,
        )
        l1_provenance = _assert_provenance(
            f"{backend}/direct/l1",
            l1_model,
            "fista",
            backend,
            expected_device,
        )
        payload["cases"].append(
            {"name": f"{backend}/direct/l1", "provenance": l1_provenance}
        )

        scad_model, scad_coef, scad_intercept = _direct_fit(
            Xb,
            yb,
            wb,
            penalty="scad",
            alpha=SCAD_ALPHA,
            device=device,
            max_iter=220,
            tol=1e-6,
        )
        scad_provenance = _assert_provenance(
            f"{backend}/direct/scad",
            scad_model,
            "proximal_irls_cd",
            backend,
            expected_device,
        )
        scad_objective = _scad_objective(
            X, y, weights, scad_coef, scad_intercept
        )
        scad_objective_error = abs(scad_objective - cpu_scad_objective)
        max_errors["scad_penalized_objective"] = max(
            max_errors["scad_penalized_objective"], scad_objective_error
        )
        if scad_objective_error > ATOL_SCAD_OBJECTIVE:
            raise AssertionError(
                f"{backend}/direct/scad penalized-objective parity failed: "
                f"{scad_objective_error:.3e}"
            )
        payload["cases"].append(
            {
                "name": f"{backend}/direct/scad",
                "provenance": scad_provenance,
                "penalized_objective": scad_objective,
                "cpu_penalized_objective": cpu_scad_objective,
                "objective_error": scad_objective_error,
            }
        )

        l2_cv = _cv_fit(
            Xb,
            yb,
            wb,
            folds,
            penalty="l2",
            alpha_grid=l2_cv_alphas,
            device=device,
            cv_strategy="strict",
            max_iter=600,
            tol=1e-9,
        )
        l2_cv_snap = _snapshot_cv(l2_cv)
        l2_cv_provenance = _assert_provenance(
            f"{backend}/cv/l2/final",
            l2_cv.estimator_,
            "irls",
            backend,
            expected_device,
        )
        score_error = _max_abs(l2_cv_snap["scores"], cpu_l2_cv_snap["scores"])
        max_errors["l2_cv_score"] = max(max_errors["l2_cv_score"], score_error)
        if score_error > ATOL_L2_SCORE:
            raise AssertionError(
                f"{backend}/cv/l2 score parity failed: {score_error:.3e}"
            )
        if l2_cv_snap["alpha"] != cpu_l2_cv_snap["alpha"]:
            raise AssertionError(
                f"{backend}/cv/l2 selected alpha drifted: "
                f"{l2_cv_snap['alpha']} != {cpu_l2_cv_snap['alpha']}"
            )
        payload["cases"].append(
            {
                "name": f"{backend}/cv/l2",
                "provenance": l2_cv_provenance,
                "selected_alpha": l2_cv_snap["alpha"],
                "score_error": score_error,
                "scores": l2_cv_snap["scores"].tolist(),
            }
        )

        l1_cv = _cv_fit(
            Xb,
            yb,
            wb,
            folds,
            penalty="l1",
            alpha_grid=sparse_cv_alphas,
            device=device,
            cv_strategy="two_stage",
            max_iter=700,
            tol=1e-7,
        )
        l1_cv_snap = _snapshot_cv(l1_cv)
        l1_cv_provenance = _assert_provenance(
            f"{backend}/cv/l1/two_stage/final",
            l1_cv.estimator_,
            "fista",
            backend,
            expected_device,
        )
        if l1_cv_snap["stage1_scores"] is None:
            raise AssertionError(
                f"{backend}/cv/l1/two_stage did not publish stage-1 scores"
            )
        if l1_cv_snap["alpha"] not in set(sparse_cv_alphas.tolist()):
            raise AssertionError(
                f"{backend}/cv/l1/two_stage selected unexpected alpha "
                f"{l1_cv_snap['alpha']}"
            )
        payload["cases"].append(
            {
                "name": f"{backend}/cv/l1/two_stage",
                "provenance": l1_cv_provenance,
                "selected_alpha": l1_cv_snap["alpha"],
                "stage1_scores": l1_cv_snap["stage1_scores"].tolist(),
                "strict_scores": l1_cv_snap["scores"].tolist(),
                "cpu_selected_alpha": cpu_l1_cv_snap["alpha"],
            }
        )

        scad_cv = _cv_fit(
            Xb,
            yb,
            wb,
            folds,
            penalty="scad",
            alpha_grid=np.asarray([SCAD_ALPHA], dtype=np.float64),
            device=device,
            cv_strategy="two_stage",
            max_iter=180,
            tol=1e-6,
        )
        scad_cv_snap = _snapshot_cv(scad_cv)
        scad_cv_provenance = _assert_provenance(
            f"{backend}/cv/scad/two_stage/final",
            scad_cv.estimator_,
            "proximal_irls_cd",
            backend,
            expected_device,
        )
        if scad_cv_snap["stage1_scores"] is None:
            raise AssertionError(
                f"{backend}/cv/scad/two_stage did not publish stage-1 scores"
            )
        if scad_cv_snap["alpha"] != SCAD_ALPHA:
            raise AssertionError(
                f"{backend}/cv/scad/two_stage selected unexpected alpha "
                f"{scad_cv_snap['alpha']}"
            )
        payload["cases"].append(
            {
                "name": f"{backend}/cv/scad/two_stage",
                "provenance": scad_cv_provenance,
                "selected_alpha": scad_cv_snap["alpha"],
                "stage1_scores": scad_cv_snap["stage1_scores"].tolist(),
                "strict_scores": scad_cv_snap["scores"].tolist(),
            }
        )

    payload["max_errors"] = max_errors
    payload["status"] = "success"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
