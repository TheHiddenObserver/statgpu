#!/usr/bin/env python3
"""Physical CUDA acceptance for explicit smooth Quantile FISTA in PR #166.

This focused validator covers explicit ``solver='fista'`` on Quantile L2 and
no-penalty objectives and the standalone non-median batched-bootstrap path
repaired during the PR #166 review/fix loop. ``solver='auto'`` intentionally
remains IRLS and is validated by the existing PR #164 artifact.

The gate exercises both CuPy CUDA and Torch CUDA, including non-uniform analytic
weights, direct L2/no-penalty fits, strict L2 CV, and a tau=0.20 standalone
bootstrap direction check.  It also replaces
``QuantileLoss.irls`` with a forbidden sentinel while the explicit-FISTA cases
run, so a passing result proves that the estimator/CV path did not silently
substitute IRLS.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model import PenalizedGLM_CV, QuantileRegression
from statgpu.linear_model.penalized import PenalizedQuantileRegression
from statgpu.losses import QuantileLoss


SCHEMA_VERSION = 2
Q = 0.35
ATOL_OBJECTIVE = 2e-5
ATOL_CV_SCORE = 2e-5
BOOTSTRAP_Q = 0.20
BOOTSTRAP_N = 80
BOOTSTRAP_B = 12
BOOTSTRAP_SEED = 16692


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _source_state():
    return _git("rev-parse", "HEAD"), not bool(_git("status", "--porcelain"))


def _require_gpu_backends():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("CuPy is required for PR166 smooth-FISTA validation") from exc
    try:
        import torch
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("Torch is required for PR166 smooth-FISTA validation") from exc

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")
    cp.cuda.Device(0).use()
    torch.cuda.set_device(0)
    return cp, torch


def _data(seed=16691, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.82, -0.37, 0.21], dtype=np.float64)[:p]
    y = (0.28 + X @ beta + rng.laplace(scale=0.20, size=n)).astype(np.float64)
    weights = np.linspace(0.55, 1.75, n, dtype=np.float64)
    rng.shuffle(weights)
    half = n // 2
    folds = [
        (np.arange(0, half), np.arange(half, n)),
        (np.arange(half, n), np.arange(0, half)),
    ]
    return X, y, weights, folds


def _host(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _objective(X, y, weights, coef, intercept, alpha):
    coef = np.asarray(coef, dtype=np.float64).ravel()
    residual = np.asarray(y, dtype=np.float64) - (
        np.asarray(X, dtype=np.float64) @ coef + float(intercept)
    )
    pinball = np.where(
        residual >= 0.0,
        Q * residual,
        (Q - 1.0) * residual,
    )
    data_fit = float(np.average(pinball, weights=np.asarray(weights, dtype=np.float64)))
    return data_fit + 0.5 * float(alpha) * float(np.dot(coef, coef))


def _native_inputs(backend, X, y, weights, cp, torch):
    if backend == "cupy":
        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            cp.asarray(weights, dtype=cp.float64),
            "cuda",
        )
    device = torch.device("cuda:0")
    return (
        torch.as_tensor(X, dtype=torch.float64, device=device),
        torch.as_tensor(y, dtype=torch.float64, device=device),
        torch.as_tensor(weights, dtype=torch.float64, device=device),
        "torch",
    )


def _standalone_bootstrap_direction_case(backend, cp, torch):
    """Exercise the backend-specific batched Quantile bootstrap at tau != 0.5."""
    X = np.ones((BOOTSTRAP_N, 1), dtype=np.float64)
    y = np.linspace(-4.0, 4.0, BOOTSTRAP_N, dtype=np.float64)
    if backend == "cupy":
        Xb = cp.asarray(X, dtype=cp.float64)
        yb = cp.asarray(y, dtype=cp.float64)
        device = "cuda:0"
    else:
        torch_device = torch.device("cuda:0")
        Xb = torch.as_tensor(X, dtype=torch.float64, device=torch_device)
        yb = torch.as_tensor(y, dtype=torch.float64, device=torch_device)
        device = str(torch_device)

    model = QuantileRegression(
        quantile=BOOTSTRAP_Q,
        fit_intercept=False,
        max_iter=400,
        tol=1e-7,
        n_bootstrap=BOOTSTRAP_B,
        random_state=BOOTSTRAP_SEED,
        device="cuda" if backend == "cupy" else "torch",
    )
    # The batched bootstrap solver only needs the already-fitted parameter
    # snapshot.  A zero coefficient makes each residual draw exactly a draw
    # from y, giving an independent direction check against known quantiles.
    model.coef_ = np.zeros(1, dtype=np.float64)
    model.intercept_ = 0.0
    boot_params, params_native, design_native = model._compute_bootstrap_batched(Xb, yb)
    estimated = np.asarray(boot_params[:, 0], dtype=np.float64)

    if backend == "cupy":
        if not isinstance(design_native, cp.ndarray):
            raise AssertionError(
                "cupy/standalone/bootstrap: numerical design left the CuPy backend"
            )
        if int(design_native.device.id) != 0:
            raise AssertionError(
                "cupy/standalone/bootstrap: numerical design left CUDA device 0"
            )
        if not isinstance(params_native, cp.ndarray):
            raise AssertionError(
                "cupy/standalone/bootstrap: parameter snapshot left the CuPy backend"
            )
    else:
        if not torch.is_tensor(design_native) or not design_native.is_cuda:
            raise AssertionError(
                "torch/standalone/bootstrap: numerical design left Torch CUDA"
            )
        if str(design_native.device) != "cuda:0":
            raise AssertionError(
                "torch/standalone/bootstrap: numerical design left CUDA device 0"
            )
        if not torch.is_tensor(params_native) or not params_native.is_cuda:
            raise AssertionError(
                "torch/standalone/bootstrap: parameter snapshot left Torch CUDA"
            )

    solver_n_iter = int(model._bootstrap_n_iter_)
    if not 1 <= solver_n_iter <= int(model.max_iter):
        raise AssertionError(
            f"{backend}/standalone/bootstrap: invalid solver_n_iter={solver_n_iter}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_batch = np.array(
        [y[rng.integers(0, BOOTSTRAP_N, size=BOOTSTRAP_N)] for _ in range(BOOTSTRAP_B)]
    )
    requested = np.quantile(y_batch, BOOTSTRAP_Q, axis=1)
    complementary = np.quantile(y_batch, 1.0 - BOOTSTRAP_Q, axis=1)
    target_error = float(np.mean(np.abs(estimated - requested)))
    wrong_direction_error = float(np.mean(np.abs(estimated - complementary)))
    median_estimate = float(np.median(estimated))

    if not np.all(np.isfinite(estimated)):
        raise AssertionError(f"{backend}/standalone/bootstrap: non-finite parameters")
    if not target_error < wrong_direction_error:
        raise AssertionError(
            f"{backend}/standalone/bootstrap: requested-quantile error "
            f"{target_error:.3e} is not below complementary-quantile error "
            f"{wrong_direction_error:.3e}"
        )
    if not median_estimate < 0.0:
        raise AssertionError(
            f"{backend}/standalone/bootstrap: median estimate {median_estimate:.3e} "
            "does not lie on the requested lower-quantile side"
        )

    return {
        "name": f"{backend}/standalone/bootstrap/q{BOOTSTRAP_Q:.2f}",
        "backend": backend,
        "device": device,
        "quantile": BOOTSTRAP_Q,
        "n_bootstrap": BOOTSTRAP_B,
        "solver": "batched_pinball_fista",
        "solver_n_iter": solver_n_iter,
        "target_error": target_error,
        "complementary_quantile_error": wrong_direction_error,
        "median_estimate": median_estimate,
    }


def _provenance(model, backend):
    expected_backend = backend
    solver = str(getattr(model, "_selected_solver", "") or "")
    selected_backend = str(getattr(model, "_selected_backend_name", "") or "")
    device = str(getattr(model, "_selected_backend_device", "") or "")
    if solver != "fista":
        raise AssertionError(f"executed solver drifted: {solver!r} != 'fista'")
    if selected_backend != expected_backend:
        raise AssertionError(
            f"backend drifted: {selected_backend!r} != {expected_backend!r}"
        )
    if device != "cuda:0":
        raise AssertionError(f"device drifted: {device!r} != 'cuda:0'")
    return {"solver": solver, "backend": selected_backend, "device": device}


def _direct(X, y, weights, *, penalty, alpha, device):
    model = PenalizedQuantileRegression(
        quantile=Q,
        penalty=penalty,
        alpha=alpha,
        solver="fista",
        device=device,
        max_iter=5000,
        tol=1e-8,
    ).fit(X, y, sample_weight=weights)
    coef = _host(model.coef_).ravel()
    intercept = float(model.intercept_)
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        raise AssertionError(f"direct {penalty}: non-finite parameters")
    return model, coef, intercept


def _cv(X, y, weights, folds, *, device):
    return PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="l2",
        alpha_grid=np.asarray([0.03, 0.015], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=166,
        solver="fista",
        device=device,
        cv_strategy="strict",
        max_iter=3500,
        tol=1e-8,
    ).fit(X, y, sample_weight=weights)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="dev/reviews/pr166_quantile_smooth_fista_gpu.json",
    )
    args = parser.parse_args()

    source_sha, source_clean = _source_state()
    cp, torch = _require_gpu_backends()
    X, y, weights, folds = _data()

    # CPU references use the same explicit public route. Hosted tests separately
    # align L2 objective with IRLS and no-penalty objective with sklearn HiGHS.
    cpu_l2, cpu_l2_coef, cpu_l2_intercept = _direct(
        X, y, weights, penalty="l2", alpha=0.02, device="cpu"
    )
    cpu_none, cpu_none_coef, cpu_none_intercept = _direct(
        X, y, weights, penalty="none", alpha=0.0, device="cpu"
    )
    cpu_cv = _cv(X, y, weights, folds, device="cpu")
    cpu_cv_scores = np.asarray(cpu_cv.cv_results_["all_scores"], dtype=np.float64)

    def forbidden_irls(*_args, **_kwargs):
        raise AssertionError("explicit Quantile FISTA physically executed IRLS")

    original_irls = QuantileLoss.irls
    QuantileLoss.irls = forbidden_irls
    cases = []
    max_objective_error = 0.0
    max_cv_score_error = 0.0
    try:
        for backend in ("cupy", "torch"):
            Xb, yb, wb, device = _native_inputs(backend, X, y, weights, cp, torch)
            cases.append(_standalone_bootstrap_direction_case(backend, cp, torch))

            for penalty, alpha, cpu_coef, cpu_intercept in (
                ("l2", 0.02, cpu_l2_coef, cpu_l2_intercept),
                ("none", 0.0, cpu_none_coef, cpu_none_intercept),
            ):
                model, coef, intercept = _direct(
                    Xb, yb, wb, penalty=penalty, alpha=alpha, device=device
                )
                gpu_obj = _objective(X, y, weights, coef, intercept, alpha)
                cpu_obj = _objective(
                    X, y, weights, cpu_coef, cpu_intercept, alpha
                )
                objective_error = abs(gpu_obj - cpu_obj)
                max_objective_error = max(max_objective_error, objective_error)
                if objective_error > ATOL_OBJECTIVE:
                    raise AssertionError(
                        f"{backend}/direct/{penalty}: objective error "
                        f"{objective_error:.3e} > {ATOL_OBJECTIVE:.3e}"
                    )
                cases.append(
                    {
                        "name": f"{backend}/direct/{penalty}",
                        "provenance": _provenance(model, backend),
                        "cpu_objective": cpu_obj,
                        "objective": gpu_obj,
                        "objective_error": objective_error,
                    }
                )

            cv = _cv(Xb, yb, wb, folds, device=device)
            if float(cv.alpha_) != float(cpu_cv.alpha_):
                raise AssertionError(
                    f"{backend}/cv/l2: selected alpha {cv.alpha_!r} "
                    f"!= CPU {cpu_cv.alpha_!r}"
                )
            scores = np.asarray(cv.cv_results_["all_scores"], dtype=np.float64)
            score_error = float(np.max(np.abs(scores - cpu_cv_scores)))
            max_cv_score_error = max(max_cv_score_error, score_error)
            if score_error > ATOL_CV_SCORE:
                raise AssertionError(
                    f"{backend}/cv/l2: score error {score_error:.3e} "
                    f"> {ATOL_CV_SCORE:.3e}"
                )
            cases.append(
                {
                    "name": f"{backend}/cv/l2",
                    "provenance": _provenance(cv.estimator_, backend),
                    "selected_alpha": float(cv.alpha_),
                    "cpu_selected_alpha": float(cpu_cv.alpha_),
                    "scores": scores.tolist(),
                    "score_error": score_error,
                }
            )
    finally:
        QuantileLoss.irls = original_irls

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": source_clean,
        "quantile": Q,
        "cases": cases,
        "max_errors": {
            "direct_objective": max_objective_error,
            "cv_score": max_cv_score_error,
        },
        "tolerances": {
            "direct_objective": ATOL_OBJECTIVE,
            "cv_score": ATOL_CV_SCORE,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_ordinal": 0,
            "cupy_device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "torch_device_name": torch.cuda.get_device_name(0),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
