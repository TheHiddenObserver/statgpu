# -*- coding: utf-8 -*-
"""Small external accuracy/runtime benchmark for GLM+penalty fits.

The benchmark is intentionally compact and auditable.  It compares statgpu
CPU/CUDA/Torch against sklearn, statsmodels, and optional R packages using
explicitly equivalent penalty parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.backends import _to_numpy
from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel


@dataclass(frozen=True)
class Case:
    name: str
    loss: str
    penalty: str
    reference: str
    alpha: float
    l1_ratio: float = 0.5
    n: int = 240
    p: int = 12
    seed: int = 123
    solver: str = "auto"
    max_iter: int = 2000
    tol: float = 1e-7
    threshold: float = 5e-3


CASES = [
    Case("gaussian_lasso_sklearn", "squared_error", "l1", "sklearn_lasso", 0.025, solver="fista", threshold=2e-3),
    Case("gaussian_enet_sklearn", "squared_error", "elasticnet", "sklearn_enet", 0.02, solver="fista", threshold=2e-3),
    Case("gaussian_ridge_sklearn", "squared_error", "l2", "sklearn_ridge", 0.04, solver="auto", threshold=1e-6),
    Case("logistic_ridge_sklearn", "logistic", "l2", "sklearn_logistic_l2", 0.03, solver="irls", threshold=5e-3),
    Case("poisson_ridge_sklearn", "poisson", "l2", "sklearn_poisson", 0.025, solver="newton", threshold=5e-3),
    Case("poisson_lasso_statsmodels", "poisson", "l1", "statsmodels_poisson_sparse", 0.018, l1_ratio=1.0, solver="fista", max_iter=3500, tol=1e-8, threshold=6e-3),
    Case("poisson_enet_statsmodels", "poisson", "elasticnet", "statsmodels_poisson_sparse", 0.015, l1_ratio=0.5, solver="fista", max_iter=3500, tol=1e-8, threshold=8e-3),
    Case("gaussian_lasso_r_glmnet", "squared_error", "l1", "r_glmnet_gaussian", 0.025, solver="fista", threshold=5e-3),
    Case("gaussian_scad_r_ncvreg", "squared_error", "scad", "r_ncvreg_gaussian", 0.04, solver="fista", max_iter=2500, tol=1e-7, threshold=2e-2),
    Case("gaussian_mcp_r_ncvreg", "squared_error", "mcp", "r_ncvreg_gaussian", 0.04, solver="fista", max_iter=2500, tol=1e-7, threshold=2e-2),
]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cpu,cuda,torch")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--include-r", action="store_true", default=False)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _make_data(loss, n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.45, size=(n, p))
    beta = np.zeros(p)
    beta[: min(5, p)] = np.linspace(0.55, -0.25, min(5, p))
    intercept = 0.18
    eta = intercept + X @ beta
    if loss == "squared_error":
        y = eta + rng.normal(scale=0.25, size=n)
    elif loss == "logistic":
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -8.0, 8.0)))
        y = (rng.random(n) < prob).astype(float)
    elif loss == "poisson":
        mu = np.exp(np.clip(eta, -2.5, 2.5))
        y = rng.poisson(mu).astype(float)
    else:
        raise ValueError(loss)
    return X.astype(np.float64), y.astype(np.float64)


def _sync_device(device):
    if device == "cuda":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif device == "torch":
        import torch

        torch.cuda.synchronize()


def _available_device(device):
    if device == "cpu":
        return True, None
    if device == "cuda":
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0, None
        except Exception as exc:
            return False, str(exc)
    if device == "torch":
        try:
            import torch

            return bool(torch.cuda.is_available()), None
        except Exception as exc:
            return False, str(exc)
    return False, f"unknown device {device!r}"


def _device_arrays(X, y, device):
    if device == "cuda":
        import cupy as cp

        return cp.asarray(X), cp.asarray(y)
    if device == "torch":
        import torch

        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        )
    return X, y


def _fit_statgpu_once(X_dev, y_dev, case, device):
    model = PenalizedGeneralizedLinearModel(
        loss=case.loss,
        penalty=case.penalty,
        alpha=case.alpha,
        l1_ratio=case.l1_ratio,
        fit_intercept=True,
        solver=case.solver,
        max_iter=case.max_iter,
        tol=case.tol,
        device=device,
        compute_inference=False,
    )
    model.fit(X_dev, y_dev)
    _sync_device(device)
    return model


def _median_timed_fit(fit_fn, repeat, warmup):
    for _ in range(max(0, warmup)):
        fit_fn()
    times = []
    result = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = fit_fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return result, float(np.median(times))


def _coef_intercept(model):
    return np.asarray(_to_numpy(model.coef_), dtype=np.float64).ravel(), float(model.intercept_)


def _objective(loss, penalty, X, y, coef, intercept, alpha, l1_ratio):
    eta = X @ coef + intercept
    if loss == "squared_error":
        value = 0.5 * float(np.mean((y - eta) ** 2))
    elif loss == "logistic":
        value = float(np.mean(np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0.0) - y * eta))
    elif loss == "poisson":
        value = float(np.mean(np.exp(np.clip(eta, -30.0, 30.0)) - y * eta))
    else:
        value = float("nan")

    if penalty == "l2":
        pen = 0.5 * alpha * float(np.sum(coef ** 2))
    elif penalty == "l1":
        pen = alpha * float(np.sum(np.abs(coef)))
    elif penalty in ("elasticnet", "en"):
        pen = alpha * (
            l1_ratio * float(np.sum(np.abs(coef)))
            + 0.5 * (1.0 - l1_ratio) * float(np.sum(coef ** 2))
        )
    elif penalty == "scad":
        a = 3.7
        abs_coef = np.abs(coef)
        pen = float(np.sum(np.where(
            abs_coef <= alpha,
            alpha * abs_coef,
            np.where(
                abs_coef <= a * alpha,
                (2 * a * alpha * abs_coef - abs_coef ** 2 - alpha ** 2) / (2 * (a - 1)),
                0.5 * (a + 1) * alpha ** 2,
            ),
        )))
    elif penalty == "mcp":
        gamma = 3.0
        abs_coef = np.abs(coef)
        pen = float(np.sum(np.where(
            abs_coef <= gamma * alpha,
            alpha * abs_coef - abs_coef ** 2 / (2 * gamma),
            0.5 * gamma * alpha ** 2,
        )))
    else:
        pen = 0.0
    return value + pen


def _metrics(case, X, y, coef, intercept, ref_coef, ref_intercept):
    pred = X @ coef + intercept
    ref_pred = X @ ref_coef + ref_intercept
    coef_delta = coef - ref_coef
    obj = _objective(case.loss, case.penalty, X, y, coef, intercept, case.alpha, case.l1_ratio)
    ref_obj = _objective(case.loss, case.penalty, X, y, ref_coef, ref_intercept, case.alpha, case.l1_ratio)
    return {
        "coef_max_abs_diff": float(np.max(np.abs(coef_delta))),
        "coef_l2_diff": float(np.linalg.norm(coef_delta)),
        "intercept_abs_diff": float(abs(intercept - ref_intercept)),
        "prediction_rmse_diff": float(math.sqrt(np.mean((pred - ref_pred) ** 2))),
        "objective": obj,
        "reference_objective": ref_obj,
        "objective_diff": float(obj - ref_obj) if np.isfinite(obj) and np.isfinite(ref_obj) else None,
    }


def _fit_sklearn(case, X, y):
    t0 = time.perf_counter()
    if case.reference == "sklearn_lasso":
        from sklearn.linear_model import Lasso

        model = Lasso(alpha=case.alpha, fit_intercept=True, max_iter=case.max_iter, tol=case.tol)
        model.fit(X, y)
        mapping = "sklearn Lasso alpha == statgpu alpha"
    elif case.reference == "sklearn_enet":
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(
            alpha=case.alpha,
            l1_ratio=case.l1_ratio,
            fit_intercept=True,
            max_iter=case.max_iter,
            tol=case.tol,
        )
        model.fit(X, y)
        mapping = "sklearn ElasticNet alpha/l1_ratio == statgpu alpha/l1_ratio"
    elif case.reference == "sklearn_ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=case.n * case.alpha, fit_intercept=True, solver="svd")
        model.fit(X, y)
        mapping = "sklearn Ridge alpha = n_samples * statgpu alpha"
    elif case.reference == "sklearn_logistic_l2":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty="l2",
            C=1.0 / (case.n * case.alpha),
            fit_intercept=True,
            solver="lbfgs",
            max_iter=case.max_iter,
            tol=case.tol,
        )
        model.fit(X, y)
        coef = model.coef_[0] if model.coef_.ndim == 2 else model.coef_
        return {
            "status": "OK",
            "coef": np.asarray(coef, dtype=np.float64),
            "intercept": float(model.intercept_[0]),
            "time_ms": (time.perf_counter() - t0) * 1000.0,
            "alpha_mapping": "sklearn LogisticRegression C = 1 / (n_samples * statgpu alpha)",
        }
    elif case.reference == "sklearn_poisson":
        from sklearn.linear_model import PoissonRegressor

        model = PoissonRegressor(alpha=case.alpha, fit_intercept=True, max_iter=case.max_iter, tol=case.tol)
        model.fit(X, y)
        mapping = "sklearn PoissonRegressor alpha == statgpu alpha"
    else:
        raise ValueError(case.reference)
    return {
        "status": "OK",
        "coef": np.asarray(model.coef_, dtype=np.float64).ravel(),
        "intercept": float(model.intercept_),
        "time_ms": (time.perf_counter() - t0) * 1000.0,
        "alpha_mapping": mapping,
    }


def _fit_statsmodels(case, X, y):
    import statsmodels.api as sm

    X_sm = sm.add_constant(X, has_constant="add")
    alpha_sm = np.concatenate([[0.0], np.full(X.shape[1], case.alpha)])
    t0 = time.perf_counter()
    result = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit_regularized(
        alpha=alpha_sm,
        L1_wt=case.l1_ratio,
        maxiter=case.max_iter,
        cnvrg_tol=case.tol,
        zero_tol=1e-10,
    )
    return {
        "status": "OK",
        "coef": np.asarray(result.params[1:], dtype=np.float64),
        "intercept": float(result.params[0]),
        "time_ms": (time.perf_counter() - t0) * 1000.0,
        "alpha_mapping": "statsmodels alpha vector = [0, alpha, ..., alpha], so intercept is unpenalized",
    }


def _run_r_script(r_code, coef_file):
    rscript = shutil.which("Rscript")
    if not rscript:
        return {"status": "SKIP", "reason": "Rscript not found"}
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as handle:
        handle.write(r_code)
        script_path = handle.name
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [rscript, script_path],
            text=True,
            capture_output=True,
            timeout=180,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            return {"status": "SKIP", "reason": proc.stderr.strip()[:300]}
        params = np.loadtxt(coef_file, delimiter=",")
        params = np.asarray(params, dtype=np.float64).ravel()
        return {
            "status": "OK",
            "coef": params[1:],
            "intercept": float(params[0]),
            "time_ms": elapsed,
        }
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def _fit_r(case, X, y):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        x_file = tmp / "x.csv"
        y_file = tmp / "y.csv"
        coef_file = tmp / "coef.csv"
        np.savetxt(x_file, X, delimiter=",", fmt="%.17g")
        np.savetxt(y_file, y, delimiter=",", fmt="%.17g")
        if case.reference == "r_glmnet_gaussian":
            r_code = f"""
suppressPackageStartupMessages(library(glmnet))
X <- as.matrix(read.csv("{x_file.as_posix()}", header=FALSE))
y <- as.numeric(read.csv("{y_file.as_posix()}", header=FALSE)[,1])
fit <- glmnet(X, y, family="gaussian", alpha=1.0, lambda=c({case.alpha}),
              standardize=FALSE, intercept=TRUE, thresh={case.tol}, maxit={case.max_iter})
params <- as.numeric(coef(fit, s={case.alpha}))
write.table(matrix(params, nrow=1), "{coef_file.as_posix()}", sep=",",
            row.names=FALSE, col.names=FALSE)
"""
            mapping = "R glmnet lambda == statgpu alpha, alpha=1, standardize=FALSE"
        elif case.reference == "r_ncvreg_gaussian":
            penalty = "SCAD" if case.penalty == "scad" else "MCP"
            r_code = f"""
suppressPackageStartupMessages(library(ncvreg))
X <- as.matrix(read.csv("{x_file.as_posix()}", header=FALSE))
y <- as.numeric(read.csv("{y_file.as_posix()}", header=FALSE)[,1])
target <- {case.alpha}
fit0 <- ncvreg(X, y, family="gaussian", penalty="{penalty}",
               standardize=FALSE, eps={case.tol}, max.iter={case.max_iter})
lambda_seq <- fit0$lambda
if (!any(abs(lambda_seq - target) <= max(1e-12, abs(target) * 1e-10))) {{
  lambda_seq <- sort(unique(c(lambda_seq, target)), decreasing=TRUE)
  fit0 <- ncvreg(X, y, family="gaussian", penalty="{penalty}",
                 lambda=lambda_seq, standardize=FALSE,
                 eps={case.tol}, max.iter={case.max_iter})
}}
params <- as.numeric(coef(fit0, lambda=target))
write.table(matrix(params, nrow=1), "{coef_file.as_posix()}", sep=",",
            row.names=FALSE, col.names=FALSE)
"""
            mapping = f"R ncvreg lambda == statgpu alpha, penalty={penalty}, standardize=FALSE"
        else:
            raise ValueError(case.reference)
        result = _run_r_script(r_code, coef_file)
        result["alpha_mapping"] = mapping
        return result


def _fit_reference(case, X, y, include_r):
    try:
        if case.reference.startswith("sklearn_"):
            return _fit_sklearn(case, X, y)
        if case.reference.startswith("statsmodels_"):
            return _fit_statsmodels(case, X, y)
        if case.reference.startswith("r_"):
            if not include_r:
                return {"status": "SKIP", "reason": "R comparisons disabled; pass --include-r"}
            return _fit_r(case, X, y)
        raise ValueError(case.reference)
    except Exception as exc:
        return {"status": "ERROR", "reason": repr(exc)}


def _jsonable_reference(ref):
    out = {k: v for k, v in ref.items() if k not in ("coef",)}
    if "intercept" in out:
        out["intercept"] = float(out["intercept"])
    if "time_ms" in out:
        out["time_ms"] = float(out["time_ms"])
    return out


def main():
    args = _parse_args()
    devices = _csv(args.devices)
    records = []

    print("=" * 108)
    print("Small external GLM+penalty benchmark")
    print("=" * 108)
    print(f"devices={devices} repeat={args.repeat} warmup={args.warmup} include_r={args.include_r}")

    for case in CASES:
        X, y = _make_data(case.loss, case.n, case.p, case.seed)
        ref = _fit_reference(case, X, y, args.include_r)
        row = {
            "case": case.name,
            "loss": case.loss,
            "penalty": case.penalty,
            "alpha": case.alpha,
            "l1_ratio": case.l1_ratio,
            "n": case.n,
            "p": case.p,
            "solver": case.solver,
            "reference_name": case.reference,
            "reference": _jsonable_reference(ref),
            "statgpu": {},
        }
        print(f"\n[{case.name}] ref={case.reference} status={ref.get('status')}")
        if ref.get("status") != "OK":
            print(f"  reference skipped/error: {ref.get('reason')}")
            records.append(row)
            continue
        ref_coef = np.asarray(ref["coef"], dtype=np.float64)
        ref_intercept = float(ref["intercept"])
        print(f"  ref_time={ref['time_ms']:.1f}ms mapping={ref.get('alpha_mapping')}")

        for device in devices:
            available, reason = _available_device(device)
            if not available:
                row["statgpu"][device] = {"status": "SKIP", "reason": reason}
                print(f"  {device:<6} SKIP {reason}")
                continue
            try:
                X_dev, y_dev = _device_arrays(X, y, device)
                fit_fn = lambda X_dev=X_dev, y_dev=y_dev, device=device: _fit_statgpu_once(
                    X_dev, y_dev, case, device
                )
                model, elapsed = _median_timed_fit(fit_fn, args.repeat, args.warmup)
                coef, intercept = _coef_intercept(model)
                metrics = _metrics(case, X, y, coef, intercept, ref_coef, ref_intercept)
                primary = max(metrics["coef_max_abs_diff"], metrics["intercept_abs_diff"])
                status = "OK" if primary <= case.threshold else "CHECK"
                row["statgpu"][device] = {
                    "status": status,
                    "time_ms": elapsed,
                    "n_iter": int(getattr(model, "n_iter_", -1)),
                    "threshold": case.threshold,
                    **metrics,
                }
                print(
                    f"  {device:<6} {status:<5} time={elapsed:8.1f}ms "
                    f"coef_max={metrics['coef_max_abs_diff']:.2e} "
                    f"int={metrics['intercept_abs_diff']:.2e} "
                    f"pred_rmse={metrics['prediction_rmse_diff']:.2e}"
                )
            except Exception as exc:
                row["statgpu"][device] = {"status": "ERROR", "reason": repr(exc)}
                print(f"  {device:<6} ERROR {exc!r}")
        records.append(row)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {out}")

    ok = 0
    checked = 0
    for row in records:
        for info in row.get("statgpu", {}).values():
            if info.get("status") in ("OK", "CHECK"):
                checked += 1
                ok += int(info.get("status") == "OK")
    print(f"\nSUMMARY statgpu OK {ok}/{checked} checked rows")


if __name__ == "__main__":
    main()
