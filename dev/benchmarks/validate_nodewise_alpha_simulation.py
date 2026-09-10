"""Deterministic diagnostic simulation for the node-wise tuning migration.

This runner compares the historical response-scaled/unstandardized node-wise
precision rule with the new standardized design-side default while holding the
main penalized fit fixed.  Coverage summaries are diagnostics, not finite-sample
theorem claims and are deliberately not encoded as brittle nominal-coverage CI
assertions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import norm

from statgpu.linear_model import PenalizedLinearRegression
from statgpu.linear_model.penalized._nodewise_precision import (
    build_nodewise_precision_numpy,
)


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def _old_precision(Xc, sigma_y):
    n, p = Xc.shape
    lam = float(sigma_y * np.sqrt(2.0 * np.log(max(p, 2)) / n))
    M = np.zeros((p, p), dtype=np.float64)
    for j in range(p):
        cols = np.concatenate([np.arange(j), np.arange(j + 1, p)])
        model = PenalizedLinearRegression(
            penalty="l1",
            alpha=lam,
            fit_intercept=False,
            device="cpu",
            solver="fista",
            max_iter=3000,
            tol=1e-8,
            compute_inference=False,
            inference_method="none",
        ).fit(Xc[:, cols], Xc[:, j])
        gamma = np.asarray(model.coef_, dtype=np.float64)
        resid = Xc[:, j] - Xc[:, cols] @ gamma
        C = float(Xc[:, j] @ resid / n)
        if not np.isfinite(C) or abs(C) < 1e-30:
            raise FloatingPointError("historical precision simulation reached degenerate normalizer")
        M[j, j] = 1.0 / C
        M[j, cols] = -gamma / C
    return M, lam


def _report_from_M(Xc, yc, beta_hat, M, beta_true):
    n, p = Xc.shape
    resid = yc - Xc @ beta_hat
    s = int(np.sum(np.abs(beta_hat) > 0.0))
    sigma2 = float(resid @ resid / max(n - s, 1))
    Sigma = Xc.T @ Xc / n
    theta = beta_hat + M @ (Xc.T @ resid) / n
    V = M @ Sigma @ M.T
    se = np.sqrt(np.maximum(0.0, sigma2 * np.diag(V) / n))
    zcrit = float(norm.ppf(0.975))
    lo = theta - zcrit * se
    hi = theta + zcrit * se
    covered = (lo <= beta_true) & (beta_true <= hi)
    return covered, hi - lo, theta


def _one(seed, n, p, rho, main_alpha):
    rng = np.random.default_rng(seed)
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    beta = np.zeros(p, dtype=np.float64)
    beta[: min(3, p)] = np.asarray([1.0, -0.8, 0.55])[: min(3, p)]
    y = X @ beta + rng.normal(scale=0.6, size=n)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    fit = PenalizedLinearRegression(
        penalty="l1",
        alpha=main_alpha,
        fit_intercept=True,
        device="cpu",
        solver="fista",
        max_iter=3000,
        tol=1e-8,
        compute_inference=False,
        inference_method="none",
    ).fit(X, y)
    beta_hat = np.asarray(fit.coef_, dtype=np.float64)
    fit_resid = y - float(fit.intercept_) - X @ beta_hat
    sigma_hat = float(np.sqrt((fit_resid @ fit_resid) / max(n - np.count_nonzero(beta_hat), 1)))

    M_old, old_alpha = _old_precision(Xc, sigma_hat)
    M_new, new_alpha, meta = build_nodewise_precision_numpy(
        Xc,
        requested_alpha=None,
        effective_n=float(n),
        weighted=False,
    )
    old_cov, old_len, old_theta = _report_from_M(Xc, yc, beta_hat, M_old, beta)
    new_cov, new_len, new_theta = _report_from_M(Xc, yc, beta_hat, M_new, beta)

    return {
        "old_coverage": old_cov,
        "new_coverage": new_cov,
        "old_length": old_len,
        "new_length": new_len,
        "old_theta": old_theta,
        "new_theta": new_theta,
        "old_nodewise_alpha": old_alpha,
        "new_nodewise_alpha": float(new_alpha),
        "new_max_kkt": float(meta["nodewise_max_kkt_residual"]),
    }


def _summarize(rows, p):
    signals = np.arange(p) < min(3, p)
    out = {}
    for prefix in ("old", "new"):
        coverage = np.stack([row[f"{prefix}_coverage"] for row in rows])
        length = np.stack([row[f"{prefix}_length"] for row in rows])
        out[prefix] = {
            "coverage_all": float(np.mean(coverage)),
            "coverage_signal": float(np.mean(coverage[:, signals])),
            "coverage_null": float(np.mean(coverage[:, ~signals])) if np.any(~signals) else None,
            "mean_interval_length": float(np.mean(length)),
            "median_interval_length": float(np.median(length)),
            "mean_nodewise_alpha": float(np.mean([row[f"{prefix}_nodewise_alpha"] for row in rows])),
        }
    out["new_max_kkt_residual"] = float(max(row["new_max_kkt"] for row in rows))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.45)
    parser.add_argument("--main-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", default="results/nodewise_alpha_simulation_schema_v1.json")
    args = parser.parse_args()
    if args.reps <= 0 or args.n <= 0 or args.p <= 1:
        raise ValueError("reps/n must be positive and p must exceed one")

    rows = [
        _one(args.seed + i, args.n, args.p, args.rho, args.main_alpha)
        for i in range(args.reps)
    ]
    result = {
        "schema_version": 1,
        "head_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _git("status", "--porcelain") == "",
        "status": "success",
        "purpose": "diagnostic old-vs-new migration evidence; not a finite-sample coverage guarantee",
        "config": vars(args) | {"output": str(args.output)},
        "summary": _summarize(rows, args.p),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
