"""
Benchmark statgpu against external frameworks (statsmodels/sklearn/R) when available.

Outputs:
  - Runtime (fit_ms)
  - Numerical differences vs statgpu CPU baseline

Coverage (best-effort):
  - LinearRegression: statsmodels OLS, sklearn LinearRegression, R lm
  - Ridge: sklearn Ridge, R glmnet(alpha=0)
  - Lasso: sklearn Lasso, statsmodels fit_regularized (params), R glmnet(alpha=1)
  - LogisticRegression: statsmodels Logit, sklearn LogisticRegression, R glm(binomial)
  - CoxPH: statsmodels PHReg, R survival::coxph
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statgpu.survival import CoxPH
from statgpu._config import set_device


@dataclass
class BenchRow:
    method: str
    framework: str
    fit_ms: float
    max_abs_coef_diff: float
    max_abs_bse_diff: float
    max_abs_p_diff: float
    notes: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External-framework benchmark for statgpu.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--p", type=int, default=12)
    p.add_argument("--lasso-alpha", type=float, default=0.05)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--cox-ties", type=str, default="breslow", choices=["breslow", "efron"])
    p.add_argument("--cox-cov-type", type=str, default="hc1", choices=["nonrobust", "hc0", "hc1", "cluster"])
    p.add_argument("--cluster-groups", type=int, default=80, help="Number of clusters when cox-cov-type=cluster")
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--skip-r", action="store_true", help="Skip R comparison even if Rscript exists")
    return p.parse_args()


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def _safe_max_abs_diff(a, b) -> float:
    if a is None or b is None:
        return np.nan
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float(np.max(np.abs(a[:n] - b[:n])))


def _make_data(seed: int, n: int, p: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + 1.2 + rng.normal(scale=0.5, size=n)
    logits = X @ (0.8 * beta) + 0.1
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    y_bin = (rng.random(n) < prob).astype(int)
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    base = 0.03
    t_true = -np.log(u) / (base * np.exp(np.clip(X @ (0.35 * beta), -20, 20)))
    censor = rng.exponential(scale=np.median(t_true), size=n)
    event = (t_true <= censor).astype(int)
    time_obs = np.minimum(t_true, censor)
    return X, y, y_bin, time_obs, event


def _run_r_if_available(script: str) -> Optional[dict]:
    if shutil.which("Rscript") is None:
        return None
    try:
        proc = subprocess.run(
            ["Rscript", "-e", script],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode != 0:
            return {"error": proc.stderr.strip() or "Rscript failed"}
        return json.loads(proc.stdout)
    except Exception as e:
        return {"error": str(e)}


def main():
    args = parse_args()
    set_device("cpu")
    X, y, y_bin, t_obs, event = _make_data(args.seed, args.n, args.p)
    rng = np.random.default_rng(args.seed + 999)
    cluster_ids = rng.integers(0, max(2, args.cluster_groups), size=args.n)
    rows: List[BenchRow] = []

    # statgpu baselines
    lin_sg, lin_ms = _time_call(lambda: LinearRegression(device="cpu").fit(X, y))
    rid_sg, rid_ms = _time_call(lambda: Ridge(alpha=args.ridge_alpha, device="cpu").fit(X, y))
    lasso_sig = inspect.signature(Lasso.__init__)
    lasso_kwargs = {"alpha": args.lasso_alpha, "device": "cpu", "max_iter": 3000}
    if "cpu_solver" in lasso_sig.parameters:
        lasso_kwargs["cpu_solver"] = "fista"
    if "solver" in lasso_sig.parameters:
        lasso_kwargs["solver"] = "fista"
    las_sg, las_ms = _time_call(lambda: Lasso(**lasso_kwargs).fit(X, y))
    log_sig = inspect.signature(LogisticRegression.__init__)
    log_kwargs = {"device": "cpu", "C": 1e10, "max_iter": 200}
    if "cov_type" in log_sig.parameters:
        log_kwargs["cov_type"] = "hc1"
    log_sg, log_ms = _time_call(lambda: LogisticRegression(**log_kwargs).fit(X, y_bin))

    cox_sig = inspect.signature(CoxPH.__init__)
    cox_kwargs = {"device": "cpu", "ties": args.cox_ties, "max_iter": 80}
    if "cov_type" in cox_sig.parameters:
        cox_kwargs["cov_type"] = args.cox_cov_type

    fit_sig = inspect.signature(CoxPH.fit)
    supports_cluster = "cluster" in fit_sig.parameters

    if args.cox_cov_type == "cluster" and supports_cluster:
        cox_sg, cox_ms = _time_call(lambda: CoxPH(**cox_kwargs).fit(X, t_obs, event, cluster=cluster_ids))
    else:
        cox_sg, cox_ms = _time_call(lambda: CoxPH(**cox_kwargs).fit(X, t_obs, event))

    rows += [
        BenchRow("LinearRegression", "statgpu", lin_ms, 0.0, 0.0, 0.0),
        BenchRow("Ridge", "statgpu", rid_ms, 0.0, np.nan, np.nan),
        BenchRow("Lasso", "statgpu", las_ms, 0.0, np.nan, np.nan),
        BenchRow("LogisticRegression", "statgpu", log_ms, 0.0, 0.0, 0.0),
        BenchRow("CoxPH", "statgpu", cox_ms, 0.0, 0.0, 0.0),
    ]

    # statsmodels
    try:
        import statsmodels.api as sm
        import statsmodels.duration.api as smd

        sm_lin, sm_lin_ms = _time_call(lambda: sm.OLS(y, sm.add_constant(X)).fit())
        rows.append(
            BenchRow(
                "LinearRegression",
                "statsmodels.OLS",
                sm_lin_ms,
                _safe_max_abs_diff(np.r_[lin_sg.intercept_, lin_sg.coef_], sm_lin.params),
                _safe_max_abs_diff(lin_sg._bse, sm_lin.bse),
                _safe_max_abs_diff(lin_sg._pvalues, sm_lin.pvalues),
            )
        )

        sm_las, sm_las_ms = _time_call(
            lambda: sm.OLS(y, sm.add_constant(X)).fit_regularized(alpha=args.lasso_alpha, L1_wt=1.0, maxiter=3000)
        )
        rows.append(
            BenchRow(
                "Lasso",
                "statsmodels.OLS.fit_regularized",
                sm_las_ms,
                _safe_max_abs_diff(np.r_[las_sg.intercept_, las_sg.coef_], sm_las.params),
                np.nan,
                np.nan,
                notes="params only",
            )
        )

        sm_log, sm_log_ms = _time_call(lambda: sm.Logit(y_bin, sm.add_constant(X)).fit(disp=0, maxiter=200, cov_type="HC1"))
        rows.append(
            BenchRow(
                "LogisticRegression",
                "statsmodels.Logit",
                sm_log_ms,
                _safe_max_abs_diff(np.r_[log_sg.intercept_, log_sg.coef_], sm_log.params),
                _safe_max_abs_diff(log_sg._bse, sm_log.bse),
                _safe_max_abs_diff(log_sg._pvalues, sm_log.pvalues),
            )
        )

        if args.cox_cov_type == "cluster":
            sm_cox, sm_cox_ms = _time_call(
                lambda: smd.PHReg(t_obs, X, status=event, ties=args.cox_ties).fit(groups=cluster_ids)
            )
        else:
            sm_cox, sm_cox_ms = _time_call(lambda: smd.PHReg(t_obs, X, status=event, ties=args.cox_ties).fit())
        rows.append(
            BenchRow(
                "CoxPH",
                "statsmodels.PHReg",
                sm_cox_ms,
                _safe_max_abs_diff(cox_sg.coef_, sm_cox.params),
                _safe_max_abs_diff(cox_sg._bse, sm_cox.bse),
                _safe_max_abs_diff(cox_sg._pvalues, sm_cox.pvalues),
            )
        )
    except Exception as e:
        rows.append(BenchRow("ALL", "statsmodels", np.nan, np.nan, np.nan, np.nan, notes=f"skipped: {e}"))

    # sklearn
    try:
        from sklearn.linear_model import LinearRegression as SkLinear
        from sklearn.linear_model import Ridge as SkRidge
        from sklearn.linear_model import Lasso as SkLasso
        from sklearn.linear_model import LogisticRegression as SkLogit

        sk_lin, sk_lin_ms = _time_call(lambda: SkLinear().fit(X, y))
        rows.append(
            BenchRow(
                "LinearRegression",
                "sklearn.LinearRegression",
                sk_lin_ms,
                _safe_max_abs_diff(np.r_[lin_sg.intercept_, lin_sg.coef_], np.r_[sk_lin.intercept_, sk_lin.coef_]),
                np.nan,
                np.nan,
            )
        )

        sk_rid, sk_rid_ms = _time_call(lambda: SkRidge(alpha=args.ridge_alpha).fit(X, y))
        rows.append(
            BenchRow(
                "Ridge",
                "sklearn.Ridge",
                sk_rid_ms,
                _safe_max_abs_diff(np.r_[rid_sg.intercept_, rid_sg.coef_], np.r_[sk_rid.intercept_, sk_rid.coef_]),
                np.nan,
                np.nan,
            )
        )

        sk_las, sk_las_ms = _time_call(lambda: SkLasso(alpha=args.lasso_alpha, max_iter=3000, tol=1e-5).fit(X, y))
        rows.append(
            BenchRow(
                "Lasso",
                "sklearn.Lasso",
                sk_las_ms,
                _safe_max_abs_diff(np.r_[las_sg.intercept_, las_sg.coef_], np.r_[sk_las.intercept_, sk_las.coef_]),
                np.nan,
                np.nan,
            )
        )

        sk_log, sk_log_ms = _time_call(
            lambda: SkLogit(C=1e10, penalty="l2", solver="lbfgs", max_iter=500).fit(X, y_bin)
        )
        rows.append(
            BenchRow(
                "LogisticRegression",
                "sklearn.LogisticRegression",
                sk_log_ms,
                _safe_max_abs_diff(np.r_[log_sg.intercept_, log_sg.coef_], np.r_[sk_log.intercept_[0], sk_log.coef_[0]]),
                np.nan,
                np.nan,
                notes="sklearn does not expose robust inference by default",
            )
        )
    except Exception as e:
        rows.append(BenchRow("ALL", "sklearn", np.nan, np.nan, np.nan, np.nan, notes=f"skipped: {e}"))

    # R (optional)
    if not args.skip_r:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "data.csv"
            x_terms = "+".join([f"x{i}" for i in range(1, X.shape[1] + 1)])
            arr = np.column_stack([X, y, y_bin, t_obs, event, cluster_ids])
            cols = [f"x{i+1}" for i in range(X.shape[1])] + ["y", "y_bin", "time", "event", "cluster"]
            np.savetxt(csv_path, arr, delimiter=",", header=",".join(cols), comments="")
            r_script = f"""
            suppressWarnings({{
              d <- read.csv("{csv_path.as_posix()}")
              X <- as.matrix(d[, grep("^x", names(d))])
              out <- list()
              t0 <- proc.time(); m_lm <- lm(y ~ {x_terms}, data=d); t1 <- proc.time()
              out$lm <- list(time_ms=as.numeric((t1-t0)[3])*1000, coef=coef(m_lm))
              if (requireNamespace("survival", quietly=TRUE)) {{
                t2 <- proc.time(); m_cox <- survival::coxph(survival::Surv(time, event) ~ {x_terms}, data=d, ties="{args.cox_ties}"{", cluster=cluster" if args.cox_cov_type == "cluster" else ""}); t3 <- proc.time()
                out$cox <- list(time_ms=as.numeric((t3-t2)[3])*1000, coef=coef(m_cox))
              }}
              if (requireNamespace("glmnet", quietly=TRUE)) {{
                t4 <- proc.time(); m_r <- glmnet::glmnet(X, d$y, alpha=0); t5 <- proc.time()
                t6 <- proc.time(); m_l <- glmnet::glmnet(X, d$y, alpha=1); t7 <- proc.time()
                out$ridge <- list(time_ms=as.numeric((t5-t4)[3])*1000)
                out$lasso <- list(time_ms=as.numeric((t7-t6)[3])*1000)
              }}
              t8 <- proc.time(); m_g <- glm(y_bin ~ {x_terms}, data=d, family=binomial()); t9 <- proc.time()
              out$logit <- list(time_ms=as.numeric((t9-t8)[3])*1000, coef=coef(m_g))
              cat(jsonlite::toJSON(out, auto_unbox=TRUE))
            }})
            """
            r_out = _run_r_if_available(r_script)
            if r_out is None:
                rows.append(BenchRow("ALL", "R", np.nan, np.nan, np.nan, np.nan, notes="Rscript not found"))
            elif "error" in r_out:
                rows.append(BenchRow("ALL", "R", np.nan, np.nan, np.nan, np.nan, notes=f"R failed: {r_out['error']}"))
            else:
                if "lm" in r_out:
                    rows.append(
                        BenchRow(
                            "LinearRegression",
                            "R::lm",
                            float(r_out["lm"]["time_ms"]),
                            _safe_max_abs_diff(np.r_[lin_sg.intercept_, lin_sg.coef_], np.asarray(r_out["lm"]["coef"])),
                            np.nan,
                            np.nan,
                        )
                    )
                if "logit" in r_out:
                    rows.append(
                        BenchRow(
                            "LogisticRegression",
                            "R::glm(binomial)",
                            float(r_out["logit"]["time_ms"]),
                            _safe_max_abs_diff(np.r_[log_sg.intercept_, log_sg.coef_], np.asarray(r_out["logit"]["coef"])),
                            np.nan,
                            np.nan,
                        )
                    )
                if "cox" in r_out:
                    rows.append(
                        BenchRow(
                            "CoxPH",
                            "R::survival::coxph",
                            float(r_out["cox"]["time_ms"]),
                            _safe_max_abs_diff(cox_sg.coef_, np.asarray(r_out["cox"]["coef"])),
                            np.nan,
                            np.nan,
                        )
                    )
                if "ridge" in r_out:
                    rows.append(BenchRow("Ridge", "R::glmnet(alpha=0)", float(r_out["ridge"]["time_ms"]), np.nan, np.nan, np.nan))
                if "lasso" in r_out:
                    rows.append(BenchRow("Lasso", "R::glmnet(alpha=1)", float(r_out["lasso"]["time_ms"]), np.nan, np.nan, np.nan))

    # print table
    print("\n=== External Framework Benchmark ===")
    print(f"{'method':<20} {'framework':<32} {'fit_ms':>10} {'coef_diff':>12} {'bse_diff':>12} {'p_diff':>12}")
    for r in rows:
        print(
            f"{r.method:<20} {r.framework:<32} "
            f"{r.fit_ms:>10.2f} {r.max_abs_coef_diff:>12.3e} "
            f"{r.max_abs_bse_diff:>12.3e} {r.max_abs_p_diff:>12.3e}"
        )
        if r.notes:
            print(f"  note: {r.notes}")

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()
