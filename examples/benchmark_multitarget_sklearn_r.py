"""
Benchmark multi-target linear regression across statgpu, sklearn, and R.

Compares:
  - Runtime: fit and predict milliseconds
  - Estimation accuracy: coefficient/intercept recovery error vs known ground truth
  - Prediction accuracy: mean R2 and RMSE on test set
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np

from multitarget_benchmark_utils import (
    MetricRow,
    coef_error,
    intercept_error,
    make_multitarget_data,
    mean_r2_score,
    rmse,
    split_train_test,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu._config import set_device
from statgpu.linear_model import LinearRegression


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-target statgpu/sklearn/R benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=8000)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-targets", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.25)
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--skip-r", action="store_true")
    return p.parse_args()


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def _run_r_multitarget(train_csv: Path, test_csv: Path) -> dict:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {"error": "Rscript not found"}
    runner = Path(__file__).with_name("r_multitarget_lm.R")
    if not runner.exists():
        return {"error": f"R helper script missing: {runner}"}
    proc = subprocess.run(
        [rscript, str(runner), str(train_csv), str(test_csv)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or "R execution failed"}
    try:
        return json.loads(proc.stdout)
    except Exception as e:
        return {"error": f"Failed to parse R JSON: {e}"}


def _format(rows: List[MetricRow]):
    print("\n=== Multi-target Linear Benchmark ===")
    print(
        f"{'framework':<28} {'fit_ms':>10} {'pred_ms':>10} "
        f"{'mean_r2':>10} {'rmse':>10} {'coef_mae':>10} {'coef_max':>10}"
    )
    for r in rows:
        print(
            f"{r.framework:<28} {r.fit_ms:>10.2f} {r.pred_ms:>10.2f} "
            f"{r.mean_r2:>10.6f} {r.rmse:>10.6f} {r.mean_abs_coef_err:>10.3e} {r.max_abs_coef_err:>10.3e}"
        )
        if r.notes:
            print(f"  note: {r.notes}")


def main():
    args = parse_args()
    X, Y, true_coef, true_intercept = make_multitarget_data(
        seed=args.seed,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_targets=args.n_targets,
        noise_std=args.noise_std,
    )
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y, test_ratio=args.test_ratio, seed=args.seed + 7)
    rows: List[MetricRow] = []

    set_device("cpu")

    sg, sg_fit = _time_call(lambda: LinearRegression(device="cpu").fit(X_train, Y_train))
    Yp_sg, sg_pred = _time_call(lambda: sg.predict(X_test))
    sg_cmae, sg_cmax = coef_error(np.asarray(sg.coef_), true_coef)
    sg_imae, sg_imax = intercept_error(np.asarray(sg.intercept_), true_intercept)
    rows.append(
        MetricRow(
            framework="statgpu.LinearRegression(cpu)",
            fit_ms=sg_fit,
            pred_ms=sg_pred,
            mean_r2=mean_r2_score(Y_test, Yp_sg),
            rmse=rmse(Y_test, Yp_sg),
            mean_abs_coef_err=sg_cmae,
            max_abs_coef_err=sg_cmax,
            mean_abs_intercept_err=sg_imae,
            max_abs_intercept_err=sg_imax,
        )
    )

    try:
        from sklearn.linear_model import LinearRegression as SkLinearRegression

        sk, sk_fit = _time_call(lambda: SkLinearRegression().fit(X_train, Y_train))
        Yp_sk, sk_pred = _time_call(lambda: sk.predict(X_test))
        sk_cmae, sk_cmax = coef_error(np.asarray(sk.coef_), true_coef)
        sk_imae, sk_imax = intercept_error(np.asarray(sk.intercept_), true_intercept)
        rows.append(
            MetricRow(
                framework="sklearn.LinearRegression",
                fit_ms=sk_fit,
                pred_ms=sk_pred,
                mean_r2=mean_r2_score(Y_test, Yp_sk),
                rmse=rmse(Y_test, Yp_sk),
                mean_abs_coef_err=sk_cmae,
                max_abs_coef_err=sk_cmax,
                mean_abs_intercept_err=sk_imae,
                max_abs_intercept_err=sk_imax,
            )
        )
    except Exception as e:
        rows.append(
            MetricRow(
                framework="sklearn.LinearRegression",
                fit_ms=np.nan,
                pred_ms=np.nan,
                mean_r2=np.nan,
                rmse=np.nan,
                mean_abs_coef_err=np.nan,
                max_abs_coef_err=np.nan,
                mean_abs_intercept_err=np.nan,
                max_abs_intercept_err=np.nan,
                notes=f"skipped: {e}",
            )
        )

    if not args.skip_r:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            x_cols = [f"x{i+1}" for i in range(X_train.shape[1])]
            y_cols = [f"y{j+1}" for j in range(Y_train.shape[1])]
            train = np.column_stack([X_train, Y_train])
            test = np.column_stack([X_test, Y_test])
            header = ",".join(x_cols + y_cols)
            train_csv = td_path / "train.csv"
            test_csv = td_path / "test.csv"
            np.savetxt(train_csv, train, delimiter=",", header=header, comments="")
            np.savetxt(test_csv, test, delimiter=",", header=header, comments="")
            r_out = _run_r_multitarget(train_csv, test_csv)
            if "error" in r_out:
                rows.append(
                    MetricRow(
                        framework="R::lm(cbind(...))",
                        fit_ms=np.nan,
                        pred_ms=np.nan,
                        mean_r2=np.nan,
                        rmse=np.nan,
                        mean_abs_coef_err=np.nan,
                        max_abs_coef_err=np.nan,
                        mean_abs_intercept_err=np.nan,
                        max_abs_intercept_err=np.nan,
                        notes=r_out["error"],
                    )
                )
            else:
                coeffs = np.asarray(r_out["coeffs"], dtype=float)
                intercept_r = coeffs[0, :]
                coef_r = coeffs[1:, :].T
                pred_r = np.asarray(r_out["pred"], dtype=float)
                if pred_r.ndim == 1:
                    pred_r = pred_r.reshape(-1, 1)
                r_cmae, r_cmax = coef_error(coef_r, true_coef)
                r_imae, r_imax = intercept_error(intercept_r, true_intercept)
                rows.append(
                    MetricRow(
                        framework="R::lm(cbind(...))",
                        fit_ms=float(r_out.get("fit_ms", np.nan)),
                        pred_ms=float(r_out.get("pred_ms", np.nan)),
                        mean_r2=mean_r2_score(Y_test, pred_r),
                        rmse=rmse(Y_test, pred_r),
                        mean_abs_coef_err=r_cmae,
                        max_abs_coef_err=r_cmax,
                        mean_abs_intercept_err=r_imae,
                        max_abs_intercept_err=r_imax,
                    )
                )

    _format(rows)

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([r.to_dict() for r in rows], indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()
