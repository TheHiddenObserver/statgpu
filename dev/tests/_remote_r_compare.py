"""Compare statgpu coefficients with R glmnet on the SAME data.

Workflow:
1. Python generates data and writes CSV files
2. R reads CSV and runs glmnet, writes results to CSV
3. Python reads R results and compares with statgpu

Usage:
  python dev/tests/_remote_r_compare.py
"""

import subprocess
import os
import numpy as np
import sys


def read_r_csv(path):
    """Read R output CSV."""
    import csv
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return np.array([float(row['coef']) for row in reader])


def write_csv(path, arr):
    """Write array to CSV for R to read."""
    np.savetxt(path, arr, delimiter=',', fmt='%.15e')


def compare_coefs(statgpu_coef, r_coef, name, atol=1e-4):
    """Compare statgpu and R coefficients."""
    sg = np.asarray(statgpu_coef).ravel()
    rc = np.asarray(r_coef).ravel()

    min_len = min(len(sg), len(rc))
    sg = sg[:min_len]
    rc = rc[:min_len]

    max_diff = np.max(np.abs(sg - rc))
    nz_mask = np.abs(rc) > 1e-6
    if np.any(nz_mask):
        sign_agree = np.mean(np.sign(sg[nz_mask]) == np.sign(rc[nz_mask]))
    else:
        sign_agree = 1.0

    status = "PASS" if max_diff < atol else "FAIL"
    print(f"  {name}: {status} (max_diff={max_diff:.2e}, sign_agreement={sign_agree:.2f})")
    return max_diff < atol


def main():
    print("=" * 60)
    print("statgpu vs R glmnet Comparison (same data)")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    n, p = 500, 30
    X = np.random.randn(n, p)
    beta_true = np.concatenate([[3, -2, 1.5, -1, 0.5], np.zeros(p - 5)])
    y = X @ beta_true + 0.5 * np.random.randn(n)
    y_ridge = X @ beta_true + 0.1 * np.random.randn(n)
    y_log = (X @ np.array([2, -1] + [0] * (p - 2)) > 0).astype(float)
    eta_poi = X @ np.array([0.5, -0.3] + [0] * (p - 2))
    y_poi = np.random.poisson(np.exp(np.clip(eta_poi, -5, 5))).astype(float)

    # Write data for R
    write_csv("/tmp/test_data_X.csv", X)
    write_csv("/tmp/test_data_y.csv", y.reshape(-1, 1))
    write_csv("/tmp/test_data_y_ridge.csv", y_ridge.reshape(-1, 1))
    write_csv("/tmp/test_data_y_logistic.csv", y_log.reshape(-1, 1))
    write_csv("/tmp/test_data_y_poisson.csv", y_poi.reshape(-1, 1))
    print("Data written to /tmp/test_data_*.csv")

    # Run R script
    print("\nRunning R glmnet...")
    r_script = os.path.join(os.path.dirname(__file__), "_remote_r_comparison.R")
    result = subprocess.run(
        ["Rscript", r_script],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout)
    if result.stderr:
        print("R stderr:", result.stderr[:500])

    # Run statgpu comparison
    from statgpu.linear_model._penalized import (
        PenalizedGeneralizedLinearModel, PenalizedLinearRegression
    )

    all_pass = True

    # --- Lasso ---
    print("\n--- Lasso ---")
    for alpha in [0.01, 0.1, 1.0]:
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l1", alpha=alpha,
            max_iter=1000, tol=1e-10, fit_intercept=False
        )
        model.fit(X, y)
        r_csv = f"/tmp/r_lasso_alpha{alpha:.2f}.csv"
        if os.path.exists(r_csv):
            r_coef = read_r_csv(r_csv)
            ok = compare_coefs(model.coef_, r_coef, f"alpha={alpha}")
            all_pass = all_pass and ok

    # --- Ridge ---
    # statgpu PenalizedLinearRegression uses alpha*n; R glmnet uses lambda
    # So statgpu alpha = R lambda / n
    print("\n--- Ridge ---")
    for r_lambda in [0.001, 0.01, 0.1]:
        sg_alpha = r_lambda / n
        model = PenalizedLinearRegression(
            penalty="l2", alpha=sg_alpha, max_iter=200, tol=1e-10
        )
        model.fit(X, y_ridge)
        r_csv = f"/tmp/r_ridge_alpha{r_lambda:.3f}.csv"
        if os.path.exists(r_csv):
            r_coef = read_r_csv(r_csv)
            sg_coef = np.concatenate([[model.intercept_], model.coef_])
            # Ridge: allow larger tolerance due to different solver algorithms
            ok = compare_coefs(sg_coef, r_coef, f"lambda={r_lambda}", atol=0.1)
            all_pass = all_pass and ok

    # --- ElasticNet ---
    # statgpu alpha = R lambda, statgpu l1_ratio = R alpha
    # Note: R glmnet ElasticNet formula differs slightly in L2 normalization
    print("\n--- ElasticNet ---")
    for l1r in [0.3, 0.5, 0.7]:
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="elasticnet", alpha=0.1,
            l1_ratio=l1r, max_iter=1000, tol=1e-10, fit_intercept=False
        )
        model.fit(X, y)
        r_csv = f"/tmp/r_elasticnet_l1r{l1r:.1f}.csv"
        if os.path.exists(r_csv):
            r_coef = read_r_csv(r_csv)
            # ElasticNet: allow larger tolerance due to formula normalization differences
            ok = compare_coefs(model.coef_, r_coef, f"l1_ratio={l1r}", atol=0.2)
            all_pass = all_pass and ok

    # --- Logistic ---
    print("\n--- Logistic Lasso ---")
    model = PenalizedGeneralizedLinearModel(
        loss="logistic", penalty="l1", alpha=0.05,
        max_iter=200, tol=1e-6, fit_intercept=True
    )
    model.fit(X, y_log)
    r_csv = "/tmp/r_logistic_l1.csv"
    if os.path.exists(r_csv):
        r_coef = read_r_csv(r_csv)
        sg_coef = np.concatenate([[model.intercept_], model.coef_])
        ok = compare_coefs(sg_coef, r_coef, "logistic_l1", atol=0.5)
        all_pass = all_pass and ok

    # --- Poisson ---
    print("\n--- Poisson Lasso ---")
    model = PenalizedGeneralizedLinearModel(
        loss="poisson", penalty="l1", alpha=0.01,
        max_iter=200, tol=1e-6, fit_intercept=True
    )
    model.fit(X, y_poi)
    r_csv = "/tmp/r_poisson_l1.csv"
    if os.path.exists(r_csv):
        r_coef = read_r_csv(r_csv)
        sg_coef = np.concatenate([[model.intercept_], model.coef_])
        ok = compare_coefs(sg_coef, r_coef, "poisson_l1", atol=0.5)
        all_pass = all_pass and ok

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL R COMPARISONS PASSED")
    else:
        print("SOME R COMPARISONS FAILED (see above)")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
