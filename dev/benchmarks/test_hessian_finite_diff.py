"""Finite-difference Hessian verification for ordered logit/probit.

Compares analytical Hessian from _ordered_hessian_analytical against
numerical Hessian (central differences) on small random data.
Tests beta-beta, beta-theta, and theta-theta blocks separately.

Usage:
    python dev/benchmarks/test_hessian_finite_diff.py
    python dev/benchmarks/test_hessian_finite_diff.py --backend torch
    python dev/benchmarks/test_hessian_finite_diff.py --backend cupy
"""

import numpy as np
import sys, os, argparse

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)


def numerical_hessian(fn, theta, h=1e-5):
    """Central-difference Hessian of fn at theta."""
    d = len(theta)
    H = np.zeros((d, d))
    f0 = fn(theta)
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h
        for j in range(i, d):
            ej = np.zeros(d)
            ej[j] = h
            fpp = fn(theta + ei + ej)
            fpm = fn(theta + ei - ej)
            fmp = fn(theta - ei + ej)
            fmm = fn(theta - ei - ej)
            Hij = (fpp - fpm - fmp + fmm) / (4 * h * h)
            H[i, j] = Hij
            H[j, i] = Hij
    return H


def test_hessian(backend="numpy", seed=42, n=200, p=4, K=4):
    """Compare analytical vs numerical Hessian for ordered model."""
    np.random.seed(seed)
    X = np.random.randn(n, p)
    beta_true = np.linspace(0.5, -0.3, p)
    y = np.digitize(0.5 + X @ beta_true + 0.5 * np.random.randn(n),
                     np.linspace(-1, 1, K - 1))

    from statgpu.linear_model._ordered_logit import OrderedLogitRegression
    from statgpu.glm_core._family import Binomial, LogitLink

    if backend == "torch":
        import torch
        m = OrderedLogitRegression(n_categories=K, compute_inference=False,
                                    max_iter=50, device="torch")
        m.fit(X, y)
        coef = m.coef_
        thresh = m._thresh_est
        fam = Binomial(link=LogitLink())
        X_arr = torch.from_numpy(X).cuda().double()
        y_arr = torch.from_numpy(y).long().cuda()
        b_arr = torch.from_numpy(coef).cuda().double()
        t_arr = torch.from_numpy(thresh).cuda().double()
        prob = m._ordered_category_probs(X_arr, b_arr, t_arr, fam, K)
        prob_c = torch.clamp(prob, 1e-15, None)
        H_analytical = m._ordered_hessian_analytical(
            X_arr, y_arr, b_arr, t_arr, fam, K, prob, prob_c)
        H_analytical = H_analytical.cpu().numpy()
        torch.cuda.synchronize()
    elif backend == "cupy":
        import cupy as cp
        m = OrderedLogitRegression(n_categories=K, compute_inference=False,
                                    max_iter=50, device="cuda")
        m.fit(X, y)
        coef = m.coef_
        thresh = m._thresh_est
        fam = Binomial(link=LogitLink())
        X_arr = cp.asarray(X, dtype=cp.float64)
        y_arr = cp.asarray(y, dtype=cp.int64)
        b_arr = cp.asarray(coef, dtype=cp.float64)
        t_arr = cp.asarray(thresh, dtype=cp.float64)
        prob = m._ordered_category_probs(X_arr, b_arr, t_arr, fam, K)
        prob_c = cp.clip(prob, 1e-15, None)
        H_analytical = m._ordered_hessian_analytical(
            X_arr, y_arr, b_arr, t_arr, fam, K, prob, prob_c)
        H_analytical = H_analytical.get()
        cp.cuda.Stream.null.synchronize()
    else:
        m = OrderedLogitRegression(n_categories=K, compute_inference=False,
                                    max_iter=50)
        m.fit(X, y)
        coef = m.coef_
        thresh = m._thresh_est
        fam = Binomial(link=LogitLink())
        prob = m._ordered_category_probs(X, coef, thresh, fam, K)
        prob_c = np.clip(prob, 1e-15, None)
        H_analytical = m._ordered_hessian_analytical(
            X, y, coef, thresh, fam, K, prob, prob_c)
        H_analytical = np.asarray(H_analytical)

    # Numerical Hessian (always on CPU via numpy)
    X_np = X
    y_np = y
    d = p + K - 1
    theta = np.concatenate([coef, thresh])

    def nll_fn(theta_vec):
        beta = theta_vec[:p]
        th = theta_vec[p:]
        prob = m._ordered_category_probs(X_np, beta, th, fam, K)
        pc = np.clip(prob, 1e-15, None)
        return -np.sum(np.log(pc[y_np, np.arange(n)])) / n

    H_numerical = numerical_hessian(nll_fn, theta, h=1e-5)

    # Analytical Hessian is full (not averaged); divide by n to match NLL scale
    H_analytical = H_analytical / n

    # Compare blocks
    max_diff = np.max(np.abs(H_analytical - H_numerical))
    bb_diff = np.max(np.abs(H_analytical[:p, :p] - H_numerical[:p, :p]))
    bth_diff = np.max(np.abs(H_analytical[:p, p:] - H_numerical[:p, p:]))
    thth_diff = np.max(np.abs(H_analytical[p:, p:] - H_numerical[p:, p:]))

    print(f"\n=== Ordered Hessian FD Test [{backend}] ===")
    print(f"  n={n} p={p} K={K} d={d}")
    print(f"  max|H_diff|           = {max_diff:.2e}")
    print(f"  max|H_diff| beta-beta = {bb_diff:.2e}")
    print(f"  max|H_diff| beta-theta= {bth_diff:.2e}")
    print(f"  max|H_diff| theta-theta={thth_diff:.2e}")

    threshold = 1e-4
    passed = max_diff < threshold
    print(f"  {'PASS' if passed else 'FAIL'} (threshold={threshold:.0e})")

    return passed, max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="numpy",
                        choices=["numpy", "cupy", "torch"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--K", type=int, default=4)
    args = parser.parse_args()

    passed, diff = test_hessian(args.backend, args.seed, args.n, args.p, args.K)
    sys.exit(0 if passed else 1)
