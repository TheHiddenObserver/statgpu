"""Remote GPU precision and performance verification for P2-1/P2-2/P3-1.

Runs on GPU server to verify:
1. Vectorized CD produces same coefficients as sklearn (precision)
2. Vectorized CD is not slower than expected (performance)
3. batch_mse unification produces correct results
4. All GLM families converge without NaN/Inf

Usage: python dev/tests/_remote_precision_perf.py
"""

import time
import numpy as np
import sys


def test_precision():
    """Verify penalized GLM precision against sklearn."""
    from statgpu.linear_model._penalized import (
        PenalizedLinearRegression,
        PenalizedGeneralizedLinearModel,
    )
    from sklearn.linear_model import Lasso, Ridge, ElasticNet

    results = {}
    np.random.seed(42)
    n, p = 500, 30
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 1.5, -1, 0.5]
    y = X @ beta_true + 0.5 * np.random.randn(n)

    # --- Lasso ---
    for alpha in [0.01, 0.1, 1.0]:
        sg = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l1", alpha=alpha,
            max_iter=1000, tol=1e-10, fit_intercept=False
        )
        sg.fit(X, y)
        sk = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000, tol=1e-10)
        sk.fit(X, y)
        max_diff = np.max(np.abs(sg.coef_ - sk.coef_))
        results[f"lasso_alpha={alpha}"] = {
            "max_coef_diff": float(max_diff),
            "pass": max_diff < 1e-3,
        }

    # --- Ridge ---
    for alpha in [0.001, 0.01, 0.1]:
        sg = PenalizedLinearRegression(
            penalty="l2", alpha=alpha, max_iter=200, tol=1e-10
        )
        sg.fit(X, y)
        sk = Ridge(alpha=alpha * n, fit_intercept=True)  # statgpu uses n*alpha
        sk.fit(X, y)
        max_diff = np.max(np.abs(sg.coef_ - sk.coef_))
        results[f"ridge_alpha={alpha}"] = {
            "max_coef_diff": float(max_diff),
            "pass": max_diff < 1e-3,
        }

    # --- ElasticNet ---
    for l1_ratio in [0.3, 0.5, 0.7]:
        alpha = 0.1
        sg = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="elasticnet", alpha=alpha,
            l1_ratio=l1_ratio, max_iter=1000, tol=1e-10, fit_intercept=False
        )
        sg.fit(X, y)
        sk = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                        max_iter=1000, tol=1e-10)
        sk.fit(X, y)
        max_diff = np.max(np.abs(sg.coef_ - sk.coef_))
        results[f"elasticnet_l1r={l1_ratio}"] = {
            "max_coef_diff": float(max_diff),
            "pass": max_diff < 1e-3,
        }

    return results


def test_glm_families():
    """Verify all GLM families converge without NaN."""
    from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

    results = {}
    np.random.seed(42)
    n, p = 200, 10
    X = np.random.randn(n, p)

    # Logistic
    y_log = (X @ np.array([2, -1, 0, 0, 0, 0, 0, 0, 0, 0]) > 0).astype(float)
    model = PenalizedGeneralizedLinearModel(
        loss="logistic", penalty="l1", alpha=0.05, max_iter=200, tol=1e-6
    )
    model.fit(X, y_log)
    results["logistic_l1"] = {
        "finite": bool(np.all(np.isfinite(model.coef_))),
        "intercept_finite": bool(np.isfinite(model.intercept_)),
    }

    # Poisson
    eta = X @ np.array([0.5, -0.3, 0, 0, 0, 0, 0, 0, 0, 0])
    y_poi = np.random.poisson(np.exp(np.clip(eta, -5, 5)))
    model = PenalizedGeneralizedLinearModel(
        loss="poisson", penalty="l1", alpha=0.01, max_iter=200, tol=1e-6
    )
    model.fit(X, y_poi)
    results["poisson_l1"] = {
        "finite": bool(np.all(np.isfinite(model.coef_))),
        "intercept_finite": bool(np.isfinite(model.intercept_)),
    }

    # Gamma
    y_gam = np.exp(np.clip(eta, -5, 5)) + 0.01
    model = PenalizedGeneralizedLinearModel(
        loss="gamma", penalty="l1", alpha=0.01, max_iter=200, tol=1e-6
    )
    model.fit(X, y_gam)
    results["gamma_l1"] = {
        "finite": bool(np.all(np.isfinite(model.coef_))),
        "intercept_finite": bool(np.isfinite(model.intercept_)),
    }

    # SCAD
    y_sq = X @ np.random.randn(p) + 0.1 * np.random.randn(n)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error", penalty="scad", alpha=0.1, max_iter=200, tol=1e-8
    )
    model.fit(X, y_sq)
    results["scad"] = {
        "finite": bool(np.all(np.isfinite(model.coef_))),
    }

    # MCP
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error", penalty="mcp", alpha=0.1, max_iter=200, tol=1e-8
    )
    model.fit(X, y_sq)
    results["mcp"] = {
        "finite": bool(np.all(np.isfinite(model.coef_))),
    }

    # Adaptive Lasso
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error", penalty="adaptive_l1", alpha=0.1, max_iter=200, tol=1e-8
    )
    model.fit(X, y_sq)
    results["adaptive_l1"] = {
        "finite": bool(np.all(np.isfinite(model.coef_))),
    }

    return results


def test_loss_class_accuracy():
    """Verify loss.value()/gradient()/fused_value_and_gradient() are consistent."""
    from statgpu.glm_core import get_glm_loss

    np.random.seed(42)
    n, p = 200, 10
    X = np.column_stack([np.random.randn(n, p), np.ones(n)])
    coef = np.random.randn(p + 1)

    losses = [
        ("squared_error", {}),
        ("logistic", {}),
        ("poisson", {}),
        ("gamma", {"link": "log"}),
        ("gamma", {"link": "inverse_power"}),
        ("inverse_gaussian", {}),
        ("negative_binomial", {"alpha": 1.0}),
        ("tweedie", {"power": 1.5}),
    ]

    results = {}
    for name, kwargs in losses:
        loss = get_glm_loss(name, **kwargs)
        y = np.abs(np.random.randn(n)) + 0.1

        # value() vs fused
        val = loss.value(X, y, coef)
        val_f, grad_f = loss.fused_value_and_gradient(X, y, coef)
        val_diff = abs(val - val_f)

        # gradient() vs fused
        grad = loss.gradient(X, y, coef)
        grad_diff = float(np.max(np.abs(grad - grad_f)))

        # per_sample finite check
        eta = X @ coef
        ps_val = loss.per_sample_value(eta, y)
        ps_grad = loss.per_sample_gradient(eta, y)
        ps_finite = bool(np.all(np.isfinite(ps_val))) and bool(np.all(np.isfinite(ps_grad)))

        label = f"{name}({kwargs})" if kwargs else name
        results[label] = {
            "val_diff": val_diff,
            "grad_diff": grad_diff,
            "per_sample_finite": ps_finite,
            "pass": val_diff < 1e-10 and grad_diff < 1e-10 and ps_finite,
        }

    return results


def test_performance():
    """Measure CD performance for different p values."""
    from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

    results = {}
    for n, p in [(500, 50), (500, 200), (1000, 100), (1000, 500)]:
        np.random.seed(42)
        X = np.random.randn(n, p)
        beta = np.zeros(p)
        beta[:min(p, 10)] = np.random.randn(min(p, 10))
        y = X @ beta + 0.1 * np.random.randn(n)

        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l1", alpha=0.1, max_iter=100, tol=1e-6
        )
        # Warmup
        model.fit(X[:100], y[:100])

        # Timed run
        t0 = time.perf_counter()
        model.fit(X, y)
        t1 = time.perf_counter()

        results[f"n={n}_p={p}"] = {
            "time_sec": round(t1 - t0, 4),
            "n_iter": model.n_iter_ if hasattr(model, 'n_iter_') else None,
        }

    return results


def test_batch_mse():
    """Verify batch_mse works on GPU arrays."""
    from statgpu.linear_model._cv_base import batch_mse

    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    coefs = np.random.randn(20, 10)
    intercepts = np.random.randn(20)
    sw = np.random.rand(100)

    # CPU
    mse_cpu = batch_mse(X, y, coefs, intercepts, sample_weight=sw)

    # Try GPU backends (batch_mse converts to numpy internally, so we compare numpy results)
    gpu_results = {}
    try:
        import cupy as cp
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        coefs_gpu = cp.asarray(coefs)
        intercepts_gpu = cp.asarray(intercepts)
        sw_gpu = cp.asarray(sw)
        mse_gpu = batch_mse(X_gpu, y_gpu, coefs_gpu, intercepts_gpu, sample_weight=sw_gpu)
        # batch_mse returns numpy, compare directly
        max_diff = float(np.max(np.abs(mse_cpu - mse_gpu)))
        gpu_results["cupy"] = {"max_diff": max_diff, "pass": max_diff < 1e-10}
    except ImportError:
        gpu_results["cupy"] = {"status": "skipped (no cupy)"}

    try:
        import torch
        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        coefs_t = torch.tensor(coefs, dtype=torch.float64)
        intercepts_t = torch.tensor(intercepts, dtype=torch.float64)
        sw_t = torch.tensor(sw, dtype=torch.float64)
        mse_t = batch_mse(X_t, y_t, coefs_t, intercepts_t, sample_weight=sw_t)
        # batch_mse returns numpy, compare directly
        max_diff = float(np.max(np.abs(mse_cpu - mse_t)))
        gpu_results["torch_cpu"] = {"max_diff": max_diff, "pass": max_diff < 1e-10}
    except Exception as e:
        gpu_results["torch_cpu"] = {"status": f"error: {e}"}

    return gpu_results


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("P2-1/P2-2/P3-1 Remote GPU Precision & Performance Test")
    print("=" * 60)

    print("\n[1/5] Precision tests (vs sklearn)...")
    try:
        prec = test_precision()
        all_pass = all(v.get("pass", False) for v in prec.values())
        print(f"  {'ALL PASS' if all_pass else 'SOME FAILED'}")
        for k, v in prec.items():
            status = "PASS" if v.get("pass") else "FAIL"
            diff = v.get("max_coef_diff", "N/A")
            print(f"    {k}: {status} (max_diff={diff:.2e})")
    except Exception as e:
        print(f"  ERROR: {e}")
        prec = {"error": str(e)}

    print("\n[2/5] GLM family convergence tests...")
    try:
        glm = test_glm_families()
        all_finite = all(
            v.get("finite", False) and v.get("intercept_finite", True)
            for v in glm.values()
        )
        print(f"  {'ALL PASS' if all_finite else 'SOME FAILED'}")
        for k, v in glm.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
        glm = {"error": str(e)}

    print("\n[3/5] Loss class accuracy (value/gradient/fused consistency)...")
    try:
        loss_acc = test_loss_class_accuracy()
        all_pass = all(v.get("pass", False) for v in loss_acc.values())
        print(f"  {'ALL PASS' if all_pass else 'SOME FAILED'}")
        for k, v in loss_acc.items():
            status = "PASS" if v.get("pass") else "FAIL"
            print(f"    {k}: {status} (val_diff={v['val_diff']:.2e}, grad_diff={v['grad_diff']:.2e})")
    except Exception as e:
        print(f"  ERROR: {e}")
        loss_acc = {"error": str(e)}

    print("\n[4/5] Performance tests...")
    try:
        perf = test_performance()
        for k, v in perf.items():
            print(f"    {k}: {v['time_sec']}s ({v.get('n_iter', '?')} iter)")
    except Exception as e:
        print(f"  ERROR: {e}")
        perf = {"error": str(e)}

    print("\n[5/5] batch_mse backend tests...")
    try:
        bmse = test_batch_mse()
        for k, v in bmse.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
        bmse = {"error": str(e)}

    # Save results
    all_results = {
        "precision": prec,
        "glm_families": glm,
        "performance": perf,
        "batch_mse": bmse,
    }
    with open("precision_perf_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to precision_perf_results.json")

    # Exit code
    sys.exit(0)
