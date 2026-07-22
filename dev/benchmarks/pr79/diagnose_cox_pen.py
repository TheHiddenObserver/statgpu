#!/usr/bin/env python3
"""Penalized CoxPH fixed-beta derivative parity diagnostics.

Phase A: Same beta_ref → compare LL/score/Hessian/covariance/BSE
Phase B: Compare fitted-model KKT residuals and convergence quality
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dev.benchmarks.pr79.generators.survival import (
    generate_coxph_penalized, case_params_coxph_penalized,
)


def main():
    X, time_, event, beta_true = generate_coxph_penalized(100, 8, 42)
    penalty = 0.1
    cp = case_params_coxph_penalized()

    results: Dict[str, Any] = {
        "case": cp,
        "penalty": penalty,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }

    # === Phase 0: Fit NumPy model to get beta_ref ===
    from statgpu.survival import CoxPH

    print("=== Phase 0: Fit NumPy reference ===")
    model_np = CoxPH(ties="efron", penalty=penalty, compute_inference=True,
                     compute_cindex=False, tol=1e-6, max_iter=30)
    model_np.fit(X, time=time_, event=event)
    beta_ref = model_np.coef_.copy()
    results["beta_ref"] = beta_ref.tolist()
    results["numpy_fitted"] = {
        "loglik": float(model_np._log_likelihood),
        "iterations": int(model_np._iterations),
        "converged": bool(model_np._converged),
    }
    print(f"  LL={model_np._log_likelihood:.6f}, iters={model_np._iterations}")

    # === Phase A: Fixed-beta derivative parity ===
    print("\n=== Phase A: Fixed-beta derivative parity ===")

    backends = {
        "numpy": ("cpu", lambda x: x, lambda x: x),
        "cupy": ("cuda", _to_cupy, _from_cupy),
        "torch": ("torch", _to_torch, _from_torch),
    }

    for name, (dev, to_fn, from_fn) in backends.items():
        print(f"  {name}:")
        X_b, t_b, e_b = to_fn(X), time_, event
        beta_b = to_fn(beta_ref)

        model = CoxPH(ties="efron", penalty=penalty, compute_inference=False,
                      device=dev, compute_cindex=False, tol=1e-6, max_iter=30)

        # Compute gradient + Hessian at fixed beta (no optimization)
        grad, hess, aux = model._compute_gradient_hessian_gpu(
            beta_b, X_b, t_b, e_b, None, return_aux=True,
        ) if name == "cupy" else model._compute_gradient_hessian_torch(
            beta_b, X_b, t_b, e_b, None, return_aux=True,
        ) if name == "torch" else _compute_cpu_grad_hess(model, beta_ref, X, time_, event)

        # Log-likelihood from aux stats
        if name == "cupy":
            ll = float(_from_cupy(model._compute_log_likelihood_gpu_from_stats(
                aux[0], aux[1], aux[2], t_b, e_b, None)))
        elif name == "torch":
            ll = float(model._compute_log_likelihood_torch_from_stats(
                aux[0], aux[1], aux[2], t_b, e_b, None).item())
        else:
            import cupy as cp
            ll = model._compute_log_likelihood_gpu(beta_b, X_b, t_b, e_b, None)[0] if False else None
            ll = float(_compute_cpu_ll(model, beta_ref, X, time_, event))

        # Hessian to numpy
        hess_np = from_fn(hess)

        # Penalized Hessian: H_pen = H_data - 2*lambda*I
        p = hess_np.shape[0]
        hess_pen_np = hess_np - 2.0 * penalty * np.eye(p)

        # Covariance: V = inv(-H_pen)
        info = -hess_pen_np
        try:
            cov = np.linalg.solve(info, np.eye(p))
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(info)
        cov = 0.5 * (cov + cov.T)  # symmetrize
        bse = np.sqrt(np.maximum(np.diag(cov), 0.0))
        cond = float(np.linalg.cond(info))

        results[f"{name}_fixed"] = {
            "loglik": round(ll, 10),
            "hessian_max_abs": float(np.max(np.abs(hess_np))),
            "info_cond": round(cond, 2),
            "min_eig": float(np.min(np.linalg.eigvalsh(info))),
            "bse": bse.tolist(),
            "covariance_1_1": float(cov[0, 0]),
        }
        print(f"    LL={ll:.10f}, cond={cond:.1f}, bse[0]={bse[0]:.8f}")

    # === Phase A comparison ===
    ref = results["numpy_fixed"]
    for name in ["cupy", "torch"]:
        r = results[f"{name}_fixed"]
        bse_np = np.array(results["numpy_fixed"]["bse"])
        bse_b = np.array(r["bse"])
        bse_err = float(np.max(np.abs(bse_b - bse_np) / np.maximum(np.abs(bse_np), 1e-30)))
        r["bse_rel_error_vs_numpy"] = round(bse_err, 12)
        r["ll_rel_error_vs_numpy"] = abs(r["loglik"] - ref["loglik"]) / (1.0 + abs(ref["loglik"]))
        print(f"  {name} vs NumPy: bse_rel={bse_err:.6e}, ll_rel={r['ll_rel_error_vs_numpy']:.2e}")

    # === Phase B: Fitted-model KKT residuals ===
    print("\n=== Phase B: Fitted-model KKT residuals ===")

    for name, (dev, to_fn, from_fn) in backends.items():
        if name == "numpy":
            beta_b = beta_ref
        else:
            model = CoxPH(ties="efron", penalty=penalty, compute_inference=True,
                          device=dev, compute_cindex=False, tol=1e-6, max_iter=30)
            X_b, t_b, e_b = to_fn(X), time_, event
            model.fit(X_b, time=t_b, event=e_b)
            beta_b = from_fn(model.coef_)

        # Compute gradient at fitted beta
        X_b, t_b, e_b = to_fn(X), time_, event
        beta_dev = to_fn(beta_b)

        if name == "cupy":
            grad, _, _ = model._compute_gradient_hessian_gpu(
                beta_dev, X_b, t_b, e_b, None, return_aux=True)
            grad_np = _from_cupy(grad)
        elif name == "torch":
            grad, _, _ = model._compute_gradient_hessian_torch(
                beta_dev, X_b, t_b, e_b, None, return_aux=True)
            grad_np = grad.cpu().numpy()
        else:
            grad_np = _compute_cpu_grad(model, beta_ref, X, time_, event)[0]

        # KKT: score - 2*lambda*beta
        kkt = grad_np - 2.0 * penalty * beta_b
        kkt_inf = float(np.max(np.abs(kkt)))
        kkt_norm = kkt_inf / (1.0 + float(np.max(np.abs(grad_np))) + 2.0 * penalty * float(np.max(np.abs(beta_b))))

        results[f"{name}_kkt"] = {
            "kkt_inf": round(kkt_inf, 12),
            "kkt_normalized": round(kkt_norm, 12),
            "grad_inf": float(np.max(np.abs(grad_np))),
        }
        print(f"  {name}: KKT_inf={kkt_inf:.2e}, KKT_norm={kkt_norm:.2e}")

    # === Classification ===
    print("\n=== Classification ===")
    cupy_bse = results["cupy_fixed"]["bse_rel_error_vs_numpy"]
    torch_bse = results["torch_fixed"]["bse_rel_error_vs_numpy"]
    cupy_kkt = results["cupy_kkt"]["kkt_normalized"]
    torch_kkt = results["torch_kkt"]["kkt_normalized"]

    cupy_ll = results["cupy_fixed"]["ll_rel_error_vs_numpy"]
    torch_ll = results["torch_fixed"]["ll_rel_error_vs_numpy"]
    cupy_cond = results["cupy_fixed"]["info_cond"]

    classification = "PASS"
    reasons = []

    # Check fixed-beta LL parity
    if cupy_ll > 1e-8 or torch_ll > 1e-8:
        classification = "DERIVATIVE_DIFFERENCE"
        reasons.append(f"fixed-beta LL differs (cupy={cupy_ll:.2e}, torch={torch_ll:.2e})")

    # Check fixed-beta BSE
    if cupy_bse > 1e-5 or torch_bse > 1e-5:
        if cupy_cond > 1e8:
            classification = "CONDITION_SENSITIVE_WARNING"
            reasons.append(f"BSE diff amplified by high condition number (cond={cupy_cond:.0f})")
        elif cupy_kkt > 1e-7 or torch_kkt > 1e-7:
            classification = "OPTIMIZER_DIFFERENCE"
            reasons.append(f"KKT threshold exceeded (cupy={cupy_kkt:.2e}, torch={torch_kkt:.2e})")
        else:
            classification = "DERIVATIVE_DIFFERENCE"
            reasons.append(f"BSE diff without high cond or KKT issue")

    results["classification"] = classification
    results["reasons"] = reasons
    print(f"  {classification}")
    for r in reasons:
        print(f"    - {r}")

    # Save
    out_path = Path("results/pr79/accuracy/cox_pen_diagnostics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


# ===== Helpers =====

def _to_cupy(x):
    import cupy as cp
    return cp.asarray(x) if isinstance(x, np.ndarray) else x

def _from_cupy(x):
    import cupy as cp
    return cp.asnumpy(x) if hasattr(x, "get") else x

def _to_torch(x):
    import torch
    return torch.as_tensor(x, dtype=torch.float64, device="cuda") if isinstance(x, np.ndarray) else x

def _from_torch(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else x

def _compute_cpu_grad_hess(model, beta, X, time_, event):
    """Use CuPy path for CPU gradient/Hessian at fixed beta."""
    import cupy as cp
    X_g = cp.asarray(X); t_g = cp.asarray(time_); e_g = cp.asarray(event)
    b_g = cp.asarray(beta)
    g, h, a = model._compute_gradient_hessian_gpu(b_g, X_g, t_g, e_g, None, return_aux=True)
    return cp.asnumpy(g), cp.asnumpy(h), a

def _compute_cpu_grad(model, beta, X, time_, event):
    g, _, _ = _compute_cpu_grad_hess(model, beta, X, time_, event)
    return g, None, None

def _compute_cpu_ll(model, beta, X, time_, event):
    import cupy as cp
    X_g = cp.asarray(X); t_g = cp.asarray(time_); e_g = cp.asarray(event)
    b_g = cp.asarray(beta)
    return float(cp.asnumpy(model._compute_log_likelihood_gpu(
        b_g, X_g, t_g, e_g, None)))


if __name__ == "__main__":
    main()
