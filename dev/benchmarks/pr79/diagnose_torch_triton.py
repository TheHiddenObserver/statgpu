#!/usr/bin/env python3
"""Torch Efron binary diagnostic: isolated grouped-GEMM vs Triton vs NumPy."""

import json, sys, os
import numpy as np
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dev.benchmarks.pr79.generators.survival import generate_coxph_small_ties, generate_coxph_no_ties


def compare(path_name, grad_t, hess_t, grad_ref, hess_ref, penalty, beta,
            n_events, p):
    """Compare Torch gradient/Hessian against NumPy reference."""
    g_err = float(np.max(np.abs(grad_t - grad_ref)))
    g_rel = g_err / max(1e-15, float(n_events))
    h_err = float(np.linalg.norm(hess_t - hess_ref, 'fro'))
    h_rel = h_err / max(1.0, float(np.linalg.norm(hess_ref, 'fro')))

    # Penalized Hessian + BSE via common NumPy inversion
    hp_t = hess_t - 2.0 * penalty * np.eye(p)
    hp_ref = hess_ref - 2.0 * penalty * np.eye(p)
    info_t, info_ref = -hp_t, -hp_ref
    try:
        cov_t = np.linalg.solve(info_t, np.eye(p))
    except np.linalg.LinAlgError:
        cov_t = np.linalg.pinv(info_t)
    try:
        cov_ref = np.linalg.solve(info_ref, np.eye(p))
    except np.linalg.LinAlgError:
        cov_ref = np.linalg.pinv(info_ref)
    cov_t = 0.5 * (cov_t + cov_t.T)
    cov_ref = 0.5 * (cov_ref + cov_ref.T)
    bse_t = np.sqrt(np.maximum(np.diag(cov_t), 0.0))
    bse_ref = np.sqrt(np.maximum(np.diag(cov_ref), 0.0))
    bse_err = float(np.max(np.abs(bse_t - bse_ref) / np.maximum(np.abs(bse_ref), 1e-30)))
    cond = float(np.linalg.cond(info_ref))

    result = {
        "path": path_name,
        "grad_max_abs": round(g_err, 12),
        "grad_per_event": round(g_rel, 12),
        "hessian_rel_fro": round(h_rel, 12),
        "bse_rel": round(bse_err, 12),
        "info_cond": round(cond, 2),
        "pass": (g_rel <= 1e-8 and h_rel <= 1e-6 and bse_err <= 1e-5),
    }
    status = "PASS" if result["pass"] else "FAIL"
    print(f"  {path_name}: grad/event={g_rel:.2e} hess={h_rel:.2e} bse={bse_err:.2e} cond={cond:.0f} → {status}")
    return result


def run_tie_config(X, time_, event, beta_ref, tie_label, penalty=0.0):
    """Run all derivative paths for one tie configuration."""
    import torch
    from statgpu.survival._cox import CoxPH
    from statgpu.survival._cox_efron_triton import compute_efron_grad_hess_triton
    import cupy as cp

    X_t = torch.as_tensor(X, dtype=torch.float64, device="cuda")
    t_t = torch.as_tensor(time_, dtype=torch.float64, device="cuda")
    e_t = torch.as_tensor(event.astype(np.int32), dtype=torch.int32, device="cuda")
    b_t = torch.as_tensor(beta_ref, dtype=torch.float64, device="cuda")

    n_events = int(event.sum())
    p = len(beta_ref)

    # Build efron_pre (matching _fit_torch setup)
    model = CoxPH(ties="efron", compute_inference=False, compute_cindex=False)
    # Sort by time descending for risk-set computation
    order = torch.argsort(t_t, descending=True)
    X_s = X_t[order]; t_s = t_t[order]; e_s = e_t[order]
    # Build efron_pre structure
    unique_times, inverse, counts = torch.unique(t_s[e_s > 0], return_inverse=True, return_counts=True)
    efron_pre = (t_s, e_s, unique_times, inverse, counts, X_s)

    results = {}

    # --- NumPy reference ---
    X_g = cp.asarray(X); t_g = cp.asarray(time_); e_g = cp.asarray(event.astype(np.int32))
    b_g = cp.asarray(beta_ref)
    grad_ref, hess_ref, _ = model._compute_gradient_hessian_gpu(
        b_g, X_g, t_g, e_g, None, return_aux=True)
    grad_ref_np = cp.asnumpy(grad_ref)
    hess_ref_np = cp.asnumpy(hess_ref)

    # --- Case A: Direct grouped-GEMM ---
    print(f"\n=== {tie_label}, penalty={penalty} ===")
    print(f"  NumPy: |grad|={float(np.max(np.abs(grad_ref_np))):.6e}")
    try:
        out = model._compute_gradient_hessian_efron_grouped_gemm_torch(b_t, X_t, efron_pre)
        grad_ge, hess_ge = out[0].cpu().numpy(), out[1].cpu().numpy()
        results["grouped_gemm"] = compare(
            "grouped-GEMM", grad_ge, hess_ge, grad_ref_np, hess_ref_np,
            penalty, beta_ref, n_events, p)
    except Exception as exc:
        print(f"  grouped-GEMM: FAILED — {exc}")
        results["grouped_gemm"] = {"path": "grouped_gemm", "error": str(exc), "pass": False}

    # --- Case B: Direct Triton ---
    try:
        triton_out = compute_efron_grad_hess_triton(X_t, b_t, efron_pre)
        if triton_out is None:
            print(f"  Triton: returned None (not available)")
            results["triton"] = {"path": "triton", "note": "returned None", "pass": None}
        else:
            grad_tr, hess_tr = triton_out[0].cpu().numpy(), triton_out[1].cpu().numpy()
            results["triton"] = compare(
                "Triton", grad_tr, hess_tr, grad_ref_np, hess_ref_np,
                penalty, beta_ref, n_events, p)
    except Exception as exc:
        print(f"  Triton: FAILED — {exc}")
        results["triton"] = {"path": "triton", "error": str(exc), "pass": False}

    return results


def main():
    import torch
    print(f"Torch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Fit a NumPy CoxPH to get beta_ref for each config
    from statgpu.survival import CoxPH

    all_results = {}

    # Config 1: small ties, no penalty
    print("\n" + "="*60)
    print("Config 1: small ties (size=3), penalty=0")
    X, t, e, _ = generate_coxph_small_ties(300, 4, 42, 3)
    base = CoxPH(ties="efron", penalty=0, compute_inference=True, compute_cindex=False,
                 tol=1e-6, max_iter=30).fit(X, time=t, event=e)
    all_results["small_ties_p0"] = run_tie_config(X, t, e, base.coef_, "tie=3,pen=0")

    # Config 2: small ties, penalty=0.1
    print("\n" + "="*60)
    print("Config 2: small ties (size=3), penalty=0.1")
    base2 = CoxPH(ties="efron", penalty=0.1, compute_inference=True, compute_cindex=False,
                  tol=1e-6, max_iter=30).fit(X, time=t, event=e)
    all_results["small_ties_p01"] = run_tie_config(X, t, e, base2.coef_, "tie=3,pen=0.1", penalty=0.1)

    # Config 3: no ties, no penalty (baseline)
    print("\n" + "="*60)
    print("Config 3: no ties, penalty=0")
    X2, t2, e2, _ = generate_coxph_no_ties(200, 4, 42)
    base3 = CoxPH(ties="efron", penalty=0, compute_inference=True, compute_cindex=False,
                  tol=1e-6, max_iter=30).fit(X2, time=t2, event=e2)
    all_results["no_ties_p0"] = run_tie_config(X2, t2, e2, base3.coef_, "no-ties,pen=0")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    for config, paths in all_results.items():
        for path_name, r in paths.items():
            status = "PASS" if r.get("pass") else ("FAIL" if r.get("pass") is False else "N/A")
            print(f"  {config}/{path_name}: {status}")

    out_path = Path("results/pr79/accuracy/torch_triton_diagnostics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
