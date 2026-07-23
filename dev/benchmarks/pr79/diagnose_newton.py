#!/usr/bin/env python3
"""Newton solver convergence diagnostic for penalized CoxPH."""
import json, sys, os
import numpy as np
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
from dev.benchmarks.pr79.generators.survival import generate_coxph_penalized

def main():
    X, t, e, _ = generate_coxph_penalized(100, 8, 42)
    penalty = 0.1

    # Fit NumPy reference
    from statgpu.survival import CoxPH
    print("=== NumPy reference ===")
    m_np = CoxPH(ties="efron", penalty=penalty, compute_inference=True, compute_cindex=False,
                 tol=1e-6, max_iter=30)
    m_np.fit(Xs, time=ts, event=es)  # Use sorted data
    b_ref = m_np.coef_.copy()
    print(f"  LL={m_np._log_likelihood:.6f}, iters={m_np._iterations}, converged={m_np._converged}")
    print(f"  termination={getattr(m_np,'_termination_reason','?')}, KKT={getattr(m_np,'_final_kkt_inf','?')}")

    # Now trace CuPy iterations manually
    import cupy as cp
    Xc = cp.asarray(X); tc = cp.asarray(t); ec = cp.asarray(e.astype(np.int32))

    from statgpu.survival._cox import CoxPH as _CoxPH
    model = _CoxPH(ties="efron", penalty=penalty, compute_inference=False, compute_cindex=False,
                   tol=1e-6, max_iter=30)

    # Sort data (time ascending) for risk-set computation — matching _fit_gpu.
    order_np = np.argsort(t, kind="stable")
    Xs = X[order_np].astype(np.float64)
    ts = t[order_np].astype(np.float64)
    es = e[order_np].astype(np.int32)

    Xc = cp.asarray(Xs); tc = cp.asarray(ts); ec = cp.asarray(es)
    efron_pre = model._efron_unique_failure_indices(ts, es)

    n_features = Xs.shape[1]
    diag_idx = cp.arange(n_features)

    print(f"\n=== CuPy trace ===")
    beta = cp.zeros(n_features, dtype=cp.float64)
    kkts = []
    for it in range(30):
        grad, hess, aux = model._compute_gradient_hessian_gpu(
            beta, Xc, tc, ec, efron_pre, return_aux=True)
        pen_grad = grad - 2 * penalty * beta

        kkt_inf = float(cp.linalg.norm(pen_grad, ord=cp.inf).item())
        kkt_norm = kkt_inf / (1.0 + float(cp.linalg.norm(grad, ord=cp.inf).item())
                              + 2 * penalty * float(cp.linalg.norm(beta, ord=cp.inf).item()))

        coef_diff = float(cp.linalg.norm(beta - cp.asarray(b_ref)).item())
        ll = float(cp.asnumpy(model._compute_log_likelihood_gpu(beta, Xc, tc, ec, efron_pre)))

        kkts.append({"iter": it, "kkt_inf": round(kkt_inf, 6), "kkt_norm": round(kkt_norm, 12),
                     "coef_diff": round(coef_diff, 6), "loglik": round(ll, 6)})

        if it < 5 or it % 5 == 0:
            print(f"  it={it}: KKT={kkt_inf:.2e}/{kkt_norm:.2e} diff={coef_diff:.2e} LL={ll:.6f}")

        if kkt_norm < 1e-9:
            print(f"  Converged at iter={it}")
            break

        # Newton + penalty
        hess_pen = hess.copy()
        hess_pen[diag_idx, diag_idx] -= 2 * penalty
        delta = model._solve_newton_delta_gpu(hess_pen, pen_grad, cp)

        # Simple line search
        old_ll = model._compute_log_likelihood_gpu(beta, Xc, tc, ec, efron_pre)
        old_obj = old_ll - penalty * cp.sum(beta * beta)

        step = 1.0
        accepted = False
        for _ in range(20):
            trial = beta - step * delta
            trial_ll = model._compute_log_likelihood_gpu(trial, Xc, tc, ec, efron_pre)
            trial_obj = trial_ll - penalty * cp.sum(trial * trial)
            if float((trial_obj - old_obj).item()) > -1e-8:
                beta = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            print(f"  Line search FAILED at iter={it}")
            break

    kkts[-1]["note"] = f"final coef_diff={kkts[-1]['coef_diff']:.2e}"

    # Compare Newton direction with NumPy at same beta
    b_cp = cp.asarray(b_ref)
    grad_cp, hess_cp, _ = model._compute_gradient_hessian_gpu(b_cp, Xc, tc, ec, efron_pre, return_aux=True)
    pen_grad_cp = grad_cp - 2 * penalty * b_cp
    hess_pen_cp = hess_cp.copy()
    hess_pen_cp[diag_idx, diag_idx] -= 2 * penalty
    delta_cp = model._solve_newton_delta_gpu(hess_pen_cp, pen_grad_cp, cp)

    ll_cp = float(cp.asnumpy(model._compute_log_likelihood_gpu(b_cp, Xc, tc, ec, efron_pre)))
    obj_cp = ll_cp - penalty * float(cp.sum(b_cp * b_cp).item())

    # Compute armijo check
    trial_beta = b_cp - delta_cp
    trial_ll = float(cp.asnumpy(model._compute_log_likelihood_gpu(trial_beta, Xc, tc, ec, efron_pre)))
    trial_obj = trial_ll - penalty * float(cp.sum(trial_beta * trial_beta).item())
    direction = -(delta_cp.flatten())
    pen_grad_vec = pen_grad_cp.flatten()
    directional_deriv = float(cp.dot(direction, pen_grad_vec).item())

    print(f"\n=== Newton direction at NumPy beta_ref ===")
    print(f"  Penalized objective at beta_ref: {obj_cp:.6f}")
    print(f"  Directional derivative (d @ pen_grad): {directional_deriv:.6e}")
    print(f"  Trial objective after full step: {trial_obj:.6f}")
    print(f"  Objective change: {trial_obj - obj_cp:.6e}")
    print(f"  step=1 Armijo (c=1e-4): {'PASS' if trial_obj >= obj_cp + 1e-4 * directional_deriv else 'FAIL'}")
    print(f"  KKT_inf at beta_ref: {float(cp.linalg.norm(pen_grad_cp, ord=cp.inf).item()):.2e}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Final KKT_inf: {kkts[-1]['kkt_inf']:.2e}")
    print(f"  Final coef_diff vs NumPy: {kkts[-1]['coef_diff']:.2e}")
    print(f"  Directional derivative sign: {'POSITIVE (ascent)' if directional_deriv > 0 else 'NEGATIVE (descent)'}")

    out_path = Path("results/pr79/accuracy/newton_diagnostics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"trace": kkts, "directional_deriv": directional_deriv}, f, indent=2)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
