"""Three-backend benchmark: IRLS-CD solver accuracy + runtime on Tesla P100."""
import sys
sys.path.insert(0, '/root/statgpu_pkg')
import numpy as np
import torch
import cupy as cp
import time

print("=" * 95)
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 95)

np.random.seed(42)
tau, alpha = 0.5, 0.1

for n, p, label in [(2000,50,"Small"), (5000,100,"Medium"), (10000,200,"Large"), (20000,200,"XL-n"), (10000,500,"XL-p")]:
    X_np = np.random.randn(n, p).astype(np.float64)
    beta_true = np.zeros(p)
    na = min(6, p)
    idx = np.linspace(0, p-1, na, dtype=int)
    vals = [3., -2.5, 1.5, -1., 2., -.5]
    for i, v in zip(idx, vals):
        beta_true[i] = v
    y_np = (X_np @ beta_true + np.random.randn(n) * .5).astype(np.float64)

    X_tc = torch.tensor(X_np, dtype=torch.float64)
    y_tc = torch.tensor(y_np, dtype=torch.float64)
    X_tg = X_tc.cuda()
    y_tg = y_tc.cuda()
    X_cp = cp.asarray(X_np)
    y_cp = cp.asarray(y_np)

    cn = np.maximum(np.sqrt(np.sum(X_np**2, axis=0)), 1e-20)
    Xs = X_np * (np.sqrt(n) / cn)
    yc = y_np - np.mean(y_np)
    lam = float(np.max(np.abs(Xs.T @ yc / n)))
    alpha_path = np.geomspace(max(lam, alpha*1.1), alpha, 3)

    from statgpu.solvers._proximal_irls_quantile import proximal_irls_quantile_solver
    from statgpu.losses._quantile import QuantileLoss
    from statgpu.penalties._scad import SCADPenalty

    loss = QuantileLoss(tau)
    penalty = SCADPenalty(alpha=alpha)

    row = f"{label:>8} |"
    coefs = {}
    for nm, (X, y) in [("numpy", (X_np, y_np)), ("torch-CPU", (X_tc, y_tc)),
                         ("torch-CUDA", (X_tg, y_tg)), ("cupy-CUDA", (X_cp, y_cp))]:
        if 'CUDA' in nm: torch.cuda.synchronize()
        t0 = time.time()
        coef, _, iters = proximal_irls_quantile_solver(
            loss, penalty, X, y, alpha_path=alpha_path,
            max_lla_per_step=2, max_iter=200, tol=1e-6, fit_intercept=True)
        if 'CUDA' in nm: torch.cuda.synchronize()
        dt = time.time() - t0
        coefs[nm] = coef
        row += f" {iters:4d}it {dt:6.3f}s |"

    d_tc = np.max(np.abs(coefs["numpy"] - coefs["torch-CPU"]))
    d_tg = np.max(np.abs(coefs["numpy"] - coefs["torch-CUDA"]))
    d_cp = np.max(np.abs(coefs["numpy"] - coefs["cupy-CUDA"]))
    max_d = max(d_tc, d_tg, d_cp)
    active = sorted(np.where(np.abs(coefs["numpy"]) > 0.05)[0])
    row += f" max|diff|={max_d:.6f}, active={active}"
    print(row)
