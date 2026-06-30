"""Test Huber/Bisquare + SCAD on all backends via fista_lla_path. Tesla P100."""
import sys; sys.path.insert(0, '/root/statgpu_pkg')
import numpy as np, torch, cupy as cp, time

print(f"Device: {torch.cuda.get_device_name(0)}")
np.random.seed(42); n, p = 5000, 100
X_np = np.random.randn(n, p).astype(np.float64)
bt = np.zeros(p)
for i, v in zip([0, 25, 50, 75], [3., -2.5, 1.5, -1.]): bt[i] = v
y_np = (X_np @ bt + np.random.randn(n) * .5).astype(np.float64)
a = [0, 25, 50, 75]

X_tc = torch.tensor(X_np, dtype=torch.float64); y_tc = torch.tensor(y_np, dtype=torch.float64)
X_tg = X_tc.cuda(); y_tg = y_tc.cuda()
X_cp = cp.asarray(X_np); y_cp = cp.asarray(y_np)

cn = np.maximum(np.sqrt(np.sum(X_np**2, axis=0)), 1e-20)
Xs = X_np * (np.sqrt(n) / cn); yc = y_np - np.mean(y_np)
lam = float(np.max(np.abs(Xs.T @ yc / n)))
ap = np.geomspace(max(lam, 0.11), 0.1, 3)

from statgpu.solvers._fista_lla import fista_lla_path
from statgpu.penalties._scad import SCADPenalty
penalty = SCADPenalty(alpha=0.1)

# === Huber + SCAD ===
print("\n=== Huber + SCAD (fista_lla_path → Proximal Newton) ===")
from statgpu.losses._huber import HuberLoss
loss = HuberLoss()
for nm, X, y in [("numpy", X_np, y_np), ("torch-CPU", X_tc, y_tc),
                  ("torch-CUDA", X_tg, y_tg), ("cupy-CUDA", X_cp, y_cp)]:
    torch.cuda.synchronize(); t0 = time.time()
    c, _, it = fista_lla_path(loss, penalty, X, y, ap, max_lla_per_step=2,
                               tol=1e-6, max_iter=[100, 200, 500], fit_intercept=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    ac = sorted(np.where(np.abs(c) > 0.05)[0])
    ok = "OK" if ac == a else "FAIL"
    print(f"  [{ok}] {nm:>14}: {it:3d}it {dt:.3f}s active={ac}")

# === Bisquare + SCAD ===
print("\n=== Bisquare + SCAD (fista_lla_path → Proximal Newton) ===")
from statgpu.losses._bisquare import BisquareLoss
loss = BisquareLoss()
bo = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
loss.estimate_scale(X_np, y_np, bo)
for nm, X, y in [("numpy", X_np, y_np), ("torch-CPU", X_tc, y_tc),
                  ("torch-CUDA", X_tg, y_tg), ("cupy-CUDA", X_cp, y_cp)]:
    torch.cuda.synchronize(); t0 = time.time()
    c, _, it = fista_lla_path(loss, penalty, X, y, ap, max_lla_per_step=2,
                               tol=1e-6, max_iter=[100, 200, 500],
                               fit_intercept=True, init_coef=bo)
    torch.cuda.synchronize(); dt = time.time() - t0
    ac = sorted(np.where(np.abs(c) > 0.05)[0])
    ok = "OK" if ac == a else "FAIL"
    print(f"  [{ok}] {nm:>14}: {it:3d}it {dt:.3f}s active={ac}")

# === CoxPH Breslow + SCAD ===
print("\n=== CoxPH Breslow + SCAD (fista_lla_path → Proximal Newton) ===")
from statgpu.losses._cox_ph import CoxPartialLikelihoodLoss
cox_n, cox_p = 500, 5
cox_X = np.random.randn(cox_n, cox_p).astype(np.float64)
cox_t = np.sort(np.random.exponential(1, cox_n))[::-1].astype(np.float64)
cox_e = np.ones(cox_n, dtype=np.float64)
cox_y = np.column_stack([cox_t, cox_e])
loss = CoxPartialLikelihoodLoss(ties='breslow')
cn = np.maximum(np.sqrt(np.sum(cox_X**2, axis=0)), 1e-20)
Xs = cox_X * (np.sqrt(cox_n) / cn); yc = cox_t - np.mean(cox_t)
lam_c = float(np.max(np.abs(Xs.T @ yc / cox_n)))
ap_c = np.geomspace(max(lam_c, 0.11), 0.1, 3)
t0 = time.time()
c, _, it = fista_lla_path(loss, penalty, cox_X, cox_y, ap_c, max_lla_per_step=2,
                           tol=1e-6, max_iter=[100, 200, 500], fit_intercept=True)
dt = time.time() - t0
print(f"  [OK] numpy: {it:3d}it {dt:.3f}s coef={np.round(c, 3)}")

print("\n=== COMPLETE ===")
