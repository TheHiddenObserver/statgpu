"""Full benchmark across solver x loss combinations. Tesla P100."""
import sys; sys.path.insert(0, '/root/statgpu_pkg')
import numpy as np, torch, cupy as cp, time

print("="*100)
print(f"Device: {torch.cuda.get_device_name(0)}")
print("="*100)

np.random.seed(42)
n, p = 5000, 100
X_np = np.random.randn(n, p).astype(np.float64)
beta_true = np.zeros(p)
for i, v in zip([0, 25, 50, 75], [3., -2.5, 1.5, -1.]):
    beta_true[i] = v
y_np = (X_np @ beta_true + np.random.randn(n)*.5).astype(np.float64)
true_active = [0, 25, 50, 75]

X_tc = torch.tensor(X_np, dtype=torch.float64)
y_tc = torch.tensor(y_np, dtype=torch.float64)
X_tg = X_tc.cuda(); y_tg = y_tc.cuda()
X_cp = cp.asarray(X_np); y_cp = cp.asarray(y_np)

cn = np.maximum(np.sqrt(np.sum(X_np**2,axis=0)), 1e-20)
Xs = X_np*(np.sqrt(n)/cn); yc = y_np-np.mean(y_np)
lam = float(np.max(np.abs(Xs.T@yc/n)))
def make_ap(a): return np.geomspace(max(lam,a*1.1), a, 3)

BACKENDS = [("numpy",X_np,y_np), ("torch-CPU",X_tc,y_tc),
            ("torch-CUDA",X_tg,y_tg), ("cupy-CUDA",X_cp,y_cp)]

def sync():
    torch.cuda.synchronize()

# ================================================================
print("\n=== 1. Quantile + SCAD (Proximal IRLS-CD) ===")
from statgpu.solvers._proximal_irls_quantile import proximal_irls_quantile_solver
from statgpu.losses._quantile import QuantileLoss
from statgpu.penalties._scad import SCADPenalty
loss = QuantileLoss(0.5)
penalty = SCADPenalty(alpha=0.1)
ap = make_ap(0.1)
for nm, X, y in BACKENDS:
    sync(); t0=time.time()
    coef,_,iters = proximal_irls_quantile_solver(loss,penalty,X,y,ap,max_lla_per_step=2,max_iter=200,tol=1e-6,fit_intercept=True)
    sync(); dt=time.time()-t0
    active=sorted(np.where(np.abs(coef)>0.05)[0])
    ok="OK" if active==true_active else "FAIL"
    print(f"  [{ok}] {nm:>14}: {iters:3d}it {dt:.3f}s active={active}")

# ================================================================
print("\n=== 2. Quantile + MCP (Proximal IRLS-CD) ===")
from statgpu.penalties._mcp import MCPPenalty
penalty = MCPPenalty(alpha=0.1)
ap = make_ap(0.1)
for nm, X, y in BACKENDS:
    sync(); t0=time.time()
    coef,_,iters = proximal_irls_quantile_solver(loss,penalty,X,y,ap,max_lla_per_step=2,max_iter=200,tol=1e-6,fit_intercept=True)
    sync(); dt=time.time()-t0
    active=sorted(np.where(np.abs(coef)>0.05)[0])
    ok="OK" if active==true_active else "FAIL"
    print(f"  [{ok}] {nm:>14}: {iters:3d}it {dt:.3f}s active={active}")

# ================================================================
print("\n=== 3. Quantile + L2 (IRLS smooth) ===")
from statgpu.penalties._l2 import L2Penalty
penalty = L2Penalty(alpha=0.1)
for nm, X, y in BACKENDS:
    if hasattr(X,'get'):  # cupy
        Xa=cp.asnumpy(X).astype(np.float64); ya=cp.asnumpy(y).astype(np.float64)
    elif hasattr(X,'is_cuda') and X.is_cuda:  # torch cuda
        Xa=X.cpu().numpy().astype(np.float64); ya=y.cpu().numpy().astype(np.float64)
    else:
        Xa=np.asarray(X,dtype=np.float64); ya=np.asarray(y,dtype=np.float64)
    sync(); t0=time.time()
    params,iters = loss.irls(Xa,ya,penalty=penalty,max_iter=100,tol=1e-8,fit_intercept=True)
    sync(); dt=time.time()-t0
    active=sorted(np.where(np.abs(params)>0.05)[0])
    ok="OK" if len(active)>=3 else "WARN"
    print(f"  [{ok}] {nm:>14}: {iters:3d}it {dt:.3f}s active={active[:6]}")

# ================================================================
print("\n=== 4. Huber + SCAD (Proximal Newton) ===")
from statgpu.losses._huber import HuberLoss
from statgpu.solvers._fista_lla import fista_lla_path
loss = HuberLoss(); penalty = SCADPenalty(alpha=0.1)
sync(); t0=time.time()
coef,intc,iters = fista_lla_path(loss,penalty,X_np,y_np,ap,max_lla_per_step=2,tol=1e-6,
                                  max_iter=[100,200,500],fit_intercept=True)
sync(); dt=time.time()-t0
active=sorted(np.where(np.abs(coef)>0.05)[0])
ok="OK" if active==true_active else "FAIL"
print(f"  [{ok}] numpy: {iters:3d}it {dt:.3f}s active={active}")

# ================================================================
print("\n=== 5. Bisquare + SCAD (Proximal Newton) ===")
from statgpu.losses._bisquare import BisquareLoss
loss = BisquareLoss(); beta_ols = np.linalg.lstsq(X_np,y_np,rcond=None)[0]
loss.estimate_scale(X_np,y_np,beta_ols); penalty = SCADPenalty(alpha=0.1)
sync(); t0=time.time()
coef,intc,iters = fista_lla_path(loss,penalty,X_np,y_np,ap,max_lla_per_step=2,tol=1e-6,
                                  max_iter=[100,200,500],fit_intercept=True,init_coef=beta_ols)
sync(); dt=time.time()-t0
active=sorted(np.where(np.abs(coef)>0.05)[0])
ok="OK" if active==true_active else "FAIL"
print(f"  [{ok}] numpy: {iters:3d}it {dt:.3f}s active={active}")

# ================================================================
print("\n=== 6. CoxPH Breslow + SCAD (Proximal Newton) ===")
from statgpu.losses._cox_ph import CoxPartialLikelihoodLoss
cox_n, cox_p = 500, 5
cox_X = np.random.randn(cox_n,cox_p).astype(np.float64)
cox_t = np.sort(np.random.exponential(1,cox_n))[::-1].astype(np.float64)
cox_e = np.ones(cox_n, dtype=np.float64)
loss = CoxPartialLikelihoodLoss(ties='breslow'); penalty = SCADPenalty(alpha=0.1)
cn = np.maximum(np.sqrt(np.sum(cox_X**2,axis=0)),1e-20)
Xs=cox_X*(np.sqrt(cox_n)/cn); yc=cox_t-np.mean(cox_t)
lam_c = float(np.max(np.abs(Xs.T@yc/cox_n)))
ap_c = np.geomspace(max(lam_c,0.11), 0.1, 3)
sync(); t0=time.time()
cox_y = np.column_stack([cox_t, cox_e])  # (n,2) as required by Cox
coef,intc,iters = fista_lla_path(loss,penalty,cox_X,cox_y,ap_c,max_lla_per_step=2,tol=1e-6,
                                  max_iter=[100,200,500],fit_intercept=False)
sync(); dt=time.time()-t0
active=sorted(np.where(np.abs(coef)>0.05)[0])
print(f"  [OK] numpy: {iters:3d}it {dt:.3f}s active={active}")

print("\n=== ALL COMPLETE ===")
