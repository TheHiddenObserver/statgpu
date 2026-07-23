"""Torch CoxPH parity + timing diagnostic."""
import numpy as np, torch, cupy as cp, sys, time
sys.path.insert(0,'.')
from dev.benchmarks.pr79.generators.survival import generate_coxph_penalized
from statgpu.survival import CoxPH

X, t, e, _ = generate_coxph_penalized(100, 8, 42)
penalty = 0.1
order = np.argsort(t, kind="stable")
Xs, ts, es = X[order], t[order], e[order].astype(np.int32)
n_features = Xs.shape[1]

# === NumPy ===
print("=== NumPy ===")
m_np = CoxPH(ties="efron", penalty=penalty, compute_inference=True, compute_cindex=False, tol=1e-6, max_iter=30)
m_np.fit(Xs, time=ts, event=es)
b_ref = m_np.coef_.copy()
print(f"  LL={m_np._log_likelihood:.6f}, iters={m_np._iterations}, KKT={getattr(m_np,'_final_kkt_inf',0):.2e}")

# === Torch fit ===
print("=== Torch ===")
m_t = CoxPH(ties="efron", penalty=penalty, compute_inference=True, compute_cindex=False, tol=1e-6, max_iter=30, device="torch")
Xt_in = torch.as_tensor(Xs, dtype=torch.float64, device="cuda")
m_t.fit(Xt_in, time=ts, event=es)
ll_t = m_t._log_likelihood
kkt_t = getattr(m_t, '_final_kkt_inf', 0)
diff = float(np.linalg.norm(m_t.coef_ - b_ref))
print(f"  LL={ll_t:.6f}, iters={m_t._iterations}, KKT={kkt_t:.2e}")
print(f"  coef_diff vs NumPy: {diff:.2e}")

# Torch fixed-beta BSE
model = CoxPH(ties="efron", penalty=penalty, compute_inference=False, compute_cindex=False, tol=1e-6, max_iter=30)
efron_pre = model._efron_unique_failure_indices(ts, es)
Xt = torch.as_tensor(Xs, dtype=torch.float64, device="cuda")
tt = torch.as_tensor(ts, dtype=torch.float64, device="cuda")
et = torch.as_tensor(es, dtype=torch.int32, device="cuda")
b_t = torch.as_tensor(b_ref, dtype=torch.float64, device="cuda")
_, hess_t, _ = model._compute_gradient_hessian_torch(b_t, Xt, tt, et, efron_pre, return_aux=True)
hess_np_t = hess_t.cpu().numpy()
hp = hess_np_t - 2*penalty*np.eye(n_features)
cov_t = np.linalg.solve(-hp, np.eye(n_features))
bse_t = np.sqrt(np.maximum(np.diag(cov_t), 0))
cov_np_arr = m_np._var_matrix
try:
    info_np = np.linalg.inv(cov_np_arr)
except:
    info_np = np.linalg.pinv(cov_np_arr)
bse_np = np.sqrt(np.maximum(np.diag(np.linalg.inv(info_np)), 0))
bse_err = float(np.max(np.abs(bse_t-bse_np)/np.maximum(np.abs(bse_np),1e-30)))
print(f"  fixed-beta BSE error: {bse_err:.6e}")

# === CuPy fit ===
print("=== CuPy ===")
m_c = CoxPH(ties="efron", penalty=penalty, compute_inference=True, compute_cindex=False, tol=1e-6, max_iter=30, device="cuda")
Xc_in = cp.asarray(Xs)
m_c.fit(Xc_in, time=ts, event=es)
ll_c = m_c._log_likelihood
kkt_c = getattr(m_c, '_final_kkt_inf', 0)
diff_c = float(np.linalg.norm(m_c.coef_ - b_ref))
print(f"  LL={ll_c:.6f}, iters={m_c._iterations}, KKT={kkt_c:.2e}")
print(f"  coef_diff vs NumPy: {diff_c:.2e}")

# === Timing (10 warm + measured) ===
print("\n=== Timing (penalized CoxPH fit, Tesla P100) ===")
for label, dev in [("NumPy","cpu"),("CuPy","cuda"),("Torch","torch")]:
    times = []
    for i in range(11):
        m = CoxPH(ties="efron", penalty=penalty, device=dev, compute_cindex=False, tol=1e-6, max_iter=30)
        if dev == "cuda": cp.cuda.Stream.null.synchronize()
        if dev == "torch": torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.fit(Xs, time=ts, event=es)
        if dev == "cuda": cp.cuda.Stream.null.synchronize()
        if dev == "torch": torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= 1: times.append(elapsed)
    ts_arr = sorted(times); n_t = len(ts_arr)
    speedup = times[0] / ts_arr[n_t//2] if len(times) > 0 else 0  # NumPy as baseline
    print(f"  {label}: median={ts_arr[n_t//2]*1000:.1f}ms, min={ts_arr[0]*1000:.1f}ms, max={ts_arr[-1]*1000:.1f}ms, iters={m._iterations}")
