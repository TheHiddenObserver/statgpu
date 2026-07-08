"""Compare pinball loss between CPU and GPU quantile bootstrap refits."""
import numpy as np, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)
from statgpu.linear_model import QuantileRegression

np.random.seed(42)
X = np.random.randn(200, 3)
y = 1.0 + X @ [0.5, -0.3, 0.8] + 0.5 * np.random.randn(200)
tau = 0.5

# Fit CPU reference to get fitted params
mc = QuantileRegression(quantile=tau, max_iter=1000, device='cpu')
mc.fit(X, y)
coef = mc.coef_
intercept = mc.intercept_
print(f"CPU fit: coef={coef} intercept={intercept:.6f}")
print(f"CPU loss: {np.mean(np.where(y >= X@coef+intercept, tau*(y-X@coef-intercept), (tau-1)*(y-X@coef-intercept))):.6f}")

# Now compare bootstrap refits: one CPU vs one GPU batched
# CPU: single bootstrap refit
rng = np.random.default_rng(42)
n = X.shape[0]
X_design = np.column_stack([np.ones(n), X])
eta = X_design @ np.concatenate([[intercept], coef])
resid = y - eta
y_fitted = eta
idx = rng.integers(0, n, size=n)
y_star = y_fitted + resid[idx]

# CPU refit
mc2 = QuantileRegression(quantile=tau, fit_intercept=False, max_iter=1000, device='cpu')
mc2.fit(X_design, y_star)
loss_cpu = np.mean(np.where(y_star >= X_design@mc2._params, tau*(y_star-X_design@mc2._params), (tau-1)*(y_star-X_design@mc2._params)))
print(f"\nCPU single refit: loss={loss_cpu:.8f} params={mc2._params}")

# GPU refit
for dev in ['cuda', 'torch']:
    mg = QuantileRegression(quantile=tau, fit_intercept=False, max_iter=1000, device=dev)
    y_star_gpu = mg._to_array(y_star, backend={'cuda':'cupy','torch':'torch'}[dev]) if hasattr(mg, '_to_array') else np.asarray(y_star)
    try:
        mg.fit(X_design, y_star)
        loss_gpu = np.mean(np.where(y_star >= X_design@mg._params, tau*(y_star-X_design@mg._params), (tau-1)*(y_star-X_design@mg._params)))
        print(f"{dev:6s} refit: loss={loss_gpu:.8f} params={mg._params} delta={loss_gpu-loss_cpu:+.8f}")
    except Exception as e:
        print(f"{dev:6s} FAIL: {e}")
