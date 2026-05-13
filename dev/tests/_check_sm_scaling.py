"""Check statsmodels vs statgpu loss scaling."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n, p = 100, 5
X = np.random.randn(n, p)
true_coef = np.array([1.0, -0.5, 0.3, 0.0, 0.0])
z = X @ true_coef
y = (1 / (1 + np.exp(-z)) > 0.5).astype(float)

X_sm = sm.add_constant(X)
m = sm.GLM(y, X_sm, family=sm.families.Binomial())
m.scaletype = 1.0

# Check deviance vs loglike
ll = m.loglike(np.zeros(p+1))
print(f"loglike(zeros) = {ll}")
print(f"-2*loglike(zeros) = {-2*ll}")

# Fit with alpha=0 (unregularized)
res0 = m.fit_regularized(alpha=0.0, L1_wt=0.0, maxiter=1000, cnvrg_tol=1e-12)
print(f"\nalpha=0: coef_norm={np.linalg.norm(res0.params):.8f}")

# Fit with alpha=0.01 (L2)
res1 = m.fit_regularized(alpha=0.01, L1_wt=0.0, maxiter=1000, cnvrg_tol=1e-12)
print(f"alpha=0.01: coef_norm={np.linalg.norm(res1.params):.8f}")

# Check what objective statsmodels minimizes
# statsmodels minimizes: (1/2) * deviance + alpha * penalty
# = (1/2) * (-2 * loglike) + alpha * penalty  [for canonical link]
# = -loglike + alpha * penalty
# BUT for non-canonical links or with scaletype != 1, it's different.
# With scaletype=1.0, it's: deviance/(2*1.0) + alpha * penalty

coef_sm = res1.params
Xb = X_sm @ coef_sm
p_pred = 1 / (1 + np.exp(-np.clip(Xb, -500, 500)))
p_pred = np.clip(p_pred, 1e-15, 1-1e-15)
nll = -np.mean(y * np.log(p_pred) + (1-y) * np.log(1-p_pred))
l2_pen = 0.5 * np.sum(coef_sm[1:]**2)

print(f"\nnll (mean) = {nll:.10f}")
print(f"l2_pen = {l2_pen:.10f}")
print(f"nll + 0.01*l2_pen = {nll + 0.01*l2_pen:.10f}")
print(f"nll + 0.005*l2_pen = {nll + 0.005*l2_pen:.10f}")

# Now check statgpu
from statgpu.glm_core._logistic import LogisticLoss
from statgpu.penalties._l2 import L2Penalty
from statgpu.glm_core._solver import lbfgs_solver

loss = LogisticLoss()
pen = L2Penalty(alpha=0.01)

# statgpu loss = (1/n) * sum(-y*z + log(1+exp(z)))
# statsmodels loss = (1/n) * sum(-y*z + log(1+exp(z)))  [same, deviance/2 = nll*n]
# Wait - statsmodels uses deviance/(2*scalar). With scaletype=1.0, scalar=1.0
# So statsmodels minimizes: deviance/2 + alpha * penalty = nll*n + alpha*penalty
# But that's nll*n, not nll (mean)!
# statgpu minimizes: nll (mean) + alpha * penalty = nll + alpha*penalty

# So the penalty relative weight differs by factor n!
# statsmodels: penalty weight = alpha / n  (relative to mean loss)
# statgpu: penalty weight = alpha  (relative to mean loss)

# To make them equivalent: statgpu alpha = statsmodels alpha / n
# Or: statsmodels alpha = statgpu alpha * n

coef_sg, iters = lbfgs_solver(loss, pen, X, y, max_iter=1000, tol=1e-12)
obj_sg = float(loss.value(X, y, coef_sg) + pen.value(coef_sg))
print(f"\nstatgpu (alpha=0.01): coef_norm={np.linalg.norm(coef_sg):.8f} obj={obj_sg:.10f}")
print(f"statsmodels (alpha=0.01): coef_norm={np.linalg.norm(coef_sm):.8f}")
print(f"diff = {np.max(np.abs(coef_sg - coef_sm[1:])):.6e}")

# Try statgpu with alpha = 0.01/n to match statsmodels
pen2 = L2Penalty(alpha=0.01/n)
coef_sg2, _ = lbfgs_solver(loss, pen2, X, y, max_iter=1000, tol=1e-12)
print(f"\nstatgpu (alpha=0.01/n={0.01/n}): coef_norm={np.linalg.norm(coef_sg2):.8f}")
print(f"diff = {np.max(np.abs(coef_sg2 - coef_sm[1:])):.6e}")

# Try statgpu with alpha = 0.005 to match deviance/2 scaling
pen3 = L2Penalty(alpha=0.005)
coef_sg3, _ = lbfgs_solver(loss, pen3, X, y, max_iter=1000, tol=1e-12)
print(f"\nstatgpu (alpha=0.005): coef_norm={np.linalg.norm(coef_sg3):.8f}")
print(f"diff = {np.max(np.abs(coef_sg3 - coef_sm[1:])):.6e}")
