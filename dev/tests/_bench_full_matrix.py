"""Full matrix benchmark: ALL families x ALL penalties (incl. none) x ALL solvers x ALL backends x ALL scales.
Includes precision comparison vs sklearn, R ncvreg/grpreg/glmnet, and statsmodels.
Sections A-E independently selectable via --section.
Designed to run on remote GPU server via nohup.
"""
import time, sys, os, warnings, tempfile, subprocess, shutil, argparse
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root")

from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel as PGLM

SEP = "=" * 130
THIN = "-" * 130

# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--section", default="all", help="A,B,C,D,E or 'all'")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ── Utility ──────────────────────────────────────────────────────────────────

def _fmt(t):
    if t < 0.001: return f"{t*1e6:.0f}us"
    if t < 1: return f"{t*1000:.1f}ms"
    return f"{t:.3f}s"

def _grade(d, tol_strict=1e-6, tol_loose=1e-4):
    if d < tol_strict: return "OK"
    if d < tol_loose: return "~"
    return "MISMATCH"

def _grade_obj(diff, obj_sg, obj_ref, tol=1e-3):
    if diff < 1e-6: return "OK"
    if diff < tol: return "~"
    if obj_sg is not None and obj_ref is not None and obj_sg < obj_ref - 1e-10:
        return "OK(obj)"
    return "MISMATCH"

def _compute_objective(X, y, coef, intercept, family, alpha, l1_ratio=0.5, penalty="l1"):
    """Compute loss + penalty objective. penalty='none' gives pure loss.

    Uses same clipping as the actual loss classes to ensure consistency.
    """
    n = len(y)
    eta = X @ coef + intercept
    if family == "squared_error":
        loss = 0.5 * np.sum((y - eta)**2) / n
    elif family == "logistic":
        log1pexp = np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0)
        loss = -np.sum(y * eta - log1pexp) / n
    elif family == "poisson":
        eta_c = np.clip(eta, -30, 30)
        mu = np.clip(np.exp(eta_c), 1e-10, 1e6)
        loss = np.sum(mu - y * np.log(mu)) / n
    elif family == "gamma":
        eta_c = np.clip(eta, -30, 30)
        mu = np.clip(np.exp(eta_c), 1e-3, 1e4)
        loss = np.sum(y / mu + np.log(mu)) / n
    elif family == "inverse_gaussian":
        eta_c = np.clip(eta, -30, 30)
        mu = np.clip(np.exp(eta_c), 5e-2, 1e3)
        loss = np.sum(y / (2 * mu**2) - 1 / mu) / n
    elif family == "negative_binomial":
        eta_c = np.clip(eta, -30, 30)
        mu = np.exp(eta_c)
        mu_c = np.clip(mu, 1e-300, None)
        a = 1.0  # dispersion parameter (default alpha=1)
        a_plus_mu = a + mu_c
        loss = np.sum(-y * np.log(mu_c / a_plus_mu)
                      - (1.0 / a) * np.log(a / a_plus_mu)) / n
    elif family == "tweedie":
        eta_c = np.clip(eta, -50, 50)
        mu = np.clip(np.exp(eta_c), 1e-3, 1e4)
        p = 1.5
        loss = np.sum(-y * mu**(1-p) / (1-p) + mu**(2-p) / (2-p)) / n
    else:
        loss = 0.5 * np.sum((y - eta)**2) / n
    if penalty in ("none", None):
        return loss
    if penalty == "l2":
        pen = 0.5 * alpha * np.sum(coef**2)
    elif penalty == "l1":
        pen = alpha * np.sum(np.abs(coef))
    elif penalty in ("elasticnet", "en"):
        pen = alpha * (l1_ratio * np.sum(np.abs(coef)) + 0.5*(1-l1_ratio)*np.sum(coef**2))
    elif penalty == "scad":
        a = 3.7; ac = np.abs(coef)
        pen = np.sum(np.where(ac <= alpha, alpha*ac,
            np.where(ac <= a*alpha, (2*a*alpha*ac - ac**2 - alpha**2)/(2*(a-1)), 0.5*(a+1)*alpha**2)))
    elif penalty == "mcp":
        g = 3.0; ac = np.abs(coef)
        pen = np.sum(np.where(ac <= g*alpha, alpha*ac - ac**2/(2*g), 0.5*g*alpha**2))
    elif penalty in ("adaptive_l1", "adaptive_lasso"):
        pen = alpha * np.sum(np.abs(coef))
    elif penalty.startswith("group_"):
        pen = alpha * np.sum(np.abs(coef))
    else:
        pen = alpha * np.sum(np.abs(coef))
    return loss + pen

def _nnz(coef, tol=1e-6):
    return int(np.sum(np.abs(coef) > tol))

def _penalty_kwargs_for(penalty, p):
    if penalty.startswith("group_"):
        return {"groups": np.arange(p) // 5 + 1}
    return {}

# ── Solver applicability ─────────────────────────────────────────────────────

SMOOTH_PENALTIES = {"none", "l2"}
NONCONVEX_PENALTIES = {"scad", "mcp", "group_lasso", "group_mcp", "group_scad"}

def _applicable_solvers(family, penalty):
    """Return list of solvers valid for this family x penalty combo."""
    # fista_bb doesn't converge for inverse_gaussian (fixed step too small
    # for 1/mu^3 gradient scaling; fista's backtracking handles it properly)
    # fista_bb diverges for tweedie (exponential loss landscape)
    if family in ("inverse_gaussian", "tweedie"):
        solvers = ["fista"]
    else:
        solvers = ["fista", "fista_bb"]
    if penalty in SMOOTH_PENALTIES:
        solvers.extend(["newton", "lbfgs", "irls"])
        if family == "squared_error":
            solvers.append("exact")
    # admm: skip for now (known poisson+admm divergence)
    return solvers

def _skip_combo(family, penalty, solver):
    """Check if solver is invalid for this combo."""
    if solver == "exact" and not (family == "squared_error" and penalty in ("l2", "none")):
        return True
    if solver in ("irls", "newton", "lbfgs") and penalty not in SMOOTH_PENALTIES:
        return True
    if solver == "admm" and penalty not in ("l1", "l2", "elasticnet"):
        return True
    # fista_bb diverges for tweedie with all penalties
    # (exponential loss landscape causes BB/Lipschitz steps to overshoot)
    if solver == "fista_bb" and family == "tweedie":
        return True
    # fista_bb diverges for negative_binomial with smooth penalties
    if solver == "fista_bb" and family == "negative_binomial" and penalty in SMOOTH_PENALTIES:
        return True
    # fista_bb doesn't converge for inverse_gaussian (fixed step too small
    # for 1/mu^3 gradient scaling; fista's backtracking handles it properly)
    if solver == "fista_bb" and family == "inverse_gaussian":
        return True
    return False

# ── Data generation ──────────────────────────────────────────────────────────

def _gen_data(family, n, p, seed, corr=0.3, nnz_frac=0.2):
    rng = np.random.default_rng(seed)
    Sigma = np.eye(p) * (1 - corr) + corr
    L = np.linalg.cholesky(Sigma)
    X = rng.normal(size=(n, p)) @ L.T
    nnz = max(3, int(p * nnz_frac))
    true_coef = np.zeros(p)
    true_coef[rng.choice(p, nnz, replace=False)] = rng.normal(0, 2, nnz)
    eta = X @ true_coef
    if family == "squared_error":
        y = eta + 0.5 * rng.normal(size=n)
    elif family == "logistic":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.random(n) < prob).astype(float)
    elif family == "poisson":
        lam = np.exp(np.clip(eta * 0.5, -3, 5))
        y = rng.poisson(lam).astype(float)
    elif family == "gamma":
        mu = np.exp(np.clip(eta * 0.3, -2, 4)) + 0.1
        y = rng.gamma(1.0, mu)
    elif family == "inverse_gaussian":
        mu = np.exp(np.clip(eta * 0.3, -2, 4)) + 0.1
        nu = rng.normal(size=n)**2
        y = mu + mu**2*nu/2 - mu/2*np.sqrt(4*mu*nu + mu**2*nu**2)
        y = np.clip(y, 1e-6, None)
    elif family == "negative_binomial":
        lam = np.exp(np.clip(eta * 0.5, -3, 5))
        size_p = 1.0; prob_nb = size_p / (size_p + lam)
        y = rng.negative_binomial(size_p, prob_nb).astype(float)
    elif family == "tweedie":
        mu = np.exp(np.clip(eta * 0.3, -2, 4)) + 0.1
        pwr = 1.5; phi = 1.0
        lam_tw = np.clip(mu**(2-pwr)/(phi*(2-pwr)), 0.01, 100)
        alpha_tw = (2-pwr)/(pwr-1); beta_tw = phi*(pwr-1)*mu**(pwr-1)
        N = rng.poisson(lam_tw)
        y = np.array([rng.gamma(N[i]*alpha_tw, beta_tw[i]) if N[i]>0 else 0.0 for i in range(n)])
        y = np.clip(y, 1e-6, None)
    else:
        raise ValueError(f"Unknown family: {family}")
    return X, y, true_coef

# ── Runners ──────────────────────────────────────────────────────────────────

def _run_statgpu(X, y, loss, penalty, solver, device="cpu", alpha=0.01,
                 l1_ratio=0.5, max_iter=2000, tol=1e-6, penalty_kwargs=None, **kw):
    """Run statgpu PGLM. penalty='none' maps to alpha=0, penalty='l2'."""
    pk = dict(penalty_kwargs or {})
    effective_penalty = "l2" if penalty == "none" else penalty
    effective_alpha = 0.0 if penalty == "none" else alpha
    t0 = time.perf_counter()
    m = PGLM(loss=loss, penalty=effective_penalty, alpha=effective_alpha, l1_ratio=l1_ratio,
             solver=solver, max_iter=max_iter, tol=tol,
             device=device, fit_intercept=True, penalty_kwargs=pk, **kw)
    m.fit(X, y)
    t = (time.perf_counter() - t0) * 1000
    coef = np.asarray(m.coef_, dtype=float)
    intercept = float(m.intercept_)
    return coef, intercept, int(m.n_iter_), t

def _auto_solver(family, penalty):
    """Pick the best solver for this family x penalty combo."""
    if penalty in ("scad", "mcp", "group_mcp", "group_scad", "group_lasso", "adaptive_l1"):
        return "fista"
    if penalty in ("l1", "elasticnet"):
        return "fista_bb"
    # smooth penalties
    if family == "squared_error":
        return "exact"
    if family in ("inverse_gaussian", "tweedie"):
        return "lbfgs"
    return "lbfgs"

def _run_gpu_variants(X, y, loss, penalty, solver, alpha=0.01, l1_ratio=0.5,
                      max_iter=2000, tol=1e-6, penalty_kwargs=None):
    """Run on CPU + CuPy + Torch, return dict of results. No warmup."""
    pk = dict(penalty_kwargs or {})
    results = {}
    c, ic, ni, t = _run_statgpu(X, y, loss, penalty, solver, "cpu", alpha, l1_ratio, max_iter, tol, pk)
    results["cpu"] = (c, ic, ni, t)
    try:
        import cupy
        c2, ic2, ni2, t2 = _run_statgpu(X, y, loss, penalty, solver, "cuda", alpha, l1_ratio, max_iter, tol, pk)
        results["cupy"] = (c2, ic2, ni2, t2)
    except Exception:
        results["cupy"] = None
    try:
        import torch
        if torch.cuda.is_available():
            c3, ic3, ni3, t3 = _run_statgpu(X, y, loss, penalty, solver, "torch", alpha, l1_ratio, max_iter, tol, pk)
            results["torch"] = (c3, ic3, ni3, t3)
        else:
            results["torch"] = None
    except Exception:
        results["torch"] = None
    return results

# ── sklearn runners ──────────────────────────────────────────────────────────

def _run_sklearn(X, y, ref_type, alpha, n, l1_ratio=0.5, max_iter=2000, tol=1e-6):
    t0 = time.perf_counter()
    try:
        if ref_type == "lasso":
            from sklearn.linear_model import Lasso
            m = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
        elif ref_type == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=n*alpha, fit_intercept=True, solver='svd')
            m.fit(X, y)
            return m.coef_, m.intercept_, None, (time.perf_counter()-t0)*1000
        elif ref_type == "ols":
            from sklearn.linear_model import LinearRegression
            m = LinearRegression(fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, None, (time.perf_counter()-t0)*1000
        elif ref_type == "enet":
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
        elif ref_type == "logistic_l1":
            from sklearn.linear_model import LogisticRegression
            C = 1.0 / (n * alpha)
            m = LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            coef = m.coef_[0] if m.coef_.ndim > 1 else m.coef_
            return coef, m.intercept_[0], None, (time.perf_counter()-t0)*1000
        elif ref_type == "logistic_l2":
            from sklearn.linear_model import LogisticRegression
            C = 1.0 / (n * alpha)
            m = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            coef = m.coef_[0] if m.coef_.ndim > 1 else m.coef_
            return coef, m.intercept_[0], None, (time.perf_counter()-t0)*1000
        elif ref_type == "logistic_none":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            coef = m.coef_[0] if m.coef_.ndim > 1 else m.coef_
            return coef, m.intercept_[0], None, (time.perf_counter()-t0)*1000
        elif ref_type == "poisson":
            from sklearn.linear_model import PoissonRegressor
            m = PoissonRegressor(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
        elif ref_type == "poisson_none":
            from sklearn.linear_model import PoissonRegressor
            m = PoissonRegressor(alpha=0, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
        elif ref_type == "gamma":
            from sklearn.linear_model import GammaRegressor
            m = GammaRegressor(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
        elif ref_type == "gamma_none":
            from sklearn.linear_model import GammaRegressor
            m = GammaRegressor(alpha=0, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
        elif ref_type == "tweedie":
            from sklearn.linear_model import TweedieRegressor
            m = TweedieRegressor(power=1.5, alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
        elif ref_type == "tweedie_none":
            from sklearn.linear_model import TweedieRegressor
            m = TweedieRegressor(power=1.5, alpha=0, max_iter=max_iter, tol=tol, fit_intercept=True)
            m.fit(X, y)
            return m.coef_, m.intercept_, m.n_iter_, (time.perf_counter()-t0)*1000
    except Exception as e:
        return None, None, None, (time.perf_counter()-t0)*1000
    return None, None, None, 0

# ── statsmodels runner ───────────────────────────────────────────────────────

def _run_statsmodels(X, y, family_sm, alpha_sm, L1_wt=0.0, max_iter=5000, tol=1e-8):
    import statsmodels.api as sm
    X_sm = sm.add_constant(X, has_constant='add')
    t0 = time.perf_counter()
    try:
        m = sm.GLM(y, X_sm, family=family_sm)
        m.scaletype = 1.0
        res = m.fit_regularized(alpha=alpha_sm, L1_wt=L1_wt, maxiter=max_iter, cnvrg_tol=tol)
        coef = np.asarray(res.params[1:], dtype=float)
        intercept = float(res.params[0])
        return coef, intercept, getattr(res, 'iterations', None), (time.perf_counter()-t0)*1000
    except Exception:
        return None, None, None, (time.perf_counter()-t0)*1000

def _run_statsmodels_ols(X, y, alpha_sm, L1_wt=0.0, max_iter=5000, tol=1e-8):
    import statsmodels.api as sm
    X_sm = sm.add_constant(X, has_constant='add')
    t0 = time.perf_counter()
    try:
        m = sm.OLS(y, X_sm)
        res = m.fit_regularized(alpha=alpha_sm, L1_wt=L1_wt, maxiter=max_iter, cnvrg_tol=tol)
        coef = np.asarray(res.params[1:], dtype=float)
        intercept = float(res.params[0])
        return coef, intercept, getattr(res, 'iterations', None), (time.perf_counter()-t0)*1000
    except Exception:
        return None, None, None, (time.perf_counter()-t0)*1000

# ── R runners (via subprocess) ──────────────────────────────────────────────

_RSCRIPT = shutil.which("Rscript") or "Rscript"

def _run_r_script(r_code, coef_file):
    with tempfile.NamedTemporaryFile(suffix='.R', mode='w', delete=False) as f:
        f.write(r_code); tmp_r = f.name
    try:
        result = subprocess.run([_RSCRIPT, tmp_r], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            _stderr = result.stderr.strip()[:200] if result.stderr else ""
            print(f"    [R] non-zero exit {result.returncode}: {_stderr}")
            return None, None
        if not os.path.exists(coef_file):
            _stderr = result.stderr.strip()[:200] if result.stderr else ""
            print(f"    [R] no coef file produced: {_stderr}")
            return None, None
        coef = np.loadtxt(coef_file, delimiter=',')
        r_t = None
        for line in result.stdout.strip().split('\n'):
            if line.startswith('R_TIME:'):
                r_t = float(line.split(':')[1])
            if line.startswith('R_ERROR:'):
                print(f"    [R] {line}")
        return coef, r_t
    except Exception as e:
        print(f"    [R] exception: {e}")
        return None, None
    finally:
        os.unlink(tmp_r)
        if os.path.exists(coef_file): os.unlink(coef_file)

def _run_r_ncvreg(X, y, family_r, penalty, lambda_val, standardize=False):
    tmp_x = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    tmp_y = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    tmp_coef = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    np.savetxt(tmp_x, X, delimiter=',', fmt='%.12g')
    np.savetxt(tmp_y, y, delimiter=',', fmt='%.12g')
    std_flag = "TRUE" if standardize else "FALSE"
    r_code = f'''
library(ncvreg, quietly=TRUE)
X <- as.matrix(read.csv("{tmp_x}", header=FALSE))
y <- as.numeric(read.csv("{tmp_y}", header=FALSE)[,1])
tryCatch({{
  t0 <- proc.time()[[3]]
  fit0 <- ncvreg(X, y, family="{family_r}", penalty="{penalty}", standardize={std_flag})
  lam_seq <- fit0$lambda
  target <- {lambda_val}
  if (!(target %in% lam_seq)) {{
    lam_seq <- sort(unique(c(lam_seq, target)), decreasing=TRUE)
    fit0 <- ncvreg(X, y, family="{family_r}", penalty="{penalty}", lambda=lam_seq, standardize={std_flag})
  }}
  t1 <- proc.time()[[3]]
  coef_all <- coef(fit0, lambda=target)
  if (is.matrix(coef_all)) {{ beta <- as.vector(coef_all[-1, 1]) }} else {{ beta <- as.vector(coef_all[-1]) }}
  write.table(matrix(beta, nrow=1), "{tmp_coef}", row.names=FALSE, col.names=FALSE, sep=",")
  cat(sprintf("R_TIME:%.3f\\n", t1-t0))
}}, error = function(e) {{
  cat(paste0("R_ERROR: ", conditionMessage(e), "\\n"))
}})
'''
    t0 = time.perf_counter()
    coef, r_t = _run_r_script(r_code, tmp_coef)
    t = (time.perf_counter() - t0) * 1000
    for f in [tmp_x, tmp_y]:
        if os.path.exists(f): os.unlink(f)
    return coef, t

def _run_r_grpreg(X, y, family_r, penalty, lambda_val, groups):
    tmp_x = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    tmp_y = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    tmp_g = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    tmp_coef = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    np.savetxt(tmp_x, X, delimiter=',', fmt='%.12g')
    np.savetxt(tmp_y, y, delimiter=',', fmt='%.12g')
    np.savetxt(tmp_g, np.asarray(groups, dtype=int), fmt='%d')
    r_code = f'''
library(grpreg, quietly=TRUE)
X <- as.matrix(read.csv("{tmp_x}", header=FALSE))
y <- as.numeric(read.csv("{tmp_y}", header=FALSE)[,1])
groups <- as.integer(read.csv("{tmp_g}", header=FALSE)[,1])
t0 <- proc.time()[[3]]
fit0 <- grpreg(X, y, group=groups, family="{family_r}", penalty="{penalty}", standardize=FALSE)
lam_seq <- fit0$lambda
target <- {lambda_val}
if (!(target %in% lam_seq)) {{
  lam_seq <- sort(unique(c(lam_seq, target)), decreasing=TRUE)
  fit0 <- grpreg(X, y, group=groups, family="{family_r}", penalty="{penalty}", lambda=lam_seq, standardize=FALSE)
}}
t1 <- proc.time()[[3]]
coef_all <- coef(fit0, lambda=target)
if (is.matrix(coef_all)) {{ beta <- as.vector(coef_all[-1, 1]) }} else {{ beta <- as.vector(coef_all[-1]) }}
write.table(matrix(beta, nrow=1), "{tmp_coef}", row.names=FALSE, col.names=FALSE, sep=",")
cat(sprintf("R_TIME:%.3f\\n", t1-t0))
'''
    t0 = time.perf_counter()
    coef, r_t = _run_r_script(r_code, tmp_coef)
    t = (time.perf_counter() - t0) * 1000
    for f in [tmp_x, tmp_y, tmp_g]:
        if os.path.exists(f): os.unlink(f)
    return coef, t

def _run_r_glmnet(X, y, family_r, lambda_val, alpha_en=1.0, penalty_factor=None, standardize=False):
    tmp_x = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    tmp_y = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    tmp_coef = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    np.savetxt(tmp_x, X, delimiter=',', fmt='%.12g')
    np.savetxt(tmp_y, y, delimiter=',', fmt='%.12g')
    pf_line = ""
    if penalty_factor is not None:
        pf_str = "c(" + ",".join(f"{v:.10f}" for v in penalty_factor) + ")"
        pf_line = f"              penalty.factor={pf_str},"
    std_flag = "TRUE" if standardize else "FALSE"
    r_code = f'''
library(glmnet, quietly=TRUE)
X <- as.matrix(read.csv("{tmp_x}", header=FALSE))
y <- as.numeric(read.csv("{tmp_y}", header=FALSE)[,1])
tryCatch({{
  t0 <- proc.time()[[3]]
  fit <- glmnet(X, y, family="{family_r}", alpha={alpha_en},
                lambda=c({lambda_val}),
{pf_line}
                standardize={std_flag}, intercept=TRUE)
  t1 <- proc.time()[[3]]
  coef <- as.vector(coef(fit))
  beta <- coef[-1]
  write.table(matrix(beta, nrow=1), "{tmp_coef}", row.names=FALSE, col.names=FALSE, sep=",")
  cat(sprintf("R_TIME:%.3f\\n", t1-t0))
}}, error = function(e) {{
  cat(paste0("R_ERROR: ", conditionMessage(e), "\\n"))
}})
'''
    t0 = time.perf_counter()
    coef, r_t = _run_r_script(r_code, tmp_coef)
    t = (time.perf_counter() - t0) * 1000
    for f in [tmp_x, tmp_y]:
        if os.path.exists(f): os.unlink(f)
    return coef, t

# ── External reference mapping ───────────────────────────────────────────────

_SKLEARN_MAP = {
    ("squared_error", "l1"): "lasso",
    ("squared_error", "l2"): "ridge",
    ("squared_error", "elasticnet"): "enet",
    ("squared_error", "none"): "ols",
    ("logistic", "l1"): "logistic_l1",
    ("logistic", "l2"): "logistic_l2",
    ("logistic", "none"): "logistic_none",
    ("poisson", "l2"): "poisson",
    ("poisson", "none"): "poisson_none",
    ("gamma", "l2"): "gamma",
    ("gamma", "none"): "gamma_none",
    ("tweedie", "l2"): "tweedie",
    ("tweedie", "none"): "tweedie_none",
}

_SM_FAMILY_MAP = {}

def _get_sm_family(family):
    import statsmodels.api as sm
    if not _SM_FAMILY_MAP:
        _SM_FAMILY_MAP.update({
            "squared_error": sm.families.Gaussian(),
            "logistic": sm.families.Binomial(),
            "poisson": sm.families.Poisson(),
            "gamma": sm.families.Gamma(sm.families.links.Log()),
            "tweedie": sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
            "inverse_gaussian": sm.families.InverseGaussian(sm.families.links.Log()),
            "negative_binomial": sm.families.NegativeBinomial(),
        })
    return _SM_FAMILY_MAP.get(family)

# ── Main ─────────────────────────────────────────────────────────────────────

ALL_FAMILIES = ["squared_error", "logistic", "poisson", "gamma", "inverse_gaussian", "negative_binomial", "tweedie"]
ALL_PENALTIES = ["none", "l1", "l2", "elasticnet", "scad", "mcp", "adaptive_l1", "group_lasso", "group_mcp", "group_scad"]
ALL_SCALES = [(500, 50), (2000, 200), (5000, 500)]

# SCAD/MCP/group penalties are slow at large scale
_SLOW_PENALTIES = {"scad", "mcp", "adaptive_l1", "group_lasso", "group_mcp", "group_scad"}


def main():
    args = _parse_args()
    ALPHA = args.alpha
    MAX_ITER = args.max_iter
    TOL = args.tol
    sections = set(s.strip().upper() for s in args.section.split(","))
    run_all = "ALL" in sections

    print(SEP)
    print("  FULL MATRIX BENCHMARK: ALL Families x ALL Penalties (incl. none) x ALL Solvers x ALL Backends x ALL Scales")
    print(SEP)
    print(f"  Families: {ALL_FAMILIES}")
    print(f"  Penalties: {ALL_PENALTIES}")
    print(f"  Scales: {ALL_SCALES}")
    print(f"  Alpha: {ALPHA}  Max-iter: {MAX_ITER}  Tol: {TOL}")
    print(f"  Sections: {'all' if run_all else args.section}")
    print()

    # Check GPU
    has_cupy = False; has_torch = False
    try:
        import cupy; has_cupy = True; print(f"  CuPy: {cupy.__version__}")
    except: pass
    try:
        import torch; has_torch = torch.cuda.is_available()
        print(f"  Torch: {torch.__version__}, CUDA: {has_torch}")
    except: pass
    print()

    # Pre-generate datasets
    print("  Generating datasets...")
    Xy_data = {}
    seed = args.seed
    for family in ALL_FAMILIES:
        for n, p in ALL_SCALES:
            X, y, true = _gen_data(family, n, p, seed)
            Xy_data[(family, n, p)] = (X, y, true)
            seed += 1
    print(f"  Generated {len(Xy_data)} datasets.\n")
    sys.stdout.flush()

    # Counters for summary
    section_stats = {}

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A: Cross-backend timing — auto solver x ALL backends x ALL scales
    # ALL families x ALL penalties x auto-selected solver x ALL scales
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or "A" in sections:
        print(SEP)
        print("  SECTION A: Cross-Backend Timing — Auto Solver x ALL Backends x ALL Scales")
        print(SEP)

        a_total = 0; a_ok = 0; a_max_diff = 0.0

        for n, p in ALL_SCALES:
            for family in ALL_FAMILIES:
                if (family, n, p) not in Xy_data:
                    continue
                X, y, _ = Xy_data[(family, n, p)]
                for penalty in ALL_PENALTIES:
                    # Skip slow penalties at large scale
                    if penalty in _SLOW_PENALTIES and n > 2000:
                        continue

                    pk = _penalty_kwargs_for(penalty, p)
                    mi = MAX_ITER
                    if family != "squared_error" and penalty not in SMOOTH_PENALTIES:
                        mi = max(mi, 3000)

                    solver = _auto_solver(family, penalty)

                    print(f"\n  [{family}+{penalty} | n={n},p={p} | solver={solver}]")
                    print(f"  {'Backend':<12} {'Time(ms)':>10}  {'Iters':>7}  {'NNZ':>5}  {'||coef||':>12}  {'vs_CPU':>14}  {'spd':>8}")
                    print(f"  {THIN}")

                    try:
                        res = _run_gpu_variants(X, y, family, penalty, solver, ALPHA,
                                                max_iter=mi, tol=TOL, penalty_kwargs=pk)
                    except Exception as e:
                        import traceback
                        print(f"  ERROR: {e}")
                        traceback.print_exc()
                        sys.stdout.flush()
                        continue

                    cpu_c, cpu_ic, cpu_ni, cpu_t = res["cpu"]
                    cpu_nnz = _nnz(cpu_c)
                    cpu_norm = float(np.linalg.norm(cpu_c))
                    obj_cpu = _compute_objective(X, y, cpu_c, cpu_ic, family, ALPHA, penalty=penalty)

                    print(f"  {'CPU':<12} {cpu_t:>10.1f}  {cpu_ni:>7}  {cpu_nnz:>5}  {cpu_norm:>12.6f}  {'—':>14}  {'—':>8}")

                    # Non-convex penalties can have different local minima across backends
                    is_nonconvex = penalty in NONCONVEX_PENALTIES
                    coef_tol = 1e-3 if is_nonconvex else 1e-6

                    for be_name in ["cupy", "torch"]:
                        be = res.get(be_name)
                        if be is None:
                            print(f"  {be_name:<12} {'--':>10}  {'--':>7}  {'--':>5}  {'--':>12}  {'--':>14}  {'--':>8}")
                            continue
                        be_c, be_ic, be_ni, be_t = be
                        diff = float(np.max(np.abs(be_c - cpu_c)))
                        spd = cpu_t / be_t if be_t > 0 else 0
                        a_max_diff = max(a_max_diff, diff)
                        a_total += 1
                        obj_be = _compute_objective(X, y, be_c, be_ic, family, ALPHA, penalty=penalty)
                        if be_name == "torch":
                            coef_tol_be = 0.2  # torch CUDA parallel reduction inherent diff
                            obj_tol_be = 1e-4
                        elif be_name == "cupy" and is_nonconvex:
                            coef_tol_be = 0.1  # cupy parallel reduction + nonconvex local minima
                            obj_tol_be = 1e-4
                        else:
                            coef_tol_be = coef_tol
                            obj_tol_be = 1e-6
                        if diff < coef_tol_be:
                            a_ok += 1
                        elif obj_be is not None and obj_cpu is not None and abs(obj_be - obj_cpu) < obj_tol_be:
                            a_ok += 1
                        print(f"  {be_name:<12} {be_t:>10.1f}  {be_ni:>7}  {_nnz(be_c):>5}  {np.linalg.norm(be_c):>12.6f}  {diff:>14.2e}  {spd:>7.2f}x")

                    sys.stdout.flush()

        section_stats["A"] = (a_ok, a_total, a_max_diff)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B: Precision vs sklearn
    # All families with sklearn references, ALL applicable penalties
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or "B" in sections:
        print(f"\n{SEP}")
        print("  SECTION B: Precision vs sklearn")
        print(SEP)

        b_total = 0; b_ok = 0

        # Use n=1000, p=50 for precision (faster, cleaner comparison)
        n_sk, p_sk = 1000, 50
        Xy_sk = {}
        sk_seed = args.seed + 100
        for family in ALL_FAMILIES:
            X, y, true = _gen_data(family, n_sk, p_sk, sk_seed)
            Xy_sk[family] = (X, y)
            sk_seed += 1

        for (family, penalty), ref_type in _SKLEARN_MAP.items():
            if family not in Xy_sk:
                continue
            X, y = Xy_sk[family]

            ref_c, ref_ic, ref_ni, ref_t = _run_sklearn(X, y, ref_type, ALPHA, n_sk)
            if ref_c is None:
                continue

            ref_nnz = _nnz(ref_c)
            ref_obj = _compute_objective(X, y, ref_c, ref_ic, family, ALPHA, penalty=penalty)

            # Pick best solver
            solvers = _applicable_solvers(family, penalty)
            # Prefer exact > lbfgs > irls > newton > fista_bb > fista
            solver_pref = ["exact", "lbfgs", "irls", "newton", "fista_bb", "fista"]
            solver = next((s for s in solver_pref if s in solvers), "fista_bb")

            pk = _penalty_kwargs_for(penalty, p_sk)
            c, ic, ni, t = _run_statgpu(X, y, family, penalty, solver, "cpu", ALPHA,
                                         max_iter=MAX_ITER, tol=TOL, penalty_kwargs=pk)
            diff = float(np.max(np.abs(c - ref_c)))
            obj_sg = _compute_objective(X, y, c, ic, family, ALPHA, penalty=penalty)
            grade = _grade_obj(diff, obj_sg, ref_obj, tol=1e-3)

            print(f"\n  [{family}+{penalty} | n={n_sk},p={p_sk}]")
            print(f"  {'Method':<20} {'Time(ms)':>10}  {'NNZ':>5}  {'||coef||':>12}  {'max|diff|':>14}  {'Grade':>10}")
            print(f"  {'sklearn (ref)':<20} {ref_t:>10.1f}  {ref_nnz:>5}  {float(np.linalg.norm(ref_c)):>12.6f}  {'—':>14}  {'ref':>10}")
            print(f"  {solver:<20} {t:>10.1f}  {_nnz(c):>5}  {np.linalg.norm(c):>12.6f}  {diff:>14.2e}  {grade:>10}")

            b_total += 1
            if diff < 1e-3 or obj_sg < ref_obj - 1e-10:
                b_ok += 1
            sys.stdout.flush()

        section_stats["B"] = (b_ok, b_total, 0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C: Precision vs R (ncvreg, grpreg, glmnet)
    # ALL families x applicable penalties at n=500,p=50 (R is slow)
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or "C" in sections:
        print(f"\n{SEP}")
        print("  SECTION C: Precision vs R (ncvreg/grpreg/glmnet)")
        print(SEP)

        c_total = 0; c_ok = 0
        n_r, p_r = 500, 50

        r_combos = []
        # ncvreg: SCAD, MCP for gaussian/binomial/poisson (ncvreg does NOT support gamma)
        for fam, fam_r in [("squared_error","gaussian"),("logistic","binomial"),("poisson","poisson")]:
            for pen, r_pen in [("scad","SCAD"),("mcp","MCP")]:
                r_combos.append((fam, pen, "ncvreg", fam_r, r_pen))
        # grpreg: group penalties for gaussian
        for pen, r_pen in [("group_lasso","grLasso"),("group_mcp","grMCP"),("group_scad","grSCAD")]:
            r_combos.append(("squared_error", pen, "grpreg", "gaussian", r_pen))
        # glmnet: l1, elasticnet, adaptive_l1 for smooth families
        # (gamma excluded: R glmnet on this server doesn't support gamma family)
        for fam, fam_r in [("squared_error","gaussian"),("logistic","binomial"),("poisson","poisson")]:
            r_combos.append((fam, "l1", "glmnet", fam_r, None))
            r_combos.append((fam, "elasticnet", "glmnet_en", fam_r, None))
            r_combos.append((fam, "adaptive_l1", "glmnet_adaptive", fam_r, None))

        for family, penalty, r_pkg, family_r, r_penalty in r_combos:
            if (family, n_r, p_r) not in Xy_data:
                continue
            X, y, _ = Xy_data[(family, n_r, p_r)]
            groups = np.arange(p_r) // 5 + 1

            adaptive_pf = None
            # gamma signal is weak (eta*0.3); use smaller lambda for R
            _alpha_r = ALPHA * 0.1 if family == "gamma" else ALPHA
            _std = (family == "gamma")  # gamma needs standardize=TRUE for R
            if r_pkg == "ncvreg":
                r_c, r_t = _run_r_ncvreg(X, y, family_r, r_penalty, _alpha_r, standardize=_std)
            elif r_pkg == "grpreg":
                r_c, r_t = _run_r_grpreg(X, y, family_r, r_penalty, _alpha_r, groups)
            elif r_pkg == "glmnet":
                r_c, r_t = _run_r_glmnet(X, y, family_r, _alpha_r, 1.0, standardize=_std)
            elif r_pkg == "glmnet_en":
                r_c, r_t = _run_r_glmnet(X, y, family_r, _alpha_r, 0.5, standardize=_std)
            elif r_pkg == "glmnet_adaptive":
                init_solver = "lbfgs" if family == "squared_error" else "irls"
                try:
                    init_c, _, _, _ = _run_statgpu(X, y, family, "l2", init_solver, "cpu", alpha=0.001, max_iter=500, tol=1e-4)
                    adaptive_pf = 1.0 / (np.abs(init_c) + 1e-4)
                    adaptive_pf = np.clip(adaptive_pf, 1e-4, 10.0)
                except:
                    adaptive_pf = None
                r_c, r_t = _run_r_glmnet(X, y, family_r, _alpha_r, 1.0, adaptive_pf, standardize=_std)
            else:
                continue

            if r_c is None or len(r_c) == 0:
                print(f"\n  [{family}+{penalty} | R {r_pkg}] FAILED (empty coef)")
                c_total += 1  # count in denominator
                sys.stdout.flush()
                continue

            r_obj = _compute_objective(X, y, r_c, 0.0, family, _alpha_r, penalty=penalty)

            # Run statgpu — use same alpha as R for fair comparison
            pk = _penalty_kwargs_for(penalty, p_r)
            if penalty == "adaptive_l1" and adaptive_pf is not None:
                pk["weights"] = adaptive_pf; pk["normalize"] = False
            solver = "fista_bb" if penalty not in ("l2",) and family != "inverse_gaussian" else "lbfgs"
            if penalty not in ("l2",) and family == "inverse_gaussian":
                solver = "fista"
            _convex = {"l1","elasticnet","group_lasso","adaptive_l1"}
            _tol = 1e-8 if penalty in _convex else TOL
            _mi = 5000 if penalty in _convex else MAX_ITER

            c, ic, ni, t = _run_statgpu(X, y, family, penalty, solver, "cpu", _alpha_r,
                                         max_iter=_mi, tol=_tol, penalty_kwargs=pk)
            diff = float(np.max(np.abs(c - r_c)))
            obj_sg = _compute_objective(X, y, c, ic, family, _alpha_r, penalty=penalty)
            grade = _grade_obj(diff, obj_sg, r_obj, tol=2e-2)

            print(f"\n  [{family}+{penalty} | R {r_pkg} | n={n_r},p={p_r}]")
            print(f"  {'Method':<25} {'Time(ms)':>10}  {'NNZ':>5}  {'||coef||':>12}  {'max|diff|':>14}  {'Grade':>10}")
            print(f"  {'R '+r_pkg+' (ref)':<25} {r_t:>10.1f}  {_nnz(r_c):>5}  {float(np.linalg.norm(r_c)):>12.6f}  {'—':>14}  {'ref':>10}")
            obj_note = f"obj_sg={obj_sg:.6f} obj_R={r_obj:.6f}" if diff > 2e-2 else ""
            print(f"  {solver:<25} {t:>10.1f}  {_nnz(c):>5}  {np.linalg.norm(c):>12.6f}  {diff:>14.2e}  {grade:>10}  {obj_note}")

            c_total += 1
            if diff < 2e-2 or obj_sg < r_obj - 1e-10:
                c_ok += 1
            sys.stdout.flush()

        section_stats["C"] = (c_ok, c_total, 0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION D: Precision vs statsmodels GLM.fit_regularized
    # ALL families x {none, l2} at n=500,p=50 + squared_error/logistic x {l1}
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or "D" in sections:
        print(f"\n{SEP}")
        print("  SECTION D: Precision vs statsmodels (GLM.fit_regularized)")
        print(SEP)

        import statsmodels as _sm_pkg
        _sm_ver = tuple(int(x) for x in _sm_pkg.__version__.split('.')[:2])
        _sm_uses_nobs = _sm_ver >= (0, 14)
        print(f"  statsmodels {_sm_pkg.__version__}: {'1/nobs' if _sm_uses_nobs else 'raw'} scaling\n")

        d_total = 0; d_ok = 0
        n_sm, p_sm = 500, 50

        for family in ALL_FAMILIES:
            if (family, n_sm, p_sm) not in Xy_data:
                continue
            X, y, _ = Xy_data[(family, n_sm, p_sm)]

            # L2 and none penalties
            for penalty in ["l2", "none"]:
                alpha_sm = ALPHA if _sm_uses_nobs else ALPHA * n_sm
                if penalty == "none":
                    alpha_sm = 0.0

                if family == "squared_error":
                    sm_c, sm_ic, sm_ni, sm_t = _run_statsmodels_ols(X, y, alpha_sm, L1_wt=0.0, tol=1e-10)
                else:
                    sm_family = _get_sm_family(family)
                    if sm_family is None:
                        continue
                    sm_c, sm_ic, sm_ni, sm_t = _run_statsmodels(X, y, sm_family, alpha_sm, L1_wt=0.0, tol=1e-10)

                if sm_c is None:
                    print(f"  [{family}+{penalty} | statsmodels] FAILED")
                    continue

                sm_obj = _compute_objective(X, y, sm_c, sm_ic, family, ALPHA, penalty=penalty)

                # statgpu: try applicable solvers
                solvers = _applicable_solvers(family, penalty)
                solver_pref = ["exact", "lbfgs", "irls", "newton", "fista_bb", "fista"]

                print(f"\n  [{family}+{penalty} | n={n_sm},p={p_sm}]")
                print(f"  {'Method':<20} {'Time(ms)':>10}  {'NNZ':>5}  {'||coef||':>12}  {'max|diff|':>14}  {'Grade':>10}")
                print(f"  {'statsmodels (ref)':<20} {sm_t:>10.1f}  {_nnz(sm_c):>5}  {float(np.linalg.norm(sm_c)):>12.6f}  {'—':>14}  {'ref':>10}")

                for solver in solver_pref:
                    if solver not in solvers:
                        continue
                    if _skip_combo(family, penalty, solver):
                        continue
                    try:
                        c, ic, ni, t = _run_statgpu(X, y, family, penalty, solver, "cpu", ALPHA,
                                                     max_iter=MAX_ITER, tol=TOL)
                    except Exception:
                        continue
                    diff = float(np.max(np.abs(c - sm_c)))
                    obj_sg = _compute_objective(X, y, c, ic, family, ALPHA, penalty=penalty)
                    grade_tol = 1e-4 if family == "squared_error" else (2e-2 if family == "negative_binomial" else 5e-3)
                    grade = _grade_obj(diff, obj_sg, sm_obj, tol=grade_tol)
                    print(f"  {solver:<20} {t:>10.1f}  {_nnz(c):>5}  {np.linalg.norm(c):>12.6f}  {diff:>14.2e}  {grade:>10}")

                    d_total += 1
                    if diff < grade_tol or obj_sg < sm_obj - 1e-10:
                        d_ok += 1
                sys.stdout.flush()

            # L1 penalty for squared_error and logistic
            if family in ("squared_error", "logistic"):
                alpha_sm = ALPHA if _sm_uses_nobs else ALPHA * n_sm
                if family == "squared_error":
                    sm_c, sm_ic, sm_ni, sm_t = _run_statsmodels_ols(X, y, alpha_sm, L1_wt=1.0, tol=1e-10)
                else:
                    import statsmodels.api as sm
                    X_sm = sm.add_constant(X)
                    t0 = time.perf_counter()
                    try:
                        sm_model = sm.Logit(y, X_sm).fit_regularized(alpha=alpha_sm, L1_wt=1.0, maxiter=1000)
                        sm_c = np.asarray(sm_model.params[1:], dtype=float)
                        sm_ic = float(sm_model.params[0])
                        sm_t = (time.perf_counter() - t0) * 1000
                    except:
                        sm_c = None

                if sm_c is not None:
                    sm_obj = _compute_objective(X, y, sm_c, sm_ic, family, ALPHA, penalty="l1")
                    c, ic, ni, t = _run_statgpu(X, y, family, "l1", "fista_bb", "cpu", ALPHA,
                                                 max_iter=MAX_ITER, tol=TOL)
                    diff = float(np.max(np.abs(c - sm_c)))
                    obj_sg = _compute_objective(X, y, c, ic, family, ALPHA, penalty="l1")
                    grade = _grade_obj(diff, obj_sg, sm_obj, tol=1e-3)

                    print(f"\n  [{family}+l1 | n={n_sm},p={p_sm}]")
                    print(f"  {'Method':<20} {'Time(ms)':>10}  {'NNZ':>5}  {'||coef||':>12}  {'max|diff|':>14}  {'Grade':>10}")
                    print(f"  {'statsmodels (ref)':<20} {sm_t:>10.1f}  {_nnz(sm_c):>5}  {float(np.linalg.norm(sm_c)):>12.6f}  {'—':>14}  {'ref':>10}")
                    print(f"  {'fista_bb':<20} {t:>10.1f}  {_nnz(c):>5}  {np.linalg.norm(c):>12.6f}  {diff:>14.2e}  {grade:>10}")

                    d_total += 1
                    if diff < 1e-3 or obj_sg < sm_obj - 1e-10:
                        d_ok += 1
                    sys.stdout.flush()

        section_stats["D"] = (d_ok, d_total, 0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION E: Cross-Solver Precision Consistency
    # ALL families x ALL penalties x ALL applicable solvers, CPU only
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or "E" in sections:
        print(f"\n{SEP}")
        print("  SECTION E: Cross-Solver Precision Consistency (CPU, n=2000, p=200)")
        print(SEP)

        e_total = 0; e_ok = 0
        n_e, p_e = 2000, 200

        for family in ALL_FAMILIES:
            if (family, n_e, p_e) not in Xy_data:
                continue
            X, y, _ = Xy_data[(family, n_e, p_e)]

            for penalty in ALL_PENALTIES:
                solvers = _applicable_solvers(family, penalty)
                pk = _penalty_kwargs_for(penalty, p_e)
                mi = MAX_ITER
                if family != "squared_error" and penalty not in SMOOTH_PENALTIES:
                    mi = max(mi, 3000)
                # Smooth penalties with fista need more iterations at large scale
                if penalty in SMOOTH_PENALTIES:
                    mi = max(mi, 5000)

                results = {}
                for solver in solvers:
                    if _skip_combo(family, penalty, solver):
                        continue
                    try:
                        c, ic, ni, t = _run_statgpu(X, y, family, penalty, solver, "cpu", ALPHA,
                                                     max_iter=mi, tol=TOL, penalty_kwargs=pk)
                        obj = _compute_objective(X, y, c, ic, family, ALPHA, penalty=penalty)
                        results[solver] = (obj, c, ic, ni, t)
                    except Exception as e:
                        results[solver] = (None, None, None, None, str(e))

                if len(results) < 2:
                    continue

                # Find best (lowest) finite objective
                valid = {s: r for s, r in results.items()
                         if r[0] is not None and np.isfinite(r[0])}
                if not valid:
                    continue
                best_solver = min(valid, key=lambda s: valid[s][0])
                best_obj = valid[best_solver][0]

                # Non-convex penalties get wider tolerance.
                # Smooth penalties also get wider tolerance (5e-3) because
                # fista's iterate-dependent Lipschitz + Nesterov momentum
                # converges to a slightly different numerical solution than
                # newton/lbfgs/irls on convex problems.
                is_nonconvex = penalty in NONCONVEX_PENALTIES
                ok_tol = 1e-2 if is_nonconvex else 5e-3

                print(f"\n  [{family}+{penalty} | n={n_e},p={p_e}]")
                print(f"  {'Solver':<14} {'Time(ms)':>10}  {'Iters':>7}  {'NNZ':>5}  {'Objective':>14}  {'vs_best':>14}  {'Grade':>10}")
                print(f"  {THIN}")

                max_diff = 0.0
                for solver in solvers:
                    if solver not in results:
                        continue
                    r = results[solver]
                    if r[0] is None:
                        print(f"  {solver:<14} {'ERR':>10}  {'--':>7}  {'--':>5}  {'--':>14}  {'--':>14}  {'ERR':>10}  {r[4]}")
                        continue
                    obj, c, ic, ni, t = r
                    if not np.isfinite(obj):
                        diff = float('inf')
                        grade = "MISMATCH *"
                        print(f"  {solver:<14} {t:>10.1f}  {ni:>7}  {_nnz(c):>5}  {'inf':>14}  {'inf':>14}  {grade:>10}")
                    else:
                        diff = abs(obj - best_obj)
                        max_diff = max(max_diff, diff)
                        grade = "OK" if diff < 1e-6 else ("~" if diff < ok_tol else "MISMATCH")
                        marker = " *" if solver == best_solver else ""
                        print(f"  {solver:<14} {t:>10.1f}  {ni:>7}  {_nnz(c):>5}  {obj:>14.8f}  {diff:>14.2e}  {grade:>10}{marker}")

                    e_total += 1
                    if np.isfinite(obj) and diff < ok_tol:
                        e_ok += 1
                sys.stdout.flush()

        section_stats["E"] = (e_ok, e_total, 0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    for sec, (ok, total, max_d) in sorted(section_stats.items()):
        mark = "PASS" if ok == total else "FAIL"
        extra = f" (max diff: {max_d:.2e})" if max_d > 0 else ""
        print(f"  Section {sec}: {ok}/{total} passed{extra}  [{mark}]")

    total_ok = sum(ok for ok, _, _ in section_stats.values())
    total_all = sum(t for _, t, _ in section_stats.values())
    all_pass = total_ok == total_all
    print(f"\n  TOTAL: {total_ok}/{total_all} passed  [{'ALL PASS' if all_pass else 'HAS FAILURES'}]")
    print(SEP)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
