# Solver algorithms

> Language: English
> Last updated: 2026-09-04
> Switch: [简体中文](../../cn/guides/solver-algorithms.md)

## Scope and objective

Most statgpu estimators minimize a composite objective

$$
\min_{\beta\in\mathbb R^p}
F(\beta)
=
f(\beta)+P(\beta),
$$

where $f$ is an average data-fit loss and $P$ is either zero, a smooth penalty such as Ridge, or a non-smooth/non-convex penalty. The intercept is normally excluded from $P$.

This page inventories **10 general-purpose solve paths** found in the current source. Eight are low-level functions exported by `statgpu.solvers`, IRLS is exported as `statgpu.glm_core.IRLSSolver`, and `exact` is an estimator-level closed-form path. Model-specific algorithms are documented only with their model; see [quantile regression](../models/quantile.md#algorithm-details) for its specialized solvers.

::: warning API boundary
The inventory is broader than any one estimator's accepted `solver=` values. Use `solver="auto"` unless you have checked the [solver × penalty compatibility matrix](solver-penalty-matrix.md). Direct solver functions expect loss and penalty objects and are mainly advanced interfaces.
:::

## Start from the model page

Choose a solver from the model you are fitting, not from the complete inventory below. Each linked page lists the public selector, its default, the resolved `auto` path, and combinations that are intentionally unavailable.

| Model surface | Solver control | Model-specific documentation |
|---|---|---|
| `GeneralizedLinearModel` and solver-enabled typed GLMs | `solver` | [Generalized linear models](../models/generalized-linear-model.md#solver-support), [Poisson](../models/poisson-regression.md#solver-support) |
| `LogisticRegression` typed wrapper | fixed IRLS; no selector | [Logistic regression](../models/logistic-regression.md#solver-support) |
| Ridge | `solver` | [Ridge](../models/ridge.md#solver-support) |
| Lasso and Elastic Net | `solver`; CV/path helpers also expose `cpu_solver` | [Lasso](../models/lasso.md#solver-support), [Elastic Net](../models/elastic-net.md#solver-support) |
| Adaptive Lasso, SCAD, MCP | routed model-specific continuation/weighted-L1 paths | [Adaptive Lasso](../models/adaptive-lasso.md#solver-support), [SCAD](../models/scad.md#solver-support), [MCP](../models/mcp.md#solver-support) |
| Robust regression | estimator-specific `solver` rules | [Robust regression](../models/robust.md#solver-support) |
| Cox and ordered response models | fixed Newton path or penalized `solver` | [CoxPH](../models/coxph.md#solver-support), [ordered models](../models/ordered.md#solver-support) |
| PCA, KernelPCA, NMF | `svd_solver`, `eigen_solver`, or `solver` | [PCA](../unsupervised/pca.md#solver-support), [kernel methods](../models/kernel-methods.md#solver-support), [NMF](../unsupervised/nmf.md#solver-support) |

Names such as `fista_lla_path`, `proximal_newton_solver`, and `lbfgs_b_solver` describe internal or low-level APIs. They are not universal values for an estimator's `solver=` parameter.

## Complete inventory

| # | Path | Public status | Main use | Backends |
|---:|---|---|---|---|
| 1 | FISTA | `fista_solver`; estimator-routed | Convex non-smooth penalties and general composite objectives | NumPy, CuPy, Torch |
| 2 | FISTA-BB | `fista_bb_solver`; estimator-routed | Sparse GLMs with adaptive spectral steps | NumPy, CuPy, Torch |
| 3 | FISTA-LLA | `fista_lla_path`; continuation subpath | SCAD, MCP, Group SCAD, and Group MCP | NumPy, CuPy, Torch |
| 4 | Newton-Raphson | `newton_solver`; estimator-routed | Smooth losses with no penalty or L2 | NumPy, CuPy, Torch |
| 5 | Proximal-Newton facade | `proximal_newton_solver`; direct low-level API | Smooth Newton path; delegates non-smooth cases to FISTA | NumPy, CuPy, Torch |
| 6 | GLM IRLS | `IRLSSolver`; estimator-routed | Ordinary GLMs and supported smooth penalized GLMs | NumPy, CuPy, Torch |
| 7 | L-BFGS | `lbfgs_solver`; estimator-routed | Smooth objectives when storing a Hessian is undesirable | NumPy, CuPy, Torch |
| 8 | L-BFGS-B | `lbfgs_b_solver`; direct low-level API | Smooth objectives with box constraints | NumPy, CuPy, Torch |
| 9 | ADMM | `admm_solver`; explicit estimator route | Split smooth-loss and proximal-penalty updates | NumPy, CuPy, Torch |
| 10 | Exact Ridge solve | estimator `solver="exact"` | Squared error with L2 | NumPy, CuPy, Torch |

Each algorithm section below explains the method in prose and uses concise author-year citations near the derivation. Full bibliographic entries are collected at the end of the page. The citations identify the mathematical method family; statgpu's backend kernels, safeguards, stopping rules, and estimator routing remain implementation-specific.

## Shared notation: proximal operator

For a step size $\gamma>0$, statgpu uses

$$
\operatorname{prox}_{\gamma P}(v)
=
\arg\min_u
\left\{
\frac{1}{2}\lVert u-v\rVert_2^2+\gamma P(u)
\right\}.
$$

For $P(\beta)=\lambda\lVert\beta\rVert_1$, this is elementwise soft thresholding:

$$
\mathcal S_{\gamma\lambda}(v_j)
=
\operatorname{sign}(v_j)
\max\left(|v_j|-\gamma\lambda,0\right).
$$

## 1. FISTA

**Source:** `statgpu/solvers/_fista.py`

FISTA combines a proximal-gradient step with Nesterov momentum, following Beck and Teboulle (2009). The implementation's restart behavior is related to the adaptive restart strategy of O'Donoghue and Candès (2015). With a local Lipschitz estimate $L_k$,

$$
\beta_{k+1}
=
\operatorname{prox}_{P/L_k}
\left(
z_k-\frac{1}{L_k}\nabla f(z_k)
\right),
$$

$$
t_{k+1}
=
\frac{1+\sqrt{1+4t_k^2}}{2},
\qquad
z_{k+1}
=
\beta_{k+1}
+
\frac{t_k-1}{t_{k+1}}
\left(\beta_{k+1}-\beta_k\right).
$$

The implementation uses backtracking when needed, tracks the best objective for divergence recovery, and can cap or disable momentum for steep exponential-link losses. GPU paths fuse the proximal and momentum operations. The default controls are `max_iter=1000` and `tol=1e-4`.

## 2. FISTA-BB

**Source:** `statgpu/solvers/_fista_bb.py`

FISTA-BB retains the accelerated proximal structure of Beck and Teboulle (2009) while using the spectral step-size construction of Barzilai and Borwein (1988). It estimates a local step from successive parameters and smooth gradients:

$$
s_k=\beta_k-\beta_{k-1},
\qquad
y_k=\nabla f(\beta_k)-\nabla f(\beta_{k-1}).
$$

The two Barzilai-Borwein candidates are

$$
\gamma_k^{\mathrm{BB1}}
=
\frac{s_k^\top s_k}{s_k^\top y_k},
\qquad
\gamma_k^{\mathrm{BB2}}
=
\frac{s_k^\top y_k}{y_k^\top y_k}.
$$

The solver alternates these candidates after a burn-in, clips them to safe bounds, and applies adaptive restart when momentum opposes descent. BB adaptation is disabled for SCAD/MCP and their group variants because changing LLA weights makes the spectral estimate unstable. Quadratic losses may delegate to standard FISTA.

## 3. FISTA-LLA

**Sources:** `statgpu/solvers/_fista_lla.py` and `_fista_lla_group_contract.py`

The SCAD/MCP setting follows the non-concave penalization framework of Fan and Li (2001), while the local linear approximation strategy follows Zou and Li (2008). It replaces a non-convex coordinate penalty near $\beta^{(m)}$ by a weighted L1 surrogate:

$$
P_\lambda(|\beta_j|)
\approx
P_\lambda(|\beta_j^{(m)}|)
+
P_\lambda'(|\beta_j^{(m)}|)
\left(|\beta_j|-|\beta_j^{(m)}|\right).
$$

Ignoring constants, each outer iteration solves

$$
\min_\beta
f(\beta)
+
\sum_{j=1}^p
\omega_j^{(m)}|\beta_j|,
\qquad
\omega_j^{(m)}
=
P_\lambda'(|\beta_j^{(m)}|),
$$

with FISTA. A decreasing continuation path $\lambda_1>\cdots>\lambda_M$ warm-starts difficult SCAD/MCP problems. Group SCAD/MCP uses the same construction on $\lVert\beta_g\rVert_2$ with a group-aware FISTA inner problem.

## 4. Newton-Raphson

**Source:** `statgpu/solvers/_newton.py`

The Newton direction and line-search treatment follow the standard presentation of Nocedal and Wright (2006). For a smooth objective, define

$$
g_k=\nabla F(\beta_k),
\qquad
H_k=\nabla^2F(\beta_k).
$$

The regularized Newton direction solves

$$
\left(H_k+\delta I\right)d_k=g_k,
\qquad
\beta_{k+1}=\beta_k-a_kd_k,
$$

where $\delta=10^{-10}$ is a numerical ridge and $a_k$ is selected by Armijo backtracking. A constant Hessian is cached. A genuine rank failure may use least squares; other numerical errors are not silently swallowed. This solver accepts only smooth penalties and only uniform `sample_weight`.

## 5. Proximal-Newton facade

**Source:** `statgpu/solvers/_proximal_newton.py`

The composite proximal-Newton framework is described by Lee, Sun, and Saunders (2014). A full proximal-Newton method for non-smooth $P$ would solve

$$
d_k
=
\arg\min_d
\left\{
g_k^\top d
+
\frac{1}{2}d^\top H_kd
+
P(\beta_k+d)
\right\}.
$$

That Hessian-metric proximal subproblem is **not implemented**. The current public function runs Newton for L2/no penalty and emits a `RuntimeWarning` before delegating any non-smooth penalty to `fista_solver`. This explicit boundary prevents an incorrect Euclidean-prox shortcut from being presented as proximal Newton.

## 6. GLM IRLS

**Source:** `statgpu/glm_core/_irls.py`

IRLS is developed by Green (1984) and in the GLM treatment of McCullagh and Nelder (1989). At the current mean $\mu_i$ and linear predictor $\eta_i=g(\mu_i)$, it builds

$$
z_i
=
\eta_i
+
(y_i-\mu_i)g'(\mu_i),
\qquad
w_i
=
\frac{\left(d\mu_i/d\eta_i\right)^2}{V(\mu_i)}.
$$

It then solves the weighted Ridge system

$$
\left(\tilde X^\top W\tilde X+\lambda D\right)\beta_{\mathrm{new}}
=
\tilde X^\top Wz,
$$

where $D$ normally leaves the intercept unpenalized. A backtracking line search checks the registered family/link objective. `IRLSSolver` supports analytic sample weights and NumPy, CuPy, and Torch arrays. On the ordinary GLM surface, `solver="auto"` currently resolves to IRLS; set `C=0` for an unpenalized IRLS fit.

## 7. L-BFGS

**Source:** `statgpu/solvers/_lbfgs.py`

The limited-memory update follows Liu and Nocedal (1989). L-BFGS approximates $H_k^{-1}g_k$ from the most recent

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k,
\qquad
\rho_k=(y_k^\top s_k)^{-1}
$$

pairs. The standard two-loop recursion produces $d_k=-B_k g_k$ without forming a dense Hessian. Armijo backtracking verifies the smooth full objective. The default history size is 10. Only L2/no penalty and uniform `sample_weight` are accepted.

## 8. L-BFGS-B

**Source:** `statgpu/solvers/_lbfgs_b.py`

The bound-constrained limited-memory method follows Byrd et al. (1995). L-BFGS-B adds componentwise constraints

$$
\ell_j\le\beta_j\le u_j
$$

and projects trial points:

$$
\beta_{k+1}
=
\Pi_{[\ell,u]}
\left(\beta_k+a_kd_k\right).
$$

Convergence uses the projected gradient: a component pointing outside an active bound is set to zero. Bounds must match the coefficient shape, contain no NaN, and satisfy $\ell_j\le u_j$. Like L-BFGS, this path accepts only smooth penalties and uniform sample weights.

## 9. ADMM

**Source:** `statgpu/solvers/_admm.py`

The splitting formulation and residual diagnostics follow Boyd et al. (2011). ADMM introduces $z$ and solves

$$
\min_{\beta,z}
f(\beta)+P(z)
\quad\text{subject to}\quad
\beta=z.
$$

With scaled dual variable $u$,

$$
\beta^{k+1}
=
\arg\min_\beta
\left[
f(\beta)
+
\frac{\rho}{2}
\lVert\beta-z^k+u^k\rVert_2^2
\right],
$$

$$
z^{k+1}
=
\operatorname{prox}_{P/\rho}
\left(\beta^{k+1}+u^k\right),
\qquad
u^{k+1}
=
u^k+\beta^{k+1}-z^{k+1}.
$$

Small constant-Hessian systems use a cached Cholesky solve. Other loss subproblems use Nesterov-accelerated gradient iterations; `cg_max_iter` is retained as the historical parameter name. The primal and dual residuals are

$$
r_{\mathrm p}=\lVert\beta-z\rVert_2,
\qquad
r_{\mathrm d}=\rho\lVert z^k-z^{k-1}\rVert_2.
$$

When enabled, adaptive $\rho$ doubles or halves it if one residual exceeds ten times the other. Only uniform sample weights are currently accepted.

## 10. Exact Ridge solve

**Source:** estimator methods `_solve_exact_numpy`, `_solve_exact_cupy`, and `_solve_exact_torch` in `statgpu/linear_model/penalized/_fit_mixin.py`

This closed-form Ridge estimator follows Hoerl and Kennard (1970). For centered/weighted $X_c$ and $y_c$, the estimator solves

$$
\hat\beta
=
\left(
X_c^\top W X_c+n_{\mathrm{eff}}\alpha I
\right)^{-1}
X_c^\top W y_c.
$$

This matches the average squared-error plus $\alpha\lVert\beta\rVert_2^2/2$ convention. The intercept is reconstructed from weighted means and is not penalized. NumPy uses a direct solve with pseudoinverse fallback; CuPy prefers Cholesky and Torch uses `torch.linalg.solve`. Automatic dispatch prefers exact Ridge on CPU but uses Newton for the corresponding GPU case because small/medium GPU factorizations can cost more than they save.

## Dispatch versus direct availability

The main automatic policy is:

| Objective | Typical automatic path |
|---|---|
| ordinary `GeneralizedLinearModel` | IRLS |
| squared error + L2 on CPU | exact |
| squared error + L2 on GPU | Newton |
| scalar or group SCAD/MCP | FISTA-LLA |
| squared error + L1/Elastic Net | FISTA |
| sparse GLMs | FISTA or FISTA-BB according to family, backend, CV mode, and size |
| smooth GLM/robust/Cox objective | Newton, with CV-specific L-BFGS exceptions |

`proximal_newton_solver` and `lbfgs_b_solver` are directly importable but are not current `auto` destinations. Estimator-specific algorithms, ordered-model trust-region Newton, fused GPU kernels, and group LLA wrappers are documented with their parent model/path rather than counted as additional solver keywords.

See the [solver × penalty compatibility matrix](solver-penalty-matrix.md) for exact accepted and rejected combinations, and the [loss × penalty × solver framework](loss-penalty-solver-framework.md) for architecture.

## Common controls and failure signals

| Control | Meaning |
|---|---|
| `max_iter` | Maximum outer iterations; continuation methods may also have inner iteration limits |
| `tol` | Gradient, parameter-step, or residual threshold depending on the algorithm |
| `history_size` | Number of $(s_k,y_k)$ pairs retained by L-BFGS/L-BFGS-B |
| `lipschitz_L` | Optional caller-provided smooth-gradient Lipschitz constant for FISTA paths |
| `rho`, `adaptive_rho` | ADMM augmented-Lagrangian scale and residual balancing |
| `alpha_path`, `max_lla_per_step`, `lla_tol` | Continuation and LLA controls for non-convex penalties |

Reaching `max_iter` produces a convergence warning on supported paths. A solver returning coefficients does not by itself prove that the statistical model is appropriate; inspect convergence metadata, objective/KKT diagnostics, and predictive validation.

## References

- Barzilai, J., & Borwein, J. M. (1988). [Two-point step size gradient methods](https://doi.org/10.1093/imanum/8.1.141). *IMA Journal of Numerical Analysis*, 8(1), 141–148.
- Beck, A., & Teboulle, M. (2009). [A fast iterative shrinkage-thresholding algorithm for linear inverse problems](https://doi.org/10.1137/080716542). *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). [Distributed optimization and statistical learning via the alternating direction method of multipliers](https://doi.org/10.1561/2200000016). *Foundations and Trends in Machine Learning*, 3(1), 1–122.
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). [A limited memory algorithm for bound constrained optimization](https://doi.org/10.1137/0916069). *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.
- Fan, J., & Li, R. (2001). [Variable selection via nonconcave penalized likelihood and its oracle properties](https://doi.org/10.1198/016214501753382273). *Journal of the American Statistical Association*, 96(456), 1348–1360.
- Green, P. J. (1984). [Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives](https://doi.org/10.1111/j.2517-6161.1984.tb01288.x). *Journal of the Royal Statistical Society: Series B*, 46(2), 149–192.
- Hoerl, A. E., & Kennard, R. W. (1970). [Ridge regression: Biased estimation for nonorthogonal problems](https://doi.org/10.1080/00401706.1970.10488634). *Technometrics*, 12(1), 55–67.
- Lee, J. D., Sun, Y., & Saunders, M. A. (2014). [Proximal Newton-type methods for minimizing composite functions](https://doi.org/10.1137/130921428). *SIAM Journal on Optimization*, 24(3), 1420–1443.
- Liu, D. C., & Nocedal, J. (1989). [On the limited memory BFGS method for large scale optimization](https://doi.org/10.1007/BF01589116). *Mathematical Programming*, 45, 503–528.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Nocedal, J., & Wright, S. J. (2006). [*Numerical Optimization* (2nd ed.)](https://doi.org/10.1007/978-0-387-40065-5). Springer.
- O'Donoghue, B., & Candès, E. (2015). [Adaptive restart for accelerated gradient schemes](https://doi.org/10.1007/s10208-013-9150-3). *Foundations of Computational Mathematics*, 15(3), 715–732.
- Zou, H., & Li, R. (2008). [One-step sparse estimates in nonconcave penalized likelihood models](https://doi.org/10.1214/07-AOS520). *Annals of Statistics*, 36(4), 1509–1533.
