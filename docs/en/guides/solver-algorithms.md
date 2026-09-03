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

This page inventories **12 core solve paths** found in the current source. Ten are low-level functions exported by `statgpu.solvers`, IRLS is exported as `statgpu.glm_core.IRLSSolver`, and `exact` is an estimator-level closed-form path. Fused GPU kernels and group-aware LLA are implementations of these paths, not extra public solver names.

::: warning API boundary
The inventory is broader than any one estimator's accepted `solver=` values. Use `solver="auto"` unless you have checked the [solver × penalty compatibility matrix](solver-penalty-matrix.md). Direct solver functions expect loss and penalty objects and are mainly advanced interfaces.
:::

## Complete inventory

| # | Path | Public status | Main use | Backends |
|---:|---|---|---|---|
| 1 | FISTA | `fista_solver`; estimator-routed | Convex non-smooth penalties and general composite objectives | NumPy, CuPy, Torch |
| 2 | FISTA-BB | `fista_bb_solver`; estimator-routed | Sparse GLMs with adaptive spectral steps | NumPy, CuPy, Torch |
| 3 | FISTA-LLA | `fista_lla_path`; continuation subpath | SCAD, MCP, Group SCAD, and Group MCP | NumPy, CuPy, Torch |
| 4 | Newton-Raphson | `newton_solver`; estimator-routed | Smooth losses with no penalty or L2 | NumPy, CuPy, Torch |
| 5 | Proximal-Newton facade | `proximal_newton_solver`; direct low-level API | Smooth Newton path; delegates non-smooth cases to FISTA | NumPy, CuPy, Torch |
| 6 | GLM IRLS | `IRLSSolver`; estimator-routed | Ordinary GLMs and supported smooth penalized GLMs | NumPy, CuPy, Torch |
| 7 | Proximal IRLS for quantiles | `proximal_irls_quantile_solver`; internal routed subpath | Quantile loss with SCAD/MCP continuation | NumPy, CuPy, Torch |
| 8 | Quantile coordinate descent | `quantile_cd_solver`; direct low-level API | Weighted-L1 LLA subproblems for quantile loss | NumPy |
| 9 | L-BFGS | `lbfgs_solver`; estimator-routed | Smooth objectives when storing a Hessian is undesirable | NumPy, CuPy, Torch |
| 10 | L-BFGS-B | `lbfgs_b_solver`; direct low-level API | Smooth objectives with box constraints | NumPy, CuPy, Torch |
| 11 | ADMM | `admm_solver`; explicit estimator route | Split smooth-loss and proximal-penalty updates | NumPy, CuPy, Torch |
| 12 | Exact Ridge solve | estimator `solver="exact"` | Squared error with L2 | NumPy, CuPy, Torch |

`quantile_cd_solver` was missing from the previous page. L-BFGS-B, ADMM, and exact were also missing from the Chinese page.

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

FISTA combines a proximal-gradient step with Nesterov momentum. With a local Lipschitz estimate $L_k$,

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

FISTA-BB estimates a local step from successive parameters and smooth gradients:

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

Local linear approximation replaces a non-convex coordinate penalty near $\beta^{(m)}$ by a weighted L1 surrogate:

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

For a smooth objective, define

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

A full proximal-Newton method for non-smooth $P$ would solve

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

At the current mean $\mu_i$ and linear predictor $\eta_i=g(\mu_i)$, IRLS builds

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

## 7. Proximal IRLS for quantile regression

**Source:** `statgpu/solvers/_proximal_irls_quantile.py`

For residual $r_i=y_i-x_i^\top\beta$ and quantile $\tau$, the check loss is

$$
\rho_\tau(r)
=
r\left(\tau-\mathbf 1\{r<0\}\right).
$$

Inside each continuation and LLA step, the implementation uses

$$
a_i
=
\frac{
\tau\mathbf 1\{r_i\ge0\}
+
(1-\tau)\mathbf 1\{r_i<0\}
}{
\max(|r_i|,\varepsilon)
},
\qquad
\omega_j=P_\lambda'(|\beta_j|).
$$

With $A=\operatorname{diag}(a_i)$,

$$
g=\tilde X^\top A(y-\tilde X\beta),
\qquad
h=\operatorname{diag}(\tilde X^\top A\tilde X),
$$

and all coordinates are updated in parallel by

$$
\beta_j^{\mathrm{new}}
=
\frac{
\mathcal S_{n\omega_j}(g_j+h_j\beta_j)
}{h_j}.
$$

This is a GPU-friendly Jacobi-style diagonal majorization step, not cyclic coordinate descent. Analytic sample weights multiply $a_i$ after being normalized to sum to $n$.

## 8. Quantile coordinate descent

**Source:** `statgpu/solvers/_quantile_cd.py`

This NumPy-only low-level solver alternates LLA weights with cyclic coordinate updates for

$$
\min_\beta
\sum_{i=1}^n\rho_\tau(y_i-x_i^\top\beta)
+
\sum_{j=1}^p\omega_j|\beta_j|.
$$

Using $\psi_\tau(r)=\tau$ for $r\ge0$ and $\psi_\tau(r)=-(1-\tau)$ otherwise, the implemented coordinate step is

$$
\beta_j
\leftarrow
\frac{
\mathcal S_{\omega_j}
\left(
\sum_i x_{ij}\psi_\tau(r_i^{(-j)})
\right)
}{
\sum_i x_{ij}^2
}.
$$

It is exported for direct advanced use but is not selected by the current automatic estimator dispatch. Its `sample_weight` argument is present in the signature but is not consumed by the current implementation; use the routed FISTA/proximal-IRLS paths when observation weights matter.

## 9. L-BFGS

**Source:** `statgpu/solvers/_lbfgs.py`

L-BFGS approximates $H_k^{-1}g_k$ from the most recent

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k,
\qquad
\rho_k=(y_k^\top s_k)^{-1}
$$

pairs. The standard two-loop recursion produces $d_k=-B_k g_k$ without forming a dense Hessian. Armijo backtracking verifies the smooth full objective. The default history size is 10. Only L2/no penalty and uniform `sample_weight` are accepted.

## 10. L-BFGS-B

**Source:** `statgpu/solvers/_lbfgs_b.py`

L-BFGS-B adds componentwise constraints

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

## 11. ADMM

**Source:** `statgpu/solvers/_admm.py`

ADMM introduces $z$ and solves

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

## 12. Exact Ridge solve

**Source:** estimator methods `_solve_exact_numpy`, `_solve_exact_cupy`, and `_solve_exact_torch` in `statgpu/linear_model/penalized/_fit_mixin.py`

For centered/weighted $X_c$ and $y_c$, the estimator solves

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
| scalar or group SCAD/MCP | FISTA-LLA; quantile loss uses proximal IRLS |
| quantile loss with convex penalty | FISTA |
| squared error + L1/Elastic Net | FISTA |
| sparse GLMs | FISTA or FISTA-BB according to family, backend, CV mode, and size |
| smooth GLM/robust/Cox objective | Newton, with CV-specific L-BFGS exceptions |

`proximal_newton_solver`, `quantile_cd_solver`, and `lbfgs_b_solver` are directly importable but are not current `auto` destinations. Estimator-specific ordered-model trust-region Newton, fused GPU kernels, and group LLA wrappers are documented with their parent model/path rather than counted as additional solver keywords.

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

- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- Barzilai, J., & Borwein, J. M. (1988). Two-point step size gradient methods. *IMA Journal of Numerical Analysis*, 8(1), 141–148.
- O'Donoghue, B., & Candès, E. (2015). Adaptive restart for accelerated gradient schemes. *Foundations of Computational Mathematics*, 15(3), 715–732.
- Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage. *Mathematics of Computation*, 35(151), 773–782.
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via ADMM. *Foundations and Trends in Machine Learning*, 3(1), 1–122.
- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood. *Journal of the American Statistical Association*, 96, 1348–1360.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509–1533.
