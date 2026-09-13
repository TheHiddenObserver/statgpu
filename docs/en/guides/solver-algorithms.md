# Solver Algorithms

> Language: English  
> Last updated: 2026-09-13  
> This page: Algorithm reference  
> Switch: [Chinese](../../cn/guides/solver-algorithms.md)

## Overview

statgpu provides first-order, second-order, proximal, and closed-form solvers. Most model users should start with `solver="auto"`; this page is the algorithm-level reference and therefore keeps the mathematical update rules, convergence criteria, backend behavior, and important capability boundaries explicit.

Three rules are useful when reading this page:

1. Backend support does not imply that every loss/penalty/weight combination is supported.
2. `sample_weight` does not change an explicitly requested solver. If the requested weighted combination is unsupported, fitting raises an error.
3. Weight support is route-specific. In particular, non-uniform direct L-BFGS weights are supported by maintained GLM losses, not automatically by every `LossBase` implementation.

For model-level dispatch, see [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md).

## Solver summary

| Solver | Best for | Backend support |
|--------|----------|:---:|
| Proximal IRLS-CD | quantile + SCAD/MCP | NumPy, CuPy, Torch |
| Proximal Newton | smooth loss + L2/no penalty; non-smooth requests use FISTA | NumPy, CuPy, Torch |
| FISTA | general non-smooth penalties | NumPy, CuPy, Torch |
| FISTA-BB | GLM + sparse penalties | NumPy, CuPy, Torch |
| FISTA-LLA | non-convex penalties via continuation/LLA | NumPy, CuPy, Torch |
| IRLS | losses with a maintained IRLS representation | NumPy, CuPy, Torch |
| Newton | smooth losses with Hessian support | NumPy, CuPy, Torch |
| L-BFGS | smooth losses, moderate dimensions | NumPy, CuPy, Torch |
| L-BFGS-B | box-constrained smooth problems | NumPy, CuPy, Torch |
| ADMM | separable/proximal formulations | NumPy, CuPy, Torch |
| `exact` | squared error + L2 closed-form path | NumPy, CuPy, Torch |

The backend column describes numerical implementation capability only. Estimator and loss contracts can further narrow the valid combinations.

---

## 1. Proximal IRLS-CD

**File**: `statgpu/solvers/_proximal_irls_quantile.py`

**Use case**: Quantile regression with SCAD/MCP penalties. It combines an IRLS quadratic majorization of the pinball loss with local linear approximation (LLA) for the non-convex penalty.

### Algorithm

For each continuation value of `alpha`:

1. **LLA outer loop.** Compute the local penalty derivative

   $$
   d_j=P'(|\beta_j|),
   $$

   with threshold

   $$
   t_j=n\,d_j.
   $$

2. **IRLS-CD inner loop.** With residuals

   $$
   r_i=y_i-x_i^\top\beta,
   $$

   define

   $$
   q_i=\begin{cases}
   \tau, & r_i\ge 0,\\
   1-\tau, & r_i<0,
   \end{cases}
   \qquad
   w_i^{\mathrm{IRLS}}=\frac{q_i}{\max(|r_i|,\varepsilon)}.
   $$

   If analytic `sample_weight=s` is supplied, statgpu first normalizes it to

   $$
   \tilde s_i=\frac{n s_i}{\sum_j s_j},
   $$

   then uses

   $$
   w_i=\tilde s_i\,w_i^{\mathrm{IRLS}}.
   $$

3. **Parallel diagonal majorization.** Let $W=\operatorname{diag}(w)$. The implementation computes

   $$
   g=X^\top W(y-X\beta),
   \qquad
   h=\operatorname{diag}(X^\top W X),
   $$

   then

   $$
   u=g+h\odot\beta,
   $$

   and updates all coordinates in parallel:

   $$
   \beta_j^{\mathrm{new}}=\frac{S(u_j,t_j)}{h_j},
   $$

   where

   $$
   S(u,t)=\operatorname{sign}(u)\max(|u|-t,0).
   $$

   This is a Jacobi-style parallel diagonal-majorization update rather than a cyclic coordinate-descent sweep.

4. **Convergence.** The IRLS inner loop checks

   $$
   \|\beta^{\mathrm{new}}-\beta\|_\infty<\texttt{tol},
   $$

   and the LLA outer loop checks

   $$
   \|\beta-\beta_{\mathrm{before\,LLA}}\|_\infty<\texttt{lla\_tol}.
   $$

### Continuation and defaults

- continuation path: `lambda_max` to target `alpha`;
- `max_lla_per_step=2`;
- `lla_tol=1e-6`;
- `tol=1e-6`;
- GPU convergence checks remain on device except for the final boolean synchronization.

---

## 2. Proximal Newton

**File**: `statgpu/solvers/_proximal_newton.py`

**Use case**: Smooth losses with L2/no penalty, where an ordinary Newton system is well defined.

For a genuinely non-smooth composite objective

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

a true Proximal Newton step should solve a Hessian-metric proximal subproblem such as

$$
\Delta_k
=\arg\min_{\Delta}
\left\{
\nabla\ell(\beta_k)^\top\Delta
+\frac12\Delta^\top H_k\Delta
+P(\beta_k+\Delta)
\right\}.
$$

Applying an ordinary Euclidean prox to a Newton step would optimize a different composite objective. The maintained implementation therefore executes Newton only on L2/no-penalty smooth routes; non-smooth requests warn and delegate to FISTA until a correct Hessian-metric proximal subproblem is implemented.

### Algorithm

At the current iterate $\beta_k$, let

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

where $P$ here is only L2 or zero. Compute

$$
g_k=\nabla\ell(\beta_k)+\nabla P(\beta_k),
$$

and

$$
H_k=\nabla^2\ell(\beta_k)+\nabla^2P(\beta_k).
$$

The implementation symmetrizes the Hessian and adds a small ridge stabilization:

$$
\widetilde H_k
=\frac12(H_k+H_k^\top)+10^{-10}I.
$$

If

$$
\|g_k\|_2\le\texttt{tol},
$$

optimization stops. Otherwise solve

$$
\widetilde H_k d_k=g_k,
$$

and form trial points with the code's subtract-direction convention,

$$
\beta_k(t)=\beta_k-t d_k.
$$

If the linear solve is recognized as singular/ill-conditioned, this solver does **not** call least squares. It falls back directly to

$$
d_k=g_k,
$$

which is steepest descent under the subtract-direction convention.

A valid descent direction must satisfy

$$
g_k^\top d_k>0.
$$

If that inner product is non-finite or non-positive, the solver again uses

$$
d_k=g_k,
\qquad
g_k^\top d_k=\|g_k\|_2^2.
$$

### Armijo backtracking

Starting from $t=1$, accept the first step satisfying

$$
F(\beta_k-t d_k)
\le
F(\beta_k)-10^{-4}t\,g_k^\top d_k.
$$

If the condition fails,

$$
t\leftarrow\frac t2.
$$

The implementation tries at most 25 backtracking steps. A successful trial gives

$$
\beta_{k+1}=\beta_k-t d_k.
$$

If all 25 trials fail, the solver keeps $\beta_{k+1}=\beta_k$, emits a line-search warning, and stops.

### Defaults and backend

- default `max_iter=50`;
- default `tol=1e-6`;
- supported NumPy/CuPy/Torch routes use the corresponding native linear algebra.

---

## 3. FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

**File**: `statgpu/solvers/_fista.py`

**Use case**: Composite objectives

$$
F(\beta)=f(\beta)+P(\beta),
$$

with smooth $f$ and a penalty $P$ that has a proximal operator.

### Proximal-gradient update

Initialize

$$
\beta_0=y_0,\qquad t_0=1.
$$

At momentum point $y_k$, compute

$$
g_k=\nabla f(y_k).
$$

For current Lipschitz constant $L_k$, use

$$
\gamma_k=\frac1{L_k},
$$

and take the proximal step

$$
\beta_{k+1}
=\operatorname{prox}_{\gamma_kP}
\left(y_k-\gamma_k g_k\right),
$$

where

$$
\operatorname{prox}_{\gamma P}(v)
=\arg\min_x\left\{\gamma P(x)+\frac12\|x-v\|_2^2\right\}.
$$

### Quadratic-majorization backtracking

On routes that backtrack, let

$$
\Delta_k=\beta_{k+1}-y_k.
$$

A trial step must satisfy the smooth-part quadratic upper bound

$$
f(\beta_{k+1})
\le
f(y_k)+g_k^\top\Delta_k
+\frac{L_k}{2}\|\Delta_k\|_2^2+\varepsilon_{\rm slack}.
$$

If not,

$$
L_k\leftarrow1.5L_k,
\qquad
\gamma_k\leftarrow\frac1{L_k},
$$

and the proximal step is recomputed, for at most 20 backtracking attempts. Supported asynchronous GPU non-smooth routes use a conservative fixed $L_k$ instead of synchronizing for every backtracking trial; the proximal update itself is unchanged.

### Nesterov momentum

After accepting $\beta_{k+1}$,

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}2,
$$

$$
y_{k+1}=\beta_{k+1}
+\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k).
$$

A typical coefficient-change criterion is

$$
\|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
$$

Some adaptive-penalty routes also use objective stability so that small coefficient oscillations at essentially unchanged objective value are not misclassified.

### Weighted path

On maintained weighted routes, if the per-observation score is $\psi_i$,

$$
g(\beta)
=\frac{X^\top(s\odot\psi)}{\sum_i s_i}.
$$

Weighted objective tracking and the weighted Lipschitz estimate use the same analytic-weight convention. A weighted FISTA implementation does not by itself imply that every model/loss combination supports weights.

### Defaults

- default `max_iter=500`;
- default `tol=1e-6`.

---

## 4. FISTA-BB (Barzilai-Borwein)

**File**: `statgpu/solvers/_fista_bb.py`

**Use case**: FISTA with local Barzilai-Borwein curvature estimates for step-size selection on supported sparse-penalty GLM routes.

### Lipschitz burn-in and BB curvature

Define the baseline step

$$
\gamma_L=\frac1L.
$$

During burn-in,

$$
\gamma_k=\gamma_L.
$$

After burn-in, define

$$
s_{k-1}=\beta_k-\beta_{k-1},
$$

$$
q_{k-1}=\nabla f(\beta_k)-\nabla f(\beta_{k-1}).
$$

When

$$
s_{k-1}^\top q_{k-1}>0
$$

and the curvature information is numerically valid, the implementation alternates

$$
\gamma_k^{\mathrm{BB1}}
=\frac{s_{k-1}^\top s_{k-1}}
{s_{k-1}^\top q_{k-1}},
$$

and

$$
\gamma_k^{\mathrm{BB2}}
=\frac{s_{k-1}^\top q_{k-1}}
{q_{k-1}^\top q_{k-1}}.
$$

The default bounds are

$$
\gamma_{\min}=10^{-3}\gamma_L,
\qquad
\gamma_{\max}=10^3\gamma_L,
$$

so the selected BB step is clipped as

$$
\gamma_k
\leftarrow
\min\{\gamma_{\max},\max(\gamma_k,\gamma_{\min})\}.
$$

If the curvature pair is invalid, the solver keeps an already-safe step instead of forcing a BB ratio.

### Proximal update and safeguard

At momentum point $y_k$,

$$
g_k=\nabla f(y_k),
$$

then

$$
v_k=y_k-\gamma_k g_k,
$$

$$
\beta_{k+1}=\operatorname{prox}_{\gamma_kP}(v_k).
$$

For non-quadratic GLMs the maintained implementation periodically checks objective and coefficient-norm safeguards. If the trial is considered too aggressive, it applies

$$
\gamma_k\leftarrow\frac{\gamma_k}{2}
$$

and recomputes the same proximal step, up to 15 safeguard retries.

### Nesterov momentum and adaptive restart

The standard momentum update is

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}2,
$$

$$
y_{k+1}=\beta_{k+1}
+\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k).
$$

The restart condition is

$$
\left(y_{k+1}-\beta_{k+1}\right)^\top
\left(\beta_{k+1}-\beta_k\right)>0.
$$

When it holds, momentum is reset:

$$
t_{k+1}=1,
\qquad
y_{k+1}=\beta_{k+1}.
$$

Convergence uses

$$
\|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
$$

Quadratic losses do not gain useful adaptation from the BB curvature estimate and therefore retain fixed-Lipschitz FISTA behavior. BB updates are also disabled for SCAD, MCP, and their group variants because non-convex reweighting can change secant curvature abruptly.

---

## 5. FISTA-LLA

**File**: `statgpu/solvers/_fista_lla.py`

**Use case**: SCAD, MCP, and other routes that can be represented by a locally weighted convex penalty.

### Continuation path

Let

$$
\alpha^{(0)}>\alpha^{(1)}>\cdots>\alpha^{(M)}=\alpha_{\rm target}.
$$

At each $\alpha^{(m)}$, run an LLA outer loop. Let $\beta^{(r)}$ be the current LLA iterate.

### LLA weights

For scalar non-convex penalties,

$$
d_j^{(r)}
=P_{\alpha^{(m)}}'\!\left(|\beta_j^{(r)}|\right).
$$

For SCAD,

$$
d_j^{(r)}=
\begin{cases}
\alpha, & |\beta_j^{(r)}|\le\alpha,\\[3pt]
\dfrac{a\alpha-|\beta_j^{(r)}|}{a-1},
& \alpha<|\beta_j^{(r)}|\le a\alpha,\\[8pt]
0, & |\beta_j^{(r)}|>a\alpha,
\end{cases}
$$

while MCP uses

$$
d_j^{(r)}=
\begin{cases}
\alpha-\dfrac{|\beta_j^{(r)}|}{\gamma},
& |\beta_j^{(r)}|\le\gamma\alpha,\\[8pt]
0, & |\beta_j^{(r)}|>\gamma\alpha.
\end{cases}
$$

When an intercept is represented by an augmented GLM column, its LLA weight is fixed at zero, so it is not penalized.

### Convex surrogate

Dropping constants independent of $\beta$, the $r$-th LLA subproblem is

$$
Q_r(\beta)
=f(\beta)+\sum_j d_j^{(r)}|\beta_j|.
$$

Thus the default inner problem is weighted L1. With a group-LLA factory, the corresponding surrogate is

$$
Q_r(\beta)
=f(\beta)+\sum_g D_g^{(r)}\|\beta_g\|_2,
$$

and the inner solve uses the matching weighted Group Lasso proximal operator.

### FISTA inner solve

For fixed LLA weights $d^{(r)}$, let

$$
\gamma_k=\frac1{L_k},
\qquad
g_k=\nabla f(y_k).
$$

Compute

$$
v_k=y_k-\gamma_k g_k.
$$

For the default weighted-L1 surrogate,

$$
\beta_{k+1,j}
=S\!\left(v_{k,j},\gamma_k d_j^{(r)}\right),
$$

where

$$
S(v,t)=\operatorname{sign}(v)\max(|v|-t,0).
$$

Then update Nesterov momentum:

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}2,
$$

$$
y_{k+1}
=\beta_{k+1}
+\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k).
$$

The typical inner stopping rule is

$$
\|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
$$

For non-quadratic losses the implementation periodically recomputes the Lipschitz estimate. If the new estimate differs from the current one by roughly more than a factor of 1.5, it updates

$$
\gamma_k=\frac1{L_k}
$$

before continuing. The unweighted squared-error GPU fast path uses

$$
g_k=\frac{X^\top Xy_k-X^\top y}{n}
$$

to avoid redundant matrix products.

### LLA outer-loop convergence

After the inner solve produces $\beta^{(r+1)}$, stop the current LLA problem when

$$
\|\beta^{(r+1)}-\beta^{(r)}\|_1
<\texttt{lla\_tol}.
$$

Otherwise recompute $d^{(r+1)}$ and solve the next weighted convex surrogate. The solution at the current continuation value seeds the next $\alpha$ value.

The generic composite route uses FISTA by default. A Proximal Newton inner solve is used only if the loss explicitly advertises a correct Hessian-metric proximal subproblem. Cox currently remains on FISTA-LLA.

---

## 6. IRLS (Iteratively Reweighted Least Squares)

**Implementation**: `statgpu/glm_core/_irls.py`, plus loss-specific `irls()` methods for selected non-GLM losses.

### Quantile IRLS

The current `QuantileLoss.irls()` implementation uses iteratively reweighted least squares (IRLS). Starting from an initial coefficient vector (OLS when no explicit initialization is supplied), each iteration computes

$$
r_i=y_i-x_i^\top\beta
$$

and

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf 1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

If analytic `sample_weight=s` is supplied, it is normalized to sum to $n$,

$$
\tilde s_i=\frac{n s_i}{\sum_j s_j},
$$

and the effective weight is

$$
w_i=\tilde s_i w_i^{\mathrm{IRLS}}.
$$

With $W=\operatorname{diag}(w)$, the unpenalized update solves

$$
(X^\top W X+\varepsilon I)\beta_{\mathrm{new}}=X^\top W y.
$$

For an L2 penalty, the maintained path adds the corresponding diagonal ridge term. When `fit_intercept=True`, the intercept coordinate is excluded from the penalty. Convergence is checked with

$$
\|\beta_{\mathrm{new}}-\beta\|_2<\texttt{tol}.
$$

### GLM IRLS

Let the link satisfy

$$
\eta=g(\mu),
\qquad
\mu=g^{-1}(\eta),
$$

with variance function $V(\mu)$. At iteration $k$,

$$
\eta_i^{(k)}=x_i^\top\beta_k,
\qquad
\mu_i^{(k)}=g^{-1}(\eta_i^{(k)}).
$$

The Fisher working weight is

$$
w_i^{\rm work}
=\frac1{V(\mu_i^{(k)})[g'(\mu_i^{(k)})]^2},
$$

and the working response is

$$
z_i^{(k)}
=\eta_i^{(k)}+
\left(y_i-\mu_i^{(k)}\right)g'(\mu_i^{(k)}).
$$

If analytic weights $s_i$ are supplied, the WLS weight becomes

$$
w_i^{(k)}=s_i\,w_i^{\rm work}.
$$

Thus analytic `sample_weight` and IRLS working weights are distinct objects: the former comes from the statistical objective, while the latter comes from the local GLM quadratic approximation.

Let

$$
W_k=\operatorname{diag}(w_1^{(k)},\ldots,w_n^{(k)}).
$$

With diagonal L2 penalty matrix $R$ and optional quadratic penalty matrix $\Omega$, the WLS candidate solves

$$
\left(X^\top W_kX+R+\Omega\right)\widetilde\beta_{k+1}
=X^\top W_k z^{(k)}.
$$

The linear solve falls back to least squares only for a genuine rank failure.

### IRLS objective backtracking

Define

$$
\Delta_k=\widetilde\beta_{k+1}-\beta_k.
$$

Starting from $t=1$, try

$$
\beta_k(t)=\beta_k+t\Delta_k.
$$

The maintained implementation requires the registered GLM objective not to increase beyond a numerical tolerance:

$$
F(\beta_k(t))
\le
F(\beta_k)+\varepsilon_F,
$$

with

$$
\varepsilon_F
=\max\left(10^{-10}|F(\beta_k)|,10^{-6}\right).
$$

If the condition fails,

$$
t\leftarrow\frac t2,
$$

for at most 30 backtracking trials. If no candidate is accepted, the old coefficients are retained and line-search failure is reported.

### GLM IRLS convergence

Define the per-observation score in the linear-predictor coordinate as

$$
u_i
=\frac{\mu_i-y_i}
{V(\mu_i)g'(\mu_i)}.
$$

With analytic weights use $s_i u_i$ and

$$
n_{\rm eff}=\sum_i s_i;
$$

without weights, $n_{\rm eff}=n$. The normalized data-fit score is

$$
g_f=\frac{X^\top u}{n_{\rm eff}},
$$

plus the corresponding L2/quadratic-penalty gradient. The maintained convergence criterion is

$$
\|g_f\|_2<\texttt{tol},
$$

rather than relying only on a potentially tiny parameter change caused by a truncated line search.

---

## 7. Newton-Raphson

**File**: `statgpu/solvers/_newton.py`

**Use case**: Smooth losses with L2/no penalty and Hessian support.

### Newton system

Let

$$
F(\beta)=\ell(\beta)+P(\beta).
$$

At iteration $k$ compute

$$
g_k=\nabla F(\beta_k),
\qquad
H_k=\nabla^2F(\beta_k).
$$

The implementation stabilizes the system with

$$
\widetilde H_k
=\frac12(H_k+H_k^\top)+10^{-10}I
$$

and solves

$$
\widetilde H_k d_k=g_k.
$$

Trial points use

$$
\beta_k(t)=\beta_k-t d_k.
$$

Unlike the Proximal Newton implementation, an ordinary Newton linear solve that encounters a genuine rank failure falls back to the least-squares solution

$$
d_k=\widetilde H_k^{+}g_k,
$$

where $\widetilde H_k^{+}$ denotes the generalized-inverse solution produced by `lstsq`.

If

$$
g_k^\top d_k\le0
$$

or the inner product is non-finite, the direction is replaced by steepest descent,

$$
d_k=g_k.
$$

The gradient stopping rule is

$$
\|g_k\|_2\le\texttt{tol}.
$$

For losses declaring a constant Hessian, that Hessian is computed once and reused.

### Armijo backtracking

Starting at $t=1$, accept the first candidate satisfying

$$
F(\beta_k-t d_k)
\le
F(\beta_k)-10^{-4}t\,g_k^\top d_k.
$$

If it fails,

$$
t\leftarrow\frac t2,
$$

for at most 20 trials. If no candidate satisfies Armijo, the solver accepts no unverified tiny step; it retains $\beta_k$ and reports line-search failure.

### Analytic `sample_weight`

For routes exposing weighted curvature, Newton uses the same normalized weighted objective for value, gradient, Hessian, and every Armijo trial:

$$
F(\beta)
=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

Multiplying all active weights by one positive constant therefore leaves the optimum unchanged. Uniform/effectively-uniform weights retain the historical unweighted numerical path where that compatibility route is defined.

---

## 8. L-BFGS / L-BFGS-B

**Files**: `statgpu/solvers/_lbfgs.py`, `statgpu/solvers/_lbfgs_b.py`

**Use case**: Limited-memory quasi-Newton optimization for smooth objectives, plus a projected box-constrained variant.

### L-BFGS curvature history

Let

$$
g_k=\nabla F(\beta_k).
$$

After accepting the next iterate define

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k.
$$

A curvature pair is stored only when

$$
y_k^\top s_k>10^{-12},
$$

with

$$
\rho_k=\frac1{y_k^\top s_k}.
$$

The default history size is

$$
m=10.
$$

Older pairs are discarded once the history exceeds $m$.

### Two-loop recursion

Start from

$$
q=g_k.
$$

Traverse history from newest to oldest:

$$
\alpha_i=\rho_i s_i^\top q,
\qquad
q\leftarrow q-\alpha_i y_i.
$$

If history is non-empty, scale the initial inverse-Hessian approximation by

$$
\gamma_k
=\frac{s_{k-1}^\top y_{k-1}}
{y_{k-1}^\top y_{k-1}},
$$

otherwise take $\gamma_k=1$. Set

$$
r=\gamma_k q.
$$

Traverse history from oldest to newest:

$$
\beta_i^{\rm loop}=\rho_i y_i^\top r,
$$

$$
r\leftarrow r+s_i
\left(\alpha_i-\beta_i^{\rm loop}\right).
$$

The search direction is

$$
p_k=-r.
$$

If

$$
g_k^\top p_k\ge0,
$$

fall back to

$$
p_k=-g_k.
$$

### L-BFGS Armijo line search

Starting with $t=1$, accept the first trial satisfying

$$
F(\beta_k+t p_k)
\le
F(\beta_k)+10^{-4}t\,g_k^\top p_k.
$$

If it fails,

$$
t\leftarrow\frac t2,
$$

for at most 25 backtracking trials.

After acceptance, recompute $g_{k+1}$ and update the curvature history. The solver stops when either

$$
\|g_k\|_2<\texttt{tol}
$$

or

$$
\|s_k\|_2<\texttt{tol}.
$$

### L-BFGS-B: projected box-constrained variant

The current `lbfgs_b_solver` is a projected-gradient L-BFGS-B route, not the full generalized-Cauchy-point/subspace-minimization algorithm. For box constraints

$$
\ell_j\le\beta_j\le u_j,
$$

define

$$
\Pi_{[\ell,u]}(v)_j
=\min\{u_j,\max(\ell_j,v_j)\}.
$$

At an active bound, the projected gradient is zeroed when the gradient points outside the box:

$$
\bar g_j=
\begin{cases}
0,& \beta_j\le\ell_j\ \text{and}\ g_j>0,\\
0,& \beta_j\ge u_j\ \text{and}\ g_j<0,\\
g_j,& \text{otherwise}.
\end{cases}
$$

The two-loop direction likewise has components removed when they would immediately leave the feasible box. Line-search candidates are

$$
\beta_k(t)
=\Pi_{[\ell,u]}\left(\beta_k+t p_k\right),
$$

with the same Armijo condition. Convergence uses

$$
\|\bar g_k\|_2<\texttt{tol}.
$$

`lbfgs_b_solver` currently accepts only omitted or uniform `sample_weight`; the non-uniform GLM weighting capability of ordinary `lbfgs_solver` should not be inferred for L-BFGS-B.

### Analytic `sample_weight` for L-BFGS

Non-uniform direct L-BFGS weights are loss-level opt-in:

| Direct L-BFGS route | Non-uniform `sample_weight` |
|---|---|
| Maintained `GLMLoss` | ✅ Supported |
| Generic robust / quantile / Cox `LossBase` consumers | ❌ Not implied by unweighted support |

For maintained GLMs, the initial gradient, current objective, every line-search candidate, and accepted-point gradient use the same normalized objective

$$
F(\beta)
=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

---

## 9. ADMM (Alternating Direction Method of Multipliers)

**File**: `statgpu/solvers/_admm.py`

Rewrite

$$
\min_w f(w)+P(w)
$$

as the consensus problem

$$
\min_{w,z} f(w)+P(z)
\quad\text{s.t.}\quad w=z.
$$

With scaled dual variable $u$, the augmented Lagrangian can be written

$$
\mathcal L_\rho(w,z,u)
=f(w)+P(z)
+\frac\rho2\|w-z+u\|_2^2
-\frac\rho2\|u\|_2^2.
$$

### Outer updates

At outer iteration $k$,

$$
w^{k+1}
=\arg\min_w
\left\{
f(w)+\frac\rho2\|w-z^k+u^k\|_2^2
\right\},
$$

$$
z^{k+1}
=\operatorname{prox}_{P/\rho}(w^{k+1}+u^k),
$$

$$
u^{k+1}=u^k+w^{k+1}-z^{k+1}.
$$

### $w$-subproblem: squared-error Cholesky path

When the loss has a constant Hessian and the feature dimension is within the maintained Cholesky threshold, pre-factor

$$
A=\frac{X^\top X}{n}+\rho I,
$$

then solve at each outer iteration

$$
Aw^{k+1}
=\frac{X^\top y}{n}+\rho(z^k-u^k).
$$

This path pins $\rho$, because changing it would invalidate the precomputed Cholesky factor.

### $w$-subproblem: Nesterov accelerated-gradient path

General GLMs use an inner accelerated-gradient solve. At inner momentum point $v_j$,

$$
g_j
=\nabla f(v_j)+\rho(v_j-z^k+u^k).
$$

The step size is

$$
\gamma=\frac1{L_f+\rho+10^{-8}},
$$

and

$$
w_{j+1}=v_j-\gamma g_j.
$$

Then

$$
t_{j+1}=\frac{1+\sqrt{1+4t_j^2}}2,
$$

$$
v_{j+1}
=w_{j+1}
+\frac{t_j-1}{t_{j+1}}(w_{j+1}-w_j).
$$

The inner loop may stop early when

$$
\|w_{j+1}-w_j\|_1
<\texttt{cg\_tol}\times p.
$$

Although the public/internal arguments are still named `cg_max_iter` and `cg_tol`, the current non-Cholesky fallback is Nesterov accelerated gradient, not conjugate gradient.

### Primal/dual residuals and adaptive $\rho$

Define

$$
r_{\rm p}^{k+1}=\|w^{k+1}-z^{k+1}\|_2,
$$

$$
r_{\rm d}^{k+1}=\rho\|z^{k+1}-z^k\|_2.
$$

With `adaptive_rho=True`,

$$
\rho\leftarrow
\begin{cases}
\min(2\rho,10^4),& r_{\rm p}>10r_{\rm d},\\
\max(\rho/2,10^{-4}),& r_{\rm d}>10r_{\rm p},\\
\rho,& \text{otherwise}.
\end{cases}
$$

After changing $\rho$, the inner step is recomputed as $\gamma=1/(L_f+\rho+10^{-8})$. Outer convergence requires

$$
r_{\rm p}<\texttt{tol}
\qquad\text{and}\qquad
r_{\rm d}<\texttt{tol}.
$$

The solver returns $z$, since $z$ is always the variable after applying the penalty proximal operator. The shared `admm_solver` currently accepts only omitted or uniform `sample_weight`; genuine non-uniform analytic weights are not part of this entry point's current capability.

---

## 10. `exact` (closed-form path)

**Implemented in**: `_fit_mixin._solve_exact_*`

**Use case**: Squared-error + L2 rows where the maintained dispatch selects the closed-form/eigendecomposition path, based on systems of the form

$$
\left(\frac{X^\top X}{n}+\alpha I\right)\beta
=\frac{X^\top y}{n},
$$

with estimator-specific intercept treatment applied separately.

---

## Solver dispatch

For direct model fitting, `solver="auto"` follows the maintained model-level table. A simplified view is:

```text
direct fit with solver="auto"
├── squared_error + L2 + NumPy/CPU → exact
├── squared_error + L2 + GPU       → Newton
├── squared_error + sparse penalty → FISTA/FISTA-BB
├── smooth non-Gaussian GLM + L2   → Newton
├── SCAD/MCP/adaptive path          → LLA + FISTA-family inner solve
├── quantile                        → quantile-specific FISTA/IRLS path
└── group penalty                   → group-aware FISTA / FISTA-LLA
```

`PenalizedGLM_CV` has a related but intentionally separate smooth-L2 policy. In particular, Gamma, Inverse-Gaussian, and Negative-Binomial L2 CV/final-refit routes use L-BFGS, while logistic, Poisson, and Tweedie L2 rows use Newton. Consult the compatibility matrix rather than inferring CV behavior from the direct-fit tree.

`sample_weight` does not change an explicitly requested solver. Unsupported weighted combinations raise instead of selecting a different solver.

## References

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- O'Donoghue, B. & Candes, E. (2015). Adaptive Restart for Accelerated Gradient Schemes. *Foundations of Computational Mathematics*, 15(3), 715-732.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods for Minimizing Composite Functions. *SIAM J. Optimization*, 24(3), 1420-1443.
- Liu, D. C. & Nocedal, J. (1989). On the Limited Memory BFGS Method for Large Scale Optimization. *Mathematical Programming*, 45, 503-528.
- Byrd, R. H., Lu, P., Nocedal, J. & Zhu, C. (1995). A Limited Memory Algorithm for Bound Constrained Optimization. *SIAM J. Scientific Computing*, 16(5), 1190-1208.
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in Machine Learning*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
- Zou, H. & Li, R. (2008). One-step Sparse Estimates in Nonconcave Penalized Likelihood Models. *Annals of Statistics*, 36(4), 1509-1533.