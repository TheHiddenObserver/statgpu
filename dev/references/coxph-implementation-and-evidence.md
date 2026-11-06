# CoxPH implementation details and historical validation

This internal reference connects the statistical objective to the algorithms used by
StatGPU. It describes the current source structure, not additional public API
guarantees or new benchmark runs.

For the user-facing contract, examples, and supported options, see
[CoxPH (English)](../../docs/en/models/coxph.md) and
[CoxPH（中文）](../../docs/cn/models/coxph.md).

## Implementation snapshot

### 1. Model and risk sets

For observation $i$, let $x_i\in\mathbb R^p$, event indicator
$\delta_i$, interval $(a_i,b_i]$, and stratum $s_i$. The Cox model is

$$
h_s(t\mid x_i)=h_{0s}(t)\exp(\eta_i),
\qquad
\eta_i=x_i^\top\beta.
$$

At a distinct failure time $t_{sk}$ in stratum $s$, define the event set
and risk set by

$$
D_{sk}=\{i:\delta_i=1,\ b_i=t_{sk},\ s_i=s\},
\qquad
R_{sk}=\{i:a_i<t_{sk}\le b_i,\ s_i=s\},
\qquad
d_{sk}=|D_{sk}|.
$$

Thus a row is excluded at its entry time and included at its exit time. The
partial likelihood compares each event group only with its contemporaneous
risk set; the baseline hazard is eliminated from coefficient estimation.

Write $w_i=\exp(\eta_i)$. For $m=0,1,2$, define the risk-set and event-set
moments

$$
S_{sk}^{(m)}
 =\sum_{j\in R_{sk}}w_jx_j^{\otimes m},
\qquad
F_{sk}^{(m)}
 =\sum_{i\in D_{sk}}w_ix_i^{\otimes m},
$$

where $x^{\otimes0}=1$, $x^{\otimes1}=x$, and
$x^{\otimes2}=xx^\top$. These three moments are sufficient for the
Breslow/Efron likelihood, score, and observed information.

### 2. Partial likelihood and tie handling

**Breslow.** All $d_{sk}$ tied events use the same risk-set denominator:

$$
\ell_B(\beta)
 =\sum_{s,k}\left[
   \sum_{i\in D_{sk}}\eta_i
   -d_{sk}\log S_{sk}^{(0)}
 \right].
$$

**Efron.** The event weights are removed fractionally across the $d_{sk}$
substeps. With $q=0,\ldots,d_{sk}-1$,

$$
\ell_E(\beta)
 =\sum_{s,k}\left[
   \sum_{i\in D_{sk}}\eta_i
   -\sum_{q=0}^{d_{sk}-1}
     \log\left\{
       S_{sk}^{(0)}-\frac{q}{d_{sk}}F_{sk}^{(0)}
     \right\}
 \right].
$$

**Exact.** The denominator sums over every size-$d_{sk}$ subset of the risk
set:

$$
Z_{sk}^{(d_{sk})}(\beta)
 =\sum_{\substack{A\subseteq R_{sk}\\|A|=d_{sk}}}
   \exp\left(\sum_{j\in A}\eta_j\right),
$$

$$
\ell_X(\beta)
 =\sum_{s,k}\left[
   \sum_{i\in D_{sk}}\eta_i-\log Z_{sk}^{(d_{sk})}(\beta)
 \right].
$$

The Exact denominator is a combinatorial sum, but StatGPU does **not**
enumerate all subsets. It evaluates the required elementary-symmetric
polynomial by dynamic programming (Section 4). When there is one event in a
failure group, all three tie methods reduce to the ordinary Cox contribution.

### 3. Score and observed information

Let

$$
U(\beta)=\frac{\partial\ell(\beta)}{\partial\beta},
\qquad
J(\beta)=-\frac{\partial^2\ell(\beta)}
                 {\partial\beta\,\partial\beta^\top}.
$$

StatGPU returns $J$ in the positive-information convention.

For Breslow, let
$\mu_{sk}=S_{sk}^{(1)}/S_{sk}^{(0)}$. Then

$$
U_B(\beta)
 =\sum_{s,k}\left[
   \sum_{i\in D_{sk}}x_i-d_{sk}\mu_{sk}
 \right],
$$

$$
J_B(\beta)
 =\sum_{s,k}d_{sk}\left[
   \frac{S_{sk}^{(2)}}{S_{sk}^{(0)}}
   -\mu_{sk}\mu_{sk}^{\top}
 \right].
$$

For Efron, define $c_{skq}=q/d_{sk}$,

$$
a_{skq}=S_{sk}^{(0)}-c_{skq}F_{sk}^{(0)},
\qquad
\mu_{skq}
 =\frac{S_{sk}^{(1)}-c_{skq}F_{sk}^{(1)}}{a_{skq}}.
$$

The score and information are

$$
U_E(\beta)
 =\sum_{s,k}\left[
   \sum_{i\in D_{sk}}x_i-\sum_{q=0}^{d_{sk}-1}\mu_{skq}
 \right],
$$

$$
J_E(\beta)
 =\sum_{s,k}\sum_{q=0}^{d_{sk}-1}
 \left[
   \frac{S_{sk}^{(2)}-c_{skq}F_{sk}^{(2)}}{a_{skq}}
   -\mu_{skq}\mu_{skq}^{\top}
 \right].
$$

For Exact ties, assign each size-$d_{sk}$ subset $A$ probability

$$
\pi_{sk}(A)
 =\frac{\exp(\sum_{j\in A}\eta_j)}
        {Z_{sk}^{(d_{sk})}},
\qquad
z_A=\sum_{j\in A}x_j.
$$

The denominator derivatives are the mean and covariance of this weighted
subset sum:

$$
m_{sk}=\mathbb E_{\pi_{sk}}(z_A),
\qquad
V_{sk}=\operatorname{Cov}_{\pi_{sk}}(z_A).
$$

Consequently,

$$
U_X(\beta)
 =\sum_{s,k}\left[
   \sum_{i\in D_{sk}}x_i-m_{sk}
 \right],
\qquad
J_X(\beta)=\sum_{s,k}V_{sk}.
$$

This is why Exact evaluation maintains the log partition, the weighted mean
subset sum, and its second moment. It obtains the score and information without
materializing the subsets.

### 4. Exact ties: elementary-symmetric dynamic programming

For one risk set, write $w_j=\exp(\eta_j)$. Let $Z_{r}^{(q)}$ be the
elementary-symmetric sum of order $r$ after processing the first $q$ risk
rows. The recurrence is

$$
Z_r^{(q)}
 =Z_r^{(q-1)}+w_qZ_{r-1}^{(q-1)},
\qquad
Z_0^{(q)}=1,
\qquad
Z_r^{(0)}=0\quad(r>0).
$$

The required Exact denominator is $Z_d^{(n_R)}$, where $n_R$ is the risk-set
size and $d$ is the tied-event count. The general per-group and batched Exact
paths evaluate the recurrence in log space,

$$
\log Z_r^{(q)}
 =\operatorname{logaddexp}\!\left(
   \log Z_r^{(q-1)},
   \eta_q+\log Z_{r-1}^{(q-1)}
 \right),
$$

The same update also carries the normalized first and second moments of the
subset covariate sum. Let $m_r^{(q)}=\mathbb E(z_A)$ and
$C_r^{(q)}=\mathbb E(z_Az_A^\top)$ under the weighted distribution over
size-$r$ subsets of the first $q$ rows. Define the mixture weights

$$
a_r^{(q)}
 =\exp\{\log Z_r^{(q-1)}-\log Z_r^{(q)}\},
\qquad
b_r^{(q)}
 =\exp\{\eta_q+\log Z_{r-1}^{(q-1)}-\log Z_r^{(q)}\}.
$$

They correspond to subsets that exclude or include row $q$, respectively.
The moment updates are

$$
m_r^{(q)}
 =a_r^{(q)}m_r^{(q-1)}
 +b_r^{(q)}\left(m_{r-1}^{(q-1)}+x_q\right),
$$

$$
\begin{aligned}
C_r^{(q)}
={}&a_r^{(q)}C_r^{(q-1)}\\
&+b_r^{(q)}\left[
C_{r-1}^{(q-1)}
+m_{r-1}^{(q-1)}x_q^\top
+x_qm_{r-1}^{(q-1)\top}
+x_qx_q^\top
\right].
\end{aligned}
$$

Initialize $m_0^{(q)}=0$ and $C_0^{(q)}=0$; subset sizes that exceed the number
of processed rows are inactive. At the required subset size $d$, the exact-tie
score and information use

$$
m_{sk}=m_d^{(n_R)},
\qquad
V_{sk}=C_d^{(n_R)}-m_{sk}m_{sk}^\top.
$$

Thus the DP produces the partition, score, and information together; it
does not need to materialize the subsets.

For ordinary right-censored data, risk sets are nested within each stratum.
Sorting rows by decreasing stop time turns each risk set into a prefix. StatGPU
therefore reuses one segmented prefix DP across failure groups instead of
restarting the DP for every group. For fixed moment width, the row/subset-size
work scales as $O(n\,d_{\max})$, where $d_{\max}$ is the largest tied-event
count; derivative-state width additionally grows with $p+p^2$. This prefix
path is used only when conservative bounds on predictor range, combinatorial
counts, and moment magnitude are safe; otherwise the normalized log-space DP
is retained.

The Torch CUDA channel-wise scan is a narrowly gated optimization. It is
eligible only for at least 2,048 rows and when a scanned moment tensor has
at most 64 channels. The row
and channel gates are controlled by
**STATGPU_TORCH_EXACT_SCAN_MIN_ROWS** and
**STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS**; **STATGPU_TORCH_EXACT_SCAN_STRATEGY**
accepts auto, native, or channelwise. The auto strategy enables the channel-wise
scan only for the benchmarked Torch 2.0 + Pascal/P100 combination; other
configurations keep the native scan. Its extra workspace is included in the
nested-DP memory check. If that workspace does not fit, the nested algorithm
uses the native Torch scan rather than falling back to the general Exact path.

Delayed entry changes membership to $a_i<t\le b_i$, so a single ordering no
longer makes every risk set a nested prefix. Those cases use batched Exact
states when the estimated workspace fits, or a bounded per-group DP otherwise.
The fallback changes the batching strategy, not the statistical likelihood or
the selected array backend.

### 5. Penalized Newton optimization

The public **CoxPH** estimator uses L2 penalty $\lambda$ on the summed
partial log-likelihood scale:

$$
Q_\lambda(\beta)
 =\ell(\beta)-\lambda\|\beta\|_2^2.
$$

Its penalized score and Newton information are

$$
U_\lambda(\beta)=U(\beta)-2\lambda\beta,
\qquad
A(\beta)=J(\beta)+2\lambda I_p.
$$

At iteration $m$, StatGPU solves

$$
A(\beta^{(m)})\Delta^{(m)}
 =U_\lambda(\beta^{(m)}),
\qquad
\beta^{(m+1)}
 =\beta^{(m)}+\alpha_m\Delta^{(m)}.
$$

The step length starts at one and is halved until the penalized objective
satisfies the ascent condition

$$
Q_\lambda(\beta^{(m)}+\alpha\Delta^{(m)})
\ge Q_\lambda(\beta^{(m)})
+10^{-4}\alpha U_\lambda(\beta^{(m)})^\top\Delta^{(m)}
-10^{-12}.
$$

A rejected line search leaves the last accepted
coefficient vector unchanged. Convergence is certified using the normalized
KKT residual

$$
r_{\mathrm{KKT}}
 =\frac{\|U_\lambda(\beta)\|_\infty}
        {1+\|U(\beta)\|_\infty+2\lambda\|\beta\|_\infty};
\qquad
r_{\mathrm{KKT}}\le\texttt{tol}.
$$

A small coefficient step or reaching **max_iter** is not, by itself, a
convergence certificate.

### 6. Where GPU parallelism occurs

The Newton iterations are **sequential**: iteration $m+1$ depends on
$\beta^{(m)}$. The parallel work is inside each objective/derivative
evaluation.

1. **Ordinary right-censored, unstratified Breslow/Efron fits.** When there is
   no delayed entry, no explicit stratum or subject label, and
   **cov_type="nonrobust"**, **CoxPH** can use the specialized
   **CoxPartialLikelihoodLoss** path. It sorts the observations once and obtains
   risk-set denominators and weighted covariate moments from stable reverse
   cumulative sums. Efron event contributions are grouped and accumulated by
   failure time. The sorted state is reused across Newton iterations.

2. **General counting-process Breslow/Efron fits.** Failure groups are formed
   within strata. For group $g$, let $(s_g,t_g)$ denote its stratum and failure
   time. For a batch of $G$ groups, define the risk and event masks

   $$
   M_{gi}=\mathbf{1}\{a_i<t_g\le b_i,\ s_i=s_g\},
   \qquad
   E_{gi}=\mathbf{1}\{\delta_i=1,\ b_i=t_g,\ s_i=s_g\},
   \qquad M,E\in\{0,1\}^{G\times n}.
   $$

   The mask is a computational representation of risk-set membership, not an
   additional model parameter. Each row represents one stratum/failure-time
   group; each column represents one observation. The GPU reduces along the
   observation axis to compute

   $$
   S_g^{(m)}=\sum_{i=1}^n M_{gi}w_ix_i^{\otimes m},
   \qquad
   F_g^{(m)}=\sum_{i=1}^n E_{gi}w_ix_i^{\otimes m},
   \qquad m=0,1,2.
   $$

   These are batched row reductions and matrix contractions, not independent
   Python loops over failure times. If a dense batch would exceed
   **STATGPU_COX_GROUP_MAX_BYTES** (512 MiB by default), the implementation
   streams observations in bounded chunks and accumulates the same moments on
   the selected backend.

3. **Exact ties.** The elementary-symmetric recurrence is the required
   dependency. The implementation vectorizes updates over subset sizes and,
   where applicable, batches failure groups. For ordinary right-censored data,
   nested risk sets permit the shared prefix DP described above. Workspace
   limits **STATGPU_EXACT_NESTED_MAX_BYTES** and
   **STATGPU_EXACT_BATCH_MAX_BYTES** (512 MiB by default) select a bounded
   alternative when a larger temporary state would exceed the configured
   ceiling.

The distinction is important: StatGPU does not parallelize the outer Newton
iterations, and it does not claim that every Exact DP dependency is independent.
It parallelizes the row/group reductions and moment operations that can be
batched, while preserving the recurrences required by the likelihood.

For numerical stability, covariates are centered within strata in the general
counting-process engine; this does not change the partial likelihood because a
common covariate shift cancels within each stratum's event/risk-set comparison.
Risk-set exponentials use local shifts. The general Exact DP uses normalized
moments and log-space **logaddexp** updates; the nested prefix path uses
per-stratum shifted weights and is guarded by the numerical bounds above.
Workspace ceilings limit the corresponding temporary arrays, not the total
memory of a fit. These are algorithmic memory fallbacks;
they are not implicit CPU fallbacks.

### 7. Baseline hazard and covariance

After estimating $\widehat\beta$, the baseline hazard uses the conventional
Breslow increment, irrespective of the tie method used for coefficient
estimation:

$$
\Delta\widehat H_{0s}(t_{sk})
 =\frac{d_{sk}}
        {\sum_{j\in R_{sk}}\exp(x_j^\top\widehat\beta)},
\qquad
\widehat S_s(t\mid x)
 =\exp\!\left\{-\widehat H_{0s}(t)
                    \exp(x^\top\widehat\beta)\right\}.
$$

Let $J=J(\widehat\beta)$ and $A=J+2\lambda I_p$. For an unpenalized
model-based fit, the covariance is $J^{-1}$. For a positive fixed L2 penalty,
the model-based frequentist covariance is

$$
\widehat{\operatorname{Var}}(\widehat\beta)
 =A^{-1}JA^{-1},
$$

not $A^{-1}$. For robust inference, let $u_g$ be the score residual
aggregated over independent unit $g$ (subject or cluster), and let
$B=\sum_g u_gu_g^\top$. The sandwich covariance is

$$
\widehat{\operatorname{Var}}_{\mathrm{robust}}(\widehat\beta)
 =A^{-1}BA^{-1}.
$$

Exact ties currently support model-based covariance only; HC0, HC1, and cluster
covariance with **ties="exact"** raise **NotImplementedError**.

### 8. Source map

- [_cox.py](../../statgpu/survival/_cox.py): public CoxPH validation,
  dispatch, fitted state, baseline output, and inference assembly.
- [_cox_counting.py](../../statgpu/survival/_cox_counting.py): backend-native
  damped Newton solver, line search, and KKT verification.
- [_risk_sets.py](../../statgpu/survival/_risk_sets.py): counting-process risk
  sets, Breslow/Efron/Exact objective, score, information, Exact DP, and
  baseline hazard.
- [_cox_ph.py](../../statgpu/losses/_cox_ph.py): specialized ordinary
  right-censored Breslow/Efron loss and reverse-scan moments.
- [_cox_inference.py](../../statgpu/survival/_cox_inference.py): covariance
  validation and backend-specific information inversion.
- [_cox_cv.py](../../statgpu/survival/_cox_cv.py): cross-validation and
  selection/refit orchestration.

## CV preparation and transfer implementation snapshot

These details describe the recorded CV implementation, not permanent operation
counts or a performance guarantee. Public field meanings are in the
[device and CV diagnostics reference](../../docs/en/reference/coxph-diagnostics.md).

The staged-safety adapter routes either experimental screening request to one
exhaustive full-precision selector call. Each effective fold is prepared once
for that pass; it does not retain a staged cache or repeatedly prepare stages.
This is distinct from reusing a complete selection result through the selection
cache.

A target-preparation count represents one complete time/event metadata
preparation. Each preparation copies two
actual target vectors, so
**candidate_target_host_vector_transfer_count** is twice
**candidate_target_host_transfer_count**. The corresponding per-call counters
reset to zero on a selection-cache hit, while selection-origin fields are
retained and current-request/final-refit device fields are updated. These
counters do not rule out input host transfer or work in the final refit.

## Historical external validation

The maintained R comparison uses R 4.4.1 with **survival** 3.8.9 and aligns tie
handling, Newton **max_iter=80**, and **tol=1e-8**. At $n=3000$, $p=10$, the
Breslow and Efron comparisons use 3,000 independent HC1 units and 120 cluster
units. The maximum StatGPU-versus-R coefficient/SE/p-value differences were
$5.55\times10^{-16}/1.39\times10^{-16}/8.00\times10^{-19}$ for HC1 and
$5.55\times10^{-16}/1.32\times10^{-16}/2.22\times10^{-16}$ for cluster
covariance.

Machine-readable R comparison artifacts:

- results/benchmark_frontend_sources/coxph_robust_inference_breslow_pr80_20260729_schema11.json;
- results/benchmark_frontend_sources/coxph_robust_inference_efron_pr80_20260729_schema11.json.

These are fixed-source, shape-specific comparisons, not universal accuracy or
performance guarantees. Exact-tie and performance conclusions remain tied to
their dedicated artifacts listed in [the PR80 review record](../reviews/pr80_review_fix.md).

### Published physical-GPU validation

A stable [published GPU validation record](https://gist.github.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139)
([raw JSON](https://gist.githubusercontent.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139/raw/pr80_final_gpu_suite_schema3.json))
is pinned to source commit
a726937a39eb0ed5a370dd03362884b63a9e9818, with artifact SHA-256
e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb.

The record certifies numerical agreement, device ownership, and error
boundaries only for its recorded source and runtime environment. It is not a
new performance-crossover benchmark or R-alignment run, and later commits do
not automatically inherit that evidence. It is not a universal guarantee for
every dataset, backend version, or GPU.
