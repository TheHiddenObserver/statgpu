# CRAN R Package Mapping & Future Directions

> Created: 2026-06-17
> Source: https://cran.r-project.org/web/packages/available_packages_by_name.html
> Cross-referenced with: dev/plans/TO_DO.md, dev/plans/panel_framework_proposal.md

## 1. Linear Models & GLM

### Current statgpu status
- Ridge, Lasso, ElasticNet, Logistic, 7 GLM families, Penalized, Ordered, CV

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `glmnet` | Lasso, ElasticNet, Ridge (GLM path) | `PenalizedGLM_CV` | Multinomial, sparse input |
| `MASS` | `glm.nb()`, `rlm()` (robust LM) | `NegativeBinomialRegression` | Robust LM |
| `robustbase` | `lmrob()`, `glmrob()` | ❌ | Robust regression |
| `quantreg` | `rq()` (quantile regression) | ❌ | Quantile regression |
| `lars` | LARS, Lasso path | ❌ | LARS algorithm |
| `biglasso` | Big data Lasso | ❌ | Large-scale penalized |
| `penalized` | L1/L2 penalized | `PenalizedGLM` | — |
| `rms` | `ols()`, `lrm()`, `Glm()` | ❌ | Regression modeling strategies |

### Priority for statgpu

**Tier 1:**
- [ ] Multinomial LogisticRegression (softmax) — `nnet::multinom()`
- [ ] Sparse input (CSR/CSC) for linear models — `Matrix` package

**Tier 2:**
- [ ] Quantile regression — `quantreg::rq()`
- [ ] Robust regression (M-estimator) — `MASS::rlm()`, `robustbase::lmrob()`

**Tier 3:**
- [ ] LARS algorithm — `lars::lars()`

---

## 2. Survival Analysis

### Current statgpu status
- CoxPH, CoxPHCV, Breslow/Efron, robust SE, cluster, delayed entry

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `survival` | `coxph()`, `survreg()`, `strata()`, `frailty()` | `CoxPH` | strata, frailty |
| `survminer` | Visualization, diagnostics | ❌ | Diagnostics |
| `flexsurv` | Flexible parametric survival | ❌ | AFT models |
| `mstate` | Multi-state models | ❌ | Multi-state |
| `cmprsk` | Competing risks | ❌ | Competing risks |
| `timereg` | Time-varying coefficients | ❌ | Time-varying Cox |
| `bshazard` | Smooth hazard | ❌ | Hazard estimation |
| `eha` | Event history analysis | ❌ | Piecewise constant |
| `frailtypack` | Frailty models | ❌ | Shared/complex frailty |
| `JM` | Joint models (survival + longitudinal) | ❌ | Joint models |

### Priority for statgpu

**Tier 1:**
- [ ] CoxPH strata — `survival::coxph(strata())`
- [ ] CoxPH frailty — `survival::coxph(frailty())`
- [ ] Time-varying covariates — `survival::coxph()` with counting process

**Tier 2:**
- [ ] Parametric survival (Weibull, log-normal) — `survival::survreg()`
- [ ] Competing risks (Fine-Gray) — `cmprsk::crr()`
- [ ] Survival diagnostics (Schoenfeld, deviance residuals) — `survival::cox.zph()`

**Tier 3:**
- [ ] Multi-state models — `mstate::msprep()`
- [ ] Joint models — `JM::jm()`

---

## 3. Panel Data / Econometrics

### Current statgpu status
- PanelOLS, RandomEffects, PooledOLS, BetweenOLS, FirstDifferenceOLS, FamaMacBeth, HAC, clustered SE

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `plm` | `plm()` (FE/RE/pooling/between/FD), `pvcm()`, `pcce()` | Most models | RE variants, CRE |
| `fixest` | `feols()`, `fepois()`, `rmarkdown` | `PanelOLS` | High-dim FE, IV |
| `linearmodels` (Py) | `PanelOLS`, `IVPanelOLS`, `FamaMacBeth` | Most models | Panel IV |
| `sandwich` | `vcovHC()`, `vcovHAC()`, `vcovCL()`, `vcovNW()` | `hac_covariance` | HC0-HC5, Driscoll-Kraay |
| `lmtest` | `coeftest()`, `waldtest()`, `bptest()` | ❌ | Specification tests |
| `AER` | `ivreg()`, `iv()` | ❌ | IV regression |
| `systemfit` | `systemfit()` (SUR, 2SLS, 3SLS) | ❌ | System of equations |
| `panelvar` | Panel VAR | ❌ | Panel VAR |
| `pder` | Panel data econometrics with R | ❌ | — |
| `pcse` | Panel-corrected SE | ❌ | Beck-Katz PCSE |
| `plm` | `pht()` (Hausman-Taylor) | ❌ | HT estimator |
| `broom` | Tidy model output | `PanelSummary` | — |

### Priority for statgpu

**Tier 1 (from panel_framework_proposal.md):**
- [ ] Hausman test (FE vs RE) — `plm::phtest()`
- [ ] Overall/Between R², Adjusted R², F-statistic
- [ ] LM test (Breusch-Pagan) — `lmtest::bptest()`
- [ ] Robust SE for RE — `plm::vcovHC()`

**Tier 2:**
- [ ] Panel IV (2SLS/GMM) — `plm::plm(iv)` or `AER::ivreg()`
- [ ] Driscoll-Kraay SE — `sandwich::vcovCL()` with DK option
- [ ] HC0/HC2/HC3 robust SEs — `sandwich::vcovHC(type="HC0"/"HC2"/"HC3")`
- [ ] High-dim FE (absorb) — `fixest::feols()` (sparse matrix techniques)
- [ ] Mundlak test / CRE — `plm::pvcm(model="random")`

**Tier 3:**
- [ ] Dynamic Panel (Arellano-Bond) — `plm::pgmm()`
- [ ] Panel-corrected SE (Beck-Katz) — `pcse::pcse()`
- [ ] Hausman-Taylor estimator — `plm::pht()`
- [ ] AR(1) errors — `plm::plm(model="random", effect="twoways")`
- [ ] DID / Event Study — `fixest::feols()` with `sunab()`
- [ ] Panel VAR — `panelvar::panelvar()`

---

## 4. ANOVA / Experimental Design

### Current statgpu status
- f_oneway, f_twoway, f_welch, tukey_hsd, bonferroni, cohens_f, partial_eta_squared

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `stats` | `aov()`, `TukeyHSD()`, `pairwise.t.test()` | Most | Repeated measures |
| `car` | `Anova()` (Type II/III), `leveneTest()` | ❌ | Type II/III SS |
| `emmeans` | Estimated marginal means | ❌ | EMM |
| `multcomp` | `glht()` (general linear hypotheses) | ❌ | Multiple comparisons |
| `lme4` | `lmer()` (mixed effects ANOVA) | ❌ | Mixed ANOVA |
| `ez` | `ezANOVA()` | ❌ | User-friendly ANOVA |
| `afex` | `aov_ez()` | ❌ | ANOVA with effect sizes |
| `effectsize` | `eta_squared()`, `cohens_f()` | `cohens_f` | More effect sizes |
| `rstatix` | `anova_test()`, `tukey_hsd()` | ❌ | Tidy ANOVA |

### Priority for statgpu

**Tier 1:**
- [ ] Repeated measures ANOVA — `stats::aov(Error(subject) + treatment)`
- [ ] Type II/III SS — `car::Anova(type="III")`

**Tier 2:**
- [ ] Levene's test for equal variances — `car::leveneTest()`
- [ ] Estimated marginal means — `emmeans::emmeans()`
- [ ] More effect sizes (omega-squared, epsilon-squared) — `effectsize::eta_squared()`

**Tier 3:**
- [ ] Mixed ANOVA (between + within) — `lme4::lmer()`
- [ ] General linear hypotheses — `multcomp::glht()`

---

## 5. Covariance Estimation

### Current statgpu status
- EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance, MinCovDet, GraphicalLasso/CV

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `stats` | `cov()`, `cor()` | `EmpiricalCovariance` | — |
| `MASS` | `cov.mve()`, `cov.mcd()` | `MinCovDet` | — |
| `robustbase` | `covMcd()`, `covRob()` | `MinCovDet` | More robust methods |
| `glasso` | `glasso()` | `GraphicalLasso` | — |
| `huge` | High-dimensional graphical lasso | ❌ | High-dim GLasso |
| `corpcor` | `cor2pcor()`, `pcor.shrink()` | ❌ | Partial correlation |
| `nlshrink` | Nonlinear shrinkage | ❌ | Oracle shrinkage |
| `rrcov` | `CovRobust()`, `CovMest()` | ❌ | M-estimator, OGK |
| `pcaPP` | Robust PCA | ❌ | Robust PCA |
| `cluster` | `clusGap()`, `pam()` | ❌ | Clustering with robust cov |
| `sandwich` | HAC, clustered cov | `hac_covariance` | — |

### Priority for statgpu

**Tier 2:**
- [ ] OGK (Orthogonalized Gnanadesikan-Kettenring) — `rrcov::CovOgk()`
- [ ] M-estimator of covariance — `rrcov::CovMest()`
- [ ] Partial correlation — `corpcor::cor2pcor()`

**Tier 3:**
- [ ] High-dim graphical lasso (adaptive) — `huge::huge()`
- [ ] Nonlinear shrinkage — `nlshrink::nlshrink()`

---

## 6. Nonparametric / Splines / GAM

### Current statgpu status
- KDE, kernel regression, bspline, natural_cubic, SplineTransformer, cyclic/thin plate, GAM

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `splines` | `bs()`, `ns()`, `poly()` | `bspline_basis`, `natural_cubic_spline_basis` | — |
| `mgcv` | `gam()`, `s()`, `te()` | `GAM` | Tensor product, adaptive |
| `np` | `npreg()`, `npdensity()` | `KernelRegression`, `KDE` | — |
| `KernSmooth` | `dpik()`, `locpoly()` | `KDE` | Local polynomial |
| `gss` | `ssanova()`, `ssden()` | ❌ | Smoothing spline ANOVA |
| `SemiPar` | `spm()` | ❌ | Semiparametric regression |
| `refund` | Functional regression | ❌ | Functional data |
| `mgcv` | `bam()` (big data GAM) | ❌ | Large-scale GAM |
| `scam` | Shape-constrained additive models | ❌ | Monotonicity constraints |

### Priority for statgpu

**Tier 1:**
- [ ] Tensor product splines — `mgcv::te()`
- [ ] GAM with automatic smoothing parameter selection (GCV/REML) — `mgcv::gam(method="REML")`

**Tier 2:**
- [ ] Shape-constrained GAM (monotonic, convex) — `scam::scam()`
- [ ] Large-scale GAM (chunked) — `mgcv::bam()`
- [ ] Smoothing spline ANOVA — `gss::ssanova()`

**Tier 3:**
- [ ] Functional data analysis — `refund::pfr()`
- [ ] Local polynomial regression — `KernSmooth::locpoly()`

---

## 7. Kernel Methods

### Current statgpu status
- 7 kernels, KernelRidge, KernelRidgeCV, Nystroem, KernelPCA, chi2_kernel

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `kernlab` | `ksvm()`, `krr()`, `kpca()` | `KernelRidge`, `KernelPCA` | SVM |
| `e1071` | `svm()` | ❌ | SVM |
| `liquidSVM` | GPU SVM | ❌ | GPU SVM |
| `KKNN` | `kknn()` (k-nearest neighbors) | ❌ | KNN |
| `kerndp` | Kernel density estimation with kernels | `KDE` | — |

### Priority for statgpu

**Tier 2:**
- [ ] Kernel SVM (SVC, SVR) — `kernlab::ksvm()`
- [ ] KNN classifier/regressor — `kknn::kknn()`

---

## 8. Inference / Multiple Testing

### Current statgpu status
- 15 distributions, p-value adjustment, bootstrap, permutation

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `stats` | `p.adjust()`, `prop.test()`, `chisq.test()` | `adjust_pvalues` | — |
| `multtest` | Multiple testing procedures | ❌ | Step-down procedures |
| `coin` | Permutation tests | `permutation_test` | — |
| `boot` | Bootstrap | `bootstrap_statistic` | Block bootstrap |
| `sandwich` | Robust inference | ❌ | — |

### Priority for statgpu

**Tier 2:**
- [ ] Block bootstrap (for time series/panel) — `boot::tsboot()`
- [ ] Step-down multiple testing — `multtest::mt.rawp2adjp()`

---

## 9. Feature Selection

### Current statgpu status
- KnockoffSelector, StepwiseSelector

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `glmnet` | CV path for Lasso/ElasticNet | `LassoCV`, `ElasticNetCV` | — |
| `knockoff` | Knockoff filter | `KnockoffSelector` | — |
| `Boruta` | Boruta feature selection | ❌ | Random forest-based |
| `rfe` (caret) | Recursive feature elimination | ❌ | RFE |
| `mRMRe` | Minimum redundancy maximum relevance | ❌ | mRMR |

### Priority for statgpu

**Tier 3:**
- [ ] Recursive feature elimination — `caret::rfe()`
- [ ] Stability selection — `stabsel::stabsel()`

---

## 10. Unsupervised / Clustering

### Current statgpu status
- PCA, IncrementalPCA, TruncatedSVD, KMeans, MiniBatchKMeans, DBSCAN, GMM, NMF, Agglomerative, UMAP, tSNE

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `stats` | `kmeans()`, `hclust()`, `prcomp()` | Most | — |
| `cluster` | `pam()`, `diana()`, `fanny()` | ❌ | PAM, DIANA |
| `mclust` | `Mclust()` (model-based clustering) | `GaussianMixture` | — |
| `dbscan` | `dbscan()`, `hdbscan()` | `DBSCAN` | HDBSCAN |
| `uwot` | UMAP | `UMAP` | — |
| `Rtsne` | t-SNE | `TSNE` | — |
| `kerdimred` | Kernel dimension reduction | ❌ | Kernel methods |

### Priority for statgpu

**Tier 2:**
- [ ] HDBSCAN — `dbscan::hdbscan()`
- [ ] PAM (Partitioning Around Medoids) — `cluster::pam()`

---

## 11. Metrics / Diagnostics

### Current statgpu status
- ROC, AUC, confusion matrix, RegressionDiagnostics

### Relevant CRAN packages

| R Package | Methods | statgpu Equivalent | Gap |
|-----------|---------|-------------------|-----|
| `pROC` | `roc()`, `ci.auc()` | `evaluate_binary_classification` | CI for AUC |
| `ROCR` | `prediction()`, `performance()` | ❌ | Precision-recall |
| `car` | `vif()`, `influence.measures()` | ❌ | VIF, influence |
| `lmtest` | `bptest()`, `dwtest()` | ❌ | BP test, DW test |
| `DHARMa` | Residual diagnostics | ❌ | Simulated residuals |

### Priority for statgpu

**Tier 2:**
- [ ] VIF (Variance Inflation Factor) — `car::vif()`
- [ ] Breusch-Pagan test — `lmtest::bptest()`
- [ ] Durbin-Watson test — `lmtest::dwtest()`
- [ ] Influence measures (Cook's D, leverage) — `car::influence.measures()`

---

## 12. Missing Domains (not in current plans)

### 12.1 Time Series

| R Package | Methods | Priority |
|-----------|---------|----------|
| `forecast` | `auto.arima()`, `ets()` | Low |
| `tseries` | `adf.test()`, `garch()` | Low |
| `vars` | VAR, SVAR | Low |
| `urca` | Unit root tests | Low |

**Note:** Time series is a large domain. Consider as a future module if demand exists.

### 12.2 Causal Inference

| R Package | Methods | Priority |
|-----------|---------|----------|
| `MatchIt` | Propensity score matching | Medium |
| `CausalImpact` | Causal impact analysis | Low |
| `dagitty` | DAG analysis | Low |
| `mediation` | Mediation analysis | Medium |

**Note:** DID is already in panel plans. Consider mediation and matching as extensions.

### 12.3 Bayesian Methods

| R Package | Methods | Priority |
|-----------|---------|----------|
| `rstanarm` | Bayesian GLM, GLMM | Low |
| `brms` | Bayesian multilevel models | Low |
| `BayesFactor` | Bayesian hypothesis testing | Low |

**Note:** Bayesian methods require fundamentally different architecture (MCMC). Not a natural fit for GPU-accelerated frequentist methods.

### 12.4 Power Analysis / Sample Size

| R Package | Methods | Priority |
|-----------|---------|----------|
| `pwr` | `pwr.t.test()`, `pwr.anova.test()` | Medium |
| `simr` | Power for mixed models | Low |

**Note:** Power analysis is a natural companion to ANOVA and t-tests. Medium priority.

---

## Summary: Recommended Next Steps

### Phase A: Complete current P2 (already in progress)
- ✅ ANOVA (two-way, Welch, post-hoc, effect sizes)
- ✅ Covariance (ShrunkCov, MinCovDet, GraphicalLasso)
- ✅ Panel (PooledOLS, BetweenOLS, FDO, FMB, HAC)
- ✅ Splines (SplineTransformer, cyclic, thin plate)
- ✅ Kernel (chi2, Nystroem, KernelPCA)

### Phase B: Panel Tier 1 (from framework proposal)
- [ ] Hausman test, LM test, F-test, R² variants, F-statistic
- [ ] Robust SE for RE
- R equivalents: `plm::phtest()`, `lmtest::bptest()`, `plm::pooltest()`

### Phase C: ANOVA + Diagnostics
- [ ] Repeated measures ANOVA
- [ ] Type II/III SS
- [ ] Levene's test
- [ ] VIF, Cook's D, BP test
- R equivalents: `car::Anova()`, `car::leveneTest()`, `car::vif()`

### Phase D: Panel Tier 2
- [ ] Panel IV (2SLS/GMM)
- [ ] Driscoll-Kraay SE
- [ ] HC0/HC2/HC3
- [ ] High-dim FE
- R equivalents: `AER::ivreg()`, `sandwich::vcovHC()`, `fixest::feols()`

### Phase E: Survival Extensions
- [ ] CoxPH strata, frailty, time-varying
- [ ] Parametric survival
- [ ] Competing risks
- R equivalents: `survival::coxph(strata())`, `survival::survreg()`, `cmprsk::crr()`

### Phase F: Advanced Methods
- [ ] DID / Event Effects
- [ ] Dynamic Panel (AB/GMM)
- [ ] Quantile regression
- [ ] Robust regression
- R equivalents: `fixest::feols(sunab())`, `plm::pgmm()`, `quantreg::rq()`, `MASS::rlm()`

---

## See Also

> **`cran_r_package_supplement.md`** — Comprehensive supplement covering 12 entirely new domains
> (Quantile Regression, Robust Regression, Mixed Effects, Meta-Analysis, Change Point Detection,
> Multivariate Statistics, IRT, Copula, SEM, EVT, Missing Data, Survey) and 5 major enhancements
> to existing plans. Includes priority ranking, GPU acceleration analysis, proposed APIs, and
> backend architecture implications.
