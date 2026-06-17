# Panel Data Module: Feature Comparison & Unified Framework Proposal

> Created: 2026-06-17
> Status: Proposal for future PRs

## 1. Feature Comparison

### 1.1 Core Estimators

| Feature | statgpu | plm (R) | linearmodels (Py) | fixest (R) | Stata |
|---|:---:|:---:|:---:|:---:|:---:|
| Pooled OLS | ✅ | ✅ | ✅ | ✅ | `regress` |
| Fixed Effects (within) | ✅ | ✅ | ✅ | ✅ | `xtreg fe` |
| Entity FE | ✅ | ✅ | ✅ | ✅ | `xtreg fe` |
| Time FE | ✅ | ✅ | ✅ | ✅ | `xtreg fe i.time` |
| Two-way FE | ✅ | ✅ | ✅ | ✅ | `reghdfe` |
| High-dim FE (absorb) | ❌ | ❌ | ❌ | ✅ | `reghdfe` |
| Random Effects (Swamy-Arora) | ✅ | ✅ | ✅ | ❌ | `xtreg re` |
| RE: Wallace-Hussain | ❌ | ✅ | ❌ | ❌ | — |
| RE: Amemiya-MaCurdy | ❌ | ✅ | ❌ | ❌ | — |
| Between OLS | ✅ | ✅ | ✅ | ❌ | `xtreg be` |
| First-Difference OLS | ✅ | ✅ | ✅ | ❌ | — |
| Fama-MacBeth | ✅ | ❌ | ✅ | ❌ | `xtfmb` |
| Correlated RE (Mundlak) | ❌ | ✅ | ❌ | ❌ | `xtreg cre` |
| Mixed Effects / REML | ❌ | ❌ | ❌ | ❌ | `xtmixed` |
| Dynamic Panel (AB/GMM) | ❌ | ✅ | ❌ | ❌ | `xtabond` |
| Panel IV (2SLS/GMM) | ❌ | ✅ | ✅ | ✅ | `xtivreg` |

### 1.2 Covariance / Standard Errors

| Feature | statgpu | plm | linearmodels | fixest | Stata |
|---|:---:|:---:|:---:|:---:|:---:|
| Nonrobust | ✅ | ✅ | ✅ | ✅ | ✅ |
| HC1 (robust) | ✅ | ❌ | ✅ | ✅ | `robust` |
| HC0/HC2/HC3 | ❌ | ❌ | ❌ | ✅ | — |
| Clustered 1-way | ✅ | ✅ | ✅ | ✅ | `vce(cluster)` |
| Clustered 2-way | ✅ | ✅ | ✅ | ✅ | `vce(cluster)` |
| HAC / Newey-West | ✅ | ✅ | ✅ | ❌ | `newey` |
| Driscoll-Kraay | ❌ | ❌ | ✅ | ❌ | `xtscc` |
| PCSE (Beck-Katz) | ❌ | ❌ | ❌ | ❌ | `xtpcse` |

### 1.3 Specification Tests

| Feature | statgpu | plm | linearmodels | fixest | Stata |
|---|:---:|:---:|:---:|:---:|:---:|
| Hausman (FE vs RE) | ❌ | ✅ | ✅ | ❌ | `xthausman` |
| F-test for pooling | ❌ | ✅ | ✅ | ❌ | `testparm` |
| LM test (Breusch-Pagan) | ❌ | ✅ | ✅ | ❌ | `xttest0` |
| Mundlak test | ❌ | ✅ | ❌ | ❌ | — |
| Pesaran CD | ❌ | ✅ | ✅ | ❌ | `xtcsd` |
| Wooldridge SC | ❌ | ✅ | ✅ | ❌ | `xtserial` |

### 1.4 Goodness of Fit

| Feature | statgpu | plm | linearmodels | fixest | Stata |
|---|:---:|:---:|:---:|:---:|:---:|
| Within R² | ✅ | ✅ | ✅ | ✅ | `e(r2_w)` |
| Overall R² | ❌ | ✅ | ✅ | ❌ | `e(r2)` |
| Between R² | ❌ | ✅ | ✅ | ❌ | `e(r2_b)` |
| Adjusted R² | ❌ | ✅ | ✅ | ✅ | `e(r2_a)` |
| F-statistic | ❌ | ✅ | ✅ | ✅ | `e(F)` |
| Log-likelihood | ❌ | ✅ | ❌ | ✅ | `e(ll)` |

### 1.5 Formula Interface

| Feature | statgpu | plm | linearmodels | fixest | Stata |
|---|:---:|:---:|:---:|:---:|:---:|
| Standard formula | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pipe syntax `\|` FE | ✅ | ❌ | ❌ | ✅ | — |
| Token syntax (EntityEffects) | ✅ | ❌ | ✅ | — | — |
| GPU acceleration | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 2. Prioritized Missing Features

### Tier 1 — Critical (core workflow, every user expects)

1. **Hausman test** — Choose between FE and RE
2. **Overall/Between R²** — Standard model fit metrics
3. **F-statistic** — Model significance
4. **LM test (Breusch-Pagan)** — RE vs pooled OLS
5. **Robust SE for RE** — Currently only nonrobust

### Tier 2 — High Priority (intermediate users)

6. **HC0/HC2/HC3 robust SEs** — Beyond HC1
7. **Panel IV (2SLS/GMM)** — Endogeneity
8. **Driscoll-Kraay SE** — Cross-section dependence
9. **High-dim FE (absorb)** — Sparse matrix techniques
10. **Adjusted R²** — Standard diagnostic

### Tier 3 — Medium Priority (advanced workflows)

11. **DID / Event Study** — Causal inference
12. **Dynamic Panel (Arellano-Bond)** — Lagged DV
13. **REML for RE** — Likelihood-based
14. **AR(1) errors** — Serial correlation correction
15. **Variance component estimators** — WH, AM, Nerlove

---

## 3. Unified Framework Architecture

### 3.1 Problem: Code Duplication

Current state: each model independently implements OLS solve, inference, and summary.

| Code | Duplicated in |
|---|---|
| OLS solve + inference | PooledOLS, BetweenOLS, FirstDifferenceOLS, FamaMacBeth |
| PanelSummary construction | All 6 models |
| Formula parsing | 6 models (slightly different patterns) |
| Covariance dispatch | PooledOLS, BetweenOLS, FirstDifferenceOLS |

### 3.2 Proposed Architecture

```
statgpu/panel/
├── _base.py              # BasePanelModel (template methods)
├── _covariance.py        # CovarianceEstimator registry
├── _formula.py           # Formula parsing (existing, extended)
├── _utils.py             # PanelSummary, demean, group ops (existing)
├── _fixed_effects.py     # PanelOLS / FixedEffects
├── _random_effects.py    # RandomEffects / RandomEffectsOLS
├── _pooled.py            # PooledOLS
├── _between.py           # BetweenOLS
├── _first_diff.py        # FirstDifferenceOLS
├── _fama_macbeth.py      # FamaMacBeth
├── _tests/               # Specification tests
│   ├── _hausman.py
│   ├── _pooling.py
│   └── _diagnostics.py
├── _iv.py                # IVPanelOLS [FUTURE]
├── _dynamic.py           # ArellanoBond [FUTURE]
└── _did.py               # DID estimators [FUTURE]
```

### 3.3 BasePanelModel Template

```python
class BasePanelModel(BaseEstimator):
    """Base class for all panel data models."""

    def __init__(self, cov_type='nonrobust', alpha=0.05, device='auto', ...):
        ...

    def fit(self, X=None, y=None, formula=None, data=None, **kwargs):
        # 1. Parse formula or arrays
        # 2. Validate inputs
        # 3. Transform data (subclass hook)
        # 4. Estimate (subclass hook)
        # 5. Inference (shared)
        ...

    def _transform_data(self, X, y, **kwargs):
        """Override in subclass. Return (X_transformed, y_transformed)."""
        raise NotImplementedError

    def _estimate(self, X, y, **kwargs):
        """Override in subclass. Return (params, resid, scale)."""
        raise NotImplementedError

    def _compute_inference(self, X, resid, params, scale, ...):
        """Shared inference logic. Delegates to CovarianceEstimator."""
        cov = self._get_covariance_estimator()
        V = cov.compute(X, resid, ...)
        bse = sqrt(diag(V))
        t = params / bse
        p = 2 * t_dist.sf(|t|, df)
        CI = params ± t_crit * bse
        ...

    def predict(self, X): ...
    def summary(self): ...
```

### 3.4 CovarianceEstimator Registry

```python
class CovarianceEstimator:
    def compute(self, X, resid, **kwargs) -> V: ...

class NonrobustCov(CovarianceEstimator):
    def compute(self, X, resid, scale, **kwargs):
        return scale * inv(X'X/n)

class RobustCov(CovarianceEstimator):
    def __init__(self, hc_type='HC1'): ...
    def compute(self, X, resid, **kwargs):
        # HC0-HC5 formulas

class ClusteredCov(CovarianceEstimator):
    def __init__(self, clusters, twoway=False): ...

class HACCov(CovarianceEstimator):
    def __init__(self, bandwidth=None, kernel='bartlett'): ...

class DriscollKraayCov(CovarianceEstimator):
    def __init__(self, bandwidth=None): ...
```

### 3.5 Specification Tests

```python
class HausmanTest:
    """Hausman test for FE vs RE."""
    def __init__(self, fe_model, re_model): ...
    def run(self) -> TestResult: ...

class LMTest:
    """Breusch-Pagan LM test for random effects."""
    def __init__(self, pooled_model, resid): ...
    def run(self) -> TestResult: ...

class FTest:
    """F-test for pooling / FE significance."""
    def __init__(self, restricted, unrestricted): ...
    def run(self) -> TestResult: ...
```

### 3.6 Implementation Roadmap

| PR | Content | Priority |
|---|---|---|
| PR-A | `BasePanelModel` refactor + `CovarianceEstimator` registry | Tier 1 |
| PR-B | Hausman test, LM test, F-test, R² variants | Tier 1 |
| PR-C | HC0/HC2/HC3, Driscoll-Kraay | Tier 2 |
| PR-D | Panel IV (2SLS/GMM) | Tier 2 |
| PR-E | High-dim FE (absorb) | Tier 2 |
| PR-F | DID / Event Study | Tier 3 |
| PR-G | Dynamic Panel (AB/GMM) | Tier 3 |

### 3.7 Key Design Principles

1. **Eliminate duplication**: `_compute_inference` lives in `BasePanelModel`, not in each subclass
2. **Pluggable covariance**: String `cov_type` for backward compat, `CovarianceEstimator` object for advanced use
3. **GPU-native**: All computations on-device, only final scalars transferred to CPU
4. **Formula-first**: Formula interface is the primary API, arrays are the fallback
5. **Lazy effects**: Entity/time effects computed on demand, not eagerly in `fit()`
