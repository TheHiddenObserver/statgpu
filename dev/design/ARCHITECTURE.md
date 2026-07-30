# statgpu Architecture

## Overview

statgpu is a GPU-accelerated statistics library that provides sklearn-style estimators with pluggable NumPy, CuPy, and Torch backends.

```
User Code
    │
    ▼
┌─────────────────────────────────────┐
│  Public API (__init__.py)           │
│  estimators, results, utilities     │
└──────────────┬──────────────────────┘
               │
    ▼──────────▼──────────▼
┌────────┐ ┌────────┐ ┌────────┐
│ Ridge  │ │ Lasso  │ │ CoxPH  │  ... (estimators)
│ _CV    │ │ _CV    │ │ _CV    │
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    ▼──────────▼──────────▼
┌─────────────────────────────────────┐
│  BaseEstimator (_base.py)           │
│  - Device management                │
│  - Backend selection                │
│  - Array conversion                 │
│  - sklearn-style parameters         │
└──────────────┬──────────────────────┘
               │
    ▼──────────▼──────────▼
┌────────┐ ┌────────┐ ┌────────┐
│ NumPy  │ │  CuPy  │ │ Torch  │  (backends)
│Backend │ │Backend │ │Backend │
└────────┘ └────────┘ └────────┘
```

## Core Design Decisions

### 1. Backend Abstraction

Computation should use `BackendBase`, the backend array namespace, or shared functional backend helpers. Estimators should not duplicate full NumPy, CuPy, and Torch implementations.

```python
class MyEstimator(BaseEstimator):
    def fit(self, X, y):
        backend = self._get_backend()
        xp = backend.xp
        X = backend.asarray(X)
        # Use xp.sum(), xp.linalg.solve(), and shared backend helpers.
```

**Why**: one statistical implementation can support NumPy CPU, CuPy CUDA, and Torch CUDA while preserving explicit device semantics.

### 2. Dual Backend Dispatch

Two dispatch patterns coexist:

- **OO dispatch**: `self._get_backend()` → `backend.xp.*`, used by estimators and public boundaries;
- **functional dispatch**: runtime array detection plus shared helpers, used by solvers, penalties, and statistical kernels.

Functional dispatch keeps performance-sensitive inner loops independent of estimator state. Direct imports of a concrete GPU framework should remain isolated to backend or explicitly specialized implementation modules.

### 3. GLM Solver Architecture

```
PenalizedGLM_CV
    │
    ├── Family (loss function)  →  glm_core/
    │   ├── SquaredError, Logistic, Poisson, Gamma
    │   ├── Tweedie, NegativeBinomial, InverseGaussian
    │   └── Custom via GLMLoss
    │
    ├── Link (transformation)
    │   ├── Identity, Logit, Log, Inverse, Cloglog
    │   └── Custom via Link
    │
    ├── Penalty (regularization)  →  penalties/
    │   ├── L1, L2, ElasticNet
    │   ├── SCAD, MCP (non-convex)
    │   ├── Adaptive L1
    │   └── Group Lasso, Adaptive Group Lasso, Group SCAD/MCP
    │
    └── Solver (optimization)  →  solvers/
        ├── fista_solver      — FISTA with backtracking line search
        ├── fista_bb_solver   — FISTA with Barzilai-Borwein step sizes
        ├── fista_lla_path    — FISTA+LLA for SCAD/MCP continuation paths
        ├── newton_solver     — Newton-Raphson with Armijo backtracking
        ├── lbfgs_solver      — Limited-memory BFGS
        ├── admm_solver       — ADMM with Nesterov-accelerated CG
        └── irls_solver       — IRLS in glm_core/_irls.py
```

Each solver handles smooth and non-smooth terms differently:

- **IRLS**: weighted quadratic updates plus penalty-specific proximal handling;
- **FISTA / FISTA-BB**: backend-native accelerated proximal iterations;
- **FISTA-LLA**: continuation and local-linear approximation for SCAD/MCP;
- **L-BFGS**: smooth loss and penalty gradients;
- **ADMM**: dual decomposition with an iterative linear-system subproblem;
- **Newton**: full Hessian with line search.

Automatic solver routing is implemented in the penalized fit layer and depends on the loss, penalty, backend, and problem contract.

### 4. linear_model Estimator Hierarchy

```
BaseEstimator
    │
    ├── LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
    │
    ├── GeneralizedLinearModel
    │   ├── LogisticRegression, LogisticRegressionCV
    │   ├── PoissonRegression, GammaRegression
    │   ├── InverseGaussianRegression, NegativeBinomialRegression, TweedieRegression
    │   └── PenalizedGeneralizedLinearModel
    │       ├── PenalizedLinearRegression
    │       ├── PenalizedLogisticRegression
    │       ├── PenalizedPoissonRegression, PenalizedGammaRegression
    │       ├── PenalizedInverseGaussianRegression
    │       ├── PenalizedNegativeBinomialRegression, PenalizedTweedieRegression
    │       ├── PenalizedCoxPHModel
    │       └── PenalizedGLM_CV
    │
    └── OrderedGeneralizedLinearModel
        ├── OrderedLogitRegression
        └── OrderedProbitRegression
```

### 5. Survival / Cox Architecture

Cox functionality is divided into two public product lines rather than one monolithic estimator.

| User need | Public estimator | Location | Current contract |
|---|---|---|---|
| Full Cox fitting, baseline prediction, formula support, and inference | `CoxPH` | `statgpu.survival` | Breslow/Efron/Exact, start-stop, strata, robust inference, NumPy/CuPy/Torch |
| L2 penalty selection by held-out partial likelihood | `CoxPHCV` | `statgpu.survival` | CV wrapper that selects a penalty and performs a final `CoxPH` refit |
| L1, L2, ElasticNet, SCAD, or MCP estimation | `PenalizedCoxPHModel` | `statgpu.linear_model` | generic penalized-solver path; currently estimation-only |

#### Canonical Cox call graph

```
CoxPH.fit
    │
    ├── public target/formula/device boundary      → _cox.py
    ├── fit-time control and label normalization  → _cox_fit_adapter.py
    ├── typed prepared input capability           → _cox_counting.py
    ├── Newton / line-search orchestration         → _cox_counting.py
    ├── risk sets, objective, score, information  → _risk_sets.py
    ├── Cox-specific covariance assembly           → _cox.py + _cox_inference.py
    ├── generic covariance spectrum / Wald policy → inference/_covariance.py
    └── fitted state, baseline, prediction, summary→ _cox.py / _cox_score.py / _numeric.py
```

`_risk_sets.py` is the canonical statistical-definition layer for delayed entry, `(start, stop]` counting-process rows, strata, tie handling, score residuals, baseline hazards, and counting-process concordance. Specialized ordinary-right-censored or accelerator kernels are optimization paths and must agree with these primitives; they do not define separate public Cox semantics.

#### Module ownership

| Module | Responsibility | Must not own |
|---|---|---|
| `survival/_cox.py` | Public `CoxPH` API, input orchestration, fitted-state transaction, covariance assembly, result publication | duplicated risk-set mathematics or framework-specific solver copies |
| `survival/_cox_fit_adapter.py` | fit-time normalization, packed targets, clone-safe controls, pre-encoded labels | objective or inference policy |
| `survival/_cox_counting.py` | prepared-state types, canonical Newton solver, line search, convergence and numerical-error routing | public formula/reporting APIs |
| `survival/_risk_sets.py` | backend-native Cox statistical primitives and correctness reference | estimator state or CV selection |
| `survival/_cox_inference.py` | independent-unit validation and backend-native observed-information inversion | generic covariance-spectrum or Wald policy |
| `inference/_covariance.py` | backend-neutral covariance classification, marginal SE validation, joint Wald availability | Cox score/meat construction |
| `survival/_cox_cv.py` | folds, caching, held-out partial likelihood, penalty selection, final refit propagation | a second Cox optimizer |
| `survival/_cox_score.py` / `_concordance.py` | public scoring boundary and bounded-memory concordance | fitting or covariance construction |
| `linear_model/penalized/_penalized_cox.py` | broad penalty registry and generic penalized solver integration | canonical Cox robust inference |
| `survival/_cox_legacy.py` and specialized kernels | regression references, compatibility, or measured fast paths | public capability definitions |

#### Prepared-state and fast-path rules

Ordinary right-censored Breslow/Efron data can reuse sorted failure-group state. Caller-owned prepared state validates both identity and content so in-place mutation cannot silently reuse stale preprocessing. CV-owned arrays use a separate typed capability because the CV layer controls their lifetime and may reuse preparation across a penalty path.

General delayed-entry, start-stop, stratified, and Exact cases use the counting-process primitives. Memory-bounded or specialized GPU routes are explicit algorithmic choices on the selected backend; an explicit `device="cuda"` or `device="torch"` request must not become an implicit CPU fallback.

#### Cox inference flow

```
observed information
    └── backend-native strict inverse → bread

counting-process score residuals
    └── subject / cluster aggregation → meat

bread @ meat @ bread
    └── p × p covariance transferred to host
        └── classify_covariance_spectrum
            ├── positive definite  → marginal inference + joint Wald
            ├── rank-deficient PSD → marginal inference; joint Wald unavailable
            └── materially indefinite → strict RuntimeError and fit-state reset
```

Cox-specific score residuals and independent-unit aggregation remain in the survival layer. Distribution functions, result containers, covariance-spectrum classification, and joint-Wald policy are shared through `statgpu.inference`.

#### CoxPHCV reuse contract

`CoxPHCV` is a model-selection wrapper, not a separate estimator implementation:

```
CoxPHCV.fit
    ├── construct or validate folds
    ├── prepare fold state on the requested backend
    ├── fit candidate L2 penalties
    ├── score held-out partial likelihood
    ├── select the best eligible penalty
    └── CoxPH(penalty=best_penalty).fit(full data)
```

The final `CoxPH` estimator owns coefficient, convergence, prediction, and inference semantics. A failed final refit resets both final-estimator state and partially published CV state.

#### Extension rules

When extending Cox functionality:

1. Put new risk-set or counting-process mathematics in `_risk_sets.py` unless it is a proven specialization of an existing primitive.
2. Validate any specialized kernel against the canonical primitives on NumPy, CuPy, and Torch.
3. Reuse backend abstractions and shared inference distributions/results; do not reproduce framework branches in `_cox.py`.
4. Add tunable canonical Cox behavior to `CoxPHCV` through candidate fitting and final `CoxPH` refit rather than a second solver.
5. Preserve strict inference: invalid covariance or information must fail transactionally unless the public API explicitly exposes a documented downgrade.
6. Keep `PenalizedCoxPHModel` and canonical `CoxPH` capability claims distinct until broad-penalty Cox inference has a validated statistical contract.

### 6. Inference Module

Shared inference infrastructure includes:

- backend-aware distribution functions;
- reusable result containers such as `ParameterInferenceResult`;
- covariance, resampling, and multiple-testing utilities;
- backend-neutral covariance-spectrum and joint-Wald policy.

Model-specific score construction, estimating equations, and independent-unit semantics remain in the corresponding model module.

## Data Flow

```
Input arrays / formula data
    │
    ▼
Public estimator boundary
    │  validate, normalize, select backend
    ▼
Statistical objective + solver
    │  remain on selected backend where supported
    ▼
Inference result / fitted state
    │
    ▼
.predict(...) / .score(...) / .summary()
```

## File Organization

```
statgpu/
├── __init__.py         # Public API
├── _config.py          # Device enum and global device selection
├── _base.py            # BaseEstimator
├── backends/
│   ├── _base.py        # BackendBase
│   ├── _numpy.py       # NumPy CPU backend
│   ├── _cupy.py        # CuPy CUDA backend
│   ├── _torch.py       # Torch backend
│   ├── _factory.py     # backend factory
│   ├── _utils.py       # cross-backend validation and conversion helpers
│   └── _array_ops.py   # functional backend operations
├── solvers/            # Generic loss-aware / penalty-aware solvers
├── cross_validation/   # Shared folds, cache, and CV infrastructure
├── linear_model/
│   ├── wrappers/       # Thin public regression and GLM wrappers
│   ├── penalized/      # Penalized GLM mixins and PenalizedCoxPHModel
│   ├── cv/             # Linear/GLM CV wrappers
│   ├── legacy/         # Compatibility implementations
│   ├── _glm_base.py
│   └── _gaussian_inference.py
├── glm_core/           # GLM families, links, losses, and IRLS
├── penalties/          # Penalty registry and implementations
├── survival/
│   ├── _cox.py              # Public canonical CoxPH orchestration
│   ├── _cox_cv.py           # L2 penalty selection and final CoxPH refit
│   ├── _cox_counting.py     # Prepared states and canonical Newton solver
│   ├── _risk_sets.py        # Cox statistical-definition primitives
│   ├── _cox_inference.py    # Cox-specific inversion and unit validation
│   ├── _cox_score.py        # Public scoring boundary
│   ├── _concordance.py      # Shared concordance implementation
│   ├── _cox_fit_adapter.py  # Fit-time input/control normalization
│   ├── _numeric.py          # Shared prediction numerical boundaries
│   ├── _cox_errors.py       # Cox numerical error types
│   └── specialized/legacy kernels and references
├── inference/          # Distributions, result containers, covariance policy, resampling
├── unsupervised/       # PCA, clustering, decomposition, manifold learning
├── panel/              # Panel-data models
├── nonparametric/      # Kernel smoothing, kernel methods, splines
├── feature_selection/  # Knockoffs and stepwise selection
├── covariance/         # Covariance estimators
├── anova/              # ANOVA methods
├── metrics/            # Metrics
├── diagnostics/        # Regression diagnostics
├── semiparametric/     # GAM
└── core/formula/       # R-style formula parser and design matrices
```
