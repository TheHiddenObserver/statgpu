# statgpu Architecture

## Overview

statgpu is a GPU-accelerated statistics library that provides sklearn-compatible estimators with transparent GPU acceleration via pluggable backends (CuPy, PyTorch).

```
User Code
    │
    ▼
┌─────────────────────────────────────┐
│  Public API (__init__.py)           │
│  ~60 exports: estimators, utils     │
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
│  - sklearn get_params/set_params    │
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

All computation goes through `BackendBase` subclasses. Estimators never import numpy/cupy/torch directly — they use `backend.xp.*` which maps to the correct array library.

```python
class MyEstimator(BaseEstimator):
    def fit(self, X, y):
        backend = self._get_backend()
        xp = backend.xp
        X = backend.asarray(X)
        # Use xp.sum(), xp.linalg.solve(), etc.
```

**Why**: Single codebase supports CPU (NumPy), GPU via CuPy, and GPU via PyTorch without code duplication.

### 2. Dual Backend Dispatch

Two dispatch patterns coexist:

- **OO dispatch**: `self._get_backend()` → `backend.xp.*` — used by estimators
- **Functional dispatch**: `_xp(arr)` runtime detection — used by solvers and penalties for performance-critical inner loops

**Why**: Functional dispatch avoids method call overhead in tight loops (FISTA iterations, IRLS steps).

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
    └── Solver (optimization)  →  solvers/ (generic, loss-agnostic)
        ├── fista_solver      — FISTA with backtracking line search
        ├── fista_bb_solver   — FISTA with Barzilai-Borwein step sizes
        ├── fista_lla_path    — FISTA+LLA for SCAD/MCP continuation paths
        ├── newton_solver     — Newton-Raphson with Armijo backtracking
        ├── lbfgs_solver      — Limited-memory BFGS
        ├── admm_solver       — ADMM with Nesterov-accelerated CG
        └── irls_solver       — IRLS (in glm_core/_irls.py, self-contained)
```

Each solver handles smooth + non-smooth terms differently:
- **IRLS**: Works with any penalty via proximal operator; self-contained in glm_core
- **FISTA / FISTA-BB**: Async GPU loop, deferred convergence checks, fused element-wise kernels
- **FISTA-LLA**: Continuation path for non-convex penalties (SCAD/MCP), per-alpha warm-start
- **L-BFGS**: Fused penalty gradient
- **ADMM**: Dual decomposition with Nesterov-accelerated CG subproblem
- **Newton**: Full Hessian with Armijo backtracking

Solver dispatch (`solver='auto'`) uses a priority table in `_fit_mixin.py` that selects
the optimal solver based on (loss, penalty, backend, l1_ratio, cv_mode, problem_size).

### 4. linear_model Estimator Hierarchy

```
BaseEstimator
    │
    ├── LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
    │
    ├── GeneralizedLinearModel (base for all GLMs)
    │   ├── LogisticRegression, LogisticRegressionCV
    │   ├── PoissonRegression, GammaRegression
    │   ├── InverseGaussianRegression, NegativeBinomialRegression, TweedieRegression
    │   └── PenalizedGeneralizedLinearModel (base for penalized GLMs)
    │       ├── PenalizedLinearRegression
    │       ├── PenalizedLogisticRegression
    │       ├── PenalizedPoissonRegression, PenalizedGammaRegression
    │       ├── PenalizedInverseGaussianRegression
    │       ├── PenalizedNegativeBinomialRegression, PenalizedTweedieRegression
    │       └── PenalizedGLM_CV (full CV over families × penalties × solvers)
    │
    └── OrderedGeneralizedLinearModel (base for ordered models)
        ├── OrderedLogitRegression
        └── OrderedProbitRegression
```

### 5. Survival Analysis

Cox PH uses custom CUDA kernels for Efron's method:
- `_cox_efron_cuda.py`: CuPy RawKernel for tied failure times
- `_cox_efron_triton.py`: Triton kernel alternative
- CPU fallback uses scipy

### 6. Inference Module

Shared across all estimators:
- Distribution backends (norm, t, chi2, F, beta, gamma)
- Multiple testing correction (Bonferroni, BH, Holm, etc.)
- Bootstrap and permutation tests
- Result classes with automatic formatting

## Data Flow

```
Input: X (n×p), y (n,)
    │
    ▼
BaseEstimator._to_array(X)  →  Convert to backend array
    │
    ▼
Solver.fit(X, y, penalty)   →  Iterative optimization
    │                          (all on GPU if available)
    ▼
InferenceResult              →  SE, p-values, CI
    │
    ▼
.predict(X_new) / .summary()
```

## File Organization

```
statgpu/
├── __init__.py         # Public API (~60 exports)
├── _config.py          # Device enum + manager singleton
├── _base.py            # BaseEstimator ABC
├── backends/
│   ├── _base.py        # BackendBase ABC
│   ├── _numpy.py       # NumpyBackend (CPU)
│   ├── _cupy.py        # CuPyBackend (GPU)
│   ├── _torch.py       # TorchBackend (GPU/CPU)
│   ├── _factory.py     # get_backend() factory
│   ├── _utils.py       # Cross-backend helpers (DLPack, xp_asarray, etc.)
│   ├── _array_ops.py   # Functional dispatch (_xp, _sigmoid, _abs_sum_dev, etc.)
│   ├── _torch_safe.py  # Resilient torch import (TORCH_LIBRARY conflict)
│   ├── _gpu_inference_cupy.py  # CuPy-specific inference acceleration
│   └── _gpu_inference_torch.py # Torch-specific inference acceleration
├── solvers/            # Generic loss-agnostic solvers (top-level module)
│   ├── _fista.py       # FISTA with backtracking line search
│   ├── _fista_bb.py    # FISTA with Barzilai-Borwein step sizes
│   ├── _fista_lla.py   # FISTA+LLA for SCAD/MCP continuation paths
│   ├── _newton.py      # Newton-Raphson with Armijo backtracking
│   ├── _lbfgs.py       # Limited-memory BFGS
│   ├── _admm.py        # ADMM with Nesterov-accelerated CG subproblem
│   ├── _utils.py       # Shared helpers (_nesterov_momentum, _call_with_weight, etc.)
│   ├── _constants.py   # Solver convergence constants and thresholds
│   └── _convergence.py # ConvergenceWarning
├── cross_validation/   # Shared CV infrastructure
│   ├── _base.py        # CVEstimatorBase, kfold_indices, hash_cv_data
│   └── _engine.py      # run_cv reference implementation
├── linear_model/
│   ├── wrappers/       # 13 model classes (thin wrappers over penalized GLM)
│   │   ├── _linear.py, _ridge.py, _lasso.py, _elasticnet.py
│   │   ├── _adaptive_lasso.py, _scad.py, _mcp.py
│   │   ├── _logistic.py, _poisson.py, _gamma.py
│   │   ├── _inverse_gaussian.py, _negative_binomial.py, _tweedie.py
│   │   └── _knockoff.py
│   ├── penalized/      # Mixin architecture for penalized GLM
│   │   ├── _base.py           # PenalizedGeneralizedLinearModel + SelectivePenalty
│   │   ├── _fit_mixin.py      # _fit_cpu, _fit_gpu_backend, _fit_loss_backend
│   │   ├── _inference_mixin.py # Debiased Lasso, Gaussian, bootstrap inference
│   │   ├── _predict_mixin.py  # predict, score, link-inverse dispatch
│   │   ├── _penalized_cv.py   # PenalizedGLM_CV (2700+ lines)
│   │   └── _penalized_*.py    # 7 family-specific subclasses
│   ├── cv/             # CV wrappers
│   │   ├── _lasso_cv.py, _ridge_cv.py, _elasticnet_cv.py, _logistic_cv.py
│   ├── legacy/         # Archived files (backward compatibility)
│   ├── _glm_base.py    # GeneralizedLinearModel base class
│   └── _gaussian_inference.py  # OLS inference utilities
├── glm_core/           # GLM families, links, and loss functions
│   ├── _base.py        # GLMLoss ABC + registry
│   ├── _fused.py       # Fused loss+gradient kernels (logistic, poisson, etc.)
│   ├── _irls.py        # IRLS solver (self-contained, backend-parameterized)
│   ├── _family.py      # Family classes (Binomial, Gaussian, Poisson, etc.)
│   ├── _squared.py, _logistic.py, _poisson.py, _gamma.py
│   ├── _inverse_gaussian.py, _negative_binomial.py, _tweedie.py
├── penalties/          # Penalty registry + implementations
├── survival/           # CoxPH + CUDA kernels
├── inference/          # Distributions, p-value adjustment, bootstrap
├── unsupervised/       # PCA, KMeans, DBSCAN, tSNE, UMAP, NMF, GMM
├── panel/              # PanelOLS, RandomEffects
├── nonparametric/      # KDE, kernel regression, splines
│   ├── kernel_smoothing/   # KDE, bandwidth selection
│   ├── kernel_methods/     # KernelRidge, KernelRidgeCV, pairwise_kernels
│   └── splines/            # B-spline, natural cubic spline basis
├── feature_selection/  # KnockoffSelector, FixedXKnockoffSelector, StepwiseSelector
├── covariance/         # LedoitWolf, OAS
├── anova/              # f_oneway
├── metrics/            # ROC, AUC, confusion matrix
├── diagnostics/        # Regression diagnostics
├── semiparametric/     # GAM
├── core/
│   └── formula/        # R-style formula parser, design matrix, terms
├── kernel_methods/     # Backward-compat shim → nonparametric.kernel_methods
└── splines/            # Backward-compat shim → nonparametric.splines + GAM
```
