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
    ├── Family (loss function)
    │   ├── SquaredError, Logistic, Poisson, Gamma
    │   ├── Tweedie, NegativeBinomial, InverseGaussian
    │   └── Custom via GLMLoss
    │
    ├── Link (transformation)
    │   ├── Identity, Logit, Log, Inverse, Cloglog
    │   └── Custom via Link
    │
    ├── Penalty (regularization)
    │   ├── L1, L2, ElasticNet
    │   ├── SCAD, MCP (non-convex)
    │   ├── Adaptive L1
    │   └── Group Lasso, Group SCAD/MCP
    │
    └── Solver (optimization)
        ├── IRLS (iteratively reweighted least squares)
        ├── FISTA / FISTA-BB (proximal gradient)
        ├── ADMM (alternating direction method)
        ├── L-BFGS (limited-memory BFGS)
        └── Newton (full Newton)
```

Each solver handles smooth + non-smooth terms differently:
- **IRLS**: Works with any penalty via proximal operator
- **FISTA**: Async GPU loop (v22e), deferred convergence checks
- **L-BFGS**: Fused penalty gradient (v23c fix)
- **ADMM**: Dual decomposition with penalty splitting

### 4. Survival Analysis

Cox PH uses custom CUDA kernels for Efron's method:
- `_cox_efron_cuda.py`: CuPy RawKernel for tied failure times
- `_cox_efron_triton.py`: Triton kernel alternative
- CPU fallback uses scipy

### 5. Inference Module

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
│   ├── _utils.py       # Cross-backend helpers (DLPack, etc.)
│   └── _array_ops.py   # Functional dispatch (_xp, _sigmoid, etc.)
├── linear_model/       # Ridge, Lasso, ElasticNet, Logistic, GLM, Ordered
├── glm_core/           # Families, links, solvers
├── penalties/          # Penalty registry + implementations
├── survival/           # CoxPH + CUDA kernels
├── inference/          # Distributions, p-value adjustment, bootstrap
├── unsupervised/       # PCA, KMeans, DBSCAN, tSNE, UMAP, NMF, GMM
├── panel/              # PanelOLS, RandomEffects
├── nonparametric/      # KDE, kernel regression, splines
├── feature_selection/  # Knockoff, stepwise
├── covariance/         # LedoitWolf, OAS
├── anova/              # f_oneway
├── metrics/            # ROC, AUC, confusion matrix
├── diagnostics/        # Regression diagnostics
├── semiparametric/     # GAM
└── core/               # Formula parser, design matrix
```
