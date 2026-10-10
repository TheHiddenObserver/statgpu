# Loss × Penalty × Solver Framework

> Language: English  
> Last updated: 2026-09-17  
> This page: Architecture guide  
> Switch: [Chinese](../../cn/guides/loss-penalty-solver-framework.md)

## Overview

statgpu separates the public statistical API from the numerical components used to fit a model:

- a **Model** owns the user-facing statistical contract and fit lifecycle;
- a **Loss** defines the data-fit term and the numerical primitives available from it;
- a **Penalty** defines regularization and its numerical primitives;
- a **Solver** consumes those primitives and performs optimization;
- a **Backend** supplies NumPy, CuPy, or Torch execution across the numerical layers;
- a **CV/meta-estimator** repeatedly reconstructs and evaluates compatible fits before a final refit.

This page explains how those pieces combine to fit a model and how their compatibility affects solver choice.

Use the following references for those details:

- [Loss Functions](../models/losses.md) — loss definitions and loss-level numerical properties
- [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md) — actual loss × penalty × solver routes
- [Solver Algorithms](solver-algorithms.md) — update equations, convergence behavior, and algorithmic assumptions
- model pages such as [GeneralizedLinearModel](../models/generalized-linear-model.md), [Quantile Regression](../models/quantile.md), [Robust Regression](../models/robust.md), and [CoxPH](../models/coxph.md) — model-specific statistical behavior

## 1. Runtime architecture

A typical fit follows this structure:

```text
User
  │
  │  estimator = Model(...)
  │  estimator.fit(X, y, sample_weight=...)
  ▼
Model / public API
  │
  ├── parse and validate data / formula inputs
  ├── resolve backend and device
  ├── construct the Loss
  ├── construct the Penalty
  ├── validate or resolve the Solver
  ├── prepare intercept / initialization / fit-local state
  │
  ▼
Optimization problem
  │
  │               F(β) = L(β) + P(β)
  │                      ▲       ▲
  │                      │       │
  │                    Loss   Penalty
  ▼
Solver
  │
  ├── evaluate the primitives required by its algorithm
  ├── iterate on the selected numerical backend
  ├── apply stopping / convergence rules
  │
  ▼
Fitted-model state
  │
  ├── coefficients / intercept
  ├── convergence and solver provenance
  ├── inference or CV state when the estimator exposes it
  └── prediction / scoring interface
```

The model class is the orchestration boundary. A loss, penalty, or low-level solver object can be useful independently, but its existence does not by itself establish a complete public estimator route.

## 2. Component responsibilities

| Component | Owns | Does not by itself establish |
|---|---|---|
| **Model / estimator** | public parameters, data validation, formula handling, loss/penalty construction, solver selection, fitted state, prediction, estimator-level inference | low-level algorithm equations |
| **Loss** | data-fit term `L(β)`, value/gradient and optional curvature or specialized primitives | which penalties or public estimators are supported |
| **Penalty** | regularization term `P(β)`, gradients/proximal maps/LLA or group metadata as applicable | whether a loss provides the primitives required by a solver |
| **Solver** | numerical update rule, line search/step policy, stopping and convergence behavior | statistical meaning of a loss, weight, estimand, or CV procedure |
| **Backend** | array type, device, linear algebra and backend-native numerical operations | statistical support for a route |
| **CV/meta-estimator** | fold-local reconstruction, scoring, selection, and final refit | permission to bypass an unsupported direct-fit contract |

This separation is important when reading APIs. For example, a solver function may accept an argument named `sample_weight`, but the complete route is supported only when the loss, solver, and estimator all define compatible semantics for that argument.

## 3. Contract composition

The generic optimization problem is

$$
F(\beta)=L(\beta)+P(\beta).
$$

Different solvers require different primitives. Depending on the algorithm, a route may need some subset of

$$
L(\beta),\qquad
\nabla L(\beta),\qquad
\nabla^2 L(\beta),\qquad
P(\beta),\qquad
\nabla P(\beta),\qquad
\operatorname{prox}_{\gamma P}(v),
$$

plus loss- or penalty-specific operations such as an IRLS majorization, a local-linear surrogate, or structured/group metadata.

A public route is coherent only when all required pieces describe the **same objective and parameterization**. Having each method separately callable is not enough. In particular:

- gradient and objective evaluations must use the same scaling and weighting convention;
- a curvature-based method needs curvature consistent with the objective it is minimizing;
- a proximal algorithm needs a penalty representation compatible with its proximal or surrogate step;
- intercept treatment must remain consistent across loss, penalty, initialization, and solver updates;
- backend/device handling must preserve the same numerical problem rather than silently changing the route.

The exact route-level combinations are listed in the [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md).

## 4. Capability matching and solver dispatch

At a high level, model fitting resolves a route in the following order:

1. Normalize public aliases and validate model parameters.
2. Build the concrete loss and penalty objects.
3. Resolve the numerical backend/device.
4. If a solver was explicitly requested, validate that the complete loss × penalty × estimator route supports it.
5. If `solver="auto"` was requested, select the model's compatible automatic route.
6. Execute that solver without silently changing an explicit user request.
7. Record the resolved solver/backend information exposed by the estimator.

`solver="auto"` is therefore a **dispatch policy**, not a numerical algorithm. Its result can vary by loss, penalty, backend, and estimator type. Direct fitting and CV may also intentionally use different automatic routes because their computational workloads differ.

Internal resolved labels may identify specialized continuation or composite paths that are not valid public `solver=` values. The compatibility matrix distinguishes public requests from such resolved implementation routes.

For the exact current dispatch, use the [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md); for how each numerical method works, use [Solver Algorithms](solver-algorithms.md).

## 5. `sample_weight` and objective consistency

`sample_weight` is not a universal solver capability. Its meaning and availability are part of the **loss × solver × estimator** contract.

When a route uses normalized analytic objective weights, the data-fit term has the form

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

The denominator fixes the scale of the data-fit term relative to the penalty. Replacing this weighted mean with a weighted sum changes that balance unless the penalty is rescaled as well. The gradient and curvature of the displayed objective inherit the same normalization.

Consequently:

- accepting `sample_weight` in a shared method signature does not imply full weighted-estimator support;
- an explicit solver request is not changed merely because weights are supplied;
- weight restrictions also apply to low-level calls: for example, shared ADMM rejects non-uniform weights on its supported losses;
- common rescaling of weights is invariant only on routes whose declared objective has that normalization.

For the meaning and availability of weights in a particular model, see its model page. For loss-level weighting of first-order quantities, see [Loss Functions](../models/losses.md).

## 6. Backend and device boundary

Backend is a cross-cutting execution dimension rather than a separate statistical model layer:

```text
                  NumPy / CuPy / Torch
                ┌──────────────────────┐
Model      ──────┤ device selection     │
Loss       ──────┤ objective primitives │
Penalty    ──────┤ prox / gradients     │
Solver     ──────┤ iterative numerics   │
                └──────────────────────┘
```

In shared penalized GLM fitting, an explicit `device="cuda"` or `device="torch"` request requires the corresponding GPU backend and raises an error if that backend is unavailable. Use `device="auto"` to allow automatic backend selection on these paths. Other model families can have different device-selection rules; see [Device and Memory](device-and-memory.md) for model-specific choices and data-transfer considerations.

Backend support is route-specific: a solver being implemented for NumPy, CuPy, and Torch does not imply that every loss, estimator, or inference procedure supports all three. Check the model documentation for preprocessing, output placement, and supported devices.

## 7. CV and meta-estimator boundary

A CV/meta-estimator adds another orchestration layer around ordinary fits. Its responsibilities include:

- constructing fold-local estimator state;
- forming candidates with the chosen loss, penalty, and tuning parameters;
- applying fold-local training data and weights;
- evaluating the declared validation score;
- selecting tuning parameters;
- performing the selected full-data final refit with the intended route.

Loss, penalty, and solver compatibility still applies inside each candidate. In `PenalizedGLM_CV`, a supported explicit solver request applies to candidate fits and the selected full-data refit.

The automatic route used by CV can differ from direct-fit `solver="auto"` for performance reasons. Those differences are part of the compatibility reference and are listed in the [CV matrix](solver-penalty-matrix.md#4-cv-solverauto-penalizedglm_cv).

For strict versus approximate CV, scoring, continuation paths, and final-refit options, see [Cross-Validation](cross-validation.md) and the relevant model page.

## 8. Public estimator APIs versus low-level APIs

For normal use, model classes are the canonical entry point. Low-level loss, penalty, and solver APIs are also public where documented, but their contracts can be narrower or simply different from estimator-level dispatch.

Do not infer estimator support from any single low-level fact such as:

- a loss exposing `gradient()` or `hessian()`;
- a penalty exposing `prox()`;
- a solver function accepting a parameter in its signature;
- a low-level call succeeding for one backend or one unweighted case.

Conversely, an estimator can combine several lower-level pieces behind one resolved route. Public support is defined by the complete estimator contract and the compatibility matrix, not by one component in isolation.

<a id="9-documentation-ownership"></a>

## 9. Find the right reference

Use this table to find the reference that answers your question:

| Question | Canonical documentation |
|---|---|
| What is this loss mathematically? What primitives does it provide? | [Loss Functions](../models/losses.md) |
| Which loss × penalty combination uses which solver? | [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md) |
| How does Newton/FISTA/ADMM/etc. update the parameters? | [Solver Algorithms](solver-algorithms.md) |
| How do Loss, Penalty, Solver, Backend, and CV compose? | **This page** |
| What does a specific model support, and why? | the corresponding model page |
| What inference procedure/estimand is available? | [Inference Modes](inference-modes.md) and model-specific inference docs |

When choosing a solver for a particular model, read its model page alongside the compatibility matrix; the model page explains any additional input restrictions or statistical assumptions.

## See also

- [Loss Functions](../models/losses.md)
- [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md)
- [Solver Algorithms](solver-algorithms.md)
- [GeneralizedLinearModel](../models/generalized-linear-model.md)
- [Quantile Regression](../models/quantile.md)
- [Robust Regression](../models/robust.md)
- [CoxPH](../models/coxph.md)
- [Inference Modes](inference-modes.md)
