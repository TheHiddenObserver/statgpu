# How statgpu Cross-Validation Works

> Language: English  
> Last updated: 2026-09-17  
> This page: public design and execution model for cross-validation  
> Switch: [Chinese](../../cn/guides/cross-validation-design.md)

## Why this page exists

The [Cross-Validation guide](cross-validation.md) explains **how to configure and use** statgpu CV estimators. This page explains the design behind that interface: how selection and final refitting are separated, where pathwise and GPU acceleration can enter, what a selection cache is allowed to reuse, and which properties remain invariant when the implementation chooses a faster execution strategy.

This is a public design document, not an implementation reference. Private helper names, cache-key fields, backend thresholds, benchmark-derived cutoffs, and validation artifacts belong in the repository's internal design and validation documentation.

## 1. The execution model

A CV estimator can be understood as a model-selection layer around a base estimator:

```text
model + tuning space + folds
          |
          v
  candidate construction
          |
          v
  training-fold fitting
          |
          v
   held-out scoring
          |
          v
      selection
          |
          v
 full-data final refit
          |
          v
 prediction / diagnostics / optional inference
```

The important distinction is between **selection work** and the **published fitted model**.

During selection, statgpu may fit many temporary models. Those fits exist to compare candidate tuning configurations. After a candidate is selected, statgpu fits the selected configuration again on all observations. Public fitted coefficients, predictions, ordinary fitted-model diagnostics, and supported coefficient inference belong to that final refit rather than to an arbitrary fold fit.

## 2. Statistical invariants versus execution freedom

Acceleration is allowed to change **how** the same CV problem is evaluated, but not silently change **which statistical problem** is being solved.

For a fixed user request, the following are part of the public statistical design:

- the candidate tuning values after public validation;
- the train/validation folds;
- the loss or validation criterion associated with the model;
- analytic-weight semantics where weights are supported;
- the candidate-selection rule;
- the selected configuration used for the full-data refit;
- explicit solver and device requests where the requested combination is supported.

By contrast, the implementation may choose among equivalent execution strategies such as:

- reusing factorizations or sufficient statistics across candidates;
- warm-starting nearby tuning values;
- evaluating several folds or candidates together on an accelerator;
- reusing previously computed selection evidence through a cache;
- choosing a backend automatically when `device="auto"` is requested.

Those strategies are performance choices. Applications should not depend on a specific batching shape, cache implementation, private fast-path name, or automatic-routing threshold.

## 3. Why selection and final refit are separate stages

Cross-validation answers a tuning question using held-out data. The final estimator answers a fitting question using all available observations after tuning is complete.

That separation explains several public behaviors:

- `alpha_` and similar attributes describe the **selected tuning configuration**;
- `coef_`, `intercept_`, prediction, and ordinary final diagnostics describe the **full-data refit**;
- inference, when supported, is performed after selection on the final refit rather than independently inside every fold;
- a CV estimator can legitimately expose one solver control for the selection path and another for the final refit.

`LassoCV` is the clearest example. `cv_solver` controls the CV path, while `solver` controls the final full-data `Lasso` fit. The two stages can use different algorithms without changing the statistical meaning of the selected `alpha`, provided each stage follows its documented objective and solver contract.

## 4. Candidate grids and folds are first-class design inputs

A CV result is meaningful only relative to the candidate set and folds that were actually evaluated.

When an estimator generates a tuning grid automatically, the grid is data/model dependent. When the user supplies a grid, that grid becomes the requested candidate set after public validation.

Similarly, folds are not merely an implementation loop. They encode the resampling design. Time-ordered, grouped, clustered, or survival data may require custom splitting rules. statgpu validates the shape and estimator-specific requirements of supplied splits, but the scientific appropriateness of those splits remains the user's modeling decision.

Conceptually, once folds and candidates are established for one fit, later acceleration should operate on that same selection problem rather than repeatedly redefining it.

## 5. Pathwise reuse and warm starts

Many regularization problems are solved over an ordered path of tuning values. Nearby candidates often have nearby solutions, so recomputing every candidate from an unrelated initial point can waste substantial work.

A CV implementation may therefore reuse information across a path, for example:

- a previous candidate's coefficient vector as the next candidate's starting point;
- a decomposition or cross-product that does not change across candidates within a fold;
- other fold-local numerical state that is valid for the same statistical objective.

This is an optimization of repeated computation, not a different CV method. Candidate scores and the final selection must still correspond to the declared candidate configurations.

For non-convex penalties or model-specific losses, the appropriate continuation/path strategy can differ from ordinary convex L1/L2 paths. Those model-specific numerical details belong to the corresponding model and solver documentation; this page only describes the common design principle.

## 6. GPU batching: accelerate repeated work, not redefine CV

A naive implementation may evaluate every `(fold, candidate)` pair through a separate sequence of small operations. On a GPU, that can leave much of the device underutilized and introduce unnecessary host/device synchronization.

Where the numerical structure permits it, statgpu may combine repeated work into larger backend-native operations. Conceptually this can include batching across folds, candidates, or both.

The public contract is not a particular batching layout. The invariant is that acceleration must preserve:

- the same effective candidate set;
- the same fold membership;
- the same model objective and validation criterion;
- the same weight semantics;
- the same selection rule;
- the same full-data final-refit interpretation.

Consequently, a future version may change its batching strategy without requiring user code to change.

## 7. Selection caches

Some CV workloads repeatedly ask the same **selection question**. Recomputing all fold/candidate scores can be unnecessary when the inputs and every selection-relevant control are unchanged.

A selection cache may reuse previously computed selection evidence under that condition. Its public semantics are deliberately narrow:

- it accelerates candidate selection;
- it does **not** replace the final full-data fitted estimator with a cached estimator;
- changing selection-relevant data, folds, tuning values, weighting, or numerical controls must not incorrectly reuse incompatible evidence;
- mutating one returned result must not corrupt later cached results;
- using a cache must not change the statistical interpretation of the selected configuration.

The exact cache identity, capacity, hashing strategy, and private helper functions are implementation details and are not public API guarantees.

## 8. Device selection during CV

Explicit device requests and automatic device selection serve different purposes.

With an explicit request (`device="cpu"`, `"cuda"`, or `"torch"`), the estimator follows the documented backend contract and reports an error when the requested accelerator path is unavailable or unsupported. CV acceleration does not have permission to silently move an explicit GPU request to a CPU fit merely because a CPU path would be easier to execute.

With `device="auto"`, statgpu may choose a backend using availability and workload characteristics. The exact crossover thresholds are performance tuning parameters, so they may change as kernels and hardware support improve.

The important design rule is that backend selection should change **where/how** an eligible CV problem is evaluated, not silently substitute another loss, penalty, candidate set, or validation criterion.

## 9. Weights must remain consistent across the CV lifecycle

Where `sample_weight` is supported, weighting is part of the statistical problem rather than a post-processing option.

A consistent weighted CV lifecycle is:

```text
full-data weights
      |
      +--> training-fold weights -> weighted candidate fit
      |
      +--> validation-fold weights -> weighted held-out score
      |
      `--> full-data weights -> selected final refit
```

The exact normalization is defined by the corresponding estimator/loss contract. An acceleration path is valid only if it preserves that same convention. Unsupported weighted combinations should report that limitation rather than silently discarding weights.

## 10. Why Cox CV is structurally different

Cox cross-validation cannot always be treated as ordinary scalar-response regression with a different loss name. Held-out partial-likelihood evidence depends on survival targets, event support, and risk-set semantics.

As a result, Cox paths may need a different preparation order and additional fold validation before candidate scoring. The common selection/refit model still applies, but the survival-specific definition of valid folds and scores belongs to the [Cox Proportional Hazards](../models/coxph.md) documentation.

This is an example of a broader design rule: shared CV infrastructure should not erase model-specific statistical structure merely to force every estimator through one identical implementation path.

## 11. Inference after tuning

When coefficient inference is supported, statgpu performs it on the selected final refit after tuning is complete.

That design avoids the meaningless alternative of publishing a separate coefficient-inference result for every temporary fold fit. It also means that ordinary intervals and p-values after CV are generally conditional on the selected tuning configuration unless a specific inference method explicitly adjusts for tuning/selection uncertainty.

See [Inference Modes](inference-modes.md) and [Penalized GLM inference](penalized-glm-inference.md) for the inferential target and limitations.

## 12. What users can and cannot rely on

| Stable public idea | Implementation detail that may change |
|---|---|
| selection is followed by a full-data refit | exact private call graph |
| explicit supported solver/device requests remain authoritative | automatic device thresholds |
| folds/candidates/weights define the statistical selection problem | batching dimensions and kernel fusion |
| `cv_solver` and final-refit `solver` may represent different stages | internal warm-start storage |
| caches may reuse selection evidence without replacing the final refit | cache key fields, capacity, hash method |
| model-specific CV semantics remain model-specific | which internal fast path evaluates them |

This distinction lets statgpu improve performance without turning every optimization choice into a permanent user-facing API commitment.

## 13. Documentation map

Use the CV documentation according to the question:

- **How do I configure and use CV?** → [Cross-Validation](cross-validation.md)
- **How is statgpu CV conceptually organized and accelerated?** → this page
- **Which loss × penalty × solver combinations are available?** → [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md)
- **How does a solver itself work?** → [Solver Algorithms](solver-algorithms.md)
- **What does inference after selection mean?** → [Inference Modes](inference-modes.md)
- **What is special about survival CV?** → [Cox Proportional Hazards](../models/coxph.md)

Repository-internal call graphs, private cache contracts, heuristic thresholds, and validation/evidence live under `dev/` and are intentionally not part of this public design page.
