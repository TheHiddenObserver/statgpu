# Penalized Robust and Quantile Dashboard Integration Plan

## 1. Decision summary

The recommended implementation keeps penalized robust and quantile regression inside the existing `robust_quantile` category and adds penalty as a first-class benchmark dimension.

The first production source should use a **staged core matrix**, rather than attempting the complete loss × penalty × solver × scale cross-product in one benchmark.

Phase 1 should cover:

- `QuantileRegression`, with median loss (`quantile=0.5`);
- `RobustRegression`, with Huber loss (`epsilon=1.35`, `method=MAD`);
- penalties `none`, `l1`, `l2`, `elasticnet`, `scad`, and `mcp`;
- scales `5K×50`, `20K×100`, `100K×50`, and `5K×500`;
- NumPy, CuPy, and Torch;
- `solver=auto`, while recording the resolved solver/implementation;
- timing, convergence, objective/KKT diagnostics, and backend agreement;
- narrowly aligned external references only where the objective is genuinely comparable.

This produces 144 statgpu timing rows before external-reference rows:

```text
2 model/loss variants × 6 penalties × 4 scales × 3 backends = 144 rows
```

The source should be generated on the canonical P100 environment and registered as a new June-or-later canonical artifact, for example:

```text
results/benchmark_frontend_sources/penalized_robust_quantile_202607xx.json
```

## 2. Current gap

The current June 23 loss-function source benchmarks only unpenalized Quantile and Huber fits. Its parser emits `penalty=None` and declares both frontend models as not supporting penalty.

The implementation itself already provides:

- `PenalizedQuantileRegression`;
- `PenalizedRobustRegression`;
- L1, L2, ElasticNet, SCAD, MCP, group, and adaptive penalties;
- backend-aware solver dispatch.

Therefore the missing work is not a frontend-only field change. A new benchmark source is required because penalized timing, convergence, and accuracy cannot be inferred from the existing unpenalized rows.

## 3. Option comparison

### Option A — relabel current rows as penalized

Change the current parser from `penalty=None` to a default such as `l2`.

**Advantages**

- Minimal implementation effort.
- The penalty filter would appear immediately.

**Disadvantages**

- Statistically incorrect: the existing benchmark used no effective regularization.
- Misrepresents timing and solver behavior.
- Creates false comparison groups and invalid speedups.
- Violates provenance and source-truth requirements.

**Decision:** reject.

### Option B — place the new rows in `penalized_glm`

Treat penalized Quantile and Robust models as part of the Penalized GLM category.

**Advantages**

- Reuses the existing penalty-heavy dashboard path.
- Makes all penalized models appear in one category.

**Disadvantages**

- Quantile, Huber, Bisquare, and Fair are `LossBase` losses, not GLM losses.
- Category semantics become statistically misleading.
- The existing `robust_quantile` category remains incomplete.
- Users would need to switch categories to compare unpenalized and penalized versions of the same model.

**Decision:** reject.

### Option C — create a new `penalized_robust_quantile` category

Add a thirteenth sidebar category dedicated to penalized Robust/Quantile methods.

**Advantages**

- Clear separation between penalized and unpenalized data.
- Focused default presentation is easy to define.

**Disadvantages**

- Duplicates the same model families across two categories.
- Prevents direct `none` versus penalized comparison in one view.
- Expands sidebar and documentation complexity.
- Requires a schema/category registry change without adding a genuinely new statistical domain.

**Decision:** reject for Phase 1.

### Option D — add penalty inside the existing `robust_quantile` category

Keep the same category and model IDs, with `penalty`, `parameters`, and resolved solver identity distinguishing configurations.

**Advantages**

- Statistically coherent: category represents the loss/model family, penalty is a method dimension.
- Supports direct comparison of `none`, convex penalties, and nonconvex penalties.
- Reuses the current progressive filter order.
- Requires no Schema v1.1 change.
- Keeps the sidebar compact.
- Extends naturally to later Quantile levels and Robust loss variants.

**Disadvantages**

- The category can become dense if the full matrix is added immediately.
- Focused-view policy must prevent information overload.
- Method identity must include all regularization parameters.

**Decision:** select.

## 4. Benchmark-scope comparison

### Scope 1 — exhaustive matrix

Benchmark all supported losses, penalties, explicit solvers, scales, backends, quantiles, Robust tuning methods, and group definitions.

A representative lower bound would already exceed:

```text
4 losses × 10 penalties × 6 solvers × 4 scales × 3 backends = 2,880 configurations
```

This excludes seeds, repeats, quantile levels, group structures, and failure retries.

**Advantages**

- Maximum coverage.
- Useful as a solver-development stress suite.

**Disadvantages**

- High GPU time and high failure/debugging cost.
- Many combinations are invalid or redundant under solver dispatch.
- Produces a dashboard that is difficult to interpret.
- Makes stable external comparison almost impossible.
- Delays delivery of the most useful results.

**Decision:** unsuitable as the first canonical dashboard source.

### Scope 2 — minimal headline benchmark

Benchmark only:

- Quantile + L1;
- Huber + L2;
- one scale;
- three backends.

**Advantages**

- Fast to implement and execute.
- Easy external comparison.

**Disadvantages**

- Does not exercise SCAD/MCP, which are among the most distinctive solver paths.
- Provides little scale information.
- Cannot show the value of Auto dispatch across smooth and nonsmooth penalties.
- Too narrow to justify a general penalty filter.

**Decision:** insufficient.

### Scope 3 — staged core matrix

Phase 1 uses two representative losses, six penalties, four scales, three backends, and Auto dispatch. Later phases add loss variants, specialized penalties, and solver diagnostics.

**Advantages**

- Covers the common baseline (`none`, L1, L2, ElasticNet).
- Covers the high-value nonconvex paths (SCAD, MCP).
- Exercises smooth, nonsmooth, and nonconvex solver dispatch.
- Includes both large-n and high-p regimes.
- Produces a comprehensible dashboard matrix.
- Can be executed and validated incrementally.

**Disadvantages**

- Group/adaptive penalties are deferred.
- Only one Quantile level and one Robust loss are covered initially.

**Decision:** select.

## 5. Phase 1 benchmark design

### 5.1 Model and loss variants

| Model ID | Loss/variant | Fixed parameters | Reason |
|---|---|---|---|
| `QuantileRegression` | `quantile`, variant `q0.5` | `quantile=0.5` | Median regression is the clearest robust Quantile baseline and has an aligned sklearn reference for none/L1. |
| `RobustRegression` | `huber`, variant `huber_mad` | `epsilon=1.35`, `method=MAD` | Huber is smooth enough to exercise Newton-like paths and is the most familiar robust regression baseline. |

Do not add `q0.1`, `q0.9`, Bisquare, Fair, or Proposal-2 in Phase 1. They should be separate variants after the core matrix is stable.

### 5.2 Penalties

Phase 1 penalties:

```text
none
l1
l2
elasticnet
scad
mcp
```

Rationale:

- `none` provides an in-source baseline;
- L1 and L2 are standard convex references;
- ElasticNet covers mixed sparsity/shrinkage;
- SCAD and MCP exercise the highest-value nonconvex paths;
- all six can be represented without extra group metadata or adaptive initial-estimator contracts.

Deferred penalties:

```text
adaptive_l1
group_lasso
group_scad
group_mcp
```

Adaptive penalties require a stable definition of the pilot estimator and weight exponent. Group penalties require canonical group membership. They should not be added until those parameters have explicit source contracts.

### 5.3 Scales

| Label | n | p | Purpose |
|---|---:|---:|---|
| `5K×50` | 5,000 | 50 | Small baseline and CPU-overhead regime. |
| `20K×100` | 20,000 | 100 | Representative medium workload. |
| `100K×50` | 100,000 | 50 | Large-n GPU-throughput regime. |
| `5K×500` | 5,000 | 500 | High-dimensional regime. |

Focused mode should choose the largest workload by the existing `n × p` rule unless the user selects a scale explicitly.

### 5.4 Backends and solver policy

Canonical headline rows should use:

```text
backend ∈ {numpy, cupy, torch}
requested_solver = auto
```

Every row must record:

```text
requested_solver
resolved_solver
implementation
```

This avoids presenting explicit solver choices as if they were equally recommended production paths. Solver-development comparisons should be placed in a separate diagnostic source or later phase.

Expected dispatch families include:

- L2/none on smooth Huber: IRLS/Newton-style path;
- L1/ElasticNet: FISTA-style path;
- Quantile SCAD/MCP: proximal IRLS-CD or the resolved current implementation;
- Huber SCAD/MCP: proximal Newton or the resolved current implementation.

The runner must record the actual resolved value rather than assuming it from the requested configuration.

### 5.5 Regularization-strength contract

A fixed raw `alpha` across different scales and feature counts is not sufficiently comparable. The runner should standardize `X`, center/standardize `y` as appropriate, and define sparsity penalties relative to the gradient at zero:

```text
alpha_max = max_j |gradient_j(beta=0)|
alpha = alpha_ratio × alpha_max
```

Recommended Phase 1 values:

```text
alpha_ratio = 0.05
l1_ratio = 0.5 for ElasticNet
```

For L2, use a separately documented normalized value, initially:

```text
alpha = 0.1
```

For `none`, use `alpha=0`.

Every run must store:

```json
{
  "alpha": 0.0123,
  "alpha_ratio": 0.05,
  "alpha_reference": "max_abs_gradient_at_zero",
  "l1_ratio": 0.5
}
```

SCAD/MCP should use the same `alpha` normalization as L1 so their shrinkage level is interpretable across scales.

Before the full P100 run, a pilot should verify that the selected `alpha_ratio` avoids both all-zero and fully dense solutions for the L1/SCAD/MCP cases. If one ratio cannot serve both loss families, store a loss-specific ratio in the source configuration rather than silently changing it per run.

### 5.6 Data generation

Use deterministic synthetic designs with a common sparse coefficient structure:

- standardized Gaussian or correlated Gaussian `X`;
- fixed sparsity ratio, initially 10% active coefficients with a minimum of 10 active variables;
- Quantile noise: asymmetric or heavy-tailed noise, with the exact distribution stored;
- Huber noise: a fixed contamination mixture, for example 90% Gaussian plus 10% large-variance Gaussian;
- three fixed data seeds per configuration;
- identical generated data across NumPy/CuPy/Torch.

Record the complete data-generation configuration in the source-level metadata.

### 5.7 Timing protocol

For each backend/configuration/seed:

- construct data outside timed regions;
- transfer/prepare backend arrays before timed regions, unless transfer-inclusive timing is explicitly requested;
- run 1–2 warmups;
- run 5 measured fits;
- synchronize CuPy/Torch before and after each timed fit;
- use float64;
- collect mean, standard deviation, minimum, maximum, and measured count;
- aggregate seeds explicitly with `seed_count=3` and documented `std_scope`.

The source should distinguish:

```text
fit_only
transfer_inclusive
```

Phase 1 canonical timing should use `fit_only`. Transfer-inclusive timing may be added later as a separate `variant` or comparison source.

## 6. Metrics and correctness gates

### 6.1 Required timing and speedup

Every statgpu row must include timing. CuPy and Torch rows must report speedup against the exactly matched statgpu NumPy row:

```text
same model
same loss/variant
same penalty
same alpha and l1_ratio
same requested/resolved solver contract
same scale
same seed aggregation
```

Use runner-reported semantics if the runner calculates the ratio from its own measured summaries. The parser should still validate the ratio against the matched timing rows.

### 6.2 Required convergence diagnostics

Record:

- converged rate;
- mean and standard deviation of iterations;
- failure count;
- final objective;
- relative objective gap versus NumPy;
- KKT or stationarity residual appropriate to the penalty;
- warning/error messages.

A timing result with failed convergence must not be treated as a successful performance row.

### 6.3 Required backend agreement

For each matched CuPy/Torch row, compare against NumPy using:

- coefficient relative L2 error;
- coefficient maximum absolute difference;
- objective relative difference;
- selected-support Jaccard for sparsity penalties;
- selected-feature count difference.

Recommended initial gates should be calibrated by pilot results, but the source must include actual tolerances rather than relying only on `PASS` labels.

### 6.4 Simulation-quality metrics

Because the data-generating support is known, record for L1/ElasticNet/SCAD/MCP:

- precision;
- recall;
- false discovery proportion;
- F1;
- support Jaccard;
- selected-set size.

These metrics should populate the existing Selection panel. This is more informative than coefficient difference alone for nonconvex penalties.

## 7. External-reference strategy

External comparisons must be narrow and objective-aligned.

### 7.1 Quantile

Use sklearn `QuantileRegressor` for:

- `penalty=none` (`alpha=0`);
- `penalty=l1` with matched `quantile`, `alpha`, intercept, standardization, and objective scaling.

These rows may participate in timing and strict objective/coefficient validation after alignment is verified.

Do not invent sklearn rows for L2, ElasticNet, SCAD, or MCP.

### 7.2 Huber

Sklearn `HuberRegressor` jointly estimates scale and has a different objective parameterization from Huber + MAD in statgpu. Therefore:

- include it only as a `partial` external timing reference if useful;
- do not use it as the default speedup reference;
- do not enforce strict coefficient equality;
- explain the parameterization difference in source metadata and tooltip text.

A stricter Huber comparison can be introduced later using an implementation with an exactly aligned loss/scale contract.

### 7.3 Headline speedup

The canonical GPU speedup for this source should remain:

```text
statgpu GPU / matched statgpu NumPy
```

External timings should be presented as reference series rather than mixed into the main GPU speedup headline.

## 8. Canonical JSON contract

Recommended top-level shape:

```json
{
  "date": "2026-07-XX",
  "benchmark": "penalized_robust_quantile",
  "environment": {},
  "config": {
    "repeats": 5,
    "warmups": 2,
    "seed_count": 3,
    "timing_scope": "fit_only",
    "dtype": "float64"
  },
  "results": []
}
```

Each result must contain enough fields to reconstruct a canonical run without parser inference:

```json
{
  "model": "QuantileRegression",
  "loss": "quantile",
  "variant": "q0.5",
  "penalty": "scad",
  "parameters": {
    "quantile": 0.5,
    "alpha": 0.0123,
    "alpha_ratio": 0.05,
    "alpha_reference": "max_abs_gradient_at_zero",
    "l1_ratio": null,
    "fit_intercept": true
  },
  "requested_solver": "auto",
  "resolved_solver": "proximal_irls_cd",
  "backend": "cupy",
  "n_samples": 20000,
  "n_features": 100,
  "timing": {},
  "convergence": {},
  "accuracy": {},
  "selection": {},
  "speedup": {}
}
```

`method_config_id` must include at least:

```text
model/loss
variant
penalty
alpha
alpha_ratio/reference
l1_ratio
loss-specific parameters
requested solver
resolved solver
fit_intercept
timing scope
```

## 9. Parser and registry work

### 9.1 New parser

Add a dedicated parser, for example:

```text
parse_penalized_robust_quantile_benchmark_v1
```

Do not overload the existing June 23 unpenalized parser. Keeping the parsers separate preserves source semantics and allows independent versioning.

The parser should:

- normalize `none` as the literal penalty string `none`;
- preserve `variant`, `loss`, and all regularization parameters;
- set `solver=auto` and `solver_display` from the requested solver;
- set `implementation` from the resolved solver;
- attach timing, speedup, convergence, accuracy, validation, and selection groups;
- create external rows only from explicit source entries;
- declare both models with `supports_penalty=True`;
- validate matched NumPy speedup references;
- reject duplicate method identities.

### 9.2 Manifest

Register the new source in `dev/benchmarks/frontend_sources.json` with:

- explicit `source_date` on or after the current cutoff;
- SHA256;
- required status;
- P100 environment ID;
- a dedicated comparison ID;
- the new parser name;
- only explicitly approved informational issue codes.

Do not remove the June 23 unpenalized source. It remains valuable as an external sklearn baseline and historical within-policy benchmark.

## 10. Frontend behavior

### 10.1 Filters

Within `Robust/Quantile`, the progressive filter path should become:

```text
Model → Variant → Penalty → Solver → Scale → Backend → External
```

Expected penalty options after the new source is selected:

```text
All
none
l1
l2
elasticnet
scad
mcp
```

The existing schema and filter state already support this dimension.

### 10.2 Focused mode

Recommended Focused rules for `robust_quantile`:

1. choose the representative largest workload unless scale is explicitly selected;
2. use Auto/resolved solver rows;
3. retain one row group per penalty;
4. retain `none` as the baseline;
5. show all three statgpu backends;
6. include aligned external rows only when explicitly enabled;
7. use compact labels such as:

```text
Quantile q0.5 · SCAD
Huber · ElasticNet
```

Do not apply the Penalized GLM rule that removes `none`; for Robust/Quantile, the direct none-versus-penalty comparison is a primary use case.

### 10.3 Full matrix

Full matrix should preserve:

- all variants;
- all penalties;
- all solver/implementation rows if diagnostic rows are added later;
- all scales;
- selected external frameworks.

### 10.4 Tooltips and table

Tooltips and table details should expose:

- alpha and alpha ratio;
- l1 ratio;
- quantile or Huber epsilon/method;
- requested and resolved solver;
- timing scope;
- convergence status;
- objective/KKT diagnostics;
- support metrics where available.

No new Schema version is needed because `Run.parameters` already supports structured configuration metadata.

## 11. Test plan

### 11.1 Runner tests

- deterministic data for fixed seeds;
- identical input data across backends;
- warmup excluded from measured counts;
- GPU synchronization around timed fits;
- complete configuration metadata;
- failure rows preserved rather than silently dropped.

### 11.2 Parser tests

- expected Phase 1 statgpu row count: 144;
- all six penalties present for both models;
- four scales and three backends present;
- `penalty=none`, never null, for unpenalized baseline rows in the new source;
- unique run IDs and method-config IDs;
- matched NumPy references for every GPU speedup;
- model registry has `supports_penalty=True`;
- selection metrics present for sparsity penalties;
- source date and SHA policy enforced.

### 11.3 Semantic tests

- speedup equals matched NumPy timing ratio within tolerance;
- failed convergence cannot have an `ok` quality status;
- external reference identities never serve unsupported penalties;
- alpha normalization metadata is present and finite;
- cross-backend objective/coef/support tolerances are enforced.

### 11.4 E2E tests

- Robust/Quantile exposes the penalty selector;
- all Phase 1 penalties are selectable;
- selecting penalty clears incompatible solver/scale state;
- Focused keeps `none` plus penalized rows;
- Full matrix row count is no smaller than Focused;
- Quantile/Huber variants do not merge;
- external references appear only in supported contexts;
- Selection and Convergence panels render when metrics are available.

## 12. Delivery phases

### Phase 0 — pilot and contract freeze

- implement a local benchmark runner;
- run a reduced matrix on NumPy and one GPU backend;
- calibrate `alpha_ratio`, convergence tolerances, and runtime;
- freeze the JSON contract and method identity;
- verify solver-resolution recording.

**Exit gate:** no all-zero/dense degeneracy across the core sparse penalties; no unexplained backend objective mismatch.

### Phase 1 — canonical core benchmark

- run the 144-row statgpu core matrix on P100;
- add aligned external rows;
- generate the final JSON artifact;
- register SHA/date/parser;
- implement parser and tests;
- update generated bundle.

**Exit gate:** all required rows parsed, every GPU speedup matched to NumPy, convergence and agreement gates satisfied.

### Phase 2 — frontend integration

- enable penalty support for both models;
- verify progressive filter options;
- implement Robust/Quantile Focused rules;
- expose alpha/resolved solver in tooltips/table;
- add E2E coverage;
- refresh deployed assets and documentation.

**Exit gate:** deterministic staleness, TypeScript, and Playwright checks green.

### Phase 3 — controlled expansion

Candidate additions, in priority order:

1. Quantile levels `q0.1` and `q0.9`;
2. Bisquare and Fair robust variants;
3. `huber_prop2` scale estimation;
4. adaptive L1 with explicit pilot-estimator contract;
5. group penalties with explicit group metadata;
6. a separate solver-diagnostic source at one representative scale;
7. transfer-inclusive timing.

Each addition should be justified by a concrete comparison question, not merely by implementation availability.

## 13. Acceptance criteria

The implementation is complete only when:

- a new June-or-later P100 source is committed and SHA-registered;
- the source contains real penalized measurements, not relabeled unpenalized rows;
- 144 required statgpu rows are present before optional external rows;
- penalty, alpha, loss parameters, requested solver, and resolved solver are explicit;
- NumPy/CuPy/Torch timing is available for every required configuration;
- all GPU speedups reference a matched NumPy row;
- convergence, objective/KKT, and backend-agreement diagnostics are present;
- selection metrics are emitted for sparse penalties;
- Robust/Quantile exposes the penalty filter;
- Focused mode gives a readable none-versus-penalty comparison;
- Full matrix preserves the complete source;
- parser, schema, semantic, TypeScript, build, staleness, and Playwright checks pass;
- documentation and deployed assets are updated in the same change.

## 14. Explicit non-goals for Phase 1

- No complete ten-penalty matrix.
- No exhaustive explicit-solver matrix.
- No new sidebar category.
- No reuse or relabeling of the June 23 unpenalized timings.
- No strict Huber coefficient parity claim against sklearn under mismatched scale parameterization.
- No transfer-inclusive speedup headline.
- No schema-version increment unless implementation discovers a field that cannot be represented safely in `parameters` or existing metric groups.
