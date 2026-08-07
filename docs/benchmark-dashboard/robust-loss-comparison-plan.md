# Robust-Loss Benchmark Coverage and Comparison Plan

## 1. Audit conclusion

The current dashboard does not provide a complete robust-regression comparison.

The eligible June 23 source contains:

- Quantile regression performance for statgpu CPU and scikit-learn;
- Huber regression performance for statgpu CPU and scikit-learn;
- small correctness checks for those two losses.

It does not contain:

- Bisquare/Tukey-biweight performance;
- Fair-loss performance;
- CuPy or Torch fit timings for Huber, Bisquare, or Fair;
- scale-estimation comparisons;
- contamination-regime comparisons;
- a structured result artifact from the existing Bisquare P100 smoke runner.

This is a benchmark-data gap, not a frontend filtering problem. The frontend must not infer Bisquare or Fair rows from Huber results.

## 2. Existing implementation and runner evidence

The implementation exposes three robust losses:

```text
Huber
Bisquare (Tukey biweight)
Fair
```

All three are available through `PenalizedRobustRegression`, support NumPy/CuPy/Torch dispatch, and expose MAD or Huber Proposal-2 scale estimation where applicable.

The repository also contains P100-oriented smoke scripts:

```text
dev/tests/remote_bench_huber_bisq.py
dev/tests/remote_bench_full.py
```

Those scripts demonstrate that Huber and Bisquare solver paths can be executed, including Bisquare + SCAD. They print results to stdout, use one principal scale, and do not write a canonical JSON artifact with repeats, seeds, provenance, external references, or aggregation metadata. They therefore cannot be registered directly as dashboard sources.

No comparable Fair-loss benchmark runner or committed result artifact was found.

## 3. Recommended separation of benchmark questions

Two statistically different questions should not be mixed into one headline chart.

### 3.1 Unpenalized robust-loss comparison

This source should compare the statistical loss functions themselves:

```text
Squared error baseline
Huber
Bisquare
Fair
```

It should hold the penalty at `none`, use the production-recommended solver for each loss, and focus on robustness, backend performance, and scale behavior.

Recommended artifact:

```text
results/benchmark_frontend_sources/robust_losses_202607xx.json
```

### 3.2 Penalized robust/quantile comparison

The separate staged plan in:

```text
docs/benchmark-dashboard/penalized-robust-quantile-plan.md
```

should remain responsible for L1/L2/ElasticNet/SCAD/MCP comparisons. Bisquare and Fair should enter a later phase after the unpenalized loss benchmark establishes correctness and stable initialization/scale contracts.

This separation avoids confounding the effect of the robust loss with the effect of regularization and solver dispatch.

## 4. Core experimental matrix

### 4.1 Loss variants

| Variant | Main behavior | Required tuning metadata |
|---|---|---|
| `squared_error` | Non-robust baseline | none |
| `huber_mad` | Convex bounded-influence transition | `epsilon=1.35`, `method=MAD` |
| `bisquare_mad` | Redescending Tukey-biweight loss | `epsilon=4.685`, `method=MAD` |
| `fair_mad` | Smooth gradual down-weighting | `epsilon=1.35`, `method=MAD` |

A later diagnostic source may add `method=huber_prop2`. It should not be silently pooled with MAD because scale estimation changes both the fitted objective and runtime.

### 4.2 Scale grid

Use the same principal scales as the staged penalized plan where feasible:

| Label | n | p | Purpose |
|---|---:|---:|---|
| `5K×50` | 5,000 | 50 | CPU and launch-overhead baseline |
| `20K×100` | 20,000 | 100 | medium workload |
| `100K×50` | 100,000 | 50 | large-n GPU throughput |
| `5K×500` | 5,000 | 500 | high-dimensional linear algebra |

Add smaller fallback cases only when an external implementation cannot complete the principal grid. Such rows must use distinct case identities and must not be presented as aligned speedups.

### 4.3 Contamination regimes

A robust-loss benchmark using only clean Gaussian noise is incomplete. At minimum include:

| Regime | Suggested construction | Purpose |
|---|---|---|
| `clean_gaussian` | Gaussian noise, no injected outliers | efficiency baseline |
| `vertical_5pct` | 5% response outliers | moderate response contamination |
| `vertical_15pct` | 15% response outliers | strong redescending-loss test |
| `heavy_tail_t3` | Student-t noise with 3 degrees of freedom | heavy-tail robustness |
| `leverage_5pct` | 5% high-leverage design rows with response contamination | leverage sensitivity |

The contamination seed and generated outlier indices must be retained in source metadata or reproducible from the stored seed.

## 5. Backend and timing contract

For every supported loss/scale/regime combination run:

```text
backend ∈ {numpy, cupy, torch}
```

Required protocol:

- float64;
- backend arrays prepared before fit-only timing;
- transfer-inclusive timing recorded separately;
- two warmups;
- five measured fits per data seed;
- at least three data seeds;
- explicit CuPy and Torch synchronization before and after each timed fit;
- mean, standard deviation, minimum, maximum, and replicate count;
- convergence status and iteration count;
- scale-estimation time recorded separately when it is not part of fit timing;
- peak GPU memory or an explicit OOM record.

The existing smoke scripts synchronize Torch globally but are not sufficient as a canonical timing protocol: they use a single data set, print only one elapsed time, and do not separately validate CuPy synchronization semantics.

## 6. Correctness and robustness metrics

Timing alone is not enough. Record:

- objective value under the same loss and scale;
- gradient norm or solver-appropriate stationarity diagnostic;
- coefficient relative error across backends;
- intercept and estimated scale agreement;
- convergence rate and iteration count;
- prediction RMSE on clean test data;
- prediction RMSE under contaminated test data;
- coefficient error relative to the known simulation truth;
- effective nonzero-weight fraction;
- fraction of observations fully rejected by Bisquare;
- finite-value checks and failure reason.

For Bisquare, initialization is part of the method contract. Store whether initialization used OLS, Huber, or a supplied coefficient vector. Do not compare differently initialized Bisquare fits under one method identity.

## 7. External-reference policy

External comparisons must match:

- loss/psi function;
- tuning constant;
- residual-scale definition;
- scale-update policy;
- intercept convention;
- stopping rule as closely as feasible.

Candidate references documented by the implementation include R/MASS-style Huber, Bisquare, and Fair behavior. Before registering an external speedup, verify the exact conventions and report any remaining mismatch as `partial` quality rather than claiming exact parity.

Where no genuinely aligned external implementation is available, use an independent NumPy formula/IRLS reference for correctness and restrict speedup to matched statgpu NumPy-versus-GPU rows.

## 8. Frontend identity

Use one model family:

```text
model_id = RobustRegression
category = robust_quantile
```

Represent the loss as a first-class field and use explicit variants:

```text
huber_mad
bisquare_mad
fair_mad
```

Method identity must include:

- loss;
- epsilon/tuning constant;
- scale method;
- requested and resolved solver;
- initialization policy;
- contamination regime only in the case identity, not the method identity.

Focused mode should not collapse the three robust losses into one row. It may choose one representative scale, but it must preserve all selected loss variants.

## 9. Acceptance gates

The source is ready for registration only when:

1. Huber, Bisquare, and Fair each have NumPy/CuPy/Torch rows at at least three scales;
2. all three contamination families—clean, vertical outlier, and heavy-tail—are represented;
3. Bisquare initialization and rejected-observation diagnostics are recorded;
4. backend coefficient/objective agreement passes documented tolerances;
5. GPU timing uses explicit synchronization;
6. no failed or OOM configuration is silently omitted;
7. external speedups are emitted only for genuinely aligned comparisons;
8. parser, schema, semantic, deterministic-generation, TypeScript, build, and Playwright checks pass.

## 10. Priority

This work should be treated as P1 for robust/quantile coverage because the current dashboard can otherwise be misread as a general robust-regression benchmark even though it contains only CPU Huber and Quantile performance rows.

Recommended order:

1. unpenalized Huber/Bisquare/Fair benchmark across backends and contamination regimes;
2. register and expose the robust-loss variants;
3. execute the Phase-1 penalized Huber/Quantile matrix;
4. add penalized Bisquare/Fair only after their initialization and scale contracts are stable.
