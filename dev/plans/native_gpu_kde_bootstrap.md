# Native GPU migration: first experimental KDE slice

Base: merged PR #251, `de7dd3e593dae4cfba4f8742c18bddd7ffce86e4`.
Status: developer-only candidate; single-P100 core validation passed at
`d4fe6f64a7ea9be9967d0f116c5184acce11ceee` (THREADS=1): 183 passed,
1 allowed two-device skip, all 38 required GPU cases executed, and four complete
benchmark JSONs. [PR #252](https://github.com/TheHiddenObserver/statgpu/pull/252)
records this exact-head evidence and retains the historical `878b966` failure.
This is not a claim that later commits have been physically rerun. Extended
scales, T=2/T=4 CPU profiles, native RNG, Torch bootstrap and public-API/default
promotion remain outside this completed core validation.

## Contract inventory and scope

| Consumer | Current contract | This change |
|---|---|---|
| public `kde_bootstrap_confidence_interval` | CPU/CuPy weighted row bootstrap, host RNG and percentile reduction; Torch weighted bootstrap unsupported | unchanged |
| public `fit_kde` / selectors / formula consumers | existing backend dispatch and tuning | unchanged |
| new developer candidate | no published API | CuPy-only, univariate Gaussian, fixed finite positive numeric bandwidth factor, float64, explicit pre-generated indices |
| bandwidth and knockoff benchmarks | existing hybrid implementations and evidence | unchanged; subsequent migration phases |

Active axes: experimental numerical refactor, device/dtype/memory, resampling, benchmark, developer documentation. No public API change, new CV objective, formula path, package/version change, or three-backend support claim. The scoped CuPy prototype tests the existing supported GPU bootstrap route; it is not a new shared production API. Torch bootstrap remains explicitly unsupported. NumPy is the reference, not a claimed device-native candidate.

## Statistical contract

Resample rows with replacement using normalized original weights as probabilities. Each realized index row then selects the original weights and normalizes them again, preserving the existing implementation's reweighting (not silently replacing it with an unweighted bootstrap). The fixed scalar bandwidth factor is held fixed; each replicate refits its weighted covariance. Conditional pointwise percentile intervals use linear quantiles, not simultaneous bands or tuning-adjusted inference. Caller-supplied indices must be integer, in range, and shape `(B, n)`; their realization is part of data identity. The candidate is conditional on these indices and does not claim to validate their sampling law.

Variance uses denominator `1 - sum(w**2)` and covariance jitter `max(var*1e-12, 1e-12)`, matching current univariate `fit_kde`. Degenerate effective degrees of freedom fail closed. Samples, queries, weights, indices, intermediate covariance/densities and quantiles remain on one concrete CuPy device. Python control and scalar validation synchronization are permitted; final host extraction is external. No host fallback or string bandwidth/kernel/multivariate expansion.

## Memory and performance experiment

Iterate replicates on device and chunk query evaluation. Retained `(B, q)` draws are explicitly budgeted. A deterministic planner bounds planned numerical buffers conservatively and rejects insufficient budgets before allocating outputs. This is an algorithm-managed workspace estimate, not a CUDA allocator, library scratch, or total process hard cap. Inputs and CUDA context are outside that estimate. Separately measure pool high-water usage when feasible and state its limitations.

Compare three variants on identical host data, weights and pre-generated indices: NumPy `fit_kde` per replicate, current CuPy `fit_kde` per replicate, and the experimental candidate. The first two are controlled-index reference workflows, not timings of the unchanged public bootstrap API. Random-index generation is excluded from all timings and disclosed; input/index transfers and final host results are included. All device timings synchronize the concrete executed device. Device-operation evidence is collected in a separate instrumented execution, never in the reported timing samples. Include initial estimate, draws and interval reduction for every variant.

## Gates and staged roadmap

1. Preimplementation independent design audit.
2. CPU planner/invalid-contract/oracle/harness tests; conditional real CuPy tests for weighted/unweighted/sparse weights, constant samples, chunk invariance, device residency, forbidden host conversions and malformed inputs.
3. Independent exact-content review and correction loop.
4. Core physical gate: P100 correctness, operation trace, memory diagnostics, then 2 warmups + 7 uninstrumented repetitions at small and GPU-relevant n/q/B points. This gate passed for the exact head recorded above; claims remain limited to its measured controlled-index workflows.
5. If equivalent and beneficial, separately design native RNG semantics/reproducibility and broader selectors/kernels/dimensions, then discuss production opt-in/default promotion.
6. Bandwidth: preserve current useful CV baseline; migrate demonstrated host numerical stages with per-method correctness and representative-scale evidence, not workflow timing as selector-only evidence.
7. Knockoff: align realized randomness and statistical work first; retain diagnostic-only status until fair comparison contracts close, then migrate identified host linear algebra.

No new large-CV resource budget, server job, dashboard publication or release is implied by this plan.

## Design audit and numerical envelope

The independent preimplementation audit approved this bounded design against the
merged base. The candidate explicitly rejects nonfinite derived variance, weight
sum, inverse variance or normalization, and overflow/underflow of bandwidth
squared or scaled variance. This narrower finite numerical envelope does not
claim to reproduce the public covariance inversion retry policy for extreme
inputs. Constant samples remain valid through covariance jitter. Index range
validation uses reductions rather than a B-by-n boolean temporary. Scalar
extraction is for validation only; covariance jitter and density normalization
remain device arithmetic.

The controlled CuPy refit path also preserves the second sampled-weight
normalization performed at `fit_kde` ingress. Since the unchanged public path's
uncentered mean/covariance is sensitive to reduction order for large location
relative to spread, the prototype rejects
`abs(weighted_mean) * float64_epsilon > sqrt(jittered_variance) * 1e-8`.
This is a conservative experimental conditioning guard, not a production
statistical acceptance threshold. Large-offset/tiny-spread or large-offset
constant samples can therefore fail explicitly, even if the public path returns
finite densities. Squared-distance/exponent overflow is likewise rejected before
an infinite intermediate can silently erase a Gaussian tail contribution.
