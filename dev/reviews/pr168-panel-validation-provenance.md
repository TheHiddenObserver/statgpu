# PR168 Panel implementation and validation provenance

Recorded: 2026-10-09. This is an engineering reference for the documentation-audience repair, not a user support matrix or a new numerical validation report.

## Source identity and evidence limits

- PR: [TheHiddenObserver/statgpu #168](https://github.com/TheHiddenObserver/statgpu/pull/168).
- Comparison base: `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.
- Documentation source head: `880b7b8474ab4728ed44a4556b1b64faaa6e7b55`.
- Scope: nine English pages under `docs/en/panel/`; their corresponding Chinese pages already describe the user-facing numerical/support contract and are unchanged by this relocation.
- The source pages were byte-verified against the Git blobs in the remote recursive tree before editing. The source-page dates below are document metadata, **not inferred execution dates**.
- This note relocates existing assertions, tolerances, runner ownership and private implementation descriptions. No historical artifact or runner was re-executed or independently re-certified by this documentation-only repair. Its descriptions record what the source pages said; they do not turn old observations into guarantees for the source head or a later commit.
- The numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` and abbreviated source label `5068da3f` are **historical**. The latter was incorrectly described as the “current PR head” in the old Fama–MacBeth page. It is not PR168 head `880b7b8474ab4728ed44a4556b1b64faaa6e7b55`. The original abbreviated label is preserved without inventing its full SHA or execution date.
- Directory labels such as `pr126_perf_fix_528d967e` and `pr126_review_fix_da3604ee` are retained artifact locations, not a substitute for each artifact's recorded source identity. The old text does not establish which full source SHA produced every file in those directories.
- Historical GPU, benchmark and R evidence applies only to the source, validator, hardware, device and environment actually recorded by that evidence. Fresh acceptance for another head requires matching evidence, or an explicitly supported unchanged-source fingerprint reuse contract. Moving these notes does not satisfy that requirement.
- Test tolerances below are assertion thresholds, not measured errors or universal numerical accuracy promises. GPU parity checks and independent statistical-reference checks answer different questions.

## Origin inventory

| Source page | Original page date | Source Git blob |
|---|---|---|
| [fama-macbeth](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/fama-macbeth.md) | 2026-08-20 | `729572b2ce6bb44abe75bb2612dd190fcf8c70e2` |
| [covariance](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/covariance.md) | 2026-08-18 | `2d42566a0f2e5fc81fd8e2edf858d264ba368eb0` |
| [pooled-ols](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/pooled-ols.md) | 2026-08-19 | `37e4695d87dde98016a30ed3aa8256952b81b42e` |
| [between-ols](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/between-ols.md) | 2026-08-19 | `ad4b97432eb0268fbb6d8903a66c1b3dcc529699` |
| [first-difference-ols](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/first-difference-ols.md) | 2026-08-19 | `71f3786e9de6cf54e9f2f21d63d9f49ed883efb5` |
| [random-effects](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/random-effects.md) | 2026-08-19 | `7cf6bcab38b5376040a561be1146f37ab2e43337` |
| [panel-ols](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/panel-ols.md) | 2026-08-15 | `84ee21b56db287bda363784771221493e10c27f6` |
| [diagnostics](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/diagnostics.md) | 2026-08-18 | `b504f48076154ee2f905a10784e39856c6681034` |
| [fit-statistics](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/panel/fit-statistics.md) | 2026-08-15 | `0715020750231e9863dfc907481f6f0bca57a9ba` |

## Reading the relocated notes

The following sections preserve the original engineering detail under its page of origin. They are archival descriptions of implementation/coverage at the source snapshot, not new execution results. References to “maintained” tests or “gates” identify the described validation responsibilities; they do not say those tests passed on PR168. Statistical formulas, parameter choices, supported devices, observed failure semantics and useful external-package interpretation remain in the public pages.

## fama-macbeth

### Identification and numerical-rank implementation

This coefficient interpretation requires every retained period-specific coefficient vector to be identified by its cross-sectional design. After the observation-count filter is applied, statgpu checks the intercept-augmented $X_t$ with the shared panel SVD rank policy. If a retained period is rank deficient, `.fit()` raises a `ValueError` before coefficient averaging or inference rather than reporting coordinate-level results from a non-unique coefficient representation.

### Period grouping, Gram certificate, right-hand-side cancellation and SVD fallback

A period is retained when it satisfies `min_obs_per_period` and the implementation's minimum count rule $n_t\ge k$, where $k$ is the intercept-augmented design width. The full-rank contract is then enforced for every retained period. NumPy, CuPy, and Torch use the same conservative exact-size Gram-certificate dispatch. Retained periods are grouped by their actual row count, without zero padding. Because every period has an exact intercept, a full-rank period may first remove a safe common response anchor along that constant direction before forming the Gram right-hand side; the intercept is restored after the solve. In addition, the exact intercept coordinate of $X_t^\top y_t$ uses a magnitude-tiered response sum rather than an ordinary BLAS reduction, so a tail such as `[2**55, 1, -2**55]` produces the representable intercept contribution `1` without forcing an SVD fallback. These two protections handle different risks: anchoring removes a huge common response level, whereas the stable intercept RHS preserves cancellation around zero.

A period may consume the Gram solve only when the backend-native spectrum satisfies $\lambda_{\min}(G_t)/\lambda_{\max}(G_t)>10^{-4}$ and the Gram matrix, centered right-hand side, and candidate solution are finite. For the single-response Fama-MacBeth path, one augmented solve returns both the candidate coefficient and the inverse-Gram information used to bound coordinatewise right-hand-side roundoff, so the precision certificate does not require a second Gram factorization. If a non-intercept coefficient lies below that certified float64 resolution, the period leaves the Gram path and is checked by the maintained SVD fallback.

An exact-zero non-intercept Gram candidate receives an additional rare-path check rather than being rejected automatically. After the Gram certificate sends that period to fallback, statgpu recomputes the original non-intercept right-hand side with the magnitude-tiered reducer. If the raw BLAS right-hand side is zero but the stable right-hand side is nonzero, a representable cancellation tail was actually lost and the period fails closed with `FloatingPointError`; a rounded SVD basis is not treated as a reliable rescue. If both raw and stable right-hand sides are zero, the zero is not classified as a lost tail and the ordinary SVD/precision certificate may proceed. This distinction preserves genuine zero coefficients while still rejecting demonstrated cancellation loss. Torch uses its documented stacked-SVD support for ordinary unsafe subsets, while NumPy/CuPy retain supported two-dimensional fallbacks.

### Reporting aliases and packed host snapshot

Successful fits also publish the shared `ParameterInferenceResult` surface used by inference-capable statgpu estimators. The public `coef_`, `bse_`, `tvalues_`, `pvalues_`, and `conf_int_` arrays remain backend-native. Distribution inference follows the selected fit backend: NumPy uses the NumPy inference backend, CuPy uses the CuPy inference backend, and Torch uses the Torch inference backend on the actual tensor device. `_inference_result` and the `_params`, `_bse`, `_tvalues`/`_zvalues`, `_pvalues`, and `_conf_int` aliases contain NumPy snapshots only for the common inference/reporting contract; those snapshots are not used to calculate p-values or confidence intervals. For GPU fits, the reporting fields are packed on the active backend and copied in one small snapshot after numerical inference has completed. `newey-west` is labeled as `z`/normal inference, while `nonrobust` is labeled as Student-t inference with $T-1$ degrees of freedom.

### Historical numerical routing, P100 scaling and physical validation record

At least two valid periods must remain after filtering; otherwise `.fit()` raises an error because the variability of the coefficient series cannot be estimated from fewer than two periods. Every retained period must also have full column rank under the shared panel SVD cutoff; a rank-deficient retained period fails closed before inference. A distinct float64 coefficient-resolution failure also fails closed: if a non-intercept coordinate is too small relative to the absolute projection error and the SVD fallback materially violates least-squares stationarity, `.fit()` raises `FloatingPointError` and identifies the retained period rather than misreporting the case as rank deficiency. Likewise, if an ordinary BLAS non-intercept RHS is zero but the magnitude-tiered fallback RHS is nonzero, the two reductions disagree on a representable cancellation tail and the period fails closed. By contrast, a raw zero that remains zero under the stable reducer is allowed to proceed and may represent a genuine zero coefficient. These cases can occur even when the period design is perfectly conditioned; the distinction concerns coefficient resolution, not rank. On every maintained backend, the Gram certificate remains only a fast-path selector; true rank boundaries remain SVD-owned.

Numerical safety is fail-closed across this path. A full-rank exact period intercept can remove a range-safe midrange response anchor before the Gram/SVD response projection and restore that anchor only to the intercept afterward. Its Gram RHS is also formed from the shared magnitude-tiered response sum, so an intercept cancellation tail can remain on the `gram-certified` fast path with zero SVD fallbacks. Cancellation-sensitive SVD projections use the shared magnitude-tiered reducer for ordinary fallback cases. A non-finite Gram matrix, centered right-hand side, or candidate solution is treated as uncertified and routed through the rank-revealing SVD fallback. If an entire full-rank design is below $\sqrt{\mathrm{DBL\_MIN}}$, it is uniformly raised to that safe working scale before SVD and the final coefficient is transformed back; this positive scalar change leaves the relative rank cutoff unchanged. Period-coefficient averaging and parameter-R² scalar/group means use the shared magnitude-tiered float64 reduction policy. Ordinary-scale panel group reductions remain on the single-scatter fast path; extra magnitude tiers are activated only when dynamic range can hide a representable low-order contribution or an unscaled same-sign reduction can overflow. Each tier is reduced on the scale of the requested target: sum reductions restore only that tier's overflow factor, while mean reductions divide a safe tier after summation and predivide only a tier whose raw same-sign sum would overflow. This avoids erasing collectively representable subnormal mean contributions merely because another tier is huge. The implementation remains float64 arithmetic rather than arbitrary-precision or exact summation. Coefficient-series covariance uses per-coordinate centered scales and restores each symmetric entry with the larger scale first, preserving representable small-coordinate variance and cross-covariance. If the final covariance itself is non-finite or has a negative diagonal variance, inference fails closed. At exactly zero estimated variance, a zero coefficient has statistic 0 while a nonzero coefficient has signed-infinite statistic, so no dimensionful fake denominator or `0/0` `NaN` is introduced.

The maintained physical scaling fixture deliberately retains three resident-array workloads: micro (64×128×4; 8,192 rows), medium (128×1,024×8; 131,072 rows), and large (128×4,096×16; 524,288 rows). Historical Tesla P100 evidence on numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` reports CuPy/Torch GPU-over-NumPy median-time ratios of **0.549/0.343** on micro, **0.204/0.168** on medium, and **0.114/0.109** on large, corresponding to about 1.82×/2.92×, 4.91×/5.97×, and 8.75×/9.16× speedups respectively. Every GPU backend × scale case used one `gram-certified` exact-size batch, one control synchronization, and zero SVD fallbacks. The old page described these measurements as a crossover on that historical P100 resident-array protocol; they are not a universal hardware guarantee. The old page reported that historical CuPy/Torch CUDA acceptance labeled `5068da3f` passed all 12 physical runners on Tesla P100 (CuPy 13.6.0 / Torch 2.0.0+cu117), including the Stage-C correctness matrix, the focused Fama-MacBeth oracle with certified-Gram provenance, HAC chronology, extreme t(2) tail, device affinity, scaling, RHS cancellation, rank precedence, and intercept cancellation; artifacts are preserved under `results/pr126_perf_fix_528d967e/` and `results/pr126_review_fix_da3604ee/`.

`dev/benchmarks/benchmark_fama_macbeth_scaling_gpu.py` records synchronized NumPy/CuPy/Torch median time, rows per second, GPU/NumPy ratio, speedup, solver provenance, numerical parity, backend versions, and thread-environment provenance. GPU input arrays are transferred before warmup/timing, so the benchmark measures resident-array `fit()` performance and explicitly excludes host-to-device input-transfer time. For that historical source, numerical differences in the accepted P100 scaling artifact remain tight: coefficient/beta/prediction differences are near machine precision and the largest reported statistic difference is below $4\times10^{-11}$.

P-values and critical values are evaluated through the selected inference backend instead of transferring the statistic vector to NumPy/SciPy. Normal inference uses the backend's two-sided normal routines; general Student-t inference uses the backend Student-t routines. The exact small-degree boundaries are kept backend-native as well: df=1 uses the Cauchy identity, and df=2 uses its elementary two-sided tail and quantile formulas so Torch versions without native `betainc` do not lose the maintained high-precision contract. Only the standardized `ParameterInferenceResult` reporting snapshot crosses to NumPy after inference has been completed.

### Zero-right-hand-side fallback check

**What happens if a period is full rank but a coefficient is below reliable float64 resolution?**  The fit raises `FloatingPointError` with a coefficient-resolution message when the available arithmetic cannot certify the coordinate. For zero non-intercept RHS values, statgpu first compares the ordinary BLAS reduction with a magnitude-tiered fallback reduction: disagreement (`raw=0`, `stable≠0`) is a demonstrated lost tail and fails closed, while agreement at zero is allowed to proceed as a genuine-zero candidate.

### External comparison and validation ownership

`dev/tests/test_fama_macbeth_linearmodels_external.py` is a maintained definition-alignment gate against pinned `linearmodels==7.0`. The fixture uses the same explicit period intercept, full-rank balanced panel, period ordering, and coefficient set in both packages. Period-by-period coefficients (`betas_` versus `all_params`) and the averaged coefficient vector are compared in both covariance modes.

For `cov_type="nonrobust"`, statgpu's covariance is aligned with linearmodels `cov_type="unadjusted", debiased=True`: covariance, standard errors, and coefficient t-statistics are compared. P-values and confidence intervals are intentionally **not** forced to match in this branch because the covariance definitions align while the two APIs use different reference degrees of freedom for post-estimation inference: statgpu uses the retained-period definition $T-1$, whereas linearmodels uses its stacked-panel residual degrees of freedom when `debiased=True`.

For `cov_type="newey-west"`, the external gate uses linearmodels `cov_type="kernel", kernel="bartlett", bandwidth=L, debiased=False` with the same fixed $L$. This aligns both the coefficient-series kernel covariance and normal-reference inference, so covariance, standard errors, test statistics, p-values, and confidence intervals are compared. The original page identified the `Panel Stage C external covariance` workflow as installing the pinned reference and running this test on relevant PR/source changes; that is a workflow-description claim, not a run result for this head.

Maintained internal regressions additionally cover formula intercept behavior, ordered-categorical versus numeric chronology on both array and formula paths, missing-row formula alignment, retained-period rank rejection, failed-refit invalidation, the SVD rank-boundary policy, conservative Gram-certificate acceptance/rejection, backend-native distribution routing, exact df=1/df=2 small-degree inference boundaries, balanced and shuffled-unbalanced exact-size GPU grouping, chronological rank-error reporting, SVD fallback ownership, direct-fit finite-validation ownership, the packed reporting snapshot, large-common-intercept Gram centering, stable exact-intercept Gram-RHS cancellation, lost nonconstant zero-RHS rejection, genuine nonconstant zero-RHS acceptance, and explicit coefficient-resolution failure reporting.

GPU consistency for the standard full-rank numeric-time `fama_macbeth_newey_west` case is tested by `dev/benchmarks/validate_panel_stage_a_gpu.py`. The historical focused gate `dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py` remains the detailed chronology/formula/rank/inference correctness oracle. The original page assigned optimized-source physical acceptance to `dev/benchmarks/validate_fama_macbeth_optimized_gpu.py`, while crossover evidence is recorded by `dev/benchmarks/benchmark_fama_macbeth_scaling_gpu.py`. The historical PR126 P100 evidence remains anchored to numerical source `8c60db00...` and is complemented by the exact-source Stage-C matrix and HAC-chronology runners; it is not current-head acceptance after the later numerical-path fixes. `dev/benchmarks/validate_panel_intercept_cancellation_gpu.py` exercises the broader shared coefficient-resolution and large-intercept panel contracts. The focused `dev/benchmarks/validate_fama_macbeth_rhs_cancellation_gpu.py` gate additionally verifies on both CuPy CUDA and Torch CUDA that an intercept cancellation tail remains `gram-certified` with zero SVD fallbacks, a demonstrated lost nonconstant RHS tail fails closed, and a genuine zero nonconstant RHS remains fit-able after fallback. Fama-MacBeth remains outside the Stage-C residual-covariance case matrix because its covariance is defined from the coefficient series rather than observation-level residual scores.

### Implementation location

Implementation: `statgpu/panel/_fama_macbeth.py`.

## covariance

### Grouped-score scaling and delayed restoration

For extreme but finite score magnitudes, grouped score reductions selectively use a group-size working scale only where a same-sign partial sum could overflow, and positive/negative contributions are accumulated separately before the final cancellation. The shared residual-covariance path also delays tiny-design scale restoration until after the covariance reduction: working-SVD projection coordinates are rescaled only when their product with the largest residual could overflow, while the residual vector itself is never globally magnitude-normalized. One-way cluster scores and Driscoll-Kraay period scores are grouped before any Gram normalization, and the subsequent per-coordinate Gram scale is the minimum needed to keep the relevant product/reduction in range. This preserves representable small groups or periods beside much larger observations without perturbing ordinary or subnormal-design paths that are already safe. As with ordinary float64 linear algebra, this is not a promise to recover arbitrary tiny remainders after catastrophically ill-conditioned upstream cancellation.

### Two-way cancellation, common Gram space and underflow boundary

Two-way clustering combines the two one-way cluster covariances and subtracts the covariance for the paired cluster labels. All three grouped-score components are formed before physical scale restoration. If one clustering dimension is nested in the other, statgpu recognizes equality of the induced partitions rather than equality of arbitrary integer codes and cancels the matching marginal/intersection component algebraically; otherwise the three components share one minimally scaled Gram space before inclusion-exclusion. If that common score scale can keep every grouped component nonzero but would still make a mathematically nonzero component self/cross product underflow before inclusion-exclusion, statgpu raises `FloatingPointError` rather than silently dropping the term. This is an explicit float64 working-range boundary for cancellations spanning more exponent range than one common Gram representation can retain; it is not reported as a zero covariance.

### Symmetrization, lag reductions and scale restoration order

Symmetric covariance combinations are evaluated with range-aware arithmetic: final symmetrization avoids overflowing a finite same-sign average, and two-way inclusion-exclusion subtracts a same-sign intersection component before a risky addition when possible. HAC/Driscoll-Kraay normalize only score coordinates whose zero-lag Gram or weighted lag product would otherwise overflow, before those products are materialized; Driscoll-Kraay performs its period grouping first. They then form symmetric lag averages and accumulate the complete lag sequence with a per-entry reduction-length working scale only where a transient partial sum could overflow. Tiny-design and projection-product restore factors remain outside this cancellation space until the final covariance is formed. Safe coordinates, groups, periods, and subnormal-design paths therefore stay on their original working scale. These reorderings are algebraically equivalent to the displayed definitions whenever the final float64 result is representable; as elsewhere, they do not claim higher-precision recovery from arbitrarily ill-conditioned cancellation.

### External reference matrix, tolerances and historical artifacts

The table below records how the statistical definitions are checked against independent implementations. GPU consistency is tested separately against NumPy so that agreement with another statistics package and agreement across hardware backends are not conflated.

| Layer | Reference | What is compared | Assertion tolerance |
|---|---|---|---|
| HC primitives | `statsmodels==0.14.6` | HC2/HC3 on a full-rank OLS regression | `rtol=5e-12`, `atol=5e-14` |
| Cluster / DK primitives | `linearmodels==7.0` | one-/two-way group-debiased clustering; Bartlett/Parzen/QS weights and DK covariance; default bandwidth and fixed-effect df adjustment | covariance `rtol=5e-12`, `atol=5e-14`; weights `rtol=5e-14`, `atol=5e-15` |
| PooledOLS / PanelOLS | `linearmodels==7.0` | coefficients plus DK covariance/BSE; PooledOLS group-debiased cluster covariance | coefficient `rtol=2e-10`, `atol=2e-11`; covariance/BSE `rtol=5e-9`, `atol=5e-11` |
| BetweenOLS / FirstDifferenceOLS | `statsmodels==0.14.6` | coefficients plus HC0/HC2/HC3 covariance/BSE after applying the same averaging/differencing transformation | coefficient `rtol=5e-10`, `atol=5e-12`; covariance/BSE `rtol=5e-9`, `atol=5e-11` |
| RandomEffects transformed regression | `linearmodels==7.0`, `statsmodels==0.14.6` | robust/HC2/HC3/DK covariance on statgpu's Swamy-Arora quasi-demeaned $X^*,y^*$; no coefficient-parity claim | covariance `rtol=5e-9`, `atol=5e-11` |
| R external checks | `plm==2.6-7`, `sandwich==3.1-3` | HC0/HC2/HC3 covariance and one-way FE coefficients | covariance `rtol=5e-9`, `atol=5e-11`; FE coefficient `rtol=5e-10`, `atol=5e-11` |
| Physical GPU | NumPy reference | 35 estimator cases + 12 covariance-primitive cases for each of CuPy and Torch | default `rtol=5e-6`, `atol=5e-7` |

The no-fixed-effect `PanelOLS` level regression is also compared with `statsmodels==0.14.6`, including coefficients, covariance/BSE, $R^2$, adjusted $R^2$, and model F statistics.

Ill-conditioned full-rank stress tests use scale-aware tolerances because the covariance entries can become very large: HC0 against statsmodels uses `rtol=2e-6, atol=5e-3`, while the stable HC2/HC3 leverage checks use `rtol=5e-11, atol=5e-3` when variances can exceed $10^{10}$.

CI tolerances in this table are pass/fail thresholds, not observed error measurements. Historical P100 validation records actual per-field `max_abs_differences` in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`, with a summary in `results/pr126_p100_fresh/validation_summary.txt`. Those artifacts predate the later shared reduction and public covariance fail-closed fixes and are reference-only; fresh exact-head CuPy/Torch CUDA validation is required for current acceptance.

The corresponding external tests are `dev/tests/test_panel_stage_c_external.py`, `dev/tests/test_panel_stage_c_external_defaults.py`, `dev/tests/test_panel_stage_c_linearmodels_estimators.py`, and `dev/tests/test_panel_stage_c_r_external.py`.

### Implementation location

Implementation: `statgpu/panel/_covariance.py`.

## pooled-ols

### SVD/BLAS response-projection route and range-safe centering

The automatically added constant is protected against response-cancellation loss. Ordinary responses keep the historical SVD/BLAS solve. When the response is classified as magnitude/cancellation sensitive, statgpu keeps the same SVD, numerical-rank cutoff, design scaling, and minimum-norm parameterization, but evaluates the SVD response projection with the shared magnitude-tiered reduction. For example, the representable intercept tail in a response such as `[2**55, 1, -2**55]` is not silently reduced to zero. Legacy pooled $R^2$ centering uses the same range-safe working-scale policy as the standardized diagnostics when physical `y-mean(y)` subtraction would overflow.

### External comparison and validation ownership

We compare the public `PooledOLS` estimator with `linearmodels==7.0` for Driscoll-Kraay coefficients, covariance, and BSE, and for group-debiased clustered covariance. Coefficients use `rtol=2e-10, atol=2e-11`; covariance/BSE use `rtol=5e-9, atol=5e-11`. Definition-level HC, clustering, Driscoll-Kraay, default-bandwidth, and R `sandwich` checks are summarized in the [validation matrix](#external-reference-matrix-tolerances-and-historical-artifacts).

GPU consistency is tested separately by comparing CuPy and Torch outputs with NumPy at default `rtol=5e-6, atol=5e-7`; observed maximum differences are stored in the PR #126 physical validation artifacts. The dedicated `dev/benchmarks/validate_panel_hac_chronology_gpu.py` gate additionally checks ordered-categorical legacy-HAC chronology, a lexical-order negative control, formula missing-row alignment, and requested/executed CuPy/Torch backend identity on the final exact source. The additional `dev/benchmarks/validate_panel_intercept_cancellation_gpu.py` gate checks the cancellation-sensitive automatic-intercept path on both physical GPU backends.

### Implementation location

Implementation: `statgpu/panel/_pooled.py`.

## between-ols

### Entity-mean SVD/BLAS reduction and centering

The automatically added intercept uses the same cancellation-sensitive SVD response-projection guard as `PooledOLS`. Ordinary entity-mean responses keep the historical SVD/BLAS solve; magnitude/cancellation-sensitive responses retain the same SVD, rank cutoff, design scaling, and minimum-norm solution while replacing only the response projection reduction with the shared magnitude-tiered reducer. Legacy between $R^2$ centering also uses a range-safe working scale when physical `y-mean(y)` subtraction would overflow.

### External comparison and validation ownership

We compare `BetweenOLS` with `statsmodels==0.14.6` after constructing the same entity-mean regression in both packages. The checks cover coefficients and HC0/HC2/HC3 standard errors/covariances: coefficients use `rtol=5e-10, atol=5e-12`, and covariance/BSE use `rtol=5e-9, atol=5e-11`. Shared covariance checks are summarized in the [validation matrix](#external-reference-matrix-tolerances-and-historical-artifacts).

GPU consistency is tested separately by comparing CuPy and Torch results with NumPy using the Stage-C physical validation tolerance `rtol=5e-6, atol=5e-7`. The dedicated `dev/benchmarks/validate_panel_intercept_cancellation_gpu.py` gate additionally verifies the cancellation-sensitive entity-mean intercept path on both physical GPU backends.

### Implementation location

Implementation: `statgpu/panel/_between.py`.

## first-difference-ols

### Coefficient-resolution and response-projection implementation

The differenced regression uses the shared certified panel least-squares policy. Cancellation-sensitive response projections use the magnitude-tiered reduction path; if a nonzero coefficient is below the numerically certifiable float64 projection resolution and the candidate materially violates least-squares stationarity, `.fit()` raises `FloatingPointError` instead of returning a finite but unreliable coefficient. This is distinct from exact collinearity. If the differenced predictors are exactly collinear, fitted values can still be computed but the coefficient vector is not unique, so coefficient-level standard errors, tests, p-values, and confidence intervals are disabled.

### Dimensionless R-squared centering

Legacy `rsquared` is also range-safe. When the physical subtraction $\Delta y-\overline{\Delta y}$ would overflow near the float64 boundary, the response and residual are placed on a common dimensionless centering scale before the scale-invariant $R^2$ ratio is formed. Ordinary-scale centering is unchanged. Invalid covariance choices or unavailable explicitly requested GPU backends raise clear errors.

### External comparison and validation ownership

We construct the identical differenced sample in `statsmodels==0.14.6` and compare coefficients plus HC0/HC2/HC3 covariance and standard errors. Coefficients use `rtol=5e-10, atol=5e-12`; covariance/BSE use `rtol=5e-9, atol=5e-11`. Shared covariance checks are listed in the [validation matrix](#external-reference-matrix-tolerances-and-historical-artifacts).

GPU consistency is tested separately by comparing CuPy and Torch results with NumPy at default `rtol=5e-6, atol=5e-7`. The original page described an exact-head physical gate, without an artifact identity in that sentence, as additionally exercising shared coefficient-resolution fail-closed behavior and an extreme differenced-response case whose physical centering would exceed float64 range.

### Implementation location

Implementation: `statgpu/panel/_first_diff.py`.

## random-effects

### Auxiliary SVD projection, variance scaling and quasi-demeaning certification

The auxiliary between/within regressions and final GLS solve use the shared cancellation-sensitive SVD response projection when the response has extreme magnitude cancellation; ordinary responses retain the historical BLAS path and the same SVD rank/minimum-norm policy.

Swamy-Arora variance-component arithmetic fails closed when a single float64 common residual scale would erase a nonzero within/between residual, including the case where the normalized residual survives but its square underflows before RSS accumulation. Quasi-demeaning is also certified against the algebraically equivalent `within + (1-theta)*mean` decomposition. If multiplication, addition, or a materially different transformed result would discard a nonzero component, `fit()` raises `FloatingPointError` rather than returning a finite but incorrect GLS coefficient. The positive square-root complement is retained before forming `theta=1-complement`, so the certificate still sees a representable complement when that subtraction rounds `theta` to exactly one. These are float64 representation limits, not alternative statistical definitions.

### External comparison and validation ownership

Random-effects coefficient estimates are **not** claimed to match another package exactly because statgpu uses its own Swamy-Arora variance-component construction. Instead, we take statgpu's quasi-demeaned $(X^*,y^*)$ regression and compare the resulting robust and Driscoll-Kraay covariance with `linearmodels==7.0`, and HC2/HC3 covariance with `statsmodels==0.14.6`. Covariance comparisons use `rtol=5e-9, atol=5e-11`; see the shared [validation matrix](#external-reference-matrix-tolerances-and-historical-artifacts).

GPU consistency is tested separately by comparing CuPy and Torch outputs with NumPy at default `rtol=5e-6, atol=5e-7`; observed differences are stored in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`. The then-new `dev/benchmarks/validate_panel_intercept_cancellation_gpu.py` gate additionally checks Pooled/Between cancellation-tail coefficients and the RandomEffects variance/quasi-demeaning fail-closed boundaries with explicit requested/executed CuPy and Torch backend evidence.

### Implementation location

Implementation: `statgpu/panel/_random_effects.py`.

## panel-ols

### External comparison and validation ownership

We compare one- and two-way fixed-effect Driscoll-Kraay results with `linearmodels==7.0`: coefficients use `rtol=2e-10, atol=2e-11`, and covariance/BSE use `rtol=5e-9, atol=5e-11`. The no-fixed-effect OLS path is compared with `statsmodels==0.14.6`, and one-way fixed-effect coefficients are also checked against R `plm==2.6-7`. Shared covariance and R tolerances are summarized in the [validation matrix](#external-reference-matrix-tolerances-and-historical-artifacts).

GPU consistency is tested separately by comparing CuPy and Torch outputs with NumPy using default `rtol=5e-6, atol=5e-7`; observed maximum differences are stored in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`.

### Implementation location

Implementation: `statgpu/panel/_fixed_effects.py`.

## diagnostics

### Backend normalization and public RSS restoration

For finite extreme-scale inputs, classical model F, pooling F, and Breusch-Pagan LM evaluate their scale-invariant quadratic reductions on backend-native normalized working values. Scalar and column centering use overflow-safe reduction-length scaling, while subnormal normalization avoids backend-specific division by a subnormal denominator. Public RSS metadata is restored to the original squared units when representable (and may be `inf` only when that squared quantity itself is outside float64); the test statistic is not allowed to become `0`, `NaN`, or `inf` merely because an avoidable intermediate overflowed or underflowed.

### Diagnostic external-test ownership

Where definitions overlap, the diagnostic and supporting covariance calculations are compared with pinned Python and R references: `linearmodels==7.0`, `statsmodels==0.14.6`, R `plm==2.6-7`, and `sandwich==3.1-3`. The corresponding checks live in the panel diagnostic tests and `dev/tests/test_panel_stage_c_r_external.py`.

### Implementation location

Implementation: `statgpu/panel/_diagnostics.py` plus the shared diagnostic-context helpers.

## fit-statistics

### CPU-suite and external-test ownership

These statistics are covered by the full CPU regression suite and by estimator-level comparisons with external packages. In particular, tests verify that selecting a robust covariance estimator changes coefficient inference without silently changing the meaning of the classical model F statistic.

### Implementation location

The calculations are implemented by the panel diagnostic/statistics helpers under `statgpu/panel/`.

## Documentation-only verification boundary

This relocation does not change numerical code, tests, algorithms, support, parameter defaults, covariance formulas or inferential targets. The intended check is that public pages retain their user-facing contract while private mechanisms and source-specific validation records live here. Any later numerical change must review and validate its own active contracts; this archive is not a replacement for that work.
