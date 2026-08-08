# PR #122 Panel Stage B physical GPU validation

Validated implementation head: `faa95ce7fb5cb204088957fbda5544c20a06fbfc`.

Raw machine-readable evidence:

- artifact: `results/pr122_p100/panel_stage_b_gpu_validation_faa95ce7.json`
- raw SHA-256: `c1ba014a3b9bb0d32cbc0ca3d844ccfe767e7149189efb9ba2969f5bc1b94b31`
- repository commit carrying the immutable raw artifact: `806789ac7c1f71d7c91656b86dad91ab0f103582`
- runner schema: 2
- top-level `git_sha`: `faa95ce7fb5cb204088957fbda5544c20a06fbfc`
- `working_tree_clean = true`
- `status = success`

Canonical frontend evidence:

- source: `results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260808.json`
- canonical SHA-256: `b8caffa6f915facfb74965b6834b06d8aa6480cb0e3822640261a77ddd1ec9ba`
- frontend source id: `panel-stage-b-pr122-20260808-b8caffa6f915`
- parser: `panel_stage_b_physical_validation` v1.0
- evidence type: validation/correctness/backend provenance only; no timing or speedup is inferred from this source.

Environment reported by the physical run:

- NVIDIA Tesla P100-SXM2-16GB
- Python 3.9.16
- statgpu 0.2.4
- NumPy 1.24.2
- SciPy 1.10.1
- PyTorch 2.0.0
- CuPy CUDA execution is proven by `executed_backend = cupy`; the runner's `importlib.metadata` package-version lookup returned `null`, so the canonical evidence deliberately does not fabricate a CuPy version.

CuPy acceptance:

- 17/17 estimator cases succeeded with `executed_backend = cupy`;
- all four Hausman diagnostic cases succeeded;
- balanced and unbalanced PooledOLS, unsorted-time HAC PooledOLS, BetweenOLS, FirstDifferenceOLS, one-way PanelOLS, RandomEffects, explicit-constant RandomEffects, and FamaMacBeth succeeded;
- balanced two-way PanelOLS succeeded;
- no CPU fallback was observed.

Torch acceptance:

- 17/17 estimator cases succeeded with `executed_backend = torch`;
- all four Hausman diagnostic cases succeeded;
- no silent CPU fallback was observed.

Explicit-constant RandomEffects repair acceptance:

- `random_effects_explicit_constant_balanced` succeeded on CuPy and Torch;
- `random_effects_explicit_constant_unbalanced` succeeded on CuPy and Torch;
- the largest explicit-constant RE coefficient difference versus NumPy was `2.220446049250313e-16`;
- `random_effects_diagnostic_contract` had zero backend-vs-NumPy difference for every RE case.

Hausman acceptance:

- `hausman_balanced`, `hausman_unbalanced`, `hausman_explicit_re_constant_balanced`, and `hausman_explicit_re_constant_unbalanced` all matched NumPy on both GPU backends;
- all generated validation datasets produced a materially indefinite covariance difference, so the test was consistently structured-inapplicable with reason `covariance difference is not positive semidefinite` rather than reporting a fabricated statistic;
- the frontend parser distinguishes standard and RE-explicit-constant Hausman parameterizations so the two cases do not collapse to indistinguishable dashboard variants.

Stage-B checks passed for the maintained model matrix:

- parameter-based within/between/overall/adjusted R-squared;
- classical model F statistic and p-value where defined;
- PooledOLS Breusch-Pagan LM statistic/p-value;
- fixed-effects pooling F statistic/p-value;
- FE-vs-RE Hausman applicability/reason parity;
- diagnostic covariance matrices;
- RandomEffects explicit-constant diagnostic metadata.

Stage-A regression checks also passed for `coef`, `bse`, `tvalues`, `pvalues`, `conf_int`, `nobs`, and `df_resid`. All fields stayed within the runner's `rtol=5e-6`, `atol=5e-7` parity contract; coefficient differences remained at machine-epsilon scale.

The canonical frontend source represents 42 validation-only runs: 34 estimator/backend rows plus eight Hausman/backend rows. It deliberately contains no timing or speedup metrics.

The dedicated identity-overhead benchmark remains accepted from the immediately preceding implementation candidate because the `faa95ce7...` repair only changes the explicit-constant RandomEffects auxiliary-within branch, while the benchmark exercises PanelOLS and no-explicit-constant RandomEffects. Its measured digest/no-digest ratios remained approximately 1.04x-1.29x over the maintained target scales.

This exact-head raw artifact supersedes the earlier `636988...` canonical physical record and the failed `9c78bf66...` correctness attempt. Generated frontend assets and hosted staleness/e2e/production gates must be refreshed after canonical promotion before the stale-evidence review thread is closed.
