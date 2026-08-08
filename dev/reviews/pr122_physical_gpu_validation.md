# PR #122 Panel Stage B physical GPU validation

Validated implementation head: `636988751bcbfad3442d24d3073cdfcd2b3ac637`.

Canonical machine-readable evidence:

- source: `results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260808.json`
- SHA256: `882892c6e3077fe3b9f6084212647311da795fd05d1ed9f12ec53da1e05d0d4d`
- frontend source id: `panel-stage-b-pr122-20260808-882892c6e307`
- parser: `panel_stage_b_physical_validation` v1.0
- evidence type: validation/correctness/backend provenance only; no timing or speedup is inferred from this source.

Environment reported by the physical run:

- NVIDIA Tesla P100
- Python 3.9.16
- CuPy 13.6.0
- PyTorch 2.0.0

Top-level artifact contract:

- `schema_version = 2` in the physical runner output; the normalized frontend source records `runner_schema_version = 2` under its source schema v1.0 wrapper;
- `git_sha = 636988751bcbfad3442d24d3073cdfcd2b3ac637`;
- `working_tree_clean = true`;
- `status = success`.

CuPy acceptance:

- 15/15 model cases succeeded with `executed_backend = cupy`;
- balanced and unbalanced PooledOLS, unsorted-time HAC PooledOLS, BetweenOLS, FirstDifferenceOLS, one-way PanelOLS, RandomEffects, and FamaMacBeth all succeeded;
- balanced two-way PanelOLS succeeded;
- balanced and unbalanced Hausman diagnostic cases completed consistently with `applicable = false`.

Torch acceptance:

- 15/15 model cases succeeded with `executed_backend = torch`;
- balanced and unbalanced diagnostic cases succeeded;
- no silent CPU fallback was observed.

Stage-B checks passed for the maintained model matrix:

- parameter-based within/between/overall/adjusted R-squared;
- classical model F statistic and p-value where defined;
- PooledOLS Breusch-Pagan LM statistic/p-value;
- fixed-effects pooling F statistic/p-value;
- FE-vs-RE Hausman applicability/result parity;
- diagnostic covariance matrices.

Stage-A regression checks also passed for `coef`, `bse`, `tvalues`, `pvalues`, `conf_int`, `nobs`, and `df_resid`. CuPy and Torch results remained within machine-precision-scale differences of the NumPy reference, so the Stage-B integration did not regress the Stage-A coefficient-inference contract.

The benchmark frontend registers this evidence as 34 validation-only runs: 30 estimator/backend rows plus four Hausman applicability rows. The generated records contain CuPy/Torch backend provenance and validation/inference status but deliberately contain no `metrics.timing` or `metrics.speedup` fields.

This record captures the physical acceptance summary supplied for the exact clean implementation head. Subsequent PR commits may update evidence, parser, generated frontend assets, tests, or documentation; the physical numerical result remains applicable only while no Stage-B statistical implementation or physical-runner code changes after the validated implementation head.
