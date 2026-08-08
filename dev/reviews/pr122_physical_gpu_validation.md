# PR #122 Panel Stage B physical GPU validation

Validated implementation head: `636988751bcbfad3442d24d3073cdfcd2b3ac637`.

Environment reported by the physical run:

- NVIDIA Tesla P100
- Python 3.9.16
- CuPy 13.6.0
- PyTorch 2.0.0

Top-level artifact contract:

- `schema_version = 2`
- `git_sha = 636988751bcbfad3442d24d3073cdfcd2b3ac637`
- `working_tree_clean = true`
- `status = success`

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

This record captures the physical acceptance summary supplied for the exact clean implementation head. PR promotion still requires the final hosted/doc-only head to remain green after any evidence-only documentation updates.
