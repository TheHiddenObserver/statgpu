# PR168: reader-oriented user-guide refinement

## Target and scope

This is a documentation/test-only follow-up to `e38fd27e280a203aea490c344cebf1962ef28352` on PR168. The freshly resolved original PR comparison base remains `3fba9af81db8624ab6e882cb153e7ede7ead7f63`. It is not a new whole-PR numerical approval. The initial export was checked against all 1,504 remote blobs and modes, rather than trusting a cached Git index.

The current public inventory is **161 pages**: root README, docs index, and active EN/CN user pages. All were read in full at paragraph level, including retained pages. **90 pages changed; 71 were retained.** The accompanying [inventory](pr168-user-guide-language-inventory.json) records 1,943 baseline section regions, 10,070 nonblank paragraph/block spans, per-page original/final hashes, retain/change reasons, and source traces. Spans include prose, examples, tables and formulas; their count is an indexing aid, not a claim that all blocks are prose.

The prior 162-page count included `docs/en/guides/statgpu_benchmark_dashboard_next_phase_plan.md`, which was already moved to developer documentation before this follow-up. It is not a missing current user page. Thirteen dashboard engineering contracts/plans and eight dated changelog/release archives are explicitly excluded and left intact. Repository contributor material is outside the user-guide inventory. Prior audit “clear” classifications were not reused as conclusions.

Public-content fingerprint: `bc7606d5b4eeb4fc5d699a2189d4b867e8f1dcf287b289ba363aa021750d0f25` (SHA-256 of sorted `path + NUL + per-file SHA256 + newline` records for all 161 pages). The record's own file is outside that content fingerprint. Final commit/tree identity and all-blob verification are recorded in the PR after publication.

## Changes and preserved boundaries

- Both device/memory guides now follow actual user tasks: choose a device; combine global and estimator settings; check known kernel/spline exceptions and use CPU alternatives; understand input/output conversion; manage GPU cache cleanup. A standalone CPU example is executable without accelerator packages.
- Removed DLPack/pinned-memory mechanics, private CPU thresholds, future-kernel evolution, intended-device promises, and instructions about which page should own content. Observable effects remain: global AUTO resolution and restoration, estimator `device="torch"` versus functional `backend="torch"`, kernel/spline CPU-placement exceptions, array checks, refusal of unwanted CPU results, explicit CPU workarounds, live-array/peak-memory limits, and reallocation costs.
- The global setting does not universally force local-AUTO penalized GLMs onto GPU. For KDE/kernel regression, CUDA/AUTO can select Torch when CuPy is unavailable; CPU inputs can then remain on CPU. These narrow caveats are source-based. No uniform behavior or new physical-GPU validation is implied.
- Applied the same reader-oriented standard to PyTorch/CV/inference/solver guides, model pages, Panel/unsupervised pages, entry pages and API references. Mathematical derivations, necessary low-level public APIs, actionable precision/resource limits, statistical provenance, known defects and workarounds remain.
- Corrected existing behavior descriptions where source verification was necessary: Panel standard residual degrees of freedom and its two within-R² definitions, failed-refit state, Quantile NumPy scoring, Cox CV error propagation, keyword-only nodewise controls, and qualified inference controls. No implementation was changed to match prose.
- Preserved model section structure and examples. Framework navigation headings keep their old anchors. Fixed a Chinese CV anchor and escaped absolute-value pipes in the two adaptive-L1 matrix rows so all four rendered cells retain the weighted penalty and controls.

## Regression protection and independent review

Five test files changed, including one new guard file. Existing phrase locks were adjusted to retain behavioral meaning rather than private machinery/editorial wording. The new tests execute both CPU examples, retain global/device/cleanup controls, protect the explicit CPU workaround within its paragraph, and inspect rendered adaptive-L1 cells. Negative controls remove CPU guidance or damage the rendered formula; they fail as expected.

Two independent reader-perspective reviews covered the complete changed-doc delta and the modified tests:

- Entry/guides/reference: 33 changed pages, both full device guides, 23 source files traced. The review corrected overly broad inference advice, stale documentation assertions and headings, and a missing CPU-workaround regression. Final scoped working-tree fingerprint: `e4f7ebd7ca4336ef95932192a65495007331e45119aa029dfefc7f988953080a`. Targeted suite: **139 passed, 36 CUDA skips, 4 expected numerical failures**. Independently rendered 90 tables and checked 546 local links; 356 display-math blocks were unchanged.
- Models/Panel/unsupervised: 57 changed pages across a 102-page inventory, plus the final Quantile documentation-test adjustment. Final combined scoped fingerprint: `e73892a6e9931640a90ec0b0c367c2f5a969bf06e99e8febaf626bbb1ba1df28`. Targeted suite: **46 passed, 1 optional linearmodels skip**; added Quantile check: **1 passed**. All **171/171** code-block, display-formula and reference preservation checks passed. Independent smoke tests confirmed failed-refit cleanup for six Panel estimators, the two PanelOLS within-R² definitions, and Huber error classes.

Both reviews found no unresolved in-scope issue after corrections. They cover this follow-up delta, not the full historical PR. Their counts overlap with the full suite and must not be added. Retained-page full reading belongs to the primary audit; the independent reviews checked retained pages as context/inventory rather than claiming a second complete 161-page audit.

## Final local validation

- Full test tree with Torch installed: **6,664 passed, 919 skipped, 166 expected failures, zero failed** (88 warnings, 122.87 seconds). The final run used the final documentation and five test-file bytes.
- Documentation contracts: **169 files passed**. Documentation path checks: **zero affected files**.
- All five changed Python test files: default Ruff, Python 3.9 grammar, and whitespace checks passed.
- Runtime: Python 3.12.14, Torch 2.14.1+cpu, NumPy 2.3.5, SciPy 1.17.0, sklearn 1.8.0, pandas 2.2.3, pytest 9.1.1. No local physical CUDA device was available.

All 161 current public pages were rendered with MarkdownIt's GFM table rule: **350 tables, 2,598 data rows, 1,879 local links**, with no inconsistent source cell counts, missing files, or missing heading/explicit-ID targets. Actual HTML cell assertions catch truncation of the adaptive-L1 formula. This is local rendering/anchor validation, not a deployed-site browser inspection.

## Evidence limits

All **286 production `statgpu/` blobs** and both `test_anova_finite_inputs.py` and `test_covariance_device_contract.py` blobs/modes match both `e38fd27` and `bf9833257d9726b4e8aaf199dc73a30cddd039c3`. The user's **222-case single-P100 acceptance remains user-reported and tied to bf983325**, with its original runtime and scope. It is not a new GPU run or evidence for changed documentation guards. **Fourteen two-GPU cases remain unrun.** Skips and expected failures are not passes.

No numerical production edit, new bug fix, Probit/#245 implementation, R run, speed benchmark, empirical-FDR or interval-coverage result is part of this follow-up. Existing issue warnings, including the RidgeCV custom-split workaround, remain. No merge, auto-merge, retargeting, or Draft/Ready transition is requested or performed. Hosted CI is checked on the eventual published head and is not inferred from baseline results.
