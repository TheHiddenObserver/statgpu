# Benchmark-data integration validation

## Source and evidence boundaries

The preserved local Git bundle retains commit `f90f5f8b42b62334e6c746ef3488f8a18a1f6217`
(tree `cd06048a4f03cfaa16ce125b73364e11006c8b9f`) as its ancestor. That commit
changes the campaign child thread setup, its precise manifest pin, and regression
tests. Its parent is the merged `91df0d9ebc35a49bde917383ba0680806016f3ae`.
The published branch instead contains `f1072b86f8e896937f2c0bc23abc0504e025aa92`,
which has that same tree and parent, but a different commit identity. The original
`f90f5f8b` physical results are source-tree-equivalent evidence, not a physical
rerun of the published commit. The local bundle ancestry must not be mistaken
for the published branch ancestry.
No `statgpu/` numerical implementation, statistical defaults, or numerical
acceptance thresholds are changed by this integration.

The following independent runs must not be joined into a same-SHA acceptance:

- Original source `91df0d9e`: 21 default jobs, 1,692 slots (1,676 passed and
  16 host-only not-applicable). Archive SHA256
  `07086b49b3e6a4d8b7a17259b561705a2a5cb0a5d6c3c5f9745cb904349c1b5b`.
  Its original preview remains 11 diagnostic jobs and 10 withheld jobs.
- Thread repair `f90f5f8b`: only the exact affected 10 jobs, 844 slots
  (836 passed and 8 host-only not-applicable), with all installed Torch runtimes
  controlled before benchmark imports and observed intra/inter-op thread counts 1.
  Archive SHA256 `1bbc72564ab30598f17a185592587cd0336efaeec6f36c1ae057a6041b27701a`.
  These jobs are NumPy/CuPy jobs, not 10 Torch-GPU jobs. All 10 diagnostic
  previews validate; 2,400 timing fields are not 2,400 independent benchmarks.

Offline conversion preserves all original raw bytes. The repaired old-data
normalizer yields 986 runs (938 original + 12 clean CV + 36 diagnostic phase
summaries), with 3,590 fields withheld from normalization. The new 10-job preview
has 996 normalizable runs and 1,404 withheld fields. Job qualification and field
normalization are different denominators. Phase diagnostics never become clean
fit timings; repeated summaries and sample aliases are not extra observations.

All results remain `UNVERIFIED_EXCLUSIVITY`, diagnostic-only, without canonical
performance eligibility or speedup comparisons. Prior conversational exclusivity
assurance does not populate either run's null operator attestation. Expanded and
independent plans were not executed by either run. No dashboard was deployed.

Warning review found 16 `cpu_solver` deprecation stderr warnings, plus 20 CSR
sparsity-efficiency and 4,000 CuPy scatter-add deprecation messages in learning
results. No numerical/convergence warning message was found. The 4,020 learning
messages lack original category/emitter-file/line metadata: message classification
is supported, but exact external emission sites were not independently proved.

## Local integration checks (2026-10-25)

- Real CPython 3.9.25, pytest 8.4.2, jsonschema 4.25.1, NumPy 2.0.2,
  SciPy 1.13.1 in a disposable CPU environment, not the user's P100 environment.
- Four focused files: frontend contracts, operation schema, qualification preview,
  optional Torch-thread child tests: **124 passed**.
- Three legacy frontend data/determinism/domain files: **39 passed** (disjoint).
- Frontend source, E2E and contract TypeScript checks passed; production build passed.
- Six operation-timing and four diagnostic-preview Playwright tests passed. These ten tests do not launch a browser.
- Browser E2E was not rerun; previous sandbox/browser launch limitation remains.
- The full numerical CPU suite was not rerun: this integration changes developer/data-display tooling, not production algorithms. Targeted parser/contract/thread tests cover the active scope.
- No numerical/GPU benchmark, new GPU test, hosted CI, push, or deployment was run
  for the integrated head. Prior physical acceptance is explicitly scoped above.

- Six relocated portable-tool test files: **101 passed**, no skips, both from
  the repository root and from the tool directory (same tests, not additive).
- Combined disjoint Python coverage is **264 tests** (124 + 39 + 101).
  The reviewer’s 96-test subset overlaps the 124 and is not added.

The final commit/source inventory accompanies the local Git bundle. Raw user result archives are intentionally not committed.
