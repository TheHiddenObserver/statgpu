# Node-wise alpha implementation review — round 1

Reviewed target:

- target kind: PR #139 / branch implementation
- base: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- reviewed head: `9f50d5911f4044917ee39623a94618ee2c45ad18`
- default branch: `master`
- review mode: fresh implementation audit after initial hosted CI closure

Verdict at this head: **NOT CLEAN — fixes required**.

## Findings

### HIGH / MAINT — implementation diverged from the reviewed architecture by stacking three install-time contract layers

The reviewed plan explicitly preferred a maintained numerical helper and rejected adding a chain of new monkeypatch contracts solely for node-wise tuning. The first implementation had:

- `_nodewise_alpha_inference_contract.py` for public API/runtime replacement;
- `_nodewise_alpha_gpu_scope_contract.py` to repair the first layer's over-broad GPU interception;
- `_nodewise_precision_cache_contract.py` to patch precision builders again for caching.

Even though the repository already contains compatibility installers from #135/#138, three new layers materially increase import-order/call-stack risk and make ownership of the numerical path difficult to audit. The implementation must consolidate this before final review. At minimum, cache must become a normal helper and GPU scope/precision routing must have one explicit owner; any remaining compatibility installer must be justified as integration with the already-installed #138 capture architecture, not as a preferred new architecture.

### MEDIUM / ARTIFACT — hosted simulation recorded PR merge checkout SHA as though it were the implementation source SHA

The dedicated PR workflow checks out GitHub's synthetic merge commit. The first schema-v1 simulation artifact recorded only `git rev-parse HEAD`, so the artifact reported merge SHA `0af5af5...` rather than reviewed source head `9f50d591...`. Evidence must record source SHA separately from checkout/merge SHA and artifact names should use the source SHA.

### MEDIUM / PERF / MAINT — restored precision cache recomputed the full Gram on misses

The first cache wrapper computed `X.T @ X / n` to build a Gram-based cache key and then called the underlying precision builder, which computed the same Gram again. This adds an avoidable O(np^2) operation to every cache miss. Cache identity may use canonical design content (the historical implementation already hashed design data) or the builder must accept/reuse a precomputed Gram; do not double-compute it.

### MEDIUM / DOC — migration changelog is still missing

The model pages document the new public parameter and statistical-default change, but a default-changing inference migration also requires a durable changelog/release-facing record. Do not rely only on the Draft PR body or #134 documentation stack.

## Evidence at reviewed head

Hosted checks at `9f50d591...` were green for the initial numerical/API implementation:

- Tests #3432: success
- Maintenance compatibility #2445: success
- Gaussian inference backend-native #477: success
- Node-wise alpha inference #1: success

The dedicated simulation (12 deterministic replicates) reported equal aggregate old/new diagnostic coverage `0.979166...`; mean interval length changed from approximately `0.24724` to `0.23780`, with new maximum independent node-wise KKT residual approximately `3.27e-7`. This is migration diagnostics only, not a nominal-coverage theorem claim.

## Fix status after this review

Subsequent commits have already begun addressing the findings. A later clean verdict must re-review the new exact head; the green checks above do not transfer automatically.
