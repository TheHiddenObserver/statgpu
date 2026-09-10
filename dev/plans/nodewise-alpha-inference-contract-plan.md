# Node-wise Lasso tuning contract — implementation plan

Status: **PLAN REVIEW CLEAN — READY FOR IMPLEMENTATION**

Target baseline:

- repository: `TheHiddenObserver/statgpu`
- implementation branch: `fix/nodewise-alpha-inference-contract`
- base `master`: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- affected capability: squared-error L1 / Elastic Net debiased inference and the approximate-precision construction used by marginal and simultaneous inference

Implementation has started on this branch. The reviewed technical contract remains the source of truth for runtime work.
