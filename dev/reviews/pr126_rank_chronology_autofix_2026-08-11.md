# PR #126 rank/chronology auto-fix review checkpoint

Status: **PARTIAL_REMOTE_PENDING / NOT MERGE-READY**

Validation tier: `local-full` for the repaired local gates; fresh `remote-full` physical CUDA evidence is pending.

## Findings fixed

- `[CRITICAL][INFER][fixed] statgpu/panel/_covariance.py` — covariance pseudoinverse and numerical rank no longer use independent backend defaults. A single backend-native SVD mask with explicit float64 cutoff `max(m,n) * eps * s_max` now determines both the pseudoinverse and rank.
- `[CRITICAL][FORMULA][fixed] statgpu/panel/_first_diff.py` — FirstDifferenceOLS now orders `time_ids` through `factorize_panel_metadata`, preserving pandas ordered-categorical chronology instead of lexically sorting labels.
- `[HIGH][MATRIX][fixed locally / needs remote GPU] dev/benchmarks/validate_panel_stage_c_gpu.py` — the physical primitive matrix now retains the six prior checks and adds rank-boundary nonrobust, HC0, HC2, HC3, cluster, and DK checks. The rank-boundary fixture is explicitly numerical rank 2 with three design columns.
- `[LOW][ARTIFACT][fixed] dev/benchmarks/validate_panel_stage_c_gpu.py` — CuPy environment provenance now checks `cupy`, `cupy-cuda11x`, and `cupy-cuda12x` distribution names.

## Local focused validation

The self-deleting auto-fix workflow applied the patch and, before committing it, successfully ran:

- `dev/tests/test_panel_stage_c_covariance.py`
- `dev/tests/test_panel_stage_c_edge_contracts.py`
- `dev/tests/test_panel_stage_c_physical_runner_contract.py`
- `dev/tests/test_panel_stage_c_api_formula.py`
- `dev/tests/test_panel_stage_c_inference_guard.py`
- Python syntax compilation for the repaired production/runner files
- `git diff --check`

The dedicated maintained Torch-CPU workflow and the full permanent hosted matrix are triggered from this connector-authored checkpoint and remain part of the re-review gate.

## Physical evidence boundary

The earlier P100 measurement `ec511f539adeaaedf310f92248200d0868577532` predates the production fixes in `statgpu/panel/_covariance.py` and `statgpu/panel/_first_diff.py` and is therefore historical, not exact-source acceptance evidence for the repaired implementation.

Fresh physical correctness must execute the expanded 26-estimator + 12-public-primitive matrix on both CuPy and Torch. Because the shared covariance implementation changed, synchronized performance evidence must also be rerun before Stage C can return to `remote-full / COMPLETE`.

PR #126 remains Draft and unmerged.