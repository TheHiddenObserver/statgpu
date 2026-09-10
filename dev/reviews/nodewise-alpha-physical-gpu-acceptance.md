# Node-wise alpha physical CUDA acceptance

Run this only from the exact candidate source intended for PR #139 acceptance.
Do not treat a dirty worktree, another branch, or a prior SHA as evidence for the
current review target.

## Environment

The validator requires both maintained GPU backends in one environment:

- CuPy with a visible CUDA device;
- PyTorch with CUDA available;
- the candidate statgpu checkout installed from the same worktree.

A Tesla P100 is acceptable; record the actual environment reported by the
validator rather than assuming a device from this prompt.

## Commands

```bash
git fetch origin
git checkout fix/nodewise-alpha-inference-contract
git pull --ff-only

git rev-parse HEAD
git status --porcelain

# Use the already prepared GPU environment; do not upgrade unrelated packages
# merely for this validation. Install this exact checkout without dependency
# churn when the environment is already complete.
python -m pip install -e . --no-deps

mkdir -p results/nodewise_alpha_gpu
python dev/benchmarks/validate_nodewise_alpha_gpu.py \
  --output results/nodewise_alpha_gpu/nodewise_alpha_gpu_schema_v1.json
```

Acceptance requires `git status --porcelain` to be empty before the run and the
validator to exit 0.

## Required cases

For **both CuPy and Torch** the schema-v1 artifact must prove:

1. automatic unweighted node-wise alpha follows the declared design-side rule;
2. unweighted CPU/GPU precision matrix and marginal params/BSE/p-values agree
   within validator limits;
3. explicit `nodewise_alpha=<auto resolved value>` reproduces the same numerical
   precision/inference while metadata reports `source="user"`;
4. rescaling only `y` does not change `nodewise_alpha_` or the precision matrix;
5. non-uniform analytic weights preserve CPU/GPU precision and marginal-report
   parity within validator limits;
6. globally rescaling analytic weights does not change the automatic precision
   problem;
7. `p=1` uses `precision_method="analytic_univariate"`, does not consume the
   requested node-wise penalty, and matches the CPU analytic precision/marginal
   report within validator limits;
8. intercept-inclusive simultaneous inference publishes finite intervals and
   reuses the same resolved node-wise alpha, validated precision matrix, and
   marginal parameter vector as the corresponding marginal fit;
9. marginal and simultaneous numerical backend/device provenance is the
   requested concrete GPU backend/device, not a silent CPU fallback;
10. the validator itself refuses to report success from a dirty worktree, so the
    artifact is evidence for the exact checked-out source rather than a modified
    local tree.

## Evidence to retain

Retain the JSON artifact and the console log. Report at least:

```text
HEAD SHA
worktree_clean
Python
CuPy
Torch
GPU name
CuPy case status
Torch case status
max KKT residuals
unweighted CPU/GPU max errors
weighted CPU/GPU max errors
p=1 CPU/GPU max errors
simultaneous precision/parameter reuse errors
marginal and simultaneous numerical device provenance
exit code
```

Do not relax validator thresholds or remove a failing case to obtain acceptance.
If a case fails, preserve the failing schema-v1 artifact and report the exact
exception/error metric so the implementation can enter another review/fix loop.
