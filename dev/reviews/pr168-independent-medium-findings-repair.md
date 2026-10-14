# PR168 independent-review repairs and native Torch model-X fix

## Scope and source

This follow-up starts from PR head
`7ef043218bf223b33071dde359714bd276318b18`, against comparison/merge base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. A fresh independent review found
two MEDIUM findings despite successful CPU tests and hosted CI: inaccurate
Torch model-X device documentation and an overbroad UMAP expected-failure guard.
Earlier clean conclusions are historical and do not override those findings.

The final scope includes the corresponding native Torch model-X implementation
repair, not merely documentation of its device exception. An intermediate
local documentation-only candidate was not published. Production changes are
limited to model-X random-allocation placement and local-generator use;
unrelated numerical, dispatch and fitted-state logic and dependency metadata
are preserved. UMAP production code is unchanged.

Existing issue [#242](https://github.com/TheHiddenObserver/statgpu/issues/242)
tracks the repaired Torch construction device/RNG problem. Existing issue
[#221](https://github.com/TheHiddenObserver/statgpu/issues/221) tracks the still
unrepaired UMAP neighbor problem. Neither a duplicate issue nor an automatic
issue closure is part of this work.

## Native Torch model-X implementation

Previously `_build_model_x_knockoffs` created a generator and random matrix
using default CUDA availability instead of the input tensor device, while the
covariance factor followed the input. CPU inputs on a CUDA-capable host, or
inputs on a nondefault CUDA device, could therefore reach mixed-device matrix
multiplication. Although a local generator was seeded, `torch.randn` did not
receive it and consumed global RNG state instead.

The repaired Torch branch creates both generator and random noise on the
input tensor's exact device and supplies the local seeded generator to the
draw. CPU construction consequently stays on CPU even when CUDA is available.
Public function, unified function and selector consumers share the same
helper. Same-seed reproducibility and different-seed draws are verified with
`corr_diff` independently of the separate seeded Lasso cache limitation.
NumPy and CuPy generator branches, covariance algebra, shrinkage and S-matrix
calculations are unchanged.

The pre-existing backend seed fallback is retained: `random_state=None` maps
to local seed zero for Torch construction. Since each public model-X draw
receives None under that setting, repeated construction draws reuse the same
noise. An explicit integer seed gives the existing distinct per-draw seed
schedule. This is stated in public help and both languages rather than
claiming that the previous buggy global-RNG output is preserved. No
cross-backend/device bitwise parity or empirical FDR claim follows from a
reproducible draw.

The obsolete six strict expected failures for ignored Torch construction
seeds become ordinary desired-behavior tests. New allocation/RNG regressions
cover CPU inputs with simulated CUDA availability, explicit/inferred backends,
public consumers, multi-draw behavior, global RNG isolation, independent
references and negative mutations. Optional real-CUDA checks are explicitly
hardware-gated; skipped checks are not physical execution evidence.

Both languages and public installed help describe the repaired contract.
Fixed-X generation, supplied-knockoff bypasses, compatibility routes and
statistical assumptions remain distinguished. The separate seeded Lasso
cache and downstream Lasso-statistic device routing remain outside this
construction repair. In particular, native Torch Lasso statistics can still
request the default CUDA device and do not inherit the construction-only CPU
and nondefault-GPU guarantee. Other documented numerical limitations are not
represented as fixed.

## UMAP expected-failure discrimination

The previous large-coordinate regression classified any fuzzy graph with fewer
than 12 nonzero entries as the known missing-neighbor defect. NaN weights,
negative weights and even empty graphs could become the same xfail.

The tightened guard validates shape, finite positive bounded weights,
symmetry, zero diagonal and the exact topology/weights of the known ten-edge
result. It recognizes the independently derived correct twelve-edge graph
first, so a real implementation repair reaches strict XPASS and requires
removal of the obsolete marker. Unrelated finite pollution, wrong topology and
malformed output fail normally. Production UMAP calculations are untouched.

Whole-file negative mutations target only large-coordinate output and retain
ordinary-scale controls. A test-only diagonal-mask repair and a separate
independent direct-distance construction demonstrate the desired positive
branch. These replays distinguish the known defect, a correct repair and an
unrelated regression instead of treating every reduced graph as expected.

## Verification and evidence boundaries

The final PR description records the frozen tree, full-path fingerprint,
independent complete-PR review, complete local CPU suite, documentation/static
checks, full remote tree verification and exact-new-head hosted results.
Earlier-head results are not relabeled as current.

All production Python files are compared against the base, with the deliberate
Torch generator/device change audited separately from documentation changes.
The existing plain `PenalizedCoxPHModel.__doc__` literal assignment retains its
target, position, guard and other executable expressions. Blanket claims that
all production logic is unchanged, or that every AST matches after stripping
initial docstrings, no longer describe this candidate.

No physical CuPy/Torch CUDA, R, benchmark, empirical FDR or coverage-calibration
run is established by this follow-up. CPU routing probes establish allocation
requests and CPU execution, not physical accelerator success. Skips and
expected failures are limitations rather than correctness passes. No merge,
auto-merge, retargeting, or Draft/Ready transition is included.
