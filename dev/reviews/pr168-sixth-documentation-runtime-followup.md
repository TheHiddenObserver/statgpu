# PR168 sixth documentation and runtime-contract follow-up

## Scope and source identity

This is a fresh whole-PR documentation/test review, not an approval inherited
from the fifth cycle. The starting PR head is
`a4bf1c7df8fd92ebd55caf3b8cf73b3620d773cd`, tree
`e6ab7f59d224c06c16d655c9e518d762d2fc8979`, compared with base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`.

The review applies `.claude/skills/code-review/`, `dev/AGENTS.md`, and
`dev/DOCUMENTATION_STYLE.md`. It covers the complete prior PR path set and
connected public claims: formulas/objective scales, constructor/method/result
inventories, runnable examples, natural Chinese, supported inference and device
semantics, runtime help, and regression quality. Production numerical repairs
remain separate issues. No merge or PR draft/ready-state change is included.

## Documentation corrections

- Added direct navigation from both language portals to the complete shared,
  linear-model, feature-selection, survival/smoothing, unsupervised,
  distribution, and inference API references. Corrected stale kernel-page and
  Torch-page link descriptions.
- Clarified shared global-device behavior: an estimator's `device="auto"`
  inherits a non-auto global policy rather than overriding it. Explicit
  non-auto estimator settings take precedence. Restored Torch and this global
  policy in the base constructor help; device-specific exceptions remain
  explicit.
- Reconciled connected Poisson, Ridge, SCAD and MCP pages with the updated GLM
  and inference guides. Ordinary Poisson auto/IRLS defaults are penalized;
  explicit `C=0` gives the unpenalized IRLS objective. Specialized SCAD/MCP
  wrappers do not accept `inference_method`; the generic Gaussian interface
  provides the explicitly requested oracle route. Examples distinguish fitting
  from the stage at which an unsupported inference request raises. Removed
  obsolete future/status assertions and completed affected constructor help.
- Documented training-derived centering for large-offset exact Ridge, including
  matched alpha scaling, reuse on held-out rows, and the raw-intercept reporting
  caveat.
- Documented very small coordinate units in covariance-factor KDE/regression,
  including training-derived coordinate scales, density/log-density Jacobians,
  unchanged regression responses, and the different meaning of numeric factors
  versus absolute widths. One-dimensional `nrd`/`nrd0` compensation and explicit
  per-feature-width controls are distinguished from the reproduced defect.
- Corrected penalized Cox's objective to `-ell/n + P(beta)`, where n counts all
  rows rather than events. A built-in L2 penalty is `alpha/2 * ||beta||^2`;
  matching canonical CoxPH's summed objective on the same data requires
  `lambda = n*alpha/2`. Penalized-family CV uses row-normalized validation
  likelihood; canonical CoxPHCV uses summed fold likelihood. Varying fold size
  prevents assuming that one fixed parameter mapping equates whole CV searches.
- Added common-positive-scaling NMF/MiniBatchNMF precautions and executable
  examples, keeping the same scale across streaming batches and later transforms
  and restoring original units without mutating the fitted dictionary. Relative
  error is checked; tiny absolute residuals alone can be misleading.
- Completed per-estimator unsupervised RNG input/state guidance. Explicit
  seed-conversion consumers are distinguished from consumers that inherit the
  installed NumPy `default_rng` seed forms; legacy `RandomState` acceptance is
  not presented as portable across all NumPy versions.

The English and Chinese changelogs describe documentation improvements and
current limitations, without claiming that the numerical defects were repaired.

## Independently reproduced implementation follow-ups

All three issues were checked against open and closed issue bodies and reproduced
independently before filing. The immutable starting numerical source is recorded
in each issue; NumPy CPU was used with Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0.

| Issue | Deterministic observation | Distinction from earlier work |
|---|---|---|
| [#239](https://github.com/TheHiddenObserver/statgpu/issues/239) | KDE/regression covariance stabilization has an absolute floor. Scaling the same nonsingular coordinates by `1e-9` changes scale-corrected density from about `0.243` to `0.001`; NW predictions change from about `±0.6385` to `±9.4e-7`. | Fixed-factor unit sensitivity, not #179 translation cancellation or #238 GAM penalty/nullspace stabilization. Independent Gaussian-mixture/SciPy and local-equation controls identify the symptom. |
| [#240](https://github.com/TheHiddenObserver/statgpu/issues/240) | Strictly positive rank-one NMF data scaled by `1e-12` reconstruct essentially as zero. Relative error becomes 1 for ordinary/mini-batch fitting and `partial_fit`, including `tol=0`. | Every feature is positive from the start; this is not #212's absent first-batch feature problem. One common positive scale restores the demonstrated rank-one controls. |
| [#241](https://github.com/TheHiddenObserver/statgpu/issues/241) | Optimized exact CPU Ridge uses unstable raw-moment centering. Adding `1e8` to features changes coefficient error by as much as 1.30466 and prediction error by 4.42396; weighted fits also fail. | Direct Ridge coefficients/predictions, not #195 PCA covariance or #229 sparse-Gaussian reporting totals. Centered normal equations use the exact statgpu `alpha*n`/`alpha*sum(w)` scale. |

The new strict expected-failure tests express desired repaired behavior. They
are not correctness passes, and safeguards in prose do not repair production.

## Preservation boundary

The 278 production Python files are compared with independently Git-blob-verified
comparison-base bytes. Ordinary module/class/function initial docstrings are
removed for the normal AST comparison. There is one explicit additional
**documentation-literal exception**:

`statgpu/linear_model/penalized/_penalized_cox_public_contract.py` assigns the
public `PenalizedCoxPHModel.__doc__` string. Only that literal's text changes to
state the true objective scale. The target, condition, assignment order and all
other expressions are unchanged, and the value is a plain string without
interpolation or effects. The dormant historical non-docstring class-body string
is left untouched. Genuine `CoxPartialLikelihoodLoss` help is corrected too.

Thus the numerical/dispatch/state logic is unchanged, but it would be inaccurate
to claim that every AST is identical after removing only initial docstrings.
The current path/count/fingerprint and exact Git-tree proof are recorded in the
PR description, including this exception.

## Regression and evidence strategy

New focused checks are in:

- `dev/tests/test_pr168_cross_page_cycle6.py`
- `dev/tests/test_pr168_linear_inference_cycle6.py`
- `dev/tests/test_pr168_survival_smoothing_cycle6.py`
- `dev/tests/test_pr168_unsupervised_cycle6.py`

They execute standalone learner examples, compare analytic Poisson score and
Cox tied likelihood/gradient/penalty-scale equations, verify unchanged training
preprocessing on new rows, inspect installed help and API navigation, and test
portable RNG semantics. Covariance-floor, NMF-collapse and Ridge raw-moment
markers use dedicated exceptions after shape/finiteness and known-symptom
checks. Unrelated runtime errors or different numerical failures must fail
normally; an intended repair must produce strict XPASS and require marker
removal. Mutation checks exercise both boundaries without editing production.

The complete original suite exposed three pre-existing Torch-CPU test-fixture
defects, independently reproduced against the original head. The two Gamma
private-solver tests now initialize `_nobs` as public `fit` does and additionally
assert promoted dtype, nonuniform weights and residual degrees of freedom. The
LassoCV test mocks both active backend-resolution boundaries with a real
Torch-CPU backend and retains the requested Torch/CUDA routing assertions. No
production behavior or original assertion is weakened, and no hardware skip is
introduced. Repairs are limited to
`test_glm_weighted_inverse_gamma_boundary.py` and
`test_pr135_canonical_audit_regressions.py`.

This cycle restores original repository tests and their fixtures/helpers from
the immutable remote tree, checking exact Git-blob hashes rather than accepting
a partial local suite as complete. All 1,456 original tracked blobs are locally
materialized. Empty local Git metadata provides root discovery for the unchanged
import-provenance tests; it contains no staged files, commit or claimed HEAD.
Remote identities continue to come from the verified GitHub metadata and blob
inventory. Final tests use a fresh Python bytecode cache
so copied cache filenames cannot falsify warning-stacklevel/source checks.
Commands include the full `python -m pytest dev/tests -q --tb=short` suite,
`python dev/validation/check_docs_contracts.py`, focused Ruff checks and the
repository's prescribed high-signal lint/compile checks. Exact executed totals,
optional skips and any residual dependency gaps belong to the final PR report.

The independent reviewer rechecks the full frozen candidate after fixes, rather
than approving only incremental paths. Publication is verified against the full
expected Git tree and current branch lease; hosted CI must belong to the exact
published head (including verification of a synthetic merge tree when used).
Earlier runs remain historical after content changes.

No physical CuPy/Torch CUDA, R, performance benchmark, population-FDR or coverage
calibration evidence is claimed here. Optional Torch checks use CPU; successful
routing inspection is not physical accelerator execution.
