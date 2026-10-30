# Development entrypoints and historical code

Use this index to find an existing entrypoint before adding another runner,
compatibility layer, or copy of an implementation. Project requirements remain
in [AGENTS.md](../AGENTS.md); this index does not replace them.

## Maintained entrypoints

| Task | Start here | Boundary |
| --- | --- | --- |
| Library implementation | [`statgpu/`](../../statgpu/) and [project map](../AGENTS.md#主要代码入口) | Reuse canonical backend, solver, inference, and CV helpers. Keep public imports and estimator contracts stable. |
| Focused regression tests | [`dev/tests/`](../tests/) | `pyproject.toml` selects this test tree. Read scripts before running them; some are remote or GPU runners rather than lightweight unit tests. |
| Review | [Code-review procedure](../../.claude/skills/code-review/SKILL.md) | Resolve the exact base/head and include dirty content when reviewing a worktree. |
| Registered benchmark campaigns | [`run_qualification_campaign.py`](../benchmarks/run_qualification_campaign.py) | Seven suites with explicit planning, execution, source identity, resource bounds, and evidence checks. |
| Registered and independent protocols | [`run_full_benchmark_matrix.py`](../benchmarks/run_full_benchmark_matrix.py) and [protocol catalog](../benchmarks/independent_benchmark_catalog.json) | A coordinator and its children are not separate statistical coverage. Distinct timing scopes remain distinct. |
| Bounded batches and diagnostic export | [Data-update tools](../benchmarks/data_update/README.md) | Follow the documented plan/execute/resume boundaries. Archived pinned helpers are not generic verifiers for a new source revision. |
| Dashboard source maintenance | [Dashboard maintenance guide](../../frontend/docs/benchmark-dashboard-maintenance.md) | Normalize stored evidence without promoting diagnostic results to performance certification. |

Existing specialized protocols and the lower-level coverage dispatcher remain
available. This index chooses starting points; it does not remove or rename CLI
entrypoints, change their arguments, or authorize a workload.

## Canonical contract owners and compatibility imports

Start at the owning definition when changing these existing contracts. The old
modules remain importable, but importing or reloading them does not replace a
function or method.

| Contract | Owning definition | Retained compatibility module |
| --- | --- | --- |
| Gaussian inference array/device conversion | [`_gaussian_inference._as_backend_array`](../../statgpu/linear_model/_gaussian_inference.py) | `_gaussian_inference_device_contract` |
| Direct Quantile IRLS validation | [`QuantileLoss.irls`](../../statgpu/losses/_quantile.py) | `_quantile_irls_validation_contract` |
| Low-degree Student-t two-sided reference routing | [`TDistributionBase`](../../statgpu/inference/_distributions_backend.py) | `_low_df_reference_contract` (also retains backend conversion helpers) |
| Adaptive Group Lasso nonsmooth curvature | [`solvers._utils._smooth_penalty_lipschitz`](../../statgpu/solvers/_utils.py) | `_adaptive_group_lipschitz_contract` |

Quantile IRLS and Student-t methods retain their historical `__wrapped__` and
`_statgpu_original` links to the unchanged unchecked numerical methods. Those
links are not new public APIs or permission to bypass validation in callers.

## Historical implementations are not interchangeable with compatibility APIs

- `statgpu/linear_model/legacy/` contains old standalone implementations **and**
  the still-used distribution compatibility module. Do not delete or exclude the
  whole package based on its directory name.
- Before moving an implementation out of the installed package, inventory
  production imports, public re-exports, old serialized object paths, and tests.
  Keep a compatibility import when required; a lack of in-repository callers is
  not proof that external callers do not exist.
- Compare module globals and exception behavior as well as function bodies;
  identical function ASTs alone do not establish interchangeable implementations.
- Prefer integrating a behavior-preserving runtime installer into its owning
  helper when the import order and all consumers are understood. If a private
  installer path is retained for compatibility, it should not keep wrapping an
  implementation that already owns the behavior.

## Preserve benchmark evidence while simplifying tooling

The [historical artifact index](historical-benchmark-artifacts.md) and
[independent protocol catalog](../benchmarks/independent_benchmark_catalog.json)
distinguish historical workload scripts, artifact processors, wrappers, and
current parameterized protocols. Read the catalog and source before discovery:
some historical scripts execute work at import time or overwrite fixed paths.

Archive related source, results, and provenance together. Preserve historical
path references and source/checksum pins; do not rewrite old evidence to make it
appear to describe a later refactor. Never move or modify a checkout while an
active campaign is using it. A refactor requires a separate source identity and
its own proportionate validation, not inherited acceptance from an old run.
