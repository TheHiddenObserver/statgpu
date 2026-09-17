# Documentation Language and Placement Policy

This file defines the repository-wide writing boundary between user-facing documentation and internal engineering material.

## Core rule

> **Internal documentation may describe capability/status; user-facing documentation should describe observable behavior/support.**

A reader of the public documentation should be able to answer what a feature does, when it is supported, what the default behavior is, what happens for an unsupported request, and why a limitation exists. They should not need to understand the repository's review, maintenance, evidence, or closure vocabulary.

## User-facing surfaces

Treat the following as user-facing unless a file clearly says otherwise:

- `README.md`;
- `docs/en/` and `docs/cn/` model, guide, API, reference, and public-design pages;
- public Python docstrings and `help()` output;
- user-visible warnings and exceptions;
- examples intended for ordinary library users.

On these surfaces, prefer language such as:

- `supported` / `unsupported`;
- `available` / `not available`;
- `solver="auto"` selects ...;
- an unsupported request `raises an error`;
- the current implementation uses ...;
- this combination is excluded because the algorithm requires ...;
- use X instead when Y is unsupported.

State the current observable contract directly. When a limitation has a mathematical or statistical reason, explain that reason rather than its project-management history.

## Public design / architecture pages

A user-facing design or architecture page may explain more than a task-oriented guide. It may describe:

- the stable execution model a user needs in order to interpret results;
- component or stage boundaries such as selection versus final refit;
- invariants that performance optimizations must preserve;
- conceptual acceleration strategies such as path reuse, batching, caching, or automatic backend choice;
- which aspects are deliberately allowed to change without changing the public statistical contract.

It should not promote current private implementation details into public guarantees. Exact helper names, cache-key fields, internal capacities, heuristic thresholds, source-level call graphs, synchronization placement, benchmark cutoffs, and validation/evidence ownership belong in `dev/` unless the user can directly configure or observe them as part of the public API.

A useful test is: **does the detail help a user understand what statgpu guarantees, or does it specify how the current source code happens to implement that guarantee?** The former can belong on a public design page; the latter belongs in internal engineering documentation.

## Internal engineering surfaces

Internal status language belongs in places such as:

- pull-request and issue descriptions/comments;
- `dev/` plans, reviews, validation notes, and evidence artifacts;
- test names/comments when they describe an engineering contract;
- benchmark provenance and exact-source records;
- internal implementation comments where lifecycle/history is relevant.

Terms such as `maintained route`, `fail closed`, `contract closure`, `evidence freshness`, `review-clean`, `exact-head validation`, and similar review/acceptance vocabulary are appropriate there when they carry useful engineering meaning.

## Terms to avoid in ordinary user documentation

Do not use internal acceptance/status wording as a substitute for describing behavior. In particular, ordinary user pages should normally avoid phrases such as:

- `maintained route` / `maintained implementation`;
- `维护中的路径` / `维护路径` when it only means an internally accepted path;
- `fail closed` / `fail-closed`;
- `closure` or `contract closure` as a support description;
- references to the current PR, issue ownership, exact commit/head, pending CI, or validation gate.

These are not globally forbidden words. They are acceptable when the lifecycle/status itself is part of a public policy—for example a documented deprecation/support lifecycle—or when the page is explicitly a changelog/release/evidence document. The test is whether the information changes something the user needs to know or do.

## Error and warning messages

Public errors should describe the unsupported request, the reason when useful, and the supported alternative. For example:

- Prefer: `FISTA-BB does not support Quantile loss because BB step sizes require smooth-gradient differences; use ordinary FISTA or IRLS as appropriate.`
- Avoid: `Quantile FISTA-BB is not a maintained route and fails closed.`

The same principle applies to warnings and runtime `help()` text.

## Documentation placement

User-facing model/guides/reference pages should contain:

- concepts and formulas;
- supported and unsupported combinations;
- defaults and dispatch behavior;
- failure semantics that users can observe;
- examples and alternatives;
- links to canonical compatibility or algorithm references.

Public design pages may additionally explain stable stage/component relationships and conceptual optimization strategies, provided they clearly distinguish those stable ideas from private implementation details.

Project-management status, review findings, follow-up ownership, exact-source evidence, remote execution details, and CI state belong in PR/issues, changelogs when release-relevant, or `dev/` evidence/review files.

A statement can be factually correct and still be a documentation defect if it is placed in the wrong layer.

## English and Chinese pages

EN/CN pages do not need word-for-word translation, but their capability claims and documentation layer must match. Internal review terminology should not be translated literally into the other language and then exposed to users.

### Chinese prose consistency

Chinese user-facing pages should read as natural Chinese technical documentation, not as English notes with Chinese connective words.

Keep English when it is the actual identifier or the established technical name readers need to map back to code, for example:

- public class, function, attribute, parameter, and enum/value names such as `LassoCV`, `sample_weight`, `cv_solver`, and `solver="auto"`;
- mathematical or algorithm names that are normally written as acronyms or proper names, such as Lasso, SCAD, MCP, FISTA, IRLS, ADMM, HAC, NumPy, CuPy, and Torch;
- a short English term in parentheses on first use when it materially helps map a Chinese concept to the literature.

Translate ordinary explanatory prose when there is a clear Chinese expression. Prefer, for example:

- `估计器` rather than prose-level `estimator`;
- `后端` rather than prose-level `backend`;
- `候选项` / `候选集合` rather than prose-level `candidate`;
- `调参网格` rather than prose-level `tuning grid`;
- `选择阶段` and `最终重拟合` rather than `selection` and `final refit`;
- `热启动` rather than prose-level `warm start`;
- `批处理` rather than prose-level `batching`;
- `缓存` rather than prose-level `cache`;
- `推断目标` rather than prose-level `inference target`;
- `已拟合状态` rather than prose-level `fitted state`;
- `模型专属` rather than prose-level `model-specific`.

Avoid embedding multi-word English noun phrases in an otherwise Chinese sentence when those words are not literal API identifiers. A sentence such as `selection cache 可以复用 selection evidence` should normally be written as `选择缓存可以复用已经计算的选择结果`.

Do not pursue artificial “zero English.” Terms that must match the API or established statistical literature should remain recognizable. The goal is **Chinese syntax and Chinese explanatory vocabulary around stable technical identifiers**.

## Review and regression protection

Documentation review should treat this boundary as part of correctness, not only style. When a PR changes user-facing support claims, reviewers should check both wording and placement.

Focused documentation-contract tests may prohibit known internal-status phrases or known prose-level English fragments on pages affected by a change. Such tests should be targeted: they must not reject legitimate API identifiers, code blocks, formulas, or standard algorithm acronyms merely because they are English.

Broader enforcement should be expanded deliberately as older documentation is cleaned up, rather than breaking unrelated pages solely to impose a repository-wide word ban in one step.
