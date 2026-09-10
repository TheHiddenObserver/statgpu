# Model documentation style guide

## Purpose

This document defines the canonical structure for public statgpu model pages.
The primary audience of a model page is a user who may know Python but does
**not** yet know the statistical or machine-learning method.

The first half of a model page must therefore teach the method before it acts as
an API reference. Solver dispatch, backend internals, validation tolerances, and
complete constructor inventories are important, but they must not displace the
conceptual path a new user needs.

Learner-first does **not** mean API-incomplete. A public model entry page must
still provide an exhaustive API inventory later on the page, or link to a
maintained dedicated reference page that provides it.

`docs/en/models/linear-regression.md` is the canonical simple-model example.
`docs/en/models/generalized-linear-model.md` is the canonical example for a
family of related models with substantial method-selection guidance.

## Documentation layers

statgpu documentation distinguishes three layers rather than forcing all
material into the first screen of one page.

### 1. Learner/model guide

This is the default public entry page for a method. It answers, in order:

- What problem does this method solve?
- What concrete situation motivates it?
- What is the core intuition?
- When should I use it, and when should I choose something else?
- What mathematical model/objective captures that idea?
- How do I run a minimal example?
- How do I interpret the result?
- Which parameters do I actually need to choose?
- How does this method differ from nearby alternatives?
- What are the most common mistakes?

The learner page should remain useful even if the reader never opens an
advanced reference page.

### 2. Statistical/API reference

Every public estimator must have a complete statistical/API lookup surface.
This may be an inline **Complete API reference** near the end of the learner
page, or a dedicated maintained reference page linked prominently from it.

The reference layer should cover, where applicable:

- the complete public constructor signature and every constructor parameter;
- the complete `fit` signature, including forwarded public keyword arguments;
- primary public methods such as `predict`, `score`, and `summary`;
- fitted attributes and their availability conditions;
- covariance and inference outputs;
- multi-output behavior;
- unsupported combinations and important failure conditions;
- family/link/penalty compatibility matrices when relevant;
- backend-specific behavior and transfer boundaries.

`linear-regression-inference.md` and `glm-family-reference.md` illustrate the
dedicated-reference form. Ridge, Lasso, and Elastic Net illustrate the inline
reference form.

The key rule is:

> **Curated teaching tables may be incomplete; the reference inventory may not.**

### 3. Numerical/implementation guide

Low-level algorithms and developer-facing details belong in method-specific
advanced sections or shared implementation guides. Examples include:

- FISTA, IRLS, Newton, L-BFGS, ADMM, coordinate-descent update equations;
- continuation/LLA internals;
- synchronization and device-transfer strategy;
- benchmark methodology;
- validation tolerances and external-comparison harnesses;
- source-file and private-helper names.

These details should not be required to understand why a method is useful.

## Canonical learner-page order

A normal model page should use the following order unless the method has a
strong reason to deviate.

### 1. Title and language switch

Keep metadata compact. Do not repeat the language switch twice.

### 2. What problem does it solve?

Open with two to four sentences in domain language rather than implementation
language. Describe the kind of outcome/data and the question the method answers.

Good:

> Lasso is useful when many predictors may be irrelevant and you want the fitted
> model to shrink some coefficients exactly to zero.

Avoid:

> `Lasso` provides an L1-regularized estimator with selectable CPU/GPU solvers.

The second statement is true but belongs later.

### 3. A concrete motivating example

For methods whose purpose is not obvious from the name, give a small practical
example before equations. Prefer a situation that exposes why a simpler method
is insufficient.

Examples:

- Lasso: 100 candidate predictors but only a small subset may matter.
- Quantile regression: the effect of experience may differ for the 10th, 50th,
  and 90th percentiles of wages.
- Robust regression: one extreme outcome can pull an OLS line away from the
  majority of observations.
- Panel fixed effects: comparing an entity with itself over time removes stable
  entity-level differences.
- PCA: many correlated variables may vary mostly along a few directions.

A tiny table or ASCII/Markdown schematic is encouraged when it clarifies the
idea faster than prose.

### 4. Intuition

Explain the method without requiring the formal objective. Introduce the key
mechanism in plain language.

The reader should be able to answer "what is the method doing differently?"
before seeing notation.

### 5. When to use it

State both positive and negative guidance.

Recommended structure:

- **Use it when**: 2-5 concrete conditions.
- **Prefer another method when**: 2-5 nearby failure modes or alternatives.

Avoid generic claims such as "use for large datasets" unless a scale threshold
or practical reason is given.

### 6. Model / objective / assumptions

Only now introduce the minimum mathematics needed to connect intuition to the
estimator.

Requirements:

- define every symbol that is not standard to the intended audience;
- explain what each objective term does;
- separate statistical assumptions from numerical requirements;
- do not lead with KKT conditions, Hessian blocks, covariance meat/bread, or
  private solver equations unless they are the essence of the method.

If a method has several equivalent interpretations, present the most intuitive
one first and label the others as alternative views.

### 7. How it works

Give a conceptual algorithm when the algorithm helps users understand the
method.

Prefer 3-6 conceptual steps. For example, Adaptive Lasso can be explained as:

1. obtain an initial coefficient estimate;
2. penalize apparently small coefficients more strongly;
3. solve a weighted L1 problem.

Detailed numerical update equations should link to an advanced section or
solver guide.

### 8. Minimal runnable example

The first code example must be self-contained whenever practical.

It should:

- import all required packages;
- construct or load its own small data;
- fit one normal, recommended configuration;
- print or expose the main outputs a beginner should inspect;
- use a fixed random seed when synthetic data is random;
- avoid requiring undefined `X`, `y`, `df`, `time`, or `entity_ids` variables.

The text immediately after the example must state what result the reader should
expect and why.

GPU examples belong later unless GPU use is intrinsic to the method.

### 9. How to read the result

Explain outputs semantically, not merely by shape.

For example:

- what does one coefficient mean?
- what natural scale does `predict()` return?
- what does a zero coefficient mean for a sparse method?
- what does a principal component represent?
- what does a cluster label *not* mean?
- what do standard errors, p-values, confidence intervals, or fit statistics
  mean under this model?

Mention important non-interpretations, such as "high R-squared does not imply
causality" or "cluster numbers have no ordinal meaning".

### 10. Key parameters and how to choose them

Do **not** reproduce the complete constructor docstring as the main teaching
table. Prioritize parameters that change the statistical meaning or normal
workflow.

The guidance column should answer "what happens when I change this?" and "how
should I choose it?"

Weak:

| Parameter | Description |
|---|---|
| `alpha` | Regularization strength |

Preferred:

| Parameter | Guidance |
|---|---|
| `alpha` | Larger values shrink coefficients more strongly. For prediction or feature selection, choose it with cross-validation rather than by hand. |

This table is explicitly **curated**. It must not be labeled as the complete API
unless it actually contains every public constructor parameter.

### 11. Compare with alternatives

Include a compact table for methods that are commonly confused.

Typical columns:

| Method | Main difference | Prefer it when |
|---|---|---|

Examples of useful comparisons:

- OLS / Ridge / Lasso / Elastic Net;
- logistic / probit / ordered logit;
- Poisson / negative binomial;
- mean / quantile / robust regression;
- pooled OLS / fixed effects / random effects;
- PCA / Truncated SVD;
- KMeans / Gaussian mixture / DBSCAN.

### 12. CPU/GPU and advanced controls

Only after the user understands the method should the page discuss:

- `device`;
- solver selection;
- exact versus approximate numerical paths;
- inference backend differences;
- performance guidance;
- formula/DataFrame convenience when it is not already central to the example.

A public solver table is useful when users genuinely choose among solver values,
but it should not appear before the motivating explanation and normal workflow.

### 13. Common pitfalls

List concrete mistakes that can produce a scientifically wrong interpretation,
not only API errors.

Examples:

- interpreting association as causation;
- treating a log-link coefficient as an additive effect;
- using Poisson under severe overdispersion without checking alternatives;
- interpreting ordinary post-selection p-values as valid selective inference;
- forgetting that fixed effects remove time-invariant regressors;
- treating PCA component signs or KMeans labels as intrinsically identified.

### 14. Complete API reference or reference link

A public learner page must end its modeling narrative with one of two options:

1. an inline `Complete API reference`; or
2. a prominent link to a maintained dedicated statistical/API reference page.

An inline complete reference should normally include:

- complete constructor signature;
- every constructor parameter and default;
- complete `fit` signature, including public forwarded keyword arguments;
- `predict`, `score`, `summary`, and other model-specific public methods;
- fitted attributes and conditions under which optional fields exist;
- important invalid combinations or fail-closed behavior.

For contract-managed pages, constructor parameter inventories should be checked
against source automatically. The current docs contract AST-checks Ridge, Lasso,
and Elastic Net English/Chinese inventories so source changes cannot silently
outgrow their API tables.

### 15. Advanced links and references

End with links to solver guides, validation documents, shared inference/API
guides, and primary literature where useful.

Validation file names and benchmark artifacts may be mentioned here, but should
not interrupt the beginner narrative.

## Method-family adaptations

The canonical order is a framework, not a rigid requirement.

### GLM/family pages

Put outcome-family selection near the top because choosing the distribution is
part of understanding the method. A family comparison table may replace the
single motivating example.

### Penalized methods

Explain the unpenalized baseline first, then explain what the penalty changes.
For L1/non-convex methods, visualize or describe sparsity before discussing KKT
conditions or proximal algorithms.

### Survival and panel methods

Lead with the data structure and estimand/identification idea. A correct
likelihood or covariance derivation is not a substitute for explaining what is
being compared.

### Unsupervised methods

Because there may be no response variable, emphasize the geometric or data-
representation goal and explain what outputs mean. State non-identifiability
facts (component signs, cluster labels) early enough to prevent misuse.

### Utility/reference pages

Pages whose explicit purpose is a family reference, covariance reference,
solver guide, loss reference, or compatibility matrix do **not** need to follow
the learner-page order. They should optimize for lookup accuracy instead.

## Content-quality requirements

A learner page is not considered migrated merely because headings were renamed
or a solver table was added. A migrated page must satisfy all of the following:

1. A reader unfamiliar with the method can explain its purpose before reaching
   the first detailed API section.
2. The page contains a motivating example or equivalent intuitive explanation.
3. The first runnable example is self-contained.
4. The page explains how to interpret at least the primary fitted outputs.
5. Parameter guidance explains choices, not only names/defaults.
6. At least the most likely competing methods are contrasted.
7. Scientifically important pitfalls are explicit.
8. Solver/backend/validation detail does not dominate the first half of the
   page.
9. English and Chinese versions preserve the same conceptual structure even
   when wording is not sentence-by-sentence identical.
10. Public claims match the current implementation and hosted validation.
11. The public API is exhaustively documented inline or through a maintained
    reference link; a curated key-parameter table alone is not sufficient.

## Review rubric

Review each learner page on a 0-2 scale for the following dimensions:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Problem framing | absent | terse/technical | clear user problem |
| Motivation / intuition | absent | partial | concrete and beginner-readable |
| Method-selection guidance | absent | one-sided | when-to-use + alternatives |
| Mathematical bridge | abrupt/undefined | correct but dense | minimal math with interpretation |
| Runnable example | incomplete | runnable but unexplained | self-contained + expected result |
| Result interpretation | absent | attribute list | semantic interpretation |
| Parameter guidance | docstring-like | mixed | choice-oriented |
| Alternative comparison | absent | prose only | compact actionable comparison |
| Pitfalls | absent | API-only | scientific + API pitfalls |
| Advanced-detail separation | dominates | mixed | learner-first, advanced linked |

A page is considered learner-first at **17/20 or higher**, with no zero in
Problem framing, Motivation / intuition, Runnable example, or Result
interpretation. **API completeness is an additional hard gate**, not an eleventh
scored dimension: a page cannot be considered fully migrated if its public API
reference is incomplete.

## Migration priority

Recommended first migration wave:

1. Ridge
2. Lasso
3. Elastic Net
4. LogisticRegression
5. PoissonRegression
6. QuantileRegression
7. Robust regression
8. CoxPH
9. PanelOLS
10. PCA and KMeans

This wave intentionally covers regularization, classification, count models,
robust/quantile estimation, survival, panel data, and unsupervised learning so
the template is tested across method families before remaining pages are
migrated in bulk.

## What should remain out of the learner-first half

Do not delete advanced information. Relocate or link it when it harms the
learning path. In particular, avoid placing the following before the normal
workflow unless the method cannot be understood without them:

- source paths and private helper names;
- full solver compatibility matrices;
- KKT tolerances and implementation-specific convergence thresholds;
- benchmark speedup claims;
- validation artifact filenames;
- complete covariance derivations;
- backend synchronization strategy;
- every fitted attribute shape;
- historical implementation fixes.

These items may and often should appear later in the complete reference layer.
The goal is not to make statgpu documentation less technical. The goal is to
make technical depth **progressive**: motivation first, working use second,
complete public API third, numerical implementation detail after that.
