# Documentation family directory migration

## Scope

This documentation-only reorganization starts from commit
`180ddc41cef544639d5c3b874430cb39a690b375`. It preserves the `models` root while
organizing its active content by statistical family. It does not change estimator
implementations, API semantics, numerical behavior, dependencies, or workflows.

The English and Chinese trees use matching paths:

- `models/regression/`: regression, GLM, penalties, ordered, quantile, and robust regression
- `models/testing/`: ANOVA and multiple testing
- `models/covariance/`: covariance overview and seven method tutorials
- `models/nonparametric/`: KDE/kernel regression, kernel methods, splines, and semiparametric models
- `models/feature-selection/`: selection overview and Knockoff tutorials
- `models/coxph.md`: retained as a single survival topic
- `panel/README.md`: the former panel overview beside the existing panel guides
- `unsupervised/README.md`: the existing selection index, with the former overview's unique contracts merged into it
- `reference/losses.md`: low-level loss definitions

The root documentation index, short usage path, model-family index, and individual
family indexes serve different navigation levels. Existing method-specific code,
formulas, examples, warnings, and complete API references remain authoritative.
The existing covariance and feature-selection overviews serve as family READMEs;
the KDE/kernel-regression tutorial is not reused as an overview for all smoothing
methods. No further broad guides/reference restructuring is included.

## Compatibility contract

The repository's documentation entry points are ordinary GitHub Markdown files.
There is no documented automatic redirect or generated documentation router.
Directory landing pages therefore use `README.md`.

All 84 moved English/Chinese pages retain their old file as a compatibility page.
Each old explicit or rendered heading anchor has a link to its corresponding
canonical page and anchor. These compatibility files intentionally remain in the
old flat directory; the maintained navigation points at the new content.

The machine-readable [migration map](pr168-document-directory-migration-map.json)
records every old path, its new path, and the anchor mapping. Tests must read the
canonical substantive pages, rather than passing against compatibility pages.
Existing semantic and numerical assertions remain in force after path migration.

Actual Markdown links throughout the repository are updated to canonical paths,
including language switches and developer-document references. Historical source
manifests and textual provenance paths remain historical records. The Lasso
runtime docstring's existing documentation URL remains reachable through the
compatibility page, avoiding an unrelated edit to a production source file.

## Validation

Validation covers all PR 168 regression tests and the affected documentation,
learner-example, API-reference, language, and navigation tests. In addition to the
existing documentation contract and link-normalization checks, validation checks
real link fragments, bilingual pairing, family reachability, complete compatibility
anchors, and source preservation. The general documentation checker strips
fragments, so its success alone is not evidence of anchor correctness.

Code fences and display formulas from the moved pages are compared with the
baseline byte-for-byte. CPU examples are executed by the applicable existing
example tests. No new physical GPU, performance, empirical FDR, browser/MathJax,
or broad runtime correctness claim follows from a directory-only migration.
