# Statistical Testing

> Language: English  
> Last updated: 2026-10-13  
> Switch: [Chinese](../../../cn/models/testing/README.md)

[Models and Statistical Methods](../README.md) / Statistical testing

Start with the hypothesis and the data's design. Comparing group means and
adjusting a collection of p-values are related tasks, but need different choices.

## Compare group means

Read the [ANOVA overview](anova.md) first for hypotheses, experimental units,
assumptions, and supported designs. Then choose the matching tutorial:

| Design or question | Tutorial | Choice cue |
|---|---|---|
| One factor with independent groups and a common-variance model | [Classical one-way ANOVA](anova-oneway.md) | Test whether all group means are equal |
| One factor with independent groups and unequal variances | [Welch ANOVA](anova-welch.md) | Account for unequal group variances |
| Two crossed factors with equal replication in every cell | [Balanced two-way ANOVA](anova-twoway.md) | Separate main effects and interaction |
| Which group pairs differ? | [Post-hoc comparisons](anova-posthoc.md) | Choose comparisons and their multiplicity adjustment |

These pages spell out input requirements and unsupported designs; a family label
alone is not evidence that a repeated-measures or unbalanced design is supported.

## Handle a collection of tests

[Multiple testing](multiple-testing.md) explains error criteria, adjustment
methods, and worked examples. Start here when you already have several p-values
and need to decide what error rate to control or what combining evidence means.

For selecting features with knockoff assumptions, use the separate
[feature-selection family](../feature-selection/README.md).

## Look up the complete interfaces

- [ANOVA API](../../reference/anova-api.md): functions, result fields, input restrictions, and device limits
- [P-value adjustment and combination API](../../guides/multiple-testing-combine-pvalues.md): signatures, axes, weights, and backend behavior
- [Inference API](../../guides/inference-api.md): related result objects, permutation tests, and bootstrap tools
- [Distribution API](../../guides/distribution-api.md): distribution functions used in statistical calculations
