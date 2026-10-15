# ANOVA: comparing group means

> Language: English\
> Last updated: 2026-10-12\
> Switch: [Chinese](../../../cn/models/testing/anova.md)

<a id="anova"></a>
<a id="overview"></a>

## What question does ANOVA answer?

Analysis of variance (ANOVA) studies how the mean of a numerical response
varies across groups or combinations of factor levels. It uses the response
observations and their grouping or factorial design to separate variation
associated with the model from variation left unexplained. The resulting F
statistics and p-values test specified hypotheses about means or factor effects,
under the assumptions of the chosen analysis.

For example, several treatments may produce different sample mean responses
even when their population means are equal. ANOVA asks whether those differences
are large relative to the background variation. Treatment groups are one
application; groups can also represent observed categories, with causal
interpretation depending on the study design.

For one factor with $k$ levels, the omnibus null hypothesis is

$$H_0: \mu_1=\mu_2=\cdots=\mu_k.$$

A small p-value is evidence against that joint hypothesis under the model
assumptions. It does not identify a differing pair, measure the size of a
difference, or establish causality. Failure to reject also does not establish
that the means are equal: small samples or noisy measurements can hide relevant
differences. Equivalence requires a different question and analysis.

<a id="factors-and-design"></a>
<a id="two-crossed-factors"></a>

## Describe the factors, groups and experimental units

A **factor** is a variable used to define conditions or categories; its values
are **levels**. If fertilizer type has three levels and each plant receives one
type, the plants at each level form a group in a one-factor design. A plant is
the experimental unit only if treatment is assigned and observations are
independent at the plant level. Several leaves measured on one plant do not
become several independent treatment replicates.

Add a second factor, watering schedule, with two levels. Crossing every
fertilizer with every schedule gives six **cells**, one for each combination.
A balanced two-factor design has the same number of independent observations
in every cell, not merely equal total counts for each factor. If the same plants
are measured under multiple schedules, the resulting dependence requires a
repeated-measures or other correlated-data model.

A two-factor analysis has separate questions: the main effect of fertilizer
averages over watering levels; the main effect of watering averages over
fertilizer levels. An **interaction** means that a difference associated with
one factor changes across levels of the other. With interaction, an average
main effect can hide a reversal or a large difference in just one condition.
The [balanced two-way tutorial](anova-twoway.md) develops these effect-specific
hypotheses and their tests. A factorial analysis is not simply several unrelated
one-way tests.

<a id="variance-intuition"></a>

## Why compare variances to learn about means?

Within a group, observations fluctuate around their group mean. Across groups,
the sample means also fluctuate, both because of sampling noise and possibly
because the population means differ. The classical one-way model partitions
total squared deviations into **between-group** and **within-group** variation.
Dividing a sum of squares by its degrees of freedom gives a mean square, a
variation estimate that accounts for the number of independent pieces of
information used to compute it.

Under the classical equal-mean, common-variance null, the between-group and
within-group mean squares estimate the same error variance. Their ratio is an
F statistic: a much larger between-group mean square is evidence against that
null. The p-value is the probability, under the null model, of an F statistic
at least as large as the observed one. It is not the probability that the null
hypothesis is true.

This is an intuition for the **classical** variance decomposition, not one
universal formula for every ANOVA. In a balanced factorial model, each effect
has its own sum of squares and degrees of freedom, and is compared with the
appropriate residual mean square. Welch ANOVA instead weights group means by
their estimated precision and adjusts the statistic and reference degrees of
freedom for unequal variances. The [classical](anova-oneway.md),
[Welch](anova-welch.md) and [two-way](anova-twoway.md) pages give their respective
models and formulas.

<a id="assumptions"></a>

## Check the design and error assumptions

- **Independence comes from data collection.** Check what was sampled or
  randomized, and whether units share a subject, household, batch or other
  source of correlation. A large sample cannot repair pseudoreplication, and
  passing a distributional check cannot establish independence.
- **Normality concerns errors within groups or cells.** The classical exact F
  reference assumes normal errors with a common variance. A pooled histogram
  can look non-normal simply because valid groups have different means. Inspect
  distributions within groups and residuals from the chosen model; robustness
  to departures depends on group sizes, imbalance and the severity of outliers.
- **The variance assumption is method-specific.** Classical one-way and the
  supported balanced two-way tests use a common error variance. Welch permits
  unequal group variances, but still assumes independent observations and uses
  an approximate F reference derived for normal within-group errors.
- **The response and model must fit the question.** Welch does not repair severe
  outliers, repeated measurements or an unsuitable outcome model. Missingness
  also needs a reasoned treatment; silently omitting observations is not an
  assumption check.

These are statistical requirements, separate from an API accepting an array.
The functions do not diagnose independence, normality or outliers for you.
The [SciPy ANOVA notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
summarize the classical assumptions and the unequal-variance alternative.

<a id="choose-the-analysis"></a>
<a id="one-way-anova"></a>
<a id="welch-anova"></a>
<a id="two-way-anova"></a>

## Choose the analysis from the design

| Your question and design | Start here | Important boundary |
|---|---|---|
| Do independent groups defined by one factor have equal means, with a common error variance? | [Classical one-way ANOVA](anova-oneway.md), `f_oneway` | Group sizes may differ; independence and equal population variances still matter |
| Do independent groups have equal means when their variances differ? | [Welch ANOVA](anova-welch.md), `f_welch` | Uses separate variance estimates and fractional denominator degrees of freedom |
| How do two crossed factors affect the outcome, and does one factor's effect depend on the other? | [Balanced two-way ANOVA](anova-twoway.md), `f_twoway` | Every factor combination needs the same sample size; decide whether interaction belongs in the model |
| Which pairs differ under a common-variance model? | [Tukey HSD](anova-posthoc.md#anova-tukey), `tukey_hsd` | All pairwise comparisons share a pooled variance; unequal group sizes use Tukey–Kramer scaling |
| Which pairs differ when variances need not be equal? | [Bonferroni pairwise Welch tests](anova-posthoc.md#anova-bonferroni), `bonferroni` | All pairwise Welch tests form one comparison family; returned p-values are raw |
| How large is the sample separation? | [Effect sizes](#effect-sizes), `cohens_f`, `partial_eta_squared` | Effect size and statistical significance answer different questions |

Choose the model from the scientific question and sampling design, rather than
trying several methods and keeping the smallest p-value. Inspect group sizes,
distributions and variance differences before interpreting a test. Retain the
same design logic for follow-up comparisons: a Welch result does not justify
switching automatically to common-variance Tukey comparisons.

<a id="post-hoc-comparisons"></a>
<a id="tukey-hsd"></a>
<a id="bonferroni-pairwise-welch-tests"></a>
<a id="pairwise-comparisons"></a>

## From an omnibus result to useful conclusions

Keep three questions separate. An **omnibus test** asks whether a joint null
about population means or a factorial effect is compatible with the data. An
**effect size** describes the magnitude of an observed difference or pattern.
A **pairwise comparison** targets the difference between two specified means.
A small omnibus p-value answers the first question, not all three.

A significant omnibus test does not make unadjusted follow-up comparisons safe.
Define the comparison family and an appropriate simultaneous procedure before
selecting individual results to report. Choosing only promising pairs after
viewing their p-values does not remove the multiplicity problem. Conversely,
a prespecified simultaneous pairwise analysis need not wait for a significant
omnibus test. The [post-hoc tutorial](anova-posthoc.md) explains comparison
direction, adjusted decisions and confidence intervals;
[multiple-testing principles](multiple-testing.md) explain family-wise error
and false discovery rate more broadly.

<a id="effect-sizes"></a>
<a id="one-factor-and-effect-sizes"></a>

## Understand what an effect size measures

For a sums-of-squares model, ordinary eta-squared for an effect is

$$\eta^2=\frac{SS_{effect}}{SS_{total}}.$$

It describes the fraction of total **sample** variation associated with that
effect. Classical one-way `eta_squared` uses the between-group sum of squares;
the two-way result reports separate ordinary eta-squared values for factor A,
factor B and interaction. None is the probability that the null is false, a
causal proportion, or a population quantity known without sampling uncertainty.

Partial eta-squared uses a different denominator:

$$\eta_p^2=\frac{SS_{effect}}{SS_{effect}+SS_{error}}.$$

`partial_eta_squared(ss_effect, ss_error)` computes this ratio. In classical
one-way ANOVA it equals ordinary eta-squared because there is only one effect.
In a multifactor design it excludes the other effects from the denominator,
so it usually differs; partial values need not sum to one. Name the measure and
model when reporting it, rather than comparing ordinary and partial values as
though they were interchangeable. The [two-way example](anova-twoway.md#effect-sizes)
shows both denominators.

`cohens_f(*groups)` computes $f=\sqrt{\eta^2/(1-\eta^2)}$ using the classical
one-way calculation. Welch's `eta_squared` is deliberately `NaN`: this API does
not supply a Welch-specific effect-size estimate. Running `cohens_f` on those
same data still gives the classical descriptive quantity; it does not apply a
Welch variance adjustment. Neither helper supplies an effect-size confidence
interval.

Effect sizes complement means and differences in the response's original
units. A large sample can make a small effect statistically detectable; a noisy
small sample can leave an important effect uncertain. Conventional labels such
as “small” or “large” cannot replace domain knowledge.

<a id="reporting"></a>

## Report the question, magnitude and uncertainty

1. Describe the response units, factors and levels, independent sampling unit,
   group or cell sizes, and how missing observations were handled.
2. Name the model and variance assumption, and state the significance level
   chosen before inspecting results. For two factors, examine interaction
   before summarizing main effects averaged over the other factor.
3. Report group or cell means and spread, F, both degrees of freedom and the
   p-value. A non-significant result leaves uncertainty; it does not establish
   equality or equivalence.
4. Report a clearly named effect size alongside the original-scale differences
   and suitable uncertainty estimates. For pairwise claims, state the comparison
   family, correction and confidence-interval interpretation.
5. Keep conclusions within the study design. A test can describe association;
   causal interpretation additionally needs a design and assumptions that
   support it.

<a id="paths"></a>
<a id="parameters"></a>
<a id="parameters-1"></a>
<a id="output"></a>
<a id="output-1"></a>
<a id="signatures-fields-and-errors"></a>

## Supported scope

The module has **seven public functions**, not seven kinds of ANOVA: three
omnibus tests, two pairwise-comparison procedures, and two effect-size helpers.
The [complete API reference](../../reference/anova-api.md) inventories all seven
functions and all five result types, including `PairwiseComparison`.

The available designs are independent-group classical one-way ANOVA, Welch
one-way ANOVA, and balanced two-way ANOVA. `f_twoway` has no Type I/II/III
sums-of-squares selector and rejects unequal cell sizes. Repeated-measures
ANOVA, ANCOVA, MANOVA, arbitrary contrasts and higher-factor ANOVA are not
provided by these functions. Omitting interaction fits an additive model and
pools the omitted interaction variation into its residual; it is not merely
a display option.

All observation-based functions reject NaN and infinite input values rather
than silently deleting observations. Finite constant data can still yield
undefined or infinite statistics. Use the per-function
[input and degeneracy rules](../../reference/anova-api.md) to distinguish an invalid
input from an undefined statistic.

<a id="backend-and-execution-boundaries"></a>
<a id="strict-and-approximate-modes"></a>
<a id="limitations-and-failure-modes"></a>
<a id="does-torch-input-require-backendtorch"></a>
<a id="why-can-the-returned-p-value-be-a-python-scalar"></a>
<a id="why-is-an-unbalanced-two-way-design-rejected"></a>
<a id="faq"></a>
<a id="cpu-and-gpu-examples"></a>
<a id="optional-gpu-execution"></a>
<a id="cupy"></a>
<a id="torch-cuda"></a>
<a id="optional-gpu-examples"></a>

### Arrays, devices and numerical limits

Group reductions support NumPy, CuPy and Torch; results contain Python scalars
and comparison records. Some final distribution calculations run on CPU,
including Tukey's SciPy studentized-range calculation. `backend="torch"` does
not by itself require CUDA. For two-way nested cells, select the backend
explicitly. `dtype=None` uses the selected backend's float64; keep this default
for inference unless you have checked the numerical effect of lower precision.

The [API reference](../../reference/anova-api.md) is the complete guide to inputs,
dtypes, conversions and devices; the [device guide](../../guides/device-and-memory.md)
explains the separate estimator `device` interface. There is no selectable
strict/approximate ANOVA mode. The [classical GPU examples](anova-oneway.md#gpu-execution)
show explicit CuPy and Torch CUDA arrays without changing the statistical model.

<a id="anova-basic"></a>
<a id="compare-two-groups-on-cpu"></a>
<a id="external-validation"></a>
<a id="comparing-implementations"></a>
<a id="compare-with-another-implementation"></a>

## Continue with a worked analysis

Use the [classical one-way](anova-oneway.md), [Welch](anova-welch.md), or
[balanced two-way](anova-twoway.md) tutorial for a self-contained analysis of
your design. Each explains its formulas, outputs and relevant limits. When
comparing software, use the matching observations, grouping and variance model:
[classical comparison settings](anova-oneway.md#comparing-implementations) and
[Welch comparison settings](anova-welch.md#comparing-implementations) differ.

Contributor validation material is in the
[validation reference](../../../../dev/reviews/pr168-model-validation-provenance.md#anova).

## References

- [NIST: one-way ANOVA](https://www.itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm).
- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). [On the comparison of several mean values](https://doi.org/10.2307/2332579).
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
