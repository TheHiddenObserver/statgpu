# ANOVA: comparing group means

> Language: English\
> Last updated: 2026-10-12\
> Switch: [Chinese](../../cn/models/anova.md)

<a id="anova"></a>
<a id="overview"></a>

## What question does ANOVA answer?

Suppose you measure a continuous outcome under several treatments. Their sample
means will differ even when their population means are equal. Analysis of
variance (ANOVA) asks whether the between-group differences are large relative
to the variation among observations within groups.

For one factor with $k$ levels, the omnibus null hypothesis is

$$
H_0: \mu_1=\mu_2=\cdots=\mu_k.
$$

A small p-value is evidence against that joint hypothesis under the model
assumptions. It does not identify a differing pair, measure the size of a
difference, or establish causality. Failure to reject also does not establish
that the means are equal: small samples or noisy measurements can hide relevant
differences.

## Why compare variances to learn about means?

The classical model writes each observation as a group mean plus random error.
Within-group variation estimates the error variance. Between-group variation
contains the same noise and, when the means differ, an additional signal. After
dividing both sums of squares by their degrees of freedom, their ratio is an
F statistic. A large ratio is difficult to explain by noise alone under the
null. The [one-way tutorial](anova-oneway.md) develops the decomposition and
shows how to interpret the test together with an effect size.

The assumptions concern the data-generating design and errors: independent
observations, approximately normal within-group errors for the usual F model,
and equal population variances for classical ANOVA. Normality is not a demand
that all groups pooled together look normal. Welch ANOVA relaxes equal
variances; it does not repair dependence, repeated measurements, severe
outliers, or an unsuitable outcome model.

## Choose the analysis from the design

| Your question and design | Start here | Important boundary |
|---|---|---|
| Do independent groups defined by one factor have equal means, with a common error variance? | [Classical one-way ANOVA](anova-oneway.md#anova-basic), `f_oneway` | Group sizes may differ; independence and equal population variances still matter |
| Do independent groups have equal means when their variances differ? | [Welch ANOVA](anova-oneway.md#anova-welch), `f_welch` | Uses separate variance estimates and fractional denominator degrees of freedom |
| How do two crossed factors affect the outcome, and does one factor's effect depend on the other? | [Balanced two-way ANOVA](anova-twoway.md), `f_twoway` | Every factor combination needs the same sample size; decide whether interaction belongs in the model |
| Which pairs differ under a common-variance model? | [Tukey HSD](anova-posthoc.md#anova-tukey), `tukey_hsd` | All pairwise comparisons share a pooled variance; unequal group sizes use Tukey–Kramer scaling |
| Which pairs differ when variances need not be equal? | [Bonferroni pairwise Welch tests](anova-posthoc.md#anova-bonferroni), `bonferroni` | All pairwise Welch tests form one comparison family; returned p-values are raw |
| How large is the sample separation? | [Effect sizes](anova-oneway.md#effect-sizes), `cohens_f`, `partial_eta_squared` | Effect size and statistical significance answer different questions |

Choose the model from the scientific question and sampling design, rather than
trying several methods and keeping the smallest p-value. Inspect group sizes,
distributions and variance differences before interpreting a test. If the same
participant appears in several groups, these independent-group functions do
not model that correlation.

## From an omnibus result to useful conclusions

1. State the population means or factorial effects you want to compare, and
   choose the significance level before inspecting results.
2. Check the design and assumptions, then inspect the omnibus statistic,
   p-value and degrees of freedom. In a two-factor model, examine interaction
   before summarizing main effects averaged over the other factor.
3. Report the observed means and a relevant effect size. An effect can be
   statistically detectable yet too small to matter in practice.
4. If pairwise claims are needed, define the comparison family and use an
   appropriate simultaneous procedure. Selecting only promising pairs after
   viewing their p-values does not remove the multiplicity problem.

A significant omnibus test does not make unadjusted follow-up comparisons safe.
Conversely, a prespecified simultaneous pairwise analysis need not be viewed as
a reward unlocked by a significant omnibus test. The
[post-hoc tutorial](anova-posthoc.md) explains the targets and confidence
intervals; [multiple-testing principles](multiple-testing.md) explain family-wise
error and false discovery rate more broadly.

## Supported scope

The module has **seven public functions**, not seven kinds of ANOVA: three
omnibus tests, two pairwise-comparison procedures, and two effect-size helpers.
The [complete API reference](../reference/anova-api.md) inventories all seven
functions and all five result types, including `PairwiseComparison`.

The available designs are independent-group one-way ANOVA, Welch one-way ANOVA,
and balanced two-way ANOVA. `f_twoway` has no Type I/II/III sums-of-squares
selector and rejects unequal cell sizes. Repeated-measures ANOVA, ANCOVA,
MANOVA, arbitrary contrasts and higher-factor ANOVA are not provided by these
functions. Omitting interaction fits an additive model and pools the omitted
interaction variation into its residual; it is not merely a display option.

All observation-based functions reject NaN and infinite input values rather
than silently deleting observations. Finite constant data can still yield
undefined or infinite statistics. Use the per-function
[input and degeneracy rules](../reference/anova-api.md) to distinguish an invalid
input from an undefined statistic.

<a id="backend-and-execution-boundaries"></a>
<a id="strict-and-approximate-modes"></a>
<a id="limitations-and-failure-modes"></a>
<a id="does-torch-input-require-backendtorch"></a>
<a id="why-can-the-returned-p-value-be-a-python-scalar"></a>
<a id="why-is-an-unbalanced-two-way-design-rejected"></a>
<a id="faq"></a>

### Arrays, devices and numerical limits

Group reductions support NumPy, CuPy and Torch. Results contain Python scalars
and comparison records, not fitted estimators. Some final scalar distribution
calculations run on CPU; Tukey's studentized-range calculation uses SciPy.
`backend="torch"` selects the functional Torch backend and does not by itself
require CUDA. For two-way nested cells, select the backend explicitly rather
than relying on input-type inference. See the
[API reference](../reference/anova-api.md) for dtype, conversion and device
restrictions, and the [device guide](../guides/device-and-memory.md) for the
separate estimator `device` interface.

There is no user-selectable strict/approximate ANOVA mode. Moving a scalar
calculation to CPU does not change the test's variance assumptions or turn
Welch inference into classical ANOVA.

## Continue with a worked analysis

The detailed material previously on this page is organized below. Existing
section links remain usable and lead to the corresponding tutorial or reference.

<a id="one-way-anova"></a>
<a id="welch-anova"></a>
<a id="anova-basic"></a>
<a id="compare-two-groups-on-cpu"></a>
<a id="effect-sizes"></a>

### One factor and effect sizes

Use the [one-way tutorial](anova-oneway.md) for classical and Welch formulas,
self-contained CPU examples, result interpretation and effect-size caveats.

<a id="two-way-anova"></a>

### Two crossed factors

Use the [two-way tutorial](anova-twoway.md) for balanced cells, main effects,
interaction, sums of squares and a complete factorial example.

<a id="post-hoc-comparisons"></a>
<a id="tukey-hsd"></a>
<a id="bonferroni-pairwise-welch-tests"></a>

### Pairwise comparisons

Use the [post-hoc tutorial](anova-posthoc.md) for Tukey and Bonferroni examples,
comparison direction, adjusted decisions and simultaneous confidence intervals.

<a id="paths"></a>
<a id="parameters"></a>
<a id="parameters-1"></a>
<a id="output"></a>
<a id="output-1"></a>

### Signatures, fields and errors

Use the [complete API reference](../reference/anova-api.md) for every function,
parameter, default, result field, input restriction and backend boundary.

<a id="cpu-and-gpu-examples"></a>
<a id="optional-gpu-execution"></a>
<a id="cupy"></a>
<a id="torch-cuda"></a>

### Optional GPU examples

The [one-way GPU examples](anova-oneway.md#gpu-execution) use the same statistical
question with explicit CuPy or Torch CUDA arrays.

<a id="external-validation"></a>
<a id="comparing-implementations"></a>

### Compare with another implementation

The [one-way tutorial](anova-oneway.md#comparing-implementations) explains which
variance assumptions and settings to align. Contributor material is in the
[validation reference](../../../dev/reviews/pr168-model-validation-provenance.md#anova).

## References

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
