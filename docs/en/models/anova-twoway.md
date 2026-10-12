# Balanced two-way ANOVA

> Language: English\
> Last updated: 2026-10-12\
> Switch: [Chinese](../../cn/models/anova-twoway.md)

<a id="the-question"></a>

## Two factors ask three different questions

Suppose you measure plant growth under two fertilizers and three watering
schedules. Every fertilizer is used at every watering level, giving six
combinations, or **cells**. With independent plants in each cell, two-way ANOVA
can separate three questions:

1. **Factor A:** do fertilizer means differ, averaging equally over watering levels?
2. **Factor B:** do watering means differ, averaging equally over fertilizers?
3. **Interaction:** does the fertilizer difference change with the watering level?

An interaction is a departure from additivity. If one fertilizer helps under
frequent watering but barely helps under sparse watering, a single average
fertilizer difference does not describe the whole pattern. Examine cell means
before summarizing an interacting system with main effects alone.

`f_twoway` supports a **balanced, crossed, fixed-effects** design: each factor
has at least two levels, every combination is present, and every cell has the
same number of observations. It does not provide unbalanced Type I/II/III
ANOVA, repeated-measures ANOVA, random-effects ANOVA, ANCOVA or MANOVA.
Use the [ANOVA overview](anova.md) to choose among the supported analyses.

<a id="assumptions"></a>

## Check the design before testing

The classical F tests assume independent errors, normally distributed within
each cell, with the same population variance across cells. Inspect cell
patterns and residual behavior; the function does not perform these checks.

- Equal cell counts are a computational requirement here, not evidence that
  variances are equal.
- Independent replicates mean different experimental units. Repeated readings
  from the same plant are not additional independent plants.
- A full interaction model needs replication within cells. With one observation
  per cell, it leaves no degrees of freedom for error and raises `ValueError`.
- Omitting interaction is a substantive additive-model assumption. It is not a
  way to hide an important interaction or automatically choose a smaller p-value.

<a id="model-and-tests"></a>

## Model, sums of squares and tests

For A level $i$, B level $j$, and replicate $r$, the full model is

$$y_{ijr}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijr}.$$

With $a$ A levels, $b$ B levels and $n$ observations per cell, $N=abn$.
Write $\bar y_{ij}$ for a cell mean, $\bar y_{i\cdot}$ and
$\bar y_{\cdot j}$ for row and column means, and $\bar y$ for the grand mean.
Balance lets us separate the following components:

$$SS_A=bn\sum_i(\bar y_{i\cdot}-\bar y)^2,\qquad
SS_B=an\sum_j(\bar y_{\cdot j}-\bar y)^2,$$

$$SS_{AB}=n\sum_{ij}(\bar y_{ij}-\bar y_{i\cdot}-\bar y_{\cdot j}+\bar y)^2,
\qquad SS_E=\sum_{ijr}(y_{ijr}-\bar y_{ij})^2.$$

Here $SS_T=SS_A+SS_B+SS_{AB}+SS_E$. The interaction is the part of cell-mean
variation that neither main effect explains. For each effect, divide its sum
of squares by its degrees of freedom, then divide by $MS_E=SS_E/df_E$:

| Term | Numerator degrees of freedom | F statistic |
|---|---:|---|
| A | $a-1$ | $(SS_A/(a-1))/MS_E$ |
| B | $b-1$ | $(SS_B/(b-1))/MS_E$ |
| A × B | $(a-1)(b-1)$ | $(SS_{AB}/((a-1)(b-1)))/MS_E$ |
| Error | $ab(n-1)$ | Common denominator |

Each p-value is an upper-tail F probability for its own effect. They are not
jointly multiplicity-adjusted. If your decision requires family-wise control
across several effects, outcomes or experiments, prespecify that family and
see [multiple testing](multiple-testing.md). The
[NIST two-way ANOVA reference](https://www.itl.nist.gov/div898/handbook/prc/section4/prc438.htm)
gives the balanced factorial decomposition.

<a id="anova-twoway"></a>

## A complete factorial CPU example

This walkthrough includes the full model, effect-size interpretation and an
additive-model comparison. Run its blocks in order; it has no dependency on
another page. Start with the imports.

<!-- example: anova-twoway -->
```python
import numpy as np
from statgpu.anova import f_twoway, partial_eta_squared
```

Build a nested list of cells. Rows represent fertilizer (A), columns watering
(B), and each cell is a one-dimensional array of 12 independent observations.
Use nested Python lists or tuples for the rows, rather than passing a raw 3-D
array. All six cells have the same size.

```python
rng = np.random.default_rng(11)
target_means = [[10.0, 10.5, 11.0], [10.0, 11.5, 13.0]]
data = [[rng.normal(mean, 1.2, 12) for mean in row]
        for row in target_means]
cell_means = np.array([[cell.mean() for cell in row] for row in data])
print(np.round(cell_means, 3))
```

Fit the model with interaction. Each test uses the same residual variance
estimate, and each returned eta-squared uses the total sum of squares.

```python
result = f_twoway(data, interaction=True, backend="numpy")
print(f"A: F={result.factor_a_statistic:.3f}, p={result.factor_a_pvalue:.3g}")
print(f"B: F={result.factor_b_statistic:.3f}, p={result.factor_b_pvalue:.3g}")
print(f"AB: F={result.interaction_statistic:.3f}, p={result.interaction_pvalue:.3g}")
print(f"residual df={result.df_within}, SS={result.ss_within:.3f}")
print(np.round([result.factor_a_eta_squared,
                result.factor_b_eta_squared,
                result.interaction_eta_squared], 3))
```

The observed cell means are approximately

| Fertilizer | Watering 0 | Watering 1 | Watering 2 |
|---|---:|---:|---:|
| A level 0 | 10.217 | 10.412 | 10.487 |
| A level 1 | 9.787 | 11.745 | 13.170 |

The full-model results are:

| Effect | F | p-value | Numerator df | Eta-squared |
|---|---:|---:|---:|---:|
| A | 24.963 | 4.55e-06 | 1 | 0.157 |
| B | 19.649 | 2.02e-07 | 2 | 0.248 |
| A × B | 14.192 | 7.47e-06 | 2 | 0.179 |

Residual degrees of freedom are `66` and residual sum of squares is `67.960`.
The interaction supplies evidence that the fertilizer difference depends on
watering. The second fertilizer is about `0.430` units lower at watering 0,
but about `2.682` units higher at watering 2. Main-effect tests summarize
averages across the other factor; they do not imply the same difference at
every level. These tests also do not provide confidence intervals or p-values
for the individual cell differences just described.

<a id="effect-sizes"></a>

## Ordinary and partial eta-squared answer different questions

The returned `factor_a_eta_squared`, `factor_b_eta_squared` and
`interaction_eta_squared` are **ordinary eta-squared**:

$$\eta_A^2=SS_A/SS_T,\qquad \eta_{AB}^2=SS_{AB}/SS_T.$$

For example, the interaction accounts for about 17.9% of total sample variation.
Partial eta-squared instead uses $SS_{effect}/(SS_{effect}+SS_E)$, excluding
other effects from its denominator. The API returns residual SS, so an effect's
SS can be recovered from its F statistic, numerator df and residual mean square:

$$SS_{effect}=F_{effect}\,df_{effect}\,(SS_E/df_E).$$

Continue with the same fitted `result` to compute the interaction's partial
eta-squared. This calculation assumes a finite, nondegenerate F statistic.

```python
mse = result.ss_within / result.df_within
ss_interaction = result.interaction_statistic * result.interaction_df * mse
partial_interaction = partial_eta_squared(ss_interaction, result.ss_within)
print(f"interaction SS={ss_interaction:.3f}")
print(f"partial eta_squared={partial_interaction:.3f}")
```

The interaction SS is `29.226` and partial eta-squared is `0.301`, rather than
ordinary eta-squared `0.179`. Neither quantity is a causal percentage or an
uncertainty interval, and partial eta-squared values should not be added across
effects. Use the same definition when comparing studies or software.

<a id="additive-model"></a>

## What changes when interaction is omitted?

With `interaction=False`, the fitted model is $\mu+\alpha_i+\beta_j$.
The unexplained interaction variation becomes part of its error term:

$$SS_{E,add}=SS_E+SS_{AB},\qquad
df_{E,add}=N-a-b+1=df_E+(a-1)(b-1).$$

Keeping only the within-cell error would discard variation the additive model
does not explain. Therefore both residual SS and residual degrees of freedom
change, and the main-effect F tests must be recomputed. This option does more
than suppress an interaction row in the output.

```python
additive = f_twoway(data, interaction=False, backend="numpy")
print(f"A: F={additive.factor_a_statistic:.3f}")
print(f"B: F={additive.factor_b_statistic:.3f}")
print(f"residual df={additive.df_within}, SS={additive.ss_within:.3f}")
print(additive.interaction_statistic, additive.interaction_pvalue)
```
<!-- example-end: anova-twoway -->

The additive residual SS is `97.186` (`67.960 + 29.226`, before rounding),
with `68` degrees of freedom. The main-effect F values become `17.985` and
`14.156`; interaction fields are `None`. Ordinary main-effect eta-squared values
stay the same because the balanced-design main-effect and total sums of squares
stay the same. Here the visible and statistically supported interaction argues
against reporting the additive model as an adequate summary.

<a id="parameters-and-pitfalls"></a>

## Parameters and limits to keep in view

- `data`: a rectangular nested list/tuple of one-dimensional cell samples.
  Supply all A × B combinations. Empty cells, inconsistent row lengths,
  unequal cell sizes, fewer than two levels for either factor, and non-finite
  observations raise `ValueError`.
- `interaction=True`: include and test interaction. With `False`, all four
  interaction result fields are `None`, and the additive residual includes
  interaction variation. One observation per cell can have positive additive
  residual df, but there is then no separate interaction-versus-error test;
  any inference relies on assuming additivity.
- `backend="auto"`: use `backend="numpy"` for this CPU tutorial. For GPU cells,
  select `"cupy"` or `"torch"` explicitly; this function's `"auto"` does not
  inspect nested cell arrays and resolves to NumPy. Torch tensors must already
  be on the intended device. See the [API reference](../reference/anova-api.md).
- `dtype=None`: uses backend float64. Retain it when small differences in large
  measurements could be sensitive to rounding.
- A zero residual mean square gives infinite F and zero p for a positive effect
  mean square, or undefined F/p when both are zero. This is a degenerate data
  configuration, not unlimited inferential certainty.
- Unequal cell counts are rejected because this API does not choose a Type I,
  II or III sums-of-squares convention. Do not delete observations merely to
  force balance; use a method that represents the actual design and hypothesis.
- These are fixed-effect tests, not random-effect variance components or
  repeated-measures tests. There is no formula interface or automatic simple-
  effects/contrast analysis in `f_twoway`.

For a significant interaction, define the scientifically relevant conditional
comparisons before choosing their error control. Treating each cell as a group
for an all-pairs procedure changes the comparison family and does not directly
reproduce the main-effect or interaction tests.

<a id="references"></a>

## References and next steps

- [NIST: models and calculations for two-way ANOVA](https://www.itl.nist.gov/div898/handbook/prc/section4/prc438.htm).
- [NIST: the hypotheses tested by ANOVA](https://www.itl.nist.gov/div898/handbook/prc/section4/prc43.htm).
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.
- [One-way and Welch ANOVA](anova-oneway.md), [post-hoc comparisons](anova-posthoc.md),
  and [complete ANOVA API reference](../reference/anova-api.md).
