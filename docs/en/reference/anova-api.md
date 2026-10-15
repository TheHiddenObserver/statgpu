# ANOVA API Reference

> Language: English\
> Last updated: 2026-10-12\
> This page: API reference\
> Switch: [Chinese](../../cn/reference/anova-api.md)

Start with the [ANOVA overview](../models/testing/anova.md) for concepts and method choice.
For complete worked analyses, see the [classical one-way](../models/testing/anova-oneway.md),
[Welch](../models/testing/anova-welch.md), [balanced two-way](../models/testing/anova-twoway.md),
and [post-hoc comparison](../models/testing/anova-posthoc.md) tutorials. This page lists
the complete public function and result-object interface, input requirements,
and numerical limitations.

<a id="public-api"></a>

## Public API

All seven functions are available from both `statgpu.anova` and `statgpu`.
The five result dataclasses are exported from `statgpu.anova`, not from the
top-level `statgpu` namespace.

| Function | Purpose | Return type |
|---|---|---|
| [`f_oneway`](#f-oneway) | Equal-variance one-way ANOVA | [`AnovaResult`](#anova-result) |
| [`f_welch`](#f-welch) | Unequal-variance one-way ANOVA | [`AnovaResult`](#anova-result) |
| [`f_twoway`](#f-twoway) | Balanced two-factor ANOVA, with optional interaction | [`TwoWayAnovaResult`](#two-way-anova-result) |
| [`tukey_hsd`](#tukey-hsd) | All pairwise pooled-variance comparisons | [`TukeyResult`](#tukey-result) |
| [`bonferroni`](#bonferroni) | All pairwise Welch tests with Bonferroni decisions and intervals | [`PosthocResult`](#posthoc-result) |
| [`cohens_f`](#cohens-f) | One-way Cohen's f from group observations | `float` |
| [`partial_eta_squared`](#partial-eta-squared) | Partial eta-squared from supplied sums of squares | `float` |

`TukeyResult.comparisons` and `PosthocResult.comparisons` contain
[`PairwiseComparison`](#pairwise-comparison) objects. These are seven functions,
not seven different ANOVA designs.

<a id="common-inputs"></a>

## Common input and parameter rules

The five `*groups` functions accept one array-like argument per group. Pass a
sequence of samples as `function(*groups)`, rather than as one combined argument.
Each group is converted to the chosen dtype and flattened with `ravel()` before
analysis. Unequal group lengths are allowed. A multidimensional argument becomes
one sample; it does not request column-wise, batched, multivariate, or
repeated-measures analysis.

| Parameter | Default | Meaning and accepted values |
|---|---|---|
| `*groups` | Required | At least two independent samples of finite real measurements. See the minimum sizes below. Group labels are their zero-based argument positions. |
| `backend` | `"auto"` | `"auto"`, `"numpy"`, `"cupy"`, or `"torch"`. Legacy aliases `"cpu"` → `"numpy"` and `"cuda"`/`"gpu"` → `"cupy"` are accepted; names are case-insensitive and surrounding whitespace is stripped. See [selection and devices](#backend-dtype-device). |
| `dtype` | `None` | Computation dtype; `None` uses the selected library's `float64`, including for float32 inputs. Prefer backend-native `float32` or `float64` dtype objects. |
| `alpha` | `0.05` | `tukey_hsd` and `bonferroni` only: finite real family-wise significance level strictly in `(0, 1)`. Intervals correspond to family confidence level `1 - alpha`. |

`backend`, `dtype`, and `alpha` are keyword-only in the `*groups` functions.
`f_twoway` instead accepts its parameters positionally or by keyword, and
`partial_eta_squared` has neither backend nor dtype controls.

| Function | Minimum observations |
|---|---|
| `f_oneway`, `cohens_f` | Every group has at least one observation; total `N` must exceed group count `k`. A singleton group is allowed when `N > k`. |
| `f_welch`, `tukey_hsd`, `bonferroni` | Every group has at least two observations. |
| `f_twoway` | At least two levels of each factor, equal nonempty cell sizes, and positive residual degrees of freedom. See [cell layout](#f-twoway). |

Use real numeric observations. The ANOVA functions do not uniformly reject
complex inputs before casting, or validate that an explicitly supplied dtype is
floating-point. A cast may discard information or the array library may reject
it. Integer and complex computation dtypes are not substitutes for the documented
floating-point calculation. Conversion errors retain the underlying library's
exception type.

<a id="backend-dtype-device"></a>

## Backend, dtype, and device behavior

- For `f_oneway`, `f_welch`, `tukey_hsd`, `bonferroni`, and `cohens_f`,
  `backend="auto"` chooses the first Torch tensor or CuPy array encountered among
  the group arguments. With only Python/NumPy inputs it chooses NumPy. It does
  not select a GPU merely because one is installed.
- For `f_twoway`, `backend="auto"` selects NumPy without inspecting the nested
  cells. Explicitly specify `"cupy"` or `"torch"` for those cell arrays.
- `backend="torch"` selects the Torch library, not a CUDA device. Torch CPU
  tensors can be used. For GPU group reductions, supply Torch CUDA tensors;
  ordinary lists/NumPy arrays follow Torch's default array-construction device,
  normally CPU. ANOVA functions have no `device` parameter and do not use the
  estimator `device="torch"` convention.
- Supply all groups/cells on the same backend and device. `f_oneway` and
  `f_twoway` preserve native tensors' devices during conversion rather than
  harmonizing mixed devices. Welch and post-hoc group preparation use a native
  input as a device reference, but arbitrary mixtures of array libraries or
  devices are not a portable input format.
- Explicit backend names do not silently select another array library. A missing
  CuPy/PyTorch installation raises `ImportError`; an unusable CUDA runtime or
  incompatible device combination can raise a library-specific runtime error.
  An invalid backend name raises `ValueError`.
- `dtype=None` means float64 computation, not preservation of the input dtype.
  Use `np.float32`/`np.float64`, `cp.float32`/`cp.float64`, or
  `torch.float32`/`torch.float64` with the matching library. In particular, do not
  assume a NumPy dtype object is accepted by every Torch function path. Float32
  can change sums of squares, p-values, and decisions near a threshold.

Group means, variances, and sums of squares use the selected array backend.
Final summaries are Python scalars; returning a scalar can synchronize a GPU.
Complete observation vectors are not moved to CPU solely for distribution
evaluation, but these functions are not guaranteed to execute entirely on GPU:

| Calculation | Distribution/scalar boundary |
|---|---|
| `f_oneway`, `f_twoway` | F-distribution evaluation uses the selected distribution backend. For Torch, this scalar step uses its default device, CUDA when available and otherwise CPU, rather than necessarily the input tensor's device. |
| `f_welch` | F-distribution evaluation uses the selected backend; Torch receives the prepared input device. |
| `tukey_hsd` | Studentized-range survival probabilities and quantiles use `scipy.stats.studentized_range` on CPU. Pairwise scalar calculations are on CPU. |
| `bonferroni` | Pairwise scalar Welch calculations and the NumPy/CPU t-distribution implementation are on CPU. |
| `cohens_f` | Uses `f_oneway`, including its p-value calculation, then converts eta-squared to a scalar effect size. |
| `partial_eta_squared` | Python/NumPy scalar calculation; no group-array backend. |

The F and t implementations can use lookup tables and numerical fallbacks;
lookup-table construction can involve SciPy on CPU. Extreme-tail subtraction
can lose accuracy, and very small tail probabilities may round to zero or
produce inaccurate/infinite critical values. See the [distribution guide's
tail limits](../guides/distribution-api.md#tail-accuracy-has-limits) and
[backend-specific numerical limits](../guides/distribution-api.md#luts-and-backend-specific-numerical-limitations).
ANOVA exposes no `use_lut`, strict/approximate, or tail-accuracy control. The
statistical definitions are the same across backends; numerical equality or
fully native GPU execution is not guaranteed.

<a id="finite-and-degenerate-inputs"></a>

## Finite and degenerate inputs

All observation-based functions, including `cohens_f`, reject `NaN`, positive
infinity, and negative infinity with `ValueError` after array conversion.
They do not omit missing observations and expose no `nan_policy`. Finiteness is
not an assumption check, and finite inputs can still overflow during arithmetic.
Inspect nonfinite outputs before interpreting a test.

| Situation | Returned result or error |
|---|---|
| `f_oneway`: all observations identical | `statistic`, `pvalue`, and `eta_squared` are `NaN`; degrees of freedom are still reported. |
| `f_oneway`: each group is constant, with different group means | `statistic=inf`, `pvalue=0.0`, `eta_squared=1.0`. |
| `cohens_f`: either preceding case | `NaN` for identical observations; `inf` for constant groups with different means. |
| `f_welch`: some but not all groups have zero sample variance | `ValueError`: the weighted Welch calculation is undefined. |
| `f_welch`: all groups have zero sample variance | `statistic=pvalue=NaN` when the group means are treated as equal; otherwise `statistic=inf`, `pvalue=0.0`. In both cases `df_within=N-k` is an integer and `eta_squared=NaN`. Equality here uses `max(abs(mean_i - mean_0)) <= 1e-12 * max(1, abs(mean_0))`. |
| `f_twoway`: residual mean square is zero | Each tested effect returns `(F, p)=(NaN, NaN)` if its mean square is zero, or `(inf, 0.0)` otherwise. With zero total SS, both main-effect eta-squared fields are `NaN`, and `interaction_eta_squared` is `None` even if `interaction=True`. |
| `tukey_hsd`: pooled MSE is zero | Each pair has a zero-width interval at its mean difference. Equal means give `pvalue=1.0`, `reject=False`; different means give `pvalue=0.0`, `reject=True`. |
| `bonferroni`: both variances in a pair are zero | The same zero-width-interval and p-value rules apply to that pair. A pair with only one zero variance is allowed. |
| `partial_eta_squared(0, 0)` | `NaN`, because the ratio is undefined. |

<a id="f-oneway"></a>

## `f_oneway`

```text
f_oneway(*groups: Any, backend: str = "auto", dtype: Any = None) -> AnovaResult
```

Tests equality of the `k` population means under independent observations,
normal errors, and a common population variance. The function does not check
these assumptions. `*groups`, `backend`, and `dtype` follow the
[common parameter rules](#common-inputs).

For total sample size `N`, group sizes $n_i$, group means $\bar y_i$, and
weighted grand mean $\bar y=\sum_i n_i\bar y_i/N$,

$$
SSB=\sum_i n_i(\bar y_i-\bar y)^2,\qquad
SSW=\sum_i\sum_j(y_{ij}-\bar y_i)^2,
\qquad F=\frac{SSB/(k-1)}{SSW/(N-k)}.
$$

Returns [`AnovaResult`](#anova-result), with F upper-tail p-value and
`eta_squared = SSB / (SSB + SSW)`. Raises `ValueError` for fewer than two groups,
an empty group, `N <= k`, or nonfinite observations. See [degenerate
cases](#finite-and-degenerate-inputs) when `SSW=0`.

<a id="f-welch"></a>

## `f_welch`

```text
f_welch(*groups: Any, backend: str = "auto", dtype: Any = None) -> AnovaResult
```

Tests equality of independent group means without assuming equal variances.
It uses sample variances $s_i^2$ with denominator $n_i-1$ and the Welch
correction. Each group must contain at least two observations. Other
parameters follow the [common rules](#common-inputs).

For $w_i=n_i/s_i^2$, $W=\sum_i w_i$,
$\bar y_w=\sum_i w_i\bar y_i/W$, and
$A=\sum_i(1-w_i/W)^2/(n_i-1)$,

$$
F_W=\frac{\sum_i w_i(\bar y_i-\bar y_w)^2/(k-1)}
{1+2(k-2)A/(k^2-1)},\qquad
df_1=k-1,\qquad df_2=\frac{k^2-1}{3A}.
$$

Returns [`AnovaResult`](#anova-result). Normally `df_within` is the
floating-point Welch–Satterthwaite `df_2`; if the computed adjustment `A <= 0`,
it is `inf`. **`eta_squared` is always `NaN`**: the pooled one-way ratio is not
reported as a Welch effect size. `cohens_f` does not provide a Welch effect
size either.

Raises `ValueError` for fewer than two groups, a group with fewer than two
observations, nonfinite data, or a mixture of zero- and positive-variance
groups. All-zero-variance behavior is specified [above](#finite-and-degenerate-inputs).
There is no trimming or correction toggle. For an aligned external comparison,
use the same samples with [statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html),
`use_var="unequal"`, `welch_correction=True`, and `trim_frac=0`.

<a id="f-twoway"></a>

## `f_twoway`

```text
f_twoway(
    data: Any,
    interaction: bool = True,
    backend: str = "auto",
    dtype: Any = None,
) -> TwoWayAnovaResult
```

Tests factor A, factor B, and optionally their interaction in a balanced
independent-observation design with common error variance and normal errors.

| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `data` | Required | Nested rows of cells, logically `(a, b, n)`: `data[i][j]` contains observations for A level `i`, B level `j`. Each row must be a Python list or tuple. Each cell is converted and flattened; all cells must have the same positive size `n`. |
| `interaction` | `True` | Boolean: include and test the interaction. If `False`, fit the additive model and pool interaction variation and degrees of freedom into the residual. |
| `backend` | `"auto"` | Common backend values; **auto chooses NumPy without inspecting cells**. |
| `dtype` | `None` | Common computation dtype; defaults to float64. |

A bare NumPy/CuPy/Torch array with shape `(a, b, n)` is not interpreted as the
nested cell layout: its array rows are treated as single cells and fail the
two-factor-level requirement. Preserve array cells inside Python row lists:

```python
from statgpu.anova import f_twoway

cells = [
    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    [[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]],
]
result = f_twoway(cells, interaction=True, backend="numpy")
```

There must be at least two levels per factor. The balanced decomposition is
$SS_{total}=SS_A+SS_B+SS_{AB}+SS_{within\ cells}$. Each tested effect uses
$F=(SS_{effect}/df_{effect})/(SS_{error}/df_{error})$.

| Model | Residual SS (`ss_within`) | Residual df (`df_within`) |
|---|---|---|
| `interaction=True` | $SS_{within\ cells}$ | $ab(n-1)$; requires `n >= 2` |
| `interaction=False` | $SS_{within\ cells}+SS_{AB}$ | $abn-a-b+1$; `n=1` is allowed when this is positive |

Omitting the interaction changes the denominator of both main-effect F tests;
it does not merely hide the interaction fields. With one observation per cell,
the additive analysis cannot separately estimate interaction and within-cell
error. Choose that model on substantive grounds.

Returns [`TwoWayAnovaResult`](#two-way-anova-result). Reported effect sizes are
ordinary $SS_{effect}/SS_{total}$, **not partial eta-squared**. All four
interaction fields are `None` when `interaction=False`.

Raises `ValueError` for empty data/cells, rows with different column counts,
fewer than two levels of either factor, unequal cell sizes, nonfinite values,
or nonpositive residual df. Unbalanced designs are rejected; no Type I, II,
or III sums-of-squares selection is exposed.

<a id="tukey-hsd"></a>

## `tukey_hsd`

```text
tukey_hsd(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> TukeyResult
```

Computes all $m=k(k-1)/2$ pairwise mean comparisons under independent normal
errors and a common variance. Each group needs at least two observations;
`alpha`, `backend`, and `dtype` follow the [common rules](#common-inputs).
Unequal sample sizes use the Tukey–Kramer standard error. The method does not
provide an unequal-variance/Games–Howell option.

With pooled $MSE=SSW/(N-k)$ and mean difference $d_{ij}=\bar y_i-\bar y_j$,

$$
SE_{q,ij}=\sqrt{\frac{MSE}{2}\left(\frac1{n_i}+\frac1{n_j}\right)},
\qquad q_{ij}=\frac{|d_{ij}|}{SE_{q,ij}}.
$$

The studentized-range distribution uses `k` groups and `N-k` residual df.
Each [`PairwiseComparison`](#pairwise-comparison) stores:

- `pvalue`: the studentized-range upper-tail probability, already adjusted for
  the family of all pairwise comparisons;
- `reject`: `pvalue < alpha`;
- `ci_lower`, `ci_upper`: $d_{ij}\pm q_{1-\alpha;k,N-k}SE_{q,ij}$,
  simultaneous confidence intervals.

Returns [`TukeyResult`](#tukey-result). Raises `ValueError` for fewer than two
groups, invalid `alpha`, undersized groups, or nonfinite observations.
Requires `scipy.stats.studentized_range`; if unavailable, raises `ImportError`.
See [zero-MSE behavior](#finite-and-degenerate-inputs). The returned p-value is
clipped to `[0, 1]`; if the distribution evaluation is nonfinite, the current
implementation substitutes `1.0` for `q=0` and `0.0` otherwise, so a finite
p-value alone is not a numerical-accuracy check.

For background on equal/unequal-size pooled-variance comparisons, see
[SciPy's Tukey HSD notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html).
The statgpu result type and options are the ones documented here.

<a id="bonferroni"></a>

## `bonferroni`

```text
bonferroni(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> PosthocResult
```

Computes all $m=k(k-1)/2$ independent-sample, two-sided Welch t-tests, with
Bonferroni adjustment of decisions and intervals. Equal variances are not
required. Each group needs at least two observations; the other parameters
follow the [common rules](#common-inputs).

For each pair, using sample variances with denominator `n_i - 1`,

$$
SE_{ij}=\sqrt{s_i^2/n_i+s_j^2/n_j},\qquad
t_{ij}=\frac{\bar y_i-\bar y_j}{SE_{ij}},\qquad
\nu_{ij}=\frac{(s_i^2/n_i+s_j^2/n_j)^2}
{(s_i^2/n_i)^2/(n_i-1)+(s_j^2/n_j)^2/(n_j-1)}.
$$

Returns [`PosthocResult`](#posthoc-result). In each comparison:

- **`pvalue` is the raw two-sided Welch p-value**, $2P(T_{\nu_{ij}}>|t_{ij}|)$
  capped at 1, not a Bonferroni-adjusted p-value.
- `reject` is `pvalue < alpha / m`, not `pvalue < alpha`.
- The interval is $d_{ij}\pm t_{1-\alpha/(2m),\nu_{ij}}SE_{ij}$.
  This uses the adjusted critical value, giving Bonferroni simultaneous
  coverage under the individual Welch approximations.

If an adjusted p-value is needed for reporting, compute
`min(1.0, comparison.pvalue * result.n_comparisons)` yourself. This does not
change the stored field. Pair-specific t statistics, standard errors, and df
are not stored in the result.

Raises `ValueError` for fewer than two groups, invalid `alpha`, undersized
groups, or nonfinite data. Both-zero-variance pairs have the behavior
[listed above](#finite-and-degenerate-inputs); a single zero variance does not
raise the mixed-zero-variance error used by `f_welch`.

<a id="cohens-f"></a>

## `cohens_f`

```text
cohens_f(*groups: Any, backend: str = "auto", dtype: Any = None) -> float
```

Computes the pooled one-way effect size
$f=\sqrt{\eta^2/(1-\eta^2)}$, using eta-squared from `f_oneway` on the supplied
groups. It has the same parameters, sample-size rules, errors, computation dtype,
and backend boundaries as [`f_oneway`](#f-oneway), including its F-distribution
evaluation. It does not take an ANOVA result or a precomputed eta-squared as
input, and is not a Welch-specific or partial effect size.

Returns a nonnegative Python `float` for an ordinary finite result, `inf` when
eta-squared is at least 1, or `NaN` when eta-squared is undefined. Conventional
0.10/0.25/0.40 descriptions are context-dependent heuristics, not test thresholds.

<a id="partial-eta-squared"></a>

## `partial_eta_squared`

```text
partial_eta_squared(ss_effect: float, ss_error: float) -> float
```

| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `ss_effect` | Required | Finite nonnegative real scalar: sum of squares for the effect of interest. |
| `ss_error` | Required | Finite nonnegative real scalar: residual/error sum of squares for the same fitted model. |

Both inputs are converted with `float()`. Inputs that cannot be converted raise
`TypeError`; nonfinite or negative converted values raise `ValueError`.
Returns the Python float
$\eta_p^2=SS_{effect}/(SS_{effect}+SS_{error})$, normally in `[0, 1]`, and
`NaN` when both sums of squares are zero. `ss_effect=0` with positive error
gives 0; positive effect with `ss_error=0` gives 1.

There is no backend, dtype, model-fitting, or degrees-of-freedom parameter.
The caller must supply compatible sums of squares. In a one-way design this
ratio equals ordinary eta-squared; in a multi-factor design it generally does
not equal the ordinary eta-squared fields in `TwoWayAnovaResult`.

<a id="result-objects"></a>

## Result objects and callable inventory

The following are plain dataclasses with required constructor fields in the
order shown. No field has a default, and constructing one directly does not
validate its statistical contents. Function-returned numeric summaries are
Python `float`/`int`, decisions are `bool`, and comparison collections are Python
lists, irrespective of the array backend.

**None of the five dataclasses has a `summary()` method or another public
statistical method.** Read their fields, use `print(result)` for the generated
representation, or use Python's `dataclasses.asdict(result)` for a dictionary.
They are not SciPy named-tuple results and do not provide tuple unpacking or a
`confidence_interval()` method; post-hoc intervals are already fields.

<a id="anova-result"></a>

### `AnovaResult`

```text
AnovaResult(
    statistic: float,
    pvalue: float,
    df_between: int,
    df_within: float,
    eta_squared: float,
) -> None
```

The signature shows the generated `__init__` return annotation; calling the
class constructs an `AnovaResult`. Returned by `f_oneway` and `f_welch`.

| Field | Runtime type | Meaning |
|---|---|---|
| `statistic` | `float` | Classical or Welch F statistic; may be `NaN` or `inf` in degenerate cases. |
| `pvalue` | `float` | F upper-tail probability; may be `NaN`. No rejection decision or alpha is stored. |
| `df_between` | `int` | Numerator df, `k - 1`. |
| `df_within` | `int` or `float` | `N-k` as an integer for `f_oneway`; normally fractional Welch denominator df, with all-constant exception described above. Its dataclass annotation is `float`. |
| `eta_squared` | `float` | `SSB / (SSB + SSW)` for `f_oneway`; always `NaN` for `f_welch`. |

<a id="two-way-anova-result"></a>

### `TwoWayAnovaResult`

```text
TwoWayAnovaResult(
    factor_a_statistic: float,
    factor_a_pvalue: float,
    factor_a_df: int,
    factor_a_eta_squared: float,
    factor_b_statistic: float,
    factor_b_pvalue: float,
    factor_b_df: int,
    factor_b_eta_squared: float,
    interaction_statistic: Optional[float],
    interaction_pvalue: Optional[float],
    interaction_df: Optional[int],
    interaction_eta_squared: Optional[float],
    df_within: int,
    ss_within: float,
) -> None
```

Constructs a `TwoWayAnovaResult`; returned by `f_twoway`.

| Field | Type | Meaning |
|---|---|---|
| `factor_a_statistic` | `float` | Factor-A F statistic using the selected model's residual mean square. |
| `factor_a_pvalue` | `float` | Factor-A F upper-tail p-value. |
| `factor_a_df` | `int` | `a - 1`. |
| `factor_a_eta_squared` | `float` | `SS_A / SS_total`; `NaN` if total SS is zero. |
| `factor_b_statistic` | `float` | Factor-B F statistic using the same residual mean square. |
| `factor_b_pvalue` | `float` | Factor-B F upper-tail p-value. |
| `factor_b_df` | `int` | `b - 1`. |
| `factor_b_eta_squared` | `float` | `SS_B / SS_total`; `NaN` if total SS is zero. |
| `interaction_statistic` | `float` or `None` | Interaction F statistic; `None` if interaction is omitted. |
| `interaction_pvalue` | `float` or `None` | Interaction F upper-tail p-value; `None` if omitted. |
| `interaction_df` | `int` or `None` | `(a - 1) * (b - 1)`; `None` if omitted. |
| `interaction_eta_squared` | `float` or `None` | `SS_AB / SS_total`; `None` if omitted **or total SS is zero**. |
| `df_within` | `int` | Residual df: `ab(n-1)` with interaction; `abn-a-b+1` without it. |
| `ss_within` | `float` | Residual SS for the chosen model, including interaction SS when `interaction=False`. |

Effect-specific sums of squares, residual mean square, and total SS are not
separate result fields. These p-values are individual effect tests, with no
automatic multiplicity adjustment across A, B, and interaction.

<a id="tukey-result"></a>

### `TukeyResult`

```text
TukeyResult(
    comparisons: List[PairwiseComparison],
    alpha: float,
    n_groups: int,
    df_within: int,
    mse: float,
) -> None
```

Constructs a `TukeyResult`; returned by `tukey_hsd`.

| Field | Type | Meaning |
|---|---|---|
| `comparisons` | `list[PairwiseComparison]` | All `k(k-1)/2` unordered pairs, with `i < j`, in order `(0,1), (0,2), …, (1,2), …`. |
| `alpha` | `float` | Family-wise alpha supplied to the function. |
| `n_groups` | `int` | Number of groups `k`. |
| `df_within` | `int` | Pooled residual df `N-k`. |
| `mse` | `float` | Pooled within-group mean square `SSW / (N-k)`. |

<a id="posthoc-result"></a>

### `PosthocResult`

```text
PosthocResult(
    comparisons: List[PairwiseComparison],
    alpha: float,
    n_comparisons: int,
) -> None
```

Constructs a `PosthocResult`; returned by `bonferroni`.

| Field | Type | Meaning |
|---|---|---|
| `comparisons` | `list[PairwiseComparison]` | All unordered pairs in the same order as `TukeyResult`. |
| `alpha` | `float` | Original family-wise alpha, **not** the pairwise threshold. |
| `n_comparisons` | `int` | `m = k(k-1)/2`; the pairwise threshold is `alpha / m`. |

<a id="pairwise-comparison"></a>

### `PairwiseComparison`

```text
PairwiseComparison(
    group_i: int,
    group_j: int,
    mean_diff: float,
    pvalue: float,
    ci_lower: float,
    ci_upper: float,
    reject: bool,
) -> None
```

Constructs a `PairwiseComparison`; elements of both post-hoc result lists.

| Field | Type | Meaning |
|---|---|---|
| `group_i` | `int` | First group's zero-based input index. |
| `group_j` | `int` | Second group's zero-based input index, greater than `group_i`. |
| `mean_diff` | `float` | Mean of group `i` minus mean of group `j`. |
| `pvalue` | `float` | Tukey: adjusted studentized-range p-value. Bonferroni: **raw** two-sided Welch p-value. |
| `ci_lower` | `float` | Lower endpoint for the `i - j` mean difference, adjusted by the corresponding procedure. |
| `ci_upper` | `float` | Upper endpoint for the same mean difference. |
| `reject` | `bool` | Tukey: `pvalue < alpha`. Bonferroni: `pvalue < alpha / m`. Equality at the threshold does not reject. |

<a id="limitations"></a>

## Scope and unsupported requests

- There is no formula/dataframe-label interface, weights, `axis`, `keepdims`,
  `nan_policy`, or estimator `fit`/`predict` interface in this module.
- Two-way ANOVA requires balanced cells and exposes no Type I/II/III choice.
  Repeated-measures ANOVA, ANCOVA, MANOVA, random/mixed-effects ANOVA, and
  higher-order factorial designs are not implemented by these functions.
- Tukey uses a pooled variance. Bonferroni uses pairwise Welch tests, with raw
  stored p-values and adjusted decisions/intervals. Neither function consumes a
  two-way fitted result or computes model-based marginal means/contrasts.
- The module does not test independence, normality, or equal variance for you.
  An omitted interaction, a small p-value, or a finite result is not evidence
  that the assumptions hold. The omnibus tests do not identify which pairs
  differ by themselves.
- No separate strict/approximate mode or numerical-tail policy is exposed;
  [backend and distribution limitations](#backend-dtype-device) still apply.

<a id="references"></a>

## References

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- [statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html): aligned Welch comparison settings.
- [SciPy `tukey_hsd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html): Tukey/Tukey–Kramer assumptions and interpretation; its API is separate from statgpu's.
- [statgpu distribution guide](../guides/distribution-api.md): distribution execution and numerical limits.
