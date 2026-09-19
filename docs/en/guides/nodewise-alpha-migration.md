# Node-wise Lasso Inference Tuning Migration

> Last updated: 2026-09-17  
> Switch: [Chinese](../../cn/guides/nodewise-alpha-migration.md)

Sparse Gaussian de-biased inference exposes a public `nodewise_alpha` control on `Lasso`, `ElasticNet`, the corresponding Gaussian penalized interfaces, and the final-refit inference configuration of `LassoCV` / `ElasticNetCV`.

## `alpha` and `nodewise_alpha` are different parameters

- `alpha` controls the penalized prediction/selection fit;
- `nodewise_alpha` controls only the node-wise Lasso regressions used to estimate the approximate design precision matrix for de-biased inference.

A finite positive value supplied by the caller is used directly. `nodewise_alpha=None` requests statgpu's automatic rule.

## Automatic rule

The automatic node-wise penalty is defined on the standardized centered/weighted working design. Its scale is

$$
\lambda_{\mathrm{nw}}
=
\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

where `n_nw=n` without analytic weights and a Kish-style effective sample size is used for non-uniform analytic weights.

The order $\sqrt{\log(p)/n}$ is motivated by high-dimensional node-wise regression theory. The exact constant and effective-sample-size convention are statgpu defaults rather than a unique theorem-mandated choice.

The previous internal response-scale-dependent rule is not exposed as a compatibility option. If a specific node-wise penalty is required, set `nodewise_alpha=` explicitly on the standardized node-wise scale.

## Precision-matrix construction

Let `X_w` denote the centered/weighted working design used by sparse Gaussian inference. statgpu standardizes its columns through

$$
d_j^2
=
\frac{1}{n}\sum_i X_{w,ij}^2,
\qquad
Z=X_wD^{-1}.
$$

For each feature, a node-wise Lasso regression is solved on `Z`. The residual normalization follows

$$
\hat\tau_j^2
=
\frac{\lVert r_j\rVert_2^2}{n}
+
\lambda_{\mathrm{nw}}\lVert\hat\gamma_j\rVert_1.
$$

The resulting standardized approximate precision matrix is transformed back to the working-feature scale.

Degenerate feature scales, non-finite precision state, an invalid normalizer, or failure of the post-solve optimality check cause inference to raise instead of publishing a placeholder precision row.

For `p=1`, there is no nuisance node-wise regression. statgpu uses the analytic univariate precision and `nodewise_alpha_` remains `None`.

## Resolved value

For multi-feature de-biased inference, `nodewise_alpha_` exposes the value actually used. This makes the automatic choice inspectable without turning the node-wise solver's internal stopping settings into additional public tuning parameters.

## Cross-validation

`LassoCV(nodewise_alpha=...)` and `ElasticNetCV(nodewise_alpha=...)` treat `nodewise_alpha` as **final-refit inference configuration only**.

It does not enter:

- the main alpha grid;
- fold scoring;
- selection of `alpha_`;
- selection of `l1_ratio_`.

The value matters only after CV has selected the prediction model and inference is run on the full-data final refit.

## Backend and analytic weights

NumPy, CuPy, and Torch use the same standardized statistical definition where the de-biased route is supported. An explicit CUDA/Torch inference request is not silently replaced by a CPU numerical calculation.

Analytic weights preserve the sparse Gaussian average-loss convention. In particular, multiplying all positive weights by the same constant does not change the automatic node-wise tuning target.

## Migration guidance

If you did not previously depend on an internal node-wise penalty value, leave `nodewise_alpha=None` and use the automatic standardized rule.

If reproducibility requires a fixed node-wise tuning value, set it explicitly:

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.05,
    compute_inference=True,
    inference_method="debiased",
    nodewise_alpha=0.08,
)
model.fit(X, y)

print(model.nodewise_alpha_)
```

Do not copy the main-model `alpha` into `nodewise_alpha` automatically: the two parameters solve different optimization problems and operate on different scales.

## Related documentation

- [Inference Modes](inference-modes.md) — choosing between de-biased, post-selection, and bootstrap inference
- [Cross-Validation](cross-validation.md) — selection and final-refit semantics
- [Lasso](../models/lasso.md) and [ElasticNet](../models/elastic-net.md) — model-specific inference controls

## References

- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217-242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869-2909.
