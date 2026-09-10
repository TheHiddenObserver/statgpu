# Node-wise Lasso inference tuning migration

> Release status: the current published release is **0.2.5**. The node-wise tuning contract described here is implemented on current `master` and is targeted for **0.2.6**; published 0.2.5 does not yet expose `nodewise_alpha` or the new automatic default.

## What changed

Sparse Gaussian debiased inference now exposes a public `nodewise_alpha` control on `Lasso`, `ElasticNet`, their maintained Gaussian penalized base surfaces, and the final-refit inference configuration of `LassoCV` / `ElasticNetCV`.

The main model `alpha` and `nodewise_alpha` are different parameters:

- `alpha` controls the penalized prediction/selection fit;
- `nodewise_alpha` controls only the node-wise Lasso problems used to approximate the design precision matrix for debiased inference.

A finite positive scalar supplied by the user is authoritative. `nodewise_alpha=None` invokes statgpu's automatic rule.

## Intentional default correction

The historical internal implementation selected the node-wise penalty with the main-response residual scale,

$$
\hat\sigma_y\sqrt{\frac{2\log(\max(p,2))}{n}}.
$$

That rule was never a public tuning contract and made the design-side precision estimate depend on the units of `y`. Starting with the behavior targeted for **0.2.6**, the automatic rule standardizes the already canonical centered/weighted working design and uses

$$
\lambda_{\mathrm{nw}}
=
\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

where `n_nw=n` without analytic weights and a Kish-style effective sample size is used for non-uniform analytic weights. The order `sqrt(log(p)/n)` is theory-motivated; the exact constant and weighted effective-sample-size convention are statgpu defaults rather than a uniquely theorem-mandated choice.

There is no compatibility selector for the old response-dependent internal rule. If a specific node-wise penalty is required, set `nodewise_alpha=` explicitly on the standardized node-wise scale.

## Precision construction

Let `X_w` be the centered/weighted average-loss working design already defined by the sparse Gaussian inference path. Set

$$
d_j^2=\frac{1}{n}\sum_iX_{w,ij}^2,
\qquad Z=X_wD^{-1}.
$$

For each feature, statgpu solves the node-wise Lasso on `Z`, uses the paper-style normalizer

$$
\hat\tau_j^2
=
\frac{\|r_j\|_2^2}{n}
+
\lambda_{\mathrm{nw}}\|\hat\gamma_j\|_1,
$$

and transforms the resulting standardized approximate precision back to the working-feature scale.

The solver stopping condition is not the final numerical acceptance test. After the solve, statgpu independently recomputes the full KKT residual; only a solution that passes that check is used to produce inference results. Degenerate scales, non-finite precision state, KKT failure, or invalid normalizers fail closed instead of publishing an identity-row fallback.

For `p=1`, no nuisance node-wise regression exists; statgpu uses the analytic univariate precision and `nodewise_alpha_` remains `None`.

## Solver and provenance

The internal node-wise FISTA stopping tolerance is intentionally tighter than the final KKT acceptance threshold. The implementation uses `coef_delta` iteration stopping with a `1e-8` internal tolerance and an iteration budget of 3000, followed by an independent `1e-5` KKT check. These are internal numerical settings, not additional public tuning parameters.

Successful multi-feature debiased inference exposes the resolved value as `nodewise_alpha_`. `_inference_result.metadata` records the requested/resolved value, source, automatic-rule identifier, weighted effective sample size, node-wise solver settings, maximum KKT residual, cache provenance, and numerical backend/device.

## Cross-validation

`LassoCV(nodewise_alpha=...)` and `ElasticNetCV(nodewise_alpha=...)` treat the parameter as **final-refit inference configuration only**. It does not enter the main regularization grid, fold scoring, selected `alpha_`, or selected `l1_ratio_`.

## Backend and weighting contract

NumPy, CuPy, and Torch use the same standardized statistical definition. Explicit CUDA/Torch inference does not numerically fall back to CPU. Analytic weights preserve the existing average-loss convention, global positive weight-scale invariance, all-one-weight identity, and zero-weight-row invariance of automatic node-wise tuning.

The precision cache may hash backend-resident working-design data for cache identity, but the node-wise solve, KKT validation, precision back-transformation, and GPU simultaneous inference remain numerical operations on the selected backend/device.

## Version migration

- **Published 0.2.5:** no public `nodewise_alpha`; the node-wise penalty remains an internal implementation detail.
- **Targeted 0.2.6:** omitting `nodewise_alpha` uses the response-scale-independent automatic rule described above.
- **Reproducing an older experiment:** do not rely on the historical private formula as a compatibility mode. Set an explicit `nodewise_alpha` on the standardized node-wise scale and record the resolved `nodewise_alpha_` instead.

## References

- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217-242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869-2909.
