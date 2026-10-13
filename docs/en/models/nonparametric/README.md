# Nonparametric Methods

> Language: English  
> Last updated: 2026-10-13  
> Switch: [Chinese](../../../cn/models/nonparametric/README.md)

[Models and Statistical Methods](../README.md) / Nonparametric methods

Choose what you need to learn: a density, a response curve, or new features.
Nonparametric methods still have assumptions; each tutorial explains its tuning,
input, prediction, and backend limits.

## Estimate a density or a local response curve

[KDE and kernel regression](nonparametric.md) starts with two different questions:
where observations are concentrated without a response, and how a continuous
response changes with predictors. It covers density estimation, local weighted
averages, local-linear regression, and bandwidth choices.

## Use similarities for prediction or representation

Read [Kernel methods](kernel-methods.md) for shared kernel ideas and method choice.
Then follow the tutorial for the output you need:

| Need | Tutorial | Choice cue |
|---|---|---|
| Predict a numeric response with a kernel | [KernelRidge](kernel-ridge.md) | Fit with a chosen regularization strength |
| Select that regularization strength using training folds | [KernelRidgeCV](kernel-ridge-cv.md) | Cross-validation followed by refitting |
| Nonlinear coordinates without a response | [KernelPCA](kernel-pca.md) | Learn an embedding rather than response predictions |
| A smaller, reusable kernel feature map | [Nystroem](kernel-nystroem.md) | Approximate the kernel with training landmarks for a later model |

## Build smooth terms or an additive model

- [Spline bases](splines.md): construct reusable basis features, including `SplineTransformer`; the basis itself does not fit a response.
- [GAM / semiparametric models](semiparametric.md): fit smooth additive effects for a continuous response. This implementation uses penalized least squares and does not expose a GLM `family` or `link` API.

For response-family models, return to [regression](../regression/README.md).
For other response-free representations and clustering, see
[unsupervised learning](../../unsupervised/README.md).

## Complete references

- [Kernel-methods API](../../reference/kernel-methods-api.md): four estimators, pairwise kernels, shapes, and numerical/backend limits
- [Survival and smoothing API](../../reference/survival-smoothing-api.md): KDE, kernel regression, bandwidth selection, spline, and GAM interfaces
- [Cross-validation](../../guides/cross-validation.md) and [device guide](../../guides/device-and-memory.md): shared workflow and backend guidance
