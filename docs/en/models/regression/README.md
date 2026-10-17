# Regression

> Language: English  
> Last updated: 2026-10-17  
> Switch: [Chinese](../../../cn/models/regression/README.md)

[Models and Statistical Methods](../README.md) / Regression

Start with the response and the quantity you want to estimate. Then decide whether
coefficient shrinkage or a different loss is needed. The tutorials below provide
worked examples; their inference, solver, and device limits remain method-specific.

## Choose a response model

| Response or question | Tutorial | Choice cue |
|---|---|---|
| Continuous response, linear conditional mean | [LinearRegression](linear-regression.md) | Ordinary least squares and coefficient inference |
| Binary response | [LogisticRegression](logistic-regression.md) | Event probabilities and classification |
| Nonnegative integer counts | [PoissonRegression](poisson-regression.md) | Expected counts under a Poisson response model |
| Ordered categories | [Ordered logit/probit](ordered.md) | Use category order without assuming equal spacing |
| Other supported response families or a shared GLM interface | [Generalized linear models](generalized-linear-model.md) | Choose a family and link, then check ordinary and penalized interfaces |

## Choose shrinkage for a linear response model

| Need | Tutorial | Choice cue |
|---|---|---|
| Shrink coefficients without an L1 selection penalty | [Ridge](ridge.md) | L2 regularization and RidgeCV |
| A sparse linear fit | [Lasso](lasso.md) | L1 regularization, LassoCV, and inference choices |
| Combine L1 selection with L2 shrinkage | [Elastic Net](elastic-net.md) | Mixed penalties and ElasticNetCV |
| Adapt the L1 penalty using an initial fit | [Adaptive Lasso](adaptive-lasso.md) | Initial estimates and feature scales affect the penalty weights |
| Reduce shrinkage of large coefficients with a nonconvex penalty | [SCAD](scad.md) and [MCP](mcp.md) | Compare their penalty shapes and optimization limitations |

## Choose a different target or residual loss

- [Quantile regression](quantile.md): estimate a conditional quantile rather than a conditional mean.
- [Robust regression](robust.md): use Huber, Bisquare, or Fair loss when large response residuals need different treatment from least squares.

## Continue the workflow

- [Lasso inference](lasso-inference.md): de-biasing, node-wise precision and simultaneous intervals
- [Cross-validation](../../guides/cross-validation.md) for tuning and final refitting
- [Inference modes](../../guides/inference-modes.md) and [regression diagnostics](../../guides/regression-diagnostics.md) for interpreting and checking a fit
- [Linear-model API](../../reference/linear-model-api.md) and [shared estimator API](../../reference/estimator-api.md) for complete interfaces
- [Loss functions](../../reference/losses.md) for low-level loss definitions and numerical properties
- [Solver × penalty matrix](../../guides/solver-penalty-matrix.md) and [device guide](../../guides/device-and-memory.md) for supported configurations

For time-to-event responses, use [survival methods](../survival/README.md). For observations with
entity/time structure, use [panel methods](../../panel/README.md).
