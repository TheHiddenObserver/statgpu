# GLM Solver Benchmark: 7 Families x 10 Penalties x 3 Backends

Date: 2026-06-23
Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)
Scale: N=100,000 x P=50
Reference: numpy + fista solver (median of 3 runs)

## Complete 3-Backend Table

| Family | Penalty | numpy best | spd | cupy best | spd | torch best | spd |
|--------|---------|------------|-----|-----------|-----|------------|-----|
| squared_error | none | admm | 1.1x | exact | 3.3x | exact | 3.5x |
| squared_error | l2 | exact | 1.0x | newton | 3.2x | exact | 3.4x |
| squared_error | l1 | fista | 1.0x | fista_bb | 2.9x | fista_bb | 3.0x |
| squared_error | elasticnet | admm | 1.0x | fista_bb | 2.9x | fista_bb | 2.9x |
| squared_error | adaptive_l1 | admm | 1.0x | fista | 1.0x | admm | 1.0x |
| squared_error | scad | fista | 1.0x | admm | 15.5x | fista | 15.6x |
| squared_error | mcp | fista | 1.0x | fista | 18.9x | admm | 19.1x |
| squared_error | group_lasso | admm | 1.0x | admm | 0.5x | admm | 0.6x |
| squared_error | group_scad | admm | 1.0x | fista_bb | 1.9x | fista_bb | 2.0x |
| squared_error | group_mcp | admm | 1.0x | fista_bb | 1.9x | fista_bb | 1.9x |
| logistic | none | lbfgs | 3.0x | irls | 18.4x | irls | 19.3x |
| logistic | l2 | lbfgs | 2.6x | newton | 13.0x | newton | 13.8x |
| logistic | l1 | fista | 1.0x | fista | 3.7x | fista | 3.7x |
| logistic | elasticnet | fista | 1.0x | fista | 2.9x | fista | 3.0x |
| logistic | adaptive_l1 | admm | 1.0x | fista_bb | 3.4x | fista_bb | 3.4x |
| logistic | scad | fista | 1.0x | fista | 6.1x | fista_bb | 6.0x |
| logistic | mcp | fista | 1.0x | fista_bb | 8.0x | fista_bb | 7.8x |
| logistic | group_lasso | admm | 1.0x | admm | 3.3x | fista_bb | 3.3x |
| logistic | group_scad | fista_bb | 1.0x | admm | 5.1x | fista | 5.1x |
| logistic | group_mcp | fista | 1.0x | fista_bb | 6.5x | fista_bb | 6.3x |
| poisson | none | lbfgs | 2.2x | newton | 12.6x | newton | 13.1x |
| poisson | l2 | lbfgs | 2.1x | newton | 12.7x | newton | 13.2x |
| poisson | l1 | fista | 1.0x | fista | 4.1x | fista | 4.3x |
| poisson | elasticnet | fista | 1.0x | fista | 3.7x | fista | 3.8x |
| poisson | adaptive_l1 | fista | 1.0x | admm | 4.1x | fista | 4.1x |
| poisson | scad | admm | 1.0x | admm | 8.8x | fista_bb | 8.8x |
| poisson | mcp | fista_bb | 1.0x | fista_bb | 8.9x | admm | 8.8x |
| poisson | group_lasso | fista_bb | 1.0x | fista | 3.8x | fista | 3.8x |
| poisson | group_scad | admm | 1.0x | fista_bb | 6.2x | fista | 6.2x |
| poisson | group_mcp | fista | 1.0x | fista | 6.1x | fista_bb | 6.2x |
| gamma | none | lbfgs | 9.8x | newton | 77.1x | newton | 82.8x |
| gamma | l2 | lbfgs | 10.4x | newton | 73.7x | newton | 77.1x |
| gamma | l1 | fista_bb | 1.2x | fista_bb | 7.5x | fista_bb | 7.7x |
| gamma | elasticnet | fista_bb | 1.3x | fista_bb | 7.4x | fista_bb | 7.5x |
| gamma | adaptive_l1 | fista | 1.0x | admm | 5.7x | fista | 5.6x |
| gamma | scad | admm | 1.0x | fista | 9.3x | fista | 9.1x |
| gamma | mcp | fista_bb | 1.0x | admm | 9.1x | fista | 9.2x |
| gamma | group_lasso | fista_bb | 1.0x | fista_bb | 5.2x | admm | 5.1x |
| gamma | group_scad | fista | 1.0x | fista | 6.6x | fista | 6.6x |
| gamma | group_mcp | admm | 1.0x | fista | 6.7x | fista | 6.6x |
| inverse_gaussian | none | lbfgs | 1.7x | lbfgs | 9.5x | lbfgs | 9.5x |
| inverse_gaussian | l2 | lbfgs | 1.2x | lbfgs | 6.7x | lbfgs | 6.2x |
| inverse_gaussian | l1 | fista_bb | 0.9x | fista | 4.0x | fista | 4.2x |
| inverse_gaussian | elasticnet | fista | 1.0x | fista_bb | 3.8x | fista | 4.0x |
| inverse_gaussian | adaptive_l1 | fista | 1.0x | admm | 4.8x | fista | 4.8x |
| inverse_gaussian | scad | fista | 1.0x | fista | 7.9x | fista | 8.0x |
| inverse_gaussian | mcp | fista | 1.0x | fista | 8.0x | fista | 8.0x |
| inverse_gaussian | group_lasso | fista | 1.0x | admm | 2.7x | fista | 2.7x |
| inverse_gaussian | group_scad | fista | 1.0x | fista | 5.9x | fista | 5.9x |
| inverse_gaussian | group_mcp | fista | 1.0x | fista | 6.0x | fista | 6.0x |
| negative_binomial | none | lbfgs | 8.9x | irls | 97.1x | irls | 101.8x |
| negative_binomial | l2 | lbfgs | 8.1x | irls | 93.6x | irls | 96.0x |
| negative_binomial | l1 | fista_bb | 1.3x | fista_bb | 6.7x | fista_bb | 6.6x |
| negative_binomial | elasticnet | fista_bb | 1.4x | fista_bb | 6.7x | fista_bb | 6.9x |
| negative_binomial | adaptive_l1 | fista | 1.0x | fista_bb | 4.2x | fista_bb | 4.2x |
| negative_binomial | scad | fista | 1.1x | fista | 8.9x | fista | 8.4x |
| negative_binomial | mcp | fista | 1.0x | fista | 8.1x | fista | 8.6x |
| negative_binomial | group_lasso | fista | 1.0x | fista_bb | 3.9x | fista | 3.9x |
| negative_binomial | group_scad | fista | 1.0x | fista | 6.2x | fista | 6.2x |
| negative_binomial | group_mcp | fista | 1.0x | fista | 6.6x | fista | 6.7x |
| tweedie | none | lbfgs | 8.5x | newton | 83.6x | newton | 88.7x |
| tweedie | l2 | lbfgs | 8.3x | newton | 82.9x | newton | 88.0x |
| tweedie | l1 | fista | 1.0x | fista | 4.9x | fista | 4.9x |
| tweedie | elasticnet | fista | 1.0x | fista | 4.7x | fista | 4.8x |
| tweedie | adaptive_l1 | fista | 1.0x | fista | 5.1x | fista_bb | 5.2x |
| tweedie | scad | fista | 1.0x | fista | 8.2x | fista | 8.4x |
| tweedie | mcp | fista | 1.0x | fista | 8.5x | fista | 8.6x |
| tweedie | group_lasso | admm | 1.0x | admm | 4.6x | admm | 4.5x |
| tweedie | group_scad | fista | 1.0x | fista | 6.6x | fista | 6.6x |
| tweedie | group_mcp | fista | 1.0x | fista | 6.3x | fista | 6.4x |

## Top 10 GPU Speedups (torch)

| Rank | Family | Penalty | Solver | Speedup |
|------|--------|---------|--------|---------|
| 1 | negative_binomial | none | irls | 101.8x |
| 2 | negative_binomial | l2 | irls | 96.0x |
| 3 | tweedie | none | newton | 88.7x |
| 4 | tweedie | l2 | newton | 88.0x |
| 5 | gamma | none | newton | 82.8x |
| 6 | gamma | l2 | newton | 77.1x |
| 7 | logistic | none | irls | 19.3x |
| 8 | squared_error | mcp | admm | 19.1x |
| 9 | squared_error | scad | fista | 15.6x |
| 10 | logistic | l2 | newton | 13.8x |

## Solver Selection Guide

| Penalty Type | Best CPU Solver | Best GPU Solver | Typical GPU Speedup |
|-------------|----------------|-----------------|--------------------|
| Smooth (none/l2) | lbfgs/newton | newton/irls | 10-102x |
| L1/ElasticNet | fista/fista_bb | fista/fista_bb | 3-7x |
| Adaptive L1 | fista/fista_bb | fista_bb | 3-5x |
| SCAD/MCP | fista | fista | 6-19x |
| Group Lasso | admm | admm/fista | 3-5x |
| Group SCAD/MCP | fista | fista | 5-7x |