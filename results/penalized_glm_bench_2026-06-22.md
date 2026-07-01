# PenalizedGLM Benchmark Results (2026-06-22)

Environment: Tesla P100-SXM2-16GB, CuPy + PyTorch CUDA

## Precision: 7 Families x 10 Penalties x 3 Backends

**210/210 PASS** -- all combinations converge with solver=auto

## Performance (100K points, 70 combos, sorted by GPU speedup)

| # | Family | Penalty | Solver | CPU (ms) | GPU (ms) | Speedup |
|---|--------|---------|--------|----------|----------|---------|
| 1 | tweedie | l1 | auto | 4351.1 | 118.2 | 36.8x |
| 2 | tweedie | elasticnet | auto | 4331.1 | 121.1 | 35.8x |
| 3 | inverse_gaussian | none | auto | 1501.0 | 79.0 | 19.0x |
| 4 | inverse_gaussian | l2 | auto | 1048.5 | 63.5 | 16.5x |
| 5 | negative_binomial | none | auto | 375.4 | 36.6 | 10.3x |
| 6 | poisson | l2 | auto | 355.0 | 35.8 | 9.9x |
| 7 | gamma | l2 | auto | 354.5 | 36.0 | 9.9x |
| 8 | gamma | none | auto | 354.4 | 35.9 | 9.9x |
| 9 | gamma | mcp | auto | 4603.5 | 468.7 | 9.8x |
| 10 | negative_binomial | l2 | auto | 361.0 | 36.8 | 9.8x |
| 11 | poisson | mcp | auto | 6346.2 | 646.3 | 9.8x |
| 12 | poisson | none | auto | 353.3 | 36.0 | 9.8x |
| 13 | poisson | scad | auto | 6632.4 | 690.3 | 9.6x |
| 14 | gamma | scad | auto | 4263.3 | 459.1 | 9.3x |
| 15 | poisson | l1 | auto | 1278.5 | 143.5 | 8.9x |
| 16 | tweedie | mcp | auto | 6925.8 | 786.3 | 8.8x |
| 17 | inverse_gaussian | mcp | auto | 4321.5 | 493.3 | 8.8x |
| 18 | poisson | elasticnet | auto | 1269.6 | 148.3 | 8.6x |
| 19 | negative_binomial | elasticnet | auto | 1740.4 | 203.6 | 8.6x |
| 20 | inverse_gaussian | scad | auto | 4662.8 | 549.9 | 8.5x |
| 21 | negative_binomial | mcp | auto | 3169.8 | 373.7 | 8.5x |
| 22 | tweedie | scad | auto | 5880.4 | 702.5 | 8.4x |
| 23 | negative_binomial | l1 | auto | 1660.1 | 198.7 | 8.3x |
| 24 | logistic | none | auto | 271.2 | 32.7 | 8.3x |
| 25 | tweedie | none | auto | 270.7 | 32.8 | 8.2x |
| 26 | tweedie | l2 | auto | 266.8 | 32.6 | 8.2x |
| 27 | logistic | l2 | auto | 262.4 | 32.5 | 8.1x |
| 28 | gamma | group_mcp | auto | 10226.5 | 1393.3 | 7.3x |
| 29 | negative_binomial | scad | auto | 3197.2 | 437.5 | 7.3x |
| 30 | gamma | group_scad | auto | 8369.6 | 1169.9 | 7.2x |
| 31 | poisson | group_mcp | auto | 9444.9 | 1371.3 | 6.9x |
| 32 | logistic | elasticnet | auto | 686.3 | 100.2 | 6.8x |
| 33 | logistic | group_mcp | auto | 1394.2 | 204.8 | 6.8x |
| 34 | poisson | group_scad | auto | 7972.1 | 1195.1 | 6.7x |
| 35 | logistic | mcp | auto | 559.0 | 88.0 | 6.3x |
| 36 | logistic | scad | auto | 559.1 | 88.5 | 6.3x |
| 37 | poisson | adaptive_l1 | auto | 1061.6 | 167.9 | 6.3x |
| 38 | negative_binomial | group_scad | auto | 7802.4 | 1246.7 | 6.3x |
| 39 | inverse_gaussian | l1 | auto | 682.1 | 111.6 | 6.1x |
| 40 | tweedie | group_scad | auto | 9215.2 | 1511.3 | 6.1x |
| 41 | inverse_gaussian | group_mcp | auto | 5291.5 | 880.5 | 6.0x |
| 42 | poisson | group_lasso | auto | 694.7 | 117.0 | 5.9x |
| 43 | gamma | l1 | auto | 645.0 | 109.8 | 5.9x |
| 44 | inverse_gaussian | group_scad | auto | 4378.6 | 751.4 | 5.8x |
| 45 | tweedie | group_mcp | auto | 10541.3 | 1836.6 | 5.7x |
| 46 | inverse_gaussian | elasticnet | auto | 678.4 | 118.5 | 5.7x |
| 47 | gamma | elasticnet | auto | 647.9 | 115.0 | 5.6x |
| 48 | gamma | adaptive_l1 | auto | 1246.1 | 222.7 | 5.6x |
| 49 | inverse_gaussian | adaptive_l1 | auto | 1327.2 | 237.3 | 5.6x |
| 50 | tweedie | adaptive_l1 | auto | 1363.0 | 248.3 | 5.5x |
| 51 | logistic | group_scad | auto | 3566.0 | 665.4 | 5.4x |
| 52 | logistic | adaptive_l1 | auto | 1173.8 | 226.6 | 5.2x |
| 53 | negative_binomial | adaptive_l1 | auto | 1325.5 | 256.5 | 5.2x |
| 54 | negative_binomial | group_mcp | auto | 8144.8 | 1588.7 | 5.1x |
| 55 | gamma | group_lasso | auto | 674.4 | 133.1 | 5.1x |
| 56 | logistic | l1 | auto | 178.9 | 35.7 | 5.0x |
| 57 | tweedie | group_lasso | auto | 733.5 | 146.6 | 5.0x |
| 58 | inverse_gaussian | group_lasso | auto | 196.3 | 46.7 | 4.2x |
| 59 | negative_binomial | group_lasso | auto | 173.2 | 47.9 | 3.6x |
| 60 | logistic | group_lasso | auto | 186.0 | 52.1 | 3.6x |
| 61 | squared_error | none | auto | 86.9 | 28.3 | 3.1x |
| 62 | squared_error | l2 | auto | 85.4 | 28.1 | 3.0x |
| 63 | squared_error | l1 | auto | 89.9 | 30.8 | 2.9x |
| 64 | squared_error | elasticnet | auto | 85.8 | 30.4 | 2.8x |
| 65 | squared_error | adaptive_l1 | auto | 488.0 | 204.8 | 2.4x |
| 66 | squared_error | group_lasso | auto | 348.7 | 223.5 | 1.6x |
| 67 | squared_error | mcp | auto | 228.4 | 269.3 | 0.8x |
| 68 | squared_error | scad | auto | 230.3 | 283.8 | 0.8x |
| 69 | squared_error | group_mcp | auto | 90.3 | 258.1 | 0.3x |
| 70 | squared_error | group_scad | auto | 87.8 | 296.0 | 0.3x |

**Summary**: 62 fast (>=3x), 4 ok (1-3x), 4 slow (<1x)
