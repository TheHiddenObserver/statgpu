# New Modules Benchmark (Complete)

Date: 2026-06-24
Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)

## Panel Data

### Performance (Large: 100K obs, 20 vars)

| Estimator | numpy (s) | cupy (s) | torch (s) | cupy spd | torch spd |
|-----------|-----------|----------|-----------|----------|-----------|

### External Comparison (vs linearmodels)

| Estimator | linearmodels (s) | statgpu torch (s) | Speedup | coef rel diff |
|-----------|-----------------|-------------------|---------|---------------|
| panel_medium_PanelOLS_numpy | 0.0714 | 0.0353 | 2.0x | 2.70e-16 |
| panel_medium_PanelOLS_cupy | 0.0714 | 0.8103 | 0.1x | 3.79e-16 |
| panel_medium_PanelOLS_torch | 0.0714 | 0.0086 | 8.3x | 3.70e-16 |
| panel_medium_RE_numpy | 0.0944 | 0.1059 | 0.9x | 5.30e-03 |
| panel_medium_RE_cupy | 0.0944 | 0.0532 | 1.8x | 5.30e-03 |
| panel_medium_RE_torch | 0.0944 | 0.0245 | 3.9x | 5.30e-03 |
| panel_large_PanelOLS_numpy | 0.4250 | 0.3378 | 1.3x | 1.01e-15 |
| panel_large_PanelOLS_cupy | 0.4250 | 0.0368 | 11.6x | 5.25e-16 |
| panel_large_PanelOLS_torch | 0.4250 | 0.0254 | 16.7x | 5.02e-16 |
| panel_large_RE_numpy | 0.4956 | 1.0130 | 0.5x | 1.74e-03 |
| panel_large_RE_cupy | 0.4956 | 0.1537 | 3.2x | 1.74e-03 |
| panel_large_RE_torch | 0.4956 | 0.1474 | 3.4x | 1.74e-03 |

## GAM

### Performance (Large: 100K obs, 10 features)

| Backend | statgpu (s) | pygam (s) | Speedup |
|---------|------------|----------|---------|
| numpy | 2.5730 | 5.7047 | 2.2x |
| cupy | 0.1194 | 5.7047 | 47.8x |
| torch | 0.1177 | 5.7047 | 48.5x |

### Precision (Aligned: uniform knots, gamma=1.4, fixed lam=1.0)

| Backend | pred rel diff |
|---------|--------------|
| numpy | 2.50e-03 |
| cupy | 2.50e-03 |
| torch | 2.50e-03 |

## ANOVA

### Performance (Large: 100K/group, 20 groups)

| Function | numpy (ms) | cupy (ms) | torch (ms) | cupy spd | torch spd |
|----------|-----------|----------|-----------|----------|-----------|
| f_oneway | 18.05 | 5.42 | 8.55 | 3.3x | 2.1x |
| f_twoway | 38.38 | 9.75 | 24.28 | 3.9x | 1.6x |
| f_welch | 5.24 | 5.22 | 10.26 | 1.0x | 0.5x |
| tukey_hsd | 795.07 | 796.19 | 799.00 | 1.0x | 1.0x |
| bonferroni | 113.10 | 721.76 | 514.16 | 0.2x | 0.2x |

### External Comparison (vs scipy, f_oneway)

| Backend | scipy (ms) | statgpu (ms) | Speedup | F rel diff |
|---------|-----------|-------------|---------|------------|
| numpy | 17.33 | 17.24 | 1.0x | 5.06e-15 |
| cupy | 17.33 | 7.73 | 2.2x | 5.46e-15 |
| torch | 17.33 | 18.98 | 0.9x | 5.33e-15 |