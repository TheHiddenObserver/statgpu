# New Modules Benchmark: Panel Data, GAM, ANOVA

Date: 2026-06-24
Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)

## Panel Data (6 estimators × 3 backends × 3 scales)

### Large Scale (2000 entities × 50 times = 100K obs, 20 vars)

| Estimator | numpy (s) | cupy (s) | torch (s) | cupy spd | torch spd |
|-----------|-----------|----------|-----------|----------|-----------|
| PooledOLS | 0.0140 | 0.0102 | 0.0098 | 1.4x | 1.4x |
| PooledOLS_hac | 0.0487 | 0.0099 | 0.0094 | 4.9x | 5.2x |
| PanelOLS_entity | 0.3291 | 0.0252 | 0.0248 | 13.1x | **13.3x** |
| PanelOLS_two_way | 1.3772 | 0.0727 | 0.0692 | 18.9x | **19.9x** |
| RandomEffects | 1.0126 | 0.1408 | 0.1415 | 7.2x | 7.2x |
| BetweenOLS | 0.6312 | 1.0297 | 1.0325 | 0.6x | 0.6x |
| FirstDifferenceOLS | 0.6037 | 0.5979 | 0.6151 | 1.0x | 1.0x |
| FamaMacBeth | 0.0475 | 0.0506 | 0.0542 | 0.9x | 0.9x |

## GAM (B-spline additive model)

| Scale | n | features | splines | numpy (s) | cupy (s) | torch (s) | cupy spd | torch spd |
|-------|---|----------|---------|-----------|----------|-----------|----------|-----------|
| small | 1K | 3 | 15 | 0.0057 | 0.0168 | 0.0164 | 0.3x | 0.3x |
| medium | 10K | 5 | 20 | 0.0684 | 0.0319 | 0.0319 | 2.1x | 2.1x |
| large | 100K | 10 | 25 | 2.5730 | 0.1194 | 0.1177 | 21.5x | **21.9x** |

## ANOVA (5 functions × 3 backends × 3 scales)

### Large Scale (100K per group, 20 groups, 2M total)

| Function | numpy (s) | cupy (s) | torch (s) | cupy spd | torch spd |
|----------|-----------|----------|-----------|----------|-----------|
| f_oneway | 0.0180 | 0.0054 | 0.0085 | **3.3x** | 2.1x |
| f_twoway | 0.0384 | 0.0098 | 0.0243 | **3.9x** | 1.6x |
| f_welch | 0.0052 | 0.0052 | 0.0103 | 1.0x | 0.5x |
| tukey_hsd | 0.7951 | 0.7962 | 0.7990 | 1.0x | 1.0x |
| bonferroni | 0.1131 | 0.7218 | 0.5142 | 0.2x | 0.2x |

**Bugs fixed & optimizations:**
- ✅ `f_oneway`: vectorized group statistics → cupy 3.3x, torch 2.1x
- ✅ `f_twoway`: vectorized SS decomposition + torch dtype fix → cupy 3.9x, torch 1.6x
- ✅ `BetweenOLS`: added time_ids parameter for API consistency

## Summary

| Module | Best GPU Speedup | Best Case |
|--------|-----------------|-----------|
| Panel Data | **19.9x** | PanelOLS_two_way (torch) |
| GAM | **21.9x** | 100K obs, 10 features (torch) |
| ANOVA | **3.9x** | f_twoway (cupy) |
