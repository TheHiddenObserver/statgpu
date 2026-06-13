# Knockoff Filter: Multi-Backend Comparison Report

**Date:** 2026-04-19  
**Experiments:** 50 independent runs  
**Configuration:**
- Data: n=400, p=80
- True signals: Features 0-11 (12 signals with increasing strength)
- Target FDR: q=0.10
- Signal strength: beta_scale=0.5 (weak signals 0-3 have β = 0.5, 1.0, 1.5, 2.0)
- Noise level: noise_scale=0.5

## Backends Compared

1. **numpy** - CPU-based numpy backend
2. **torch** - PyTorch backend (CUDA GPU accelerated)

## Summary Results

| Metric | NumPy | Torch | Difference |
|--------|-------|-------|------------|
| Selected Features | 13.1 ± 1.9 | 12.9 ± 1.2 | p=0.49 (ns) |
| True Positives | 12.0 ± 0.0 | 12.0 ± 0.0 | - |
| False Positives | 1.1 ± 1.9 | 0.9 ± 1.2 | p=0.49 (ns) |
| **Power** | **100.0%** | **100.0%** | - |
| FDR | 7.2% ± 9.8% | 6.4% ± 7.6% | p=0.63 (ns) |
| Separation | 79.4x ± 27.8x | 74.8x ± 17.5x | p=0.26 (ns) |
| Time (s) | 1.19 ± 0.59 | 1.19 ± 0.52 | p=0.90 (ns) |

## Key Findings

### 1. Power (True Positive Rate)

**Both backends achieve 100% power** - all 12 true signals were detected in all 50 experiments.

### 2. Weak Signal Detection (Features 0-3)

Weak signals have the smallest coefficients (β = 0.5, 1.0, 1.5, 2.0) and are the most challenging to detect.

| Feature | Signal Strength | NumPy Detection | Torch Detection |
|---------|-----------------|-----------------|-----------------|
| 0 | 0.5 | **100%** | **100%** |
| 1 | 1.0 | **100%** | **100%** |
| 2 | 1.5 | **100%** | **100%** |
| 3 | 2.0 | **100%** | **100%** |

**Both backends achieve 100% detection rate for all weak signals.**

### 3. False Positive Control

- NumPy: Average 1.1 false positives per experiment
- Torch: Average 0.9 false positives per experiment
- Both are well-controlled (FDR < 10% target)

### 4. Separation (Signal vs Noise)

Separation = Mean |W| of true signals / Max |W| of noise features

- NumPy: 79.4x ± 27.8x
- Torch: 74.8x ± 17.5x
- p=0.26 (no significant difference)

Higher separation indicates better distinguishability between signals and noise.

### 5. Computational Efficiency

- NumPy: 1.19 ± 0.59 seconds per experiment
- Torch: 1.19 ± 0.52 seconds per experiment
- No significant difference in this problem size (n=400, p=80)

## Statistical Comparison (Paired t-test)

| Metric | t-statistic | p-value | Significant? |
|--------|-------------|---------|--------------|
| Selected | 0.693 | 0.4915 | No |
| False Positives | 0.693 | 0.4915 | No |
| FDR | 0.490 | 0.6263 | No |
| Separation | 1.134 | 0.2623 | No |
| Time | 0.124 | 0.9018 | No |

**Conclusion:** No statistically significant differences between NumPy and Torch backends across any metric.

## Per-Experiment Results

### NumPy Backend

| Exp | Selected | TP | FP | Power | FDR | Weak | Time(s) |
|-----|----------|----|----|-------|-----|------|---------|
| 1 | 12 | 12 | 0 | 1.00 | 0.00 | 4 | 1.86 |
| 2 | 12 | 12 | 0 | 1.00 | 0.00 | 4 | 0.23 |
| 3 | 13 | 12 | 1 | 1.00 | 0.08 | 4 | 0.23 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 50 | 12 | 12 | 0 | 1.00 | 0.00 | 4 | 0.25 |

### Torch Backend

| Exp | Selected | TP | FP | Power | FDR | Weak | Time(s) |
|-----|----------|----|----|-------|-----|------|---------|
| 1 | 12 | 12 | 0 | 1.00 | 0.00 | 4 | 0.34 |
| 2 | 13 | 12 | 1 | 1.00 | 0.08 | 4 | 0.26 |
| 3 | 12 | 12 | 0 | 1.00 | 0.00 | 4 | 0.27 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 50 | 13 | 12 | 1 | 1.00 | 0.08 | 4 | 0.28 |

## Conclusions

1. **Both backends are statistically equivalent** - No significant differences in any performance metric.

2. **100% power achieved** - All 12 true signals detected in every experiment by both backends.

3. **Excellent weak signal detection** - Even the weakest signal (Feature 0, β=0.5) was detected 100% of the time.

4. **FDR well-controlled** - Average FDR (7.2% for NumPy, 6.4% for Torch) is below the 10% target.

5. **High separation** - Signal W statistics are ~75-80x larger than maximum noise W statistics.

6. **Similar runtime** - For this problem size, both backends have similar execution times.

## Recommendations

1. **Use NumPy backend for CPU-only systems** - No performance penalty compared to GPU for moderate problem sizes.

2. **Consider Torch backend for:**
   - Larger problems (n > 1000, p > 200) where GPU parallelism can help
   - Systems with dedicated GPU
   - Integration with PyTorch-based workflows

3. **Both backends are production-ready** - Statistically equivalent performance with 100% power and controlled FDR.

---

*Generated from 50 independent experiments with seeds 20260406-20260455.*
