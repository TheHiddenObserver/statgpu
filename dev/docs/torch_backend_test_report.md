# StatGPU Torch Backend - Remote Test Report

**Test Date**: 2026-04-17  
**Remote Server**: hz-4.matpool.com:27609  
**Conda Environment**: `myconda`  
**PyTorch Version**: 2.0.0+cu117  
**CuPy Version**: 13.6.0  
**GPU Device**: Tesla P100-SXM2-16GB  

---

## Test Summary

| Test | Status | Device |
|------|--------|--------|
| Distribution functions (norm) | PASS | cuda:0 |
| Distribution functions (t) | PASS | cuda:0 |
| GPU utils (compute_inference) | PASS | cuda |
| LinearRegression CPU fit | PASS | cpu |
| LinearRegression GPU fit | PASS | cuda |

**All tests passed!**

---

## Detailed Results

### [1] Distribution Functions

```
norm.two_sided_pvalue([1.96, 2.58]) = tensor([0.0500, 0.0099], device='cuda:0')
t.two_sided_pvalue(df=10, [2.0, 3.0]) = tensor([0.0734, 0.0133], device='cuda:0')
```

**Notes**: 
- Both functions work on GPU with PyTorch 2.0.0+cu117
- `torch.special.betainc` available in PyTorch 2.0+

### [2] GPU Utils (`compute_inference_torch`)

```
Device: cuda
Coefficients: [ 0.99735315  1.97967866 -0.99961518  0.47913899]
Std Errors:   [0.01076714 0.01125083 0.01203754 0.01198577]
R² = 0.9975
```

**Notes**: 
- All inference statistics computed on Torch GPU
- R² matches expected value

### [3] LinearRegression CPU Fit

```
NumPy reference: [ 0.06392205  0.97496502 -1.96644711  0.506478    0.02700706  1.52078524]
Torch CPU coef:  [ 0.97496502 -1.96644711  0.506478    0.02700706  1.52078524]
R² = 0.9659
Max coef diff vs NumPy: 0.00e+00
```

**Notes**: 
- **Perfect numerical accuracy** - coefficients match NumPy exactly
- R² matches expected value

### [4] LinearRegression GPU Fit

```
Device: cuda
GPU fit coef: [ 0.97496502 -1.96644711  0.506478    0.02700706  1.52078524]
R² = 0.9659
```

**Notes**: 
- GPU fit successful on Tesla P100
- Coefficients match CPU fit exactly

---

## Key Fixes Applied

1. **betainc fallback**: Added `scipy_beta_cdf_torch()` fallback for older PyTorch versions without `torch.special.betainc`
2. **Import path fix**: Changed import in `_gpu_utils_torch.py` to `from .inference._distributions_torch import t`

---

## Environment Notes

### Working Configuration (myconda)
```
PyTorch: 2.0.0+cu117
CuPy: 13.6.0
CUDA: 11.7
GPU: Tesla P100-SXM2-16GB
Driver: 460.32.03
```

### Non-Working Configuration (base)
```
PyTorch: 2.11.0+cu130
CUDA: 13.0 (driver too old)
GPU: Not accessible
```

**Important**: Use `myconda` environment for GPU testing. The `base` environment has PyTorch 2.11.0+cu130 which requires a newer NVIDIA driver.

---

## Conclusions

1. **Torch backend is numerically accurate** - LinearRegression coefficients match NumPy with 0.00e+00 difference
2. **GPU path works correctly** - All tests pass on Tesla P100 with PyTorch 2.0.0+cu117
3. **Distribution functions work** - Both norm and t distributions produce correct p-values on GPU
4. **Environment matters** - Must use `myconda` environment, not `base`

---

## Next Steps

### Phase 4: Performance Benchmarks (In Progress)
- Compare Torch vs CuPy performance on:
  - LinearRegression fit time
  - Inference computation time
  - Memory usage

### Phase 5: Documentation Updates (Pending)
- Update README.md with Torch installation instructions
- Update USAGE.md with Torch usage examples
- Document environment requirements (PyTorch 2.0+ recommended)

---

**Test Script**: `dev/scripts/remote_torch_test_inline.py`  
**Report Generated**: 2026-04-17
