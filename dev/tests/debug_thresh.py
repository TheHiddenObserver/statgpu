# -*- coding: utf-8 -*-
"""Debug CuPy ElementwiseKernel with zero threshold."""
import numpy as np
import cupy as cp

fused = cp.ElementwiseKernel(
    'T coef, T grad, T step, T thresh, T coef_old, T beta',
    'T coef_new, T y_k',
    '''
    T w = coef - step * grad;
    T abs_w = abs(w);
    T sign_w = (w > 0) ? 1 : ((w < 0) ? -1 : 0);
    coef_new = (abs_w > thresh) ? sign_w * (abs_w - thresh) : 0;
    y_k = coef_new + beta * (coef_new - coef_old);
    ''',
    'test_fused',
)

p = 5
coef = cp.array([1.0, -2.0, 0.5, -0.01, 3.0])
grad = cp.array([0.1, -0.2, 0.05, -0.001, 0.3])
step = 0.01
beta = 0.5
coef_old = coef.copy()

# Test with thresh=0
thresh_zero = cp.zeros(p)
coef_z, yk_z = fused(coef.copy(), grad, step, thresh_zero, coef_old, beta)

# Test with thresh=1e-15
thresh_tiny = cp.full(p, 1e-15)
coef_t, yk_t = fused(coef.copy(), grad, step, thresh_tiny, coef_old, beta)

# Test with thresh=0.01
thresh_big = cp.full(p, 0.01)
coef_b, yk_b = fused(coef.copy(), grad, step, thresh_big, coef_old, beta)

print(f"thresh=0:     coef={coef_z.get()}")
print(f"thresh=1e-15: coef={coef_t.get()}")
print(f"thresh=0.01:  coef={coef_b.get()}")
print(f"match(0 vs 1e-15): {cp.allclose(coef_z, coef_t)}")
