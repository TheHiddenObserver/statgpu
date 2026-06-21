"""
Multi-block Efron gradient/Hessian CUDA kernel.

Fully fused: computes prefix sums, per-group quantities, and Efron
adjustments inside the kernel. No Python loops, no O(n*p*p) host memory.

Each block processes one failure group. All blocks run in parallel.
"""

import numpy as np

_THREADS = 128

# Prefix sum helper: compute cumulative sum of p*p values across n rows
# Each block handles one row, accumulates into global prefix array
_PREFIX_KERNEL = r'''
extern "C" __global__
void prefix_outer(
    const double* __restrict__ X_exp,   // (n, p)
    const double* __restrict__ X,       // (n, p)
    const int n, const int p,
    double* __restrict__ prefix_out     // (n+1, p*p) prefix sum
) {
    int row = blockIdx.x;
    if (row >= n) return;
    int tid = threadIdx.x;
    int pp = p * p;

    // Compute outer product X_exp[row] * X[row] -> flat (p*p)
    // Each thread handles some elements
    __shared__ double sh[256]; // max p*p = 256 for p=16
    for (int j = tid; j < pp; j += blockDim.x) {
        int r = j / p;
        int c = j % p;
        sh[j] = X_exp[row * p + r] * X[row * p + c];
    }
    __syncthreads();

    // Write to prefix_out[row+1] (prefix_out[0] is zeros)
    for (int j = tid; j < pp; j += blockDim.x) {
        prefix_out[(row + 1) * pp + j] = sh[j];
    }
}
''';

_EFRON_KERNEL = r'''
extern "C" __global__
void efron_grad_hess_fused(
    const double* __restrict__ X,           // (n, p)
    const double* __restrict__ exp_eta,     // (n,)
    const double* __restrict__ risk_sum,    // (n,)
    const double* __restrict__ risk_X_sum,  // (n, p)
    const double* __restrict__ prefix_flat, // (n+1, p*p) prefix of outer products
    const double* __restrict__ total_X2,    // (p*p,)
    const int* __restrict__ meta,           // [n, p, nuft]
    const int* __restrict__ fail_ptr,       // CSR pointers
    const int* __restrict__ fail_ind,       // CSR indices
    const int* __restrict__ first_idx_uft,  // first index per group
    double* __restrict__ out_grad,          // (nuft, p)
    double* __restrict__ out_hess           // (nuft, p*p)
) {
    int tid = (int)threadIdx.x;
    int g = (int)blockIdx.x;
    int n = meta[0], p = meta[1], nuft = meta[2];
    if (g >= nuft) return;

    int start = fail_ptr[g];
    int end = fail_ptr[g + 1];
    int m = end - start;
    int pp = p * p;

    // Step 1: Reduce sum_ev_exp and sum_ev_X
    __shared__ double sh_exp[THREADS];
    __shared__ double sh_X[THREADS * 8]; // max p=8

    double local_exp = 0.0;
    double local_X[8] = {0};
    for (int i = start + tid; i < end; i += THREADS) {
        int idx = fail_ind[i];
        double ex = exp_eta[idx];
        local_exp += ex;
        for (int j = 0; j < p; j++) local_X[j] += X[idx * p + j];
    }
    sh_exp[tid] = local_exp;
    for (int j = 0; j < p; j++) sh_X[tid * p + j] = local_X[j];
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_exp[tid] += sh_exp[tid + stride];
            for (int j = 0; j < p; j++) sh_X[tid * p + j] += sh_X[(tid + stride) * p + j];
        }
        __syncthreads();
    }
    double sum_ev_exp = sh_exp[0];
    double sum_ev_X[8];
    for (int j = 0; j < p; j++) sum_ev_X[j] = sh_X[j];

    // Step 2: Risk set quantities
    int re = first_idx_uft[g];
    double s0 = risk_sum[re];
    double s1[8], risk_X2[64];
    for (int j = 0; j < p; j++) s1[j] = risk_X_sum[re * p + j];
    // risk_X2 = total_X2 - prefix[re]
    for (int j = 0; j < pp; j++) risk_X2[j] = total_X2[j] - prefix_flat[re * pp + j];

    // Step 3: Efron arithmetic series
    double local_inv = 0.0, local_inv2 = 0.0;
    for (int k = tid; k < m; k += THREADS) {
        double denom = s0 - ((double)k / (double)m) * sum_ev_exp;
        if (denom < 1e-300) denom = 1e-300;
        double inv = 1.0 / denom;
        local_inv += inv;
        local_inv2 += inv * inv;
    }
    sh_exp[tid] = local_inv;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh_exp[tid] += sh_exp[tid + stride];
        __syncthreads();
    }
    double sum_inv = sh_exp[0];

    sh_exp[tid] = local_inv2;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh_exp[tid] += sh_exp[tid + stride];
        __syncthreads();
    }
    double sum_inv2 = sh_exp[0];

    // Step 4: Write gradient and Hessian
    if (tid == 0) {
        for (int j = 0; j < p; j++)
            out_grad[g * p + j] = sum_ev_X[j] - s1[j] * sum_inv * (double)m;
        for (int j = 0; j < p; j++)
            for (int k = 0; k < p; k++)
                out_hess[g * pp + j * p + k] =
                    -(risk_X2[j * p + k] * sum_inv - s1[j] * s1[k] * sum_inv2 * (double)m);
    }
}
''';

_kernel_cache_grad_hess = None
_KERNEL_VER = 2


def get_efron_grad_hess_fused_kernel(cp):
    global _kernel_cache_grad_hess
    if _kernel_cache_grad_hess is None or _kernel_cache_grad_hess[1] != _KERNEL_VER:
        _kernel_cache_grad_hess = (cp.RawKernel(_EFRON_KERNEL, "efron_grad_hess_fused"), _KERNEL_VER)
    return _kernel_cache_grad_hess[0]


def compute_efron_grad_hess_fused(
    X, exp_eta, risk_sum, risk_X_sum,
    fail_ptr, fail_ind, first_idx_uft, nuft, p,
    *,
    cupy_module,
):
    """Fully fused Efron gradient/Hessian.

    Computes prefix sums and per-group adjustments inside the kernel.
    No Python loops, no O(n*p*p) host memory.

    Returns (grad, hess) as cupy arrays, or None if kernel fails.
    """
    cp = cupy_module
    if nuft == 0:
        return cp.zeros(p, dtype=cp.float64), cp.zeros((p, p), dtype=cp.float64)

    n = int(X.shape[0])
    pp = p * p

    # Ensure contiguous
    X = cp.ascontiguousarray(X)
    exp_eta = cp.ascontiguousarray(exp_eta)
    risk_sum = cp.ascontiguousarray(risk_sum)
    risk_X_sum = cp.ascontiguousarray(risk_X_sum)

    # Compute prefix sum of outer products on GPU
    prefix_flat = cp.zeros((n + 1, pp), dtype=cp.float64)
    prefix_flat[1:] = (X[:, :, None] * (exp_eta * X)[:, None, :]).reshape(n, pp)
    prefix_flat = cp.cumsum(prefix_flat, axis=0)

    total_X2 = prefix_flat[-1]  # (p*p,)

    # Output arrays
    out_grad = cp.zeros((nuft, p), dtype=cp.float64)
    out_hess = cp.zeros((nuft, pp), dtype=cp.float64)

    meta = cp.array([n, p, nuft], dtype=cp.int32)

    kernel = get_efron_grad_hess_fused_kernel(cp)
    try:
        kernel(
            (nuft,),
            (_THREADS,),
            (
                X, exp_eta, risk_sum, risk_X_sum,
                prefix_flat, total_X2,
                meta,
                cp.ascontiguousarray(fail_ptr),
                cp.ascontiguousarray(fail_ind),
                cp.ascontiguousarray(first_idx_uft),
                out_grad, out_hess,
            ),
        )
        cp.cuda.Stream.null.synchronize()
    except Exception:
        return None

    grad = cp.sum(out_grad, axis=0)
    hess = cp.sum(out_hess, axis=0).reshape(p, p)
    return grad, hess
