"""
Multi-block Efron gradient/Hessian CUDA kernel.

Launches one block per failure group (like the loglik kernel),
so all groups are processed in parallel. Each block computes
the gradient and Hessian contribution from its group using
the Efron arithmetic-series formula.
"""

import numpy as np

_EFRON_GRAD_HESS_THREADS = 128

_EFRON_GRAD_HESS_KERNEL = r"""
#define THREADS 128
extern "C" __global__
void efron_grad_hess_by_group(
    const double* __restrict__ X,           // (n, p)
    const double* __restrict__ exp_eta,     // (n,)
    const double* __restrict__ risk_sum,    // (n,)
    const double* __restrict__ risk_X_sum,  // (n, p)
    const double* __restrict__ risk_X2_sum, // (n, p*p)  prefix sum of X_exp^T @ X
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

    // Shared memory for reductions
    __shared__ double sh_exp[THREADS];
    __shared__ double sh_X[THREADS * 16]; // up to p=16 in shared

    // Step 1: Compute sum_ev_exp and sum_ev_X for tied events
    double local_exp = 0.0;
    double local_X[16] = {0}; // max p=16
    for (int i = start + tid; i < end; i += THREADS) {
        int idx = fail_ind[i];
        double ex = exp_eta[idx];
        local_exp += ex;
        for (int j = 0; j < p; j++) {
            local_X[j] += X[idx * p + j];
        }
    }

    sh_exp[tid] = local_exp;
    for (int j = 0; j < p; j++) {
        sh_X[tid * p + j] = local_X[j];
    }
    __syncthreads();

    // Reduce
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_exp[tid] += sh_exp[tid + stride];
            for (int j = 0; j < p; j++) {
                sh_X[tid * p + j] += sh_X[(tid + stride) * p + j];
            }
        }
        __syncthreads();
    }

    double sum_ev_exp = sh_exp[0];
    double sum_ev_X[16];
    for (int j = 0; j < p; j++) sum_ev_X[j] = sh_X[j];

    // Step 2: Risk set quantities
    int re = first_idx_uft[g];
    double s0 = risk_sum[re];
    double s1[16], risk_X2[256]; // p=8 max
    for (int j = 0; j < p; j++) s1[j] = risk_X_sum[re * p + j];
    for (int j = 0; j < p * p; j++) risk_X2[j] = total_X2[j] - risk_X2_sum[re * p * p + j];

    // Step 3: Efron arithmetic series
    // sum(1/denom_k) and sum(1/denom_k^2) for k=0..m-1
    // denom_k = s0 - (k/m)*sum_ev_exp
    double sum_inv = 0.0;
    double sum_inv2 = 0.0;
    for (int k = tid; k < m; k += THREADS) {
        double denom = s0 - ((double)k / (double)m) * sum_ev_exp;
        if (denom < 1e-300) denom = 1e-300;
        double inv = 1.0 / denom;
        sum_inv += inv;
        sum_inv2 += inv * inv;
    }

    sh_exp[tid] = sum_inv;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh_exp[tid] += sh_exp[tid + stride];
        __syncthreads();
    }
    sum_inv = sh_exp[0];

    sh_exp[tid] = sum_inv2;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh_exp[tid] += sh_exp[tid + stride];
        __syncthreads();
    }
    sum_inv2 = sh_exp[0];

    // Step 4: Write gradient and Hessian contributions
    if (tid == 0) {
        // grad_g = sum_ev_X - s1 * sum_inv * m
        for (int j = 0; j < p; j++) {
            out_grad[g * p + j] = sum_ev_X[j] - s1[j] * sum_inv * (double)m;
        }
        // hess_g = -(risk_X2 * sum_inv - outer(s1, s1) * sum_inv2 * m)
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < p; k++) {
                out_hess[g * p * p + j * p + k] =
                    -(risk_X2[j * p + k] * sum_inv - s1[j] * s1[k] * sum_inv2 * (double)m);
            }
        }
    }
}
"""

_kernel_cache = None
_KERNEL_VER = 1


def get_efron_grad_hess_kernel(cp):
    global _kernel_cache
    if _kernel_cache is None or not isinstance(_kernel_cache, tuple) or _kernel_cache[1] != _KERNEL_VER:
        _kernel_cache = (cp.RawKernel(_EFRON_GRAD_HESS_KERNEL, "efron_grad_hess_by_group"), _KERNEL_VER)
    return _kernel_cache[0]


def compute_efron_grad_hess_multiblock(
    X, exp_eta, risk_sum, risk_X_sum, risk_X2_sum, total_X2,
    fail_ptr, fail_ind, first_idx_uft, nuft, p,
    *,
    cupy_module,
):
    """Compute Efron gradient/Hessian using multi-block CUDA kernel.

    Launches one block per failure group. Works for any nuft (no 512 limit).

    Returns (grad, hess) as cupy arrays, or None if kernel fails.
    """
    cp = cupy_module
    if nuft == 0:
        return cp.zeros(p, dtype=cp.float64), cp.zeros((p, p), dtype=cp.float64)

    n = int(X.shape[0])
    meta = cp.array([n, p, nuft], dtype=cp.int32)

    out_grad = cp.zeros((nuft, p), dtype=cp.float64)
    out_hess = cp.zeros((nuft, p * p), dtype=cp.float64)

    # Ensure all arrays are contiguous (CUDA kernels require it)
    X = cp.ascontiguousarray(X)
    exp_eta = cp.ascontiguousarray(exp_eta)
    risk_sum = cp.ascontiguousarray(risk_sum)
    risk_X_sum = cp.ascontiguousarray(risk_X_sum)
    risk_X2_sum = cp.ascontiguousarray(risk_X2_sum)
    total_X2 = cp.ascontiguousarray(total_X2)
    fail_ptr = cp.ascontiguousarray(fail_ptr)
    fail_ind = cp.ascontiguousarray(fail_ind)
    first_idx_uft = cp.ascontiguousarray(first_idx_uft)

    kernel = get_efron_grad_hess_kernel(cp)
    try:
        kernel(
            (nuft,),                      # one block per group
            (_EFRON_GRAD_HESS_THREADS,),  # threads per block
            (
                X, exp_eta, risk_sum, risk_X_sum, risk_X2_sum, total_X2,
                meta, fail_ptr, fail_ind, first_idx_uft,
                out_grad, out_hess,
            ),
        )
        cp.cuda.Stream.null.synchronize()
    except Exception:
        return None

    # Sum across groups
    grad = cp.sum(out_grad, axis=0)
    hess = cp.sum(out_hess, axis=0).reshape(p, p)
    return grad, hess
