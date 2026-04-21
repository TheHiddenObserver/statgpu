"""
CUDA RawKernel for Cox PH Efron backward gradient/Hessian.

Sequential scan over unique failure times (ii). Enter/exit/failure-at-risk updates are
commutative; large index lists use ``atomicAdd`` (double, sm_60+), small lists (<= ``seq_thresh``)
use thread-0 sequential adds to avoid atomic overhead. Failure accumulation for large ``m`` is
parallel; Efron formulas remain on thread 0. Workspace ends with a scratch double for parallel
``xp0f`` sum.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np

_KERNEL_SOURCE = r"""
/* sm_60+ double atomicAdd. Small batches: thread0 sequential (no atomics). Large: parallel atomics. */
#define EFRON_MAX_P_STACK 128
// seq_thresh is passed via meta[3] for runtime tuning (see python launch code).

extern "C" __global__
void efron_backward_scan(
    const double* __restrict__ X,
    const double* __restrict__ e_eta,
    const int* __restrict__ meta,
    const int* __restrict__ enter_ptr,
    const int* __restrict__ enter_ind,
    const int* __restrict__ exit_ptr,
    const int* __restrict__ exit_ind,
    const int* __restrict__ fail_ptr,
    const int* __restrict__ fail_ind,
    double* __restrict__ grad_out,
    double* __restrict__ hess_out,
    double* __restrict__ workspace
) {
    int n = meta[0];
    int p = meta[1];
    int nuft = meta[2];
    int seq_thresh = meta[3];
    (void)n;
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;
    if (threadIdx.y != 0 || threadIdx.z != 0) return;

    double* xp0_ptr = workspace;
    double* xp1 = xp0_ptr + 1;
    double* xp2 = xp1 + p;
    double* hess_acc = xp2 + p * p;
    double* xp1f = hess_acc + p * p;
    double* xp2f = xp1f + p;
    double* scratch_xp0f = xp2f + p * p;

    int ws_doubles = 2 + 2 * p + 3 * p * p;
    for (int i = threadIdx.x; i < ws_doubles; i += blockDim.x) {
        workspace[i] = 0.0;
    }
    for (int j = threadIdx.x; j < p; j += blockDim.x) {
        grad_out[j] = 0.0;
    }
    __syncthreads();

    for (int ii = nuft - 1; ii >= 0; ii--) {
        int e0 = enter_ptr[ii];
        int e1 = enter_ptr[ii + 1];
        int nt = e1 - e0;
        if (nt <= seq_thresh) {
            if (threadIdx.x == 0) {
                for (int t = e0; t < e1; t++) {
                    int idx = enter_ind[t];
                    const double* Xrow = X + (size_t)idx * (size_t)p;
                    double elx = e_eta[idx];
                    *xp0_ptr += elx;
                    if (p <= EFRON_MAX_P_STACK) {
                        double row[EFRON_MAX_P_STACK];
                        for (int j = 0; j < p; j++) row[j] = Xrow[j];
                        for (int j = 0; j < p; j++) xp1[j] += elx * row[j];
                        for (int j = 0; j < p; j++)
                            for (int k = 0; k < p; k++)
                                xp2[j * p + k] += elx * row[j] * row[k];
                    } else {
                        for (int j = 0; j < p; j++) xp1[j] += elx * Xrow[j];
                        for (int j = 0; j < p; j++) {
                            double vj = Xrow[j];
                            for (int k = 0; k < p; k++)
                                xp2[j * p + k] += elx * vj * Xrow[k];
                        }
                    }
                }
            }
        } else {
            // Stage A: block-local accumulation for small p to reduce global atomics.
            // Stage B: one global writeback per aggregated entry.
            if (p <= 64) {
                __shared__ double sh_xp0;
                __shared__ double sh_xp1[64];
                __shared__ double sh_xp2[4096];
                if (threadIdx.x == 0) sh_xp0 = 0.0;
                for (int j = threadIdx.x; j < p; j += blockDim.x) {
                    sh_xp1[j] = 0.0;
                }
                for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
                    sh_xp2[j] = 0.0;
                }
                __syncthreads();
                for (int tt = threadIdx.x; tt < nt; tt += blockDim.x) {
                    int idx = enter_ind[e0 + tt];
                    const double* Xrow = X + (size_t)idx * (size_t)p;
                    double elx = e_eta[idx];
                    atomicAdd(&sh_xp0, elx);
                    for (int j = 0; j < p; j++) {
                        double vj = Xrow[j];
                        atomicAdd(sh_xp1 + j, elx * vj);
                    }
                    for (int j = 0; j < p; j++) {
                        double vj = Xrow[j];
                        for (int k = 0; k < p; k++) {
                            atomicAdd(sh_xp2 + j * p + k, elx * vj * Xrow[k]);
                        }
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0) atomicAdd(xp0_ptr, sh_xp0);
                for (int j = threadIdx.x; j < p; j += blockDim.x) {
                    atomicAdd(xp1 + j, sh_xp1[j]);
                }
                for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
                    atomicAdd(xp2 + j, sh_xp2[j]);
                }
            } else {
                for (int tt = threadIdx.x; tt < nt; tt += blockDim.x) {
                    int idx = enter_ind[e0 + tt];
                    const double* Xrow = X + (size_t)idx * (size_t)p;
                    double elx = e_eta[idx];
                    atomicAdd(xp0_ptr, elx);
                    if (p <= EFRON_MAX_P_STACK) {
                        double row[EFRON_MAX_P_STACK];
                        for (int j = 0; j < p; j++) row[j] = Xrow[j];
                        for (int j = 0; j < p; j++) atomicAdd(xp1 + j, elx * row[j]);
                        for (int j = 0; j < p; j++)
                            for (int k = 0; k < p; k++)
                                atomicAdd(xp2 + j * p + k, elx * row[j] * row[k]);
                    } else {
                        for (int j = 0; j < p; j++) atomicAdd(xp1 + j, elx * Xrow[j]);
                        for (int j = 0; j < p; j++) {
                            double vj = Xrow[j];
                            for (int k = 0; k < p; k++)
                                atomicAdd(xp2 + j * p + k, elx * vj * Xrow[k]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        int f0 = fail_ptr[ii];
        int f1 = fail_ptr[ii + 1];
        int m = f1 - f0;
        if (m > 0) {
            for (int j = threadIdx.x; j < p; j += blockDim.x) {
                xp1f[j] = 0.0;
            }
            for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
                xp2f[j] = 0.0;
            }
            __syncthreads();

            if (m <= seq_thresh) {
                if (threadIdx.x == 0) {
                    double xp0v = *xp0_ptr;
                    double xp0f = 0.0;
                    for (int t = f0; t < f1; t++) {
                        int idx = fail_ind[t];
                        const double* Xrow = X + (size_t)idx * (size_t)p;
                        double elx = e_eta[idx];
                        xp0f += elx;
                        if (p <= EFRON_MAX_P_STACK) {
                            double row[EFRON_MAX_P_STACK];
                            for (int j = 0; j < p; j++) row[j] = Xrow[j];
                            for (int j = 0; j < p; j++) {
                                xp1f[j] += elx * row[j];
                                grad_out[j] += row[j];
                            }
                            for (int j = 0; j < p; j++)
                                for (int k = 0; k < p; k++)
                                    xp2f[j * p + k] += elx * row[j] * row[k];
                        } else {
                            for (int j = 0; j < p; j++) {
                                double vj = Xrow[j];
                                xp1f[j] += elx * vj;
                                grad_out[j] += vj;
                            }
                            for (int j = 0; j < p; j++) {
                                double vj = Xrow[j];
                                for (int k = 0; k < p; k++)
                                    xp2f[j * p + k] += elx * vj * Xrow[k];
                            }
                        }
                    }
                    double sum_inv_c0 = 0.0;
                    double sum_J_c0 = 0.0;
                    double sum_aa = 0.0;
                    double sum_bb = 0.0;
                    double sum_ab = 0.0;
                    for (int kk = 0; kk < m; kk++) {
                        double Jk = (double)kk / (double)m;
                        double c0 = xp0v - Jk * xp0f;
                        if (c0 < 1e-300) c0 = 1e-300;
                        double ak = 1.0 / c0;
                        double bk = Jk * ak;
                        sum_inv_c0 += ak;
                        sum_J_c0 += Jk / c0;
                        sum_aa += ak * ak;
                        sum_bb += bk * bk;
                        sum_ab += ak * bk;
                    }
                    for (int j = 0; j < p; j++) {
                        grad_out[j] -= (xp1[j] * sum_inv_c0 - xp1f[j] * sum_J_c0);
                    }
                    for (int j = 0; j < p * p; j++) {
                        hess_acc[j] += xp2[j] * sum_inv_c0;
                        hess_acc[j] -= xp2f[j] * sum_J_c0;
                    }
                    for (int j1 = 0; j1 < p; j1++) {
                        for (int j2 = j1; j2 < p; j2++) {
                            double o11 = xp1[j1] * xp1[j2];
                            double off = xp1f[j1] * xp1f[j2];
                            double cross = xp1[j1] * xp1f[j2] + xp1f[j1] * xp1[j2];
                            double hsub = sum_aa * o11 + sum_bb * off - sum_ab * cross;
                            hess_acc[j1 * p + j2] -= hsub;
                            if (j2 != j1) hess_acc[j2 * p + j1] -= hsub;
                        }
                    }
                }
            } else {
                if (p <= 64) {
                    __shared__ double sh_xp0f;
                    __shared__ double sh_xp1f[64];
                    __shared__ double sh_grad[64];
                    __shared__ double sh_xp2f[4096];
                    if (threadIdx.x == 0) sh_xp0f = 0.0;
                    for (int j = threadIdx.x; j < p; j += blockDim.x) {
                        sh_xp1f[j] = 0.0;
                        sh_grad[j] = 0.0;
                    }
                    for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
                        sh_xp2f[j] = 0.0;
                    }
                    __syncthreads();
                    for (int tt = threadIdx.x; tt < m; tt += blockDim.x) {
                        int idx = fail_ind[f0 + tt];
                        const double* Xrow = X + (size_t)idx * (size_t)p;
                        double elx = e_eta[idx];
                        atomicAdd(&sh_xp0f, elx);
                        for (int j = 0; j < p; j++) {
                            double vj = Xrow[j];
                            atomicAdd(sh_xp1f + j, elx * vj);
                            atomicAdd(sh_grad + j, vj);
                        }
                        for (int j = 0; j < p; j++) {
                            double vj = Xrow[j];
                            for (int k = 0; k < p; k++) {
                                atomicAdd(sh_xp2f + j * p + k, elx * vj * Xrow[k]);
                            }
                        }
                    }
                    __syncthreads();
                    for (int j = threadIdx.x; j < p; j += blockDim.x) {
                        xp1f[j] = sh_xp1f[j];
                        atomicAdd(grad_out + j, sh_grad[j]);
                    }
                    for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
                        xp2f[j] = sh_xp2f[j];
                    }
                    if (threadIdx.x == 0) {
                        *scratch_xp0f = sh_xp0f;
                    }
                } else {
                    if (threadIdx.x == 0) {
                        *scratch_xp0f = 0.0;
                    }
                    __syncthreads();
                    for (int tt = threadIdx.x; tt < m; tt += blockDim.x) {
                        int idx = fail_ind[f0 + tt];
                        const double* Xrow = X + (size_t)idx * (size_t)p;
                        double elx = e_eta[idx];
                        atomicAdd(scratch_xp0f, elx);
                        if (p <= EFRON_MAX_P_STACK) {
                            double row[EFRON_MAX_P_STACK];
                            for (int j = 0; j < p; j++) row[j] = Xrow[j];
                            for (int j = 0; j < p; j++) {
                                atomicAdd(xp1f + j, elx * row[j]);
                                atomicAdd(grad_out + j, row[j]);
                            }
                            for (int j = 0; j < p; j++)
                                for (int k = 0; k < p; k++)
                                    atomicAdd(xp2f + j * p + k, elx * row[j] * row[k]);
                        } else {
                            for (int j = 0; j < p; j++) {
                                double vj = Xrow[j];
                                atomicAdd(xp1f + j, elx * vj);
                                atomicAdd(grad_out + j, vj);
                            }
                            for (int j = 0; j < p; j++) {
                                double vj = Xrow[j];
                                for (int k = 0; k < p; k++)
                                    atomicAdd(xp2f + j * p + k, elx * vj * Xrow[k]);
                            }
                        }
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    double xp0v = *xp0_ptr;
                    double xp0f = *scratch_xp0f;
                    double sum_inv_c0 = 0.0;
                    double sum_J_c0 = 0.0;
                    double sum_aa = 0.0;
                    double sum_bb = 0.0;
                    double sum_ab = 0.0;
                    for (int kk = 0; kk < m; kk++) {
                        double Jk = (double)kk / (double)m;
                        double c0 = xp0v - Jk * xp0f;
                        if (c0 < 1e-300) c0 = 1e-300;
                        double ak = 1.0 / c0;
                        double bk = Jk * ak;
                        sum_inv_c0 += ak;
                        sum_J_c0 += Jk / c0;
                        sum_aa += ak * ak;
                        sum_bb += bk * bk;
                        sum_ab += ak * bk;
                    }
                    for (int j = 0; j < p; j++) {
                        grad_out[j] -= (xp1[j] * sum_inv_c0 - xp1f[j] * sum_J_c0);
                    }
                    for (int j = 0; j < p * p; j++) {
                        hess_acc[j] += xp2[j] * sum_inv_c0;
                        hess_acc[j] -= xp2f[j] * sum_J_c0;
                    }
                    for (int j1 = 0; j1 < p; j1++) {
                        for (int j2 = j1; j2 < p; j2++) {
                            double o11 = xp1[j1] * xp1[j2];
                            double off = xp1f[j1] * xp1f[j2];
                            double cross = xp1[j1] * xp1f[j2] + xp1f[j1] * xp1[j2];
                            double hsub = sum_aa * o11 + sum_bb * off - sum_ab * cross;
                            hess_acc[j1 * p + j2] -= hsub;
                            if (j2 != j1) hess_acc[j2 * p + j1] -= hsub;
                        }
                    }
                }
            }
        }
        __syncthreads();

        int x0 = exit_ptr[ii];
        int x1 = exit_ptr[ii + 1];
        int nx = x1 - x0;
        if (nx <= seq_thresh) {
            if (threadIdx.x == 0) {
                for (int t = x0; t < x1; t++) {
                    int idx = exit_ind[t];
                    const double* Xrow = X + (size_t)idx * (size_t)p;
                    double elx = e_eta[idx];
                    *xp0_ptr -= elx;
                    if (p <= EFRON_MAX_P_STACK) {
                        double row[EFRON_MAX_P_STACK];
                        for (int j = 0; j < p; j++) row[j] = Xrow[j];
                        for (int j = 0; j < p; j++) xp1[j] -= elx * row[j];
                        for (int j = 0; j < p; j++)
                            for (int k = 0; k < p; k++)
                                xp2[j * p + k] -= elx * row[j] * row[k];
                    } else {
                        for (int j = 0; j < p; j++) xp1[j] -= elx * Xrow[j];
                        for (int j = 0; j < p; j++) {
                            double vj = Xrow[j];
                            for (int k = 0; k < p; k++)
                                xp2[j * p + k] -= elx * vj * Xrow[k];
                        }
                    }
                }
            }
        } else {
            // Stage A/B block accumulation for exit updates, mirroring enter side.
            if (p <= 64) {
                __shared__ double sh_xp0;
                __shared__ double sh_xp1[64];
                __shared__ double sh_xp2[4096];
                if (threadIdx.x == 0) sh_xp0 = 0.0;
                for (int j = threadIdx.x; j < p; j += blockDim.x) {
                    sh_xp1[j] = 0.0;
                }
                for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
                    sh_xp2[j] = 0.0;
                }
                __syncthreads();
                for (int tt = threadIdx.x; tt < nx; tt += blockDim.x) {
                    int idx = exit_ind[x0 + tt];
                    const double* Xrow = X + (size_t)idx * (size_t)p;
                    double elx = e_eta[idx];
                    atomicAdd(&sh_xp0, -elx);
                    for (int j = 0; j < p; j++) {
                        double vj = Xrow[j];
                        atomicAdd(sh_xp1 + j, -elx * vj);
                    }
                    for (int j = 0; j < p; j++) {
                        double vj = Xrow[j];
                        for (int k = 0; k < p; k++) {
                            atomicAdd(sh_xp2 + j * p + k, -elx * vj * Xrow[k]);
                        }
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0) atomicAdd(xp0_ptr, sh_xp0);
                for (int j = threadIdx.x; j < p; j += blockDim.x) {
                    atomicAdd(xp1 + j, sh_xp1[j]);
                }
                for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
                    atomicAdd(xp2 + j, sh_xp2[j]);
                }
            } else {
                for (int tt = threadIdx.x; tt < nx; tt += blockDim.x) {
                    int idx = exit_ind[x0 + tt];
                    const double* Xrow = X + (size_t)idx * (size_t)p;
                    double elx = e_eta[idx];
                    atomicAdd(xp0_ptr, -elx);
                    if (p <= EFRON_MAX_P_STACK) {
                        double row[EFRON_MAX_P_STACK];
                        for (int j = 0; j < p; j++) row[j] = Xrow[j];
                        for (int j = 0; j < p; j++) atomicAdd(xp1 + j, -elx * row[j]);
                        for (int j = 0; j < p; j++)
                            for (int k = 0; k < p; k++)
                                atomicAdd(xp2 + j * p + k, -elx * row[j] * row[k]);
                    } else {
                        for (int j = 0; j < p; j++) atomicAdd(xp1 + j, -elx * Xrow[j]);
                        for (int j = 0; j < p; j++) {
                            double vj = Xrow[j];
                            for (int k = 0; k < p; k++)
                                atomicAdd(xp2 + j * p + k, -elx * vj * Xrow[k]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int j = 0; j < p * p; j++) {
            hess_out[j] = -hess_acc[j];
        }
    }
}
"""

_KERNEL_SOURCE_SERIAL = r"""
#define EFRON_MAX_P_STACK 128
extern "C" __global__
void efron_backward_scan_serial(
    const double* __restrict__ X,
    const double* __restrict__ e_eta,
    const int* __restrict__ meta,
    const int* __restrict__ enter_ptr,
    const int* __restrict__ enter_ind,
    const int* __restrict__ exit_ptr,
    const int* __restrict__ exit_ind,
    const int* __restrict__ fail_ptr,
    const int* __restrict__ fail_ind,
    double* __restrict__ grad_out,
    double* __restrict__ hess_out,
    double* __restrict__ workspace
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int p = meta[1];
    int nuft = meta[2];

    double* xp0_ptr = workspace;
    double* xp1 = xp0_ptr + 1;
    double* xp2 = xp1 + p;
    double* hess_acc = xp2 + p * p;
    double* xp1f = hess_acc + p * p;
    double* xp2f = xp1f + p;

    *xp0_ptr = 0.0;
    for (int j = 0; j < p; j++) {
        grad_out[j] = 0.0;
        xp1[j] = 0.0;
        xp1f[j] = 0.0;
    }
    for (int j = 0; j < p * p; j++) {
        xp2[j] = 0.0;
        xp2f[j] = 0.0;
        hess_acc[j] = 0.0;
    }

    for (int ii = nuft - 1; ii >= 0; ii--) {
        int e0 = enter_ptr[ii];
        int e1 = enter_ptr[ii + 1];
        for (int t = e0; t < e1; t++) {
            int idx = enter_ind[t];
            const double* Xrow = X + (size_t)idx * (size_t)p;
            double elx = e_eta[idx];
            *xp0_ptr += elx;
            for (int j = 0; j < p; j++) xp1[j] += elx * Xrow[j];
            for (int j = 0; j < p; j++) {
                double vj = Xrow[j];
                for (int k = 0; k < p; k++) xp2[j * p + k] += elx * vj * Xrow[k];
            }
        }

        int f0 = fail_ptr[ii];
        int f1 = fail_ptr[ii + 1];
        int m = f1 - f0;
        if (m > 0) {
            for (int j = 0; j < p; j++) xp1f[j] = 0.0;
            for (int j = 0; j < p * p; j++) xp2f[j] = 0.0;
            double xp0v = *xp0_ptr;
            double xp0f = 0.0;

            for (int t = f0; t < f1; t++) {
                int idx = fail_ind[t];
                const double* Xrow = X + (size_t)idx * (size_t)p;
                double elx = e_eta[idx];
                xp0f += elx;
                for (int j = 0; j < p; j++) {
                    xp1f[j] += elx * Xrow[j];
                    grad_out[j] += Xrow[j];
                }
                for (int j = 0; j < p; j++) {
                    double vj = Xrow[j];
                    for (int k = 0; k < p; k++) xp2f[j * p + k] += elx * vj * Xrow[k];
                }
            }

            double sum_inv_c0 = 0.0;
            double sum_J_c0 = 0.0;
            double sum_aa = 0.0;
            double sum_bb = 0.0;
            double sum_ab = 0.0;
            for (int kk = 0; kk < m; kk++) {
                double Jk = (double)kk / (double)m;
                double c0 = xp0v - Jk * xp0f;
                if (c0 < 1e-300) c0 = 1e-300;
                double ak = 1.0 / c0;
                double bk = Jk * ak;
                sum_inv_c0 += ak;
                sum_J_c0 += Jk / c0;
                sum_aa += ak * ak;
                sum_bb += bk * bk;
                sum_ab += ak * bk;
            }

            for (int j = 0; j < p; j++) grad_out[j] -= (xp1[j] * sum_inv_c0 - xp1f[j] * sum_J_c0);
            for (int j = 0; j < p * p; j++) {
                hess_acc[j] += xp2[j] * sum_inv_c0;
                hess_acc[j] -= xp2f[j] * sum_J_c0;
            }
            for (int j1 = 0; j1 < p; j1++) {
                for (int j2 = j1; j2 < p; j2++) {
                    double o11 = xp1[j1] * xp1[j2];
                    double off = xp1f[j1] * xp1f[j2];
                    double cross = xp1[j1] * xp1f[j2] + xp1f[j1] * xp1[j2];
                    double hsub = sum_aa * o11 + sum_bb * off - sum_ab * cross;
                    hess_acc[j1 * p + j2] -= hsub;
                    if (j2 != j1) hess_acc[j2 * p + j1] -= hsub;
                }
            }
        }

        int x0 = exit_ptr[ii];
        int x1 = exit_ptr[ii + 1];
        for (int t = x0; t < x1; t++) {
            int idx = exit_ind[t];
            const double* Xrow = X + (size_t)idx * (size_t)p;
            double elx = e_eta[idx];
            *xp0_ptr -= elx;
            for (int j = 0; j < p; j++) xp1[j] -= elx * Xrow[j];
            for (int j = 0; j < p; j++) {
                double vj = Xrow[j];
                for (int k = 0; k < p; k++) xp2[j * p + k] -= elx * vj * Xrow[k];
            }
        }
    }

    for (int j = 0; j < p * p; j++) hess_out[j] = -hess_acc[j];
}
"""

# Workspace: xp0(1) + xp1(p) + xp2(p*p) + hess_acc(p*p) + xp1f(p) + xp2f(p*p) + scratch(1)
EFRON_BACKWARD_THREADS: int = 128

_kernel_cache: Any = None
_KERNEL_VER = 9
_kernel_cache_serial: Any = None
_KERNEL_VER_SERIAL = 1


def _env_int(name: str, default: int) -> int:
    """Parse env int with a safe fallback."""
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _pick_backward_launch_params(p: int, nuft: int, n: int) -> Tuple[int, int]:
    """Choose (seq_thresh, threads) with env override + sane defaults.

    - `STATGPU_EFRON_SEQ_THRESH` >= 0 overrides seq threshold.
    - `STATGPU_EFRON_BACKWARD_THREADS` > 0 overrides threads.
    - Otherwise choose a heavier sequential threshold for small/medium `p`,
      which often reduces atomic overhead in heavy-ties cases.
    """
    seq_env = _env_int("STATGPU_EFRON_SEQ_THRESH", -1)
    th_env = _env_int("STATGPU_EFRON_BACKWARD_THREADS", -1)

    if th_env > 0:
        threads = int(th_env)
    else:
        # Small/medium shapes are usually latency-sensitive; fewer threads can help.
        if p <= 24 and nuft <= 512:
            threads = 64
        elif p <= 64 and nuft <= 512:
            threads = 64
        else:
            threads = EFRON_BACKWARD_THREADS

    if seq_env >= 0:
        seq_thresh = int(seq_env)
    else:
        # Favor single-thread local accumulation for moderate group sizes.
        avg_group = float(n) / max(1.0, float(nuft))
        if p <= 24:
            seq_thresh = 64
        elif p <= 64:
            # Heavy ties (large n/nuft) are often dominated by atomic contention.
            # Favor larger sequential groups to reduce p^2 atomic pressure.
            if avg_group >= 64.0:
                seq_thresh = 256
            elif avg_group >= 32.0:
                seq_thresh = 128
            else:
                seq_thresh = 64
        else:
            seq_thresh = 16

    return seq_thresh, threads


def _pack_csr(groups: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    ptr = [0]
    ind: List[int] = []
    for g in groups:
        ind.extend(int(x) for x in g)
        ptr.append(len(ind))
    return np.asarray(ptr, dtype=np.int32), np.asarray(ind, dtype=np.int32)


def efron_indices_to_csr(
    uft_ix: List[List[int]], risk_enter: List[List[int]], risk_exit: List[List[int]], nuft: int
) -> Tuple[np.ndarray, ...]:
    enter_ptr, enter_ind = _pack_csr(risk_enter)
    exit_ptr, exit_ind = _pack_csr(risk_exit)
    fail_ptr, fail_ind = _pack_csr(uft_ix)
    assert enter_ptr.size == nuft + 1 and exit_ptr.size == nuft + 1 and fail_ptr.size == nuft + 1
    return enter_ptr, enter_ind, exit_ptr, exit_ind, fail_ptr, fail_ind


def get_efron_backward_kernel(cp):
    global _kernel_cache
    if (
        _kernel_cache is None
        or not isinstance(_kernel_cache, tuple)
        or _kernel_cache[1] != _KERNEL_VER
    ):
        _kernel_cache = (cp.RawKernel(_KERNEL_SOURCE, "efron_backward_scan"), _KERNEL_VER)
    return _kernel_cache[0]


def get_efron_backward_kernel_serial(cp):
    global _kernel_cache_serial
    if (
        _kernel_cache_serial is None
        or not isinstance(_kernel_cache_serial, tuple)
        or _kernel_cache_serial[1] != _KERNEL_VER_SERIAL
    ):
        _kernel_cache_serial = (
            cp.RawKernel(_KERNEL_SOURCE_SERIAL, "efron_backward_scan_serial"),
            _KERNEL_VER_SERIAL,
        )
    return _kernel_cache_serial[0]


_LOGLIK_THREADS: int = 128

_LOGLIK_KERNEL_SOURCE = r"""
#define EFRON_LOGLIK_THREADS 128
extern "C" __global__
void efron_loglik_by_group(
    const double* __restrict__ eta,
    const double* __restrict__ exp_eta,
    const double* __restrict__ risk_sum,
    const int* __restrict__ meta,      // meta[0] = nuft
    const int* __restrict__ fail_ptr,
    const int* __restrict__ fail_ind,
    const int* __restrict__ first_idx_uft,
    double* __restrict__ out_ll
) {
    int tid = (int)threadIdx.x;
    int g = (int)blockIdx.x;
    int nuft = meta[0];
    if (g >= nuft) return;

    int start = fail_ptr[g];
    int end = fail_ptr[g + 1];
    int m = end - start;

    __shared__ double sh_events[EFRON_LOGLIK_THREADS];
    __shared__ double sh_eta[EFRON_LOGLIK_THREADS];
    __shared__ double sh_logs[EFRON_LOGLIK_THREADS];

    double local_events = 0.0;
    double local_eta = 0.0;
    for (int i = start + tid; i < end; i += (int)blockDim.x) {
        int idx = fail_ind[i];
        double ex = exp_eta[idx];
        local_events += ex;
        local_eta += eta[idx];
    }

    sh_events[tid] = local_events;
    sh_eta[tid] = local_eta;
    __syncthreads();

    // Reduce sum_events and sum_eta.
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_events[tid] += sh_events[tid + stride];
            sh_eta[tid] += sh_eta[tid + stride];
        }
        __syncthreads();
    }

    double sum_events = sh_events[0];
    double sum_eta = sh_eta[0];
    double risk_at_t = risk_sum[first_idx_uft[g]];

    double local_logs = 0.0;
    for (int k = tid; k < m; k += (int)blockDim.x) {
        double Jk = (double)k / (double)m;
        double denom = risk_at_t - Jk * sum_events;
        if (denom < 1e-300) denom = 1e-300;
        local_logs += log(denom);
    }

    sh_logs[tid] = local_logs;
    __syncthreads();

    // Reduce sum_logs.
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_logs[tid] += sh_logs[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // ll = sum(eta[idx]) - sum_{k=0..m-1} log(risk_at_t - k/m * sum_events)
        out_ll[g] = sum_eta - sh_logs[0];
    }
}
"""

_kernel_cache_loglik: Any = None
_KERNEL_VER_LOGLIK = 1


def get_efron_loglik_kernel(cp):
    global _kernel_cache_loglik
    if (
        _kernel_cache_loglik is None
        or not isinstance(_kernel_cache_loglik, tuple)
        or _kernel_cache_loglik[1] != _KERNEL_VER_LOGLIK
    ):
        _kernel_cache_loglik = (
            cp.RawKernel(_LOGLIK_KERNEL_SOURCE, "efron_loglik_by_group"),
            _KERNEL_VER_LOGLIK,
        )
    return _kernel_cache_loglik[0]


def compute_efron_loglik_raw_csr(
    eta,
    exp_eta,
    risk_sum,
    fail_ptr,
    fail_ind,
    first_idx_uft,
    nuft: int,
    *,
    cupy_module,
) -> Any:
    """
    Compute scalar Efron log partial likelihood on GPU using a single kernel.
    `fail_ptr/fail_ind` are CSR arrays for uft_ix; `first_idx_uft` is int32.
    """
    cp = cupy_module
    if nuft == 0:
        return cp.array(0.0, dtype=cp.float64)

    # RawKernel assumes contiguous storage; avoid redundant copies.
    if not getattr(eta, "flags", None) or not eta.flags.c_contiguous:
        eta = cp.ascontiguousarray(eta)
    if not getattr(exp_eta, "flags", None) or not exp_eta.flags.c_contiguous:
        exp_eta = cp.ascontiguousarray(exp_eta)
    if not getattr(risk_sum, "flags", None) or not risk_sum.flags.c_contiguous:
        risk_sum = cp.ascontiguousarray(risk_sum)

    fail_ptr_g = cp.asarray(fail_ptr, dtype=cp.int32)
    fail_ind_g = cp.asarray(fail_ind, dtype=cp.int32)
    first_idx_uft_g = cp.asarray(first_idx_uft, dtype=cp.int32)

    out_ll = cp.zeros(int(nuft), dtype=cp.float64)
    meta = cp.array([int(nuft)], dtype=cp.int32)
    kernel = get_efron_loglik_kernel(cp)
    try:
        kernel(
            (int(nuft),),
            (_LOGLIK_THREADS,),
            (
                eta,
                exp_eta,
                risk_sum,
                meta,
                fail_ptr_g,
                fail_ind_g,
                first_idx_uft_g,
                out_ll,
            ),
        )
        return cp.sum(out_ll)
    except Exception:
        # If kernel launch fails, let caller fallback to Python loop.
        raise


def compute_efron_loglik_raw(eta, exp_eta, risk_sum, time, efron_pre, *, cupy_module):
    """
    Scalar partial log-likelihood (Efron) on GPU.

    Uses a CuPy loop over cached failure groups; inner Efron sum over k is vectorized.
    When ``efron_pre`` includes ``first_idx_uft`` (from ``_efron_unique_failure_indices``), avoids host ``searchsorted`` per group.
    
    When cupy_module=None, uses NumPy for CPU computation (used by Torch backend fallback).
    """
    cp = cupy_module
    
    # NumPy fallback for Torch backend
    if cp is None:
        return _compute_efron_loglik_raw_numpy(eta, exp_eta, risk_sum, time, efron_pre)
    
    if len(efron_pre) == 6:
        uft_arr, uft_ix, _, _, nuft, first_idx_uft = efron_pre
    else:
        uft_arr, uft_ix, _, _, nuft = efron_pre
        first_idx_uft = None
    if nuft == 0:
        return cp.array(0.0, dtype=cp.float64)

    ll = cp.zeros((), dtype=cp.float64)
    fi_gpu = cp.asarray(first_idx_uft, dtype=cp.int32) if first_idx_uft is not None else None
    time_np = None
    if fi_gpu is None:
        time_np = cp.asnumpy(time).astype(np.float64, copy=False)

    for i in range(nuft):
        ix_ev = uft_ix[i]
        d = len(ix_ev)
        if d == 0:
            continue
        if fi_gpu is not None:
            first_idx = fi_gpu[i]
        else:
            first_idx = int(np.searchsorted(time_np, float(uft_arr[i]), side="left"))
        risk_at_t = risk_sum[first_idx]
        idx = cp.asarray(ix_ev, dtype=cp.int32)
        sum_events = cp.sum(exp_eta[idx])
        kd = float(d)
        k = cp.arange(d, dtype=cp.float64)
        denom = risk_at_t - (k / kd) * sum_events
        ll -= cp.sum(cp.log(cp.maximum(denom, 1e-300)))
        ll += cp.sum(eta[idx])
    return ll


def _compute_efron_loglik_raw_numpy(eta, exp_eta, risk_sum, time, efron_pre):
    """
    NumPy implementation of Efron log-likelihood for Torch backend fallback.
    
    Parameters
    ----------
    eta : ndarray
        Linear predictor values (n_samples,)
    exp_eta : ndarray
        exp(eta) values (n_samples,)
    risk_sum : ndarray
        Cumulative risk sums (n_samples,)
    time : ndarray
        Event times (n_samples,)
    efron_pre : tuple
        Precomputed failure time indices from _efron_unique_failure_indices
        
    Returns
    -------
    float
        Log-likelihood value
    """
    if len(efron_pre) == 6:
        uft_arr, uft_ix, _, _, nuft, first_idx_uft = efron_pre
    else:
        uft_arr, uft_ix, _, _, nuft = efron_pre
        first_idx_uft = None
    
    if nuft == 0:
        return 0.0
    
    ll = 0.0
    time_np = None
    if first_idx_uft is None:
        time_np = time.astype(np.float64, copy=False)
    
    for i in range(nuft):
        ix_ev = uft_ix[i]
        d = len(ix_ev)
        if d == 0:
            continue
        
        if first_idx_uft is not None:
            first_idx = first_idx_uft[i]
        else:
            first_idx = int(np.searchsorted(time_np, float(uft_arr[i]), side="left"))
        
        risk_at_t = risk_sum[first_idx]
        sum_events = np.sum(exp_eta[ix_ev])
        kd = float(d)
        
        # Efron correction: loop over k=0..d-1
        k = np.arange(d, dtype=np.float64)
        denom = risk_at_t - (k / kd) * sum_events
        ll -= np.sum(np.log(np.maximum(denom, 1e-300)))
        ll += np.sum(eta[ix_ev])
    
    return ll


def compute_efron_grad_hess_raw(
    X,
    beta,
    efron_pre,
    *,
    cupy_module,
    efron_csr=None,
) -> Tuple[Any, Any]:
    """
    Returns (grad, hess) as cupy arrays. Falls back to None if launch fails (caller uses Python path).
    """
    cp = cupy_module
    if efron_csr is not None:
        # (enter_ptr, enter_ind, exit_ptr, exit_ind, fail_ptr, fail_ind, first_idx_uft, nuft)
        enter_ptr, enter_ind, exit_ptr, exit_ind, fail_ptr, fail_ind, _, nuft = efron_csr
    else:
        if len(efron_pre) == 6:
            _, uft_ix, risk_enter, risk_exit, nuft, _ = efron_pre
        else:
            _, uft_ix, risk_enter, risk_exit, nuft = efron_pre
        enter_ptr, enter_ind, exit_ptr, exit_ind, fail_ptr, fail_ind = efron_indices_to_csr(
            uft_ix, risk_enter, risk_exit, nuft
        )

    if nuft == 0:
        p = int(X.shape[1])
        return cp.zeros(p, dtype=cp.float64), cp.zeros((p, p), dtype=cp.float64)

    n, p = int(X.shape[0]), int(X.shape[1])
    linpred = X @ beta
    linpred = linpred - cp.max(linpred)
    e_eta = cp.exp(linpred)

    enter_ptr_g = cp.asarray(enter_ptr)
    enter_ind_g = cp.asarray(enter_ind)
    exit_ptr_g = cp.asarray(exit_ptr)
    exit_ind_g = cp.asarray(exit_ind)
    fail_ptr_g = cp.asarray(fail_ptr)
    fail_ind_g = cp.asarray(fail_ind)

    grad_out = cp.zeros(p, dtype=cp.float64)
    hess_out = cp.zeros((p, p), dtype=cp.float64)
    ws = 2 + 2 * p + 3 * p * p
    workspace = cp.zeros(ws, dtype=cp.float64)

    seq_thresh, threads = _pick_backward_launch_params(p, int(nuft), int(n))
    use_serial = (p <= 24 and int(nuft) <= 512 and _env_int("STATGPU_EFRON_SERIAL_KERNEL", 1) == 1)
    meta = cp.array([n, p, nuft, seq_thresh], dtype=cp.int32)
    kernel = get_efron_backward_kernel_serial(cp) if use_serial else get_efron_backward_kernel(cp)
    try:
        kernel(
            (1,),
            ((1,) if use_serial else (threads,)),
            (
                X,
                e_eta,
                meta,
                enter_ptr_g,
                enter_ind_g,
                exit_ptr_g,
                exit_ind_g,
                fail_ptr_g,
                fail_ind_g,
                grad_out,
                hess_out.reshape(-1),
                workspace,
            ),
        )
        # Surface asynchronous kernel execution errors at this call site so fallback
        # behavior is reliable and diagnostics point to the correct launch.
        cp.cuda.Stream.null.synchronize()
    except Exception:
        return None

    return grad_out, hess_out


_BRESLOW_KERNEL_VER = 1
_breslow_kernel_cache = None

_BRESLOW_KERNEL_SOURCE = r"""
extern "C" __global__
void breslow_backward_hess_scan(
    const double* __restrict__ X,
    const double* __restrict__ e_eta,
    const int* __restrict__ first_idx,
    const double* __restrict__ counts,
    const int* __restrict__ meta,   // [n, p, nuft, seq_thresh]
    double* __restrict__ hess_out,
    double* __restrict__ workspace
) {
    int n = meta[0];
    int p = meta[1];
    int nuft = meta[2];
    int seq_thresh = meta[3];
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;
    if (threadIdx.y != 0 || threadIdx.z != 0) return;

    double* xp0_ptr = workspace;             // 1
    double* xp1 = xp0_ptr + 1;               // p
    double* xp2 = xp1 + p;                   // p*p
    double* hess_acc = xp2 + p * p;          // p*p
    int ws_doubles = 1 + p + 2 * p * p;

    for (int i = threadIdx.x; i < ws_doubles; i += blockDim.x) {
        workspace[i] = 0.0;
    }
    __syncthreads();

    for (int ii = nuft - 1; ii >= 0; --ii) {
        int start = first_idx[ii];
        int end = (ii == nuft - 1) ? n : first_idx[ii + 1];
        int nt = end - start;

        if (nt > 0) {
            if (nt <= seq_thresh) {
                if (threadIdx.x == 0) {
                    for (int r = start; r < end; ++r) {
                        const double* Xrow = X + (size_t)r * (size_t)p;
                        double elx = e_eta[r];
                        *xp0_ptr += elx;
                        for (int j = 0; j < p; ++j) {
                            double vj = Xrow[j];
                            xp1[j] += elx * vj;
                        }
                        for (int j = 0; j < p; ++j) {
                            double vj = Xrow[j];
                            for (int k = 0; k < p; ++k) {
                                xp2[j * p + k] += elx * vj * Xrow[k];
                            }
                        }
                    }
                }
            } else {
                for (int rr = threadIdx.x; rr < nt; rr += blockDim.x) {
                    int r = start + rr;
                    const double* Xrow = X + (size_t)r * (size_t)p;
                    double elx = e_eta[r];
                    atomicAdd(xp0_ptr, elx);
                    for (int j = 0; j < p; ++j) {
                        atomicAdd(xp1 + j, elx * Xrow[j]);
                    }
                    for (int j = 0; j < p; ++j) {
                        double vj = Xrow[j];
                        for (int k = 0; k < p; ++k) {
                            atomicAdd(xp2 + j * p + k, elx * vj * Xrow[k]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            double rs = *xp0_ptr;
            double w = counts[ii];
            if (rs > 1e-300 && w != 0.0) {
                double inv = 1.0 / rs;
                double inv2 = inv * inv;
                for (int j1 = 0; j1 < p; ++j1) {
                    double x1 = xp1[j1];
                    for (int j2 = 0; j2 < p; ++j2) {
                        double exx = xp2[j1 * p + j2] * inv;
                        double ex1ex2 = x1 * xp1[j2] * inv2;
                        hess_acc[j1 * p + j2] += w * (exx - ex1ex2);
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int j = threadIdx.x; j < p * p; j += blockDim.x) {
        hess_out[j] = -hess_acc[j];
    }
}
"""


def get_breslow_hess_kernel(cp):
    global _breslow_kernel_cache
    if (
        _breslow_kernel_cache is None
        or not isinstance(_breslow_kernel_cache, tuple)
        or _breslow_kernel_cache[1] != _BRESLOW_KERNEL_VER
    ):
        _breslow_kernel_cache = (
            cp.RawKernel(_BRESLOW_KERNEL_SOURCE, "breslow_backward_hess_scan"),
            _BRESLOW_KERNEL_VER,
        )
    return _breslow_kernel_cache[0]


def compute_breslow_hess_raw(
    X,
    first_idx_uft,
    counts_uft,
    *,
    cupy_module,
    exp_eta=None,
    beta=None,
):
    """
    Fused CuPy RawKernel Hessian for Breslow grouped ties.
    Returns hess (cupy array) or None on launch/compile failure.
    """
    cp = cupy_module
    nuft = int(first_idx_uft.size)
    p = int(X.shape[1])
    if nuft == 0:
        return cp.zeros((p, p), dtype=cp.float64)

    n = int(X.shape[0])
    if exp_eta is None:
        if beta is None:
            raise ValueError("compute_breslow_hess_raw requires either exp_eta or beta")
        linpred = X @ beta
        linpred = linpred - cp.max(linpred)
        e_eta = cp.exp(linpred)
    else:
        e_eta = exp_eta

    first_idx_g = cp.asarray(first_idx_uft, dtype=cp.int32)
    counts_g = cp.asarray(counts_uft, dtype=cp.float64)
    hess_out = cp.zeros((p, p), dtype=cp.float64)
    workspace = cp.zeros(1 + p + 2 * p * p, dtype=cp.float64)
    seq_thresh, threads = _pick_backward_launch_params(p, nuft, n)
    meta = cp.array([n, p, nuft, seq_thresh], dtype=cp.int32)
    kernel = get_breslow_hess_kernel(cp)
    try:
        kernel(
            (1,),
            (threads,),
            (
                X,
                e_eta,
                first_idx_g,
                counts_g,
                meta,
                hess_out.reshape(-1),
                workspace,
            ),
        )
    except Exception:
        return None
    return hess_out
