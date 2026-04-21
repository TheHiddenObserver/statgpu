# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

"""
Cython-optimized Efron gradient and Hessian computation.
"""

from libc.math cimport exp, log
import numpy as np
cimport numpy as np

ctypedef np.float64_t DOUBLE_t

# Clip bounds for exp to prevent overflow
cdef double MAX_LINPRED = 700.0
cdef double MIN_LINPRED = -700.0


cpdef tuple efron_grad_hess(
    const DOUBLE_t[::1] linpred,
    const DOUBLE_t[:, ::1] X,
    object risk_enter,
    object risk_exit,
    object uft_ix,
    int nuft,
):
    """
    Cython-optimized Efron gradient and Hessian computation.

    Parameters
    ----------
    linpred : ndarray (n_samples,)
        Linear predictor = X @ beta. Will be clipped to [-700, 700] before exp.
    X : ndarray (n_samples, n_features)
        Covariate matrix.
    risk_enter : list of list
        Indices of samples entering risk set at each unique failure time.
    risk_exit : list of list
        Indices of samples exiting risk set at each unique failure time.
    uft_ix : list of list
        Indices of samples experiencing event at each unique failure time.
    nuft : int
        Number of unique failure times.

    Returns
    -------
    grad : ndarray (n_features,)
        Gradient of log partial likelihood.
    hess : ndarray (n_features, n_features)
        Hessian of log partial likelihood (negative definite).
    """
    cdef:
        int n_features = X.shape[1]
        int i, j, k, l, m, idx
        double xp0 = 0.0
        double xp0f = 0.0
        double elx, c0
        double sum_inv_c0, sum_J_c0, sum_aa, sum_bb, sum_ab
        double ak, bk, J_val
        double xjk, xjl, lp

    cdef object ix
    cdef object ixf

    cdef DOUBLE_t[::1] grad = np.zeros(n_features, dtype=np.float64)
    cdef DOUBLE_t[:, ::1] hess_inner = np.zeros((n_features, n_features), dtype=np.float64)

    # Working arrays for risk set accumulation
    cdef DOUBLE_t[::1] xp1 = np.zeros(n_features, dtype=np.float64)
    cdef DOUBLE_t[:, ::1] xp2 = np.zeros((n_features, n_features), dtype=np.float64)
    cdef DOUBLE_t[::1] xp1f = np.zeros(n_features, dtype=np.float64)
    cdef DOUBLE_t[:, ::1] xp2f = np.zeros((n_features, n_features), dtype=np.float64)

    # Backward scan over unique failure times.
    # Keep this algebra identical to `efron_grad_hess_python` for numerical parity.
    for i in range(nuft - 1, -1, -1):
        ix = risk_enter[i]
        m = len(ix)
        if m > 0:
            for idx in range(m):
                j = ix[idx]
                lp = linpred[j]
                if lp > MAX_LINPRED:
                    lp = MAX_LINPRED
                elif lp < MIN_LINPRED:
                    lp = MIN_LINPRED
                elx = exp(lp)
                xp0 += elx
                for k in range(n_features):
                    xjk = X[j, k]
                    xp1[k] += elx * xjk
                    for l in range(n_features):
                        xjl = X[j, l]
                        xp2[k, l] += elx * xjk * xjl

        ixf = uft_ix[i]
        m = len(ixf)
        if m > 0:
            xp0f = 0.0
            for k in range(n_features):
                xp1f[k] = 0.0
                for l in range(n_features):
                    xp2f[k, l] = 0.0

            for idx in range(m):
                j = ixf[idx]
                lp = linpred[j]
                if lp > MAX_LINPRED:
                    lp = MAX_LINPRED
                elif lp < MIN_LINPRED:
                    lp = MIN_LINPRED
                elx = exp(lp)
                xp0f += elx
                for k in range(n_features):
                    xjk = X[j, k]
                    xp1f[k] += elx * xjk
                    for l in range(n_features):
                        xjl = X[j, l]
                        xp2f[k, l] += elx * xjk * xjl

            sum_inv_c0 = 0.0
            sum_J_c0 = 0.0
            sum_aa = 0.0
            sum_bb = 0.0
            sum_ab = 0.0

            for k in range(m):
                J_val = <double>k / <double>m
                c0 = xp0 - J_val * xp0f
                if c0 < 1e-300:
                    c0 = 1e-300
                ak = 1.0 / c0
                bk = J_val * ak
                sum_inv_c0 += ak
                sum_J_c0 += bk
                sum_aa += ak * ak
                sum_bb += bk * bk
                sum_ab += ak * bk

            for idx in range(m):
                j = ixf[idx]
                for k in range(n_features):
                    grad[k] += X[j, k]

            for k in range(n_features):
                grad[k] -= xp1[k] * sum_inv_c0 - xp1f[k] * sum_J_c0

            for k in range(n_features):
                for l in range(n_features):
                    hess_inner[k, l] += xp2[k, l] * sum_inv_c0
                    hess_inner[k, l] -= xp2f[k, l] * sum_J_c0
                    hess_inner[k, l] -= sum_aa * xp1[k] * xp1[l]
                    hess_inner[k, l] -= sum_bb * xp1f[k] * xp1f[l]
                    hess_inner[k, l] += sum_ab * (
                        xp1[k] * xp1f[l] + xp1f[k] * xp1[l]
                    )

        ix = risk_exit[i]
        m = len(ix)
        if m > 0:
            for idx in range(m):
                j = ix[idx]
                lp = linpred[j]
                if lp > MAX_LINPRED:
                    lp = MAX_LINPRED
                elif lp < MIN_LINPRED:
                    lp = MIN_LINPRED
                elx = exp(lp)
                xp0 -= elx
                for k in range(n_features):
                    xjk = X[j, k]
                    xp1[k] -= elx * xjk
                    for l in range(n_features):
                        xjl = X[j, l]
                        xp2[k, l] -= elx * xjk * xjl

    return np.asarray(grad), -np.asarray(hess_inner)


def efron_grad_hess_python(linpred, X, risk_enter, risk_exit, uft_ix, nuft):
    """Pure Python fallback implementation."""
    import numpy as np

    n_features = X.shape[1]
    grad = np.zeros(n_features, dtype=np.float64)
    hess_inner = np.zeros((n_features, n_features), dtype=np.float64)
    xp0 = 0.0
    xp1 = np.zeros(n_features, dtype=np.float64)
    xp2 = np.zeros((n_features, n_features), dtype=np.float64)
    e_linpred = np.exp(linpred)

    for i in range(nuft)[::-1]:
        ix = risk_enter[i]
        if len(ix) > 0:
            elx = e_linpred[ix]
            v = X[ix]
            xp0 += elx.sum()
            xp1 += (elx[:, None] * v).sum(axis=0)
            xp2 += np.einsum("ij,ik,i->jk", v, v, elx)

        ixf = uft_ix[i]
        if len(ixf) > 0:
            ixf = np.asarray(ixf, dtype=np.intp)
            v = X[ixf]
            elx = e_linpred[ixf]
            xp0f = elx.sum()
            xp1f = (elx[:, None] * v).sum(axis=0)
            xp2f = np.einsum("ij,ik,i->jk", v, v, elx)
            m = len(ixf)
            J = np.arange(m, dtype=np.float64) / max(m, 1)
            c0 = xp0 - J * xp0f
            c0 = np.maximum(c0, 1e-300)
            inv = 1.0 / c0
            ak = inv
            bk = J * inv
            sum_inv_c0 = np.sum(ak)
            sum_J_c0 = np.sum(bk)
            sum_aa = np.sum(ak * ak)
            sum_bb = np.sum(bk * bk)
            sum_ab = np.sum(ak * bk)
            grad += v.sum(axis=0)
            grad -= xp1 * sum_inv_c0 - xp1f * sum_J_c0
            hess_inner += xp2 * sum_inv_c0
            hess_inner -= xp2f * sum_J_c0
            hess_inner -= (
                sum_aa * np.outer(xp1, xp1)
                + sum_bb * np.outer(xp1f, xp1f)
                - sum_ab * (np.outer(xp1, xp1f) + np.outer(xp1f, xp1))
            )

        ix = risk_exit[i]
        if len(ix) > 0:
            elx = e_linpred[ix]
            v = X[ix]
            xp0 -= elx.sum()
            xp1 -= (elx[:, None] * v).sum(axis=0)
            xp2 -= np.einsum("ij,ik,i->jk", v, v, elx)

    return np.asarray(grad), -np.asarray(hess_inner)
