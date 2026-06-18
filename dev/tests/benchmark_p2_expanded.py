"""P2 Expanded Benchmark: Large-scale + 3-backend comparison.

Tests at n=500/2000/5000, across numpy/cupy/torch, with external framework comparison.
Run on remote server with GPU: python dev/tests/benchmark_p2_expanded.py
"""

import time
import json
import os
import traceback

import numpy as np


def timeit(func, n_repeats=None):
    """Run func n_repeats times, return (result, median_time_ms)."""
    if n_repeats is None:
        n_repeats = 3
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, np.median(times)


def _to_np(x):
    """Convert any backend array to numpy."""
    if hasattr(x, 'get'):
        return x.get()
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_backend_module(name):
    """Get array module for backend name."""
    if name == "numpy":
        return np
    elif name == "cupy":
        import cupy
        return cupy
    elif name == "torch":
        import torch
        return torch
    return np


def _make_torch_cuda(arr):
    """Convert numpy array to torch CUDA tensor."""
    import torch
    return torch.from_numpy(arr).cuda().double()


# ============================================================================
# Section 1: ANOVA — 3-backend + scipy comparison
# ============================================================================

def bench_anova_3backend():
    """ANOVA across numpy/cupy/torch + scipy."""
    results = {}
    np.random.seed(42)

    for n in [5000, 20000, 50000, 100000]:
        g1 = np.random.randn(n) + 0
        g2 = np.random.randn(n) + 1
        g3 = np.random.randn(n) + 2

        key = f"n={n}"
        results[key] = {}

        # scipy baseline
        from scipy.stats import f_oneway as sp_f
        r_scipy, t_scipy = timeit(lambda: sp_f(g1, g2, g3))
        results[key]["scipy"] = {
            "F": float(r_scipy.statistic), "time_ms": t_scipy
        }

        # numpy backend
        from statgpu.anova import f_oneway
        r_np, t_np = timeit(lambda: f_oneway(g1, g2, g3, backend="numpy"))
        results[key]["numpy"] = {
            "F": r_np.statistic, "p": r_np.pvalue,
            "max_diff_F": abs(r_np.statistic - float(r_scipy.statistic)),
            "time_ms": t_np,
        }

        # cupy backend
        try:
            import cupy as cp
            g1_c = cp.asarray(g1)
            g2_c = cp.asarray(g2)
            g3_c = cp.asarray(g3)
            r_cp, t_cp = timeit(lambda: f_oneway(g1_c, g2_c, g3_c, backend="cupy"))
            results[key]["cupy"] = {
                "F": r_cp.statistic,
                "max_diff_F": abs(r_cp.statistic - float(r_scipy.statistic)),
                "time_ms": t_cp,
            }
            del g1_c, g2_c, g3_c
        except Exception as e:
            results[key]["cupy"] = {"error": str(e)}

        # torch backend
        try:
            import torch
            g1_t = _make_torch_cuda(g1)
            g2_t = _make_torch_cuda(g2)
            g3_t = _make_torch_cuda(g3)
            r_t, t_t = timeit(lambda: f_oneway(g1_t, g2_t, g3_t, backend="torch"))
            results[key]["torch"] = {
                "F": r_t.statistic,
                "max_diff_F": abs(r_t.statistic - float(r_scipy.statistic)),
                "time_ms": t_t,
            }
            del g1_t, g2_t, g3_t
        except Exception as e:
            results[key]["torch"] = {"error": str(e)}

    return results


# ============================================================================
# Section 2: Covariance — 3-backend + sklearn comparison
# ============================================================================

def bench_covariance_3backend():
    """Covariance across numpy/cupy/torch + sklearn."""
    results = {}
    np.random.seed(42)

    for n in [5000, 20000, 50000]:
        p = 50
        X = np.random.randn(n, p)
        key = f"n={n},p={p}"
        results[key] = {}

        # sklearn baseline
        from sklearn.covariance import LedoitWolf as SkLW
        sk_lw, t_sk = timeit(lambda: SkLW().fit(X))
        results[key]["sklearn_LedoitWolf"] = {"time_ms": t_sk}

        # numpy
        from statgpu.covariance import LedoitWolf
        lw_np, t_np = timeit(lambda: LedoitWolf(device='cpu').fit(X))
        cov_np = _to_np(lw_np.covariance_)
        results[key]["numpy"] = {
            "corr": float(np.corrcoef(cov_np.ravel(), sk_lw.covariance_.ravel())[0, 1]),
            "time_ms": t_np,
        }

        # cupy
        try:
            import cupy as cp
            X_c = cp.asarray(X)
            lw_cp, t_cp = timeit(lambda: LedoitWolf(device='cuda').fit(X_c))
            cov_cp = _to_np(lw_cp.covariance_)
            results[key]["cupy"] = {
                "corr": float(np.corrcoef(cov_cp.ravel(), sk_lw.covariance_.ravel())[0, 1]),
                "time_ms": t_cp,
            }
            del X_c
        except Exception as e:
            results[key]["cupy"] = {"error": str(e)}

        # torch
        try:
            import torch
            X_t = _make_torch_cuda(X)
            lw_t, t_t = timeit(lambda: LedoitWolf(device='torch').fit(X_t))
            cov_t = _to_np(lw_t.covariance_)
            results[key]["torch"] = {
                "corr": float(np.corrcoef(cov_t.ravel(), sk_lw.covariance_.ravel())[0, 1]),
                "time_ms": t_t,
            }
            del X_t
        except Exception as e:
            results[key]["torch"] = {"error": str(e)}

    return results


# ============================================================================
# Section 3: Panel — 3-backend + statsmodels comparison
# ============================================================================

def bench_panel_3backend():
    """Panel models across numpy/cupy/torch + statsmodels."""
    results = {}
    np.random.seed(42)

    for n in [5000, 20000, 50000]:
        n_entities = min(500, n // 10)
        n_per = n // n_entities
        X = np.random.randn(n, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(n) * 0.5
        eids = np.repeat(np.arange(n_entities), n_per)[:n]
        tids = np.tile(np.arange(n_per), n_entities)[:n]

        key = f"n={n}"
        results[key] = {}

        # statsmodels baseline
        try:
            import statsmodels.api as sm
            X_sm = sm.add_constant(X)
            m_sm, t_sm = timeit(lambda: sm.OLS(y, X_sm).fit())
            results[key]["statsmodels"] = {"time_ms": t_sm}
        except ImportError:
            results[key]["statsmodels"] = {"error": "not installed"}

        # numpy
        from statgpu.panel import PooledOLS
        m_np, t_np = timeit(lambda: PooledOLS(device='cpu').fit(X, y))
        results[key]["numpy_PooledOLS"] = {"time_ms": t_np, "R2": m_np.rsquared}

        # cupy
        try:
            import cupy as cp
            X_c = cp.asarray(X)
            y_c = cp.asarray(y)
            m_cp, t_cp = timeit(lambda: PooledOLS(device='cuda').fit(X_c, y_c))
            results[key]["cupy_PooledOLS"] = {"time_ms": t_cp}
            del X_c, y_c
        except Exception as e:
            results[key]["cupy_PooledOLS"] = {"error": str(e)}

        # torch
        try:
            import torch
            X_t = _make_torch_cuda(X)
            y_t = _make_torch_cuda(y)
            m_t, t_t = timeit(lambda: PooledOLS(device='torch').fit(X_t, y_t))
            results[key]["torch_PooledOLS"] = {"time_ms": t_t}
            del X_t, y_t
        except Exception as e:
            results[key]["torch_PooledOLS"] = {"error": str(e)}

        # FamaMacBeth
        from statgpu.panel import FamaMacBeth
        m_fmb, t_fmb = timeit(lambda: FamaMacBeth(device='cpu').fit(X, y, time_ids=tids))
        results[key]["FamaMacBeth"] = {"time_ms": t_fmb, "n_periods": m_fmb.n_periods}

    return results


# ============================================================================
# Section 4: Splines — 3-backend + sklearn comparison
# ============================================================================

def bench_splines_3backend():
    """Splines across numpy/cupy/torch + sklearn."""
    results = {}
    np.random.seed(42)

    for n in [5000, 20000, 50000]:
        p = 10
        X = np.random.randn(n, p)
        key = f"n={n}"
        results[key] = {}

        # sklearn baseline
        try:
            from sklearn.preprocessing import SplineTransformer as SkST
            sk_st, t_sk = timeit(lambda: SkST(n_knots=10).fit_transform(X))
            results[key]["sklearn"] = {"shape": sk_st.shape, "time_ms": t_sk}
        except Exception as e:
            results[key]["sklearn"] = {"error": str(e)}

        # numpy
        from statgpu.nonparametric.splines import SplineTransformer
        st_np, t_np = timeit(lambda: SplineTransformer(n_knots=10, device='cpu').fit_transform(X))
        results[key]["numpy"] = {"shape": st_np.shape, "time_ms": t_np}

        # cupy
        try:
            import cupy as cp
            X_c = cp.asarray(X)
            st_cp, t_cp = timeit(lambda: SplineTransformer(n_knots=10, device='cuda').fit_transform(X_c))
            results[key]["cupy"] = {"shape": _to_np(st_cp).shape, "time_ms": t_cp}
            del X_c
        except Exception as e:
            results[key]["cupy"] = {"error": str(e)}

        # torch
        try:
            import torch
            X_t = _make_torch_cuda(X)
            st_t, t_t = timeit(lambda: SplineTransformer(n_knots=10, device='torch').fit_transform(X_t))
            results[key]["torch"] = {"shape": _to_np(st_t).shape, "time_ms": t_t}
            del X_t
        except Exception as e:
            results[key]["torch"] = {"error": str(e)}

    return results


# ============================================================================
# Section 5: Kernel — 3-backend + sklearn comparison
# ============================================================================

def bench_kernel_3backend():
    """Kernel methods across numpy/cupy/torch + sklearn."""
    results = {}
    np.random.seed(42)

    for n in [5000, 20000, 50000]:
        p = 20
        X = np.random.randn(n, p)
        key = f"n={n}"
        results[key] = {}

        n_comp = min(100, n // 10)

        # rbf_kernel: 3-backend (uses xp parameter)
        from statgpu.nonparametric.kernel_methods import rbf_kernel, chi2_kernel
        K_rbf_np, t_rbf_np = timeit(lambda: rbf_kernel(X, xp=np))
        results[key]["numpy_rbf"] = {"time_ms": t_rbf_np}

        try:
            import cupy as cp
            X_c = cp.asarray(X)
            K_rbf_cp, t_rbf_cp = timeit(lambda: rbf_kernel(X_c, xp=cp))
            results[key]["cupy_rbf"] = {"time_ms": t_rbf_cp}
            del X_c, K_rbf_cp
        except Exception as e:
            results[key]["cupy_rbf"] = {"error": str(e)}

        try:
            import torch
            X_t = _make_torch_cuda(X)
            K_rbf_t, t_rbf_t = timeit(lambda: rbf_kernel(X_t, xp=torch))
            results[key]["torch_rbf"] = {"time_ms": t_rbf_t}
            del X_t, K_rbf_t
        except Exception as e:
            results[key]["torch_rbf"] = {"error": str(e)}

        # chi2_kernel: 3-backend (uses xp parameter, not device)
        X_pos = np.abs(X)  # chi2 needs non-negative
        K_np, t_chi2_np = timeit(lambda: chi2_kernel(X_pos, xp=np))
        results[key]["numpy_chi2"] = {"time_ms": t_chi2_np}

        try:
            import cupy as cp
            X_c = cp.asarray(X_pos)
            K_cp, t_chi2_cp = timeit(lambda: chi2_kernel(X_c, xp=cp))
            results[key]["cupy_chi2"] = {"time_ms": t_chi2_cp}
            del X_c, K_cp
        except Exception as e:
            results[key]["cupy_chi2"] = {"error": str(e)}

        try:
            import torch
            X_t = _make_torch_cuda(X_pos)
            K_t, t_chi2_t = timeit(lambda: chi2_kernel(X_t, xp=torch))
            results[key]["torch_chi2"] = {"time_ms": t_chi2_t}
            del X_t, K_t
        except Exception as e:
            results[key]["torch_chi2"] = {"error": str(e)}

        # sklearn baseline
        try:
            from sklearn.kernel_approximation import Nystroem as SkNy
            sk_ny, t_sk = timeit(lambda: SkNy(n_components=n_comp, random_state=42).fit_transform(X))
            results[key]["sklearn_Nystroem"] = {"shape": sk_ny.shape, "time_ms": t_sk}
        except Exception as e:
            results[key]["sklearn_Nystroem"] = {"error": str(e)}

        # numpy
        from statgpu.nonparametric.kernel_methods import Nystroem, KernelPCA
        ny_np, t_np = timeit(lambda: Nystroem(n_components=n_comp, random_state=42, device='cpu').fit_transform(X))
        results[key]["numpy_Nystroem"] = {"shape": ny_np.shape, "time_ms": t_np}

        # cupy
        try:
            import cupy as cp
            X_c = cp.asarray(X)
            ny_cp, t_cp = timeit(lambda: Nystroem(n_components=n_comp, random_state=42, device='cuda').fit_transform(X_c))
            results[key]["cupy_Nystroem"] = {"shape": _to_np(ny_cp).shape, "time_ms": t_cp}
            del X_c
        except Exception as e:
            results[key]["cupy_Nystroem"] = {"error": str(e)}

        # torch
        try:
            import torch
            X_t = _make_torch_cuda(X)
            ny_t, t_t = timeit(lambda: Nystroem(n_components=n_comp, random_state=42, device='torch').fit_transform(X_t))
            results[key]["torch_Nystroem"] = {"shape": _to_np(ny_t).shape, "time_ms": t_t}
            del X_t
        except Exception as e:
            results[key]["torch_Nystroem"] = {"error": str(e)}

        # KernelPCA (only for smaller n since it's O(n^2))
        if n <= 20000:
            kpca_np, t_kpca_np = timeit(lambda: KernelPCA(n_components=5, device='cpu').fit_transform(X))
            results[key]["numpy_KernelPCA"] = {"shape": kpca_np.shape, "time_ms": t_kpca_np}

            try:
                X_c = cp.asarray(X)
                kpca_cp, t_kpca_cp = timeit(lambda: KernelPCA(n_components=5, device='cuda').fit_transform(X_c))
                results[key]["cupy_KernelPCA"] = {"shape": _to_np(kpca_cp).shape, "time_ms": t_kpca_cp}
                del X_c
            except Exception as e:
                results[key]["cupy_KernelPCA"] = {"error": str(e)}

            try:
                X_t = _make_torch_cuda(X)
                kpca_t, t_kpca_t = timeit(lambda: KernelPCA(n_components=5, device='torch').fit_transform(X_t))
                results[key]["torch_KernelPCA"] = {"shape": _to_np(kpca_t).shape, "time_ms": t_kpca_t}
                del X_t
            except Exception as e:
                results[key]["torch_KernelPCA"] = {"error": str(e)}

    return results


# ============================================================================
# Section 6: MinCovDet precision verification
# ============================================================================

def bench_mincovdet_precision():
    """Verify MinCovDet precision vs sklearn at multiple scales."""
    results = {}
    np.random.seed(42)

    for n in [500, 2000, 10000]:
        p = 20
        X = np.random.randn(n, p)

        from statgpu.covariance import MinCovDet
        mcd = MinCovDet(random_state=42, device='cpu').fit(X)
        from sklearn.covariance import MinCovDet as SkMCD
        sk = SkMCD(random_state=42).fit(X)

        mcd_cov = _to_np(mcd.covariance_)
        results[f"n={n}"] = {
            "corr": float(np.corrcoef(mcd_cov.ravel(), sk.covariance_.ravel())[0, 1]),
            "max_diff": float(np.max(np.abs(mcd_cov - sk.covariance_))),
            "location_diff": float(np.max(np.abs(_to_np(mcd.location_) - sk.location_))),
            "support_agree": int((mcd.support_ == sk.support_).sum()),
            "support_total": n,
        }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("P2 Expanded Benchmark: Large-scale + 3-backend Comparison")
    print("=" * 70)

    all_results = {}

    sections = [
        ("ANOVA_3backend", bench_anova_3backend),
        ("Covariance_3backend", bench_covariance_3backend),
        ("Panel_3backend", bench_panel_3backend),
        ("Splines_3backend", bench_splines_3backend),
        ("Kernel_3backend", bench_kernel_3backend),
        ("MinCovDet_precision", bench_mincovdet_precision),
    ]

    for name, func in sections:
        print(f"\n[{name}]")
        try:
            result = func()
            all_results[name] = result
            # Print summary
            for k, v in result.items():
                if isinstance(v, dict):
                    summary = {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                    print(f"  {k}: {summary}")
        except Exception as e:
            all_results[name] = {"error": str(e)}
            print(f"  ERROR: {e}")
            traceback.print_exc()

    os.makedirs("results", exist_ok=True)
    out_path = "results/p2_benchmark_expanded.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
