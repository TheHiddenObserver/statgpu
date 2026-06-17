"""P2 Module Benchmark: Precision and Timing across 3 backends + external frameworks.

Run on remote server with: python dev/tests/benchmark_p2_remote.py
Requires: statgpu, scipy, sklearn, numpy, cupy, torch
"""

import time
import json
import os
import sys
import traceback

import numpy as np

# ============================================================================
# Timing utility
# ============================================================================

def timeit(func, n_repeats=3):
    """Run func n_repeats times, return (result, median_time_ms)."""
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, np.median(times)


# ============================================================================
# Benchmark sections
# ============================================================================

def bench_anova():
    """Benchmark ANOVA: f_oneway, f_twoway, f_welch, post-hoc, effect sizes."""
    results = {}
    np.random.seed(42)

    # Data
    g1 = np.random.randn(100) + 0
    g2 = np.random.randn(100) + 1
    g3 = np.random.randn(100) + 2
    data_2way = [[np.random.randn(50) + i + j for j in range(3)] for i in range(2)]

    # -- f_oneway vs scipy --
    from statgpu.anova import f_oneway
    r_statgpu, t_statgpu = timeit(lambda: f_oneway(g1, g2, g3))
    from scipy.stats import f_oneway as sp_f_oneway
    r_scipy, t_scipy = timeit(lambda: sp_f_oneway(g1, g2, g3))

    results["f_oneway"] = {
        "statgpu_F": r_statgpu.statistic,
        "scipy_F": float(r_scipy.statistic),
        "statgpu_p": r_statgpu.pvalue,
        "scipy_p": float(r_scipy.pvalue),
        "max_diff_F": abs(r_statgpu.statistic - float(r_scipy.statistic)),
        "max_diff_p": abs(r_statgpu.pvalue - float(r_scipy.pvalue)),
        "time_statgpu_ms": t_statgpu,
        "time_scipy_ms": t_scipy,
    }

    # -- f_twoway --
    from statgpu.anova import f_twoway
    r2, t2 = timeit(lambda: f_twoway(data_2way, interaction=True))
    results["f_twoway"] = {
        "F_A": r2.factor_a_statistic,
        "F_B": r2.factor_b_statistic,
        "F_AB": r2.interaction_statistic,
        "time_ms": t2,
    }

    # -- f_welch --
    from statgpu.anova import f_welch
    r3, t3 = timeit(lambda: f_welch(g1, g2, g3))
    results["f_welch"] = {"F": r3.statistic, "p": r3.pvalue, "time_ms": t3}

    # -- tukey_hsd --
    from statgpu.anova import tukey_hsd
    r4, t4 = timeit(lambda: tukey_hsd(g1, g2, g3))
    results["tukey_hsd"] = {"n_comparisons": len(r4.comparisons), "time_ms": t4}

    # -- bonferroni --
    from statgpu.anova import bonferroni
    r5, t5 = timeit(lambda: bonferroni(g1, g2, g3))
    results["bonferroni"] = {"n_comparisons": len(r5.comparisons), "time_ms": t5}

    return results


def _to_np(x):
    """Convert any backend array to numpy."""
    if hasattr(x, 'get'):
        return x.get()
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def bench_covariance():
    """Benchmark Covariance: ShrunkCovariance, MinCovDet, GraphicalLasso vs sklearn."""
    results = {}
    np.random.seed(42)
    X = np.random.randn(200, 10)

    # -- ShrunkCovariance vs sklearn --
    from statgpu.covariance import ShrunkCovariance
    sg, t_sg = timeit(lambda: ShrunkCovariance(shrinkage=0.3, device='cpu').fit(X))
    from sklearn.covariance import ShrunkCovariance as SkShrunk
    sk, t_sk = timeit(lambda: SkShrunk(shrinkage=0.3).fit(X))
    sg_cov = _to_np(sg.covariance_)
    results["ShrunkCovariance"] = {
        "max_diff": float(np.max(np.abs(sg_cov - sk.covariance_))),
        "corr": float(np.corrcoef(sg_cov.ravel(), sk.covariance_.ravel())[0, 1]),
        "time_statgpu_ms": t_sg,
        "time_sklearn_ms": t_sk,
    }

    # -- MinCovDet vs sklearn --
    from statgpu.covariance import MinCovDet
    mcd, t_mcd = timeit(lambda: MinCovDet(random_state=42, device='cpu').fit(X))
    from sklearn.covariance import MinCovDet as SkMCD
    sk_mcd, t_sk_mcd = timeit(lambda: SkMCD(random_state=42).fit(X))
    mcd_cov = _to_np(mcd.covariance_)
    results["MinCovDet"] = {
        "corr": float(np.corrcoef(mcd_cov.ravel(), sk_mcd.covariance_.ravel())[0, 1]),
        "time_statgpu_ms": t_mcd,
        "time_sklearn_ms": t_sk_mcd,
    }

    # -- GraphicalLasso vs sklearn --
    from statgpu.covariance import GraphicalLasso
    gl, t_gl = timeit(lambda: GraphicalLasso(alpha=0.1, max_iter=50, device='cpu').fit(X))
    from sklearn.covariance import GraphicalLasso as SkGL
    sk_gl, t_sk_gl = timeit(lambda: SkGL(alpha=0.1, max_iter=50).fit(X))
    gl_prec = _to_np(gl.precision_)
    results["GraphicalLasso"] = {
        "corr": float(np.corrcoef(gl_prec.ravel(), sk_gl.precision_.ravel())[0, 1]),
        "n_iter": gl.n_iter_,
        "time_statgpu_ms": t_gl,
        "time_sklearn_ms": t_sk_gl,
    }

    return results


def bench_panel():
    """Benchmark Panel: PooledOLS, BetweenOLS, FDOLS, FamaMacBeth, HAC."""
    results = {}
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(n) * 0.5
    eids = np.repeat(np.arange(50), 10)
    tids = np.tile(np.arange(10), 50)

    # -- PooledOLS vs statsmodels OLS --
    from statgpu.panel import PooledOLS
    m1, t1 = timeit(lambda: PooledOLS().fit(X, y))
    try:
        import statsmodels.api as sm
        X_sm = sm.add_constant(X)
        m_sm, t_sm = timeit(lambda: sm.OLS(y, X_sm).fit())
        results["PooledOLS"] = {
            "coef_diff": float(np.max(np.abs(m1.coef_ - m_sm.params))),
            "time_statgpu_ms": t1,
            "time_statsmodels_ms": t_sm,
        }
    except ImportError:
        results["PooledOLS"] = {"coef": m1.coef_.tolist(), "time_ms": t1}

    # -- BetweenOLS --
    from statgpu.panel import BetweenOLS
    m2, t2 = timeit(lambda: BetweenOLS().fit(X, y, entity_ids=eids))
    results["BetweenOLS"] = {"nobs": m2.nobs, "time_ms": t2}

    # -- FirstDifferenceOLS --
    from statgpu.panel import FirstDifferenceOLS
    m3, t3 = timeit(lambda: FirstDifferenceOLS().fit(X, y, entity_ids=eids, time_ids=tids))
    results["FirstDifferenceOLS"] = {"nobs": m3.nobs, "time_ms": t3}

    # -- FamaMacBeth --
    from statgpu.panel import FamaMacBeth
    m4, t4 = timeit(lambda: FamaMacBeth().fit(X, y, time_ids=tids))
    results["FamaMacBeth"] = {"n_periods": m4.n_periods, "time_ms": t4}

    # -- HAC --
    from statgpu.panel import hac_covariance
    X_int = np.column_stack([np.ones(n), X])
    resid = y - X_int @ np.linalg.lstsq(X_int, y, rcond=None)[0]
    _, t_hac = timeit(lambda: hac_covariance(X_int, resid, bandwidth=10))
    results["HAC"] = {"time_ms": t_hac}

    return results


def bench_splines():
    """Benchmark Splines: SplineTransformer, cyclic cubic, thin plate."""
    results = {}
    np.random.seed(42)
    X = np.random.randn(500, 3)

    # -- SplineTransformer vs sklearn --
    from statgpu.nonparametric.splines import SplineTransformer
    st, t_st = timeit(lambda: SplineTransformer(n_knots=10).fit_transform(X))
    try:
        from sklearn.preprocessing import SplineTransformer as SkST
        sk_st, t_sk = timeit(lambda: SkST(n_knots=10).fit_transform(X))
        results["SplineTransformer"] = {
            "shape": st.shape,
            "time_statgpu_ms": t_st,
            "time_sklearn_ms": t_sk,
        }
    except ImportError:
        results["SplineTransformer"] = {"shape": st.shape, "time_ms": t_st}

    # -- Cyclic cubic --
    from statgpu.nonparametric.splines import cyclic_cubic_spline_basis
    x = np.linspace(0, 1, 500)
    knots = np.array([0.2, 0.4, 0.6, 0.8])
    _, t_cyc = timeit(lambda: cyclic_cubic_spline_basis(x, knots))
    results["cyclic_cubic"] = {"time_ms": t_cyc}

    # -- Thin plate --
    from statgpu.nonparametric.splines import thin_plate_spline_basis
    _, t_tp = timeit(lambda: thin_plate_spline_basis(x, knots))
    results["thin_plate"] = {"time_ms": t_tp}

    return results


def bench_kernel():
    """Benchmark Kernel: chi2_kernel, Nystroem, KernelPCA vs sklearn."""
    results = {}
    np.random.seed(42)
    X = np.abs(np.random.randn(200, 10))

    # -- chi2_kernel vs sklearn --
    from statgpu.nonparametric.kernel_methods import chi2_kernel
    K, t_ours = timeit(lambda: chi2_kernel(X))
    try:
        from sklearn.metrics.pairwise import chi2_kernel as sk_chi2
        K_sk, t_sk = timeit(lambda: sk_chi2(X))
        results["chi2_kernel"] = {
            "max_diff": float(np.max(np.abs(K - K_sk))),
            "time_statgpu_ms": t_ours,
            "time_sklearn_ms": t_sk,
        }
    except ImportError:
        results["chi2_kernel"] = {"shape": K.shape, "time_ms": t_ours}

    # -- Nystroem vs sklearn --
    X_pos = np.random.randn(200, 10)
    from statgpu.nonparametric.kernel_methods import Nystroem
    ny, t_ny = timeit(lambda: Nystroem(n_components=50, random_state=42).fit_transform(X_pos))
    try:
        from sklearn.kernel_approximation import Nystroem as SkNy
        sk_ny, t_sk_ny = timeit(lambda: SkNy(n_components=50, random_state=42).fit_transform(X_pos))
        results["Nystroem"] = {
            "shape": ny.shape,
            "time_statgpu_ms": t_ny,
            "time_sklearn_ms": t_sk_ny,
        }
    except ImportError:
        results["Nystroem"] = {"shape": ny.shape, "time_ms": t_ny}

    # -- KernelPCA vs sklearn --
    from statgpu.nonparametric.kernel_methods import KernelPCA
    kpca, t_kpca = timeit(lambda: KernelPCA(n_components=3).fit_transform(X_pos))
    try:
        from sklearn.decomposition import KernelPCA as SkKPCA
        sk_kpca, t_sk_kpca = timeit(lambda: SkKPCA(n_components=3).fit_transform(X_pos))
        results["KernelPCA"] = {
            "shape": kpca.shape,
            "time_statgpu_ms": t_kpca,
            "time_sklearn_ms": t_sk_kpca,
        }
    except ImportError:
        results["KernelPCA"] = {"shape": kpca.shape, "time_ms": t_kpca}

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("P2 Module Benchmark: Precision and Timing")
    print("=" * 70)

    all_results = {}

    sections = [
        ("ANOVA", bench_anova),
        ("Covariance", bench_covariance),
        ("Panel", bench_panel),
        ("Splines", bench_splines),
        ("Kernel", bench_kernel),
    ]

    for name, func in sections:
        print(f"\n[{name}]")
        try:
            result = func()
            all_results[name] = result
            print(f"  OK: {json.dumps(result, indent=2, default=str)[:500]}")
        except Exception as e:
            all_results[name] = {"error": str(e)}
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = "results/p2_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
