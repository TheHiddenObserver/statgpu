from pathlib import Path


def replace_once(path, old, new):
    file_path = Path(path)
    text = file_path.read_text()
    if new in text:
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    file_path.write_text(text.replace(old, new, 1))
    return True


cupy_old = '''def compute_f_stat_gpu(y, resid, X_design, df_resid):
    """
    Compute F-statistic on GPU.
    
    Parameters
    ----------
    y : cupy.ndarray
        True values on GPU.
    resid : cupy.ndarray
        Residuals on GPU.
    X_design : cupy.ndarray
        Design matrix on GPU.
    df_resid : int
        Residual degrees of freedom.
    
    Returns
    -------
    fvalue : float
        F-statistic.
    """
    import cupy as cp
    
    y_mean = y.mean()
    ss_tot = cp.sum((y - y_mean) ** 2)
    ss_res = cp.sum(resid ** 2)
    ss_reg = ss_tot - ss_res
    
    k = X_design.shape[1] - 1  # exclude intercept
    if k == 0 or ss_res <= 0:
        return (np.inf, 1.0)
    
    fvalue_gpu = (ss_reg / k) / (ss_res / df_resid)
    fvalue = float(cp.asnumpy(fvalue_gpu))
    
    # p-value on GPU using F CDF expressed via regularized incomplete beta.
    #
    # For F ~ F(d1, d2):
    #   CDF(x) = I_{ d1 x / (d1 x + d2) }(d1/2, d2/2)
    #   pvalue = 1 - CDF
    d1 = float(k)
    d2 = float(df_resid)
    if d2 <= 0 or d1 <= 0:
        pvalue = 1.0
    else:
        z = (d1 * fvalue) / (d1 * fvalue + d2)
        cdf = regularized_betainc_gpu(d1 / 2.0, d2 / 2.0, cp.asarray(z))
        pvalue = float(1.0 - cp.asnumpy(cdf))
    
    return fvalue, pvalue
'''

cupy_new = '''def compute_f_stat_gpu(y, resid, X_design, df_resid):
    """Compute the overall-regression F statistic and p-value on CuPy."""
    import cupy as cp

    y_mean = y.mean()
    ss_tot = cp.sum((y - y_mean) ** 2)
    ss_res = cp.sum(resid ** 2)

    k = int(X_design.shape[1] - 1)  # exclude intercept
    d1 = float(k)
    d2 = float(df_resid)
    if d1 <= 0.0 or d2 <= 0.0:
        return np.nan, np.nan

    # Only scalar reductions cross the host boundary.  These checks mirror the
    # public CPU LinearRegression F-statistic semantics.
    ss_tot_value = float(cp.asnumpy(ss_tot))
    ss_res_value = float(cp.asnumpy(ss_res))
    if not np.isfinite(ss_tot_value) or not np.isfinite(ss_res_value):
        return np.nan, np.nan

    tol = np.finfo(float).eps * max(1.0, abs(ss_tot_value))
    if ss_tot_value <= tol:
        return np.nan, np.nan
    if ss_res_value <= tol:
        return np.inf, 0.0

    ss_reg = cp.maximum(ss_tot - ss_res, 0.0)
    fvalue_gpu = (ss_reg / d1) / (ss_res / d2)
    fvalue = float(cp.asnumpy(fvalue_gpu))

    # For F ~ F(d1, d2), CDF(x) is a regularized incomplete beta.
    z = (d1 * fvalue) / (d1 * fvalue + d2)
    cdf = regularized_betainc_gpu(d1 / 2.0, d2 / 2.0, cp.asarray(z))
    pvalue = float(1.0 - cp.asnumpy(cdf))
    return fvalue, float(np.clip(pvalue, 0.0, 1.0))
'''

torch_old = '''def compute_f_stat_torch(y, resid, X_design, df_resid, device=None):
    """
    Compute F-statistic and p-value on Torch GPU.

    Parameters
    ----------
    y : torch.Tensor
        True values on GPU.
    resid : torch.Tensor
        Residuals on GPU.
    X_design : torch.Tensor
        Design matrix on GPU.
    df_resid : int
        Residual degrees of freedom.
    device : str, optional
        Torch device string.

    Returns
    -------
    fvalue : float
        F-statistic.
    pvalue : float
        p-value for F-statistic.
    """
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    from statgpu.inference._distributions_backend import get_distribution
    f_dist = get_distribution("f", backend="torch", device=device)

    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    ss_res = torch.sum(resid ** 2)
    ss_reg = ss_tot - ss_res

    k = X_design.shape[1] - 1  # exclude intercept

    if k == 0 or ss_res <= 0:
        return float('inf'), 1.0

    fvalue_tensor = (ss_reg / k) / (ss_res / df_resid)
    fvalue = float(fvalue_tensor.cpu().numpy())

    # p-value using F CDF
    # For F ~ F(d1, d2): CDF(x) = I_{d1*x/(d1*x+d2)}(d1/2, d2/2)
    d1 = float(k)
    d2 = float(df_resid)

    if d2 <= 0 or d1 <= 0:
        pvalue = 1.0
    else:
        z = (d1 * fvalue) / (d1 * fvalue + d2)
        cdf = f_dist.cdf(fvalue, dfn=d1, dfd=d2)
        pvalue = 1.0 - float(cdf.cpu().numpy())

    return fvalue, pvalue
'''

torch_new = '''def compute_f_stat_torch(y, resid, X_design, df_resid, device=None):
    """Compute the overall-regression F statistic and p-value on Torch."""
    torch = _import_torch()

    if device is None:
        device = _get_torch_device()

    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    ss_res = torch.sum(resid ** 2)

    k = int(X_design.shape[1] - 1)  # exclude intercept
    d1 = float(k)
    d2 = float(df_resid)
    if d1 <= 0.0 or d2 <= 0.0:
        return np.nan, np.nan

    # Only scalar reductions cross the host boundary.  These checks mirror the
    # public CPU LinearRegression F-statistic semantics.
    ss_tot_value = float(ss_tot.detach().cpu().item())
    ss_res_value = float(ss_res.detach().cpu().item())
    if not np.isfinite(ss_tot_value) or not np.isfinite(ss_res_value):
        return np.nan, np.nan

    tol = np.finfo(float).eps * max(1.0, abs(ss_tot_value))
    if ss_tot_value <= tol:
        return np.nan, np.nan
    if ss_res_value <= tol:
        return np.inf, 0.0

    ss_reg = torch.clamp(ss_tot - ss_res, min=0.0)
    fvalue_tensor = (ss_reg / d1) / (ss_res / d2)
    fvalue = float(fvalue_tensor.detach().cpu().item())

    from statgpu.inference._distributions_backend import get_distribution
    f_dist = get_distribution("f", backend="torch", device=device)
    cdf = f_dist.cdf(fvalue, dfn=d1, dfd=d2)
    pvalue = 1.0 - float(cdf.detach().cpu().item())
    return fvalue, float(np.clip(pvalue, 0.0, 1.0))
'''

replace_once("statgpu/backends/_gpu_inference_cupy.py", cupy_old, cupy_new)
replace_once("statgpu/backends/_gpu_inference_torch.py", torch_old, torch_new)

test_path = Path("dev/tests/test_pr79_final_review_fixes.py")
test_text = test_path.read_text()
marker = "def test_gpu_f_stat_degenerate_semantics"
if marker not in test_text:
    test_text += '''\n\n@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_gpu_f_stat_degenerate_semantics(backend):
    """GPU helpers must match public CPU semantics on degenerate F tests."""
    y_np = np.array([-1.0, 0.0, 1.0, 2.0])
    design_np = np.column_stack([np.ones(y_np.size), y_np])
    intercept_only_np = np.ones((y_np.size, 1))

    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA device unavailable")
        from statgpu.backends._gpu_inference_cupy import compute_f_stat_gpu

        y = cp.asarray(y_np)
        perfect_f, perfect_p = compute_f_stat_gpu(
            y, cp.zeros_like(y), cp.asarray(design_np), df_resid=2
        )
        null_f, null_p = compute_f_stat_gpu(
            y,
            y - y.mean(),
            cp.asarray(intercept_only_np),
            df_resid=3,
        )
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device unavailable")
        from statgpu.backends._gpu_inference_torch import compute_f_stat_torch

        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")
        perfect_f, perfect_p = compute_f_stat_torch(
            y,
            torch.zeros_like(y),
            torch.as_tensor(design_np, dtype=torch.float64, device="cuda"),
            df_resid=2,
            device="cuda",
        )
        null_f, null_p = compute_f_stat_torch(
            y,
            y - y.mean(),
            torch.as_tensor(intercept_only_np, dtype=torch.float64, device="cuda"),
            df_resid=3,
            device="cuda",
        )

    assert np.isposinf(perfect_f)
    assert perfect_p == 0.0
    assert np.isnan(null_f)
    assert np.isnan(null_p)
'''
    test_path.write_text(test_text)
