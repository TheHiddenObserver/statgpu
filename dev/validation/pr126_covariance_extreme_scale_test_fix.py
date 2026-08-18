from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Test the group reducer at its actual contract boundary rather than mixing in
# SVD row-roundoff from a catastrophically scaled constant design.
replace_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    '''    _dk_kernel_weights,\n    _symmetrize,\n    clustered_covariance,\n''',
    '''    _dk_kernel_weights,\n    _grouped_score_sums,\n    _symmetrize,\n    clustered_covariance,\n''',
    "covariance test grouped helper import",
)
replace_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    '''def test_cluster_group_reduction_survives_finite_partial_sum_overflow():\n    X = np.ones((6, 1), dtype=np.float64) * 1.0e-154\n    resid = np.asarray([1.0e155, 1.0e155, -1.0e155, -1.0e155, 1.0, -1.0])\n    groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n    cov = clustered_covariance(X, resid, groups)\n    assert np.all(np.isfinite(cov))\n    np.testing.assert_allclose(cov, np.zeros((1, 1)), rtol=0.0, atol=0.0)\n\n\n''',
    '''def test_grouped_score_reduction_survives_finite_partial_sum_overflow():\n    amplitude = 1.6e308\n    scores = np.asarray(\n        [[amplitude], [amplitude], [-amplitude], [-amplitude], [1.0], [-1.0]],\n        dtype=np.float64,\n    )\n    groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n    grouped = _grouped_score_sums(scores, groups, n_groups=2, xp=np)\n    assert np.all(np.isfinite(grouped))\n    np.testing.assert_array_equal(grouped, np.zeros((2, 1)))\n\n\n''',
    "covariance grouped reduction regression",
)

replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''from statgpu.panel._covariance import (\n    clustered_covariance,\n    driscoll_kraay_covariance,\n    hac_covariance,\n    two_way_clustered_covariance,\n)\n''',
    '''from statgpu.panel._covariance import (\n    _grouped_score_sums,\n    clustered_covariance,\n    driscoll_kraay_covariance,\n    hac_covariance,\n    two_way_clustered_covariance,\n)\n''',
    "torch grouped helper import",
)
replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''    X_tiny = torch.ones((6, 1), dtype=torch.float64) * 1.0e-154\n    resid_cancel = torch.tensor(\n        [1.0e155, 1.0e155, -1.0e155, -1.0e155, 1.0, -1.0],\n        dtype=torch.float64,\n    )\n    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n    cancel_cov = clustered_covariance(X_tiny, resid_cancel, cancel_groups, xp=torch)\n    assert_allclose(cancel_cov, np.zeros((1, 1)), rtol=0.0, atol=0.0)\n\n''',
    '''    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n    score_amplitude = 1.6e308\n    scores = torch.tensor(\n        [[score_amplitude], [score_amplitude], [-score_amplitude], [-score_amplitude], [1.0], [-1.0]],\n        dtype=torch.float64,\n    )\n    grouped = _grouped_score_sums(scores, cancel_groups, n_groups=2, xp=torch)\n    assert_allclose(grouped, np.zeros((2, 1)), rtol=0.0, atol=0.0)\n\n''',
    "torch grouped reducer regression",
)

replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "from statgpu.panel._covariance import hac_covariance, ols_covariance, two_way_clustered_covariance\n",
    "from statgpu.panel._covariance import _grouped_score_sums, hac_covariance, ols_covariance, two_way_clustered_covariance\n",
    "physical grouped helper import",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    X_cancel_np = np.ones((6, 1), dtype=np.float64) * 1.0e-154\n    resid_cancel_np = np.asarray([1.0e155, 1.0e155, -1.0e155, -1.0e155, 1.0, -1.0])\n    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n''',
    '''    score_amplitude = 1.6e308\n    scores_np = np.asarray(\n        [[score_amplitude], [score_amplitude], [-score_amplitude], [-score_amplitude], [1.0], [-1.0]],\n        dtype=np.float64,\n    )\n    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n''',
    "physical grouped synthetic fixture",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X, resid = X_np, resid_np\n        X_cancel, resid_cancel = X_cancel_np, resid_cancel_np\n        X_hac, resid_hac = X_hac_np, resid_hac_np\n''',
    '''        X, resid = X_np, resid_np\n        scores = scores_np\n        X_hac, resid_hac = X_hac_np, resid_hac_np\n''',
    "physical numpy grouped fixture",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X, resid = cp.asarray(X_np), cp.asarray(resid_np)\n        X_cancel, resid_cancel = cp.asarray(X_cancel_np), cp.asarray(resid_cancel_np)\n        X_hac, resid_hac = cp.asarray(X_hac_np), cp.asarray(resid_hac_np)\n''',
    '''        X, resid = cp.asarray(X_np), cp.asarray(resid_np)\n        scores = cp.asarray(scores_np)\n        X_hac, resid_hac = cp.asarray(X_hac_np), cp.asarray(resid_hac_np)\n''',
    "physical cupy grouped fixture",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")\n        resid = torch.as_tensor(resid_np, dtype=torch.float64, device="cuda")\n        X_cancel = torch.as_tensor(X_cancel_np, dtype=torch.float64, device="cuda")\n        resid_cancel = torch.as_tensor(resid_cancel_np, dtype=torch.float64, device="cuda")\n        X_hac = torch.as_tensor(X_hac_np, dtype=torch.float64, device="cuda")\n''',
    '''        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")\n        resid = torch.as_tensor(resid_np, dtype=torch.float64, device="cuda")\n        scores = torch.as_tensor(scores_np, dtype=torch.float64, device="cuda")\n        X_hac = torch.as_tensor(X_hac_np, dtype=torch.float64, device="cuda")\n''',
    "physical torch grouped fixture",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    cancellation = _array(clustered_covariance(X_cancel, resid_cancel, cancel_groups, xp=xp))\n''',
    '''    cancellation = _array(\n        _grouped_score_sums(scores, cancel_groups, n_groups=2, xp=xp)\n    )\n''',
    "physical grouped reduction execution",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    np.testing.assert_allclose(cancellation, np.zeros((1, 1)), rtol=0.0, atol=0.0)\n''',
    '''    np.testing.assert_array_equal(cancellation, np.zeros((2, 1)))\n''',
    "physical grouped reduction expectation",
)
