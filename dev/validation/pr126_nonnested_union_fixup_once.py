from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/panel/_covariance.py",
    '''def _cross_reduction_is_safe(left, right, xp) -> bool:\n    """Return whether ``left.T @ right`` has a conservative float64 range bound.\n\n    This is deliberately only a sufficient condition.  It is used to select a\n    cancellation-preserving two-way-cluster shortcut; if the bound is not met,\n    callers retain the existing scaled covariance path.\n    """\n    n_terms = max(1, int(left.shape[0]))\n    left_max = _to_float_scalar(xp.max(xp.abs(left)))\n    right_max = _to_float_scalar(xp.max(xp.abs(right)))\n    if not np.isfinite(left_max) or not np.isfinite(right_max):\n        return False\n    if left_max == 0.0 or right_max == 0.0:\n        return True\n    # Keep a factor-four range margin, matching the Gram working-scale policy.\n    per_term_limit = 0.25 * float(np.finfo(np.float64).max) / float(n_terms)\n    return bool(left_max <= per_term_limit / right_max)\n''',
    '''def _cross_reduction_is_safe(left, right, xp) -> bool:\n    """Return whether ``left.T @ right`` has a conservative rowwise range bound.\n\n    Matrix-product terms pair values from the same observation.  Bounding the\n    product of unrelated column maxima can therefore reject a safe structural\n    cancellation (for example, a huge left score where the corresponding right\n    score is exactly zero).  Use per-row infinity norms instead.  This remains\n    only a sufficient condition: any uncertain case stays on the scaled\n    component path.\n    """\n    n_terms = max(1, int(left.shape[0]))\n    if _is_torch(xp):\n        left_row_max = xp.max(xp.abs(left), dim=1).values\n        right_row_max = xp.max(xp.abs(right), dim=1).values\n    else:\n        left_row_max = xp.max(xp.abs(left), axis=1)\n        right_row_max = xp.max(xp.abs(right), axis=1)\n\n    finite = xp.isfinite(left_row_max) & xp.isfinite(right_row_max)\n    large = xp.maximum(left_row_max, right_row_max)\n    small = xp.minimum(left_row_max, right_row_max)\n    one = xp.ones_like(large)\n    safe_large = xp.maximum(large, one)\n    per_term_limit = 0.25 * float(np.finfo(np.float64).max) / float(n_terms)\n    safe = finite & (small <= float(per_term_limit) / safe_large)\n    return bool(_to_float_scalar(xp.all(safe)))\n''',
)

replace_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    '''    n = 6\n    amplitude = 2.0e307\n    small = 1.0e-150\n    X = np.ones((n, 1), dtype=np.float64)\n    target_scores = np.asarray(\n        [-amplitude, 0.0, amplitude, 0.0, small, -small], dtype=np.float64\n    )\n    resid = float(n) * target_scores\n    cluster1 = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)\n    cluster2 = np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int64)\n\n    actual = two_way_clustered_covariance(X, resid, cluster1, cluster2)\n    expected = np.asarray([[2.0 * small * small]], dtype=np.float64)\n''',
    '''    n = 4\n    amplitude = 1.0e154\n    small = 1.0e-154\n    # ||X||_2 == 1 exactly, so the one-column SVD pseudoinverse is exactly 0.5\n    # in float64.  This isolates the covariance-combination issue from upstream\n    # least-squares roundoff.\n    X = np.full((n, 1), 0.5, dtype=np.float64)\n    target_scores = np.asarray(\n        [-amplitude, small, amplitude, -small], dtype=np.float64\n    )\n    resid = 2.0 * target_scores\n    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)\n\n    actual = two_way_clustered_covariance(X, resid, cluster1, cluster2)\n    expected = np.asarray([[-4.0 * amplitude * small]], dtype=np.float64)\n''',
)

replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''    n = 6\n    amplitude = 2.0e307\n    small = 1.0e-150\n    X = torch.ones((n, 1), dtype=torch.float64)\n    target_scores = torch.tensor(\n        [-amplitude, 0.0, amplitude, 0.0, small, -small], dtype=torch.float64\n    )\n    resid = float(n) * target_scores\n    cluster1 = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)\n    cluster2 = np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int64)\n''',
    '''    n = 4\n    amplitude = 1.0e154\n    small = 1.0e-154\n    X = torch.full((n, 1), 0.5, dtype=torch.float64)\n    target_scores = torch.tensor(\n        [-amplitude, small, amplitude, -small], dtype=torch.float64\n    )\n    resid = 2.0 * target_scores\n    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)\n''',
)
replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''        actual, np.asarray([[2.0 * small * small]]), rtol=2e-12, atol=0.0\n''',
    '''        actual, np.asarray([[-4.0 * amplitude * small]]), rtol=2e-12, atol=0.0\n''',
)

replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    nonnested_n = 6\n    nonnested_amplitude = 2.0e307\n    nonnested_small = 1.0e-150\n    X_nonnested_np = np.ones((nonnested_n, 1), dtype=np.float64)\n    nonnested_scores = np.asarray(\n        [-nonnested_amplitude, 0.0, nonnested_amplitude, 0.0,\n         nonnested_small, -nonnested_small],\n        dtype=np.float64,\n    )\n    resid_nonnested_np = float(nonnested_n) * nonnested_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)\n    nonnested_cluster2 = np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int64)\n''',
    '''    nonnested_n = 4\n    nonnested_amplitude = 1.0e154\n    nonnested_small = 1.0e-154\n    X_nonnested_np = np.full((nonnested_n, 1), 0.5, dtype=np.float64)\n    nonnested_scores = np.asarray(\n        [-nonnested_amplitude, nonnested_small,\n         nonnested_amplitude, -nonnested_small],\n        dtype=np.float64,\n    )\n    resid_nonnested_np = 2.0 * nonnested_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    nonnested_cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        np.asarray([[2.0 * nonnested_small * nonnested_small]]),\n''',
    '''        np.asarray([[-4.0 * nonnested_amplitude * nonnested_small]]),\n''',
)

for path in (
    "dev/tests/test_panel_stage_c_covariance.py",
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
):
    p = Path(path)
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
