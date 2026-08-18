from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:180]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


p = Path("statgpu/panel/_diagnostics.py")
text = p.read_text(encoding="utf-8")
old = '''    D = np.asarray(_symmetrize(D), dtype=np.float64)\n    eigvals, eigvecs = np.linalg.eigh(D)\n    norm_D = float(np.linalg.norm(D, ord=2)) if D.size else 0.0\n    tol = _relative_tolerance(norm_D, factor=256.0 * max(1, d.size))\n    meta = {\n        "eigen_tolerance": tol,\n        "minimum_eigenvalue": float(eigvals.min()),\n        "maximum_eigenvalue": float(eigvals.max()),\n    }\n    if float(eigvals.min()) < -tol:\n'''
new = '''    D = np.asarray(_symmetrize(D), dtype=np.float64)\n    matrix_scale = float(np.max(np.abs(D))) if D.size else 0.0\n    if not np.isfinite(matrix_scale):\n        return _inapplicable(\n            null=null,\n            alternative=alternative,\n            distribution="chi2",\n            reason="covariance difference contains non-finite values",\n        )\n    if matrix_scale == 0.0:\n        D_work = D\n    else:\n        D_work = D / matrix_scale\n    eigvals_work, eigvecs = np.linalg.eigh(D_work)\n    norm_work = float(np.linalg.norm(D_work, ord=2)) if D_work.size else 0.0\n    tol_work = _relative_tolerance(\n        norm_work, factor=256.0 * max(1, d.size)\n    )\n\n    def _restore_linear(value: float) -> float:\n        value = float(value)\n        if value == 0.0 or matrix_scale == 0.0:\n            return 0.0\n        limit = float(np.finfo(np.float64).max) / abs(value)\n        if matrix_scale > limit:\n            return float(np.copysign(np.inf, value))\n        return float(value * matrix_scale)\n\n    meta = {\n        "eigen_scale": float(matrix_scale),\n        "eigen_tolerance": _restore_linear(tol_work),\n        "eigen_tolerance_normalized": float(tol_work),\n        "minimum_eigenvalue": _restore_linear(float(eigvals_work.min())),\n        "maximum_eigenvalue": _restore_linear(float(eigvals_work.max())),\n        "minimum_eigenvalue_normalized": float(eigvals_work.min()),\n        "maximum_eigenvalue_normalized": float(eigvals_work.max()),\n    }\n    if float(eigvals_work.min()) < -tol_work:\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("Hausman eigenscale anchor not found")
    text = text.replace(old, new, 1)
text = text.replace(
    '''    positive = eigvals > tol\n''',
    '''    positive = eigvals_work > tol_work\n''',
    1,
)
old_stat = '''    coordinates = basis.T @ d\n    standardized = coordinates / np.sqrt(eigvals[positive])\n    statistic = float(np.sum(standardized * standardized))\n    meta["quadratic_evaluation"] = "standardized_eigencoordinates"\n'''
new_stat = '''    coordinates = basis.T @ d\n    if matrix_scale == 0.0:\n        standardized = coordinates\n    else:\n        standardized = coordinates / np.sqrt(matrix_scale)\n        standardized = standardized / np.sqrt(eigvals_work[positive])\n    statistic = float(np.sum(standardized * standardized))\n    meta["quadratic_evaluation"] = "scaled_standardized_eigencoordinates"\n'''
if new_stat not in text:
    if old_stat not in text:
        raise RuntimeError("Hausman statistic anchor not found")
    text = text.replace(old_stat, new_stat, 1)
p.write_text(text.rstrip() + "\n", encoding="utf-8")

p = Path("dev/tests/test_panel_stage_b_hausman_covariance.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_hausman_quadratic_normalizes_dense_large_covariance_scale():
    result = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e154]),
        np.full((2, 2), 1.0e308, dtype=np.float64),
    )
    assert result.applicable, result.reason
    assert result.df == 1.0
    assert result.metadata["eigen_scale"] == 1.0e308
    assert np.isinf(result.metadata["maximum_eigenvalue"])
    assert_allclose(result.metadata["maximum_eigenvalue_normalized"], 2.0, rtol=0.0, atol=0.0)
    assert result.metadata["quadratic_evaluation"] == "scaled_standardized_eigencoordinates"
    assert_allclose(result.statistic, 1.0, rtol=5e-13, atol=0.0)
    assert np.isfinite(result.pvalue)
'''
if "test_hausman_quadratic_normalizes_dense_large_covariance_scale" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_stage_b_torch_cpu_hausman_dense_large_covariance_scale():
    result = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e154]),
        np.full((2, 2), 1.0e308, dtype=np.float64),
    )
    assert result.applicable, result.reason
    np.testing.assert_allclose(result.statistic, 1.0, rtol=5e-13, atol=0.0)
'''
if "test_stage_b_torch_cpu_hausman_dense_large_covariance_scale" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
old = '''    singular = _hausman_quadratic(\n        np.asarray([1.0e154, 1.0e200]),\n        np.diag(np.asarray([1.0e308, 0.0])),\n    )\n    if singular.applicable or not singular.reason or (\n        "outside the identified covariance-difference range" not in singular.reason\n    ):\n        raise AssertionError(\n            f"{backend}: large singular Hausman range guard failed: {singular}"\n        )\n    return {\n        "status": "success",\n        "backend": backend,\n        "cases": results,\n        "large_singular_range_rejected": True,\n    }\n'''
new = '''    singular = _hausman_quadratic(\n        np.asarray([1.0e154, 1.0e200]),\n        np.diag(np.asarray([1.0e308, 0.0])),\n    )\n    if singular.applicable or not singular.reason or (\n        "outside the identified covariance-difference range" not in singular.reason\n    ):\n        raise AssertionError(\n            f"{backend}: large singular Hausman range guard failed: {singular}"\n        )\n    dense = _hausman_quadratic(\n        np.asarray([1.0e154, 1.0e154]),\n        np.full((2, 2), 1.0e308, dtype=np.float64),\n    )\n    if not dense.applicable:\n        raise AssertionError(\n            f"{backend}: dense large Hausman scale became inapplicable: {dense.reason}"\n        )\n    np.testing.assert_allclose(\n        dense.statistic, 1.0, rtol=5e-13, atol=0.0,\n        err_msg=f"{backend}: dense large Hausman statistic",\n    )\n    return {\n        "status": "success",\n        "backend": backend,\n        "cases": results,\n        "large_singular_range_rejected": True,\n        "dense_large_statistic": float(dense.statistic),\n    }\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("Hausman physical audit extension anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text.rstrip() + "\n", encoding="utf-8")

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
old = '''    assert audit["large_singular_range_rejected"] is True\n    for label in ("large", "subnormal"):\n'''
new = '''    assert audit["large_singular_range_rejected"] is True\n    np.testing.assert_allclose(audit["dense_large_statistic"], 1.0, rtol=5e-13, atol=0.0)\n    for label in ("large", "subnormal"):\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("Hausman physical contract anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text.rstrip() + "\n", encoding="utf-8")

for path in (
    "statgpu/panel/_diagnostics.py",
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
):
    p = Path(path)
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
