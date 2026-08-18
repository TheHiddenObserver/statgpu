from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/panel/_diagnostics.py",
    """    D = 0.5 * (D + D.T)\n    eigvals, eigvecs = np.linalg.eigh(D)\n""",
    """    # Reuse the covariance layer's range-safe symmetric average.  The\n    # mathematical input is a covariance difference, so doubling two finite\n    # same-sign entries near DBL_MAX must not turn a representable matrix into\n    # Inf before the eigendecomposition.\n    from statgpu.panel._covariance import _symmetrize\n\n    D = np.asarray(_symmetrize(D), dtype=np.float64)\n    eigvals, eigvecs = np.linalg.eigh(D)\n""",
)

replace_once(
    "statgpu/panel/_diagnostics.py",
    """    inv_eigs = 1.0 / eigvals[positive]\n    statistic = float((basis.T @ d).T @ (inv_eigs * (basis.T @ d)))\n""",
    """    # Evaluate d' D+ d in standardized eigencoordinates instead of\n    # materializing 1/lambda.  For a positive subnormal eigenvalue, 1/lambda\n    # can overflow even when (u'd)^2/lambda is perfectly representable.\n    coordinates = basis.T @ d\n    standardized = coordinates / np.sqrt(eigvals[positive])\n    statistic = float(np.sum(standardized * standardized))\n    meta[\"quadratic_evaluation\"] = \"standardized_eigencoordinates\"\n""",
)

replace_once(
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
    "from statgpu.panel import PanelOLS, RandomEffects\n",
    "from statgpu.panel import PanelOLS, RandomEffects\nfrom statgpu.panel._diagnostics import _hausman_quadratic\n",
)

p = Path("dev/tests/test_panel_stage_b_hausman_covariance.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_hausman_quadratic_is_scale_safe_at_float64_extremes():
    cases = (
        (1.0e308, np.sqrt(1.0e308)),
        (1.0e-320, np.sqrt(1.0e-320)),
    )
    results = []
    for variance, difference in cases:
        result = _hausman_quadratic(
            np.asarray([difference], dtype=np.float64),
            np.asarray([[variance]], dtype=np.float64),
        )
        assert result.applicable, result.reason
        assert result.df == 1.0
        assert result.metadata["quadratic_evaluation"] == "standardized_eigencoordinates"
        assert_allclose(result.statistic, 1.0, rtol=3e-12, atol=0.0)
        assert np.isfinite(result.pvalue)
        results.append(result)

    assert_allclose(results[0].pvalue, results[1].pvalue, rtol=3e-12, atol=0.0)
'''
if "test_hausman_quadratic_is_scale_safe_at_float64_extremes" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "from statgpu.panel._diagnostics import _diagnostic_identity, _fingerprints_match\n",
    "from statgpu.panel._diagnostics import (\n    _diagnostic_identity,\n    _fingerprints_match,\n    _hausman_quadratic,\n)\n",
)

p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_stage_b_torch_cpu_hausman_host_quadratic_is_scale_safe():
    # Hausman forms the small covariance-difference quadratic on host after the
    # backend-specific FE/RE fits.  Keep both floating-point scale extremes in
    # the maintained Torch job so GPU-originated results cannot regress here.
    for variance, difference in (
        (1.0e308, np.sqrt(1.0e308)),
        (1.0e-320, np.sqrt(1.0e-320)),
    ):
        result = _hausman_quadratic([difference], [[variance]])
        assert result.applicable, result.reason
        assert_allclose(result.statistic, 1.0, rtol=3e-12, atol=0.0)
'''
if "test_stage_b_torch_cpu_hausman_host_quadratic_is_scale_safe" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    """from statgpu.panel._diagnostics import (\n    _classical_model_f,\n    _scaled_group_means,\n    _scaled_mean,\n)\n""",
    """from statgpu.panel._diagnostics import (\n    _classical_model_f,\n    _hausman_quadratic,\n    _scaled_group_means,\n    _scaled_mean,\n)\n""",
)

replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "\ndef _zero_variance_inference_audit(backend):\n",
    r'''

def _hausman_scale_audit(backend):
    results = {}
    for label, variance, difference in (
        ("large", 1.0e308, np.sqrt(1.0e308)),
        ("subnormal", 1.0e-320, np.sqrt(1.0e-320)),
    ):
        result = _hausman_quadratic([difference], [[variance]])
        if not result.applicable:
            raise AssertionError(
                f"{backend}: Hausman {label} scale became inapplicable: {result.reason}"
            )
        np.testing.assert_allclose(
            result.statistic, 1.0, rtol=3e-12, atol=0.0,
            err_msg=f"{backend}: Hausman {label} scale statistic",
        )
        if not np.isfinite(result.pvalue):
            raise AssertionError(f"{backend}: Hausman {label} p-value is non-finite")
        results[label] = {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "df": float(result.df),
        }
    np.testing.assert_allclose(
        results["large"]["pvalue"], results["subnormal"]["pvalue"],
        rtol=3e-12, atol=0.0,
    )
    return {"status": "success", "backend": backend, "cases": results}


def _zero_variance_inference_audit(backend):
''',
)

replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    """            \"covariance_extreme_scale\": _covariance_extreme_scale_audit(backend),\n            \"zero_variance_inference\": _zero_variance_inference_audit(backend),\n""",
    """            \"covariance_extreme_scale\": _covariance_extreme_scale_audit(backend),\n            \"hausman_scale\": _hausman_scale_audit(backend),\n            \"zero_variance_inference\": _zero_variance_inference_audit(backend),\n""",
)

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_stage_c_runner_hausman_scale_audit_is_executable():
    audit = _MOD._hausman_scale_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    for label in ("large", "subnormal"):
        case = audit["cases"][label]
        assert case["df"] == 1.0
        assert np.isfinite(case["statistic"])
        assert np.isfinite(case["pvalue"])
        np.testing.assert_allclose(case["statistic"], 1.0, rtol=3e-12, atol=0.0)
'''
if "test_stage_c_runner_hausman_scale_audit_is_executable" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

# Keep staged test files compliant with git's whitespace checker: exactly one
# newline at EOF, no additional blank line introduced by the append blocks.
for path in (
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
):
    p = Path(path)
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
