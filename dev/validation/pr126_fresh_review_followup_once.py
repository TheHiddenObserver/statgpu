from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---- Physical GPU runner: exercise the newly hardened diagnostic reductions on
# actual CuPy/Torch CUDA and keep tiny-design rank policy aligned with the solver.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    """from statgpu.panel._covariance import ols_covariance\nfrom statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean\nfrom statgpu.panel._linalg import (\n    panel_lstsq,\n    panel_lstsq_batched,\n    panel_lstsq_gram_certified_batched,\n)\n""",
    """from statgpu.panel._covariance import ols_covariance\nfrom statgpu.panel._diagnostic_context import (\n    bp_lm_from_residuals,\n    pooling_f_from_level_arrays,\n)\nfrom statgpu.panel._diagnostics import (\n    _classical_model_f,\n    _scaled_group_means,\n    _scaled_mean,\n)\nfrom statgpu.panel._linalg import (\n    panel_lstsq,\n    panel_lstsq_batched,\n    panel_lstsq_gram_certified_batched,\n    panel_matrix_rank,\n)\n""",
    "physical diagnostic imports",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    """def _tiny_design_lstsq_audit(backend):\n    tiny = 1.0e-320\n""",
    """def _diagnostic_scale_audit(backend):\n    if backend == \"numpy\":\n        xp = np\n    elif backend == \"cupy\":\n        import cupy as cp\n        xp = cp\n    elif backend == \"torch\":\n        import torch\n        xp = torch\n    else:\n        raise ValueError(backend)\n\n    # Pooling F: naive column/scalar means overflow after this common scaling,\n    # while the centered regression and scale-invariant statistic are finite.\n    n = 24\n    t = np.linspace(-1.0, 1.0, n)\n    X = np.column_stack(\n        [1.15 + 0.18 * t, 0.95 - 0.11 * t + 0.03 * t * t]\n    ).astype(np.float64)\n    y = (\n        1.05\n        + 0.42 * t\n        - 0.08 * t * t\n        + 0.025 * np.sin(np.arange(n))\n    ).astype(np.float64)\n    Xc = X - X.mean(axis=0)\n    yc = y - y.mean()\n    beta, _ = panel_lstsq(Xc, yc, np)\n    effects_resid = 0.55 * (yc - Xc @ beta)\n    dummy = np.arange(n, dtype=np.int64)\n\n    def _pooling(scale):\n        Xb, yb, _eb, _tb = _to_backend(\n            scale * X, scale * y, dummy, dummy, backend\n        )\n        _X2, effects_b, _e2, _t2 = _to_backend(\n            scale * X, scale * effects_resid, dummy, dummy, backend\n        )\n        return pooling_f_from_level_arrays(\n            yb,\n            Xb,\n            xp=xp,\n            rss_effects=0.0,\n            df_resid_effects=n - 6,\n            has_constant=False,\n            resid_effects=effects_b,\n        )\n\n    pooling_reference = _pooling(1.0)\n    pooling_large = _pooling(1.0e307)\n    if not pooling_reference.applicable or not pooling_large.applicable:\n        raise AssertionError(f\"{backend}: pooling-F scale audit became inapplicable\")\n    np.testing.assert_allclose(\n        pooling_large.statistic, pooling_reference.statistic,\n        rtol=5e-8, atol=1e-10,\n    )\n    np.testing.assert_allclose(\n        pooling_large.pvalue, pooling_reference.pvalue,\n        rtol=5e-8, atol=1e-12,\n    )\n\n    # Classical model F: use a subnormal response scale so direct backend scalar\n    # division is exercised. The statistic must remain invariant to response units.\n    x = np.linspace(-1.0, 1.0, 12)\n    Xf = np.column_stack([np.ones(x.size), x]).astype(np.float64)\n    yf = (\n        0.8\n        + 0.45 * x\n        + np.asarray(\n            [0.08, -0.04, 0.03, -0.06, 0.05, -0.02,\n             0.01, 0.04, -0.03, 0.02, -0.01, 0.05],\n            dtype=np.float64,\n        )\n    )\n    entity_f = np.arange(x.size, dtype=np.int64)\n\n    def _model_f(scale):\n        Xb, yb, _eb, _tb = _to_backend(\n            Xf, scale * yf, entity_f, entity_f, backend\n        )\n        params, rank = panel_lstsq(Xb, yb, xp)\n        result = _classical_model_f(\n            yb, Xb, params, xp=xp,\n            df_resid=x.size - int(rank), has_constant=True,\n        )\n        if result[0] is None or not np.isfinite(result[0]):\n            raise AssertionError(f\"{backend}: classical-F scale audit is not finite\")\n        return result\n\n    model_f_reference = _model_f(1.0)\n    model_f_tiny = _model_f(1.0e-310)\n    np.testing.assert_allclose(\n        model_f_tiny[0], model_f_reference[0], rtol=5e-4, atol=5e-6\n    )\n    np.testing.assert_allclose(\n        model_f_tiny[1], model_f_reference[1], rtol=5e-4, atol=5e-8\n    )\n\n    # Baltagi-Li BP-LM is also response-scale invariant and uses grouped backend\n    # reductions, so verify the subnormal path on the requested physical backend.\n    groups = np.repeat(np.arange(5), 4).astype(np.int64)\n    pattern = np.asarray(\n        [1.0, -0.4, 0.6, -0.3, 0.8, -0.2, 0.5, -0.7, 1.1, -0.5,\n         0.3, -0.1, 0.7, -0.6, 0.4, -0.2, 0.9, -0.3, 0.2, -0.4],\n        dtype=np.float64,\n    )\n    dummy_x = np.arange(pattern.size, dtype=np.float64)[:, None]\n    dummy_time = np.arange(pattern.size, dtype=np.int64)\n\n    def _bp(scale):\n        _xb, resid_b, groups_b, _tb = _to_backend(\n            dummy_x, scale * pattern, groups, dummy_time, backend\n        )\n        result = bp_lm_from_residuals(resid_b, groups_b, xp=xp)\n        if not result.applicable or not np.isfinite(result.statistic):\n            raise AssertionError(f\"{backend}: BP-LM scale audit is not finite/applicable\")\n        return result\n\n    bp_reference = _bp(1.0)\n    bp_tiny = _bp(1.0e-310)\n    np.testing.assert_allclose(\n        bp_tiny.statistic, bp_reference.statistic, rtol=5e-8, atol=1e-10\n    )\n    np.testing.assert_allclose(\n        bp_tiny.pvalue, bp_reference.pvalue, rtol=5e-8, atol=1e-12\n    )\n\n    return {\n        \"status\": \"success\",\n        \"backend\": backend,\n        \"pooling_f_statistic\": float(pooling_large.statistic),\n        \"pooling_f_pvalue\": float(pooling_large.pvalue),\n        \"classical_f_statistic\": float(model_f_tiny[0]),\n        \"classical_f_pvalue\": float(model_f_tiny[1]),\n        \"bp_lm_statistic\": float(bp_tiny.statistic),\n        \"bp_lm_pvalue\": float(bp_tiny.pvalue),\n    }\n\n\ndef _tiny_design_lstsq_audit(backend):\n    tiny = 1.0e-320\n""",
    "physical diagnostic scale audit",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    """    if rank != 2:\n        raise AssertionError(f\"{backend}: tiny full-rank design rank drifted to {rank}\")\n    np.testing.assert_allclose(params_np, np.asarray([1.0, 2.0]), rtol=5e-11, atol=0.0)\n    return {\"status\": \"success\", \"backend\": backend, \"rank\": rank, \"params\": params_np.tolist()}\n""",
    """    matrix_rank = int(panel_matrix_rank(Xb, torch if backend == \"torch\" else cp))\n    if rank != 2 or matrix_rank != rank:\n        raise AssertionError(\n            f\"{backend}: tiny-design rank policy drifted: solver={rank}, matrix_rank={matrix_rank}\"\n        )\n    np.testing.assert_allclose(params_np, np.asarray([1.0, 2.0]), rtol=5e-11, atol=0.0)\n    return {\n        \"status\": \"success\", \"backend\": backend,\n        \"rank\": rank, \"matrix_rank\": matrix_rank, \"params\": params_np.tolist(),\n    }\n""",
    "physical tiny-design rank consistency",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    """        payload[\"numerical_primitives\"] = {\n            \"tiny_design_lstsq\": _tiny_design_lstsq_audit(backend),\n            \"gram_overflow_certificate\": _gram_overflow_certificate_audit(backend),\n            \"cancellation_safe_mean\": _cancellation_safe_mean_audit(backend),\n        }\n""",
    """        diagnostic_scale = _diagnostic_scale_audit(backend)\n        diagnostic_reference = _diagnostic_scale_audit(\"numpy\")\n        diagnostic_diffs = {}\n        for field in (\n            \"pooling_f_statistic\", \"pooling_f_pvalue\",\n            \"classical_f_statistic\", \"classical_f_pvalue\",\n            \"bp_lm_statistic\", \"bp_lm_pvalue\",\n        ):\n            tolerance = 5e-4 if field.startswith(\"classical_f\") else max(args.rtol, 5e-8)\n            np.testing.assert_allclose(\n                diagnostic_scale[field], diagnostic_reference[field],\n                rtol=tolerance, atol=max(args.atol, 5e-8),\n                err_msg=f\"diagnostic_scale.{field}\",\n            )\n            diagnostic_diffs[field] = abs(\n                float(diagnostic_scale[field]) - float(diagnostic_reference[field])\n            )\n        diagnostic_scale[\"max_abs_differences_vs_numpy\"] = diagnostic_diffs\n        payload[\"numerical_primitives\"] = {\n            \"tiny_design_lstsq\": _tiny_design_lstsq_audit(backend),\n            \"gram_overflow_certificate\": _gram_overflow_certificate_audit(backend),\n            \"cancellation_safe_mean\": _cancellation_safe_mean_audit(backend),\n            \"diagnostic_scale_reductions\": diagnostic_scale,\n        }\n""",
    "physical diagnostic payload",
)

# Hosted runner contract executes the NumPy reference for the new physical case.
physical_test = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = physical_test.read_text(encoding="utf-8")
marker = "def test_stage_c_runner_diagnostic_scale_audit_is_executable():"
if marker in text:
    raise RuntimeError("physical diagnostic contract already present")
text += r'''


def test_stage_c_runner_diagnostic_scale_audit_is_executable():
    audit = _MOD._diagnostic_scale_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    for field in (
        "pooling_f_statistic", "pooling_f_pvalue",
        "classical_f_statistic", "classical_f_pvalue",
        "bp_lm_statistic", "bp_lm_pvalue",
    ):
        assert np.isfinite(audit[field]), field
'''
physical_test.write_text(text, encoding="utf-8")

# Hosted Torch CPU coverage for panel_matrix_rank must agree with the same tiny
# working-scale solve policy used by panel_lstsq.
final_test = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = final_test.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel._linalg import panel_lstsq\n",
    "from statgpu.panel._linalg import panel_lstsq, panel_matrix_rank\n",
    1,
)
marker = "def test_torch_tiny_design_matrix_rank_matches_shared_solver():"
if marker in text:
    raise RuntimeError("tiny matrix-rank regression already present")
text += r'''


def test_torch_tiny_design_matrix_rank_matches_shared_solver():
    torch = pytest.importorskip("torch")
    tiny = 1.0e-320
    X = torch.eye(2, dtype=torch.float64) * tiny
    y = torch.tensor([tiny, 2.0 * tiny], dtype=torch.float64)
    _params, solver_rank = panel_lstsq(X, y, torch)
    direct_rank = panel_matrix_rank(X, torch)
    assert solver_rank == 2
    assert direct_rank == solver_rank
'''
final_test.write_text(text, encoding="utf-8")

# ---- Documentation and durable evidence semantics.
replace_once(
    "docs/en/panel/diagnostics.md",
    "> Last updated: 2026-08-15  \n",
    "> Last updated: 2026-08-18  \n",
    "EN diagnostics date",
)
replace_once(
    "docs/en/panel/diagnostics.md",
    """`PanelTestResult` reports `statistic`, `pvalue`, the reference distribution, degrees of freedom, null and alternative text, and an `applicable` flag. When a test cannot be computed under its documented definition, inspect `reason` to see why; statgpu does not return a different test under the same method name.\n""",
    """`PanelTestResult` reports `statistic`, `pvalue`, the reference distribution, degrees of freedom, null and alternative text, and an `applicable` flag. When a test cannot be computed under its documented definition, inspect `reason` to see why; statgpu does not return a different test under the same method name.\n\nFor finite extreme-scale inputs, classical model F, pooling F, and Breusch-Pagan LM evaluate their scale-invariant quadratic reductions on backend-native normalized working values. Scalar and column centering use overflow-safe reduction-length scaling, while subnormal normalization avoids backend-specific division by a subnormal denominator. Public RSS metadata is restored to the original squared units when representable (and may be `inf` only when that squared quantity itself is outside float64); the test statistic is not allowed to become `0`, `NaN`, or `inf` merely because an avoidable intermediate overflowed or underflowed.\n""",
    "EN diagnostics numerical behavior",
)
replace_once(
    "docs/cn/panel/diagnostics.md",
    "> 最后更新：2026-08-15  \n",
    "> 最后更新：2026-08-18  \n",
    "CN diagnostics date",
)
replace_once(
    "docs/cn/panel/diagnostics.md",
    """`PanelTestResult` 提供 `statistic`、`pvalue`、reference distribution、degrees of freedom、null/alternative text 与 `applicable` flag。若某个检验在文档定义下无法计算，可以查看 `reason` 了解具体原因；statgpu 不会用同一个 method name 返回另一种 test。\n""",
    """`PanelTestResult` 提供 `statistic`、`pvalue`、reference distribution、degrees of freedom、null/alternative text 与 `applicable` flag。若某个检验在文档定义下无法计算，可以查看 `reason` 了解具体原因；statgpu 不会用同一个 method name 返回另一种 test。\n\n对于 finite extreme-scale inputs，classical model F、pooling F 与 Breusch-Pagan LM 会在当前 backend 上用归一化 working values 计算其 scale-invariant quadratic reductions。scalar/column centering 只在 reduction 可能 overflow 时按 reduction length 做缩放；subnormal normalization 也不会直接除以 subnormal denominator。公开的 RSS metadata 会在原始平方尺度可表示时恢复到该尺度；只有真实平方量超出 float64 表示范围时才允许为 `inf`。检验 statistic 不应仅因为可避免的中间 overflow/underflow 而错误变成 `0`、`NaN` 或 `inf`。\n""",
    "CN diagnostics numerical behavior",
)
replace_once(
    "CHANGELOG.md",
    """- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth averages and parameter-R² scalar/group means use only reduction-length scaling when overflow is possible, preserving representable cancellation remainders; coefficient-series covariance uses per-coordinate scales with symmetric large-scale-first restoration. Genuinely unrepresentable covariance still fails closed, while exact-zero variance avoids `0/0` inference NaNs.\n""",
    """- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth averages and parameter-R² scalar/group means use only reduction-length scaling when overflow is possible, preserving representable cancellation remainders; coefficient-series covariance uses per-coordinate scales with symmetric large-scale-first restoration. Genuinely unrepresentable covariance still fails closed, while exact-zero variance avoids `0/0` inference NaNs.\n- **Panel diagnostics extreme-scale correctness**: classical model F, pooling F, Breusch-Pagan LM, adjusted/legacy fit-statistic reductions, and estimator-side RSS/TSS reporting now share overflow-safe centering plus subnormal-safe backend normalization. Scale-invariant statistics are computed before restoring squared units, so representable finite results are not converted to false exact fits or `Inf/Inf`/underflow artifacts; the physical Stage-C runner now exercises these branches on both CuPy and Torch CUDA.\n""",
    "root changelog diagnostic stability",
)

review = Path("dev/reviews/pr126_final_review_2026-08-17.md")
text = review.read_text(encoding="utf-8")
text = text.replace(
    "# PR #126 final technical review and physical acceptance record\n\nDate: 2026-08-17\n",
    "# PR #126 historical physical acceptance snapshot\n\nDate: 2026-08-17\n\n> **Historical evidence only (status updated 2026-08-18).** Subsequent PR #126 review/fix loops changed valid Fama-MacBeth, shared panel least-squares, covariance, and diagnostic numerical paths after the source recorded below. This snapshot must not be used as current-head merge acceptance. Fresh exact-head CuPy and Torch CUDA revalidation is required and remains pending until new physical artifacts are produced.\n",
    1,
)
text = text.replace("## Validated numerical source\n", "## Historical validated numerical source\n", 1)
text = text.replace(
    "The final production numerical source validated on physical GPU is:\n",
    "The historical numerical source validated on physical GPU in this snapshot is:\n",
    1,
)
text = text.replace(
    "Production numerical code remained frozen after this source during final\nphysical-evidence promotion, documentation reconciliation, artifact cleanup,\nand deterministic benchmark-data regeneration.\n",
    "At the time of this snapshot, production numerical code remained frozen after this source during its physical-evidence promotion, documentation reconciliation, artifact cleanup, and deterministic benchmark-data regeneration. Later review/fix loops intentionally changed that numerical code, so the evidence below is historical rather than current-head acceptance.\n",
    1,
)
text = text.replace(
    "## Final Fama-MacBeth solver contract\n\nNumPy remains the serial rank-revealing SVD statistical reference. GPU retained\n",
    "## Historical Fama-MacBeth solver contract at `8c60db00...`\n\nAt this historical source, NumPy remained the serial rank-revealing SVD statistical reference. GPU retained\n",
    1,
)
text = text.replace(
    "## Final Tesla P100 physical acceptance\n",
    "## Historical Tesla P100 physical acceptance\n",
    1,
)
review.write_text(text, encoding="utf-8")

print("PR126 fresh review evidence follow-up staged")
