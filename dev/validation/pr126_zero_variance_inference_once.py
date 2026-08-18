from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_before(path, anchor, addition, label):
    replace_once(path, anchor, addition + anchor, label)


# Shared exact-zero statistic ratio. A positive absolute variance floor is not
# scale equivariant; exact zero needs an explicit statistical convention instead.
append_before(
    "statgpu/panel/_utils.py",
    "def ols_inference_nonrobust(params, X, scale, df, alpha=0.05):\n",
    '''def _zero_safe_statistic_ratio(params, bse, xp):\n    """Return parameter statistics with explicit exact-zero variance semantics.\n\n    Positive standard errors are used unchanged. At exactly zero standard error,\n    a zero coefficient maps to statistic 0 while a nonzero coefficient maps to\n    signed infinity. This avoids both a dimensionful variance floor and 0/0 NaN.\n    """\n    zero = bse == 0.0\n    denominator = xp.where(zero, xp.ones_like(bse), bse)\n    statistic = params / denominator\n    positive_inf = xp.full_like(statistic, float("inf"))\n    negative_inf = xp.full_like(statistic, float("-inf"))\n    statistic = xp.where(zero & (params > 0.0), positive_inf, statistic)\n    statistic = xp.where(zero & (params < 0.0), negative_inf, statistic)\n    return xp.where(zero & (params == 0.0), xp.zeros_like(statistic), statistic)\n\n\n''',
    "zero-safe statistic helper",
)
replace_once(
    "statgpu/panel/_utils.py",
    "    tvalues = params / np.maximum(bse, np.finfo(np.float64).tiny)\n",
    "    tvalues = _zero_safe_statistic_ratio(params, bse, np)\n",
    "legacy nonrobust exact-zero statistic",
)
replace_once(
    "statgpu/panel/_utils.py",
    '''    tvalues_dev = params / xp_maximum(\n        bse_dev, np.finfo(np.float64).tiny, xp\n    )\n''',
    "    tvalues_dev = _zero_safe_statistic_ratio(params, bse_dev, xp)\n",
    "legacy panel exact-zero statistic",
)

# Shared public OLS inference: prohibit positive absolute variance floors and use
# the explicit exact-zero statistic contract for every maintained backend.
replace_once(
    "statgpu/panel/_base.py",
    '''    validate_panel_alpha,\n    validate_panel_numeric_data,\n)\n''',
    '''    validate_panel_alpha,\n    validate_panel_numeric_data,\n    _zero_safe_statistic_ratio,\n)\n''',
    "base zero-safe helper import",
)
replace_once(
    "statgpu/panel/_base.py",
    "        diag_floor=1e-30,\n",
    "        diag_floor=0.0,\n",
    "base inference default floor",
)
replace_once(
    "statgpu/panel/_base.py",
    '''        # Normalize signed zero before any historical diagonal floor is used.\n        diag = xp_maximum(diag, 0.0, xp)\n        if diag_floor is not None:\n            diag = xp_maximum(diag, float(diag_floor), xp)\n        bse_dev = xp.sqrt(diag)\n        if diag_floor is None:\n            tvalues_dev = params / bse_dev\n        else:\n            denominator = xp_maximum(\n                bse_dev, np.finfo(np.float64).tiny, xp\n            )\n            tvalues_dev = params / denominator\n''',
    '''        # Positive absolute variance floors are dimensionful and break\n        # outcome-scale equivariance. Keep the private compatibility argument,\n        # but fail closed if a caller tries to reintroduce such a floor.\n        if diag_floor not in (None, 0, 0.0):\n            raise ValueError(\n                "positive absolute covariance diagonal floors are not supported"\n            )\n        diag = xp_maximum(diag, 0.0, xp)\n        bse_dev = xp.sqrt(diag)\n        tvalues_dev = _zero_safe_statistic_ratio(params, bse_dev, xp)\n''',
    "shared inference floor removal",
)

# All estimator integrations now opt into the same zero-only normalization.
for path in (
    "statgpu/panel/_between.py",
    "statgpu/panel/_first_diff.py",
):
    replace_once(
        path,
        "            diag_floor=1e-30,\n",
        "            diag_floor=0.0,\n",
        f"{path} remove absolute variance floor",
    )
replace_once(
    "statgpu/panel/_pooled.py",
    "            diag_floor=None,\n",
    "            diag_floor=0.0,\n",
    "pooled exact-zero inference convention",
)

# Fama-MacBeth shares the same exact-zero result-surface semantics.
replace_once(
    "statgpu/panel/_fama_macbeth.py",
    "from statgpu.panel._utils import factorize_panel_labels, factorize_panel_metadata\n",
    '''from statgpu.panel._utils import (\n    _zero_safe_statistic_ratio,\n    factorize_panel_labels,\n    factorize_panel_metadata,\n)\n''',
    "fmb zero-safe helper import",
)
replace_once(
    "statgpu/panel/_fama_macbeth.py",
    '''        # Match the shared panel inference convention at an exactly zero\n        # estimated variance: 0/0 should not leak NaN into the public result\n        # surface. Positive standard errors are unchanged.\n        bse_for_stat = xp.where(\n            bse > 0.0,\n            bse,\n            xp.full_like(bse, np.finfo(np.float64).tiny),\n        )\n        tvalues = avg_beta / bse_for_stat\n''',
    '''        # Exact-zero variance is handled without a dimensionful fake\n        # denominator: beta=0 gives statistic 0; beta!=0 gives signed infinity.\n        tvalues = _zero_safe_statistic_ratio(avg_beta, bse, xp)\n''',
    "fmb exact-zero statistic",
)

# Hosted NumPy regressions: exact-zero semantics and public scale equivariance.
p = Path("dev/tests/test_panel_stage_c_inference_guard.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel import PanelOLS, RandomEffects, _covariance\n",
    "from statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, RandomEffects, _covariance\n",
    1,
)
text = text.replace("diag_floor=1.0e-30", "diag_floor=0.0")
anchor = "def test_hausman_rejects_rank_deficient_nonunique_coefficients():\n"
addition = '''def test_exact_zero_variance_statistics_do_not_use_a_fake_denominator(monkeypatch):\n    def _zero_covariance(*args, **kwargs):\n        return np.zeros((3, 3), dtype=np.float64)\n\n    monkeypatch.setattr(_covariance, "ols_covariance", _zero_covariance)\n    model = _DummyPanelModel()\n    backend = model._get_backend(backend="auto")\n    tiny = np.nextafter(0.0, 1.0)\n    params = np.asarray([0.0, tiny, -tiny], dtype=np.float64)\n    model._panel_store_ols_inference(\n        np.eye(3),\n        np.zeros(3),\n        params,\n        scale=0.0,\n        df_resid=3,\n        backend=backend,\n        fit_rank=3,\n        cov_type="hc0",\n        allowed=("hc0",),\n        diag_floor=0.0,\n    )\n\n    np.testing.assert_array_equal(model.bse_, np.zeros(3))\n    assert model.tvalues_[0] == 0.0\n    assert np.isposinf(model.tvalues_[1])\n    assert np.isneginf(model.tvalues_[2])\n    np.testing.assert_array_equal(model.pvalues_, np.asarray([1.0, 0.0, 0.0]))\n    np.testing.assert_array_equal(model.conf_int_, np.column_stack([params, params]))\n\n\ndef test_between_and_first_difference_inference_is_outcome_scale_equivariant():\n    rng = np.random.default_rng(12989)\n    n_entities, n_times = 12, 5\n    entity = np.repeat(np.arange(n_entities), n_times)\n    time = np.tile(np.arange(n_times), n_entities)\n    x = rng.normal(size=entity.size)\n    X = x[:, None]\n    alpha = np.repeat(rng.normal(scale=0.35, size=n_entities), n_times)\n    y = 0.7 * x + alpha + rng.normal(scale=0.18, size=entity.size)\n    response_scale = 1.0e-20\n\n    cases = (\n        (BetweenOLS(cov_type="hc0"), {"entity_ids": entity}),\n        (FirstDifferenceOLS(cov_type="hc0"), {"entity_ids": entity, "time_ids": time}),\n    )\n    for estimator, kwargs in cases:\n        reference = estimator.fit(X, y, **kwargs)\n        scaled = type(estimator)(cov_type="hc0").fit(\n            X, response_scale * y, **kwargs\n        )\n        np.testing.assert_allclose(\n            scaled.coef_, response_scale * reference.coef_, rtol=2e-10, atol=0.0\n        )\n        np.testing.assert_allclose(\n            scaled.bse_, response_scale * reference.bse_, rtol=2e-9, atol=0.0\n        )\n        np.testing.assert_allclose(\n            scaled.tvalues_, reference.tvalues_, rtol=2e-9, atol=2e-12\n        )\n        np.testing.assert_allclose(\n            scaled.pvalues_, reference.pvalues_, rtol=2e-9, atol=2e-14\n        )\n\n\n'''
if anchor not in text:
    raise RuntimeError("inference guard append anchor missing")
p.write_text(text.replace(anchor, addition + anchor, 1), encoding="utf-8")

# Maintained Torch-CPU coverage for the shared exact-zero primitive and public
# Between/FD response-scale equivariance.
p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel import FamaMacBeth, PanelOLS, PooledOLS, RandomEffects\n",
    "from statgpu.panel import BetweenOLS, FamaMacBeth, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects\n",
    1,
)
text = text.replace(
    "from statgpu.panel._diagnostics import _diagnostic_identity, _fingerprints_match\n",
    "from statgpu.panel._diagnostics import _diagnostic_identity, _fingerprints_match\nfrom statgpu.panel._utils import _zero_safe_statistic_ratio\n",
    1,
)
text += '''\n\ndef test_stage_c_torch_cpu_zero_variance_and_response_scale_equivariance():\n    tiny = np.nextafter(0.0, 1.0)\n    params = torch.tensor([0.0, tiny, -tiny], dtype=torch.float64)\n    bse = torch.zeros(3, dtype=torch.float64)\n    statistic = _zero_safe_statistic_ratio(params, bse, torch).detach().cpu().numpy()\n    assert statistic[0] == 0.0\n    assert np.isposinf(statistic[1])\n    assert np.isneginf(statistic[2])\n\n    rng = np.random.default_rng(1230)\n    n_entities, n_times = 10, 5\n    entity = np.repeat(np.arange(n_entities), n_times)\n    time = np.tile(np.arange(n_times), n_entities)\n    x = rng.normal(size=entity.size)\n    X = x[:, None]\n    alpha = np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)\n    y = 0.65 * x + alpha + rng.normal(scale=0.16, size=entity.size)\n    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)\n    response_scale = 1.0e-20\n\n    reference_between = BetweenOLS(cov_type="hc0").fit(X_t, y_t, entity_ids=entity_t)\n    scaled_between = BetweenOLS(cov_type="hc0").fit(\n        X_t, response_scale * y_t, entity_ids=entity_t\n    )\n    reference_fd = FirstDifferenceOLS(cov_type="hc0").fit(\n        X_t, y_t, entity_ids=entity_t, time_ids=time_t\n    )\n    scaled_fd = FirstDifferenceOLS(cov_type="hc0").fit(\n        X_t, response_scale * y_t, entity_ids=entity_t, time_ids=time_t\n    )\n    for reference, scaled in (\n        (reference_between, scaled_between),\n        (reference_fd, scaled_fd),\n    ):\n        assert_allclose(scaled.coef_, response_scale * reference.coef_, rtol=2e-9, atol=0.0)\n        assert_allclose(scaled.bse_, response_scale * reference.bse_, rtol=2e-8, atol=0.0)\n        assert_allclose(scaled.tvalues_, reference.tvalues_, rtol=2e-8, atol=2e-11)\n        assert_allclose(scaled.pvalues_, reference.pvalues_, rtol=2e-8, atol=2e-13)\n'''
p.write_text(text, encoding="utf-8")

# Physical CUDA primitive for exact-zero and tiny-positive standard-error paths.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''from statgpu.panel._linalg import (\n    panel_lstsq,\n    panel_lstsq_batched,\n    panel_lstsq_gram_certified_batched,\n    panel_matrix_rank,\n)\n''',
    '''from statgpu.panel._linalg import (\n    panel_lstsq,\n    panel_lstsq_batched,\n    panel_lstsq_gram_certified_batched,\n    panel_matrix_rank,\n)\nfrom statgpu.panel._utils import _zero_safe_statistic_ratio\n''',
    "physical zero-safe helper import",
)
append_before(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "def _tiny_design_lstsq_audit(backend):\n",
    '''def _zero_variance_inference_audit(backend):\n    tiny = np.nextafter(0.0, 1.0)\n    params_np = np.asarray([0.0, tiny, -tiny], dtype=np.float64)\n    bse_np = np.zeros(3, dtype=np.float64)\n    regular_params_np = np.asarray([2.0e-20, -3.0e-20], dtype=np.float64)\n    regular_bse_np = np.asarray([1.0e-20, 1.0e-20], dtype=np.float64)\n    if backend == "numpy":\n        xp = np\n        params, bse = params_np, bse_np\n        regular_params, regular_bse = regular_params_np, regular_bse_np\n    elif backend == "cupy":\n        import cupy as cp\n        xp = cp\n        params, bse = cp.asarray(params_np), cp.asarray(bse_np)\n        regular_params, regular_bse = cp.asarray(regular_params_np), cp.asarray(regular_bse_np)\n    elif backend == "torch":\n        import torch\n        xp = torch\n        params = torch.as_tensor(params_np, dtype=torch.float64, device="cuda")\n        bse = torch.as_tensor(bse_np, dtype=torch.float64, device="cuda")\n        regular_params = torch.as_tensor(regular_params_np, dtype=torch.float64, device="cuda")\n        regular_bse = torch.as_tensor(regular_bse_np, dtype=torch.float64, device="cuda")\n    else:\n        raise ValueError(backend)\n\n    exact = _array(_zero_safe_statistic_ratio(params, bse, xp))\n    regular = _array(_zero_safe_statistic_ratio(regular_params, regular_bse, xp))\n    if exact[0] != 0.0 or not np.isposinf(exact[1]) or not np.isneginf(exact[2]):\n        raise AssertionError(f"{backend}: exact-zero inference semantics drifted: {exact}")\n    np.testing.assert_allclose(regular, np.asarray([2.0, -3.0]), rtol=0.0, atol=0.0)\n    return {\n        "status": "success",\n        "backend": backend,\n        "zero_coefficient_statistic": float(exact[0]),\n        "positive_zero_variance_is_inf": bool(np.isposinf(exact[1])),\n        "negative_zero_variance_is_inf": bool(np.isneginf(exact[2])),\n        "tiny_positive_bse_statistics": regular.tolist(),\n    }\n\n\n''',
    "physical exact-zero audit",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),\n            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),\n            "diagnostic_scale_reductions": diagnostic_scale,\n''',
    '''            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),\n            "zero_variance_inference": _zero_variance_inference_audit(backend),\n            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),\n            "diagnostic_scale_reductions": diagnostic_scale,\n''',
    "physical numerical primitive payload",
)

# Hosted contract executes the new validator primitive without a GPU.
p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
text += '''\n\ndef test_stage_c_runner_zero_variance_inference_audit_is_executable():\n    audit = _MOD._zero_variance_inference_audit("numpy")\n    assert audit["status"] == "success"\n    assert audit["backend"] == "numpy"\n    assert audit["zero_coefficient_statistic"] == 0.0\n    assert audit["positive_zero_variance_is_inf"] is True\n    assert audit["negative_zero_variance_is_inf"] is True\n    np.testing.assert_array_equal(\n        audit["tiny_positive_bse_statistics"], np.asarray([2.0, -3.0])\n    )\n'''
p.write_text(text, encoding="utf-8")
