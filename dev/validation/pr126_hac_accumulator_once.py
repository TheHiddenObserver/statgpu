from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_before(path, anchor, addition, label):
    replace_once(path, anchor, addition + anchor, label)


append_before(
    "statgpu/panel/_covariance.py",
    "def _stable_inclusion_exclusion(V1, V2, V12, xp):\n",
    '''def _covariance_accumulator_start(initial, *, max_terms: int, xp):\n    """Start a per-entry reduction-length covariance accumulator.\n\n    Entries are scaled only when one of at most ``max_terms`` same-sign\n    contributions could overflow a representable final reduction. Safe and\n    subnormal entries remain on their original scale.\n    """\n    max_terms = max(1, int(max_terms))\n    threshold = float(np.finfo(np.float64).max) / float(max_terms)\n    scaled = xp.abs(initial) > threshold\n    divisor = xp.where(\n        scaled,\n        xp.full_like(initial, float(max_terms)),\n        xp.ones_like(initial),\n    )\n    return initial / divisor, scaled\n\n\ndef _covariance_accumulator_add(\n    accumulator,\n    scaled,\n    base,\n    multiplier: float,\n    *,\n    max_terms: int,\n    xp,\n):\n    """Add ``multiplier * base`` without materializing a risky term/sum."""\n    max_terms = max(1, int(max_terms))\n    multiplier = float(multiplier)\n    if multiplier == 0.0:\n        return accumulator, scaled\n\n    abs_multiplier = abs(multiplier)\n    max_float = float(np.finfo(np.float64).max)\n    accumulator_threshold = max_float / float(max_terms)\n    base_threshold = max_float / (float(max_terms) * abs_multiplier)\n    needs_scale = (~scaled) & (\n        (xp.abs(accumulator) > accumulator_threshold)\n        | (xp.abs(base) > base_threshold)\n    )\n\n    divisor = xp.where(\n        needs_scale,\n        xp.full_like(accumulator, float(max_terms)),\n        xp.ones_like(accumulator),\n    )\n    accumulator = accumulator / divisor\n    scaled = scaled | needs_scale\n    coefficient = xp.where(\n        scaled,\n        xp.full_like(base, multiplier / float(max_terms)),\n        xp.full_like(base, multiplier),\n    )\n    return accumulator + base * coefficient, scaled\n\n\ndef _covariance_accumulator_finish(accumulator, scaled, *, max_terms: int, xp):\n    """Restore the original scale after safe lag-term accumulation."""\n    max_terms = max(1, int(max_terms))\n    factor = xp.where(\n        scaled,\n        xp.full_like(accumulator, float(max_terms)),\n        xp.ones_like(accumulator),\n    )\n    return accumulator * factor\n\n\n''',
    "covariance lag accumulator helpers",
)

replace_once(
    "statgpu/panel/_covariance.py",
    '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)\n    cov = influence.T @ influence\n    for h in range(1, bandwidth + 1):\n        w = 1.0 - h / (bandwidth + 1.0)\n        gamma_h = influence[h:].T @ influence[: n - h]\n        cov = cov + _weighted_symmetric_sum(gamma_h, float(w))\n    return _symmetrize(cov)\n''',
    '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)\n    max_terms = int(bandwidth) + 1\n    cov, scaled = _covariance_accumulator_start(\n        _symmetrize(influence.T @ influence), max_terms=max_terms, xp=xp\n    )\n    for h in range(1, bandwidth + 1):\n        w = 1.0 - h / (bandwidth + 1.0)\n        gamma_h = influence[h:].T @ influence[: n - h]\n        cov, scaled = _covariance_accumulator_add(\n            cov,\n            scaled,\n            _symmetrize(gamma_h),\n            2.0 * float(w),\n            max_terms=max_terms,\n            xp=xp,\n        )\n    cov = _covariance_accumulator_finish(\n        cov, scaled, max_terms=max_terms, xp=xp\n    )\n    return _symmetrize(cov)\n''',
    "HAC lag accumulator",
)

replace_once(
    "statgpu/panel/_covariance.py",
    '''    cov = grouped.T @ grouped\n    for lag in range(1, n_periods):\n        if weights_np[lag] == 0.0:\n            continue\n        gamma = grouped[lag:].T @ grouped[: n_periods - lag]\n        cov = cov + _weighted_symmetric_sum(gamma, weights[lag])\n\n    scale = float(n) / float(denom)\n    cov = _symmetrize(scale * cov)\n''',
    '''    max_terms = 1 + int(np.count_nonzero(weights_np[1:]))\n    cov, scaled = _covariance_accumulator_start(\n        _symmetrize(grouped.T @ grouped), max_terms=max_terms, xp=xp\n    )\n    for lag in range(1, n_periods):\n        if weights_np[lag] == 0.0:\n            continue\n        gamma = grouped[lag:].T @ grouped[: n_periods - lag]\n        cov, scaled = _covariance_accumulator_add(\n            cov,\n            scaled,\n            _symmetrize(gamma),\n            2.0 * float(weights_np[lag]),\n            max_terms=max_terms,\n            xp=xp,\n        )\n    cov = _covariance_accumulator_finish(\n        cov, scaled, max_terms=max_terms, xp=xp\n    )\n\n    scale = float(n) / float(denom)\n    cov = _symmetrize(scale * cov)\n''',
    "DK lag accumulator",
)

# Public NumPy regression with a finite final HAC/DK covariance but an overflowing\n# historical intermediate after the first positive Bartlett lag.
p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
text += '''\n\ndef test_hac_and_dk_lag_accumulator_survives_finite_final_after_overflowing_partial_sum():\n    n = 7\n    bandwidth = 4\n    influence_sq = 2.0e307\n    influence_amp = float(np.sqrt(influence_sq))\n    signs = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])\n    X = np.ones((n, 1), dtype=np.float64)\n    resid = n * influence_amp * signs\n    time = np.arange(n, dtype=np.int64)\n\n    # In exact arithmetic the Bartlett meat coefficients are\n    # 7 + 3.2 - 3.6 - 1.6 + 0.4 = 5.4. The historical sequential path\n    # overflowed after 7 + 3.2 = 10.2 even though the final result is finite.\n    expected_hac = np.asarray([[5.4 * influence_sq]], dtype=np.float64)\n    expected_dk = expected_hac * (n / (n - 1.0))\n    hac = hac_covariance(X, resid, bandwidth=bandwidth)\n    dk = driscoll_kraay_covariance(\n        X, resid, time, bandwidth=bandwidth, kernel="bartlett"\n    )\n    assert np.all(np.isfinite(hac))\n    assert np.all(np.isfinite(dk))\n    np.testing.assert_allclose(hac, expected_hac, rtol=1.2e-14, atol=0.0)\n    np.testing.assert_allclose(dk, expected_dk, rtol=1.5e-14, atol=0.0)\n'''
p.write_text(text, encoding="utf-8")

# Maintained Torch 2.0.1 CPU regression.
p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
text += '''\n\ndef test_stage_c_torch_cpu_lag_accumulator_preserves_finite_hac_and_dk():\n    n = 7\n    bandwidth = 4\n    influence_sq = 2.0e307\n    influence_amp = float(np.sqrt(influence_sq))\n    signs_np = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])\n    X = torch.ones((n, 1), dtype=torch.float64)\n    resid = torch.as_tensor(n * influence_amp * signs_np, dtype=torch.float64)\n    time = np.arange(n, dtype=np.int64)\n    expected_hac = np.asarray([[5.4 * influence_sq]], dtype=np.float64)\n    assert_allclose(\n        hac_covariance(X, resid, bandwidth=bandwidth, xp=torch),\n        expected_hac,\n        rtol=1.2e-13,\n        atol=0.0,\n    )\n    assert_allclose(\n        driscoll_kraay_covariance(\n            X, resid, time, bandwidth=bandwidth, kernel="bartlett", xp=torch\n        ),\n        expected_hac * (n / (n - 1.0)),\n        rtol=1.5e-13,\n        atol=0.0,\n    )\n'''
p.write_text(text, encoding="utf-8")

# Extend physical CUDA audit with the same lag-accumulation counterexample.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
old = '''    n = 16\n    influence_amplitude = 3.0e153\n    X_hac_np = np.ones((n, 1), dtype=np.float64)\n    resid_hac_np = n * influence_amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)\n    time = np.arange(n, dtype=np.int64)\n'''
new = '''    n = 16\n    influence_amplitude = 3.0e153\n    X_hac_np = np.ones((n, 1), dtype=np.float64)\n    resid_hac_np = n * influence_amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)\n    time = np.arange(n, dtype=np.int64)\n\n    lag_n = 7\n    lag_bandwidth = 4\n    lag_influence_sq = 2.0e307\n    lag_influence_amp = float(np.sqrt(lag_influence_sq))\n    lag_signs = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])\n    X_lag_np = np.ones((lag_n, 1), dtype=np.float64)\n    resid_lag_np = lag_n * lag_influence_amp * lag_signs\n    lag_time = np.arange(lag_n, dtype=np.int64)\n'''
if old not in text:
    raise RuntimeError("physical lag fixture anchor missing")
text = text.replace(old, new, 1)
text = text.replace(
    '''        X_hac, resid_hac = X_hac_np, resid_hac_np\n''',
    '''        X_hac, resid_hac = X_hac_np, resid_hac_np\n        X_lag, resid_lag = X_lag_np, resid_lag_np\n''',
    1,
)
text = text.replace(
    '''        X_hac, resid_hac = cp.asarray(X_hac_np), cp.asarray(resid_hac_np)\n''',
    '''        X_hac, resid_hac = cp.asarray(X_hac_np), cp.asarray(resid_hac_np)\n        X_lag, resid_lag = cp.asarray(X_lag_np), cp.asarray(resid_lag_np)\n''',
    1,
)
text = text.replace(
    '''        X_hac = torch.as_tensor(X_hac_np, dtype=torch.float64, device="cuda")\n        resid_hac = torch.as_tensor(resid_hac_np, dtype=torch.float64, device="cuda")\n''',
    '''        X_hac = torch.as_tensor(X_hac_np, dtype=torch.float64, device="cuda")\n        resid_hac = torch.as_tensor(resid_hac_np, dtype=torch.float64, device="cuda")\n        X_lag = torch.as_tensor(X_lag_np, dtype=torch.float64, device="cuda")\n        resid_lag = torch.as_tensor(resid_lag_np, dtype=torch.float64, device="cuda")\n''',
    1,
)
old = '''    hac = _array(hac_covariance(X_hac, resid_hac, bandwidth=1, xp=xp))\n    dk = _array(driscoll_kraay_covariance(X_hac, resid_hac, time, bandwidth=1, xp=xp))\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk)):\n'''
new = '''    hac = _array(hac_covariance(X_hac, resid_hac, bandwidth=1, xp=xp))\n    dk = _array(driscoll_kraay_covariance(X_hac, resid_hac, time, bandwidth=1, xp=xp))\n    lag_hac = _array(hac_covariance(X_lag, resid_lag, bandwidth=lag_bandwidth, xp=xp))\n    lag_dk = _array(\n        driscoll_kraay_covariance(\n            X_lag, resid_lag, lag_time, bandwidth=lag_bandwidth, kernel="bartlett", xp=xp\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk)):\n'''
if old not in text:
    raise RuntimeError("physical lag execution anchor missing")
text = text.replace(old, new, 1)
old = '''    np.testing.assert_allclose(hac, expected_hac, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(dk, expected_hac * (n / (n - 1.0)), rtol=8e-13, atol=0.0)\n    return {\n'''
new = '''    np.testing.assert_allclose(hac, expected_hac, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(dk, expected_hac * (n / (n - 1.0)), rtol=8e-13, atol=0.0)\n    expected_lag_hac = np.asarray([[5.4 * lag_influence_sq]], dtype=np.float64)\n    np.testing.assert_allclose(lag_hac, expected_lag_hac, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(\n        lag_dk,\n        expected_lag_hac * (lag_n / (lag_n - 1.0)),\n        rtol=8e-13,\n        atol=0.0,\n    )\n    return {\n'''
if old not in text:
    raise RuntimeError("physical lag assertion anchor missing")
text = text.replace(old, new, 1)
text = text.replace(
    '''        "driscoll_kraay": dk.tolist(),\n''',
    '''        "driscoll_kraay": dk.tolist(),\n        "lag_accumulator_hac": lag_hac.tolist(),\n        "lag_accumulator_driscoll_kraay": lag_dk.tolist(),\n''',
    1,
)
p.write_text(text, encoding="utf-8")

# Hosted executable-contract assertion for the new physical payload fields.
p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
old = '''    for key in ("one_way", "two_way", "group_cancellation", "hac", "driscoll_kraay"):\n        assert np.all(np.isfinite(np.asarray(audit[key], dtype=np.float64))), key\n'''
new = '''    for key in (\n        "one_way",\n        "two_way",\n        "group_cancellation",\n        "hac",\n        "driscoll_kraay",\n        "lag_accumulator_hac",\n        "lag_accumulator_driscoll_kraay",\n    ):\n        assert np.all(np.isfinite(np.asarray(audit[key], dtype=np.float64))), key\n'''
if old not in text:
    raise RuntimeError("physical contract covariance key anchor missing")
p.write_text(text.replace(old, new, 1), encoding="utf-8")
