from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1. Public cluster covariance must not truth-cast arbitrary group_debias values.
replace_once(
    "statgpu/panel/_covariance.py",
    '''def _group_debias_factor(n_groups: int, nobs: int) -> float:\n    n_groups = int(n_groups)\n    nobs = int(nobs)\n    if n_groups < 2:\n        raise ValueError("group_debias requires at least two groups")\n    if nobs <= 0:\n        raise ValueError("group_debias requires a positive observation count")\n    return (n_groups / (n_groups - 1.0)) * ((nobs - 1.0) / nobs)\n\n\ndef clustered_covariance(\n''',
    '''def _group_debias_factor(n_groups: int, nobs: int) -> float:\n    n_groups = int(n_groups)\n    nobs = int(nobs)\n    if n_groups < 2:\n        raise ValueError("group_debias requires at least two groups")\n    if nobs <= 0:\n        raise ValueError("group_debias requires a positive observation count")\n    return (n_groups / (n_groups - 1.0)) * ((nobs - 1.0) / nobs)\n\n\ndef _validate_group_debias(value) -> bool:\n    if not isinstance(value, (bool, np.bool_)):\n        raise ValueError("group_debias must be boolean")\n    return bool(value)\n\n\ndef clustered_covariance(\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    xp = _ensure_xp(xp)\n\n    X = xp_asarray(X, dtype=xp.float64, xp=xp)\n''',
    '''    xp = _ensure_xp(xp)\n    group_debias = _validate_group_debias(group_debias)\n\n    X = xp_asarray(X, dtype=xp.float64, xp=xp)\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    \"\"\"Two-way clustered covariance with exact intersection factorization.\"\"\"\n    xp = _ensure_xp(xp)\n    n = int(X.shape[0])\n''',
    '''    \"\"\"Two-way clustered covariance with exact intersection factorization.\"\"\"\n    xp = _ensure_xp(xp)\n    group_debias = _validate_group_debias(group_debias)\n    n = int(X.shape[0])\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    \"\"\"Dispatch residual-based panel covariance definitions.\"\"\"\n    xp = _ensure_xp(xp)\n    name = normalize_covariance_type(cov_type)\n''',
    '''    \"\"\"Dispatch residual-based panel covariance definitions.\"\"\"\n    xp = _ensure_xp(xp)\n    group_debias = _validate_group_debias(group_debias)\n    name = normalize_covariance_type(cov_type)\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    if bool(group_debias) and name != "clustered":\n''',
    '''    if group_debias and name != "clustered":\n''',
)

# 2. Lock the public primitive fail-closed behavior.
edge = Path("dev/tests/test_panel_stage_c_edge_contracts.py")
edge_text = edge.read_text(encoding="utf-8")
if "test_cluster_primitives_reject_nonboolean_group_debias" not in edge_text:
    edge_text += '''\n\n@pytest.mark.parametrize("value", ["false", 0, 1, None])\ndef test_cluster_primitives_reject_nonboolean_group_debias(value):\n    X = np.column_stack([np.ones(8), np.arange(8.0)])\n    resid = np.linspace(-0.2, 0.3, 8)\n    groups = np.repeat(np.arange(4), 2)\n    with pytest.raises(ValueError, match="group_debias must be boolean"):\n        clustered_covariance(X, resid, groups, group_debias=value)\n'''
    edge.write_text(edge_text, encoding="utf-8")

# 3. Performance runner: retain bounded base cases but add an explicit high-T QS scenario.
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''from statgpu.panel import PanelOLS, PooledOLS, RandomEffects\n\n\ndef _git_sha():\n''',
    '''from statgpu.panel import PanelOLS, PooledOLS, RandomEffects\n\n\nPERFORMANCE_SCHEMA_VERSION = 2\nDEFAULT_HIGH_T_SCALE = "10000x2x200"\nHIGH_T_CASES = ("pooled_dk_qs", "panel_entity_dk_qs")\n\n\ndef _git_sha():\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''def _parse_scales(text):\n    values = []\n    for token in text.split(","):\n        n_text, k_text = token.strip().lower().split("x", 1)\n        n, k = int(n_text), int(k_text)\n        if n <= 0 or k <= 0:\n            raise ValueError("scales must be positive NxK pairs")\n        values.append((n, k))\n    return values\n\n\ndef _sync(backend):\n''',
    '''def _parse_scales(text):\n    values = []\n    for token in text.split(","):\n        n_text, k_text = token.strip().lower().split("x", 1)\n        n, k = int(n_text), int(k_text)\n        if n <= 0 or k <= 0:\n            raise ValueError("scales must be positive NxK pairs")\n        values.append((n, k))\n    return values\n\n\ndef _parse_high_t_scale(text):\n    parts = text.strip().lower().split("x")\n    if len(parts) != 3:\n        raise ValueError("high-T scale must be an NxKxT triple")\n    n, k, n_times = (int(v) for v in parts)\n    if n <= 0 or k <= 0 or n_times < 2 or n < n_times:\n        raise ValueError("high-T scale requires positive N/K, T>=2, and N>=T")\n    return n, k, n_times\n\n\ndef _timing_row(*, backend, case, scenario, n, k, n_times, repeats, samples):\n    return {\n        "backend": backend,\n        "case": case,\n        "scenario": scenario,\n        "n_samples": int(n),\n        "n_features": int(k),\n        "n_times": int(n_times),\n        "repeats": int(repeats),\n        "median_seconds": float(np.median(samples)),\n        "samples_seconds": [float(v) for v in samples],\n    }\n\n\ndef _sync(backend):\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''def _dataset(n, k, seed):\n    rng = np.random.default_rng(seed)\n    n_times = 20\n''',
    '''def _dataset(n, k, seed, *, n_times=20):\n    if int(n_times) < 2 or int(n) < int(n_times):\n        raise ValueError("dataset requires n_times>=2 and n>=n_times")\n    rng = np.random.default_rng(seed)\n    n_times = int(n_times)\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''    if case == "panel_entity_dk":\n        return PanelOLS(\n            entity_effects=True, cov_type="dk", bandwidth=2, device=device\n        ).fit(X, y, entity_ids=entity, time_ids=time_ids)\n    if case == "random_effects_nonrobust":\n''',
    '''    if case == "panel_entity_dk":\n        return PanelOLS(\n            entity_effects=True, cov_type="dk", bandwidth=2, device=device\n        ).fit(X, y, entity_ids=entity, time_ids=time_ids)\n    if case == "panel_entity_dk_qs":\n        return PanelOLS(\n            entity_effects=True, cov_type="dk", bandwidth=2, kernel="qs", device=device\n        ).fit(X, y, entity_ids=entity, time_ids=time_ids)\n    if case == "random_effects_nonrobust":\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''    parser.add_argument("--scales", default="10000x2,100000x2,100000x10")\n    parser.add_argument("--repeats", type=int, default=3)\n''',
    '''    parser.add_argument("--scales", default="10000x2,100000x2,100000x10")\n    parser.add_argument(\n        "--high-t-scale",\n        default=DEFAULT_HIGH_T_SCALE,\n        help="additional NxKxT scenario used only for QS all-lag cases",\n    )\n    parser.add_argument("--repeats", type=int, default=3)\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''        X_np, y_np, entity_np, time_np, clusters = _dataset(\n            n, k, 20260812 + scale_idx\n        )\n''',
    '''        X_np, y_np, entity_np, time_np, clusters = _dataset(\n            n, k, 20260812 + scale_idx, n_times=20\n        )\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''                rows.append(\n                    {\n                        "backend": backend,\n                        "case": case,\n                        "n_samples": n,\n                        "n_features": k,\n                        "n_times": int(len(np.unique(time_np))),\n                        "repeats": args.repeats,\n                        "median_seconds": float(np.median(samples)),\n                        "samples_seconds": [float(v) for v in samples],\n                    }\n                )\n\n    payload = {\n        "schema_version": 1,\n''',
    '''                rows.append(\n                    _timing_row(\n                        backend=backend,\n                        case=case,\n                        scenario="base",\n                        n=n,\n                        k=k,\n                        n_times=len(np.unique(time_np)),\n                        repeats=args.repeats,\n                        samples=samples,\n                    )\n                )\n\n    high_n, high_k, high_t = _parse_high_t_scale(args.high_t_scale)\n    X_np, y_np, entity_np, time_np, clusters = _dataset(\n        high_n, high_k, 20260899, n_times=high_t\n    )\n    for backend in backends:\n        X, y, entity, time_ids = _to_backend(\n            X_np, y_np, entity_np, time_np, backend\n        )\n        for case in HIGH_T_CASES:\n            _timed(case, X, y, entity, time_ids, clusters, backend)\n            samples = [\n                _timed(case, X, y, entity, time_ids, clusters, backend)\n                for _ in range(args.repeats)\n            ]\n            rows.append(\n                _timing_row(\n                    backend=backend,\n                    case=case,\n                    scenario="high_t_qs",\n                    n=high_n,\n                    k=high_k,\n                    n_times=len(np.unique(time_np)),\n                    repeats=args.repeats,\n                    samples=samples,\n                )\n            )\n\n    payload = {\n        "schema_version": PERFORMANCE_SCHEMA_VERSION,\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''        "timing_scope": "synchronized end-to-end estimator fit",\n        "environment": {\n''',
    '''        "timing_scope": "synchronized end-to-end estimator fit",\n        "input_residency": (\n            "X/y/entity/time preloaded on selected GPU backend; "\n            "cluster labels remain CPU metadata"\n        ),\n        "high_t_scale": args.high_t_scale,\n        "environment": {\n''',
)

# 4. Add a pure CPU contract test for performance schema/high-T coverage.
perf_test = Path("dev/tests/test_panel_stage_c_performance_runner_contract.py")
perf_test.write_text(
    '''"""Hosted contract checks for the Stage-C physical performance runner."""\n\nfrom __future__ import annotations\n\nimport importlib.util\nfrom pathlib import Path\n\nimport numpy as np\nimport pytest\n\n\ndef _runner():\n    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_panel_stage_c_covariance.py"\n    spec = importlib.util.spec_from_file_location("panel_stage_c_perf_runner", path)\n    module = importlib.util.module_from_spec(spec)\n    assert spec.loader is not None\n    spec.loader.exec_module(module)\n    return module\n\n\ndef test_high_t_scenario_is_explicit_and_qs_only():\n    mod = _runner()\n    n, k, n_times = mod._parse_high_t_scale(mod.DEFAULT_HIGH_T_SCALE)\n    assert (n, k, n_times) == (10000, 2, 200)\n    assert n_times >= 200\n    assert set(mod.HIGH_T_CASES) == {"pooled_dk_qs", "panel_entity_dk_qs"}\n\n\ndef test_dataset_honors_requested_time_dimension():\n    mod = _runner()\n    X, y, entity, time, clusters = mod._dataset(1000, 2, 42, n_times=100)\n    assert X.shape == (1000, 2)\n    assert y.shape == (1000,)\n    assert clusters.shape == (1000, 2)\n    assert len(np.unique(time)) == 100\n    assert len(entity) == 1000\n    with pytest.raises(ValueError, match="n>=n_times"):\n        mod._dataset(10, 2, 42, n_times=20)\n\n\ndef test_timing_row_schema_records_scenario_and_time_dimension():\n    mod = _runner()\n    row = mod._timing_row(\n        backend="cupy",\n        case="pooled_dk_qs",\n        scenario="high_t_qs",\n        n=10000,\n        k=2,\n        n_times=200,\n        repeats=3,\n        samples=[0.3, 0.2, 0.4],\n    )\n    assert row["scenario"] == "high_t_qs"\n    assert row["n_times"] == 200\n    assert row["median_seconds"] == pytest.approx(0.3)\n    assert row["samples_seconds"] == [0.3, 0.2, 0.4]\n    assert mod.PERFORMANCE_SCHEMA_VERSION >= 2\n''',
    encoding="utf-8",
)

# 5. Pinned external oversized-bandwidth regression.
ext = Path("dev/tests/test_panel_stage_c_external_defaults.py")
ext_text = ext.read_text(encoding="utf-8")
if "test_driscoll_kraay_oversized_bandwidth_matches_linearmodels_7_0" not in ext_text:
    ext_text += '''\n\n@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])\ndef test_driscoll_kraay_oversized_bandwidth_matches_linearmodels_7_0(kernel):\n    rng = np.random.default_rng(12707)\n    n_entities, n_times = 10, 5\n    entity = np.repeat(np.arange(n_entities), n_times)\n    time = np.tile(np.arange(n_times), n_entities)\n    X = np.column_stack([np.ones(entity.size), rng.normal(size=(entity.size, 2))])\n    beta = np.array([0.15, 0.55, -0.2])\n    y = X @ beta + rng.normal(scale=0.2, size=entity.size)\n    params = np.linalg.lstsq(X, y, rcond=None)[0]\n    resid = y - X @ params\n    bandwidth = 9\n\n    meta = {}\n    actual = driscoll_kraay_covariance(\n        X, resid, time, bandwidth=bandwidth, kernel=kernel, metadata=meta\n    )\n    expected = DriscollKraay(\n        y[:, None],\n        X,\n        params[:, None],\n        entity[:, None],\n        time[:, None],\n        debiased=True,\n        extra_df=0,\n        kernel=kernel,\n        bandwidth=float(bandwidth),\n    ).cov\n\n    assert meta["bandwidth"] == bandwidth\n    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)\n'''
    ext.write_text(ext_text, encoding="utf-8")

# 6. Keep the reviewed plan synchronized with the corrected bandwidth/perf contract.
replace_once(
    "dev/plans/panel_p1_stage_c_covariance_plan.md",
    '''Default:\n\n```text\nbw = floor(4 * (T/100)^(2/9)).\n```\n\nKernel aliases/formulas follow linearmodels:\n''',
    '''Default:\n\n```text\nbw = floor(4 * (T/100)^(2/9)).\n```\n\nAn explicit bandwidth is retained even when `bw > T-1`. Only observed lags\n`1,...,T-1` can contribute, but Bartlett/Parzen still use the requested `bw`\nin their weight denominator and QS keeps it as the smoothing scale. Stage C\ndoes not silently replace an oversized bandwidth by `T-1`.\n\nKernel aliases/formulas follow linearmodels:\n''',
)
replace_once(
    "dev/plans/panel_p1_stage_c_covariance_plan.md",
    '''- bandwidth default/zero/cap/validation;\n''',
    '''- bandwidth default/zero/oversized/validation;\n''',
)
replace_once(
    "dev/plans/panel_p1_stage_c_covariance_plan.md",
    '''- QS all-lag cost is reported at representative T and does not accidentally become `O(n^2)` in observations;\n''',
    '''- QS all-lag cost is reported at representative T and does not accidentally become `O(n^2)` in observations; the default runner includes a bounded same-order `N=10,000`, `k=2`, `T=200` high-T QS scenario in addition to the `T=20` base cases;\n''',
)
