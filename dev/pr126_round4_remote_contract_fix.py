from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Fresh correctness artifacts now carry inference-applicability and prediction
# provenance fields, so distinguish them from the frozen schema-1 v3 lineage.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "from statgpu.panel._covariance import ols_covariance\n\n\ndef _git_sha() -> str:\n",
    "from statgpu.panel._covariance import ols_covariance\n\n\nCORRECTNESS_SCHEMA_VERSION = 2\n\n\ndef _git_sha() -> str:\n",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '        "schema_version": 1,\n',
    '        "schema_version": CORRECTNESS_SCHEMA_VERSION,\n',
)

# Performance schema v3 adds a dedicated unbalanced two-way FE scenario so the
# newly hardened alternating projection is performance-observable on physical GPU.
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''PERFORMANCE_SCHEMA_VERSION = 2\nDEFAULT_HIGH_T_SCALE = "10000x2x200"\nHIGH_T_CASES = ("pooled_dk_qs", "panel_entity_dk_qs")\n''',
    '''PERFORMANCE_SCHEMA_VERSION = 3\nDEFAULT_HIGH_T_SCALE = "10000x2x200"\nDEFAULT_TWO_WAY_UNBALANCED_SCALE = "10000x2x20"\nHIGH_T_CASES = ("pooled_dk_qs", "panel_entity_dk_qs")\nTWO_WAY_UNBALANCED_CASE = "panel_two_way_nonrobust"\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''    clusters = np.column_stack([entity, time_ids])\n    return X, y.astype(np.float64), entity, time_ids, clusters\n\n\ndef _to_backend(X, y, entity, time_ids, backend):\n''',
    '''    clusters = np.column_stack([entity, time_ids])\n    return X, y.astype(np.float64), entity, time_ids, clusters\n\n\ndef _unbalanced_two_way_dataset(n, k, seed, *, n_times=20):\n    """Deterministic incomplete panel for alternating-projection timing."""\n    if int(n_times) < 2 or int(n) < int(n_times):\n        raise ValueError("two-way dataset requires n_times>=2 and n>=n_times")\n    rng = np.random.default_rng(seed)\n    n_times = int(n_times)\n    n_entities = max(2, int(np.ceil((1.2 * n) / n_times)))\n    entity_full = np.repeat(np.arange(n_entities), n_times)\n    time_full = np.tile(np.arange(n_times), n_entities)\n    keep = ((7 * entity_full + 11 * time_full) % 17) != 0\n    retained = np.flatnonzero(keep)\n    if retained.size < int(n):\n        raise RuntimeError("deterministic two-way mask did not retain enough rows")\n    idx = retained[: int(n)]\n    X_full = rng.normal(size=(entity_full.size, k)).astype(np.float64)\n    beta = np.linspace(0.15, 0.75, k, dtype=np.float64)\n    alpha = rng.normal(scale=0.35, size=n_entities)\n    tau = rng.normal(scale=0.2, size=n_times)\n    y_full = (\n        X_full @ beta\n        + alpha[entity_full]\n        + tau[time_full]\n        + rng.normal(scale=0.25, size=entity_full.size)\n    )\n    entity = entity_full[idx]\n    time_ids = time_full[idx]\n    X = X_full[idx]\n    y = y_full[idx].astype(np.float64)\n    clusters = np.column_stack([entity, time_ids])\n    if np.unique(clusters, axis=0).shape[0] != int(n):\n        raise RuntimeError("two-way performance dataset contains duplicate panel cells")\n    return X, y, entity, time_ids, clusters\n\n\ndef _to_backend(X, y, entity, time_ids, backend):\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''    if case == "panel_entity_dk_qs":\n        return PanelOLS(\n            entity_effects=True, cov_type="dk", bandwidth=2, kernel="qs", device=device\n        ).fit(X, y, entity_ids=entity, time_ids=time_ids)\n    if case == "random_effects_nonrobust":\n''',
    '''    if case == "panel_entity_dk_qs":\n        return PanelOLS(\n            entity_effects=True, cov_type="dk", bandwidth=2, kernel="qs", device=device\n        ).fit(X, y, entity_ids=entity, time_ids=time_ids)\n    if case == "panel_two_way_nonrobust":\n        return PanelOLS(\n            entity_effects=True,\n            time_effects=True,\n            cov_type="nonrobust",\n            device=device,\n        ).fit(X, y, entity_ids=entity, time_ids=time_ids)\n    if case == "random_effects_nonrobust":\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''    parser.add_argument(\n        "--high-t-scale",\n        default=DEFAULT_HIGH_T_SCALE,\n        help="additional NxKxT scenario used only for QS all-lag cases",\n    )\n    parser.add_argument("--repeats", type=int, default=3)\n''',
    '''    parser.add_argument(\n        "--high-t-scale",\n        default=DEFAULT_HIGH_T_SCALE,\n        help="additional NxKxT scenario used only for QS all-lag cases",\n    )\n    parser.add_argument(\n        "--two-way-unbalanced-scale",\n        default=DEFAULT_TWO_WAY_UNBALANCED_SCALE,\n        help="NxKxT scenario for the iterative two-way FE performance path",\n    )\n    parser.add_argument("--repeats", type=int, default=3)\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''    for backend in backends:\n        X, y, entity, time_ids = _to_backend(\n            X_np, y_np, entity_np, time_np, backend\n        )\n        for case in HIGH_T_CASES:\n            _timed(case, X, y, entity, time_ids, clusters, backend)\n            samples = [\n                _timed(case, X, y, entity, time_ids, clusters, backend)\n                for _ in range(args.repeats)\n            ]\n            rows.append(\n                _timing_row(\n                    backend=backend,\n                    case=case,\n                    scenario="high_t_qs",\n                    n=high_n,\n                    k=high_k,\n                    n_times=len(np.unique(time_np)),\n                    repeats=args.repeats,\n                    samples=samples,\n                )\n            )\n\n    payload = {\n''',
    '''    for backend in backends:\n        X, y, entity, time_ids = _to_backend(\n            X_np, y_np, entity_np, time_np, backend\n        )\n        for case in HIGH_T_CASES:\n            _timed(case, X, y, entity, time_ids, clusters, backend)\n            samples = [\n                _timed(case, X, y, entity, time_ids, clusters, backend)\n                for _ in range(args.repeats)\n            ]\n            rows.append(\n                _timing_row(\n                    backend=backend,\n                    case=case,\n                    scenario="high_t_qs",\n                    n=high_n,\n                    k=high_k,\n                    n_times=len(np.unique(time_np)),\n                    repeats=args.repeats,\n                    samples=samples,\n                )\n            )\n\n    tw_n, tw_k, tw_t = _parse_high_t_scale(args.two_way_unbalanced_scale)\n    X_np, y_np, entity_np, time_np, clusters = _unbalanced_two_way_dataset(\n        tw_n, tw_k, 20260900, n_times=tw_t\n    )\n    for backend in backends:\n        X, y, entity, time_ids = _to_backend(\n            X_np, y_np, entity_np, time_np, backend\n        )\n        _timed(\n            TWO_WAY_UNBALANCED_CASE, X, y, entity, time_ids, clusters, backend\n        )\n        samples = [\n            _timed(\n                TWO_WAY_UNBALANCED_CASE,\n                X,\n                y,\n                entity,\n                time_ids,\n                clusters,\n                backend,\n            )\n            for _ in range(args.repeats)\n        ]\n        rows.append(\n            _timing_row(\n                backend=backend,\n                case=TWO_WAY_UNBALANCED_CASE,\n                scenario="two_way_unbalanced",\n                n=tw_n,\n                k=tw_k,\n                n_times=len(np.unique(time_np)),\n                repeats=args.repeats,\n                samples=samples,\n            )\n        )\n\n    payload = {\n''',
)
replace_once(
    "dev/benchmarks/benchmark_panel_stage_c_covariance.py",
    '''        "high_t_scale": args.high_t_scale,\n        "environment": {\n''',
    '''        "high_t_scale": args.high_t_scale,\n        "two_way_unbalanced_scale": args.two_way_unbalanced_scale,\n        "environment": {\n''',
)

# Hosted runner contracts pin the fresh schemas and the dedicated unbalanced path.
replace_once(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    '''_SPEC.loader.exec_module(_MOD)\n\n\ndef test_stage_c_runner_numpy_reference_matrix_is_complete_and_executable():\n''',
    '''_SPEC.loader.exec_module(_MOD)\n\n\ndef test_stage_c_runner_fresh_schema_version_is_explicit():\n    assert _MOD.CORRECTNESS_SCHEMA_VERSION == 2\n\n\ndef test_stage_c_runner_numpy_reference_matrix_is_complete_and_executable():\n''',
)
replace_once(
    "dev/tests/test_panel_stage_c_performance_runner_contract.py",
    '''def test_high_t_scenario_is_explicit_and_qs_only():\n    mod = _runner()\n''',
    '''def test_high_t_scenario_is_explicit_and_qs_only():\n    mod = _runner()\n    assert mod.PERFORMANCE_SCHEMA_VERSION == 3\n''',
)
replace_once(
    "dev/tests/test_panel_stage_c_performance_runner_contract.py",
    '''def test_dataset_honors_requested_time_dimension():\n    mod = _runner()\n''',
    '''def test_two_way_unbalanced_scenario_is_explicit_and_incomplete():\n    mod = _runner()\n    assert mod.DEFAULT_TWO_WAY_UNBALANCED_SCALE == "10000x2x20"\n    assert mod.TWO_WAY_UNBALANCED_CASE == "panel_two_way_nonrobust"\n    X, y, entity, time, clusters = mod._unbalanced_two_way_dataset(\n        1000, 2, 43, n_times=20\n    )\n    assert X.shape == (1000, 2)\n    assert y.shape == (1000,)\n    assert clusters.shape == (1000, 2)\n    assert np.unique(clusters, axis=0).shape[0] == 1000\n    counts = np.bincount(entity)\n    assert np.unique(counts[counts > 0]).size > 1\n    assert len(np.unique(time)) == 20\n\n\ndef test_dataset_honors_requested_time_dimension():\n    mod = _runner()\n''',
)

print("PR126 round4 remote evidence contract fixes applied")
