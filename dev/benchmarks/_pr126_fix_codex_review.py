from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected anchor missing in {path}")
    text2 = text.replace(old, new, 1)
    p.write_text(text2, encoding="utf-8")


# Preserve ordered pandas categorical chronology in DK factorization.
replace_once(
    "statgpu/panel/_covariance.py",
    '''def _factorize_1d_labels(values, *, nobs: int, name: str):\n    raw = np.asarray(_to_numpy(values))\n''',
    '''def _ordered_categorical_factorization(values, *, nobs: int, name: str):\n    """Return observed ordered-categorical labels/codes without losing chronology.\n\n    pandas ordered categoricals carry an explicit semantic order that can differ\n    from lexical label order (for example ``t1, t2, t10``).  Preserve that\n    metadata before any NumPy coercion.  Unused categories are omitted so the\n    existing Stage-C contract continues to operate on distinct *observed*\n    periods, while observed categories retain their declared relative order.\n    """\n    candidate = getattr(values, "array", values)\n    dtype = getattr(candidate, "dtype", None)\n    categories = getattr(dtype, "categories", None)\n    if categories is None or not bool(getattr(dtype, "ordered", False)):\n        return None\n    codes = getattr(candidate, "codes", None)\n    if codes is None:\n        return None\n\n    codes_np = np.asarray(codes, dtype=np.int64).ravel()\n    if codes_np.shape[0] != int(nobs):\n        raise ValueError(f"{name} must be one-dimensional with length n_samples")\n    if np.any(codes_np < 0):\n        raise ValueError(f"{name} must not contain missing or non-finite values")\n\n    observed = np.unique(codes_np)\n    labels = np.asarray(categories)[observed]\n    remapped = np.searchsorted(observed, codes_np).astype(np.int64, copy=False)\n    return labels, remapped\n\n\ndef _factorize_1d_labels(values, *, nobs: int, name: str):\n    categorical = _ordered_categorical_factorization(\n        values, nobs=nobs, name=name\n    )\n    if categorical is not None:\n        return categorical\n\n    raw = np.asarray(_to_numpy(values))\n''',
)

# Preserve ordered-categorical metadata across formula row alignment.
replace_once(
    "statgpu/panel/_formula.py",
    '''def _align_formula_side_array(values, design_info, expected_n=None, name="array"):\n    """Align an observation-level side array with rows retained by Patsy."""\n    if values is None:\n        return None\n    arr = np.asarray(values)\n    if arr.ndim == 0:\n        raise ValueError(f"{name} must be observation-level")\n    positions = getattr(design_info, "_statgpu_row_positions", None)\n    if positions is None:\n        if expected_n is not None and arr.shape[0] != expected_n:\n            raise ValueError(f"{name} must have {expected_n} observations")\n        return arr\n    positions = np.asarray(positions, dtype=np.int64)\n    if arr.shape[0] == positions.shape[0]:\n        return arr\n    if positions.size and arr.shape[0] > int(positions.max()):\n        return arr[positions]\n    if positions.size == 0 and arr.shape[0] == 0:\n        return arr\n    raise ValueError(f"{name} has {arr.shape[0]} observations and cannot be aligned to the {positions.shape[0]} rows retained by the formula")\n''',
    '''def _ordered_categorical_array(values):\n    """Return an ordered categorical array-like without importing pandas."""\n    candidate = getattr(values, "array", values)\n    dtype = getattr(candidate, "dtype", None)\n    if (\n        getattr(dtype, "categories", None) is not None\n        and bool(getattr(dtype, "ordered", False))\n        and getattr(candidate, "codes", None) is not None\n    ):\n        return candidate\n    return None\n\n\ndef _align_formula_side_array(values, design_info, expected_n=None, name="array"):\n    """Align an observation-level side array with rows retained by Patsy."""\n    if values is None:\n        return None\n\n    categorical = _ordered_categorical_array(values)\n    if categorical is None:\n        arr = np.asarray(values)\n        if arr.ndim == 0:\n            raise ValueError(f"{name} must be observation-level")\n        n_values = int(arr.shape[0])\n    else:\n        arr = None\n        n_values = int(len(categorical))\n\n    positions = getattr(design_info, "_statgpu_row_positions", None)\n    if positions is None:\n        if expected_n is not None and n_values != expected_n:\n            raise ValueError(f"{name} must have {expected_n} observations")\n        return categorical if categorical is not None else arr\n\n    positions = np.asarray(positions, dtype=np.int64)\n    if n_values == positions.shape[0]:\n        return categorical if categorical is not None else arr\n    if positions.size and n_values > int(positions.max()):\n        if categorical is not None:\n            return categorical.take(positions)\n        return arr[positions]\n    if positions.size == 0 and n_values == 0:\n        return categorical if categorical is not None else arr\n    raise ValueError(f"{name} has {n_values} observations and cannot be aligned to the {positions.shape[0]} rows retained by the formula")\n''',
)

# Regression coverage for direct DK and formula alignment.
replace_once(
    "dev/tests/test_panel_stage_c_edge_contracts.py",
    '''from statgpu.panel import driscoll_kraay_covariance\n''',
    '''from statgpu.panel import PooledOLS, driscoll_kraay_covariance\n''',
)

p = Path("dev/tests/test_panel_stage_c_edge_contracts.py")
text = p.read_text(encoding="utf-8")
append = r'''


def test_dk_preserves_ordered_categorical_chronology():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(12951)
    labels = np.tile(np.array(["t1", "t2", "t10"], dtype=object), 9)
    numeric = np.tile(np.arange(3), 9)
    ordered = pd.Categorical(
        labels,
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    X = np.column_stack([np.ones(labels.size), rng.normal(size=labels.size)])
    resid = rng.normal(size=labels.size)

    actual = driscoll_kraay_covariance(
        X, resid, ordered, bandwidth=1, kernel="bartlett"
    )
    expected = driscoll_kraay_covariance(
        X, resid, numeric, bandwidth=1, kernel="bartlett"
    )
    lexical = driscoll_kraay_covariance(
        X, resid, np.asarray(ordered, dtype=object), bandwidth=1, kernel="bartlett"
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)
    assert not np.allclose(actual, lexical, rtol=1e-10, atol=1e-12)


def test_dk_ordered_categorical_rejects_missing_codes():
    pd = pytest.importorskip("pandas")
    labels = pd.Categorical(
        ["t1", "t2", None, "t10"],
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    X = np.column_stack([np.ones(4), np.arange(4.0)])
    resid = np.linspace(-0.2, 0.3, 4)
    with pytest.raises(ValueError, match="must not contain missing or non-finite values"):
        driscoll_kraay_covariance(X, resid, labels, bandwidth=1)


def test_pooled_formula_dk_preserves_ordered_categorical_after_row_alignment():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(12952)
    labels = np.tile(np.array(["t1", "t2", "t10"], dtype=object), 12)
    numeric = np.tile(np.arange(3), 12)
    ordered = pd.Categorical(
        labels,
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    x = rng.normal(size=labels.size)
    y = 0.4 + 0.7 * x + rng.normal(scale=0.2, size=labels.size)
    x_with_gap = x.copy()
    x_with_gap[5] = np.nan
    data = pd.DataFrame({"y": y, "x": x_with_gap})

    categorical_fit = PooledOLS(cov_type="dk", bandwidth=1).fit(
        formula="y ~ x", data=data, time_index=ordered
    )
    numeric_fit = PooledOLS(cov_type="dk", bandwidth=1).fit(
        formula="y ~ x", data=data, time_index=numeric
    )

    np.testing.assert_allclose(
        categorical_fit.coef_, numeric_fit.coef_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        categorical_fit.bse_, numeric_fit.bse_, rtol=2e-13, atol=2e-15
    )
    assert categorical_fit._covariance_metadata["n_periods"] == 3
'''
if "test_dk_preserves_ordered_categorical_chronology" in text:
    raise SystemExit("Stage-C categorical chronology tests already present")
p.write_text(text + append, encoding="utf-8")

# Refresh EN/CN model docs and document time-label ordering semantics.
replace_once(
    "docs/en/models/panel.md",
    "> Last updated: 2026-08-09",
    "> Last updated: 2026-08-10",
)
replace_once(
    "docs/en/models/panel.md",
    "Stage C of the Tier-1 panel roadmap completes the residual-sandwich covariance layer on top of the Stage-B diagnostics: historical defaults remain unchanged, while HC0/HC2/HC3, robust RandomEffects inference, explicit cluster group debiasing, and Driscoll-Kraay covariance are added with NumPy/CuPy/Torch-native accumulation. Final physical CUDA acceptance for this PR remains a separate evidence gate until the exact-head artifact is recorded.",
    "Stage C of the Tier-1 panel roadmap completes the residual-sandwich covariance layer on top of the Stage-B diagnostics: historical defaults remain unchanged, while HC0/HC2/HC3, robust RandomEffects inference, explicit cluster group debiasing, and Driscoll-Kraay covariance are added with NumPy/CuPy/Torch-native accumulation. Physical CUDA acceptance is recorded from exact-clean-head Tesla P100 runs: all 26 estimator cases plus two direct public covariance primitives pass on both CuPy and Torch, together with synchronized performance evidence including the bounded high-T QS scenario.",
)
replace_once(
    "docs/en/models/panel.md",
    "`bandwidth=None` uses `floor(4*(T/100)^(2/9))`, where `T` is the number of distinct observed periods. Bartlett/Newey-West and Parzen/Gallant are truncated at the bandwidth. Quadratic Spectral (`qs`, Andrews) treats bandwidth as a smoothing scale and applies weights to **all observed lags** when bandwidth is positive; it is not truncated at `bw`.",
    "`bandwidth=None` uses `floor(4*(T/100)^(2/9))`, where `T` is the number of distinct observed periods. Bartlett/Newey-West and Parzen/Gallant are truncated at the bandwidth. Quadratic Spectral (`qs`, Andrews) treats bandwidth as a smoothing scale and applies weights to **all observed lags** when bandwidth is positive; it is not truncated at `bw`. Numeric and datetime time keys use their natural sorted order. An ordered pandas categorical preserves its declared category chronology, restricted to observed categories. Plain string/object labels retain deterministic sorted-label ordering; when chronological order differs from lexical order, pass an ordered categorical or an explicit numeric/datetime time key.",
)
replace_once(
    "docs/en/models/panel.md",
    "Hosted Stage-C tests pin HC2/HC3 against analytic/statsmodels fit-space calculations and cluster/Driscoll-Kraay definitions against `linearmodels==7.0`. The exact-head CuPy/Torch physical correctness and performance artifacts are a separate acceptance gate and must be recorded before PR #126 is promoted from Draft.",
    "Hosted Stage-C tests pin HC2/HC3 against analytic/statsmodels fit-space calculations and cluster/Driscoll-Kraay definitions against `linearmodels==7.0`. Exact-clean-head Tesla P100 acceptance is recorded for both CuPy and Torch: 26/26 estimator covariance cases plus 2/2 direct public covariance primitives per backend, with requested/executed backend identity and no CPU fallback. The synchronized performance artifact also covers the explicit `N=10,000`, `k=2`, `T=200` QS all-lag scenario; it does not encode a speedup or CPU-baseline claim.",
)

replace_once(
    "docs/cn/models/panel.md",
    "> 最后更新：2026-08-09",
    "> 最后更新：2026-08-10",
)
replace_once(
    "docs/cn/models/panel.md",
    "Tier-1 Panel 路线的 Stage C 在 Stage-B diagnostics 之上补齐 residual-sandwich covariance 层：历史默认行为保持不变，同时加入 HC0/HC2/HC3、RandomEffects robust inference、显式 cluster group debias 与 Driscoll-Kraay，并保持 NumPy/CuPy/Torch 数值累积后端原生。当前 PR 的最终 physical CUDA acceptance 仍需由 exact-head 机器产物单独闭合。",
    "Tier-1 Panel 路线的 Stage C 在 Stage-B diagnostics 之上补齐 residual-sandwich covariance 层：历史默认行为保持不变，同时加入 HC0/HC2/HC3、RandomEffects robust inference、显式 cluster group debias 与 Driscoll-Kraay，并保持 NumPy/CuPy/Torch 数值累积后端原生。physical CUDA acceptance 已由 exact-clean-head Tesla P100 产物闭合：CuPy 与 Torch 均通过 26 个 estimator case 和 2 个 direct public covariance primitive，并记录了包含 bounded high-T QS 场景的同步 performance evidence。",
)
replace_once(
    "docs/cn/models/panel.md",
    "`bandwidth=None` 使用 `floor(4*(T/100)^(2/9))`，其中 `T` 是 observed distinct time period 数。Bartlett/Newey-West 与 Parzen/Gallant 在 bandwidth 处截断。Quadratic Spectral（`qs`、Andrews）把 bandwidth 当作平滑尺度；bandwidth 为正时对**所有 observed lag**赋权，而不是在 `bw` 截断。",
    "`bandwidth=None` 使用 `floor(4*(T/100)^(2/9))`，其中 `T` 是 observed distinct time period 数。Bartlett/Newey-West 与 Parzen/Gallant 在 bandwidth 处截断。Quadratic Spectral（`qs`、Andrews）把 bandwidth 当作平滑尺度；bandwidth 为正时对**所有 observed lag**赋权，而不是在 `bw` 截断。numeric 与 datetime time key 按自然排序处理；ordered pandas categorical 保留显式声明的 category chronology，并仅压缩实际 observed categories。普通 string/object label 仍采用 deterministic sorted-label order；若时间顺序与字典序不同，应传入 ordered categorical 或显式 numeric/datetime time key。",
)
replace_once(
    "docs/cn/models/panel.md",
    "hosted Stage-C tests 已将 HC2/HC3 与 analytic/statsmodels fit-space 计算对齐，并将 cluster/Driscoll-Kraay definition 与 `linearmodels==7.0` 固定版本对齐。exact-head CuPy/Torch physical correctness 与 performance artifact 是独立 acceptance gate；PR #126 在这些产物闭合前继续保持 Draft。",
    "hosted Stage-C tests 已将 HC2/HC3 与 analytic/statsmodels fit-space 计算对齐，并将 cluster/Driscoll-Kraay definition 与 `linearmodels==7.0` 固定版本对齐。exact-clean-head Tesla P100 acceptance 已对 CuPy 与 Torch 闭合：每个 backend 均通过 26/26 estimator covariance case 与 2/2 direct public covariance primitive，并验证 requested/executed backend identity、无 CPU fallback。同步 performance artifact 还覆盖显式 `N=10,000`、`k=2`、`T=200` 的 QS all-lag 场景；其中不编码 speedup 或 CPU-baseline claim。",
)

print("PR126 Codex review patch applied")
