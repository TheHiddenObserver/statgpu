"""One-shot cleanup for fresh PR126 FE-df review metadata/docs."""
from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '    """Return rank-consistent Stage-B FE diagnostic df without changing legacy df."""\n',
    '    """Return the standard rank-consistent fixed-effect df decomposition."""\n',
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '        "legacy_df_unchanged": True,\n',
    '        "legacy_df_unchanged": False,\n        "public_df_uses_standard_effect_rank": True,\n',
)

replace_once(
    "dev/tests/test_panel_stage_c_fresh_df_rank_guard.py",
    '    assert model.fit_statistics_.metadata["public_df_resid_basis"] == "standard"\n',
    '    assert model.fit_statistics_.metadata["public_df_resid_basis"] == "standard"\n    diagnostic_df = model.fit_statistics_.metadata["diagnostic_df"]\n    assert diagnostic_df["legacy_df_unchanged"] is False\n    assert diagnostic_df["public_df_uses_standard_effect_rank"] is True\n',
)

replace_once(
    "docs/en/panel/covariance.md",
    '''HC2 and HC3 require $1-h_i$ to be numerically positive. If an observation has leverage effectively equal to 1, these corrections are undefined and statgpu raises an error rather than returning an infinite or unstable variance.\n''',
    '''HC2 and HC3 require $1-h_i$ to be numerically positive. For a full-rank estimator fit or a direct covariance-primitive call, an observation with leverage effectively equal to 1 therefore raises rather than returning an infinite or unstable variance. If the estimator fit-space is already rank deficient, coefficient-level inference is unavailable regardless of covariance type; in that case statgpu keeps the fitted values and does not attempt an HC2/HC3 coordinate covariance that can be undefined at unit leverage.\n''',
)
replace_once(
    "docs/cn/panel/covariance.md",
    '''HC2/HC3 要求 $1-h_i$ 在数值上为正。如果某个 observation 的 leverage 实际等于 1，这两个 correction 无法定义，statgpu 会报错，而不是返回无穷大或数值不稳定的 variance。\n''',
    '''HC2/HC3 要求 $1-h_i$ 在数值上为正。对于 full-rank estimator fit 或直接调用 covariance primitive，如果某个 observation 的 leverage 在数值上等于 1，statgpu 会直接报错，而不是返回无穷大或不稳定的 variance。若 estimator 的 fit-space 本身已经 rank deficient，则 coefficient-level inference 无论选择哪种 covariance 都不可用；此时 statgpu 保留 fitted values，并且不会再强行构造可能在 unit leverage 下无定义的 HC2/HC3 coordinate covariance。\n''',
)
