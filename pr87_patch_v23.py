from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"documentation anchor missing in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "CHANGELOG.md",
    '''- Addressed Issue #81 with backend-native finite-value validation at public
  estimator boundaries without full GPU-array transfers.
- Addressed Issue #82 by preserving exact raw constructor arguments for
''',
    '''- Addressed Issue #81 with backend-native finite-value validation at public
  estimator boundaries without full GPU-array transfers.
- Aligned formula sample weights after missing-row filtering across linear,
  GLM, and penalized estimators; retained Torch/CuPy weights on device; and
  corrected Gaussian GLM FISTA to use weighted centering and the intended
  weighted squared-loss intercept.
- Addressed Issue #82 by preserving exact raw constructor arguments for
''',
)

replace_once(
    "docs/en/changelog.md",
    "> Last updated: 2026-08-04<br>",
    "> Last updated: 2026-08-05<br>",
)
replace_once(
    "docs/en/changelog.md",
    '''  and panel identifiers while preserving formula-owned missing-row semantics.

### Estimator and test contracts
''',
    '''  and panel identifiers while preserving formula-owned missing-row semantics.
- Formula sample weights are aligned only after Patsy selects retained rows,
  then checked for shape, finite values, non-negativity, and positive total
  weight. Torch and CuPy alignment and inference weights remain device-native.
- Gaussian GLM FISTA now profiles the intercept with weighted feature and
  response means, matching the declared weighted squared-loss objective and
  closed-form weighted least squares when the penalty is zero.

### Estimator and test contracts
''',
)

replace_once(
    "docs/cn/changelog.md",
    "> 最后更新：2026-08-04<br>",
    "> 最后更新：2026-08-05<br>",
)
replace_once(
    "docs/cn/changelog.md",
    '''  fit/predict/transform、inverse-transform、scoring、初始化数组和 panel ID，
  同时保留 formula 路径对缺失行的专属语义。

### Estimator 与测试契约
''',
    '''  fit/predict/transform、inverse-transform、scoring、初始化数组和 panel ID，
  同时保留 formula 路径对缺失行的专属语义。
- formula sample weight 在 Patsy 确定保留行之后才进行对齐，并检查一维形状、
  finite、非负性与正权重和；Torch/CuPy 的对齐及 inference 权重保持在设备端。
- Gaussian GLM 的 FISTA 路径改用加权的特征均值与响应均值 profile intercept；
  在零惩罚时与闭式 weighted least squares 一致，不再优化错误的未加权中心化目标。

### Estimator 与测试契约
''',
)
