from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"missing anchor in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_after(path, marker, addition):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"missing marker in {path}: {marker!r}")
    p.write_text(text.replace(marker, marker + addition, 1), encoding="utf-8")


# English covariance: match the implementation's actual convergence contract.
replace(
    "docs/en/models/covariance.md",
    "`GraphicalLasso` and `GraphicalLassoCV` have two convergence-related parameters: `max_iter` (outer iterations) and `tol` (dual gap tolerance). The inner coordinate descent for each feature uses a fixed 100-iteration cap with tolerance \\(10^{-6}\\).",
    "`GraphicalLasso` and `GraphicalLassoCV` have two convergence-related parameters: `max_iter` (outer iterations) and `tol` (maximum absolute covariance-update tolerance). The inner coordinate descent uses a 1000-iteration cap and tolerance `min(1e-8, 0.1 * tol)`.",
)

# Chinese covariance: remove the remaining three-estimator-era language.
replace(
    "docs/cn/models/covariance.md",
    "- **EmpiricalCovariance**：样本协方差 \\(\\hat{S} = X^\\top X / n\\) 直接计算。精度矩阵 \\(\\hat{S}^{-1}\\) 通过抖动稳定矩阵求逆获得（当矩阵接近奇异时逐步增加对角增量）。",
    "- **EmpiricalCovariance**：样本协方差 \\(\\hat{S} = X^\\top X / n\\) 直接计算。先尝试精确求逆；仅当求逆失败或结果非有限时才逐步增加对角 jitter。",
)
replace(
    "docs/cn/models/covariance.md",
    "- `shrinkage_`：收缩强度 \\(\\alpha\\)，取值范围 \\([0, 1]\\) 的浮点数（仅 LedoitWolf 和 OAS）。",
    "- `shrinkage_`：收缩强度 \\(\\alpha\\)，取值范围 \\([0, 1]\\) 的浮点数（LedoitWolf、OAS 与 ShrunkCovariance）。",
)
replace(
    "docs/cn/models/covariance.md",
    "协方差估计器没有单独的 strict 或 approx 模式。三种估计器均使用直接解析公式，无迭代求解器，因此无需调节收敛容差。",
    "协方差估计器没有单独的 strict 或 approx 模式。经验/收缩估计器使用直接公式；MinCovDet 使用内部 C-step；GraphicalLasso/CV 使用 `max_iter` 和以协方差最大变化量定义的 `tol`。",
)

# English panel: FamaMacBeth does not expose an rsquared attribute.
replace(
    "docs/en/models/panel.md",
    "| `rsquared` | scalar | R-squared (PooledOLS, BetweenOLS, FirstDifferenceOLS, FamaMacBeth) |",
    "| `rsquared` | scalar | R-squared (PooledOLS, BetweenOLS, FirstDifferenceOLS) |",
)

# Chinese panel: complete the six-model inventory and backend example.
insert_after(
    "docs/cn/models/panel.md",
    "- `statgpu.panel.two_way_clustered_covariance`\n",
    "- `statgpu.panel.hac_covariance`\n",
)
insert_after(
    "docs/cn/models/panel.md",
    "`RandomEffects` 默认在准去均值数据上使用非稳健 OLS 推断。\n",
    "\n`PooledOLS` 支持 `nonrobust`、`robust`、`clustered` 与 Bartlett HAC；"
    "`BetweenOLS` 支持 `nonrobust`、`robust`、`clustered`；"
    "`FirstDifferenceOLS` 支持 `nonrobust` 与 `robust`；"
    "`FamaMacBeth` 支持 `nonrobust` 或对系数时间序列使用 `newey-west`。\n",
)
replace(
    "docs/cn/models/panel.md",
    "`fit()` 后的输出：`coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared_within`（PanelOLS）。",
    "`fit()` 后的公共输出包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`；PanelOLS 另有 `rsquared_within`，Pooled/Between/FirstDifference 另有 `rsquared`，FamaMacBeth 另有 `betas_`、`cov_params_` 和 `n_periods`。",
)
replace(
    "docs/cn/models/panel.md",
    "fe_torch = PanelOLS(entity_effects=True, cov_type='robust', device='cuda')",
    "fe_torch = PanelOLS(entity_effects=True, cov_type='robust', device='torch')",
)
insert_after(
    "docs/cn/models/panel.md",
    "- `'clustered'`：聚类稳健标准误，允许组内任意相关。p 值使用正态分布。支持单向和双向聚类。\n",
    "- `'hac'` / `'newey-west'`：使用 Bartlett 权重的 Newey-West HAC；PooledOLS 作用于按时间排序的 score，FamaMacBeth 作用于分期系数路径。\n",
)
insert_after(
    "docs/cn/models/panel.md",
    "| `rsquared_within` | 标量 | 组内 R 方（仅 PanelOLS） |\n",
    "| `rsquared` | 标量 | R 方（PooledOLS、BetweenOLS、FirstDifferenceOLS） |\n",
)
insert_after(
    "docs/cn/models/panel.md",
    "| `variance_components_` | dict | `{'sigma2_e': float, 'sigma2_a': float}`（仅 RandomEffects） |\n",
    "| `betas_` | `(T, k)` | 每期横截面系数路径（仅 FamaMacBeth） |\n| `cov_params_` | `(k, k)` | 系数均值的协方差（仅 FamaMacBeth） |\n| `n_periods` | int | 纳入的时期数（仅 FamaMacBeth） |\n",
)
insert_after(
    "docs/cn/models/panel.md",
    "| `fit(y, X, entity_ids, ...)` | `self` | 拟合面板模型。需要 `entity_ids`（一维个体标签数组）。可选：`time_ids`、`cluster`。 |\n",
    "| `fit(X, y, ...)` | `self` | 拟合 PooledOLS、BetweenOLS、FirstDifferenceOLS 或 FamaMacBeth；所需 entity/time/cluster 参数见上文。 |\n",
)

# Chinese spline outputs and parity wording.
replace(
    "docs/cn/models/splines.md",
    "样条基计算没有 strict/approx 模式区分。De Boor 递归是确定性算法，在所有后端（NumPy、CuPy、Torch）上产生相同结果（浮点精度范围内）。",
    "样条基计算没有 strict/approx 模式。NumPy、CuPy 与 Torch 使用同一递推；已验证 NumPy/Torch-CPU 紧容差一致性，但真实 CUDA parity 与性能仍待验证。",
)
insert_after(
    "docs/cn/models/splines.md",
    "**natural_cubic_spline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + 1)`。\n",
    "\n**cyclic_cubic_spline_basis**：返回满足周期边界约束的三次样条基。\n\n"
    "**thin_plate_spline_basis**：返回径向基与低阶多项式列组成的矩阵。\n\n"
    "**SplineTransformer**：`fit()` 后提供 `knots_`、`boundary_lo_`、`boundary_hi_`、"
    "`n_features_in_` 和 `n_features_out_`；`transform()` 返回与输入/所选后端一致的数组。\n",
)

# Chinese ANOVA: document the additional result contracts.
insert_after(
    "docs/cn/models/anova.md",
    "| `eta_squared` | float | 效应量 $\\eta^2$ |\n",
    "\nWelch ANOVA 的 `df_within` 为 Welch-Satterthwaite 小数自由度；"
    "`TwoWayAnovaResult` 分别报告 factor A、factor B、interaction 与 residual 项；"
    "`TukeyResult`/`PosthocResult` 返回每一对组别的均值差、p 值、置信区间和拒绝标记。\n",
)

print("final PR79 documentation consistency pass completed")
