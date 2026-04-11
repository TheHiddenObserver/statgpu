# Knockoff 特征选择

> 语言: 中文  
> 最后更新: 2026-04-11  
> 页面定位: 方法文档  
> 切换: [English](../en/models/knockoff.md)

语言切换：[English](../en/models/knockoff.md)

路径：
- statgpu.feature_selection.knockoff_filter
- statgpu.feature_selection.fixed_x_knockoff_filter
- statgpu.feature_selection.model_x_knockoff_filter
- statgpu.feature_selection.KnockoffSelector
- statgpu.feature_selection.FixedXKnockoffSelector

顶层别名：
- statgpu.knockoff_filter
- statgpu.fixed_x_knockoff_filter
- statgpu.model_x_knockoff_filter
- statgpu.KnockoffSelector
- statgpu.FixedXKnockoffSelector

## 方法概览

当前实现提供两类 knockoff 路径：

- fixed-X knockoff
  - 适用于设计矩阵可视为固定的场景。
  - 约束：通常要求 n >= 2p，否则无法构造 fixed-X knockoff。
- model-X knockoff
  - 基于高斯二阶近似（协方差估计 + S-matrix）构造 knockoff。
  - 支持多次 draw 后对 W 统计量做平均。

统一入口为 knockoff_filter，通过参数 knockoff_type 在 fixed_x/model_x 之间切换。

## 统一入口参数（knockoff_filter）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| knockoff_type | fixed_x | fixed_x 或 model_x |
| q | 0.1 | FDR 控制目标，必须在 (0, 1) |
| method | corr_diff | W 统计量：corr_diff / ols_coef_diff / lasso_coef_diff |
| fdr_control | knockoff_plus | 阈值规则：knockoff_plus 或 knockoff |
| random_state | None | 随机种子 |
| backend | auto | auto / numpy / cupy |
| Xk | None | 外部提供 knockoff 设计矩阵（形状必须与 X 相同） |
| compat_mode | statgpu | statgpu 或 knockpy |
| lasso_cv_impl | auto | auto / statgpu / sklearn |
| lasso_fast_profile | off | lasso 快速配置开关 |
| modelx_covariance_shrinkage | 0.20 | model-X 协方差收缩系数 |
| modelx_s_scale | 0.999 | model-X S 缩放系数 |
| modelx_draws | None | model-X draw 次数（None 时按统计量给默认） |
| modelx_shrinkage | ledoitwolf | knockpy 兼容路径下的协方差估计策略 |
| modelx_smatrix_method | mvr | knockpy 兼容路径下的 S-matrix 方法 |
| knockpy_sampler | None | 可选分发入口（gaussian/fx/metro/artk 等） |
| knockpy_sampler_method | None | gaussian 分发子方法（mvr/sdp/maxent/equi/ci） |

## 返回对象（KnockoffResult）

fit/过滤函数返回 KnockoffResult，核心字段：

- selected_features: 被选中的特征下标
- W: 每个特征对应的 W 统计量
- threshold: 当前 q 对应阈值
- estimated_fdr: 估计 FDR
- q_trajectory: 阈值扫描轨迹
- metadata: 运行元数据（draw 次数、兼容模式、Xk 来源等）

## 快速示例

### 1) fixed-X 直接调用

```python
from statgpu import fixed_x_knockoff_filter

res = fixed_x_knockoff_filter(
    X,
    y,
    q=0.1,
    method="ols_coef_diff",
    backend="auto",
)

print(res.selected_features)
print(res.threshold, res.estimated_fdr)
```

### 2) model-X 统一入口

```python
from statgpu import knockoff_filter

res = knockoff_filter(
    X,
    y,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    compat_mode="knockpy",
    modelx_draws=3,
)
```

### 3) sklearn 风格选择器

```python
from statgpu import KnockoffSelector

selector = KnockoffSelector(
    knockoff_type="model_x",
    q=0.1,
    method="corr_diff",
    backend="auto",
)
selector.fit(X, y)
X_sel = selector.transform(X)
mask = selector.get_support()
```

## 统一分发接口说明（新增）

当前已提供 knockpy_sampler_dispatch 统一分发接口，并在 model_x 路径下支持参数透传：

- knockpy_sampler: gaussian / gaussian_mvr / gaussian_sdp / gaussian_maxent / gaussian_equi / gaussian_ci / fx / metro / artk
- knockpy_sampler_method: 当 sampler=gaussian 时可选 mvr/sdp/maxent/equi/ci

注意：
- 目前这些分发目标接口仍处于占位阶段（函数体 pass）。
- 若显式传入 knockpy_sampler，会触发 NotImplementedError（用于避免静默走错路径）。
- 不传该参数时，仍走当前稳定实现路径。

## 约束与排错

- q 不在 (0, 1) 会报错。
- X 必须为二维，y 行数需与 X 一致。
- 传入 Xk 时，Xk 形状必须与 X 相同。
- fixed-X 对样本数与秩有要求，若不满足会报错。
- cupy 后端需要环境中已安装 CuPy。

## 相关脚本

- dev/benchmarks/benchmark_knockoff_fixedx.py
- dev/benchmarks/benchmark_knockoff_vs_baselines.py
- dev/benchmarks/benchmark_knockoff_same_xk_parity.py
