from __future__ import annotations

from pathlib import Path


def replace_section(path: str, start: str, end: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if text.count(start) != 1:
        raise RuntimeError(f"{path}: start marker is not unique: {start!r}")
    lo = text.index(start)
    hi = text.find(end, lo + len(start))
    if hi < 0:
        raise RuntimeError(
            f"{path}: end marker not found after start: {end!r}"
        )
    p.write_text(text[:lo] + replacement + text[hi:], encoding="utf-8")


replace_section(
    "docs/en/models/elastic-net.md",
    "## Parameters\n",
    "## CPU/GPU Examples\n",
    '''## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.0` | Overall regularization strength |
| `l1_ratio` | `0.5` | L1 mixing proportion: 0=Ridge, 1=Lasso |
| `fit_intercept` | `True` | Fit an unpenalized intercept |
| `max_iter` | `1000` | Maximum solver iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `stopping` | `"coef_delta"` | `"coef_delta"` or `"kkt"` stopping rule |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` |
| `n_jobs` | `None` | CPU parallelism where supported |
| `solver` | `"fista"` | Backend-aware optimization method |
| `cpu_solver` | `"fista"` | CPU solver override |
| `lipschitz_L` | `None` | Optional user-supplied Lipschitz constant |
| `gpu_memory_cleanup` | `False` | Release backend memory pools after fit where supported |
| `compute_inference` | `False` | Compute post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"cpu_ols"`, or `"bootstrap"` |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where supported |

The public wrapper does not accept separate `backend`, `warm_start`, or
`random_state` constructor parameters. Backend selection is controlled by
`device`; a one-fit warm start can be supplied through `fit(initial_coef=...)`.

''',
)

replace_section(
    "docs/en/models/elastic-net.md",
    "## strict/approx difference\n",
    "## Outputs\n",
    '''## Solver and Inference Semantics

The default estimator uses FISTA for the declared Elastic Net objective. The
`stopping` option changes only the convergence diagnostic (`coef_delta` versus
KKT violation); it does not define a separate statistical approximation mode.

`compute_inference=False` returns the penalized estimate only. With
`compute_inference=True`, the same fitted coefficients are retained and the
selected post-fit inference method is run afterward. The standalone
`ElasticNet` wrapper and the final full-data refit of `ElasticNetCV` both support
this contract directly; users do not need to switch estimator classes merely
to request debiased inference.

''',
)

replace_section(
    "docs/cn/models/elastic-net.md",
    "## 参数\n",
    "## CPU/GPU 示例\n",
    '''## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | `1.0` | 总体正则化强度 |
| `l1_ratio` | `0.5` | L1 混合比例：0=Ridge，1=Lasso |
| `fit_intercept` | `True` | 拟合不受惩罚的截距 |
| `max_iter` | `1000` | 最大求解迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `stopping` | `"coef_delta"` | `"coef_delta"` 或 `"kkt"` 停止准则 |
| `device` | `"auto"` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"` |
| `n_jobs` | `None` | 适用 CPU 路径的并行度 |
| `solver` | `"fista"` | 后端感知的优化方法 |
| `cpu_solver` | `"fista"` | CPU 求解器覆盖选项 |
| `lipschitz_L` | `None` | 可选的用户指定 Lipschitz 常数 |
| `gpu_memory_cleanup` | `False` | 在支持的后端上于拟合后释放内存池 |
| `compute_inference` | `False` | 计算拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 适用方法中的协方差约定 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的滞后阶数 |

公开 wrapper 不接受单独的 `backend`、`warm_start` 或 `random_state`
构造参数。后端由 `device` 控制；单次拟合的 warm start 可通过
`fit(initial_coef=...)` 提供。

''',
)

replace_section(
    "docs/cn/models/elastic-net.md",
    "## strict/approx 区别\n",
    "## 输出属性\n",
    '''## 求解器与推断语义

默认估计器使用 FISTA 优化声明的 Elastic Net 目标函数。`stopping` 仅改变
收敛诊断（`coef_delta` 或 KKT violation），并不定义不同的统计近似模式。

`compute_inference=False` 只返回 penalized estimate。设置
`compute_inference=True` 后，原拟合系数保持不变，并在拟合完成后运行所选推断方法。
独立 `ElasticNet` wrapper 与 `ElasticNetCV` 的最终全数据重拟合都直接支持该契约；
用户不需要仅为了 debiased inference 而切换到其他估计器类。

''',
)

for changelog, bullet in (
    (
        "CHANGELOG.md",
        "- Reconciled the ElasticNet API documentation with the implementation by correcting constructor defaults, removing nonexistent parameters, and replacing stale strict/approx guidance with the actual FISTA and post-fit inference semantics.\n",
    ),
    (
        "docs/en/changelog.md",
        "- Reconciled the ElasticNet API documentation with the implementation by correcting constructor defaults, removing nonexistent parameters, and replacing stale strict/approx guidance with the actual FISTA and post-fit inference semantics.\n",
    ),
    (
        "docs/cn/changelog.md",
        "- 统一 ElasticNet API 文档与实现：修正构造参数默认值、删除不存在的参数，并用实际 FISTA 与拟合后推断语义替换过时的 strict/approx 说明。\n",
    ),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
