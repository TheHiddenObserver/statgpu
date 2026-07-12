from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return (ROOT / path).read_text(encoding="utf-8")


def save(path, text):
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(path, old, new):
    text = load(path)
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"missing replacement anchor in {path}: {old[:80]!r}")
    save(path, text.replace(old, new, 1))


def insert_after(path, marker, block):
    text = load(path)
    if block.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"missing insertion anchor in {path}: {marker!r}")
    save(path, text.replace(marker, marker + block, 1))


def insert_before(path, marker, block):
    text = load(path)
    if block.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"missing insertion anchor in {path}: {marker!r}")
    save(path, text.replace(marker, block + marker, 1))


# ---------------------------------------------------------------------------
# Changelogs
# ---------------------------------------------------------------------------
insert_after(
    "CHANGELOG.md",
    "## 2026-07-12\n\n",
    """### PR #79 — Native three-backend execution follow-up

- Removed complete numeric-array NumPy fallbacks from `GraphicalLasso`,
  `GraphicalLassoCV`, `MinCovDet`, `SplineTransformer`, and `FamaMacBeth`.
- Kept Graphical Lasso block-coordinate descent/CV, FAST-MCD C-steps and
  reweighting, spline Cox–de Boor recurrence, and Fama–MacBeth regressions/HAC
  covariance on the selected NumPy, CuPy, or Torch backend.
- Kept Tukey/Bonferroni group reductions on-device; only scalar distribution
  CDF/quantile evaluations cross the CPU boundary.
- Added NumPy/Torch parity and backend-preservation tests plus optional CuPy CUDA
  checks. Physical CuPy/Torch CUDA memory, runtime, convergence, and repeated-fit
  validation remains `PARTIAL_REMOTE_PENDING`.
- Synchronized README, bilingual implemented-method lists, model pages, and all
  three changelogs with the corrected execution and validation boundaries.

""",
)

insert_after(
    "docs/en/changelog.md",
    "## 2026-07\n\n",
    """### Improved (2026-07-12) — PR #79 native three-backend execution

- Replaced complete-design NumPy fallbacks in Graphical Lasso/CV, MinCovDet,
  SplineTransformer, and Fama–MacBeth with NumPy/CuPy/Torch-native core
  numerical paths.
- Kept post-hoc group reductions on the selected backend and restricted SciPy
  use to scalar studentized-range/t distribution evaluations.
- Added NumPy/Torch parity, output-backend, and source-boundary regression tests;
  optional CuPy checks run only when a CUDA runtime is available.
- Updated public README, bilingual method inventories, and ANOVA/covariance/panel/
  spline model pages. Physical CUDA validation remains pending.

""",
)

insert_after(
    "docs/cn/changelog.md",
    "## 2026-07\n\n",
    """### 改进（2026-07-12）— PR #79 原生三后端执行

- 移除 Graphical Lasso/CV、MinCovDet、SplineTransformer 与 Fama–MacBeth
  对完整数值设计矩阵的 NumPy 回退，使核心计算保留在 NumPy、CuPy 或 Torch 后端。
- 事后检验的组内归约保留在所选后端，仅将 studentized-range/t 分布的标量
  CDF/分位数计算交给 SciPy。
- 新增 NumPy/Torch 数值一致性、输出后端与源码边界测试；只有存在 CUDA runtime
  时才运行可选 CuPy 检查。
- 同步根 README、中英文方法清单以及 ANOVA、协方差、面板和样条模型页。
  真实 CUDA 数值、显存、性能与重复拟合验证仍待完成。

""",
)

# ---------------------------------------------------------------------------
# Root README and documentation indexes
# ---------------------------------------------------------------------------
replace_once(
    "README.md",
    "- 🚀 **3 Backends**: NumPy (CPU), CuPy (CUDA), PyTorch (CUDA) — automatic device selection",
    "- 🚀 **3 Backends**: NumPy (CPU), CuPy (CUDA), PyTorch (CUDA) — automatic device selection\n"
    "- 🧭 **Backend transparency**: core numerical paths preserve backend arrays; intentional CPU boundaries are limited to formula/label metadata and unsupported scalar distribution functions",
)
replace_once(
    "README.md",
    "| **ANOVA** | 2 functions | `f_oneway`, `f_twoway` — GPU-accelerated |\n"
    "| **Covariance** | 3 classes | EmpiricalCovariance, LedoitWolf, OAS |\n"
    "| **Panel Data** | 2 classes | PanelOLS, RandomEffects |\n"
    "| **Nonparametric** | 5 classes | KernelRidge, KernelRidgeCV, pairwise_kernels, bspline_basis, natural_cubic_spline_basis |",
    "| **ANOVA** | 7 functions | `f_oneway`, `f_twoway`, `f_welch`, Tukey/Bonferroni post-hoc, effect sizes |\n"
    "| **Covariance** | 7 classes | Empirical/shrinkage covariance, MinCovDet, GraphicalLasso, GraphicalLassoCV |\n"
    "| **Panel Data** | 6 classes | PanelOLS, RandomEffects, PooledOLS, BetweenOLS, FirstDifferenceOLS, FamaMacBeth |\n"
    "| **Nonparametric** | 10+ classes/functions | KDE/kernel regression, KernelRidge/CV, KernelPCA, Nystroem, spline bases and SplineTransformer |",
)
insert_before(
    "README.md",
    "## Installation\n",
    """## Backend execution status

`GraphicalLasso`/`GraphicalLassoCV`, `MinCovDet`, `SplineTransformer`, and
`FamaMacBeth` now keep their main numerical computation on NumPy, CuPy, or Torch.
Tukey and Bonferroni keep group reductions on-device and synchronize only scalar
statistics for distributions not implemented by CuPy/Torch. NumPy/Torch-CPU parity
is covered by CI; physical CuPy CUDA and Torch CUDA convergence, memory, runtime,
and repeated-fit validation is still tracked as `PARTIAL_REMOTE_PENDING`.

""",
)

for path in ["docs/en/README.md", "docs/cn/README.md"]:
    replace_once(path, "[Panel](models/panel.md) — fixed/random effects panel models" if path.startswith("docs/en") else "[Panel](models/panel.md) — 固定/随机效应面板模型",
                 "[Panel](models/panel.md) — six panel estimators including pooled, between, first-difference, and Fama–MacBeth" if path.startswith("docs/en") else "[Panel](models/panel.md) — 六类面板估计器，含 pooled、between、first-difference 与 Fama–MacBeth")
    replace_once(path, "[Splines](models/splines.md) — B-spline basis, penalized splines" if path.startswith("docs/en") else "[样条](models/splines.md) — B 样条基、惩罚样条",
                 "[Splines](models/splines.md) — B/natural/cyclic/thin-plate splines and SplineTransformer" if path.startswith("docs/en") else "[样条](models/splines.md) — B/自然/周期/薄板样条与 SplineTransformer")
    replace_once(path, "[ANOVA](models/anova.md) — analysis of variance" if path.startswith("docs/en") else "[ANOVA](models/anova.md) — 方差分析",
                 "[ANOVA](models/anova.md) — one/two-way, Welch, post-hoc, and effect sizes" if path.startswith("docs/en") else "[ANOVA](models/anova.md) — 单/双因素、Welch、事后检验与效应量")
    replace_once(path, "[Covariance](models/covariance.md) — covariance estimation, shrinkage" if path.startswith("docs/en") else "[Covariance](models/covariance.md) — 协方差估计、收缩",
                 "[Covariance](models/covariance.md) — empirical/shrinkage, robust MCD, and sparse precision" if path.startswith("docs/en") else "[Covariance](models/covariance.md) — 经验/收缩、稳健 MCD 与稀疏精度矩阵")

# ---------------------------------------------------------------------------
# Bilingual implemented-method inventories
# ---------------------------------------------------------------------------
for path in ["docs/en/guides/implemented-methods.md", "docs/cn/guides/implemented-methods.md"]:
    replace_once(path, "> Last updated: 2026-06-14" if path.startswith("docs/en") else "> 最后更新：2026-06-14",
                 "> Last updated: 2026-07-12" if path.startswith("docs/en") else "> 最后更新：2026-07-12")

replace_once(
    "docs/en/guides/implemented-methods.md",
    "| `f_oneway` | GPU-accelerated one-way ANOVA |",
    "| `f_oneway` | One-way ANOVA |\n| `f_twoway` | Balanced two-way ANOVA, full or additive model |\n| `f_welch` | Welch one-way ANOVA with fractional denominator df |\n| `tukey_hsd` | Tukey HSD simultaneous post-hoc comparisons |\n| `bonferroni` | Bonferroni-adjusted pairwise Welch tests |\n| `cohens_f` | Cohen's f effect size |\n| `partial_eta_squared` | Partial eta-squared effect size |",
)
replace_once(
    "docs/cn/guides/implemented-methods.md",
    "| `f_oneway` | GPU-accelerated one-way ANOVA |",
    "| `f_oneway` | 单因素 ANOVA |\n| `f_twoway` | 平衡设计双因素 ANOVA（完整或加性模型） |\n| `f_welch` | Welch 单因素 ANOVA，保留小数分母自由度 |\n| `tukey_hsd` | Tukey HSD 同时事后比较 |\n| `bonferroni` | Bonferroni 校正的两两 Welch 检验 |\n| `cohens_f` | Cohen's f 效应量 |\n| `partial_eta_squared` | 偏 eta 平方效应量 |",
)

cov_extra_en = """| `ShrunkCovariance` | User-specified covariance shrinkage | CPU, CuPy, Torch |
| `MinCovDet` | Robust FAST-MCD covariance with backend-native C-steps | CPU, CuPy, Torch |
| `GraphicalLasso` | Sparse inverse covariance via block coordinate descent | CPU, CuPy, Torch |
| `GraphicalLassoCV` | Cross-validated Graphical Lasso | CPU, CuPy, Torch |"""
cov_extra_cn = """| `ShrunkCovariance` | 用户指定强度的协方差收缩 | CPU, CuPy, Torch |
| `MinCovDet` | 后端原生 C-step 的稳健 FAST-MCD | CPU, CuPy, Torch |
| `GraphicalLasso` | 块坐标下降稀疏逆协方差 | CPU, CuPy, Torch |
| `GraphicalLassoCV` | 交叉验证 Graphical Lasso | CPU, CuPy, Torch |"""
insert_after("docs/en/guides/implemented-methods.md", "| `OAS` | Oracle Approximating Shrinkage estimator | CPU, CuPy, Torch |\n", cov_extra_en + "\n")
insert_after("docs/cn/guides/implemented-methods.md", "| `OAS` | Oracle Approximating Shrinkage estimator | CPU, CuPy, Torch |\n", cov_extra_cn + "\n")

panel_extra_en = """| `PooledOLS` | Stacked OLS with robust/clustered/HAC covariance | CPU, CuPy, Torch |
| `BetweenOLS` | OLS on entity means | CPU, CuPy, Torch |
| `FirstDifferenceOLS` | Within-entity first-difference OLS | CPU, CuPy, Torch |
| `FamaMacBeth` | Per-period cross-sectional regressions with Newey-West inference | CPU, CuPy, Torch |"""
panel_extra_cn = """| `PooledOLS` | 堆叠 OLS，支持稳健/聚类/HAC 协方差 | CPU, CuPy, Torch |
| `BetweenOLS` | 个体均值上的 OLS | CPU, CuPy, Torch |
| `FirstDifferenceOLS` | 个体内一阶差分 OLS | CPU, CuPy, Torch |
| `FamaMacBeth` | 分期横截面回归与 Newey-West 推断 | CPU, CuPy, Torch |"""
insert_after("docs/en/guides/implemented-methods.md", "| `RandomEffects` | Swamy-Arora feasible GLS random effects | CPU, CuPy, Torch |\n", panel_extra_en + "\n")
insert_after("docs/cn/guides/implemented-methods.md", "| `RandomEffects` | Swamy-Arora feasible GLS random effects | CPU, CuPy, Torch |\n", panel_extra_cn + "\n")

nonparam_extra_en = """| `KernelPCA` | Centered-kernel principal component embedding |
| `Nystroem` | Low-rank kernel feature approximation via stable SVD normalization |
| `KernelDensity` / kernel regression | Backend-native kernel smoothing estimators |
| `cyclic_cubic_spline_basis` | Periodic cubic spline basis |
| `thin_plate_spline_basis` | Multi-dimensional thin-plate radial basis |
| `SplineTransformer` | sklearn-style backend-native B-spline transformer with four extrapolation modes |"""
nonparam_extra_cn = """| `KernelPCA` | 中心化核主成分嵌入 |
| `Nystroem` | 稳定 SVD 归一化的低秩核特征近似 |
| `KernelDensity` / 核回归 | 后端原生核平滑估计器 |
| `cyclic_cubic_spline_basis` | 周期三次样条基 |
| `thin_plate_spline_basis` | 多维薄板径向基 |
| `SplineTransformer` | 支持四种外推模式的后端原生 sklearn 风格 B 样条变换器 |"""
insert_after("docs/en/guides/implemented-methods.md", "| `natural_cubic_spline_basis` | Natural cubic spline basis |\n", nonparam_extra_en + "\n")
insert_after("docs/cn/guides/implemented-methods.md", "| `natural_cubic_spline_basis` | Natural cubic spline basis |\n", nonparam_extra_cn + "\n")

insert_before(
    "docs/en/guides/implemented-methods.md",
    "## Semiparametric Models\n",
    """### Backend execution boundary

Graphical Lasso/CV, MinCovDet, SplineTransformer, and Fama–MacBeth keep their
main numerical work on NumPy/CuPy/Torch. Formula and categorical-label parsing,
integer fold/subset metadata, and unsupported scalar distribution CDF/quantiles
remain intentional CPU boundaries. NumPy/Torch-CPU parity is tested; physical
CUDA validation remains pending.

""",
)
insert_before(
    "docs/cn/guides/implemented-methods.md",
    "## 半参数模型\n",
    """### 后端执行边界

Graphical Lasso/CV、MinCovDet、SplineTransformer 与 Fama–MacBeth 的主要数值
计算保留在 NumPy/CuPy/Torch 后端。formula 与分类标签解析、fold/subset 整数元数据，
以及后端缺失的标量分布 CDF/分位数计算仍是有意的 CPU 边界。已验证 NumPy 与
Torch-CPU 一致性；真实 CUDA 验证仍待完成。

""",
)

# ---------------------------------------------------------------------------
# English model pages
# ---------------------------------------------------------------------------
for path in [
    "docs/en/models/covariance.md",
    "docs/en/models/splines.md",
    "docs/en/models/panel.md",
    "docs/en/models/anova.md",
]:
    replace_once(path, "> Last updated: 2026-06-17", "> Last updated: 2026-07-12")

replace_once(
    "docs/en/models/covariance.md",
    "- \\alpha\\|\\Theta\\|_1",
    "- \\alpha\\|\\Theta\\|_{1,\\mathrm{off}}",
)
replace_once(
    "docs/en/models/covariance.md",
    "Convergence is checked via the dual gap.",
    "Convergence is checked by the maximum absolute covariance update between outer iterations; the precision diagonal is not L1-penalized.",
)
replace_once(
    "docs/en/models/covariance.md",
    "The precision matrix \\(\\hat{S}^{-1}\\) is obtained via jitter-stabilized matrix inversion (progressive diagonal augmentation if the matrix is near-singular).",
    "The precision matrix \\(\\hat{S}^{-1}\\) is computed by exact inversion first; progressive diagonal jitter is used only when the exact inverse fails or is non-finite.",
)
replace_once(
    "docs/en/models/covariance.md",
    "For \\(n > 500\\), the data is partitioned into subsets of ~300 with 500 total trials, followed by the same top-10 refinement.",
    "For larger data, 50 seeded random starts are used; candidate subsets are refined by backend-native C-steps and the best positive-definite support is retained.",
)
replace_once(
    "docs/en/models/covariance.md",
    "via cyclical coordinate descent with soft-thresholding (up to 100 inner iterations). Convergence is checked by the dual gap \\(\\|\\operatorname{tr}(W\\Theta) - p\\|\\).",
    "via cyclical coordinate descent with soft-thresholding (up to 1000 inner iterations). Outer convergence uses the maximum covariance update.",
)
insert_before(
    "docs/en/models/covariance.md",
    "## strict/approx difference\n",
    """## Backend execution and validation boundary

`GraphicalLasso` and `GraphicalLassoCV` keep centering, covariance updates,
coordinate descent, inversion, fold fitting, and held-out scoring on the selected
NumPy, CuPy, or Torch backend. `MinCovDet` keeps C-steps, Mahalanobis distances,
sorting, support masks, reweighting, and final covariance/precision on the selected
backend. Only seeded integer indices, convergence/CV scalars, and chi-square scalar
CDF/quantile calculations cross the CPU boundary.

NumPy/Torch-CPU parity and output-backend preservation are covered by regression
tests. Physical CuPy CUDA and Torch CUDA convergence, memory, runtime, and repeated-fit
validation remains `PARTIAL_REMOTE_PENDING`.

""",
)
replace_once(
    "docs/en/models/covariance.md",
    "Fitted `covariance_`, `precision_`, `location_`, and `shrinkage_` values match to relative error < 1e-15 on test datasets. `MinCovDet` matches sklearn with consistency correction factors and multi-stage FAST-MCD. `GraphicalLasso` implements the block coordinate descent of Friedman et al. (2008). Consistency checks are maintained in `dev/tests/test_external_consistency.py`.",
    "Empirical and shrinkage estimators are compared with scikit-learn at tight numerical tolerances. `MinCovDet` is checked through robust-location/covariance and support invariants, while `GraphicalLasso` is checked against reference solutions and covariance/precision structural identities. NumPy/Torch-CPU parity is covered in `dev/tests/test_three_backend_native_followup.py`; physical CUDA parity is not yet claimed.",
)

replace_once(
    "docs/en/models/splines.md",
    "`SplineTransformer` delegates to `bspline_basis` per feature.",
    "`SplineTransformer` evaluates each feature with its own backend-native Cox–de Boor recurrence and explicit extrapolation semantics.",
)
replace_once(
    "docs/en/models/splines.md",
    "| `extrapolation` | `'constant'` | Extrapolation mode: `'constant'` (clamp), `'linear'`, or `'continue'` (extend with boundary slope) |",
    "| `extrapolation` | `'constant'` | `'error'`, `'constant'` (clamp), `'linear'` (boundary tangent), or `'continue'` (continue the boundary polynomial piece) |",
)
replace_once(
    "docs/en/models/splines.md",
    "Spline basis computation has no strict/approx mode distinction. The De Boor recursion is a deterministic algorithm that produces identical results across all backends (NumPy, CuPy, Torch) up to floating-point precision.",
    "Spline basis computation has no strict/approx mode. The same recurrence is used across NumPy, CuPy, and Torch. NumPy/Torch-CPU parity is tested at tight tolerance; physical CUDA parity and performance remain pending.",
)
insert_before(
    "docs/en/models/splines.md",
    "## strict / approx Difference\n",
    """## Backend execution and extrapolation boundary

`SplineTransformer.fit()` learns knots on the selected backend and `transform()`
constructs the full basis there; it no longer transfers the complete input to SciPy.
`error`, `constant`, `linear`, and polynomial `continue` modes share the same
NumPy/CuPy/Torch recurrence. Moving a fitted transformer to another backend transfers
only knot metadata.

NumPy/Torch-CPU extrapolation parity is covered by CI. Physical CuPy CUDA and Torch
CUDA memory/runtime validation remains pending.

""",
)
replace_once(
    "docs/en/models/splines.md",
    "- **GPU speedup for splines?** The B-spline basis construction is vectorized over all sample points. For large $n$ (5000+), expect 2-3x speedup on GPU.",
    "- **GPU speedup for splines?** The recurrence is vectorized over observations and remains on-device, but speedup depends on sample size, degree, knot count, and backend. No general speedup claim is made until the current CUDA benchmark pass is completed.",
)
insert_after(
    "docs/en/models/splines.md",
    "- `SplineTransformer` output validated against `sklearn.preprocessing.SplineTransformer` for uniform and quantile knot strategies.\n",
    "- Constant, linear, and continue extrapolation are checked for NumPy/Torch-CPU parity; optional CuPy tests require a physical CUDA runtime.\n",
)

insert_before(
    "docs/en/models/panel.md",
    "## strict/approx difference\n",
    """## Backend execution and metadata boundary

For array input, `FamaMacBeth` keeps cross-sectional regressions, coefficient paths,
Newey-West covariance, inference arrays, and prediction on NumPy, CuPy, or Torch.
Panel formula construction and categorical/time/cluster label factorization remain CPU
metadata operations; only compact integer codes are copied to the numerical backend.
Scalar t/normal CDF and quantile evaluations are also intentional CPU boundaries.

Formula-side arrays are aligned to Patsy's retained rows after missing-value deletion.
NumPy/Torch-CPU parity is tested for Fama–MacBeth HAC fit and prediction; physical CUDA
validation remains pending.

""",
)
replace_once(
    "docs/en/models/panel.md",
    "fe_torch = PanelOLS(entity_effects=True, cov_type='robust', device='cuda')",
    "fe_torch = PanelOLS(entity_effects=True, cov_type='robust', device='torch')",
)

replace_once(
    "docs/en/models/anova.md",
    "`f_oneway` performs one-way Analysis of Variance (ANOVA), testing whether group means are equal. It is a GPU-accelerated drop-in replacement for `scipy.stats.f_oneway`, supporting numpy, cupy, and torch backends.",
    "The ANOVA module provides one-way ANOVA, balanced two-way ANOVA, Welch ANOVA, Tukey HSD, Bonferroni-adjusted pairwise Welch tests, and effect-size helpers. Group reductions support NumPy, CuPy, and Torch backends.",
)
replace_once(
    "docs/en/models/anova.md",
    "No strict/approx modes. Single computation path with backend selection.",
    "No strict/approx modes. Backend-native reductions share one statistical definition; unsupported distribution functions use scalar CPU calls.",
)
insert_after(
    "docs/en/models/anova.md",
    "| `\"auto\"` | Automatically selects the best available backend |\n",
    """
### Execution boundary

One-way, two-way, Welch, and post-hoc group reductions remain on the selected backend.
Tukey's studentized-range distribution and Welch/t/normal/F distribution CDF or
quantile evaluations may use CPU scalar calls where CuPy/Torch provide no equivalent.
Complete group vectors are not transferred to NumPy. NumPy/Torch-CPU parity is tested;
physical CUDA validation remains pending.
""",
)
replace_once(
    "docs/en/models/anova.md",
    "`f_twoway` performs a two-factor analysis of variance, testing the effects of factor A, factor B, and their interaction. It accepts data as a nested list of cell observations and supports both full (with interaction) and additive (without interaction) models.",
    "`f_twoway` performs a two-factor analysis of variance for balanced cell sizes, testing factor A, factor B, and optionally their interaction. Unbalanced designs are rejected until the API exposes an explicit Type I/II/III sums-of-squares convention. In the additive model, interaction variation is included in the residual term.",
)
replace_once(
    "docs/en/models/anova.md",
    "| `df_within` | int | Approximate within-group df (Welch-Satterthwaite) |",
    "| `df_within` | float | Fractional Welch-Satterthwaite denominator degrees of freedom |",
)

# ---------------------------------------------------------------------------
# Chinese model pages: update inventories and execution boundaries
# ---------------------------------------------------------------------------
for path in [
    "docs/cn/models/covariance.md",
    "docs/cn/models/splines.md",
    "docs/cn/models/panel.md",
    "docs/cn/models/anova.md",
]:
    replace_once(path, "> 最后更新: 2026-05-28", "> 最后更新: 2026-07-12")

replace_once(
    "docs/cn/models/covariance.md",
    "`covariance` 模块提供协方差矩阵估计，包含三种估计器：`EmpiricalCovariance`（经验协方差）、`LedoitWolf`（Ledoit & Wolf 2004 收缩估计）和 `OAS`（Oracle Approximating Shrinkage，Chen et al. 2010）。三者均支持 CPU、CuPy 和 PyTorch 后端，并具备自动设备检测功能。`LedoitWolf` 和 `OAS` 在 `EmpiricalCovariance` 基础上增加了向缩放单位矩阵目标的解析最优收缩，即使特征数接近或超过样本量时也能产生良态的协方差估计。",
    "`covariance` 模块包含七种估计器：`EmpiricalCovariance`、`LedoitWolf`、`OAS`、`ShrunkCovariance`、稳健的 `MinCovDet`，以及稀疏精度矩阵估计 `GraphicalLasso`/`GraphicalLassoCV`。七者均支持 NumPy、CuPy 和 Torch 后端；Graphical Lasso 的坐标下降和 FAST-MCD 的 C-step 已改为后端原生执行。",
)
replace_once(
    "docs/cn/models/covariance.md",
    "- `statgpu.covariance.OAS`",
    "- `statgpu.covariance.OAS`\n- `statgpu.covariance.ShrunkCovariance`\n- `statgpu.covariance.MinCovDet`\n- `statgpu.covariance.GraphicalLasso`\n- `statgpu.covariance.GraphicalLassoCV`",
)
insert_before(
    "docs/cn/models/covariance.md",
    "## 估计方程（Estimating Equation）\n",
    """### 其他估计器

`ShrunkCovariance` 使用用户指定的收缩强度。`MinCovDet` 通过 FAST-MCD
选择协方差行列式较小的支持子集，并使用卡方阈值进行重加权。
`GraphicalLasso` 求解

$$
\\max_{\\Theta \\succ 0}\\; \\log\\det(\\Theta)-\\operatorname{tr}(S\\Theta)
-\\alpha\\|\\Theta\\|_{1,\\mathrm{off}},
$$

其中对角元素不受 L1 惩罚；`GraphicalLassoCV` 通过留出对数似然选择
`alpha`。

""",
)
replace_once(
    "docs/cn/models/covariance.md",
    "三种估计器均使用直接计算，而非迭代优化：",
    "经验与收缩估计器使用直接计算；稳健和稀疏估计器使用迭代算法：",
)
insert_after(
    "docs/cn/models/covariance.md",
    "- **OAS**：与 Ledoit-Wolf 相同的闭式方法，但使用 OAS 收缩公式。该公式在高斯假设下推导，当 \\(n > p\\) 时渐近最优。\n",
    "- **ShrunkCovariance**：使用用户指定的收缩强度。\n- **MinCovDet**：30/50 个 seeded 随机起点，经后端原生 C-step 精炼并重加权。\n- **GraphicalLasso**：协方差块坐标下降，内层使用软阈值坐标更新；外层以协方差最大变化量判断收敛。\n- **GraphicalLassoCV**：在各 fold 上拟合并按留出高斯对数似然选择 `alpha`。\n",
)
insert_before(
    "docs/cn/models/covariance.md",
    "## strict/approx 差异（strict/approx difference）\n",
    """## 后端执行与验证边界

Graphical Lasso/CV 的中心化、协方差更新、坐标下降、求逆与 fold 评分，以及
MinCovDet 的 C-step、马氏距离、排序、支持集和重加权均保留在 NumPy/CuPy/Torch
后端。CPU 仅处理随机/fold 整数索引、收敛标量和卡方分布标量。

已验证 NumPy 与 Torch-CPU 数值一致性和输出后端；真实 CuPy/Torch CUDA 的
收敛、显存、性能与重复拟合验证仍为 `PARTIAL_REMOTE_PENDING`。

""",
)
replace_once(
    "docs/cn/models/covariance.md",
    "以上参数由 `EmpiricalCovariance`、`LedoitWolf` 和 `OAS` 共享。",
    "以上参数由七种估计器共享；`MinCovDet` 另有 `support_fraction`、`random_state`，Graphical Lasso 另有 `alpha`、`max_iter`、`tol`，CV 版本另有 `alphas` 与 `cv`。",
)
replace_once(
    "docs/cn/models/covariance.md",
    "三种估计器均针对其 scikit-learn 对应类进行验证：",
    "七种估计器均有参考实现或结构不变量测试：",
)
insert_after(
    "docs/cn/models/covariance.md",
    "- `sklearn.covariance.OAS`\n",
    "- `sklearn.covariance.ShrunkCovariance`\n- `sklearn.covariance.MinCovDet`\n- `sklearn.covariance.GraphicalLasso`\n- `sklearn.covariance.GraphicalLassoCV`\n",
)
replace_once(
    "docs/cn/models/covariance.md",
    "拟合的 `covariance_`、`precision_`、`location_` 和 `shrinkage_` 值在测试数据集上相对误差 < 1e-15。一致性检查维护在 `dev/tests/test_external_consistency.py` 中。",
    "经验与收缩估计器在严格容差下对照 scikit-learn；MinCovDet 与 Graphical Lasso 还检查支持集、互逆性、对角线和稀疏结构。NumPy/Torch-CPU parity 见 `dev/tests/test_three_backend_native_followup.py`，尚不宣称完成真实 CUDA parity。",
)

replace_once(
    "docs/cn/models/splines.md",
    "样条模块提供样条基函数构造工具。`bspline_basis` 使用 De Boor 递归算法评估 B 样条基矩阵。`natural_cubic_spline_basis` 构造带边界约束（边界节点处二阶导数为零）的自然三次样条基。两者均支持 CPU、CuPy 和 Torch 后端。",
    "样条模块提供 `bspline_basis`、`natural_cubic_spline_basis`、`cyclic_cubic_spline_basis`、`thin_plate_spline_basis` 以及 sklearn 风格的 `SplineTransformer`。这些接口支持 NumPy、CuPy 和 Torch；SplineTransformer 使用后端原生 Cox–de Boor 递推。",
)
replace_once(
    "docs/cn/models/splines.md",
    "`statgpu.nonparametric.splines.bspline_basis`、`statgpu.nonparametric.splines.natural_cubic_spline_basis`",
    "- `statgpu.nonparametric.splines.bspline_basis`\n- `statgpu.nonparametric.splines.natural_cubic_spline_basis`\n- `statgpu.nonparametric.splines.cyclic_cubic_spline_basis`\n- `statgpu.nonparametric.splines.thin_plate_spline_basis`\n- `statgpu.nonparametric.splines.SplineTransformer`",
)
insert_before(
    "docs/cn/models/splines.md",
    "## 估计方程（Estimating Equation）\n",
    """**周期三次样条**在两端约束函数值、一阶导数与二阶导数连续。
**薄板样条**使用径向核；二维且惩罚阶数为 2 时为
\\(\\phi(r)=r^2\\log r\\)。

`SplineTransformer` 为每个特征学习 uniform、quantile 或自定义节点，并支持：

- `error`：超出边界时报错；
- `constant`：钳制到边界；
- `linear`：沿边界切线延拓；
- `continue`：继续边界处的多项式片段。

""",
)
replace_once(
    "docs/cn/models/splines.md",
    "评估是直接的递归计算，无需求解线性系统。",
    "评估采用直接递推，无需求解回归系统。SplineTransformer 不再将完整数组交给 SciPy，而是在所选后端构造完整基矩阵。",
)
insert_before(
    "docs/cn/models/splines.md",
    "## strict / approx 区别\n",
    """## 后端执行与验证边界

SplineTransformer 的节点学习和四种外推均使用 NumPy/CuPy/Torch 共享递推。
在已拟合对象切换输入后端时，仅转移节点元数据，不转移完整训练设计。
已验证 NumPy/Torch-CPU 外推一致性；真实 CUDA 显存与性能验证仍待完成。

""",
)
insert_after(
    "docs/cn/models/splines.md",
    "| `xp` | `None` | 数组模块；若为 `None` 则从 `x` 推断 |\n",
    """
**SplineTransformer**：`n_knots=5`、`degree=3`、`knots='uniform'`、
`include_bias=True`、`extrapolation='constant'`，并支持 `device='auto'`。
""",
)
replace_once(
    "docs/cn/models/splines.md",
    "**样条的 GPU 加速效果如何？** B 样条基构造在所有样本点上向量化。对于大 $n$（5000+），GPU 上可期望 2-3 倍加速。",
    "**样条的 GPU 加速效果如何？** 递推已向量化并保留在设备端，但加速取决于样本量、次数、节点数和后端；完成当前 CUDA benchmark 前不作统一倍数承诺。",
)

replace_once(
    "docs/cn/models/panel.md",
    "`panel` 模块提供面板数据（纵向数据）模型。`PanelOLS` 估计固定效应（个体效应和/或时间效应），支持非稳健、HC1 稳健和聚类标准误。`RandomEffects` 使用 Swamy-Arora 方差分量估计器实现可行 GLS 随机效应。两个类均支持 CPU、CuPy 和 PyTorch 后端，并自动检测设备。",
    "`panel` 模块包含 `PanelOLS`、`RandomEffects`、`PooledOLS`、`BetweenOLS`、`FirstDifferenceOLS` 和 `FamaMacBeth` 六类估计器，并提供 clustered、two-way clustered 与 HAC 协方差工具。所有模型支持 NumPy、CuPy 和 Torch 后端。",
)
replace_once(
    "docs/cn/models/panel.md",
    "- `statgpu.panel.RandomEffects`\n- `statgpu.panel.clustered_covariance`",
    "- `statgpu.panel.RandomEffects`\n- `statgpu.panel.PooledOLS`\n- `statgpu.panel.BetweenOLS`\n- `statgpu.panel.FirstDifferenceOLS`\n- `statgpu.panel.FamaMacBeth`\n- `statgpu.panel.clustered_covariance`",
)
insert_before(
    "docs/cn/models/panel.md",
    "## 估计方程（Estimating Equation）\n",
    """**PooledOLS** 在堆叠数据上直接做 OLS；**BetweenOLS** 在个体均值上做 OLS；
**FirstDifferenceOLS** 在个体内一阶差分后做无截距 OLS。**FamaMacBeth** 在每个时期
执行横截面 OLS，再对系数路径取平均，并可使用 Newey-West HAC 推断。

""",
)
insert_before(
    "docs/cn/models/panel.md",
    "## strict/approx 差异（strict/approx difference）\n",
    """## 后端执行与元数据边界

对于数组输入，FamaMacBeth 的分期回归、系数路径、Newey-West 协方差、推断数组
和预测均保留在 NumPy/CuPy/Torch 后端。Patsy formula 构造和时间/聚类标签 factorize
属于 CPU 元数据操作，只将紧凑整数编码复制到数值后端；t/normal 分布只接收标量。
formula 删除缺失行后，entity/time/cluster 等侧数组会同步对齐。

已验证 NumPy/Torch-CPU 的 FamaMacBeth HAC 拟合与预测一致性；真实 CUDA 验证仍待完成。

""",
)
insert_after(
    "docs/cn/models/panel.md",
    "### RandomEffects\n\n| 参数 | 默认值 | 说明 |\n|---|---:|---|\n| `device` | `\"auto\"` | 计算设备：`\"cpu\"`、`\"cuda\"` 或 `\"auto\"` |\n",
    """
### 其他模型

- `PooledOLS(cov_type='nonrobust', bandwidth=None, kernel='bartlett')`
- `BetweenOLS(cov_type='nonrobust')`
- `FirstDifferenceOLS(cov_type='nonrobust')`
- `FamaMacBeth(cov_type='newey-west', bandwidth=None, min_obs_per_period=1)`

以上模型均支持 `alpha` 和 `device`；相应 `fit()` 需要 entity/time/cluster 元数据。
""",
)
replace_once(
    "docs/cn/models/panel.md",
    "from statgpu.panel import PanelOLS, RandomEffects",
    "from statgpu.panel import (PanelOLS, RandomEffects, PooledOLS,\n                           BetweenOLS, FirstDifferenceOLS, FamaMacBeth)",
)

replace_once(
    "docs/cn/models/anova.md",
    "`f_oneway` 执行单因素方差分析（One-Way ANOVA），检验各组均值是否相等。它是 `scipy.stats.f_oneway` 的 GPU 加速替代实现，支持 numpy、cupy 和 torch 后端。",
    "ANOVA 模块提供 `f_oneway`、平衡设计 `f_twoway`、`f_welch`、`tukey_hsd`、`bonferroni` 以及 `cohens_f`/`partial_eta_squared` 效应量工具。组内归约支持 NumPy、CuPy 和 Torch。",
)
replace_once(
    "docs/cn/models/anova.md",
    "`statgpu.anova.f_oneway`、`statgpu.anova.AnovaResult`",
    "- `statgpu.anova.f_oneway` / `AnovaResult`\n- `statgpu.anova.f_twoway` / `TwoWayAnovaResult`\n- `statgpu.anova.f_welch`\n- `statgpu.anova.tukey_hsd` / `TukeyResult`\n- `statgpu.anova.bonferroni` / `PosthocResult`\n- `statgpu.anova.cohens_f` / `partial_eta_squared`",
)
insert_before(
    "docs/cn/models/anova.md",
    "## 参数（Parameters）\n",
    """### 双因素、Welch 与事后检验

`f_twoway` 支持包含交互项的完整模型和不含交互项的加性模型。当前只接受各 cell
样本量相同的平衡设计；非平衡设计在 API 明确 Type I/II/III 平方和前会报错。
加性模型会把交互变异并入残差。

`f_welch` 用于异方差组，并保留 Welch-Satterthwaite 的小数分母自由度。
`tukey_hsd` 使用 studentized-range 分布，`bonferroni` 执行 Bonferroni 校正的
两两 Welch t 检验。

""",
)
insert_before(
    "docs/cn/models/anova.md",
    "## strict/approx 差异（strict/approx difference）\n",
    """## 后端执行与分布边界

单/双因素、Welch 与事后检验的组内均值、方差和平方和保留在所选后端。
studentized-range、t、normal 或 F 分布在后端缺少实现时只接收标量并在 CPU 计算；
不会把完整组向量传回 NumPy。已验证 NumPy/Torch-CPU 一致性，真实 CUDA 验证仍待完成。

""",
)
replace_once(
    "docs/cn/models/anova.md",
    "无 strict/approx 模式区分。单一计算路径，仅需选择后端。",
    "无 strict/approx 模式。各后端共享同一统计定义；后端不支持的分布函数仅使用 CPU 标量调用。",
)

print("PR79 public documentation synchronization completed")
