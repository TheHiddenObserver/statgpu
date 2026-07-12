"""Synchronize PR #79 public-module audit documentation without truncating history."""

from pathlib import Path


def insert_once(path: str, marker: str, insertion: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    heading = insertion.splitlines()[0]
    if heading in text:
        return
    if marker not in text:
        raise RuntimeError(f"{path}: insertion marker not found: {marker!r}")
    file_path.write_text(text.replace(marker, marker + insertion, 1), encoding="utf-8")


root_section = '''### PR #79 — Public module statistical-contract follow-up

- Extended the repository review beyond Ridge to every top-level public module family,
  combining full-package high-signal static analysis with targeted numerical invariants,
  nested-model checks, and parity comparisons against established reference libraries.
- Corrected two-way ANOVA residual and balance semantics, Welch/post-hoc degenerate cases,
  chi-square kernels, KernelRidge/KernelRidgeCV scoring, KernelPCA embedding consistency,
  and Nystroem normalization for indefinite kernels.
- Corrected empirical precision estimation, Graphical Lasso block-coordinate updates,
  MinCovDet centered semantics, panel cluster/HAC contracts, Patsy side-array alignment,
  and rank-deficient panel regression fallbacks.
- Implemented real spline extrapolation modes; hardened B-spline, KDE, kernel regression,
  GAM, and binary-metric input contracts.
- Added three focused regression suites and expanded the permanent Python 3.9–3.12,
  full-CPU, static-contract, compilation, and complete-collection gates.
- Validation remains `PARTIAL_REMOTE_PENDING`: all hosted CPU/static gates pass, while
  physical CuPy/Torch CUDA numerical, memory, type/device, and performance validation is
  still required for affected GPU paths.

'''
insert_once("CHANGELOG.md", "## 2026-07-12\n\n", root_section)

english_section = '''### Fixed and hardened (2026-07-12) — PR #79 public-module follow-up

- Extended the review beyond Ridge to ANOVA, kernel methods, covariance, panel models,
  KDE/kernel regression, splines, GAM, and binary metrics.
- Fixed two-way ANOVA model decomposition, Welch/post-hoc degeneracies, chi-square
  kernel domain/fallback logic, KernelRidge scoring/CV, KernelPCA consistency, and
  Nystroem normalization for indefinite kernels.
- Fixed empirical precision, Graphical Lasso coordinate descent, MinCovDet centering,
  clustered/HAC panel covariance, formula side-array alignment, and rank-deficient panel
  regression fallbacks.
- Implemented actual `error`/`constant`/`linear`/`continue` spline extrapolation and
  hardened finite-value, parameter, shape, and degeneracy contracts across B-splines,
  KDE, kernel regression, GAM, and classification metrics.
- Added focused numerical regression suites and expanded the permanent multi-version,
  full-CPU, static, compilation, and collection gates. Status remains
  `PARTIAL_REMOTE_PENDING` until physical CuPy/Torch CUDA validation is complete.

'''
insert_once("docs/en/changelog.md", "## 2026-07\n\n", english_section)
en_path = Path("docs/en/changelog.md")
en = en_path.read_text(encoding="utf-8").replace(
    "> Last updated: 2026-07-11", "> Last updated: 2026-07-12", 1
)
en_path.write_text(en, encoding="utf-8")

chinese_section = '''### 修复与加固（2026-07-12）— PR #79 公开模块后续审查

- 将审查范围从 Ridge 扩展到 ANOVA、核方法、协方差、面板模型、KDE/核回归、
  样条、GAM 与二分类指标等全部顶层公开模块族。
- 修复双因素 ANOVA 模型分解、Welch/事后检验退化情形、卡方核定义域与 fallback、
  KernelRidge 评分/CV、KernelPCA 一致性及不定核下 Nystroem 归一化。
- 修复经验精度矩阵、Graphical Lasso 坐标下降、MinCovDet 中心化语义、面板聚类/HAC
  协方差、formula 侧数组行对齐及秩亏面板回归的稳定回退。
- 为样条实现真实的 `error`/`constant`/`linear`/`continue` 外推，并加固 B-spline、
  KDE、核回归、GAM 与分类指标的有限性、参数、形状和退化情形契约。
- 新增专项数值回归测试，扩展永久多版本、完整 CPU、静态、编译与收集门禁。
  当前仍为 `PARTIAL_REMOTE_PENDING`，需完成真实 CuPy/Torch CUDA 验证。

'''
insert_once("docs/cn/changelog.md", "## 2026-07\n\n", chinese_section)
cn_path = Path("docs/cn/changelog.md")
cn = cn_path.read_text(encoding="utf-8").replace(
    "> 最后更新：2026-07-11", "> 最后更新：2026-07-12", 1
)
cn_path.write_text(cn, encoding="utf-8")

report_path = Path("dev/reviews/pr79_full_repository_review.md")
report = report_path.read_text(encoding="utf-8")
public_section = '''### Post-Ridge public module audit

1. **ANOVA and post-hoc inference**
   - Additive two-way ANOVA now absorbs omitted interaction variation into the
     residual instead of inflating main-effect F statistics.
   - The balanced-design sums-of-squares implementation rejects unbalanced cells
     until the API exposes an explicit Type I/II/III convention.
   - Welch ANOVA preserves fractional denominator degrees of freedom and rejects
     mixed zero-variance groups rather than silently changing the null hypothesis.
   - Tukey and Bonferroni comparisons handle identical constant groups without
     NaN/Inf artifacts and accept optional GPU inputs through an explicit boundary.
2. **Kernel methods**
   - The chi-square kernel rejects negative features and its chunked NumPy fallback
     matches the reference definition.
   - KernelRidge validates inputs, uses stable solve fallback, and implements
     force-finite uniform-average multi-output R-squared.
   - KernelRidgeCV validates folds/grids, avoids unused eigenvectors, and reports
     actual mean fold R-squared.
   - KernelPCA uses the unregularized centered-kernel eigenvalues for embeddings so
     training `fit_transform` and `transform` agree.
   - Nystroem uses SVD normalization for indefinite kernels instead of converting
     negative eigenvalues into enormous artificial features.
3. **Covariance estimators**
   - EmpiricalCovariance computes the exact precision when possible and adds jitter
     only as a singular fallback.
   - GraphicalLasso uses covariance block-coordinate descent, leaves the precision
     diagonal unpenalized, preserves the empirical covariance diagonal, and returns
     mutually consistent covariance/precision matrices.
   - GraphicalLassoCV validates folds and alpha grids; MinCovDet validates support
     fractions and honors `assume_centered` throughout its C-steps.
4. **Panel estimators**
   - Cluster labels are factorized before GPU conversion; clustered/HAC covariance
     validates labels, lengths, kernels, and bandwidths.
   - Formula-side entity/time/cluster arrays follow Patsy's retained rows after
     missing-value filtering.
   - Pooled, between, first-difference, fixed-effects, and Fama-MacBeth paths use
     stable pseudoinverse fallbacks and explicit residual-degree/period checks.
5. **Smoothing, splines, GAM, and metrics**
   - SplineTransformer now implements real `error`, `constant`, `linear`, and
     `continue` extrapolation rather than silently returning zero/ignoring modes.
   - B-spline, KDE, and kernel-regression shared utilities reject non-finite inputs,
     invalid knots, and non-finite weights.
   - GAM validates smoothing parameters, shapes, finite data, and constant features,
     and accepts one-dimensional prediction points for one-feature fits.
   - Binary evaluation rejects non-finite decision thresholds.

'''
if "### Post-Ridge public module audit" not in report:
    marker = "### Test and CI quality\n\n"
    if marker not in report:
        raise RuntimeError("review report test section marker not found")
    report = report.replace(marker, public_section + marker, 1)

report = report.replace(
    "7. Ridge-specific tests cover:\n",
    "7. Ridge-specific tests cover:\n",
    1,
)
if "8. Three post-Ridge public-module suites cover:" not in report:
    marker = "8. CI includes Python 3.9-3.12 regression gates,"
    addition = '''8. Three post-Ridge public-module suites cover:
   - nested additive/two-way ANOVA identities and degenerate post-hoc cases;
   - sklearn/reference parity and invariants for kernel/covariance estimators;
   - formula missing-row alignment and panel covariance/rank-deficiency contracts;
   - spline extrapolation, smoothing/GAM finite-value contracts, and metric edges.
9. CI includes Python 3.9-3.12 regression gates,'''
    if marker not in report:
        raise RuntimeError("review report CI-list marker not found")
    report = report.replace(marker, addition, 1)

report = report.replace(
    "- run the affected UMAP/NNDescent, Cox, knockoff, inference, and ElasticNetCV\n  suites on both CuPy CUDA and Torch CUDA;\n",
    "- run the affected UMAP/NNDescent, Cox, knockoff, inference, and ElasticNetCV\n  suites on both CuPy CUDA and Torch CUDA;\n"
    "- validate kernel, covariance, panel, KDE/kernel-regression, spline, GAM, and\n"
    "  post-hoc paths for numerical parity, output type/device, memory, and runtime;\n",
    1,
)
report = report.replace("GitHub Actions run **#228** passed", "GitHub Actions run **#268** passed", 1)
report = report.replace(
    "including both Ridge CV implementations and the penalized fit/inference paths;",
    "including the Ridge, ANOVA, kernel, covariance, panel, smoothing, spline, GAM, and metrics paths;",
    1,
)
report_path.write_text(report, encoding="utf-8")

print("Post-Ridge public-module audit documentation synchronized")
