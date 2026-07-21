from pathlib import Path


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text()
    start_pos = text.index(start)
    end_pos = text.index(end, start_pos)
    p.write_text(text[:start_pos] + replacement.rstrip() + "\n\n" + text[end_pos:])


# Root changelog: replace the current final-validation summary only.
replace_between(
    "CHANGELOG.md",
    "### PR #79 — Final physical GPU validation and correctness hardening",
    "## 2026-07-14",
    """### PR #79 — Final physical GPU validation and correctness hardening

- Completed GPU smoke, three-backend correctness, metamorphic, device-purity,
  memory-leak, performance, external-validation, and full CPU/GPU gates on Tesla P100.
- Full campaign result on `2f18e5d`: 1100 passed, 0 failed, 124 skipped, and
  1 version-limited strict XFAIL; all 40 initial Gate B failures were eliminated or
  formally dispositioned.
- Completed a subsequent review-fix cycle covering backend-native `LinearRegression`,
  PooledOLS HAC ordering and effective rank, formula-weight alignment, validator integrity,
  weighted CPU/CuPy/Torch fitting, and degenerate GPU F-statistic semantics.
- Exact-head physical GPU acceptance on clean SHA
  `786af9e2eb4742a56e5203b4380b03aec63a3ac8`: **17 passed, 0 failed, 0 skipped**
  in 7.28 seconds, with CuPy and Torch CUDA tests both executed.
- Degenerate F tests now agree across backends: perfect non-constant fit returns
  `(inf, 0.0)`; intercept-only and otherwise undefined overall tests return `(nan, nan)`.
- Standard GitHub Actions Tests run #483 also passed on the exact cleaned head.
- Follow-up issues #81 and #82 remain non-blocking; see
  `dev/reviews/pr79_physical_gpu_validation.md`.""",
)


# English changelog: replace the post-validation and latest-head boundary sections.
replace_between(
    "docs/en/changelog.md",
    "### Fixed (2026-07-21) — post-validation review-fix loop",
    "### Performance baseline — Tesla P100",
    """### Fixed (2026-07-21) — post-validation review-fix loop

A further review → fix → test → re-review cycle was completed after the full GPU campaign.
The exact cleaned acceptance head is
`786af9e2eb4742a56e5203b4380b03aec63a3ac8`.

Additional repairs:

- preserved backend-native `LinearRegression.fit` and `predict` inputs until backend
  resolution instead of performing eager NumPy conversion;
- made PooledOLS HAC ordering explicit through validated, stable `time_index` sorting;
- used effective design rank for PooledOLS residual degrees of freedom;
- hardened the remote validator with shell `pipefail`, exact required SHAs, immutable base
  worktrees, and reset/clean verification;
- separated formula-controlled intercept semantics from the public clone-visible
  `fit_intercept` constructor parameter;
- corrected weighted `LinearRegression` on CPU, CuPy, and Torch, including intercept
  weighting, multi-output broadcasting, validation, residual state, singular fallback,
  diagnostics, and weighted R-squared;
- aligned original-length formula sample weights after Patsy removes missing rows;
- aligned CuPy and Torch degenerate overall F-test semantics with the CPU contract.

Permanent coverage in `dev/tests/test_pr79_final_review_fixes.py` includes reference-library
parity, rank-deficient and HAC invariants, formula behavior, invalid weights, multi-output
WLS, exact-SHA validator checks, backend-to-NumPy transfer guards, and physical CuPy/Torch
F-statistic edge cases.

### Validation (2026-07-21) — exact-head acceptance passed

On a clean Tesla P100 worktree at exact SHA
`786af9e2eb4742a56e5203b4380b03aec63a3ac8`:

- `STATGPU_REQUIRE_PHYSICAL_GPU=1` forced both CUDA backends to execute;
- `dev/tests/test_pr79_final_review_fixes.py` completed with
  **17 passed, 0 failed, 0 skipped in 7.28 seconds**;
- CuPy and Torch weighted fit/predict parity passed;
- formula missing-row and original-length weight alignment passed;
- perfect non-constant fits return `(inf, 0.0)` for the overall F test;
- intercept-only and otherwise undefined overall F tests return `(nan, nan)`;
- the exact SHA and clean-worktree state were recorded.

Standard GitHub Actions Tests run #483 also passed on the cleaned head, including the
Python 3.9–3.12 regression matrices, static contracts, compilation, complete collection,
and full CPU suite. PR #79 is therefore ready for review and squash merge.""",
)


# Chinese changelog: same evidence and boundary.
replace_between(
    "docs/cn/changelog.md",
    "### 修复（2026-07-21）— 验证后的 review-fix 循环",
    "### 性能基线 — Tesla P100",
    """### 修复（2026-07-21）— 验证后的 review-fix 循环

完整 GPU 验证后又完成了一轮 review → fix → test → re-review。最终 exact-head 验收
代码为 `786af9e2eb4742a56e5203b4380b03aec63a3ac8`。

新增修复包括：

- `LinearRegression.fit` 与 `predict` 在后端解析前保留 CuPy/Torch 原生输入；
- PooledOLS HAC 使用经过验证的 `time_index` 稳定排序；
- PooledOLS 使用有效设计秩计算 residual degrees of freedom；
- 远程验证器加入 shell `pipefail`、精确 SHA、不可变 base worktree 与 reset/clean 检查；
- 将公式控制的截距语义与公开、clone 可见的 `fit_intercept` 参数分离；
- 修正 CPU、CuPy、Torch 带权 `LinearRegression` 的截距加权、multi-output 广播、
  权重验证、残差状态、奇异设计 fallback、diagnostics 与 weighted R-squared；
- Patsy 删除缺失行后，按保留的原始行位置对齐 formula sample weights；
- 统一 CuPy、Torch 与 CPU 的退化 overall F-test 语义。

永久测试 `dev/tests/test_pr79_final_review_fixes.py` 覆盖外部库对齐、秩亏与 HAC
不变量、公式语义、非法权重、multi-output WLS、精确 SHA 验证器、
backend-to-NumPy 传输保护，以及真实 CuPy/Torch F-stat 边界情况。

### 验证（2026-07-21）— exact-head 最终验收通过

在 Tesla P100 的 clean worktree 上，对精确 SHA
`786af9e2eb4742a56e5203b4380b03aec63a3ac8` 执行最终验收：

- 设置 `STATGPU_REQUIRE_PHYSICAL_GPU=1`，强制 CuPy 与 Torch CUDA 测试实际执行；
- `dev/tests/test_pr79_final_review_fixes.py` 结果为
  **17 passed，0 failed，0 skipped，耗时 7.28 秒**；
- CuPy 与 Torch weighted fit/predict parity 通过；
- formula 缺失行与原始长度 sample weights 对齐通过；
- perfect non-constant fit 的 overall F test 返回 `(inf, 0.0)`；
- intercept-only 及其他未定义 overall F test 返回 `(nan, nan)`；
- 已记录 exact SHA 与 clean-worktree 状态。

标准 GitHub Actions Tests run #483 同样通过，包括 Python 3.9–3.12 regression matrix、
static contracts、编译、完整测试收集与 full CPU suite。因此 PR #79 已满足 Ready for
review 与 squash merge 条件。""",
)


# Final report: replace decision, evidence boundary, table row, and pending section.
report = Path("dev/reviews/pr79_physical_gpu_validation.md")
text = report.read_text()
text = text.replace(
    "Latest cleaned code head after the review-fix loop: `ff72424071ec7ca52399146dbd8a556534c9e6c3`",
    "Exact-head physical-GPU acceptance SHA: `786af9e2eb4742a56e5203b4380b03aec63a3ac8`",
)
report.write_text(text)
replace_between(
    str(report),
    "## Decision",
    "## Evidence boundary",
    """## Decision

**MERGE-READY.** The complete Tesla P100 Gate A–G campaign passed, the subsequent
review-fix cycle was completed, and the exact cleaned head
`786af9e2eb4742a56e5203b4380b03aec63a3ac8` passed the mandatory focused physical-GPU
acceptance suite with **17 passed, 0 failed, and 0 skipped in 7.28 seconds**.

CuPy CUDA and Torch CUDA both executed under `STATGPU_REQUIRE_PHYSICAL_GPU=1`. The exact
SHA and clean-worktree state were recorded. Standard GitHub Actions Tests run #483 also
passed. No unresolved CRITICAL/HIGH defect or PR-introduced regression is known.

Issues #81 and #82 and the Torch Cox Hessian memory optimization remain explicitly tracked,
non-blocking follow-ups.""",
)
replace_between(
    str(report),
    "### Post-validation review-fix evidence",
    "## Additional defects fixed by the post-validation review-fix loop",
    """### Post-validation review-fix and exact-head evidence

The post-validation review repaired additional backend-routing, PooledOLS, WLS, formula,
validator, and GPU inference edge cases. Standard GitHub Actions Tests run #483 completed
successfully on the cleaned head with:

- regression matrices on Python 3.9, 3.10, 3.11, and 3.12;
- static-contract, compilation, and complete-collection gates;
- the complete CPU test suite.

The mandatory Tesla P100 exact-head acceptance then ran on clean SHA
`786af9e2eb4742a56e5203b4380b03aec63a3ac8`:

```text
17 passed in 7.28s
```

Both CuPy and Torch CUDA parameterizations executed with no skips. The suite confirmed
weighted fit/predict parity, formula missing-row weight alignment, device-purity guards,
and backend-consistent degenerate F-statistic semantics.""",
)
text = report.read_text()
needle = "| Formula sample weights | Patsy could drop rows while `sample_weight` retained original length. Formula evaluation now returns retained row positions and aligns weights deterministically. | HIGH |"
replacement = needle + "\n| GPU overall F-test edge cases | The early return mixed perfect-fit and intercept-only cases and returned an incorrect p-value. CuPy/Torch now return `(inf, 0.0)` for perfect non-constant fits and `(nan, nan)` when the overall test is undefined. | HIGH |"
if replacement not in text:
    text = text.replace(needle, replacement)
report.write_text(text)
replace_between(
    str(report),
    "## Required exact-head physical-GPU recheck",
    "## Previously fixed production defects from the full GPU campaign",
    """## Exact-head physical-GPU acceptance — PASS

Command:

```bash
STATGPU_REQUIRE_PHYSICAL_GPU=1 \\
python -m pytest dev/tests/test_pr79_final_review_fixes.py -q -rs --tb=short
```

Recorded result on clean SHA `786af9e2eb4742a56e5203b4380b03aec63a3ac8`:

```text
17 passed in 7.28s
```

Acceptance results:

1. CuPy CUDA available and executed: PASS.
2. Torch CUDA available and executed: PASS.
3. No GPU parameterization skipped: PASS.
4. Weighted fit/predict parity: PASS.
5. Formula missing-row and sample-weight alignment: PASS.
6. Perfect-fit overall F test `(inf, 0.0)` on CuPy and Torch: PASS.
7. Intercept-only overall F test `(nan, nan)` on CuPy and Torch: PASS.
8. Exact SHA and clean-worktree state recorded: PASS.

The physical-GPU validation loop is closed. PR #79 may be marked Ready for review.""",
)
