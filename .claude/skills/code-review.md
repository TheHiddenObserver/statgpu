---
name: code-review
description: Comprehensive code review of statgpu modules - bugs, readability, maintainability, extensibility, performance
---

# Code Review Skill

When the user invokes this skill, perform a thorough code review of the specified files across 5 dimensions.

## Input

The user provides:
- File paths or module name to review
- Focus areas (optional): bugs, readability, maintainability, extensibility, performance

## Review checklist

### 1. Bug 检查 (Correctness)
- [ ] 公式正确性（对标论文、sklearn、R）
- [ ] 边界情况处理（空输入、单观测、奇异矩阵）
- [ ] 自由度计算正确
- [ ] 数值稳定性（除零、溢出、下溢）
- [ ] 后端一致性：GPU 输入 → GPU 输出，无静默 numpy 转换
- [ ] `hasattr(x, 'is_cuda')` 检测 torch（不用 `hasattr(x, 'device')`）
- [ ] API 契约：方法与文档一致，`fit()` 设置 `_fitted = True`
- [ ] 错误处理：异常信息清晰，不静默失败

### 2. 可读性 (Readability)
- [ ] 无魔法数字（使用命名常量）
- [ ] 文档完整（Parameters, Returns, Raises, Examples）
- [ ] 注释解释"为什么"，而非"是什么"
- [ ] 命名一致（函数/变量/类遵循项目惯例）
- [ ] 代码结构清晰（函数长度合理，职责单一）
- [ ] 缩进和格式一致

### 3. 可维护性 (Maintainability)
- [ ] 无代码重复（提取共享函数/基类）
- [ ] 修改一处不需要改多处（DRY 原则）
- [ ] 错误处理不依赖隐式行为
- [ ] 硬编码值提取为常量或参数
- [ ] 依赖关系清晰（导入不循环，不引入不必要的依赖）
- [ ] 向后兼容：已有 API 不被破坏

### 4. 可拓展性 (Extensibility)
- [ ] 新增参数不影响已有调用（默认值兼容）
- [ ] 抽象层次合适（基类/接口可复用）
- [ ] 注册表/工厂模式便于扩展（如 kernel registry, covariance registry）
- [ ] 钩子/回调点设计合理（如 `_transform_data`, `_estimate`）
- [ ] 配置可外部化（不硬编码设备、阈值等）

### 5. 性能优化 (Performance)
- [ ] 无不必要的 n×n 临时数组
- [ ] 使用 in-place 操作减少内存分配
- [ ] Python 循环向量化（用 numpy/cupy 操作替代）
- [ ] 大矩阵使用 float32 降低内存带宽
- [ ] 分块计算避免 OOM（chunked computation）
- [ ] 缓存避免重复计算（如 KPCA 缓存训练核统计）
- [ ] 小矩阵避免 GPU kernel launch 开销

## Output format

For each file, report:
- **CLEAN** if no issues found
- **severity | dimension | file:line | description | fix** for each issue

Severity levels:
- **CRITICAL**: Wrong results, crashes, data loss
- **HIGH**: Silent incorrect behavior, API violations
- **MEDIUM**: Performance issues, code quality, missing edge cases
- **LOW**: Style, documentation, minor improvements

Dimensions:
- `BUG` — correctness issue
- `READ` — readability issue
- `MAINT` — maintainability issue
- `EXT` — extensibility issue
- `PERF` — performance issue

## Review loop

After the initial review:
1. Fix all CRITICAL and HIGH issues
2. Run tests: `pytest dev/tests/ -q`
3. Re-review the fixed files
4. Repeat until all files are CLEAN

## Example output

```
CRITICAL | BUG  | _graphical_lasso.py:146 | Precision off-diagonal not divided by Schur complement | theta_j[not_j] = -beta / c
HIGH     | MAINT | _shrinkage.py:230     | OAS denominator doesn't match sklearn | Revert to (n+1)
MEDIUM   | PERF  | _kpca.py:184          | Training kernel recomputed every transform() | Cache during fit()
LOW      | READ  | _welch.py:122         | df_within rounded to int, loses precision | Change to float
```
