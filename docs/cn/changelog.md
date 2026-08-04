# Changelog

> 语言：中文<br>
> 最后更新：2026-08-04<br>
> 页面定位：变更记录<br>
> 切换：[English](../en/changelog.md)

## 2026-08

### 修复（2026-08-04）— PR #80 精确源码 CV 复审后续

- 规范物理 GPU suite 现在会把受审计的 Git checkout 放在 `PYTHONPATH`
  首位、禁用 user site，核验实际导入模块的路径均位于该 checkout 内，并记录这些
  实际导入文件的 SHA-256；child 与 nested runner 继承同一受控环境。
- 请求 CoxPHCV two-stage 或 successive-halving 后，NumPy、CuPy 与 Torch 现在都只
  执行一次显式 exhaustive full-precision candidate pass。公开诊断记录
  `staged_safety_strategy="single_pass_exhaustive"`；不筛除任何 candidate，CuPy 也不再
  重复完整 grid。
- 一次性 `CoxPHCV.cv_splits` iterator 会私下 materialize 一次，并在重复 fit、
  scikit-learn clone、旧版参数重建与 pickle 中复用；fit 期间公开构造参数对象保持不变。
- Hosted workflow #946 已在精确 head
  `a726937a39eb0ed5a370dd03362884b63a9e9818` 上通过：完整 CPU suite 为
  1879 passed、662 skipped，static、文档及 Python 3.9–3.12 regression job 全部通过。
- 该 head 的原始物理结果现已持久发布为
  [最终 promotion artifact](https://gist.github.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139)。
  Artifact 记录 134/134 项检查通过、所有 return code 为 0、所有 gate-failure 数组为空，
  SHA-256 为
  `e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb`。
- 本后续提交将 final aggregation format 正式升级为 machine schema 3，同步 CoxPH
  主模型页，并把 `.markdown` 历史页重新纳入维护文档检查。由于这些提交产生了新的
  head，最终批准前必须对新 head 再运行一次 exact-head physical suite；上述 Gist
  仍只证明 `a726937...`。

## 更早的历史记录

截至 2026-08-03 的详细条目保留在
[归档 changelog](changelog-history-through-2026-08-03.markdown)。
