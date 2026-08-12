from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: replacement count={count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "dev/plans/panel_p1_stage_c_covariance_plan.md",
    "- strict inference rejects every strictly negative final covariance diagonal; IEEE signed zero may be normalized to zero, but no absolute or row/column-scale tolerance may turn a genuinely negative variance into a valid-looking standard error because such tolerances are not invariant to outcome/regressor units;",
    "- strict inference first requires the entire final covariance matrix to be finite, then rejects every strictly negative diagonal entry; IEEE signed zero may be normalized to zero, but no absolute or row/column-scale tolerance may turn a genuinely negative variance into a valid-looking standard error because such tolerances are not invariant to outcome/regressor units;",
)
replace_once(
    "docs/en/models/panel.md",
    "The previous exact-clean Tesla P100 measurement `3dc7df19...` (CuPy/Torch **39/39 per backend**, plus 58 synchronized performance rows) remains immutable historical evidence, but it is no longer current acceptance evidence after the 2026-08-12 rank-deficient-df and strict covariance-validity fixes. The current implementation still uses one backend-native SVD mask with cutoff `max(n,k) * eps64 * s_max`, now also uses identified rank for supported rank-deficient residual/auxiliary degrees of freedom, and rejects every strictly negative final covariance diagonal. Fresh physical acceptance is pending on the current exact source with an expanded target of **35 estimator integrations + 12 public primitives = 47/47 per backend**, plus the unchanged 58-row performance matrix.",
    "The previous exact-clean Tesla P100 measurement `3dc7df19...` (CuPy/Torch **39/39 per backend**, plus 58 synchronized performance rows) remains immutable historical evidence, but it is no longer current acceptance evidence after the 2026-08-12 rank-deficient-df and strict covariance-validity fixes. The current implementation still uses one backend-native SVD mask with cutoff `max(n,k) * eps64 * s_max`, now also uses identified rank for supported rank-deficient residual/auxiliary degrees of freedom, and fails closed if the final covariance contains any non-finite value or any strictly negative diagonal variance (IEEE signed zero may normalize to zero). Fresh physical acceptance is pending on the current exact source with an expanded target of **35 estimator integrations + 12 public primitives = 47/47 per backend**, plus the unchanged 58-row performance matrix.",
)
replace_once(
    "docs/cn/models/panel.md",
    "此前 exact-clean `3dc7df19...` Tesla P100 结果（CuPy/Torch **每个 backend 39/39**，以及 58 行同步 performance）继续作为不可变历史证据保留，但在 2026-08-12 的 rank-deficient df 与 strict covariance-validity 修复后已不再是当前 acceptance evidence。当前实现继续使用同一个 backend-native SVD mask 与 cutoff `max(n,k) * eps64 * s_max`，并进一步让受支持的秩亏 residual/auxiliary df 使用 identified rank，同时对任何严格为负的最终 covariance diagonal fail closed。新的 physical acceptance 尚待在当前 exact source 上执行扩展矩阵：**35 个 estimator integration + 12 个 public primitive = 每个 backend 47/47**，performance 目标仍为 58 行。",
    "此前 exact-clean `3dc7df19...` Tesla P100 结果（CuPy/Torch **每个 backend 39/39**，以及 58 行同步 performance）继续作为不可变历史证据保留，但在 2026-08-12 的 rank-deficient df 与 strict covariance-validity 修复后已不再是当前 acceptance evidence。当前实现继续使用同一个 backend-native SVD mask 与 cutoff `max(n,k) * eps64 * s_max`，并进一步让受支持的秩亏 residual/auxiliary df 使用 identified rank；若最终 covariance 含任何非有限值或任何严格为负的 diagonal variance，则 strict inference fail closed（IEEE signed zero 可正规化为 0）。新的 physical acceptance 尚待在当前 exact source 上执行扩展矩阵：**35 个 estimator integration + 12 个 public primitive = 每个 backend 47/47**，performance 目标仍为 58 行。",
)
