from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "dev/plans/panel_p1_stage_c_covariance_plan.md"
text = PATH.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"plan: expected exactly one match, got {count}: {old[:80]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "`CV` is non-tunable/not applicable for all touched capabilities. A covariance option is not declared supported until its NumPy/CuPy/Torch inference path is tested.\n",
    "`CV` is non-tunable/not applicable for all touched capabilities. A covariance option is not declared supported until its NumPy/CuPy/Torch inference path is tested. The `inference=supported` capability decision applies to identified/full-column-rank coefficient coordinates. Exact rank-deficient fits remain supported for minimum-norm estimation, prediction, residuals, fit statistics, and fit-space covariance, but ordinary coordinate-wise BSE/test/p-value/CI output is explicitly unavailable because the original coefficient coordinates are not uniquely identified.\n",
)
replace_once(
    "9. On supported rank-deficient fits, residual and auxiliary-regression degrees of freedom use the identified numerical rank. Adding an exactly redundant column must not change identified fitted values, Swamy-Arora variance components/theta, fit-space covariance, or inference merely by increasing the raw column count.\n",
    "9. On supported rank-deficient fits, residual and auxiliary-regression degrees of freedom use the identified numerical rank. Adding an exactly redundant column must not change identified fitted values, Swamy-Arora variance components/theta, or fit-space covariance merely by increasing the raw column count. The coefficient vector is the shared Moore-Penrose minimum-norm representation, while ordinary coordinate-wise BSE/test/p-value/CI output is unavailable and summaries fail closed because the original coordinates are not uniquely identified.\n",
)
replace_once(
    "- covariance/BSE/t-or-z/p/CI consistent;\n",
    "- on full-column-rank fits, covariance/BSE/t-or-z/p/CI consistent; on exact rank-deficient fits, fit-space covariance remains auditable while coordinate-wise BSE/t-or-z/p/CI is explicitly unavailable;\n",
)
replace_once(
    "Record exact SHA/clean tree, requested/executed backend, covariance/SE/t/p/CI vs NumPy, coefficient and Stage-B-stat invariance, covariance config/effective bandwidth/support/group counts/rank extension, and environment/GPU metadata.\n\nA validator change after an accepted artifact invalidates acceptance for the changed validator contract per `RELEASING.md`.\n",
    "Fresh correctness evidence uses schema v2. Record exact SHA/clean tree, requested/executed fit backend, full-rank covariance/SE/t/p/CI vs NumPy, rank-deficient fit rank/parameter count plus explicit coefficient-inference unavailability, `PanelOLS` and `RandomEffects` prediction execution backend on representative full-rank cases, coefficient and Stage-B-stat invariance where identified, covariance config/effective bandwidth/support/group counts/rank extension, and environment/GPU metadata. The maintained matrix remains 35 estimator integrations plus 12 public covariance primitives, i.e. 47/47 checks per requested GPU backend.\n\nA validator change after an accepted artifact invalidates acceptance for the changed validator contract per `RELEASING.md`. Historical schema-v1 evidence and its immutable parser/source identities are never overwritten.\n",
)
replace_once(
    "- GPU metadata conversion/synchronization is not transfer-dominated at target sizes.\n\nNo speedup claim is planned. Performance becomes blocking only for material regression/pathological complexity/transfer dominance. Optimization budget: one profile, at most two algorithmic/kernel attempts, one rebenchmark each. Timing JSON remains separate from correctness evidence.\n",
    "- GPU metadata conversion/synchronization is not transfer-dominated at target sizes;\n- the iterative two-way FE path changed in the final review loop, so schema-v3 performance evidence includes an explicit incomplete/unbalanced `PanelOLS(entity_effects=True, time_effects=True, cov_type='nonrobust')` case at `N=10,000`, `k=2`, `T=20` on both CuPy and Torch.\n\nNo speedup claim is planned. Fresh schema-v3 performance evidence contains 60 rows: the historical 54 base rows, 4 bounded high-T QS rows, and 2 unbalanced two-way FE rows. Performance becomes blocking only for material regression/pathological complexity/transfer dominance. Optimization budget: one profile, at most two algorithmic/kernel attempts, one rebenchmark each. Timing JSON remains separate from correctness evidence. Historical schema-v2 timing artifacts remain immutable and are not reinterpreted as current evidence.\n",
)
replace_once(
    "- [ ] Every supported new public covariance works NumPy/CuPy/Torch without fallback.\n",
    "- [ ] Every supported new public covariance works NumPy/CuPy/Torch without fallback.\n- [ ] Exact rank-deficient fits preserve identified fit-space quantities while ordinary coordinate-wise BSE/test/p-value/CI output is explicitly unavailable.\n- [ ] `PanelOLS.predict()` and `RandomEffects.predict()` execute linear prediction on the selected numerical backend before returning the historical NumPy-visible output.\n",
)
replace_once(
    "- [ ] Performance gate has no material unresolved regression.\n",
    "- [ ] Fresh schema-v3 performance gate contains 60 synchronized rows, including the two CuPy/Torch unbalanced two-way FE cases, with no material unresolved regression.\n",
)

PATH.write_text(text, encoding="utf-8")
print("PR126 reviewed plan synchronized")
