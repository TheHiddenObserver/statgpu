"""Guard the model-page wording and numerical distinctions clarified in #168."""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.survival import CoxPH

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "dev/references/coxph-implementation-and-evidence.md"


def _text(path):
    return (ROOT / path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "page,fragments",
    (
        ("elastic-net", ("KKT 次梯度违反", "标准化设计侧", "解析权重", "去偏报告截距")),
        ("coxph", ("保守门禁", "收敛证书", "预算耗尽", "具有组合复杂度")),
        ("feature-selection", ("高斯风格", "该退路", "全部 worker")),
        ("linear-regression", ("worker 循环",)),
        ("logistic-regression", ("sandwich 的 bread", "解析权重")),
    ),
)
def test_model_pages_avoid_reviewed_translationese(page, fragments):
    text = _text(f"docs/cn/models/{page}.md")
    for fragment in fragments:
        assert fragment not in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_cox_separates_exact_calculation_from_asymptotic_inference(language):
    text = _text(f"docs/{language}/models/coxph.md")
    if language == "en":
        assert "large-sample normal approximation" in text
        assert "finite-sample exact tests" in text
        assert not re.search(r"both (?:are exact|use exact supported inference)", text)
        assert "likelihood is combinatorial" not in text
        assert "duplicating the observations doubles" not in text
        assert "dynamic programming" in text
    else:
        assert "大样本正态近似" in text
        assert "有限样本精确检验" in text
        assert not re.search(r"两者均执行(?:已支持的)?精确推断", text)
        assert "具有组合复杂度" not in text
        assert "复制全部观测会使似然与得分贡献加倍" not in text
        assert "动态规划" in text
    assert "coxph-implementation-and-evidence.md" in text
    # Exact-source provenance belongs in the developer reference, not a learner page.
    assert "a726937a39eb0ed5a370dd03362884b63a9e9818" not in text
    assert "e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb" not in text


def test_cox_reference_retains_source_bound_evidence_and_configuration():
    text = REFERENCE.read_text(encoding="utf-8")
    for required in (
        "a726937a39eb0ed5a370dd03362884b63a9e9818",
        "e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb",
        "https://gist.github.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139",
        "pr80_final_gpu_suite_schema3.json",
        "../reviews/pr80_review_fix.md",
        "STATGPU_EXACT_NESTED_MAX_BYTES",
        "STATGPU_EXACT_BATCH_MAX_BYTES",
        "STATGPU_COX_GROUP_MAX_BYTES",
        "STATGPU_TORCH_EXACT_SCAN_MIN_ROWS",
        "STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS",
        "STATGPU_TORCH_EXACT_SCAN_STRATEGY",
        "512 MiB",
        "2,048",
        "64",
        "later commits do\nnot automatically inherit that evidence",
    ):
        assert required in text


def test_robust_cox_modes_share_normal_reference_inference():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(718)
    X = rng.normal(size=(120, 2))
    event_time = rng.exponential(scale=np.exp(-(X @ [0.4, -0.3])))
    censor_time = rng.exponential(scale=2.0, size=len(X))
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.int64)
    fits = []
    for mode in ("strict", "approx"):
        model = CoxPH(
            device="cpu", ties="efron", cov_type="hc1",
            inference_mode=mode, compute_cindex=False,
        ).fit(X, time, event)
        assert model.converged_
        assert model.inference_approximate_ is False
        np.testing.assert_allclose(
            model._pvalues, 2 * stats.norm.sf(np.abs(model._zvalues)),
        )
        delta = stats.norm.ppf(0.975) * model._bse
        np.testing.assert_allclose(
            model._conf_int, np.column_stack((model.coef_ - delta, model.coef_ + delta)),
        )
        fits.append(model)
    for name in ("coef_", "_bse", "_pvalues", "_conf_int"):
        np.testing.assert_allclose(getattr(fits[0], name), getattr(fits[1], name))
