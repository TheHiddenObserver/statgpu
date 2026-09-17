"""Cross-validation documentation ownership contracts."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def test_cv_public_guide_links_public_design_layer():
    en = _read("docs/en/guides/cross-validation.md")
    cn = _read("docs/cn/guides/cross-validation.md")

    assert "[How statgpu Cross-Validation Works](cross-validation-design.md)" in en
    assert "[statgpu 的交叉验证如何工作](cross-validation-design.md)" in cn

    for text in (en, cn):
        assert "cv_solver" in text
        assert "final refit" in text
        assert "_effective_cv_device" not in text
        assert "_array_identity_token" not in text
        assert "_make_lasso_cv_auto_cache_key" not in text
        assert "STATGPU_LASSO_CV_CACHE_SIZE" not in text


def test_cv_public_design_explains_stable_execution_model():
    en = _read("docs/en/guides/cross-validation-design.md")
    cn = _read("docs/cn/guides/cross-validation-design.md")

    required_en = (
        "selection work",
        "final refit",
        "Statistical invariants versus execution freedom",
        "Pathwise reuse and warm starts",
        "GPU batching",
        "Selection caches",
        "Device selection during CV",
        "Weights must remain consistent across the CV lifecycle",
        "Why Cox CV is structurally different",
        "Inference after tuning",
    )
    for fragment in required_en:
        assert fragment in en

    required_cn = (
        "selection work",
        "final refit",
        "统计不变量与执行自由度",
        "Pathwise reuse 与 warm start",
        "GPU batching",
        "Selection cache",
        "CV 中的 device 选择",
        "权重必须贯穿整个 CV lifecycle",
        "为什么 Cox CV 在结构上不同",
        "Tuning 完成后的 inference",
    )
    for fragment in required_cn:
        assert fragment in cn

    assert "[Cross-Validation](cross-validation.md)" in en
    assert "[交叉验证](cross-validation.md)" in cn


def test_cv_public_design_does_not_publish_private_implementation_contracts():
    en = _read("docs/en/guides/cross-validation-design.md")
    cn = _read("docs/cn/guides/cross-validation-design.md")

    private_tokens = (
        "_effective_cv_device",
        "_array_identity_token",
        "_make_lasso_cv_auto_cache_key",
        "STATGPU_LASSO_CV_CACHE_SIZE",
        "dev/tests/",
        "dev/benchmarks/",
        "dev/reviews/",
        "PR #",
        "Issue #",
    )
    for text in (en, cn):
        for token in private_tokens:
            assert token not in text

    assert "maintained route" not in en.lower()
    assert "fail closed" not in en.lower()
    assert "维护中的路径" not in cn
    assert "维护路径" not in cn
    assert "fail closed" not in cn.lower()


def test_cv_internal_design_keeps_private_contracts_and_names_public_layers():
    internal = _read("dev/design/CROSS_VALIDATION.md")

    for fragment in (
        "docs/en/guides/cross-validation.md",
        "docs/en/guides/cross-validation-design.md",
        "_effective_cv_device()",
        "_array_identity_token",
        "_make_lasso_cv_auto_cache_key",
        "gpu_cv_mixed_precision",
        "STATGPU_LASSO_CV_CACHE_SIZE",
        "64",
        "private call graphs",
        "heuristic thresholds",
    ):
        assert fragment in internal


def test_cv_design_is_discoverable_from_user_portals():
    en_readme = _read("docs/en/README.md")
    cn_readme = _read("docs/cn/README.md")
    en_usage = _read("docs/en/usage.md")
    cn_usage = _read("docs/cn/usage.md")

    assert "guides/cross-validation-design.md" in en_readme
    assert "guides/cross-validation-design.md" in cn_readme
    assert "guides/cross-validation-design.md" in en_usage
    assert "guides/cross-validation-design.md" in cn_usage


def test_documentation_policy_defines_public_design_layer():
    style = _read("dev/DOCUMENTATION_STYLE.md")

    assert "## Public design / architecture pages" in style
    assert "stable execution model" in style
    assert "conceptual acceleration strategies" in style
    assert "Exact helper names" in style
    assert "heuristic thresholds" in style
