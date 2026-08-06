"""Documentation contracts for CoxPHCV staged-screening safety."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "path",
    [
        "docs/en/guides/cox-cv-staged-safety.md",
        "docs/cn/guides/cox-cv-staged-safety.md",
    ],
)
def test_staged_safety_guides_publish_backend_and_diagnostic_contract(path):
    text = Path(path).read_text(encoding="utf-8")
    required = {
        "STATGPU_COXPHCV_TWO_STAGE",
        "STATGPU_COXPHCV_SUCCESSIVE_HALVING",
        "exhaustive_safety_fallback",
        "single_pass_exhaustive",
        "two_stage_requested",
        "two_stage_enabled",
        "successive_halving_requested",
        "successive_halving_enabled",
        "fast_pass_candidate_mask",
        "full_precision_candidate_mask",
        "screened_out_candidate_mask",
        "CuPy",
        "Torch",
    }
    assert all(token in text for token in required)


@pytest.mark.parametrize(
    "index_path",
    ["docs/en/README.md", "docs/cn/README.md"],
)
def test_staged_safety_guides_are_linked_from_language_indexes(index_path):
    text = Path(index_path).read_text(encoding="utf-8")
    assert "guides/cox-cv-staged-safety.md" in text


@pytest.mark.parametrize(
    "path,obsolete_phrases",
    [
        (
            "docs/en/models/coxph.md",
            (
                "reuses it across every staged penalty pass",
                "stages repeat fold preparation",
                "Current audited evidence",
            ),
        ),
        (
            "docs/cn/models/coxph.md",
            (
                "由所有 staged penalty pass 复用",
                "超限时各 stage 会重新准备 fold",
                "当前可审计证据",
            ),
        ),
    ],
)
def test_primary_cox_model_pages_publish_single_pass_and_durable_evidence(
    path, obsolete_phrases
):
    text = Path(path).read_text(encoding="utf-8")
    assert 'staged_safety_strategy="single_pass_exhaustive"' in text
    assert "ebbb7f2401f45b124069a30d3510c139" in text
    assert "e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb" in text
    for phrase in obsolete_phrases:
        assert phrase not in text
