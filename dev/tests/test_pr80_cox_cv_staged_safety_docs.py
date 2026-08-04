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
    assert required.issubset(text)
