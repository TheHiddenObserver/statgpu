"""Keep device guidance actionable without turning implementation policy into API promises."""
import re
from pathlib import Path

import numpy as np
import pytest

from dev.tests.doc_examples import run_example

ROOT = Path(__file__).resolve().parents[2]


def _assert_cpu_workaround(text):
    paragraphs = text.split("\n\n")
    assert any(
        all(term in paragraph for term in ("NumPy", 'device="cpu"', 'backend="numpy"'))
        and re.search(r"alternative|替代", paragraph, re.IGNORECASE)
        for paragraph in paragraphs
    )


@pytest.mark.parametrize("language", ("en", "cn"))
def test_device_guide_cpu_example_is_self_contained(language):
    path = ROOT / f"docs/{language}/guides/device-and-memory.md"
    result = run_example(path.read_text(encoding="utf-8"), "device-cpu", path,
                         allow_legacy=True)
    assert result["model"].device == "cpu"
    assert isinstance(result["predictions"], np.ndarray)
    assert result["predictions"].shape == (3,)
    assert np.isfinite(result["predictions"]).all()


@pytest.mark.parametrize("language", ("en", "cn"))
def test_device_guide_keeps_settings_exceptions_and_observable_checks(language):
    text = (ROOT / f"docs/{language}/guides/device-and-memory.md").read_text(encoding="utf-8")
    _assert_cpu_workaround(text)
    for value in ("cpu", "cuda", "torch", "auto"):
        assert f'device="{value}"' in text
    for term in ("statgpu.set_device", "get_device()", "KernelDensityEstimator",
                 "KernelRegression", "SplineTransformer", "KernelPCA", "Nystroem",
                 "samples_", "knots_", ".is_cuda", ".device", 'backend="numpy"',
                 'backend="torch"', "return_cpu=False", "Ridge.coef_"):
        assert term in text, term
    # These were known audience defects, not a ban on statistical/algorithm terms.
    for term in ("DLPack", "pinned memory", "intended convention", "预期约定",
                 "as kernels and benchmarks improve", "随着数值内核和基准测试",
                 "does not duplicate those matrices", "本页不再重复"):
        assert term not in text, term
    if language == "en":
        assert re.search(r"restor\w*[^.]*polic", text, re.IGNORECASE)
        assert re.search(r"(?:do not accept|reject)[^.]*CPU result", text)
        assert "penalized GLMs" in text and "global GPU setting" in text
    else:
        assert re.search(r"恢复[^。]*策略", text)
        assert re.search(r"(?:不要接受|拒绝)[^。]*CPU 结果", text)
        assert "惩罚 GLM" in text and "全局设置为 GPU" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_device_workaround_guard_rejects_missing_cpu_instructions(language):
    text = (ROOT / f"docs/{language}/guides/device-and-memory.md").read_text(encoding="utf-8")
    damaged = "\n\n".join(
        paragraph for paragraph in text.split("\n\n")
        if not all(term in paragraph for term in
                   ("NumPy", 'device="cpu"', 'backend="numpy"'))
    )
    assert damaged != text
    with pytest.raises(AssertionError):
        _assert_cpu_workaround(damaged)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_memory_guidance_retains_cost_and_live_state_limits(language):
    text = (ROOT / f"docs/{language}/guides/device-and-memory.md").read_text(encoding="utf-8")
    assert "gpu_memory_cleanup=False" in text
    assert "gpu_memory_cleanup=True" in text
    assert "predict()" in text and "score()" in text
    if language == "en":
        assert "live arrays" in text and "peak memory" in text
        assert re.search(r"allocat\w*[^.]*again|reallocat", text)
    else:
        assert "仍被引用的数组" in text and "峰值显存" in text
        assert "重新分配内存" in text


def _assert_adaptive_penalty_row(text):
    from dev.tests.test_glm_parameter_table_rendering import _render_tables

    rows = [row for _, table in _render_tables(text) for row in table
            if row and row[0] == "adaptive_l1"]
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 4
    assert "α" in row[1] and "w_j" in row[1] and "|β_j|" in row[1]
    assert "alpha" in row[3] and "weights" in row[3]


@pytest.mark.parametrize("language", ("en", "cn"))
def test_adaptive_l1_penalty_and_controls_survive_markdown_rendering(language):
    path = ROOT / f"docs/{language}/guides/solver-penalty-matrix.md"
    _assert_adaptive_penalty_row(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("damage", ("raw_pipes", "missing_weight"))
def test_rendered_adaptive_penalty_guard_rejects_truncation(language, damage):
    path = ROOT / f"docs/{language}/guides/solver-penalty-matrix.md"
    text = path.read_text(encoding="utf-8")
    if damage == "raw_pipes":
        text = text.replace(r"\|β_j\|", "|β_j|")
    else:
        text = text.replace("w_j", "")
    with pytest.raises(AssertionError):
        _assert_adaptive_penalty_row(text)
