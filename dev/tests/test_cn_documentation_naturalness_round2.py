"""Regression checks for natural Chinese prose in the second cleanup batch."""

import re
from inspect import signature
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

_PAGES = (
    "docs/cn/guides/distribution-api.md",
    "docs/cn/models/linear-regression.md",
    "docs/cn/models/logistic-regression.md",
    "docs/cn/models/semiparametric.md",
    "docs/cn/models/nonparametric.md",
    "docs/cn/models/elastic-net.md",
    "docs/cn/models/feature-selection.md",
    "docs/cn/models/unsupervised.md",
    "docs/cn/unsupervised/README.md",
    "docs/cn/models/coxph.md",
)

_FORBIDDEN_HEADINGS = (
    "## 概览（Overview）",
    "## 路径（Path）",
    "## 外部验证（External Validation）",
)

_FORBIDDEN_PROSE = re.compile(
    r"(?<![A-Za-z0-9_])(?:estimator|backend-native|candidate fit|final refit|"
    r"selection 来源|history 与 cache|benchmark baseline|production estimator code|"
    r"strict inference|approximation fallback reason|ElasticNet wrapper|"
    r"direct-fit solver|核回归 以|核回归 输出|不同的 交叉验证折|"
    r"失败组 × 样本|penalty 口径)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _prose(text: str) -> str:
    lines = []
    fenced = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"`[^`]*`", "", line)
        # Link destinations are identifiers/paths, not translated prose.
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        # The rule targets mixed Chinese/English explanatory prose, not
        # English-only bibliographic titles or paper names.
        if re.search(r"[\u3400-\u9fff]", line):
            lines.append(line)
    return "\n".join(lines)


def test_second_batch_uses_chinese_first_headings():
    for path in _PAGES:
        text = (_ROOT / path).read_text(encoding="utf-8")
        for heading in _FORBIDDEN_HEADINGS:
            assert heading not in text, f"{path}: stale bilingual heading {heading!r}"


def test_second_batch_avoids_known_translationese_fragments():
    for path in _PAGES:
        text = _prose((_ROOT / path).read_text(encoding="utf-8"))
        match = _FORBIDDEN_PROSE.search(text)
        assert match is None, f"{path}: mixed-language prose {match.group(0)!r}"


def test_cleaned_learner_pages_avoid_internal_validation_inventories():
    cox = (_ROOT / "docs/cn/models/coxph.md").read_text(encoding="utf-8")
    index = (_ROOT / "docs/cn/unsupervised/README.md").read_text(encoding="utf-8")
    for fragment in ("remote-full-final-promotion-suite", "134/134", "机器 schema"):
        assert fragment not in cox
    for fragment in ("dev/tests/", "dev/benchmarks/", "results/"):
        assert fragment not in index


# The standalone solver capability may evolve (issue #131 / PR #172).
# Guard agreement with the public API, rather than permanently forbidding it.
_NO_SOLVER_STATEMENTS = {
    "en": "This estimator does not expose a `solver` selection parameter.",
    "cn": "该估计器不提供 `solver` 选择参数。",
}


def _check_logistic_solver_documentation(estimator, text, language):
    constructor = signature(estimator)
    has_solver = "solver" in constructor.parameters
    limitation = _NO_SOLVER_STATEMENTS[language]
    assert (limitation in text) == (not has_solver), (
        f"{language}: solver limitation disagrees with the public constructor"
    )
    for control in ("IRLS", "max_iter", "tol"):
        assert control in text, f"{language}: missing fitting control {control}"

    documented_solvers = set(re.findall(r"""\bsolver\s*=\s*["']([^"']+)["']""", text))
    if has_solver:
        assert documented_solvers, f"{language}: document concrete solver choices"
        for solver in documented_solvers:
            # Check keyword acceptance and constructor validation, without
            # fitting data or requiring an optional numerical backend.
            constructor.bind_partial(solver=solver)
            estimator(solver=solver)
    else:
        assert not documented_solvers, f"{language}: unsupported solver keyword"
        assert "lbfgs" not in text.lower().replace("-", ""), (
            f"{language}: unsupported L-BFGS capability"
        )


def test_logistic_documentation_matches_public_solver_capability():
    from statgpu.linear_model import LogisticRegression

    for language in ("en", "cn"):
        text = (_ROOT / f"docs/{language}/models/logistic-regression.md").read_text(
            encoding="utf-8"
        )
        _check_logistic_solver_documentation(LogisticRegression, text, language)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_solver_documentation_guard_accepts_current_and_extended_apis(language):
    def irls_only():
        pass

    def selectable(*, solver="auto"):
        if solver not in ("auto", "irls", "lbfgs"):
            raise ValueError("unsupported solver")

    controls = "IRLS max_iter tol"
    limitation = controls + " " + _NO_SOLVER_STATEMENTS[language]
    selection = controls + ' solver="auto" solver="irls" solver="lbfgs"'

    _check_logistic_solver_documentation(irls_only, limitation, language)
    _check_logistic_solver_documentation(selectable, selection, language)

    # Neither retaining stale prose after capability expansion nor advertising
    # future support before it exists is a passing documentation state.
    with pytest.raises(AssertionError):
        _check_logistic_solver_documentation(selectable, limitation, language)
    with pytest.raises(AssertionError):
        _check_logistic_solver_documentation(irls_only, selection, language)
    with pytest.raises(AssertionError):
        _check_logistic_solver_documentation(selectable, controls, language)
    with pytest.raises(AssertionError):
        _check_logistic_solver_documentation(
            irls_only, limitation + ' solver="lbfgs"', language
        )
    with pytest.raises(ValueError, match="unsupported solver"):
        _check_logistic_solver_documentation(
            selectable, controls + ' solver="unknown"', language
        )


def test_prose_guard_ignores_link_paths_but_keeps_visible_labels():
    assert _FORBIDDEN_PROSE.search(_prose("参见[共享方法](../reference/estimator-api.md)")) is None
    assert _FORBIDDEN_PROSE.search(_prose("这个 estimator 需要重拟合")) is not None
