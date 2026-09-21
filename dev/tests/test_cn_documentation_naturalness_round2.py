"""Regression checks for natural Chinese prose in the second cleanup batch."""

import re
from pathlib import Path


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
    "## strict/approx 差异（strict/approx difference）",
)

_FORBIDDEN_PROSE = re.compile(
    r"(?<![A-Za-z0-9_])(?:estimator|backend-native|candidate fit|final refit|"
    r"selection 来源|history 与 cache|benchmark baseline|production estimator code|"
    r"strict inference|approximation fallback reason|ElasticNet wrapper|"
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
