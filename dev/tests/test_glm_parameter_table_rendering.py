"""Render bilingual API tables so Markdown delimiters cannot hide GLM guidance."""

from html.parser import HTMLParser
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIONS = {
    "en": (
        "Ordinary IRLS adds ∥beta∥²/(4C) for positive C; C=0 removes it. "
        "Explicit newton/lbfgs/fista ignore C."
    ),
    "cn": (
        "普通 IRLS 在 C>0 时加入 ∥beta∥²/(4C)；C=0 取消惩罚。"
        "显式 newton/lbfgs/fista 不使用 C。"
    ),
}


class _RenderedTables(HTMLParser):
    """Collect visible cell text, retaining each table's level-two section."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.section = ""
        self.heading = None
        self.rows = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self.heading = []
        elif tag == "table":
            self.rows = []
        elif tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.cell = []

    def handle_data(self, data):
        if self.heading is not None:
            self.heading.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "h2":
            self.section = "".join(self.heading).strip()
            self.heading = None
        elif tag in ("th", "td"):
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)
            self.row = None
        elif tag == "table":
            self.tables.append((self.section, self.rows))
            self.rows = None


def _read_reference(language):
    return (ROOT / "docs" / language / "reference/linear-model-api.md").read_text(
        encoding="utf-8"
    )


def _render_tables(source):
    # GFM's table rule splits unescaped pipes even inside code spans. This is
    # a required test dependency, not an importorskip or a source-text proxy.
    html = MarkdownIt("commonmark").enable("table").render(source)
    parser = _RenderedTables()
    parser.feed(html)
    parser.close()
    return parser.tables


def _assert_glm_c_description(tables, language):
    parameter_label = "Parameter" if language == "en" else "参数"
    matches = [
        rows for section, rows in tables
        if section == "GeneralizedLinearModel" and rows[0][0] == parameter_label
    ]
    assert len(matches) == 1, "expected one rendered GLM parameter table"
    rows = matches[0]
    assert all(len(row) == 3 for row in rows), "expected three rendered columns"
    c_rows = [row for row in rows[1:] if row[0] == "C"]
    assert len(c_rows) == 1, "expected one rendered C parameter row"
    assert c_rows[0] == ["C", "1.0", DESCRIPTIONS[language]], (
        "rendered C description lost its norm/normalization, positive-C or "
        "C=0 behavior, or explicit-solver exceptions"
    )


@pytest.mark.parametrize("language", ["en", "cn"])
def test_glm_parameter_table_renders_complete_c_description(language):
    _assert_glm_c_description(_render_tables(_read_reference(language)), language)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_linear_model_reference_tables_render_consistent_column_counts(language):
    tables = _render_tables(_read_reference(language))
    assert tables, "reference did not render any tables"
    for section, rows in tables:
        assert rows and len(rows) > 1, section
        assert all(len(row) == len(rows[0]) for row in rows), section


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("damage", ["raw_pipes", "normalization", "zero", "solvers", "extra_column"])
def test_rendered_glm_guard_rejects_lost_content_and_extra_cells(language, damage):
    before, section = _read_reference(language).split("## GeneralizedLinearModel\n", 1)
    if damage == "raw_pipes":
        section = section.replace("∥beta∥²/(4C)", "||beta||²/(4C)")
    elif damage == "normalization":
        section = section.replace("/(4C)", "")
    elif damage == "zero":
        section = section.replace("C=0", "")
    elif damage == "solvers":
        section = section.replace("newton/lbfgs/fista", "")
    else:
        lines = section.splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("|"))
        lines[header] += " Extra |"
        lines[header + 1] += "---|"
        section = "\n".join(lines)
    damaged_source = before + "## GeneralizedLinearModel\n" + section
    with pytest.raises(AssertionError, match="rendered (C description|columns)"):
        _assert_glm_c_description(_render_tables(damaged_source), language)
