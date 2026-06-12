"""Tests for the full-matrix benchmark report parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_parser_module():
    path = Path(__file__).with_name("_bench_report_parser.py")
    spec = importlib.util.spec_from_file_location("_bench_report_parser", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_full_matrix_section_a_output_fixture():
    parser = _load_parser_module()
    fixture = Path(__file__).with_name("_bench_full_matrix_output_section_A.txt")

    summary = parser.parse_benchmark_text(fixture.read_text(encoding="utf-8"))

    assert summary["counts"]["gpu_rows"] == 816
    assert summary["counts"]["groups"] > 0
    assert summary["backend_counts"]["cupy"] == 408
    assert summary["backend_counts"]["torch"] == 408
    assert summary["total_summary"]["passed"] == 816
    assert summary["total_summary"]["status"] == "ALL PASS"
    assert summary["section_summaries"][0]["section"] == "Section A"


def test_markdown_summary_contains_key_sections():
    parser = _load_parser_module()
    text = """
  [squared_error+l2 | n=10,p=3 | solvers=exact]
  Solver         Backend    Time(ms)    Iters    NNZ      ||coef||          vs_CPU       spd
  exact          CPU             1.0        1      3      1.000000               -         -
  exact          cupy            0.5        1      3      1.000000        1.00e-12     2.00x
  Section A: 2/2 passed (max diff: 1.00e-12)  [PASS]
  TOTAL: 2/2 passed  [ALL PASS]
"""

    summary = parser.parse_benchmark_text(text)
    markdown = parser.summary_to_markdown(summary)

    assert "Benchmark Summary" in markdown
    assert "Backend Rows" in markdown
    assert "Section A: 2/2 passed" in markdown
    assert "Fastest GPU Rows" in markdown


def test_fail_on_alerts_returns_nonzero_for_precision_alert(tmp_path):
    parser = _load_parser_module()
    bench_log = tmp_path / "bench.txt"
    bench_log.write_text(
        """
  [squared_error+l2 | n=10,p=3 | solvers=exact]
  Solver         Backend    Time(ms)    Iters    NNZ      ||coef||          vs_CPU       spd
  exact          CPU             1.0        1      3      1.000000               -         -
  exact          cupy            0.5        1      3      1.000000        2.00e-03     2.00x
  Section A: 2/2 passed (max diff: 2.00e-03)  [PASS]
  TOTAL: 2/2 passed  [ALL PASS]
""",
        encoding="utf-8",
    )

    rc = parser.main([str(bench_log), "--fail-on-alerts", "--quiet"])

    assert rc == 1
