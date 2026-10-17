"""Freeze the bilingual directory migration without weakening body-level tests.

The independent page inventory and baseline anchor fingerprints prevent a
truncated migration report from silently dropping canonical pages or fragments.
Statistical examples and API inventories remain in their existing test modules.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
import pytest

from doc_paths import (
    MODEL_PAGES,
    linked_pages,
    markdown_anchors,
    model_doc,
    peer_link,
    reachable_pages,
)

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = json.loads((ROOT / "dev/reviews/pr168-document-directory-migration-map.json").read_text())
BASELINE = json.loads(Path(__file__).with_name("document_migration_baseline.json").read_text())
ENTRIES = MIGRATION["pages"]
LANGUAGES = ("en", "cn")
FAMILIES = {
    "models/regression/README": {
        "linear-regression", "ridge", "lasso", "elastic-net", "adaptive-lasso",
        "scad", "mcp", "generalized-linear-model", "logistic-regression",
        "poisson-regression", "ordered", "quantile", "robust",
    },
    "models/testing/README": {
        "anova", "anova-oneway", "anova-welch", "anova-twoway", "anova-posthoc",
        "multiple-testing", "multiple-testing-fwer", "multiple-testing-fdr",
        "pvalue-combination",
    },
    "models/covariance/README": {
        "covariance", "covariance-empirical", "covariance-shrunk",
        "covariance-ledoit-wolf", "covariance-oas", "covariance-min-cov-det",
        "covariance-graphical-lasso", "covariance-graphical-lasso-cv",
    },
    "models/nonparametric/README": {
        "nonparametric", "kernel-methods", "kernel-ridge", "kernel-ridge-cv",
        "kernel-pca", "kernel-nystroem", "splines", "semiparametric",
    },
    "models/feature-selection/README": {
        "feature-selection", "knockoff", "knockoff-fixed-x", "knockoff-model-x",
    },
    "models/survival/README": {"coxph"},
    "panel/README": {"panel"},
    "unsupervised/README": {"unsupervised"},
}


# The original manifest and fingerprints stay frozen. Only the subsequently
# split multiple-testing sections relocate, to explicit, individually checked
# destinations. No unrelated legacy fragment may disappear behind this map.
MULTIPLE_TESTING_RELOCATIONS = {
    "en": {
        "adjustment-formulas": ("models/testing/multiple-testing-fwer", "adjustment-formulas"),
        "make-individual-decisions-with-holm": ("models/testing/multiple-testing-fwer", "make-individual-decisions-with-holm"),
        "compare-with-bh-for-fdr-control": ("models/testing/multiple-testing-fdr", "compare-with-bh-for-fdr-control"),
        "ask-a-global-question-with-fisher": ("models/testing/pvalue-combination", "ask-a-global-question-with-fisher"),
        "combination-formulas-and-assumptions": ("models/testing/pvalue-combination", "combination-formulas-and-assumptions"),
        "fisher-add-log-evidence": ("models/testing/pvalue-combination", "fisher-add-log-evidence"),
        "stouffer-add-standardized-evidence": ("models/testing/pvalue-combination", "stouffer-add-standardized-evidence"),
        "cauchyacat-emphasize-small-p-values-through-a-heavy-tail": ("models/testing/pvalue-combination", "cauchyacat-emphasize-small-p-values-through-a-heavy-tail"),
        "choose-a-combination-method-and-its-weights": ("models/testing/pvalue-combination", "choose-a-combination-method-and-its-weights"),
        "practical-limits": ("guides/multiple-testing-combine-pvalues", "numerical-limits"),
    },
    "cn": {
        "校正公式": ("models/testing/multiple-testing-fwer", "校正公式"),
        "用-holm-逐项作出判断": ("models/testing/multiple-testing-fwer", "用-holm-逐项作出判断"),
        "对照-bh-的-fdr-控制": ("models/testing/multiple-testing-fdr", "对照-bh-的-fdr-控制"),
        "用-fisher-回答总体问题": ("models/testing/pvalue-combination", "用-fisher-回答总体问题"),
        "合并公式与适用条件": ("models/testing/pvalue-combination", "合并公式与适用条件"),
        "fisher累加对数证据": ("models/testing/pvalue-combination", "fisher累加对数证据"),
        "stouffer累加标准化证据": ("models/testing/pvalue-combination", "stouffer累加标准化证据"),
        "cauchyacat用重尾变换突出小-p-值": ("models/testing/pvalue-combination", "cauchyacat用重尾变换突出小-p-值"),
        "选择合并方法及权重": ("models/testing/pvalue-combination", "选择合并方法及权重"),
        "实际使用限制": ("guides/multiple-testing-combine-pvalues", "数值限制"),
    },
}


def _digest(anchors):
    return hashlib.sha256("\n".join(sorted(anchors)).encode()).hexdigest()


def _links(text):
    """Read Markdown destinations, ignoring examples and preserving fragments."""
    for token in MarkdownIt().parse(text):
        for child in token.children or ():
            if child.type == "link_open":
                yield child.attrGet("href")


def test_complete_bijective_mapping_matches_the_independent_page_inventory():
    expected = {
        f"docs/{language}/models/{name}.md": f"docs/{language}/{canonical}.md"
        for language in LANGUAGES
        for name, canonical in MODEL_PAGES.items()
        # These three pages were added after the frozen directory migration.
        if name not in {"coxph", "multiple-testing-fwer", "multiple-testing-fdr",
                        "pvalue-combination"}
    }
    actual = {entry["old"]: entry["new"] for entry in ENTRIES}
    assert len(ENTRIES) == len(actual) == len(set(actual.values())) == 84
    assert actual == expected
    assert set(BASELINE["pages"]) == set(expected)
    assert MIGRATION["baseline"] == BASELINE["source_commit"] == "180ddc41cef544639d5c3b874430cb39a690b375"
    assert sum(item["anchor_count"] for item in BASELINE["pages"].values()) == 1763
    for language in LANGUAGES:
        flat = {page.name for page in (ROOT / "docs" / language / "models").glob("*.md")}
        assert flat == {"coxph.md", "README.md"}
        retained = ROOT / "docs" / language / (model_doc("coxph") + ".md")
        assert "<!-- example: coxph-cpu-walkthrough -->" in retained.read_text()
        assert str(retained.relative_to(ROOT)) not in actual


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["old"])
def test_old_paths_are_removed_and_canonical_bodies_keep_original_fragments(entry):
    old, new = ROOT / entry["old"], ROOT / entry["new"]
    assert not old.exists(), "Removed paths must not reappear as compatibility stubs"
    assert new.is_file()
    new_text = new.read_text()
    assert not new_text.startswith(("# Documentation moved", "# 文档已迁移"))
    baseline = BASELINE["pages"][entry["old"]]
    assert len(entry["anchors"]) == baseline["anchor_count"]
    assert _digest(entry["anchors"]) == baseline["anchors_sha256"]
    if new.name == "multiple-testing.md":
        language = Path(entry["new"]).parts[1]
        relocations = MULTIPLE_TESTING_RELOCATIONS[language]
        assert len(relocations) == 10
        assert set(relocations) < set(entry["anchors"].values())
        for anchor in entry["anchors"].values():
            if anchor in relocations:
                page, fragment = relocations[anchor]
                destination = ROOT / "docs" / language / (page + ".md")
                assert destination.is_file(), (anchor, destination)
                assert fragment in markdown_anchors(destination.read_text()), (anchor, destination)
            else:
                assert anchor in markdown_anchors(new_text)
    else:
        assert set(entry["anchors"].values()) <= markdown_anchors(new_text)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["new"])
def test_canonical_bodies_have_the_matching_language_peer(entry):
    path = ROOT / entry["new"]
    _, language, *relative = Path(entry["new"]).with_suffix("").parts
    page = "/".join(relative)
    target = peer_link(language, page)
    assert target in set(_links(path.read_text())), (path, target)
    peer = (path.parent / target).resolve()
    assert peer.is_file()
    other = "cn" if language == "en" else "en"
    assert peer == (ROOT / "docs" / other / (page + ".md")).resolve()
    assert peer_link(other, page) in set(_links(peer.read_text()))


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("family,members", FAMILIES.items())
def test_family_readmes_link_every_canonical_child_directly(language, family, members):
    language_root = ROOT / "docs" / language
    path = language_root / (family + ".md")
    assert path.is_file()
    assert peer_link(language, family) in set(_links(path.read_text()))
    destinations = linked_pages(path) | {path.resolve()}
    expected = {(language_root / (model_doc(member) + ".md")).resolve() for member in members}
    assert expected <= destinations, (family, expected - destinations)


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("entry", ("README", "usage", "models/README"))
def test_layered_entry_points_reach_all_families_and_bodies(language, entry):
    language_root = ROOT / "docs" / language
    reached = reachable_pages(language_root / (entry + ".md"), language_root)
    expected = {
        (language_root / (relative + ".md")).resolve()
        for relative in set(MODEL_PAGES.values()) | set(FAMILIES)
    }
    assert expected <= reached, (entry, expected - reached)


def test_repository_links_use_canonical_paths_and_valid_migrated_fragments():
    legacy = {(ROOT / entry["old"]).resolve() for entry in ENTRIES}
    canonical = {(ROOT / entry["new"]).resolve() for entry in ENTRIES}
    anchors = {path: markdown_anchors(path.read_text()) for path in canonical}
    # Parse actual Markdown/HTML links throughout the repository. Historical
    # plain-text paths and source manifests are evidence, not live navigation.
    for source in ROOT.rglob("*.md"):
        text = source.read_text()
        targets = list(_links(text))
        for token in MarkdownIt().parse(text):
            for child in [token] + (token.children or []):
                if child.type in {"html_inline", "html_block"}:
                    targets.extend(re.findall(r'href=["\']([^"\']+)', child.content))
        for target in targets:
            link = urlsplit(target)
            if link.scheme or link.netloc:
                # Commit-specific URLs deliberately refer to historical trees.
                match = re.fullmatch(
                    r"/TheHiddenObserver/statgpu/(?:blob|tree)/(?:master|main)/(.+)",
                    unquote(link.path),
                ) if link.netloc.lower() == "github.com" else None
                if not match:
                    continue
                destination = (ROOT / match.group(1)).resolve()
            else:
                destination = ((ROOT / unquote(link.path).lstrip("/")) if link.path.startswith("/")
                               else (source.parent / unquote(link.path))).resolve() if link.path else source.resolve()
            assert destination not in legacy, (source, target)
            if destination in canonical:
                assert destination.is_file(), (source, target)
                if link.fragment:
                    assert unquote(link.fragment) in anchors[destination], (source, target)


def test_runtime_documentation_references_do_not_point_to_removed_paths():
    old_paths = {entry["old"] for entry in ENTRIES}
    for source in (ROOT / "statgpu").rglob("*.py"):
        text = source.read_text()
        assert not any(old in text for old in old_paths), source
    lasso = (ROOT / "statgpu/linear_model/wrappers/_lasso.py").read_text()
    assert "docs/en/models/regression/lasso.md" in lasso


def test_heading_parser_preserves_rendered_labels_duplicate_slugs_and_explicit_ids():
    text = '# [A linked heading](other.md)\n\n# A linked heading\n\n<a id="legacy"></a>\n\nSetext\n------\n\n```text\n# Not a heading\n```\n'
    assert markdown_anchors(text) == {"a-linked-heading", "a-linked-heading-1", "legacy", "setext"}
