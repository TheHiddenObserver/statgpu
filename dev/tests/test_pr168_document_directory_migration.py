"""Freeze the bilingual directory migration without weakening body-level tests.

The independent page inventory and baseline anchor fingerprints prevent a
truncated migration report from silently dropping legacy pages or fragments.
Statistical examples and API inventories remain in their existing test modules.
"""

import hashlib
import json
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
        "multiple-testing",
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
    "panel/README": {"panel"},
    "unsupervised/README": {"unsupervised"},
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
        if name != "coxph"
    }
    actual = {entry["old"]: entry["new"] for entry in ENTRIES}
    assert len(ENTRIES) == len(actual) == len(set(actual.values())) == 84
    assert actual == expected
    assert set(BASELINE["pages"]) == set(expected)
    assert MIGRATION["baseline"] == BASELINE["source_commit"] == "180ddc41cef544639d5c3b874430cb39a690b375"
    assert sum(item["anchor_count"] for item in BASELINE["pages"].values()) == 1763
    for language in LANGUAGES:
        flat = {page.name for page in (ROOT / "docs" / language / "models").glob("*.md")}
        assert flat == {name + ".md" for name in MODEL_PAGES} | {"README.md"}
        retained = ROOT / "docs" / language / (model_doc("coxph") + ".md")
        assert "<!-- example: coxph-cpu-walkthrough -->" in retained.read_text()
        assert str(retained.relative_to(ROOT)) not in actual


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["old"])
def test_every_old_page_and_original_fragment_forwards_to_the_canonical_body(entry):
    old, new = ROOT / entry["old"], ROOT / entry["new"]
    assert old.is_file() and new.is_file()
    old_text, new_text = old.read_text(), new.read_text()
    assert old_text.startswith(("# Documentation moved", "# 文档已迁移"))
    assert not new_text.startswith(("# Documentation moved", "# 文档已迁移"))
    assert "```" not in old_text, "Compatibility pages must not duplicate tutorial bodies"
    baseline = BASELINE["pages"][entry["old"]]
    assert len(entry["anchors"]) == baseline["anchor_count"]
    assert _digest(entry["anchors"]) == baseline["anchors_sha256"]
    old_anchors, new_anchors = markdown_anchors(old_text), markdown_anchors(new_text)
    assert set(entry["anchors"]) <= old_anchors
    assert set(entry["anchors"].values()) <= new_anchors
    destinations = list(_links(old_text))
    resolved = {
        ((old.parent / unquote(urlsplit(target).path)).resolve(),
         unquote(urlsplit(target).fragment))
        for target in destinations
    }
    assert (new.resolve(), "") in resolved
    for old_anchor, new_anchor in entry["anchors"].items():
        # Each anchor's own pointer must preserve that fragment, not merely a
        # same-page link somewhere else on the compatibility page.
        marker = f'<a id="{old_anchor}"></a>'
        assert old_text.count(marker) == 1
        section = old_text.split(marker, 1)[1].split('<a id="', 1)[0]
        own_links = list(_links(section))
        assert len(own_links) == 1
        link = urlsplit(own_links[0])
        assert (old.parent / unquote(link.path)).resolve() == new.resolve()
        assert unquote(link.fragment) == new_anchor


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


@pytest.mark.parametrize("language", LANGUAGES)
def test_maintained_document_links_use_canonical_paths_and_valid_fragments(language):
    legacy = {(ROOT / entry["old"]).resolve() for entry in ENTRIES}
    canonical = {(ROOT / entry["new"]).resolve() for entry in ENTRIES}
    anchors = {path: markdown_anchors(path.read_text()) for path in canonical}
    paths = [ROOT / "README.md", ROOT / "docs/index.md"]
    paths.extend((ROOT / "docs" / language).rglob("*.md"))
    for source in paths:
        if source.resolve() in legacy:
            continue
        for target in _links(source.read_text()):
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            destination = (source.parent / unquote(link.path)).resolve() if link.path else source.resolve()
            assert destination not in legacy, (source, target)
            if destination in canonical and link.fragment:
                assert unquote(link.fragment) in anchors[destination], (source, target)


def test_heading_parser_preserves_rendered_labels_duplicate_slugs_and_explicit_ids():
    text = '# [A linked heading](other.md)\n\n# A linked heading\n\n<a id="legacy"></a>\n\nSetext\n------\n\n```text\n# Not a heading\n```\n'
    assert markdown_anchors(text) == {"a-linked-heading", "a-linked-heading-1", "legacy", "setext"}
