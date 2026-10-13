"""Canonical documentation names for tests that span several model families.

This explicit inventory is intentionally independent of the migration report:
regression tests must read the maintained body, never the old compatibility page.
The migration tests compare this inventory with every report entry.
"""

import posixpath
import re
from urllib.parse import unquote, urlsplit


MODEL_PAGES = {'adaptive-lasso': 'models/regression/adaptive-lasso',
 'anova': 'models/testing/anova',
 'anova-oneway': 'models/testing/anova-oneway',
 'anova-posthoc': 'models/testing/anova-posthoc',
 'anova-twoway': 'models/testing/anova-twoway',
 'anova-welch': 'models/testing/anova-welch',
 'covariance': 'models/covariance/README',
 'covariance-empirical': 'models/covariance/covariance-empirical',
 'covariance-graphical-lasso': 'models/covariance/covariance-graphical-lasso',
 'covariance-graphical-lasso-cv': 'models/covariance/covariance-graphical-lasso-cv',
 'covariance-ledoit-wolf': 'models/covariance/covariance-ledoit-wolf',
 'covariance-min-cov-det': 'models/covariance/covariance-min-cov-det',
 'covariance-oas': 'models/covariance/covariance-oas',
 'covariance-shrunk': 'models/covariance/covariance-shrunk',
 'coxph': 'models/coxph',
 'elastic-net': 'models/regression/elastic-net',
 'feature-selection': 'models/feature-selection/README',
 'generalized-linear-model': 'models/regression/generalized-linear-model',
 'kernel-methods': 'models/nonparametric/kernel-methods',
 'kernel-nystroem': 'models/nonparametric/kernel-nystroem',
 'kernel-pca': 'models/nonparametric/kernel-pca',
 'kernel-ridge': 'models/nonparametric/kernel-ridge',
 'kernel-ridge-cv': 'models/nonparametric/kernel-ridge-cv',
 'knockoff': 'models/feature-selection/knockoff',
 'knockoff-fixed-x': 'models/feature-selection/knockoff-fixed-x',
 'knockoff-model-x': 'models/feature-selection/knockoff-model-x',
 'lasso': 'models/regression/lasso',
 'linear-regression': 'models/regression/linear-regression',
 'logistic-regression': 'models/regression/logistic-regression',
 'losses': 'reference/losses',
 'mcp': 'models/regression/mcp',
 'multiple-testing': 'models/testing/multiple-testing',
 'nonparametric': 'models/nonparametric/nonparametric',
 'ordered': 'models/regression/ordered',
 'panel': 'panel/README',
 'poisson-regression': 'models/regression/poisson-regression',
 'quantile': 'models/regression/quantile',
 'ridge': 'models/regression/ridge',
 'robust': 'models/regression/robust',
 'scad': 'models/regression/scad',
 'semiparametric': 'models/nonparametric/semiparametric',
 'splines': 'models/nonparametric/splines',
 'unsupervised': 'unsupervised/README'}


def model_doc(name):
    """Return a canonical language-relative page name, without its .md suffix."""
    return MODEL_PAGES[name]


def peer_link(language, page):
    """Compute the peer link from the actual canonical directory depth."""
    other = "cn" if language == "en" else "en"
    return posixpath.relpath(f"{other}/{page}.md", posixpath.dirname(f"{language}/{page}.md"))


def linked_pages(path):
    """Resolve local Markdown destinations while keeping URL/fragment handling explicit."""
    result = set()
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        link = urlsplit(target)
        if not link.scheme and not link.netloc and link.path.endswith(".md"):
            result.add((path.parent / unquote(link.path)).resolve())
    return result


def reachable_pages(start, language_root, max_hops=3):
    """Follow index layers only, so unrelated tutorial links cannot mask missing navigation."""
    boundary = language_root.resolve()
    reached = {start.resolve()}
    frontier = {start.resolve()}
    for _ in range(max_hops):
        following = set()
        for source in frontier:
            for target in linked_pages(source):
                if not target.is_relative_to(boundary):
                    continue
                assert target.is_file(), (source, target)
                if target not in reached and target.name == "README.md":
                    following.add(target)
                reached.add(target)
        frontier = following
    return reached


def markdown_anchors(text):
    """Read real Markdown headings, including linked and Setext headings."""
    from markdown_it import MarkdownIt

    anchors = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', text))
    tokens = MarkdownIt().parse(text)
    seen = {}
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        title = "".join(child.content for child in tokens[index + 1].children
                        if child.type in {"text", "code_inline"})
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if not count else f"{slug}-{count}")
    return anchors
