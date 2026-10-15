"""Website routes preserve the current bilingual source documentation layout."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "website_doc_contracts", ROOT / "dev/validation/check_docs_contracts.py"
)
CONTRACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACTS)


def test_sidebar_has_matching_valid_canonical_locale_routes():
    config = (ROOT / "docs/.vitepress/config.ts").read_text()
    routes = set(re.findall(r'''["']link["']?:\s*["'](/(?:en|cn)/[^"']+)["']''', config))
    en = {route[3:] for route in routes if route.startswith('/en/')}
    cn = {route[3:] for route in routes if route.startswith('/cn/')}
    assert en == cn
    for route in routes:
        assert CONTRACTS.validate_links(ROOT / 'docs/index.md', f'[page]({route})') == []
    for section in ('regression', 'testing', 'covariance', 'nonparametric', 'feature-selection'):
        assert f'/models/{section}/README' in en
    assert '/reference/linear-model-api' in en
    assert '/models/lasso' not in en
    assert '/models/index' not in en
    assert '/models/' not in en


@pytest.mark.parametrize('locale', ['en', 'cn'])
def test_model_roots_have_no_resurrected_flat_pages(locale):
    assert {p.name for p in (ROOT / 'docs' / locale / 'models').glob('*.md')} == {'README.md', 'coxph.md'}


@pytest.mark.parametrize('route', [
    '/en/models/regression/lasso', '/cn/reference/linear-model-api',
    '/en/models/README', '/cn/panel/README', '/en/', '/cn/',
    '/dashboard/', '/dashboard/data/benchmark_data.json',
    '/dashboard/data/parse_report.json', '/dashboard/data/source_inventory.json',
])
def test_source_validator_recognizes_known_website_routes(route):
    assert CONTRACTS.validate_links(ROOT / 'docs/index.md', f'[page]({route})') == []


@pytest.mark.parametrize('route', [
    '/en/models/lasso', '/cn/models/ridge', '/en/models/not-a-page',
    '/dashboard/data/missing.json', '/images/missing.png', '/unrecognized/route',
])
def test_source_validator_does_not_hide_missing_website_routes(route):
    assert CONTRACTS.validate_links(ROOT / 'docs/index.md', f'[page]({route})')


def test_documentation_has_no_links_to_removed_dashboard_bundle():
    for page in (ROOT / 'docs').rglob('*.md'):
        if '.vitepress' in page.parts:
            continue
        for target in CONTRACTS.iter_link_targets(page.read_text()):
            assert 'assets/benchmarks' not in target, (page, target)
