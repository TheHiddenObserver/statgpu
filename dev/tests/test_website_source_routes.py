"""Website routes preserve the current bilingual source documentation layout."""

import importlib.util
import json
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
    for section in ('regression', 'testing', 'covariance', 'nonparametric', 'feature-selection', 'survival'):
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


@pytest.mark.parametrize('page', [
    'local-model-benchmark-execution-prompt.md',
    'parser-contracts.md',
    'source-catalog-contract.md',
])
def test_active_benchmark_integration_contracts_use_assembled_outputs(page):
    text = (ROOT / 'docs/benchmark-dashboard' / page).read_text()
    assert 'docs/assets/benchmarks' not in text
    assert 'Regenerate committed frontend data' not in text
    assert 'generated bundle or deployment assets are stale' not in text
    assert 'read-only staleness gate' not in text
    for directory in ('frontend/public/data/', 'frontend/dist/', '.site-dist/dashboard/'):
        assert directory in text
    assert 'website-deployment.md' in text


def test_integration_reference_reuses_complete_production_recipe():
    text = (ROOT / 'docs/benchmark-dashboard/local-model-benchmark-execution-prompt.md').read_text()
    reference = text.split('## 12. Repo integration reference\n', 1)[1]
    assert '#7-full-integration-继续提示词' in reference
    assert '```bash' not in reference
    assert 'cd frontend' not in text
    assert 'npm run build\n' not in text
    assert 'npm run test:e2e --prefix frontend' in text
    assert 'npm run test:e2e:production --prefix frontend' in text
    assert 'npm run --silent site:hash' in text
    assert 'STATGPU_SITE_BASE=/' in text


@pytest.mark.parametrize('locale,function,label', [
    ('en', 'englishModelReference', 'Survival analysis'),
    ('cn', 'chineseModelReference', '生存分析'),
])
def test_survival_sidebar_groups_the_overview_and_existing_cox_tutorial(locale, function, label):
    config = (ROOT / 'docs/.vitepress/config.ts').read_text()
    block = config.split(f'function {function}() {{', 1)[1].split('return ', 1)[1]
    sidebar, _ = json.JSONDecoder().raw_decode(block)
    groups = [item for item in sidebar['items'] if item['text'] == label]
    assert len(groups) == 1
    assert groups[0]['collapsed'] is True
    assert [item['link'] for item in groups[0]['items']] == [
        f'/{locale}/models/survival/README', f'/{locale}/models/coxph',
    ]
    assert all(item.get('link') != f'/{locale}/models/coxph' for item in sidebar['items'])


@pytest.mark.parametrize('locale', ['en', 'cn'])
def test_survival_entry_points_use_a_family_overview_and_keep_the_method_url(locale):
    root = ROOT / 'docs' / locale
    overview = root / 'models/survival/README.md'
    tutorial = root / 'models/coxph.md'
    for source, target in (
        (root / 'README.md', 'models/survival/README.md'),
        (root / 'models/README.md', 'survival/README.md'),
        (root / 'models/regression/README.md', '../survival/README.md'),
        (tutorial, 'survival/README.md'),
    ):
        assert f']({target})' in source.read_text(), source
    text = overview.read_text()
    peer = 'cn' if locale == 'en' else 'en'
    for target in ('../README.md', '../coxph.md', f'../../../{peer}/models/survival/README.md'):
        assert f']({target})' in text
    assert CONTRACTS.validate_links(overview, text) == []
    assert '<!-- example: coxph-cpu-walkthrough -->' in tutorial.read_text()


@pytest.mark.parametrize('locale', ['en', 'cn'])
def test_survival_estimator_choice_lives_in_the_overview_not_the_directory(locale):
    root = ROOT / 'docs' / locale / 'models'
    directory = (root / 'README.md').read_text()
    overview = (root / 'survival/README.md').read_text()
    for estimator in ('CoxPH', 'CoxPHCV', 'PenalizedCoxPHModel'):
        assert f'`{estimator}`' in overview
        assert f'`{estimator}`' not in directory
    for contract in ('Breslow/Efron/Exact', '(start, stop]', 'NumPy/CuPy/Torch',
                     'L1', 'L2', 'ElasticNet', 'SCAD', 'MCP', 'compute_inference=True'):
        assert contract in overview
    if locale == 'en':
        assert 'not interchangeable aliases' in overview
        assert 'estimation-only' in overview
    else:
        assert '不是可以互换的别名' in overview
        assert '只提供参数估计' in overview
