"""README user navigation works independently of its GitHub/PyPI location."""
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def _readme_methods():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    return text.split('## Implemented Methods\n', 1)[1].split('\n## ', 1)[0]


def _anchors(text):
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
    seen = {}
    for heading in re.findall(r'^#{1,6} (.+)$', text, re.MULTILINE):
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if not count else f'{slug}-{count}')
    return anchors


def test_method_links_have_canonical_hosted_routes_and_existing_sources():
    section = _readme_methods()
    links = re.findall(r'\]\(([^)]+)\)', section)
    assert links
    for target in links:
        link = urlsplit(target)
        assert link.scheme == 'https' and link.netloc == 'statgpu.github.io', target
        assert link.path.startswith('/en/'), target
        assert not link.path.endswith('.md') and not link.query, target
        source = ROOT / 'docs' / (unquote(link.path).lstrip('/') + '.md')
        assert source.is_file(), target
        if link.fragment:
            assert unquote(link.fragment) in _anchors(source.read_text(encoding='utf-8')), target
        for base in ('https://pypi.org/project/statgpu/',
                     'https://github.com/TheHiddenObserver/statgpu/blob/master/README.md'):
            assert urljoin(base, target) == target


def test_all_method_categories_and_english_knockoff_navigation_remain():
    section = _readme_methods()
    rows = [line for line in section.splitlines() if line.startswith('| [**')]
    assert len(rows) == 14
    feature = next(row for row in rows if '[**Feature selection**]' in row)
    for page in ('knockoff', 'knockoff-fixed-x', 'knockoff-model-x'):
        assert f'https://statgpu.github.io/en/models/feature-selection/{page})' in feature
    assert 'https://statgpu.github.io/en/reference/feature-selection-api)' in feature
    assert not re.search(r'^Knockoff(?: 中文)?:', section, re.MULTILINE)
    assert '/cn/' not in section


def test_device_limitations_live_in_guides_not_readme_feature_section():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert '## Known device-routing limitations' not in text
    assert 'Explicit backend semantics' not in text
    assert '[Device and GPU Memory](docs/en/guides/device-and-memory.md)' in text
    for language in ('en', 'cn'):
        assert f'[Source](docs/{language})' in text or f'[源码](docs/{language})' in text
        guide = (ROOT / f'docs/{language}/guides/device-and-memory.md').read_text(encoding='utf-8')
        assert '<a id="current-smoothing-and-spline-exceptions"></a>' in guide
        for name in ('KernelRegression', 'SplineTransformer', 'KernelPCA', 'Nystroem'):
            assert name in guide
        assert ('can still produce Torch CPU features' in guide if language == 'en'
                else '仍可能得到 Torch CPU 特征' in guide)


def test_readme_preserves_generic_chinese_entrances_without_model_links():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert '[中文文档](https://statgpu.github.io/cn/README)' in text
    assert '**Chinese docs**: [在线文档](https://statgpu.github.io/cn/README) · [源码](docs/cn)' in text
    targets = re.findall(r'\]\(([^)]+)\)', text)
    chinese_targets = [target for target in targets
                       if '/cn/' in target or target == 'docs/cn']
    assert set(chinese_targets) == {'https://statgpu.github.io/cn/README', 'docs/cn'}
