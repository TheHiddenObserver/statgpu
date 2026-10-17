"""Protect the bilingual GPU lesson without treating CPU checks as GPU evidence."""
import ast
import importlib.util
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from doc_examples import parse_examples, run_example

ROOT = Path(__file__).resolve().parents[2]
ANCHORS = (
    'why-matrix-work-can-benefit', 'when-gpu-is-worth-trying',
    'when-cpu-is-the-better-start', 'count-the-whole-journey',
    'try-it-with-an-explicit-device', 'measure-your-own-workload',
)
EXAMPLES = ('gpu-decision-cpu', 'gpu-decision-cupy', 'gpu-decision-torch')


def guide(locale):
    return ROOT / f'docs/{locale}/guides/when-to-use-gpu.md'


@pytest.mark.parametrize('locale', ['en', 'cn'])
def test_lesson_puts_principles_before_code_and_covers_decision_boundaries(locale):
    text = guide(locale).read_text()
    for anchor in ANCHORS:
        assert text.count(f'id="{anchor}"') == 1
    assert text.index('id="when-cpu-is-the-better-start"') < text.index('```python')
    assert text.index('id="count-the-whole-journey"') < text.index('```python')
    for contract in ('device="cpu"', 'device="cuda"', 'device="torch"',
                     'device="auto"', 'return_cpu=False', 'coef_',
                     'current-smoothing-and-spline-exceptions', '160,000,000'):
        assert contract in text
    assert 'statgpu_benchmark_dashboard.md#open-dashboard' in text
    peer = 'cn' if locale == 'en' else 'en'
    assert f'](../../{peer}/guides/when-to-use-gpu.md)' in text
    for phrase in ('exact-head', 'PR #', 'physical GPU passed', 'review-clean'):
        assert phrase not in text
    if locale == 'en':
        assert 'not measured speedups' in text
        assert 'not performance demonstrations' in text
        assert 'asynchronous' in text
        assert 'not automatically a statgpu GPU-versus-CPU speedup' in text
    else:
        assert '不是实测' in text
        assert '异步' in text


def test_bilingual_examples_have_identical_executable_statements_and_dependencies():
    examples = [parse_examples(guide(locale).read_text(), guide(locale)) for locale in ['en', 'cn']]
    for inventory in examples:
        assert tuple(inventory) == EXAMPLES
        assert all(example.explicit_end for example in inventory.values())
        assert inventory['gpu-decision-cpu'].requires == ()
        for name in EXAMPLES[1:]:
            assert inventory[name].requires == ('gpu-decision-cpu',)
    for name in EXAMPLES:
        assert ast.dump(ast.parse(examples[0][name].code)) == ast.dump(ast.parse(examples[1][name].code))


@pytest.mark.parametrize('locale', ['en', 'cn'])
def test_cpu_lesson_runs_and_agrees_with_independent_ridge_solution(locale):
    path = guide(locale)
    state = run_example(path.read_text(), 'gpu-decision-cpu', path)
    X, y = state['X'], state['y']
    assert X.shape == (2000, 20)
    assert state['cpu_model'].device == 'cpu'
    assert isinstance(state['cpu_predictions'], np.ndarray)
    assert state['cpu_predictions'].shape == (3,)
    # statgpu uses mean squared loss: alpha adds n * alpha to centered X'X.
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    coef = np.linalg.solve(Xc.T @ Xc + X.shape[0] * 0.1 * np.eye(X.shape[1]), Xc.T @ yc)
    expected = (X[:3] - X.mean(axis=0)) @ coef + y.mean()
    np.testing.assert_allclose(state['cpu_predictions'], expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('locale', ['en', 'cn'])
def test_gpu_guide_is_reachable_from_home_device_and_quickstart(locale):
    root = ROOT / 'docs' / locale
    for page in ['index.md', 'getting-started/quickstart.md', 'guides/device-and-memory.md']:
        assert 'when-to-use-gpu' in (root / page).read_text()
    assert f"link: '/{locale}/guides/when-to-use-gpu'" in (ROOT / 'docs/.vitepress/config.ts').read_text()
    assert 'en/guides/when-to-use-gpu' in (ROOT / 'docs/index.md').read_text()


@pytest.mark.parametrize('name', ['gpu-work-en', 'gpu-work-cn', 'gpu-journey-en', 'gpu-journey-cn'])
def test_illustrations_are_accessible_and_self_contained(name):
    svg = ROOT / f'docs/public/images/{name}.svg'
    assert svg.stat().st_size < 50_000
    tree = ET.fromstring(svg.read_text())
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    assert tree.attrib['role'] == 'img'
    labels = tree.attrib['aria-labelledby'].split()
    for tag in ('title', 'desc'):
        node = tree.find(f'svg:{tag}', ns)
        assert node is not None and node.text.strip()
        assert node.attrib['id'] in labels
    ids = [node.attrib['id'] for node in tree.iter() if 'id' in node.attrib]
    assert len(ids) == len(set(ids))
    for node in tree.iter():
        assert node.tag.rsplit('}', 1)[-1] not in {'script', 'foreignObject', 'image'}
        for key, value in node.attrib.items():
            assert not key.lower().startswith('on')
            if key.rsplit('}', 1)[-1] == 'href':
                assert value.startswith('#')


@pytest.mark.parametrize('locale', ['en', 'cn'])
def test_guide_images_have_localized_alternative_text(locale):
    text = guide(locale).read_text()
    images = re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', text)
    assert len(images) == 2
    for alt, target in images:
        assert len(alt) >= 30
        assert target.endswith(f'-{locale}.svg')
        assert (ROOT / 'docs/public' / target.lstrip('/')).is_file()


@pytest.mark.parametrize('page,title', [
    ('docs/index.md', 'Statistical computing, accelerated'),
    ('docs/en/index.md', 'Statistical computing, accelerated'),
    ('docs/cn/index.md', '让统计计算更快'),
])
def test_homepage_keeps_acceleration_identity_and_existing_structure(page, title):
    text = (ROOT / page).read_text()
    frontmatter = text.split('---', 2)[1]
    assert f'  text: {title}\n' in frontmatter
    assert 'light: /images/statgpu-compute-hero-light.jpg' in frontmatter
    assert 'dark: /images/statgpu-compute-hero.webp' in frontmatter
    assert frontmatter.count('    - theme:') == 3
    assert frontmatter.count('  - title:') == 3
    assert 'home-content' not in text
    assert 'gpu-overview.svg' not in text
    assert text.count('when-to-use-gpu.md') == 1
    if page == 'docs/cn/index.md':
        assert '多种回归模型支持 R 风格的公式输入' in frontmatter
        assert '潜在优势' in frontmatter
    else:
        assert 'R-style formula input supported by multiple regression models' in frontmatter
        assert 'potential GPU benefits' in frontmatter
        english = (ROOT / 'docs/en/index.md').read_text().split('---', 2)[1]
        assert frontmatter.split('features:\n')[1] == english.split('features:\n')[1]


@pytest.mark.parametrize('page', ['docs/index.md', 'docs/en/index.md', 'docs/cn/index.md'])
def test_homepage_task_order_formula_target_and_public_contact(page):
    text = (ROOT / page).read_text()
    rows = [line for line in text.splitlines() if re.match(r'\| [1-5]\. ', line)]
    assert len(rows) == 5
    assert [int(line[2]) for line in rows] == [1, 2, 3, 4, 5]
    assert 'models/README.md' in rows[0]
    assert 'when-to-use-gpu.md' in rows[1]
    assert '<br>' in rows[1] and 'device-and-memory.md' in rows[1]
    assert 'getting-started/quickstart.md' in rows[2]
    assert 'reference/linear-model-api.md#formula-inputs' in rows[2]
    assert 'cross-validation.md' in rows[3]
    assert 'inference-modes.md' in rows[4] and 'inference-api.md' in rows[4]
    assert '](en/usage.md)' not in text and '](usage.md)' not in text
    assert 'https://github.com/TheHiddenObserver/statgpu/issues' in text
    assert 'https://github.com/TheHiddenObserver/statgpu/blob/master/CONTRIBUTING.md' in text
    assert '[linziqian@fzu.edu.cn](mailto:linziqian@fzu.edu.cn)' in text
    assert re.findall(r'mailto:([^)]*)', text) == ['linziqian@fzu.edu.cn']
    assert not re.search(r'^## (?:Acknowledg|致谢)', text, re.M)
