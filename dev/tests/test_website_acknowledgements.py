"""Acknowledgements preserve supplied names, affiliations, ordering, and routes."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('locale,headings,groups', [
    ('en', ['Contributors', 'LLM API providers', 'GPU providers'], [
        ['Zhan Gao (Peking University)', 'Zeqi Wu (Renmin University of China)', 'Baichen Yu (Peking University)'],
        ['Qiuxin Gao (Johns Hopkins University)', 'Huangang Lu (Peking University)', 'Xiaomi MiMo Orbit 百万亿 Token 创造者激励计划'],
        ['狗熊会 (Clubear)'],
    ]),
    ('cn', ['贡献者', 'LLM API 提供者', 'GPU 提供者'], [
        ['Zhan Gao（北京大学）', 'Zeqi Wu（中国人民大学）', 'Baichen Yu（北京大学）'],
        ['Qiuxin Gao（约翰斯·霍普金斯大学）', 'Huangang Lu（北京大学）', 'Xiaomi MiMo Orbit 百万亿 Token 创造者激励计划'],
        ['狗熊会（Clubear）'],
    ]),
])
def test_acknowledgements_have_exact_supplied_groups_and_surname_order(locale, headings, groups):
    text = (ROOT / f'docs/{locale}/acknowledgements.md').read_text()
    sections = re.split(r'^## (.+)\n', text, flags=re.M)
    assert sections[1::2] == headings
    assert [re.findall(r'^- (.+)$', body, re.M) for body in sections[2::2]] == groups
    counterpart = 'cn' if locale == 'en' else 'en'
    assert f'../{counterpart}/acknowledgements.md' in text
    assert 'surname' in text if locale == 'en' else '个人按姓氏' in text
    assert 'not by the size of their contribution' in text if locale == 'en' else '不代表贡献大小' in text
    assert 'http' not in text and 'mailto:' not in text


@pytest.mark.parametrize('page,target,label', [
    ('docs/index.md', 'en/acknowledgements.md', 'Acknowledgements'),
    ('docs/en/index.md', 'acknowledgements.md', 'Acknowledgements'),
    ('docs/cn/index.md', 'acknowledgements.md', '致谢'),
])
def test_homepage_has_one_compact_acknowledgements_link_and_keeps_contact(page, target, label):
    text = (ROOT / page).read_text()
    assert text.count(f'[{label}]({target})') == 1
    assert 'mailto:linziqian@fzu.edu.cn' in text
    assert 'https://github.com/TheHiddenObserver/statgpu/issues' in text
    assert 'https://github.com/TheHiddenObserver/statgpu/blob/master/CONTRIBUTING.md' in text
    assert not re.search(r'^## (?:Acknowledgements|致谢)$', text, re.M)


def test_project_sidebars_link_both_acknowledgements_pages():
    config = (ROOT / 'docs/.vitepress/config.ts').read_text()
    for locale in ('en', 'cn'):
        assert config.count(f"link: '/{locale}/acknowledgements'") == 1
