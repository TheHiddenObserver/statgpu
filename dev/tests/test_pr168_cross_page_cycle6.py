"""Sixth-cycle navigation and global-device documentation checks.

No GPU execution is claimed: the routing probe inspects enum resolution only.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from statgpu import LinearRegression
from statgpu._base import BaseEstimator
from statgpu._config import Device, _device_manager

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_documentation_index_exposes_all_completed_api_references(language):
    index = ROOT / 'docs' / language / 'README.md'
    text = index.read_text(encoding='utf-8')
    for relative in (
        'reference/estimator-api.md',
        'reference/linear-model-api.md',
        'reference/feature-selection-api.md',
        'reference/survival-smoothing-api.md',
        'unsupervised/api-reference.md',
        'guides/distribution-api.md',
        'guides/inference-api.md',
    ):
        assert f']({relative})' in text
        assert (index.parent / relative).is_file()
    # The kernel-methods page covers feature maps/ridge, rather than KDE.
    kernel_line = next(line for line in text.splitlines()
                       if '](models/kernel-methods.md)' in line)
    assert 'KernelPCA' in kernel_line and 'Nystroem' in kernel_line
    assert 'KDE' not in kernel_line


@pytest.mark.parametrize('global_device', [Device.CPU, Device.CUDA, Device.TORCH])
@pytest.mark.parametrize('local_device', ['auto', 'cpu', 'cuda', 'torch'])
def test_shared_auto_device_inherits_global_without_running_on_gpu(
    monkeypatch, global_device, local_device,
):
    monkeypatch.setattr(_device_manager, '_current_device', global_device)
    estimator = LinearRegression(device=local_device)
    expected = global_device if local_device == 'auto' else Device(local_device)
    assert estimator._get_compute_device() is expected
    assert estimator.device == local_device


def test_installed_base_device_help_lists_torch_and_global_auto_policy():
    text = inspect.getdoc(BaseEstimator.__init__)
    for choice in ('cpu', 'cuda', 'torch', 'auto'):
        assert f"'{choice}'" in text
    assert 'global' in text and 'overrides' in text


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_torch_guide_links_global_auto_exception(language):
    text = (ROOT / 'docs' / language / 'guides/pytorch-backend.md').read_text()
    assert 'sg.set_device("auto")' in text
    assert 'device="cpu"' in text
    fragment = ('global-settings-and-estimator-settings' if language == 'en'
                else '全局设置与估计器设置')
    assert f'](device-and-memory.md#{fragment})' in text


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_usage_portal_links_complete_api_index(language):
    text = (ROOT / 'docs' / language / 'usage.md').read_text()
    fragment = 'complete-api-references' if language == 'en' else '完整-api-参考'
    assert f'](README.md#{fragment})' in text
