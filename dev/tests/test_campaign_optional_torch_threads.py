"""Real isolated child processes with fake modules, never numerical/GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'benchmarks/run_qualification_campaign.py'
SPEC = importlib.util.spec_from_file_location('thread_campaign', PATH)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)

TORCH = '''import os
from pathlib import Path
def event(value):
    with Path(os.environ['TRACE']).open('a') as stream: stream.write(value + '\\n')
event('torch_import')
intra = inter = 12
def set_num_threads(n):
    global intra
    event('set_intra')
    intra = n
def set_num_interop_threads(n):
    global inter
    event('set_inter')
    inter = n
def get_num_threads(): return intra
def get_num_interop_threads(): return inter
'''


def child(tmp_path, backend='numpy', torch_source=TORCH, workload='', pool_source=None):
    # -S excludes site-packages so "absent" never depends on host Torch installs.
    if torch_source is not None:
        (tmp_path / 'torch.py').write_text(torch_source)
    (tmp_path / 'threadpoolctl.py').write_text(pool_source or
        "def threadpool_info(): return [{'num_threads': 1, 'internal_api': 'fake'}]\n")
    (tmp_path / 'workload.py').write_text('''import os
from pathlib import Path
with Path(os.environ['TRACE']).open('a') as stream: stream.write('benchmark_import\\n')
''' + workload + '''
def main(argv):
    with Path(os.environ['TRACE']).open('a') as stream: stream.write('benchmark_main\\n')
    if 'mutate' in globals(): mutate()
    return 0
''')
    bootstrap = tmp_path / 'bootstrap.py'
    bootstrap.write_text(CAMPAIGN.CHILD_BOOTSTRAP)
    env = {**os.environ, 'TRACE': str(tmp_path / 'trace'), 'PYTHONDONTWRITEBYTECODE': '1'}
    env.update({key: '1' for key in CAMPAIGN.THREAD_ENV})
    result = subprocess.run([sys.executable, '-S', str(bootstrap), str(tmp_path),
        'workload', backend, '1', str(tmp_path / 'runtime')],
        capture_output=True, env=env, timeout=15)
    report = json.loads((tmp_path / 'runtime.after.json').read_text())
    trace = (tmp_path / 'trace').read_text().splitlines() if (tmp_path / 'trace').exists() else []
    return result, report, trace


@pytest.mark.parametrize('backend', ['numpy', 'cupy', 'torch'])
def test_controls_optional_torch_before_import_for_every_backend(tmp_path, backend):
    result, report, trace = child(tmp_path, backend, workload='''import torch
assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
''')
    assert result.returncode == 0, result.stderr.decode()
    assert trace == ['torch_import', 'set_intra', 'set_inter', 'benchmark_import', 'benchmark_main']
    assert report['torch_control']['controlled_before_benchmark_import'] is True
    assert report['torch_control']['observed_after_control'] == {'intraop': 1, 'interop': 1}
    for phase in ('before', 'after'):
        assert report[phase]['cap_verified'] is True
        assert report[phase]['torch_threads'] == {'intraop': 1, 'interop': 1}


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_absent_torch_is_distinct_and_allowed_without_torch_backend(tmp_path, backend):
    result, report, trace = child(tmp_path, backend, torch_source=None)
    assert result.returncode == 0, result.stderr.decode()
    assert report['torch_control']['status'] == 'not_installed'
    assert report['before']['torch_threads'] is None
    assert trace == ['benchmark_import', 'benchmark_main']


def test_absent_requested_torch_fails_before_benchmark_import(tmp_path):
    result, report, trace = child(tmp_path, 'torch', torch_source=None)
    assert result.returncode == 86
    assert report['torch_control']['status'] == 'not_installed'
    assert report['resource_guard'] is True
    assert trace == []


@pytest.mark.parametrize('source', [
    "raise RuntimeError('broken install')\n",
    "raise ModuleNotFoundError('missing Torch dependency', name='torch_dependency')\n",
    TORCH.replace("    inter = n", "    raise RuntimeError('interop already initialized')"),
    TORCH.replace("    inter = n", "    inter = 12"),
    TORCH.replace("    inter = n", "    inter = True"),
    TORCH.replace("    intra = n", "    intra = 0"),
])
def test_broken_import_reinitialization_or_unhonored_cap_fails_closed(tmp_path, source):
    result, report, trace = child(tmp_path, torch_source=source)
    assert result.returncode == 86
    assert report['torch_control']['status'] == 'initialization_failed'
    assert report['resource_guard'] is True
    assert 'benchmark_import' not in trace


@pytest.mark.parametrize('backend', ['numpy', 'cupy', 'torch'])
@pytest.mark.parametrize('assignment', ['torch.inter = 12', 'torch.intra = 2', 'torch.inter = True'])
def test_late_torch_bounds_violation_is_observed_and_rejected(tmp_path, backend, assignment):
    result, report, trace = child(tmp_path, backend, workload='''import torch
def mutate():
    ''' + assignment + '\n')
    assert result.returncode == 86
    assert report['before']['cap_verified'] is True
    assert report['after']['cap_verified'] is False
    assert report['resource_guard'] is True
    assert trace[-1] == 'benchmark_main'


@pytest.mark.parametrize('statement', [
    "sys.modules.pop('torch')", "sys.modules['torch'] = types.SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1)",
])
def test_torch_disappearance_or_replacement_cannot_hide_reinitialization(tmp_path, statement):
    result, report, trace = child(tmp_path, workload='''import sys, types
def mutate():
    ''' + statement + '\n')
    assert result.returncode == 86
    assert report['before']['cap_verified'] is True
    assert report['after']['cap_verified'] is False


def test_late_torch_after_absence_is_not_silently_uncapped(tmp_path):
    result, report, trace = child(tmp_path, torch_source=None, workload='''import sys, types
def mutate():
    sys.modules['torch'] = types.SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1)
''')
    assert result.returncode == 86
    assert report['torch_control']['status'] == 'not_installed'
    assert report['after']['cap_verified'] is False


@pytest.mark.parametrize('value', ['[]', "[{'num_threads': 12}]", "[{'num_threads': True}]", "[{}]"])
def test_native_pool_validation_still_fails_closed(tmp_path, value):
    result, report, trace = child(tmp_path,
        pool_source='def threadpool_info(): return ' + value + '\n')
    assert result.returncode == 86
    assert report['before']['cap_verified'] is False
    assert 'benchmark_main' not in trace
