#!/usr/bin/env python3
"""Read-only source/environment preflight; CUDA work requires --probe-gpu."""
import argparse
import datetime
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PIN = '91df0d9ebc35a49bde917383ba0680806016f3ae'
TREE = 'bdf93d2c86fca661ed3a444b6754325635566843'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

def verify_tracked_source(repo):
    entries = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-rz', '--full-tree', 'HEAD'])
    manifest = {}
    for entry in entries.split(b'\0'):
        if not entry: continue
        metadata, rawpath = entry.split(b'\t', 1)
        mode, kind, expected = metadata.decode().split()
        relative = rawpath.decode('utf-8')
        path = repo / relative
        if kind != 'blob' or mode not in ('100644', '100755', '120000'):
            raise RuntimeError('Unsupported tracked object: ' + relative)
        if mode == '120000':
            if not path.is_symlink(): raise RuntimeError('Tracked symlink changed: ' + relative)
            data = os.readlink(path).encode('utf-8')
        else:
            if path.is_symlink() or not path.is_file(): raise RuntimeError('Tracked file changed: ' + relative)
            if not path.resolve().is_relative_to(repo.resolve()): raise RuntimeError('Tracked path escapes checkout')
            data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected: raise RuntimeError('Tracked bytes differ from HEAD: ' + relative)
        manifest[relative] = hashlib.sha256(data).hexdigest()
    return manifest

def verify(repo, gpu=False, threads=1):
    report = {'schema': 'statgpu-runtime-preflight-v1', 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.version, 'errors': [], 'gpu_probe_requested': gpu,
              'exclusivity': 'UNVERIFIED_EXCLUSIVITY', 'performance_eligible': False}
    report['source'] = {k: git(repo, *args) for k, args in {
        'head': ('rev-parse', 'HEAD'), 'tree': ('rev-parse', 'HEAD^{tree}'),
        'status': ('status', '--porcelain'), 'shallow': ('rev-parse', '--is-shallow-repository')}.items()}
    if report['source'] != {'head': PIN, 'tree': TREE, 'status': '', 'shallow': 'false'}:
        report['errors'].append('Requires the exact clean non-shallow merged source pin')
    try:
        report['tracked_sha256'] = verify_tracked_source(repo)
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    report['versions'] = {}
    for name in ('numpy','scipy','scikit-learn','statsmodels','pandas','patsy','pytest','threadpoolctl','jsonschema','torch','cupy-cuda11x','cupy-cuda12x','cupy'):
        try: report['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['versions'][name] = None
    report['thread_environment'] = {k: os.environ.get(k) for k in THREAD_KEYS}
    if any(v != str(threads) for v in report['thread_environment'].values()):
        report['errors'].append('Set all six thread environment variables before Python starts')
    if not sys.flags.utf8_mode:
        report['errors'].append('Start Python with PYTHONUTF8=1')
    if report['errors'] or not gpu:
        report['status'] = 'failed' if report['errors'] else 'source_and_environment_only'
        return report
    sys.path.insert(0, str(repo))
    try:
        import statgpu
        report['statgpu_import'] = str(Path(statgpu.__file__).resolve())
        if not Path(statgpu.__file__).resolve().is_relative_to(repo.resolve()):
            raise RuntimeError('statgpu import is outside checkout')
        import jsonschema
        jsonschema.Draft202012Validator({'type':'string','format':'date-time'}, format_checker=jsonschema.FormatChecker()).validate('2026-10-25T00:00:00Z')
        import cupy as cp
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
        if not torch.cuda.is_available(): raise RuntimeError('Torch CUDA unavailable')
        with cp.cuda.Device(0):
            x = cp.arange(16, dtype=cp.float64).reshape(4,4)
            y = x @ x.T
            cp.cuda.Device(0).synchronize()
            if float(y[0,0].item()) != 14.0: raise RuntimeError('CuPy float64 probe mismatch')
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props['name']; name = name.decode() if isinstance(name,bytes) else str(name)
            report['cupy'] = {'device': 0, 'name': name, 'dtype': str(y.dtype), 'float64_matmul_synced': True}
        x = torch.arange(16, dtype=torch.float64, device='cuda:0').reshape(4,4)
        y = x @ x.T
        torch.cuda.synchronize(0)
        if y[0,0].item() != 14.0: raise RuntimeError('Torch float64 probe mismatch')
        report['torch'] = {'device': str(y.device), 'name': torch.cuda.get_device_name(0), 'dtype': str(y.dtype),
                           'float64_matmul_synced': True, 'intraop': torch.get_num_threads(), 'interop': torch.get_num_interop_threads()}
        import threadpoolctl
        report['pools'] = threadpoolctl.threadpool_info()
        if not report['pools'] or any(p['num_threads'] > threads for p in report['pools']):
            raise RuntimeError('Numerical thread pools unavailable or above requested cap')
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    report['status'] = 'failed' if report['errors'] else 'device_preflight_pass'
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--probe-gpu',action='store_true')
    p.add_argument('--threads',type=int,choices=(1,2,4,8),default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Refusing existing output')
    r=verify(a.repo,a.probe_gpu,a.threads)
    with a.output.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(r['status']);return 2 if r['errors'] else 0
if __name__=='__main__':sys.exit(main())
