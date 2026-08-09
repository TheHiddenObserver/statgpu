from pathlib import Path

path = Path('dev/benchmarks/benchmark_panel_stage_c_covariance.py')
text = path.read_text(encoding='utf-8')

needle = '''def _device(backend):
    return {"cupy": "cuda", "torch": "torch"}[backend]


def _fit_case(case, X, y, entity, time_ids, clusters, backend):
'''
replacement = '''def _device(backend):
    return {"cupy": "cuda", "torch": "torch"}[backend]


def _gpu_name(backend):
    if backend == "cupy":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch backend requested but CUDA is unavailable")
        return torch.cuda.get_device_name(0)
    raise ValueError(backend)


def _fit_case(case, X, y, entity, time_ids, clusters, backend):
'''
if needle not in text:
    raise SystemExit('expected device block not found')
text = text.replace(needle, replacement, 1)

old_row = '''                        "n_samples": n,
                        "n_features": k,
                        "repeats": args.repeats,
'''
new_row = '''                        "n_samples": n,
                        "n_features": k,
                        "n_times": int(len(np.unique(time_np))),
                        "repeats": args.repeats,
'''
if old_row not in text:
    raise SystemExit('expected performance row block not found')
text = text.replace(old_row, new_row, 1)

old_env = '''        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _version(name)
                for name in ("statgpu", "numpy", "cupy", "torch")
            },
        },
        "rows": rows,
'''
new_env = '''        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu_by_backend": {backend: _gpu_name(backend) for backend in backends},
            "packages": {
                name: _version(name)
                for name in ("statgpu", "numpy", "cupy", "torch")
            },
        },
        "rows": rows,
'''
if old_env not in text:
    raise SystemExit('expected performance environment block not found')
text = text.replace(old_env, new_env, 1)

path.write_text(text, encoding='utf-8')
