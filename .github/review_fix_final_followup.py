from pathlib import Path

# Preserve explicitly replaced generators while snapshotting only stale iterators.
p = Path('statgpu/_base.py')
text = p.read_text(encoding='utf-8')
old = '''        for key, value in tuple(direct.items()):
            if isinstance(value, Iterator):
                snapshot = getattr(self, "_cox_cv_split_snapshot", None)
                if snapshot is None:
                    snapshot = list(value)
                direct[key] = copy.deepcopy(snapshot)
'''
new = '''        explicitly_updated = {
            key.partition("__")[0]
            for key in params
            if "__" not in key
        }
        for key, value in tuple(direct.items()):
            if isinstance(value, Iterator) and key not in explicitly_updated:
                snapshot = getattr(self, "_cox_cv_split_snapshot", None)
                if snapshot is None:
                    snapshot = list(value)
                direct[key] = copy.deepcopy(snapshot)
'''
if text.count(old) != 1:
    raise SystemExit(f'iterator anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# Restrict name-based estimator typing to actual model modules.
old = '''        if "_estimator_type" not in cls.__dict__:
            name = cls.__name__.lower()
            if "classifier" in name or "logistic" in name:
                cls._estimator_type = "classifier"
            elif any(
                token in name
                for token in (
                    "regression",
                    "regressor",
                    "ridge",
                    "lasso",
                    "elasticnet",
                    "quantile",
                    "cox",
                    "panel",
                    "ols",
                    "effects",
                    "fama",
                    "kernelridge",
                    "gam",
                )
            ):
                cls._estimator_type = "regressor"
'''
new = '''        if "_estimator_type" not in cls.__dict__:
            name = cls.__name__.lower()
            module = cls.__module__
            classifier_module = module.startswith("statgpu.linear_model")
            regression_module = module.startswith(
                (
                    "statgpu.linear_model",
                    "statgpu.panel",
                    "statgpu.survival",
                    "statgpu.semiparametric",
                )
            )
            if ("classifier" in name or "logistic" in name) and classifier_module:
                cls._estimator_type = "classifier"
            elif (
                "regressor" in name
                or "kernelridge" in name
                or (
                    regression_module
                    and any(
                        token in name
                        for token in (
                            "regression",
                            "ridge",
                            "lasso",
                            "elasticnet",
                            "quantile",
                            "cox",
                            "panel",
                            "ols",
                            "effects",
                            "fama",
                            "gam",
                        )
                    )
                )
            ):
                cls._estimator_type = "regressor"
'''
if text.count(old) != 1:
    raise SystemExit(f'estimator type anchor count={text.count(old)}')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Bound compile diagnostic retention.
p = Path('statgpu/backends/_torch_compile.py')
text = p.read_text(encoding='utf-8')
text = text.replace('import functools\nimport os\n', 'import functools\nimport os\nfrom collections import deque\n', 1)
text = text.replace('_COMPILE_DIAGNOSTICS = []', '_COMPILE_DIAGNOSTICS = deque(maxlen=256)', 1)
p.write_text(text, encoding='utf-8')

# Strengthen tests: generator identity, covariance non-regressor, actual Dynamo graph.
p = Path('dev/tests/test_maintenance_024_025.py')
text = p.read_text(encoding='utf-8')
old = '''    assert is_classifier(LogisticRegression())
    assert is_regressor(Ridge(compute_inference=False))
'''
new = '''    from statgpu.covariance import GraphicalLasso

    assert is_classifier(LogisticRegression())
    assert is_regressor(Ridge(compute_inference=False))
    assert not is_regressor(GraphicalLasso())
    assert not is_classifier(GraphicalLasso())
'''
if text.count(old) != 1:
    raise SystemExit(f'tag test anchor count={text.count(old)}')
text = text.replace(old, new, 1)

old = '''    compiled = compile_torch(add_one, workload="iterative")
    x = torch.arange(16, device="cuda", dtype=torch.float64)
    result = compiled(x)
    torch.cuda.synchronize()
    assert compiled.__statgpu_compile_status__ == "compiled"
    assert torch.allclose(result, x + 1)
    assert get_torch_compile_diagnostics(clear=True)[-1]["status"] == "compiled"
'''
new = '''    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    before_graphs = int(counters["stats"].get("unique_graphs", 0))
    compiled = compile_torch(add_one, workload="iterative")
    x = torch.arange(16, device="cuda", dtype=torch.float64)
    result = compiled(x)
    torch.cuda.synchronize()
    after_graphs = int(counters["stats"].get("unique_graphs", 0))
    assert compiled.__statgpu_compile_status__ == "compiled"
    assert after_graphs > before_graphs
    assert torch.allclose(result, x + 1)
    assert get_torch_compile_diagnostics(clear=True)[-1]["status"] == "compiled"
'''
if text.count(old) != 1:
    raise SystemExit(f'physical compile anchor count={text.count(old)}')
text = text.replace(old, new, 1)

old = '''    get_torch_compile_diagnostics(clear=True)

    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
'''
new = '''    get_torch_compile_diagnostics(clear=True)
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    before_graphs = int(counters["stats"].get("unique_graphs", 0))

    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
'''
if text.count(old) != 1:
    raise SystemExit(f'penalty graph before anchor count={text.count(old)}')
text = text.replace(old, new, 1)
old = '''    events = get_torch_compile_diagnostics(clear=True)
    assert len([event for event in events if event["status"] == "compiled"]) >= len(penalties)
    assert [event for event in events if "fallback" in event["status"]] == []
'''
new = '''    after_graphs = int(counters["stats"].get("unique_graphs", 0))
    events = get_torch_compile_diagnostics(clear=True)
    assert after_graphs > before_graphs
    assert len([event for event in events if event["status"] == "compiled"]) >= len(penalties)
    assert [event for event in events if "fallback" in event["status"]] == []
'''
if text.count(old) != 1:
    raise SystemExit(f'penalty graph after anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# Make the original Lasso acceptance explicitly inspect compile diagnostics.
old = '''    from statgpu.backends import _to_numpy
    from statgpu.linear_model import Lasso

    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
'''
new = '''    from statgpu.backends import _to_numpy
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    from statgpu.linear_model import Lasso

    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
    get_torch_compile_diagnostics(clear=True)
'''
if text.count(old) != 1:
    raise SystemExit(f'lasso diagnostics import anchor count={text.count(old)}')
text = text.replace(old, new, 1)
old = '''    assert np.isfinite(second).all()
    np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-8)
'''
new = '''    assert np.isfinite(second).all()
    np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-8)
    events = get_torch_compile_diagnostics(clear=True)
    assert any(event["status"] == "compiled" for event in events)
    assert not any("fallback" in event["status"] for event in events)
'''
if text.count(old) != 1:
    raise SystemExit(f'lasso diagnostics assertion anchor count={text.count(old)}')
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')

import compileall
for path in ('statgpu/_base.py', 'statgpu/backends/_torch_compile.py', 'dev/tests/test_maintenance_024_025.py'):
    if not compileall.compile_file(path, quiet=1):
        raise SystemExit(f'compile failed: {path}')
