"""Control-flow tests for the installed universal-wheel NumPy smoke matrix.

These do not replace the clean artifact installations performed by CI.
"""
from pathlib import Path
import subprocess
import textwrap
import venv

import pytest


WORKFLOW = Path('.github/workflows/release-package.yml')


def _smoke_script():
    source = WORKFLOW.read_text(encoding='utf-8')
    marker = '      - name: Smoke-install universal wheel\n'
    block = source.split(marker, 1)[1].split('        run: |\n', 1)[1]
    return textwrap.dedent(block)


def test_explicit_numpy_rows_reuse_the_validated_universal_wheel():
    source = WORKFLOW.read_text(encoding='utf-8')
    job = source.split('  smoke-install-wheel:', 1)[1]
    assert 'needs: validate-distributions' in job
    assert 'actions/download-artifact@v4' in job
    assert 'name: statgpu-release-distributions' in job
    assert 'name: ${{ matrix.job-name }}' in job
    for os_name in ('ubuntu-latest', 'windows-latest', 'macos-latest'):
        assert (f'- os: {os_name}\n            numpy: default\n'
                f'            python: "3.11"\n            job-name: wheel smoke ({os_name})') in job
    for version in ('1.26.4', '2.0.2'):
        assert f'- os: ubuntu-latest\n            numpy: "{version}"\n            python: "3.9"' in job
    assert 'python-version: ${{ matrix.python }}' in job
    assert 'STATGPU_NO_EXT: "1"' in source


@pytest.mark.parametrize('numpy_version', ['default', '1.26.4', '2.0.2'])
def test_install_pins_numpy_with_exact_artifact_and_checks_dependencies(monkeypatch, tmp_path, numpy_version):
    root = tmp_path / 'checkout'
    (root / 'statgpu').mkdir(parents=True)
    (root / 'statgpu' / '__init__.py').write_text('__version__ = "0.3.0"\n')
    wheel = root / 'dist' / 'statgpu-0.3.0-py3-none-any.whl'
    wheel.parent.mkdir()
    wheel.touch()
    monkeypatch.setenv('GITHUB_WORKSPACE', str(root))
    monkeypatch.setattr(venv.EnvBuilder, 'create', lambda self, path: None)
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs)))
    namespace = {}
    exec(compile(_smoke_script().replace('${{ matrix.numpy }}', numpy_version), str(WORKFLOW), 'exec'), namespace)
    assert len(calls) == 4
    install = calls[1][0]
    assert install[1:5] == ['-m', 'pip', 'install', str(wheel)]
    assert install[5:] == ([] if numpy_version == 'default' else [f'numpy=={numpy_version}'])
    assert calls[2][0][1:] == ['-m', 'pip', 'check']
    assert all(kwargs['check'] for _, kwargs in calls)
    args, kwargs = calls[3]
    assert args[1] == '-c'
    assert kwargs['cwd'] != root
    assert kwargs['env']['EXPECTED_NUMPY_VERSION'] == numpy_version
    assert kwargs['env']['EXPECTED_STATGPU_VERSION'] == '0.3.0'
    smoke = args[2]
    compile(smoke, '<installed-wheel-smoke>', 'exec')
    assert 'np.__version__ == expected_numpy' in smoke
    assert 'package_dir.is_relative_to' in smoke
    assert 'Ridge(alpha=0.1, device="cpu")' in smoke
    assert 'np.testing.assert_allclose(prediction, y' in smoke


def test_missing_downloaded_wheel_fails_before_install(monkeypatch, tmp_path):
    (tmp_path / 'statgpu').mkdir()
    (tmp_path / 'statgpu' / '__init__.py').write_text('__version__ = "0.3.0"\n')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('must not install'))
    with pytest.raises(SystemExit, match='missing downloaded wheel'):
        exec(compile(_smoke_script(), str(WORKFLOW), 'exec'), {})


def _installed_smoke():
    import ast
    tree = ast.parse(_smoke_script())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'smoke_code' for t in node.targets))


@pytest.mark.parametrize('failure', ['wrong_numpy', 'outside_venv', 'compiled_binary'])
def test_installed_smoke_rejects_wrong_runtime_or_artifact(monkeypatch, tmp_path, failure):
    import sys
    from types import SimpleNamespace
    import numpy as np

    prefix = tmp_path / 'venv'
    package = (tmp_path / 'checkout' if failure == 'outside_venv' else prefix) / 'statgpu'
    package.mkdir(parents=True)
    if failure == 'compiled_binary':
        (package / 'unwanted.so').touch()
    monkeypatch.setattr(sys, 'prefix', str(prefix))
    monkeypatch.setitem(sys.modules, 'statgpu', SimpleNamespace(__version__='0.3.0', __file__=str(package / '__init__.py')))
    monkeypatch.setitem(sys.modules, 'statgpu.linear_model', SimpleNamespace(LinearRegression=object, Ridge=object))
    monkeypatch.setitem(sys.modules, 'statgpu.survival', SimpleNamespace(CoxPH=object, CoxPHCV=object))
    monkeypatch.setenv('EXPECTED_STATGPU_VERSION', '0.3.0')
    monkeypatch.setenv('EXPECTED_NUMPY_VERSION', 'unexpected' if failure == 'wrong_numpy' else np.__version__)
    with pytest.raises(AssertionError):
        exec(compile(_installed_smoke(), '<installed-wheel-smoke>', 'exec'), {})
