# Releasing statgpu to PyPI

This document is for maintainers preparing an official `statgpu` release. The repository currently publishes from GitHub Actions when a tag matching `v*` is pushed. The workflow is defined in [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

## Release model

The package version is maintained in two files and must match:

- `pyproject.toml`: `project.version`;
- `statgpu/__init__.py`: `__version__`.

A release tag must use the same version with a leading `v`, for example:

```text
package version: 0.2.2
tag:             v0.2.2
```

PyPI release files are immutable. A broken upload cannot be replaced under the same version; prepare a new patch version instead.

## 1. Prepare a focused release pull request

Start from the latest `master` after the intended feature/fix pull requests are merged.

Update both version declarations:

```toml
# pyproject.toml
version = "0.2.2"
```

```python
# statgpu/__init__.py
__version__ = "0.2.2"
```

Update release-facing documentation:

- `CHANGELOG.md`;
- `docs/en/changelog.md`;
- `docs/cn/changelog.md`;
- README or model documentation when installation, compatibility, or public behavior changed.

Keep release-only changes separate from large implementation work. The release pull request should primarily contain version, packaging, changelog, and release-validation updates.

## 2. Validate the release candidate

At minimum, run the full CPU suite:

```bash
python -m pip install -e ".[dev,validation,formula]"
python -m pytest dev/tests -q --tb=short
```

Run focused physical-GPU acceptance for changes that affect CuPy, Torch, inference, device routing, or performance. Record the exact commit, GPU, CUDA/CuPy/Torch versions, and whether any test was skipped.

Confirm that both version declarations agree:

```bash
python - <<'PY'
import pathlib
import re

pyproject = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
init_file = pathlib.Path("statgpu/__init__.py").read_text(encoding="utf-8")

project_version = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.M).group(1)
package_version = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_file, re.M).group(1)
assert project_version == package_version, (project_version, package_version)
print(project_version)
PY
```

## 3. Build clean artifacts locally

Remove stale packaging output first:

```bash
rm -rf build dist *.egg-info statgpu.egg-info
python -m pip install --upgrade build twine
```

The official PyPI workflow sets `STATGPU_NO_EXT=1`. This produces a universal pure-Python wheel while retaining optional Cython sources in the sdist:

```bash
STATGPU_NO_EXT=1 python -m build
python -m twine check dist/*
ls -lh dist/
```

Expected artifacts:

```text
statgpu-X.Y.Z-py3-none-any.whl
statgpu-X.Y.Z.tar.gz
```

`MANIFEST.in` includes the `.pyx` and `.pxd` files required by users who choose to build the optional CPU extensions from the sdist.

## 4. Test the wheel and sdist in clean environments

Do not validate only from the source checkout. Install each artifact in a fresh environment.

### Wheel

```bash
python -m venv /tmp/statgpu-wheel-test
/tmp/statgpu-wheel-test/bin/python -m pip install --upgrade pip
/tmp/statgpu-wheel-test/bin/python -m pip install dist/statgpu-X.Y.Z-py3-none-any.whl
/tmp/statgpu-wheel-test/bin/python - <<'PY'
import statgpu
print(statgpu.__version__)
from statgpu.linear_model import LinearRegression
print(LinearRegression)
PY
```

### Source distribution

```bash
python -m venv /tmp/statgpu-sdist-test
/tmp/statgpu-sdist-test/bin/python -m pip install --upgrade pip
STATGPU_NO_EXT=1 /tmp/statgpu-sdist-test/bin/python -m pip install dist/statgpu-X.Y.Z.tar.gz
/tmp/statgpu-sdist-test/bin/python - <<'PY'
import statgpu
print(statgpu.__version__)
PY
```

On Windows, replace `/tmp/.../bin/python` with the environment's `Scripts/python.exe`.

For packaging changes, also inspect the artifact contents and confirm that no credentials, benchmark caches, local configuration, or unrelated result bundles are included.

## 5. Optional TestPyPI rehearsal

A TestPyPI upload is recommended when changing packaging metadata, package discovery, build behavior, dependencies, or release automation.

```bash
python -m twine upload --repository testpypi dist/*
```

Install with PyPI available for dependencies:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  statgpu==X.Y.Z
```

TestPyPI and PyPI require separate credentials/tokens.

## 6. Merge the release pull request

Before merging, verify:

- version fields match;
- changelogs describe the release accurately;
- CI is green on the exact release head;
- required physical-GPU tests are recorded;
- wheel and sdist both pass `twine check` and clean-install tests;
- the target version does not already exist on PyPI.

Merge the focused release pull request into `master`.

## 7. Create and push the release tag

Update local `master` and tag the exact merge commit:

```bash
git checkout master
git pull --ff-only origin master
git tag -a vX.Y.Z -m "statgpu X.Y.Z"
git push origin vX.Y.Z
```

Pushing the tag starts the `Publish to PyPI` workflow. The current workflow:

1. checks out the tagged commit;
2. sets up Python 3.11;
3. installs `build` and `twine`;
4. verifies that the tag matches `pyproject.toml`;
5. builds a pure-Python wheel and sdist with `STATGPU_NO_EXT=1`;
6. runs `twine check`;
7. uploads `dist/*` to PyPI using the repository secret `PYPI_TOKEN`.

The PyPI API token should be project-scoped and stored only as a GitHub Actions secret. Never place it in source files, command history committed to the repository, issue comments, or documentation examples.

## 8. Verify the published release

After the workflow succeeds, verify the PyPI release in a new environment:

```bash
python -m venv /tmp/statgpu-pypi-test
/tmp/statgpu-pypi-test/bin/python -m pip install --upgrade pip
/tmp/statgpu-pypi-test/bin/python -m pip install --no-cache-dir statgpu==X.Y.Z
/tmp/statgpu-pypi-test/bin/python - <<'PY'
import statgpu
print(statgpu.__version__)
PY
```

Also verify:

- the PyPI project page renders the README correctly;
- the wheel is `py3-none-any` as intended;
- the sdist is present;
- dependency extras are displayed;
- the homepage and repository links are valid.

Create a GitHub Release from the same tag and use the changelog as the basis for release notes.

## 9. Failure handling

### Version mismatch

If the tag and package version differ, the workflow stops before uploading. Correct the version in a new commit and create a new tag. Do not move an already published tag.

### Upload partially succeeds

PyPI may accept one artifact before another fails. Because filenames and versions are immutable, inspect the project release and normally issue a new patch version rather than attempting to replace uploaded files.

### Bad release already published

- mark the PyPI release as yanked when appropriate;
- fix the problem in a new patch release;
- document the incident and migration path in the changelog;
- do not delete or recreate Git history to reuse the version.

## Recommended automation improvement

The current workflow uses a project-scoped API token through `PYPI_TOKEN`. PyPI Trusted Publishing is preferable for long-term maintenance because it removes the stored upload token and binds publishing to a specific GitHub repository/workflow/environment. Migrating should be handled in a dedicated release-infrastructure pull request and tested before removing the existing token path.
