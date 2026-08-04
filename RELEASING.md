# Releasing statgpu to PyPI

This document is for maintainers preparing an official `statgpu` release.
The repository publishes from GitHub Actions when a tag matching `v*` is pushed.
The upload workflow is defined in [`.github/workflows/publish.yml`](.github/workflows/publish.yml),
and pull-request package validation is defined in
[`.github/workflows/release-package.yml`](.github/workflows/release-package.yml).

## Release model

The package version is maintained in two files and must match:

- `pyproject.toml`: `project.version`;
- `statgpu/__init__.py`: `__version__`.

A release tag uses the same version with a leading `v`:

```text
package version: 0.2.3
tag:             v0.2.3
```

PyPI release files are immutable. A broken upload cannot be replaced under the
same version; prepare a new patch version instead.

## 1. Prepare a focused release pull request

Start from the latest `master` after the intended feature and fix pull requests
are merged.

Update both version declarations and the release-facing documentation:

- `pyproject.toml`;
- `statgpu/__init__.py`;
- `CHANGELOG.md`;
- `docs/en/changelog.md`;
- `docs/cn/changelog.md`;
- README or model documentation when installation, compatibility, or public
  behavior changed.

Keep release-only changes separate from implementation work. A release pull
request should primarily contain version, packaging, changelog, and
release-validation changes.

## 2. Validate the release pull request

The normal `Tests` workflow must pass, including the complete CPU suite, static
contracts, documentation contracts, and the Python 3.9–3.12 regression matrix.
For changes affecting CuPy, Torch, inference, device routing, or performance,
record physical-GPU acceptance on the exact release source commit.

The `Release package validation` workflow automatically:

1. checks that `pyproject.toml` and `statgpu/__init__.py` declare the same version;
2. builds a pure-Python wheel and source distribution with `STATGPU_NO_EXT=1`;
3. runs `twine check`;
4. requires exactly `statgpu-X.Y.Z-py3-none-any.whl` and
   `statgpu-X.Y.Z.tar.gz`;
5. rejects unsafe paths, credential-like files, cache directories, and compiled
   binaries in the universal wheel;
6. confirms that the sdist contains every `.pyx` or `.pxd` source that currently
   exists in the repository;
7. installs the sdist in a clean Ubuntu virtual environment and checks its
   version;
8. uploads the validated wheel and sdist as a short-lived workflow artifact;
9. downloads that exact wheel artifact on Ubuntu, Windows, and macOS, installs it
   in a fresh virtual environment, imports the public linear-model and Cox APIs,
   and runs a CPU `LinearRegression` fit/predict smoke test.

The cross-platform matrix validates portability of the published
`py3-none-any` CPU wheel. It does not claim Apple MPS support or replace the
separate physical-NVIDIA-GPU acceptance required for CUDA behavior.

For a local rehearsal, run:

```bash
python -m pip install -e ".[dev,validation,formula]"
python -m pytest dev/tests -q --tb=short

rm -rf build dist *.egg-info statgpu.egg-info
python -m pip install --upgrade build twine
STATGPU_NO_EXT=1 python -m build
python -m twine check dist/*
ls -lh dist/
```

Do not validate only from the source checkout. Install the wheel and sdist in
fresh environments, or rely on the successful release-package workflow for the
exact PR head.

## 3. Optional TestPyPI rehearsal

A TestPyPI rehearsal is recommended when changing packaging metadata, package
discovery, build behavior, dependencies, or release automation:

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  statgpu==X.Y.Z
```

TestPyPI and PyPI use separate credentials.

## 4. Merge the release pull request

Before merging, verify:

- both version declarations match the intended release;
- all release notes are accurate and synchronized in English and Chinese;
- required GitHub Actions jobs are green on the exact release head;
- required physical-GPU evidence is recorded for the exact source commit;
- wheel and sdist validation, the Ubuntu sdist clean-install check, and the
  Ubuntu/Windows/macOS wheel smoke matrix pass;
- the target version does not already exist on PyPI;
- the `PYPI_TOKEN` repository secret remains valid and project-scoped.

Merge the focused release pull request into `master`. Do not add unrelated
commits after release validation; changes after validation require the release
checks to run again.

## 5. Create and push the release tag

Update local `master` and tag the exact release-PR merge commit:

```bash
git checkout master
git pull --ff-only origin master
git tag -a vX.Y.Z -m "statgpu X.Y.Z"
git push origin vX.Y.Z
```

Pushing the tag starts `Publish to PyPI`. The workflow:

1. checks out the tagged commit;
2. verifies that the tag, `pyproject.toml`, and `statgpu.__version__` agree;
3. builds a pure-Python wheel and sdist with `STATGPU_NO_EXT=1`;
4. runs `twine check`;
5. installs the wheel in a clean environment and checks its version and core
   imports;
6. uploads `dist/*` to PyPI using the repository secret `PYPI_TOKEN`.

Never place the PyPI token in source files, committed command output, issues,
pull requests, or documentation examples.

## 6. Verify the published release

After the workflow succeeds, verify the release from a new environment:

```bash
python -m venv /tmp/statgpu-pypi-test
/tmp/statgpu-pypi-test/bin/python -m pip install --upgrade pip
/tmp/statgpu-pypi-test/bin/python -m pip install --no-cache-dir statgpu==X.Y.Z
/tmp/statgpu-pypi-test/bin/python - <<'PY'
import statgpu
from statgpu.survival import CoxPH, CoxPHCV

print(statgpu.__version__)
print(CoxPH, CoxPHCV)
PY
```

Also verify:

- the PyPI page renders the README correctly;
- the wheel is `py3-none-any`;
- the sdist is present;
- dependency extras and supported Python versions are correct;
- project, documentation, issue, and changelog links work;
- a GitHub Release is created from the same tag using the changelog as the basis
  for release notes.

## 7. Failure handling

### Version mismatch

The publish workflow stops before upload. Correct the version in a new commit,
merge a new release PR, and create a new tag. Do not move a published tag.

### Partial upload

PyPI may accept one artifact before another fails. Because filenames and
versions are immutable, inspect the release and normally issue a new patch
version rather than trying to replace the accepted file.

### Bad release already published

- yank the PyPI release when appropriate;
- fix the problem in a new patch release;
- document the incident and migration path in the changelog;
- do not rewrite Git history or reuse the released version.

## Future infrastructure improvement

The current workflow uses a project-scoped API token through `PYPI_TOKEN`.
PyPI Trusted Publishing is preferable for long-term maintenance because it
removes the stored upload token and binds publishing to a specific repository,
workflow, and optional environment. Migrate in a dedicated infrastructure pull
request and verify the trusted-publisher configuration before removing the token
path.
