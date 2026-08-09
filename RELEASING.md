# Releasing statgpu to PyPI and GitHub

This document is for maintainers preparing an official `statgpu` release.
The repository publishes when a tag matching `v*` is pushed.

The release automation is defined in:

- [`.github/workflows/publish.yml`](.github/workflows/publish.yml) for PyPI and GitHub Release publication;
- [`.github/workflows/release-package.yml`](.github/workflows/release-package.yml) for wheel and sdist validation;
- [`.github/workflows/release-notes.yml`](.github/workflows/release-notes.yml) for versioned GitHub Release-note validation.

## Release model

The package version is maintained in two files and must match:

- `pyproject.toml`: `project.version`;
- `statgpu/__init__.py`: `__version__`.

A release tag uses the same version with a leading `v`:

```text
package version: 0.2.3
tag:             v0.2.3
```

Each release also has one authoritative GitHub Release body:

```text
.github/releases/vX.Y.Z.md
```

For example, the GitHub Release notes for 0.2.3 are stored at:

```text
.github/releases/v0.2.3.md
```

The tag workflow publishes this file verbatim as the GitHub Release body. Do not
rely on GitHub's automatically generated PR list as the primary release notes,
and do not compose the final release body manually in the GitHub UI.

PyPI release files are immutable. A broken upload cannot be replaced under the
same version; prepare a new patch version instead.

## 1. Prepare a focused release pull request

Start from the latest `master` after the intended feature and fix pull requests
are merged.

Update both version declarations and all release-facing sources:

- `pyproject.toml`;
- `statgpu/__init__.py`;
- `.github/releases/vX.Y.Z.md`;
- `CHANGELOG.md`;
- `docs/en/changelog.md`;
- `docs/cn/changelog.md`;
- README or model documentation when installation, compatibility, limitations,
  or public behavior changed.

The versioned GitHub Release document must be user-facing. It should explain:

- what major capability was added or changed;
- which public APIs and workflows are affected;
- installation and platform support;
- behavioral changes and upgrade implications;
- known limitations and unsupported combinations;
- validation evidence without turning the document into an internal audit log;
- links to the main implementation PR, release PR, version comparison, and
  repository changelog.

Keep release-only changes separate from implementation work. A release pull
request should primarily contain version, packaging, changelog, release notes,
and release-validation changes.

## 2. Validate the release pull request

The normal `Tests` workflow must pass, including the complete CPU suite, static
contracts, documentation contracts, and the Python 3.9–3.12 regression matrix.
For changes affecting CuPy, Torch, inference, device routing, or performance,
record physical-GPU acceptance on the exact release source commit.

Physical evidence is tied to both the numerical implementation and the validator
that defines the acceptance matrix. If a physical-validation runner changes after
an artifact has been accepted—for example, because review adds a previously
uncovered inference branch—the old artifact remains useful historical evidence
but no longer proves the new acceptance contract. Return the affected pull
request to a pending/draft state, rerun the changed validator on an exact clean
candidate head, and only then promote new canonical evidence or restore a
Ready/merge-ready conclusion.

### Package validation

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

### GitHub Release-note validation

The `Release notes validation` workflow requires:

- `.github/releases/vX.Y.Z.md` matching the package version;
- a title of the form `# statgpu X.Y.Z`;
- substantive Highlights, Installation and platform support, Validation,
  Upgrade notes and known limits, and Full change history sections;
- no unresolved `TODO`, `TBD`, `X.Y.Z`, or similar placeholders;
- explicit installation, platform, and version-comparison information;
- the same version to appear in the root, English, and Chinese changelogs.

This gate prevents a release tag from being prepared with a generic or
incomplete GitHub Release description.

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
- `.github/releases/vX.Y.Z.md` accurately describes the user-visible release;
- root, English, and Chinese changelogs are synchronized;
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
3. verifies that `.github/releases/vX.Y.Z.md` exists and has the required
   versioned sections;
4. builds a pure-Python wheel and sdist with `STATGPU_NO_EXT=1`;
5. runs `twine check`;
6. installs the wheel in a clean environment and checks its version and core
   imports;
7. retains the validated distributions as a workflow artifact;
8. uploads the distributions to PyPI using the repository secret `PYPI_TOKEN`;
9. only after the PyPI job succeeds, creates or updates the GitHub Release using
   `.github/releases/vX.Y.Z.md` as the exact release body and attaches the same
   wheel and sdist.

PyPI publication and GitHub Release creation are separate jobs. If the GitHub
Release job fails after PyPI succeeds, rerun only the failed job; the successful
PyPI upload does not need to be repeated.

Never place the PyPI token in source files, committed command output, issues,
pull requests, or documentation examples.

## 6. Verify the published release

After the workflow succeeds, verify the PyPI package from a new environment:

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
- the GitHub Release title is `statgpu X.Y.Z`;
- the GitHub Release body matches `.github/releases/vX.Y.Z.md` rather than an
  automatically generated PR summary;
- the GitHub Release includes the same wheel and sdist published by the tag
  workflow.

## 7. Failure handling

### Version or release-note mismatch

The publish workflow stops before upload. Correct the version or release-note
file in a new commit, merge a new release PR, and create a new tag. Do not move a
published tag.

### Partial PyPI upload

PyPI may accept one artifact before another fails. Because filenames and
versions are immutable, inspect the release and normally issue a new patch
version rather than trying to replace the accepted file.

### GitHub Release publication failure

If the PyPI job succeeded and only the GitHub Release job failed, rerun the
failed GitHub Release job. It is idempotent: an existing release is updated from
the versioned notes file and attached artifacts are uploaded with replacement.

### Bad release already published

- yank the PyPI release when appropriate;
- fix the problem in a new patch release;
- document the incident and migration path in the changelog and versioned release
  notes;
- do not rewrite Git history or reuse the released version.

## Future infrastructure improvement

The current workflow uses a project-scoped API token through `PYPI_TOKEN`.
PyPI Trusted Publishing is preferable for long-term maintenance because it
removes the stored upload token and binds publishing to a specific repository,
workflow, and optional environment. Migrate in a dedicated infrastructure pull
request and verify the trusted-publisher configuration before removing the token
path.
