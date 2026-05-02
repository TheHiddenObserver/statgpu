from __future__ import annotations

import warnings
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build optional C extensions without making installation fail."""

    def run(self):
        try:
            super().run()
        except Exception as exc:
            warnings.warn(f"optional Cython extensions were not built: {exc}", RuntimeWarning)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            warnings.warn(f"optional extension {ext.name!r} was not built: {exc}", RuntimeWarning)

build_commands = {"build", "build_ext", "bdist_wheel", "develop", "install"}
if build_commands.intersection(sys.argv):
    try:
        import numpy as np
        from Cython.Build import cythonize
    except Exception:
        ext_modules = []
    else:
        ext_modules = cythonize(
            [
                Extension(
                    "statgpu.unsupervised._dbscan_cpu",
                    ["statgpu/unsupervised/_dbscan_cpu.pyx"],
                    include_dirs=[np.get_include()],
                )
            ],
            compiler_directives={"language_level": "3"},
        )
else:
    ext_modules = []


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
