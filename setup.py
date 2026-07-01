"""statgpu setup.py — builds all optional Cython extensions.

Usage:
    python setup.py build_ext --inplace   # compile locally
    pip install -e .                       # editable install
"""
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


def get_ext_modules():
    """Return list of Cython extension modules to build."""
    try:
        import numpy as np
        from Cython.Build import cythonize
    except Exception:
        return []

    extensions = [
        # Survival
        Extension(
            "statgpu.survival._cox_efron_cy",
            ["statgpu/survival/_cox_efron_cy.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        ),
        # Unsupervised — DBSCAN
        Extension(
            "statgpu.unsupervised._dbscan_cpu",
            ["statgpu/unsupervised/_dbscan_cpu.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "statgpu.unsupervised._dbscan_cy_fast",
            ["statgpu/unsupervised/_dbscan_cy_fast.pyx"],
            include_dirs=[np.get_include()],
        ),
        # Unsupervised — KD-tree
        Extension(
            "statgpu.unsupervised._kdtree",
            ["statgpu/unsupervised/_kdtree.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        ),
        # Unsupervised — Union-Find
        Extension(
            "statgpu.unsupervised._unionfind",
            ["statgpu/unsupervised/_unionfind.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ]

    return cythonize(extensions, compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False})


build_commands = {"build", "build_ext", "bdist_wheel", "develop", "install"}
if build_commands.intersection(sys.argv):
    ext_modules = get_ext_modules()
else:
    ext_modules = []


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
