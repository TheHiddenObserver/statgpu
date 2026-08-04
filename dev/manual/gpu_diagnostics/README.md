# Manual GPU diagnostics

This directory is for intentionally ad-hoc GPU reproduducers and
hardware-specific exploratory scripts.  Files here are not collected by
the maintained pytest gate.

Maintained regression coverage belongs under `dev/tests/test_*.py` and
must:

- expose discoverable pytest functions or classes;
- avoid substantial work at module import time;
- use explicit CuPy/Torch/CUDA availability checks and deterministic skip
  reasons;
- run from a clean checkout without ignored local fixtures;
- assert the current public inference and device contracts.

The historical scripts named in issue #83 were ignored local diagnostics,
not versioned test assets.  Their still-relevant contracts are represented
by maintained backend, Cox, inference, and maintenance regression tests;
future one-off scripts should be placed here rather than hidden inside
`dev/tests/`.
