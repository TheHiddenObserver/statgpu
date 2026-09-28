"""Chinese prose naturalization contracts for previously uncleaned pages."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

PAGES = (
    "docs/cn/unsupervised/agglomerative-clustering.md",
    "docs/cn/unsupervised/umap.md",
    "docs/cn/unsupervised/dbscan.md",
    "docs/cn/unsupervised/minibatch-nmf.md",
    "docs/cn/unsupervised/incremental-pca.md",
    "docs/cn/unsupervised/gaussian-mixture.md",
)

FRAGMENTS_TO_AVOID = (
    "dense mini-batches",
    "Frobenius reconstruction loss",
    "strict statistical inference",
    "batch order",
    "running mean",
    "SVD basis",
    "compact matrix",
    "mean-correction",
    "sparse input",
    "CD solver",
    "beta loss",
    "density-connected components",
    "dense Euclidean",
    "Python fallback",
    "label permutation-invariant",
    "within-cluster squared error",
    "backend-resident",
    "device-native",
    "host-side",
    "host memory",
    "fuzzy-set cross-entropy",
    "unseen samples",
    "最新远程 artifact",
    "reconstruction loss",
    "batch activations",
    "explained variance",
    "batch/streaming",
)


@pytest.mark.parametrize("page", PAGES)
def test_naturalized_pages_avoid_literal_translation_fragments(page):
    text = (ROOT / page).read_text(encoding="utf-8")
    present = [fragment for fragment in FRAGMENTS_TO_AVOID if fragment in text]
    assert not present, f"{page}: literal-translation fragments {present!r}"
