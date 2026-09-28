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
    "docs/cn/unsupervised/pca.md",
    "docs/cn/unsupervised/README.md",
    "docs/cn/unsupervised/kmeans.md",
    "docs/cn/unsupervised/minibatch-kmeans.md",
    "docs/cn/unsupervised/nmf.md",
    "docs/cn/unsupervised/truncated-svd.md",
    "docs/cn/unsupervised/tsne.md",
    "docs/cn/models/unsupervised.md",
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
    "batch/streaming",
    "centered dense data",
    "retained variance",
    "sign-aware comparison",
    "transformed scores",
    "component scores",
    "host SciPy graph assembly",
    "Fuzzy graph cross-entropy",
    "cluster centers",
    "greedy `k-means++`",
    "callable init",
    "squared center movement",
    "Cluster ID",
    "permutation-invariant label metrics",
    "dense observations",
    "exact Lloyd polishing",
    "batch 顺序",
    "latent dimension",
    "reconstruction differences",
    "reconstruction error",
    "component sign convention",
    "sign/subspace invariant",
    "Gaussian affinity",
    "Student-t affinity",
    "adaptive gains",
    "exact dense t-SNE",
    "external baseline",
    "finite-input",
    "standard errors",
    "confidence intervals",
    "p-values",
    "Frobenius loss",
    "dense data",
    "dense input",
    "dense 数据",
    "dense 输入",
)


@pytest.mark.parametrize("page", PAGES)
def test_naturalized_pages_avoid_literal_translation_fragments(page):
    text = (ROOT / page).read_text(encoding="utf-8")
    present = [fragment for fragment in FRAGMENTS_TO_AVOID if fragment in text]
    assert not present, f"{page}: literal-translation fragments {present!r}"
