"""Reproducible CPU teaching figures for hierarchical and streaming clustering.

Run from the repository root with
``PYTHONPATH=. python dev/figures/clustering_hierarchy_stream.py``.
The validation helpers deliberately reconstruct statistical quantities from raw
observations, rather than treating estimator output as its own reference.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from statgpu import AgglomerativeClustering, MiniBatchKMeans

try:
    from .clustering_common import setup, save
except ImportError:
    from clustering_common import setup, save

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def _partition(labels):
    """Compare memberships without assigning meaning to cluster numbers."""
    return {frozenset(np.flatnonzero(labels == label)) for label in np.unique(labels)}


def _tree_partition(children, n, k):
    groups = {i: frozenset([i]) for i in range(n)}
    for step, (a, b) in enumerate(children[:n - k]):
        groups[n + step] = groups.pop(int(a)) | groups.pop(int(b))
    return set(groups.values())


def build_agglomerative_data():
    """Fit the actual Ward tree and the two independently checked cuts."""
    X = np.array([[0., 0.], [.3, .45], [1.2, .1], [1.5, .65],
                  [4.1, .15], [4.45, .8]])
    models = {k: AgglomerativeClustering(n_clusters=k, linkage="ward",
                                        device="cpu").fit(X) for k in (3, 2)}
    children = np.asarray(models[3].children_).copy()
    heights = np.asarray(models[3].distances_).copy()
    sizes = np.ones(2 * len(X) - 1)
    for i, (a, b) in enumerate(children):
        sizes[len(X) + i] = sizes[a] + sizes[b]
    data = dict(X=X, children=children, heights=heights,
                linkage=np.column_stack([children, heights, sizes[len(X):]]),
                labels={k: np.asarray(m.labels_).copy() for k, m in models.items()},
                cuts={k: float((heights[len(X)-k-1] + heights[len(X)-k]) / 2)
                      for k in (3, 2)})
    validate_agglomerative_data(data)
    return data


def validate_agglomerative_data(data):
    """Check greedy Ward costs, tree sizes, height scale, and both memberships."""
    X, children, heights = data["X"], data["children"], data["heights"]
    n = len(X)
    assert children.shape == (n - 1, 2)
    assert np.all(np.diff(heights) > 0)
    active = {i: [i] for i in range(n)}
    def cost(a, b):
        A, B = X[active[a]], X[active[b]]
        return len(A) * len(B) / (len(A) + len(B)) * np.sum((A.mean(0)-B.mean(0))**2)
    for step, (a, b) in enumerate(children):
        a, b = int(a), int(b)
        assert a in active and b in active and a != b
        delta = cost(a, b)
        candidates = [cost(i, j) for i in active for j in active if i < j]
        np.testing.assert_allclose(delta, min(candidates), atol=1e-12)
        np.testing.assert_allclose(heights[step], np.sqrt(2 * delta), atol=1e-12)
        active[n + step] = active.pop(a) + active.pop(b)
        assert data["linkage"][step, 3] == len(active[n + step])
    assert sorted(next(iter(active.values()))) == list(range(n))
    expected = {3: {frozenset([0, 1]), frozenset([2, 3]), frozenset([4, 5])},
                2: {frozenset([0, 1, 2, 3]), frozenset([4, 5])}}
    for k in (3, 2):
        assert _partition(data["labels"][k]) == expected[k]
        assert _tree_partition(children, n, k) == expected[k]
        assert np.count_nonzero(heights < data["cuts"][k]) == n-k
    return True


def build_minibatch_data():
    """Record four actual partial_fit updates, with explicit finite initialization."""
    initial = np.array([[-2.5, -1.], [2.5, -1.]])
    means = np.array([[[-2.4, 0.], [2.3, 0.]], [[-2., 1.], [2., 1.]],
                      [[-1.6, 2.], [1.6, 2.]], [[-1.3, 3.], [1.2, 3.]]])
    batch_sizes = [(3, 2), (2, 4), (4, 2), (3, 4)]
    batches = []
    for centers, sizes in zip(means, batch_sizes):
        chunks = []
        for center, count in zip(centers, sizes):
            t = np.linspace(-1., 1., count)
            chunks.append(center + np.column_stack([.16*t, .10*t]))
        batches.append(np.vstack(chunks))
    model = MiniBatchKMeans(n_clusters=2, init=initial.copy(), n_init=1,
                           random_state=17, device="cpu")
    centers, counts = [initial.copy()], [np.zeros(2)]
    for batch in batches:
        model.partial_fit(batch)
        centers.append(np.asarray(model.cluster_centers_).copy())
        counts.append(np.asarray(model.counts_).copy())
    data = dict(initial=initial, batches=batches, centers=np.array(centers),
                counts=np.array(counts), n_steps=model.n_steps_)
    validate_minibatch_data(data)
    return data


def validate_minibatch_data(data):
    """Reconstruct each assignment, cumulative count, sum, and center from scratch."""
    centers = data["initial"].copy()
    sums, counts = np.zeros_like(centers), np.zeros(len(centers))
    np.testing.assert_allclose(data["centers"][0], centers)
    np.testing.assert_allclose(data["counts"][0], counts)
    for step, batch in enumerate(data["batches"], 1):
        labels = np.argmin(np.sum((batch[:, None, :] - centers[None, :, :])**2, axis=2), axis=1)
        for j in range(len(centers)):
            selected = batch[labels == j]
            counts[j] += len(selected)
            sums[j] += selected.sum(axis=0)
            if counts[j]:
                centers[j] = sums[j] / counts[j]
        np.testing.assert_allclose(data["centers"][step], centers, atol=1e-12)
        np.testing.assert_array_equal(data["counts"][step], counts)
        assert counts.sum() == sum(map(len, data["batches"][:step]))
    np.testing.assert_array_equal(data["counts"][-1], [12, 12])
    assert data["n_steps"] == 4
    assert np.all(np.diff(data["counts"], axis=0) >= 0)
    return True


def _finish_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.17)
    ax.set_axisbelow(True)


def plot_agglomerative(output_dir, lang="en"):
    setup(lang)
    data = build_agglomerative_data()
    cn = lang == "cn"
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.subplots_adjust(left=.065, right=.985, bottom=.24, top=.80, wspace=.38)
    fig.suptitle("层次聚类：同一棵 Ward 树，两种切分" if cn else
                 "Agglomerative clustering: one Ward tree, two cuts", y=.96, fontsize=15)
    ax = axes[0]
    dendrogram(data["linkage"], ax=ax, labels=[str(i+1) for i in range(6)],
               color_threshold=0, above_threshold_color="#6B7280", leaf_font_size=11)
    for letter, k, color, style in [("A", 3, BLUE, "--"), ("B", 2, ORANGE, ":")]:
        h = data["cuts"][k]
        ax.axhline(h, color=color, ls=style, lw=1.8)
        ax.text(1, h+.14, f"{letter}: h={h:.2f}, k={k}", color=color, fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
    ax.set_title("合并树" if cn else "Merge tree", fontsize=12)
    ax.set_xlabel("样本编号" if cn else "Observation ID")
    ax.set_ylabel("Ward 高度  √(2ΔSSE)" if cn else "Ward height  √(2ΔSSE)")
    ax.set_ylim(0, data["heights"][-1]*1.12)
    _finish_axes(ax)
    for ax, k, letter, cut_color in zip(axes[1:], (3, 2), ("A", "B"), (BLUE, ORANGE)):
        groups = sorted(_partition(data["labels"][k]), key=min)
        colors = [BLUE, ORANGE, GREEN] if k == 3 else [BLUE, GREEN]
        markers = ["o", "s", "^"] if k == 3 else ["o", "^"]
        for group, color, marker in zip(groups, colors, markers):
            idx = sorted(group)
            ax.scatter(*data["X"][idx].T, c=color, marker=marker, s=65,
                       edgecolors="white", linewidths=.7, zorder=3)
        for i, (x, y) in enumerate(data["X"]):
            ax.annotate(str(i+1), (x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=11)
        ax.set_title(f"{letter}: k={k}  (h={data['cuts'][k]:.2f})", color=cut_color, fontsize=12)
        ax.set_xlim(-.5, 5)
        ax.set_ylim(-1.8, 2.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("特征 1" if cn else "Feature 1")
        ax.set_ylabel("特征 2" if cn else "Feature 2")
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([-1, 0, 1, 2])
        _finish_axes(ax)
    fig.text(.5, .085, "ΔSSE 为合并导致的簇内平方和增量；高度不是概率。" if cn else
             "ΔSSE is the increase in within-cluster squared error; height is not a probability.",
             ha="center", fontsize=11)
    return save(fig, output_dir, "agglomerative", lang)


def plot_minibatch(output_dir, lang="en"):
    setup(lang)
    data = build_minibatch_data()
    cn = lang == "cn"
    fig, (ax, count_ax) = plt.subplots(1, 2, figsize=(10, 4.4),
                                     gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.07, right=.975, bottom=.23, top=.80, wspace=.28)
    fig.suptitle("小批量 K-Means：逐批更新，累积计数" if cn else
                 "MiniBatchKMeans: update by batch, retain cumulative counts", y=.96, fontsize=15)
    X = np.vstack(data["batches"])
    ax.scatter(*X.T, s=15, c="#9CA3AF", alpha=.65,
               label="_nolegend_")
    ax.text(.5, .97, "灰点：全部批次样本" if cn else "Gray: samples from all batches",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555555")
    for j, (color, marker) in enumerate([(BLUE, "s"), (ORANGE, "D")]):
        trajectory = data["centers"][:, j]
        ax.plot(*trajectory.T, color=color, marker=marker, markersize=6, lw=1.3,
                label=f"中心 {j+1}" if cn else f"Center {j+1}")
        for step, point in enumerate(trajectory):
            ax.annotate(str(step), point, xytext=(-13 if j == 0 else 11, 0),
                        textcoords="offset points", color=color, ha="center", va="center", fontsize=11)
        for start, end in zip(trajectory[:-1], trajectory[1:]):
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=5, shrinkB=5))
        count_ax.plot(np.arange(5), data["counts"][:, j], color=color, marker=marker,
                      lw=1.8, linestyle="-" if j == 0 else "--",
                      label=f"中心 {j+1}" if cn else f"Center {j+1}")
    ax.set_title("中心轨迹（0 为初始值）" if cn else "Center paths (0 = initialization)", fontsize=12)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.5, 3.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("特征 1" if cn else "Feature 1")
    ax.set_ylabel("特征 2" if cn else "Feature 2")
    ax.legend(loc="center", fontsize=9, frameon=False)
    count_ax.set_title("累计分配样本数" if cn else "Cumulative assigned observations", fontsize=12)
    count_ax.set_xlabel("partial_fit 批次" if cn else "partial_fit batch")
    count_ax.set_ylabel("累计计数" if cn else "Cumulative count")
    count_ax.set_xticks(range(5))
    count_ax.set_yticks(range(0, 13, 3))
    count_ax.set_ylim(-.5, 13)
    count_ax.legend(loc="upper left", frameon=False, fontsize=10)
    for axis in (ax, count_ax):
        _finish_axes(axis)
    fig.text(.5, .09, "中心 = 历次分配样本之和 / 累计计数；旧样本的权重不会自动衰减。" if cn else
             "Center = sum of observations assigned so far / cumulative count; no automatic forgetting.",
             ha="center", fontsize=11)
    return save(fig, output_dir, "minibatch-kmeans", lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/unsupervised"))
    parser.add_argument("--lang", choices=["en", "cn", "both"], default="both")
    args = parser.parse_args()
    for language in (["en", "cn"] if args.lang == "both" else [args.lang]):
        plot_agglomerative(args.output_dir, language)
        plot_minibatch(args.output_dir, language)
