"""Deterministic, analytically checked GMM and DBSCAN teaching figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from statgpu import GaussianMixture, DBSCAN

try:
    from .clustering_common import setup, save
except ImportError:
    from clustering_common import setup, save

BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'


def ellipse_points(mean, covariance, radius, n=241):
    """Points at a fixed fitted-component Mahalanobis radius (not a CI)."""
    values, vectors = np.linalg.eigh(covariance)
    theta = np.linspace(0, 2 * np.pi, n)
    unit = np.vstack((np.cos(theta), np.sin(theta)))
    return mean + (vectors @ (np.sqrt(values)[:, None] * unit) * radius).T


def build_gmm_data():
    rng = np.random.default_rng(20261014)
    X = np.vstack((
        rng.multivariate_normal([-0.9, -0.15], [[0.85, 0.42], [0.42, 0.65]], 110),
        rng.multivariate_normal([1.05, 0.35], [[0.75, -0.3], [-0.3, 0.55]], 110),
    ))
    model = GaussianMixture(n_components=2, covariance_type='full',
                            random_state=17, n_init=5, max_iter=500,
                            tol=1e-7, reg_covar=1e-6, device='cpu').fit(X)
    # Canonicalize display identities only; component numbers have no intrinsic order.
    order = np.argsort(np.asarray(model.means_)[:, 0])
    means = np.asarray(model.means_)[order]
    covariances = np.asarray(model.covariances_)[order]
    weights = np.asarray(model.weights_)[order]
    responsibilities = np.asarray(model.predict_proba(X))[:, order]
    return dict(X=X, model=model, order=order, means=means,
                covariances=covariances, weights=weights,
                responsibilities=responsibilities,
                labels=np.argmax(responsibilities, axis=1),
                ellipses=[[ellipse_points(m, c, r) for r in (1., 2.)]
                          for m, c in zip(means, covariances)])


def validate_gmm_data(data):
    X = data['X']
    assert data['model'].converged_
    density = []
    for w, mean, cov in zip(data['weights'], data['means'], data['covariances']):
        assert np.all(np.linalg.eigvalsh(cov) > 0)
        diff = X - mean
        quad = np.einsum('ni,ij,nj->n', diff, np.linalg.inv(cov), diff)
        density.append(w * np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(cov))))
    density = np.asarray(density).T
    analytic = density / density.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(data['responsibilities'], analytic, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(data['responsibilities'].sum(axis=1), 1., atol=1e-12)
    np.testing.assert_array_equal(data['labels'], np.argmax(analytic, axis=1))
    assert np.count_nonzero((analytic[:, 1] > .2) & (analytic[:, 1] < .8)) >= 15
    for mean, cov, ellipses in zip(data['means'], data['covariances'], data['ellipses']):
        for radius, points in zip((1., 2.), ellipses):
            diff = points - mean
            q = np.einsum('ni,ij,nj->n', diff, np.linalg.inv(cov), diff)
            np.testing.assert_allclose(q, radius ** 2, atol=1e-12)
    return True


def build_dbscan_data():
    cross = np.array([[0., 0.], [.25, 0.], [-.25, 0.], [0., .25], [0., -.25]])
    X = np.vstack((cross, cross + [2.4, .4], [[.75, 0.], [3.15, .4]],
                   [[1.2, 1.4], [-.8, -.8], [3.1, -.8]]))
    eps, min_samples = .52, 5
    model = DBSCAN(eps=eps, min_samples=min_samples, device='cpu').fit(X)
    # Independent dense distances, including the diagonal/self-neighbor.
    adjacency = np.linalg.norm(X[:, None] - X[None, :], axis=2) <= eps
    counts = adjacency.sum(axis=1)
    core = counts >= min_samples
    labels = np.full(len(X), -1, dtype=int)
    component = 0
    for root in np.flatnonzero(core):
        if labels[root] != -1:
            continue
        labels[root] = component
        stack = [root]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(adjacency[i] & core & (labels == -1)):
                labels[j] = component
                stack.append(j)
        component += 1
    for i in np.flatnonzero(~core):
        neighbors = np.flatnonzero(adjacency[i] & core)
        if len(neighbors):
            # Fixture has no ambiguous border shared by disconnected core components.
            assert len(np.unique(labels[neighbors])) == 1
            labels[i] = labels[neighbors[0]]
    return dict(X=X, model=model, eps=eps, min_samples=min_samples,
                adjacency=adjacency, counts=counts, core=core,
                labels=np.asarray(model.labels_), reference_labels=labels,
                border=(~core) & (labels >= 0), noise=labels == -1,
                circle_center=X[0], circle_radius=eps)


def validate_dbscan_data(data):
    labels, reference = data['labels'], data['reference_labels']
    np.testing.assert_array_equal(labels == -1, reference == -1)
    # Partition comparison is invariant to arbitrary cluster label permutations.
    np.testing.assert_array_equal(labels[:, None] == labels[None, :],
                                  reference[:, None] == reference[None, :])
    np.testing.assert_array_equal(data['model'].core_sample_indices_, np.flatnonzero(data['core']))
    np.testing.assert_array_equal(data['counts'], data['adjacency'].sum(axis=1))
    assert (data['core'].sum(), data['border'].sum(), data['noise'].sum()) == (10, 2, 3)
    assert data['counts'][0] == 5
    assert data['circle_radius'] == data['model'].eps
    np.testing.assert_array_equal(data['circle_center'], data['X'][0])
    return True


def _axes(ax, cn):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('特征 1' if cn else 'Feature 1')
    ax.set_ylabel('特征 2' if cn else 'Feature 2')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)


def make_gmm(output_dir, lang='en'):
    setup(lang)
    cn = lang == 'cn'
    d = build_gmm_data()
    validate_gmm_data(d)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    fig.subplots_adjust(left=.07, right=.90, bottom=.34, top=.85, wspace=.30)
    cmap = plt.get_cmap('cividis')
    for ax in axes:
        _axes(ax, cn)
        ax.set_xlim(-3.5, 3.8)
        ax.set_ylim(-2.6, 2.8)
    for k, (color, marker) in enumerate(zip((BLUE, ORANGE), ('o', '^'))):
        subset = d['labels'] == k
        for ax, soft in zip(axes, (False, True)):
            ax.scatter(*d['X'][subset].T, s=20,
                       c=d['responsibilities'][subset, 1] if soft else color,
                       **dict(cmap=cmap, vmin=0, vmax=1) if soft else {},
                       marker=marker, edgecolors='#333333', linewidths=.3, zorder=3)
        for radius, points in zip((1, 2), d['ellipses'][k]):
            axes[0].plot(*points.T, color=color, lw=1.6,
                         ls='-' if radius == 1 else '--')
    axes[0].set_title('硬分配与拟合分量' if cn else 'Hard assignments + fitted components', fontsize=12)
    axes[1].set_title('软分配：分量 2 的概率' if cn else 'Soft assignments: component 2 probability', fontsize=12)
    cax = fig.add_axes([.92, .31, .018, .46])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, ticks=[0, .5, 1])
    bar.solids.set_rasterized(False)  # Keep the probability colorbar fully vector.
    bar.ax.tick_params(labelsize=10)
    fig.suptitle('GMM：重叠区域可以保留归属的不确定性' if cn else 'GMM preserves uncertain membership in overlap', fontsize=14)
    fig.text(.5, .16, '形状 = 最大概率分量：圆形 1，三角形 2' if cn else 'Shape = largest-probability component: circle 1, triangle 2', ha='center', fontsize=11)
    fig.text(.5, .095, '椭圆：拟合分量的等密度线（马氏距离 1、2），不是置信区间。' if cn else 'Ellipses: fitted iso-density contours (Mahalanobis radii 1, 2), not confidence intervals.', ha='center', fontsize=11)
    fig.text(.5, .035, '颜色表示分量归属概率，不是预测正确的概率。' if cn else 'Color shows component membership probability, not probability of being correct.', ha='center', fontsize=11)
    result = save(fig, output_dir, 'gmm', lang)
    plt.close(fig)
    return result


def make_dbscan(output_dir, lang='en'):
    setup(lang)
    cn = lang == 'cn'
    d = build_dbscan_data()
    validate_dbscan_data(d)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=.07, right=.65, bottom=.30, top=.86)
    _axes(ax, cn)
    ax.set_xlim(-1.2, 3.65)
    ax.set_ylim(-1.15, 1.8)
    for k, color in enumerate((BLUE, ORANGE)):
        for mask, fill in ((d['core'], True), (d['border'], False)):
            subset = mask & (d['reference_labels'] == k)
            ax.scatter(*d['X'][subset].T, s=105, marker='o' if k == 0 else '^',
                       facecolors=color if fill else 'white', edgecolors=color,
                       linewidths=1.8, zorder=3)
    ax.scatter(*d['X'][d['noise']].T, marker='x', c='#555555', s=70, linewidths=1.8, zorder=3)
    ax.add_patch(Circle(d['circle_center'], d['eps'], fill=False, ec=GREEN, lw=1.8, ls='--'))
    ax.annotate('eps = 0.52', xy=(0, -.52), xytext=(.1, -.91), fontsize=11,
                arrowprops=dict(arrowstyle='-', color=GREEN), color=GREEN)
    ax.annotate('边界点' if cn else 'Border', xy=(.75, 0), xytext=(1., -.4), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#555555'))
    fig.suptitle('DBSCAN：局部密度决定核心点、边界点和噪声' if cn else 'DBSCAN: core points, border points and noise', fontsize=14)
    text = ('min_samples = 5（含自身）\n\n实心：核心点（10 个）\n邻域内至少有 5 个点\n\n空心：边界点（2 个）\n非核心点，但邻近核心点\n\n×：噪声（3 个）\n不在任何核心点邻域内' if cn else
            'min_samples = 5 (includes self)\n\nFilled: core (10 points)\nAt least 5 points in its neighborhood\n\nHollow: border (2 points)\nNot core; within eps of a core point\n\n×: noise (3 points)\nOutside every core neighborhood')
    fig.text(.68, .79, text, va='top', fontsize=11, linespacing=1.25)
    fig.text(.5, .10, '圆内有 5 个点，包含圆心自身。蓝色圆形与橙色三角形表示两个簇。' if cn else 'The circle contains 5 points, including its center. Blue circles and orange triangles identify two clusters.', ha='center', fontsize=11)
    fig.text(.5, .04, '相邻的核心点相连成簇；边界点不会继续扩展簇。' if cn else 'Connected core points form clusters; border points do not extend a cluster.', ha='center', fontsize=11)
    result = save(fig, output_dir, 'dbscan', lang)
    plt.close(fig)
    return result


if __name__ == '__main__':
    output = Path(__file__).resolve().parents[2] / 'docs/assets/unsupervised'
    for language in ('en', 'cn'):
        make_gmm(output, language)
        make_dbscan(output, language)
