"""One analytically computed Lloyd assignment/update, not a fit trace."""
import numpy as np
import matplotlib.pyplot as plt
try:
    from .clustering_common import setup, save, COLORS, MARKERS
except ImportError:
    from clustering_common import setup, save, COLORS, MARKERS


def build_kmeans_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [3, 1], [4, 0], [4, 1], [4, 2]], dtype=float)
    old = np.array([[0, 0], [4, 2]], dtype=float)
    distances = ((X[:, None] - old[None, :])**2).sum(axis=2)
    labels = distances.argmin(axis=1)
    new = np.array([X[labels == k].mean(axis=0) for k in range(2)])
    before = distances[np.arange(len(X)), labels].sum()
    after = ((X - new[labels])**2).sum()
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_allclose(new, [[.5, .5], [3.75, 1]])
    np.testing.assert_allclose([before, after], [11, 4.75])
    assert after <= before
    return dict(X=X, old=old, new=new, labels=labels, before=before, after=after)


def make_kmeans(output_dir, lang):
    setup(lang)
    d = build_kmeans_data()
    cn = lang == 'cn'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    fig.subplots_adjust(left=.065, right=.98, bottom=.26, top=.84, wspace=.22)
    titles = ['1. 按最近中心分配', '2. 固定分配，移至均值'] if cn else ['1. Assign to the nearest center', '2. Hold assignments; move to means']
    for j, ax in enumerate(axes):
        for k in range(2):
            points = d['X'][d['labels'] == k]
            ax.scatter(*points.T, c=COLORS[k], marker=MARKERS[k], s=64, edgecolors='white', linewidths=.5)
            center = d['old'][k] if j == 0 else d['new'][k]
            for point in points:
                ax.plot([point[0],center[0]],[point[1],center[1]], color=COLORS[k], alpha=.35, lw=1, zorder=0)
            ax.scatter(*d['old'][k], marker='X', c=COLORS[k], s=145, edgecolors='black', linewidths=.8)
            if j:
                ax.scatter(*center, marker='*', c=COLORS[k], s=220, edgecolors='black', linewidths=.8, zorder=5)
                ax.annotate('', xy=center, xytext=d['old'][k], arrowprops=dict(arrowstyle='->', color='black', lw=1.6))
        ax.set(title=titles[j], xlabel='$x_1$', ylabel='$x_2$', xlim=(-.5,4.6), ylim=(-.5,2.6), xticks=range(5), yticks=range(3))
        ax.set_aspect('equal')
        ax.grid(alpha=.15)
        ax.text(.5,-.26, ('固定分配的平方距离和' if cn else 'Squared-distance sum, fixed assignments')+f" = {d['before'] if j == 0 else d['after']:g}", transform=ax.transAxes, ha='center', fontsize=10)
    from matplotlib.lines import Line2D
    handles = [Line2D([],[],color=COLORS[k],marker=MARKERS[k],ls='',label=(f'分组 {k}' if cn else f'Group {k}')) for k in range(2)]
    handles += [Line2D([],[],color='black',marker='X',ls='',label='旧中心' if cn else 'Old center'), Line2D([],[],color='black',marker='*',markersize=12,ls='',label='更新后的均值' if cn else 'Updated mean')]
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,.015),frameon=False)
    return save(fig, output_dir, 'kmeans', lang)
