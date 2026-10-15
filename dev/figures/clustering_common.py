"""Shared deterministic rendering style for clustering teaching figures."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/statgpu-clustering-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
MARKERS = ['o', '^', 's', 'D']
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')


def setup(lang):
    if lang not in ('en', 'cn'):
        raise ValueError('language must be en or cn')
    if lang == 'cn':
        if not FONT.exists():
            raise RuntimeError('Chinese figures require Noto Sans CJK Regular; install this font before regeneration')
        font_manager.fontManager.addfont(str(FONT))
        family = font_manager.FontProperties(fname=str(FONT)).get_name()
    else:
        family = 'DejaVu Sans'
    plt.rcParams.update({'font.family': family, 'font.size': 11,
        'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'svg.fonttype': 'path',
        'svg.hashsalt': 'statgpu-clustering-teaching-v1',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white'})


def save(fig, output_dir, slug, lang):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{slug}-{lang}.svg'
    fig.savefig(path, format='svg', metadata={'Date': None, 'Creator': 'statgpu deterministic teaching figures'})
    plt.close(fig)
    return path
