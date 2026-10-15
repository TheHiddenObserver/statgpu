"""Generate all EN/CN scientific clustering figures from the repository root."""
from pathlib import Path
import argparse
import sys
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/statgpu-clustering-mpl')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev.figures.clustering_common import setup
from dev.figures.clustering_kmeans import make_kmeans
from dev.figures.clustering_density import make_gmm, make_dbscan
from dev.figures.clustering_hierarchy_stream import plot_agglomerative, plot_minibatch

GENERATORS = (make_kmeans, make_gmm, make_dbscan, plot_agglomerative, plot_minibatch)


def generate(output_dir, languages=('en', 'cn')):
    return [fn(output_dir, lang) for lang in languages for fn in GENERATORS]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'docs/assets/unsupervised')
    args = parser.parse_args()
    for path in generate(args.output_dir):
        print(path)
