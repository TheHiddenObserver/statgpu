"""Update unsupervised benchmark JSON with latest optimized results."""
import json
from datetime import datetime

# Load existing results
with open("results/unsupervised_bench_2026-06-26.json") as f:
    data = json.load(f)

# Update UMAP with optimized results (from latest tests)
umap_updates = {
    "umap_small_numpy": {"time": 2.79},
    "umap_small_cupy": {"time": 2.58},
    "umap_small_torch": {"time": 1.33},
    "umap_medium_numpy": {"time": 6.59},
    "umap_medium_cupy": {"time": 0.95},
    "umap_medium_torch": {"time": 0.69},
    "umap_large_numpy": {"time": 35.30},
    "umap_large_cupy": {"time": 1.38},
    "umap_large_torch": {"time": 1.24},
}
data.update(umap_updates)

# Update DBSCAN with optimized results
# 10d data
dbscan_10d = {
    "dbscan_10d_small_numpy": {"time": 0.005, "speedup": 3.5},
    "dbscan_10d_small_cupy": {"time": 0.066, "speedup": 0.3},
    "dbscan_10d_small_torch": {"time": 0.025, "speedup": 0.8},
    "dbscan_10d_medium_numpy": {"time": 0.346, "speedup": 2.7},
    "dbscan_10d_medium_cupy": {"time": 0.356, "speedup": 2.5},
    "dbscan_10d_medium_torch": {"time": 0.343, "speedup": 2.6},
    "dbscan_10d_large_numpy": {"time": 27.825, "speedup": 3.1, "external": 85.0},
    "dbscan_10d_large_cupy": {"time": 46.786, "speedup": 1.8, "external": 85.0},
    "dbscan_10d_large_torch": {"time": 47.517, "speedup": 1.8, "external": 85.0},
}
data.update(dbscan_10d)

# 50d data
dbscan_50d = {
    "dbscan_50d_small_numpy": {"time": 0.008, "speedup": 39.6},
    "dbscan_50d_small_cupy": {"time": 0.011, "speedup": 29.0},
    "dbscan_50d_small_torch": {"time": 0.112, "speedup": 2.9},
    "dbscan_50d_medium_numpy": {"time": 0.099, "speedup": 1.9},
    "dbscan_50d_medium_cupy": {"time": 0.223, "speedup": 0.9},
    "dbscan_50d_medium_torch": {"time": 0.184, "speedup": 1.0},
    "dbscan_50d_large_numpy": {"time": 8.935, "speedup": 0.4, "external": 3.7},
    "dbscan_50d_large_cupy": {"time": 41.684, "speedup": 0.1, "external": 3.7},
    "dbscan_50d_large_torch": {"time": 41.808, "speedup": 0.1, "external": 3.7},
}
data.update(dbscan_50d)

# Update TSNE with medium scale
tsne_updates = {
    "tsne_medium_numpy": {"time": 4.02},
    "tsne_medium_cupy": {"time": 0.47},
    "tsne_medium_torch": {"time": 0.34},
}
data.update(tsne_updates)

# Save
with open("results/unsupervised_bench_2026-06-26.json", "w") as f:
    json.dump(data, f, indent=2, default=str)

print(f"Updated: {len(data)} entries")
