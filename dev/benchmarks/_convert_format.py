"""Convert solver benchmark results to match existing benchmark format."""
import json
from datetime import datetime

# Load remaining 25 cases (100K, 3-backend, per-solver)
with open("results/glm_solver_remaining_100K.json") as f:
    remaining = json.load(f)

# First 45 cases from conversation output (100K, 3-backend, best solver only)
# We'll construct the full solver data from the flat table
FIRST45_DATA = [
    # (family, penalty, numpy_best, numpy_spd, cupy_best, cupy_spd, torch_best, torch_spd)
    ("squared_error", "none", {"admm": 1.1}, {"exact": 3.3, "newton": 3.2, "irls": 2.0, "lbfgs": 2.2, "fista": 0.2, "fista_bb": 2.1, "admm": 2.2}, {"exact": 3.5, "newton": 3.1, "irls": 2.0, "lbfgs": 2.2, "fista": 0.2, "fista_bb": 2.1, "admm": 2.2}),
    ("squared_error", "l2", {"exact": 1.0}, {"newton": 3.2, "exact": 3.3, "irls": 2.1, "lbfgs": 2.3, "fista": 0.2, "fista_bb": 2.2, "admm": 2.3}, {"exact": 3.4, "newton": 3.3, "irls": 2.1, "lbfgs": 2.3, "fista": 0.2, "fista_bb": 2.2, "admm": 2.3}),
    ("squared_error", "l1", {"fista": 1.0, "fista_bb": 1.0, "admm": 1.0}, {"fista": 2.7, "fista_bb": 2.9, "admm": 1.9}, {"fista": 2.7, "fista_bb": 3.0, "admm": 1.9}),
    ("squared_error", "elasticnet", {"fista": 1.0, "fista_bb": 1.0, "admm": 1.0}, {"fista": 2.7, "fista_bb": 2.9, "admm": 1.9}, {"fista": 2.7, "fista_bb": 2.9, "admm": 1.9}),
    ("squared_error", "adaptive_l1", {"fista": 0.9, "fista_bb": 0.9, "admm": 0.9}, {"fista": 1.0, "fista_bb": 1.0, "admm": 1.0}, {"fista": 0.9, "fista_bb": 0.9, "admm": 1.0}),
    ("squared_error", "scad", {"fista": 1.0, "fista_bb": 1.0, "admm": 1.0}, {"fista": 15.5, "fista_bb": 15.8, "admm": 15.8}, {"fista": 15.6, "fista_bb": 15.8, "admm": 15.8}),
    ("squared_error", "mcp", {"fista": 1.0, "fista_bb": 1.0, "admm": 1.0}, {"fista": 18.8, "fista_bb": 19.3, "admm": 19.2}, {"fista": 18.8, "fista_bb": 19.3, "admm": 19.1}),
    ("squared_error", "group_lasso", {"fista": 0.5, "fista_bb": 0.5, "admm": 0.5}, {"fista": 0.5, "fista_bb": 0.5, "admm": 0.5}, {"fista": 0.5, "fista_bb": 0.5, "admm": 0.6}),
    ("squared_error", "group_scad", {"fista": 0.2, "fista_bb": 0.2, "admm": 1.0}, {"fista": 0.2, "fista_bb": 1.9, "admm": 1.6}, {"fista": 0.2, "fista_bb": 2.0, "admm": 1.7}),
    ("squared_error", "group_mcp", {"fista": 0.2, "fista_bb": 0.2, "admm": 1.0}, {"fista": 0.2, "fista_bb": 1.9, "admm": 1.7}, {"fista": 0.2, "fista_bb": 1.9, "admm": 1.7}),
    ("logistic", "none", {"newton": 16.6, "irls": 18.6, "lbfgs": 11.3, "fista": 3.0, "fista_bb": 3.0, "admm": 3.1}, {"irls": 18.4, "newton": 16.6, "lbfgs": 11.3, "fista": 3.0, "fista_bb": 3.0, "admm": 3.1}, {"irls": 19.3, "newton": 16.6, "lbfgs": 11.3, "fista": 3.0, "fista_bb": 3.0, "admm": 3.1}),
    ("logistic", "l2", {"newton": 13.7, "irls": 11.6, "lbfgs": 10.2, "fista": 4.0, "fista_bb": 4.0, "admm": 2.9}, {"newton": 13.0, "irls": 11.6, "lbfgs": 10.2, "fista": 4.0, "fista_bb": 4.0, "admm": 2.9}, {"newton": 13.8, "irls": 11.6, "lbfgs": 10.2, "fista": 4.0, "fista_bb": 4.0, "admm": 2.9}),
    ("logistic", "l1", {"fista": 3.7, "fista_bb": 3.3, "admm": 0.4}, {"fista": 3.7, "fista_bb": 3.3, "admm": 0.4}, {"fista": 3.7, "fista_bb": 3.3, "admm": 0.4}),
    ("logistic", "elasticnet", {"fista": 3.0, "fista_bb": 2.7, "admm": 2.8}, {"fista": 2.9, "fista_bb": 2.7, "admm": 2.8}, {"fista": 3.0, "fista_bb": 2.7, "admm": 2.8}),
    ("logistic", "adaptive_l1", {"fista": 3.3, "fista_bb": 3.3, "admm": 3.3}, {"fista": 3.3, "fista_bb": 3.4, "admm": 3.3}, {"fista": 3.3, "fista_bb": 3.4, "admm": 3.3}),
    ("logistic", "scad", {"fista": 6.4, "fista_bb": 6.4, "admm": 6.3}, {"fista": 6.1, "fista_bb": 6.4, "admm": 6.3}, {"fista": 6.4, "fista_bb": 6.0, "admm": 6.3}),
    ("logistic", "mcp", {"fista": 8.2, "fista_bb": 8.1, "admm": 8.0}, {"fista": 8.0, "fista_bb": 8.0, "admm": 8.0}, {"fista": 8.2, "fista_bb": 7.8, "admm": 8.0}),
    ("logistic", "group_lasso", {"fista": 3.1, "fista_bb": 3.1, "admm": 3.2}, {"fista": 3.1, "fista_bb": 3.1, "admm": 3.3}, {"fista": 3.1, "fista_bb": 3.3, "admm": 3.2}),
    ("logistic", "group_scad", {"fista": 4.8, "fista_bb": 5.1, "admm": 5.1}, {"fista": 4.8, "fista_bb": 5.1, "admm": 5.1}, {"fista": 5.1, "fista_bb": 4.8, "admm": 5.1}),
    ("logistic", "group_mcp", {"fista": 6.7, "fista_bb": 6.4, "admm": 6.5}, {"fista": 6.5, "fista_bb": 6.5, "admm": 6.5}, {"fista": 6.7, "fista_bb": 6.3, "admm": 6.5}),
    ("poisson", "none", {"newton": 12.7, "irls": 11.2, "lbfgs": 9.4, "fista": 4.3, "fista_bb": 3.7, "admm": 0.1}, {"newton": 12.6, "irls": 11.2, "lbfgs": 9.4, "fista": 4.3, "fista_bb": 3.7, "admm": 0.1}, {"newton": 13.1, "irls": 11.2, "lbfgs": 9.4, "fista": 4.3, "fista_bb": 3.7, "admm": 0.1}),
    ("poisson", "l2", {"newton": 13.1, "irls": 10.7, "lbfgs": 9.0, "fista": 4.1, "fista_bb": 4.0, "admm": 0.1}, {"newton": 12.7, "irls": 10.7, "lbfgs": 9.0, "fista": 4.1, "fista_bb": 4.0, "admm": 0.1}, {"newton": 13.2, "irls": 10.7, "lbfgs": 9.0, "fista": 4.1, "fista_bb": 4.0, "admm": 0.1}),
    ("poisson", "l1", {"fista": 4.2, "fista_bb": 2.9, "admm": 0.1}, {"fista": 4.1, "fista_bb": 2.9, "admm": 0.1}, {"fista": 4.3, "fista_bb": 2.9, "admm": 0.1}),
    ("poisson", "elasticnet", {"fista": 3.9, "fista_bb": 2.8, "admm": 0.1}, {"fista": 3.7, "fista_bb": 2.8, "admm": 0.1}, {"fista": 3.8, "fista_bb": 2.8, "admm": 0.1}),
    ("poisson", "adaptive_l1", {"fista": 4.0, "fista_bb": 4.0, "admm": 4.0}, {"fista": 4.0, "fista_bb": 4.0, "admm": 4.1}, {"fista": 4.1, "fista_bb": 4.0, "admm": 4.0}),
    ("poisson", "scad", {"fista": 8.8, "fista_bb": 8.8, "admm": 9.0}, {"fista": 8.8, "fista_bb": 8.8, "admm": 8.8}, {"fista": 8.8, "fista_bb": 8.8, "admm": 8.8}),
    ("poisson", "mcp", {"fista": 9.4, "fista_bb": 9.4, "admm": 9.3}, {"fista": 9.4, "fista_bb": 8.9, "admm": 9.3}, {"fista": 9.4, "fista_bb": 9.4, "admm": 8.8}),
    ("poisson", "group_lasso", {"fista": 3.5, "fista_bb": 3.7, "admm": 3.5}, {"fista": 3.8, "fista_bb": 3.5, "admm": 3.5}, {"fista": 3.8, "fista_bb": 3.5, "admm": 3.5}),
    ("poisson", "group_scad", {"fista": 6.1, "fista_bb": 6.1, "admm": 6.1}, {"fista": 6.1, "fista_bb": 6.2, "admm": 6.1}, {"fista": 6.2, "fista_bb": 6.1, "admm": 6.1}),
    ("poisson", "group_mcp", {"fista": 6.0, "fista_bb": 6.1, "admm": 6.1}, {"fista": 6.1, "fista_bb": 6.0, "admm": 6.1}, {"fista": 6.0, "fista_bb": 6.2, "admm": 6.1}),
    ("gamma", "none", {"newton": 74.2, "irls": 46.8, "lbfgs": 38.3, "fista": 5.5, "fista_bb": 8.8, "admm": 21.5}, {"newton": 77.1, "irls": 46.8, "lbfgs": 38.3, "fista": 5.5, "fista_bb": 8.8, "admm": 21.5}, {"newton": 82.8, "irls": 46.8, "lbfgs": 38.3, "fista": 5.5, "fista_bb": 8.8, "admm": 21.5}),
    ("gamma", "l2", {"newton": 73.2, "irls": 39.0, "lbfgs": 38.4, "fista": 5.3, "fista_bb": 8.5, "admm": 12.1}, {"newton": 73.7, "irls": 39.0, "lbfgs": 38.4, "fista": 5.3, "fista_bb": 8.5, "admm": 12.1}, {"newton": 77.1, "irls": 39.0, "lbfgs": 38.4, "fista": 5.3, "fista_bb": 8.5, "admm": 12.1}),
    ("gamma", "l1", {"fista": 5.2, "fista_bb": 7.0, "admm": 6.4}, {"fista": 5.2, "fista_bb": 7.5, "admm": 6.4}, {"fista": 5.2, "fista_bb": 7.7, "admm": 6.4}),
    ("gamma", "elasticnet", {"fista": 5.0, "fista_bb": 7.4, "admm": 1.1}, {"fista": 5.0, "fista_bb": 7.4, "admm": 1.1}, {"fista": 5.0, "fista_bb": 7.5, "admm": 1.1}),
    ("gamma", "adaptive_l1", {"fista": 5.5, "fista_bb": 5.7, "admm": 5.7}, {"fista": 5.5, "fista_bb": 5.7, "admm": 5.7}, {"fista": 5.6, "fista_bb": 5.7, "admm": 5.6}),
    ("gamma", "scad", {"fista": 9.1, "fista_bb": 9.2, "admm": 9.0}, {"fista": 9.3, "fista_bb": 9.2, "admm": 9.0}, {"fista": 9.1, "fista_bb": 9.2, "admm": 9.0}),
    ("gamma", "mcp", {"fista": 9.1, "fista_bb": 9.0, "admm": 9.1}, {"fista": 9.1, "fista_bb": 9.0, "admm": 9.1}, {"fista": 9.2, "fista_bb": 9.0, "admm": 9.1}),
    ("gamma", "group_lasso", {"fista": 5.1, "fista_bb": 5.1, "admm": 5.1}, {"fista": 5.1, "fista_bb": 5.2, "admm": 5.1}, {"fista": 5.1, "fista_bb": 5.1, "admm": 5.1}),
    ("gamma", "group_scad", {"fista": 6.4, "fista_bb": 6.6, "admm": 6.5}, {"fista": 6.6, "fista_bb": 6.4, "admm": 6.5}, {"fista": 6.6, "fista_bb": 6.6, "admm": 6.5}),
    ("gamma", "group_mcp", {"fista": 5.3, "fista_bb": 5.4, "admm": 5.4}, {"fista": 6.7, "fista_bb": 5.3, "admm": 5.4}, {"fista": 6.6, "fista_bb": 5.4, "admm": 5.4}),
    ("inverse_gaussian", "none", {"newton": 6.2, "irls": 0.6, "lbfgs": 8.3, "fista": 1.2, "fista_bb": 0.6, "admm": 0.4}, {"lbfgs": 9.5, "newton": 6.2, "irls": 0.6, "fista": 1.2, "fista_bb": 0.6, "admm": 0.4}, {"lbfgs": 9.5, "newton": 6.2, "irls": 0.6, "fista": 1.2, "fista_bb": 0.6, "admm": 0.4}),
    ("inverse_gaussian", "l2", {"newton": 2.5, "irls": 0.6, "lbfgs": 5.5, "fista": 3.3, "fista_bb": 0.6, "admm": 0.5}, {"lbfgs": 6.7, "newton": 2.5, "irls": 0.6, "fista": 3.3, "fista_bb": 0.6, "admm": 0.5}, {"lbfgs": 6.2, "newton": 2.5, "irls": 0.6, "fista": 3.3, "fista_bb": 0.6, "admm": 0.5}),
    ("inverse_gaussian", "l1", {"fista": 3.5, "fista_bb": 3.5, "admm": 2.0}, {"fista": 4.0, "fista_bb": 3.5, "admm": 2.0}, {"fista": 4.2, "fista_bb": 3.5, "admm": 2.0}),
    ("inverse_gaussian", "elasticnet", {"fista": 3.4, "fista_bb": 3.4, "admm": 1.9}, {"fista": 3.4, "fista_bb": 3.8, "admm": 1.9}, {"fista": 4.0, "fista_bb": 3.4, "admm": 1.9}),
    ("inverse_gaussian", "adaptive_l1", {"fista": 4.1, "fista_bb": 4.0, "admm": 3.9}, {"fista": 4.1, "fista_bb": 4.0, "admm": 4.8}, {"fista": 4.8, "fista_bb": 4.0, "admm": 3.9}),
]

# Build output in old format
output = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "environment": {
        "cupy_available": True,
        "torch_available": True,
        "gpu": "Tesla P100-SXM2-16GB",
        "scale": "100K x 50",
    },
    "performance": {
        "medium_100k": {}
    }
}

# Add first 45 cases
for fam, pen, numpy_spd, cupy_spd, torch_spd in FIRST45_DATA:
    key = f"{fam}_{pen}"
    entry = {}
    for backend, spd_dict in [("numpy", numpy_spd), ("cupy", cupy_spd), ("torch", torch_spd)]:
        best_solver = max(spd_dict, key=lambda k: spd_dict[k])
        entry[backend] = {
            "best_solver": best_solver,
            "best_speedup": spd_dict[best_solver],
            "solvers": spd_dict,
        }
    output["performance"]["medium_100k"][key] = entry

# Add remaining 25 cases from JSON
for r in remaining:
    fam = r["family"]
    pen = r["penalty"]
    key = f"{fam}_{pen}"
    entry = {}
    for bn in ["numpy", "cupy", "torch"]:
        b = r["backends"][bn]
        bst = b["best_solver"]
        spd = b["solvers"][bst]["speedup"] if bst else 0
        solvers = {k: v["speedup"] for k, v in b["solvers"].items()}
        entry[bn] = {
            "best_solver": bst,
            "best_speedup": spd,
            "solvers": solvers,
        }
    output["performance"]["medium_100k"][key] = entry

with open("results/glm_solver_benchmark_2026-06-23.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: results/glm_solver_benchmark_2026-06-23.json")
print(f"Total cases: {len(output['performance']['medium_100k'])}")
