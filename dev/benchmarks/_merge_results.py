"""Merge all 70 cases into final 3-backend table."""
import json

# First 45 cases from conversation output (100K scale)
FIRST45 = [
    ("squared_error", "none", "admm", 1.1, "exact", 3.3, "exact", 3.5),
    ("squared_error", "l2", "exact", 1.0, "newton", 3.2, "exact", 3.4),
    ("squared_error", "l1", "fista", 1.0, "fista_bb", 2.9, "fista_bb", 3.0),
    ("squared_error", "elasticnet", "admm", 1.0, "fista_bb", 2.9, "fista_bb", 2.9),
    ("squared_error", "adaptive_l1", "admm", 1.0, "fista", 1.0, "admm", 1.0),
    ("squared_error", "scad", "fista", 1.0, "admm", 15.5, "fista", 15.6),
    ("squared_error", "mcp", "fista", 1.0, "fista", 18.9, "admm", 19.1),
    ("squared_error", "group_lasso", "admm", 1.0, "admm", 0.5, "admm", 0.6),
    ("squared_error", "group_scad", "admm", 1.0, "fista_bb", 1.9, "fista_bb", 2.0),
    ("squared_error", "group_mcp", "admm", 1.0, "fista_bb", 1.9, "fista_bb", 1.9),
    ("logistic", "none", "lbfgs", 3.0, "irls", 18.4, "irls", 19.3),
    ("logistic", "l2", "lbfgs", 2.6, "newton", 13.0, "newton", 13.8),
    ("logistic", "l1", "fista", 1.0, "fista", 3.7, "fista", 3.7),
    ("logistic", "elasticnet", "fista", 1.0, "fista", 2.9, "fista", 3.0),
    ("logistic", "adaptive_l1", "admm", 1.0, "fista_bb", 3.4, "fista_bb", 3.4),
    ("logistic", "scad", "fista", 1.0, "fista", 6.1, "fista_bb", 6.0),
    ("logistic", "mcp", "fista", 1.0, "fista_bb", 8.0, "fista_bb", 7.8),
    ("logistic", "group_lasso", "admm", 1.0, "admm", 3.3, "fista_bb", 3.3),
    ("logistic", "group_scad", "fista_bb", 1.0, "admm", 5.1, "fista", 5.1),
    ("logistic", "group_mcp", "fista", 1.0, "fista_bb", 6.5, "fista_bb", 6.3),
    ("poisson", "none", "lbfgs", 2.2, "newton", 12.6, "newton", 13.1),
    ("poisson", "l2", "lbfgs", 2.1, "newton", 12.7, "newton", 13.2),
    ("poisson", "l1", "fista", 1.0, "fista", 4.1, "fista", 4.3),
    ("poisson", "elasticnet", "fista", 1.0, "fista", 3.7, "fista", 3.8),
    ("poisson", "adaptive_l1", "fista", 1.0, "admm", 4.1, "fista", 4.1),
    ("poisson", "scad", "admm", 1.0, "admm", 8.8, "fista_bb", 8.8),
    ("poisson", "mcp", "fista_bb", 1.0, "fista_bb", 8.9, "admm", 8.8),
    ("poisson", "group_lasso", "fista_bb", 1.0, "fista", 3.8, "fista", 3.8),
    ("poisson", "group_scad", "admm", 1.0, "fista_bb", 6.2, "fista", 6.2),
    ("poisson", "group_mcp", "fista", 1.0, "fista", 6.1, "fista_bb", 6.2),
    ("gamma", "none", "lbfgs", 9.8, "newton", 77.1, "newton", 82.8),
    ("gamma", "l2", "lbfgs", 10.4, "newton", 73.7, "newton", 77.1),
    ("gamma", "l1", "fista_bb", 1.2, "fista_bb", 7.5, "fista_bb", 7.7),
    ("gamma", "elasticnet", "fista_bb", 1.3, "fista_bb", 7.4, "fista_bb", 7.5),
    ("gamma", "adaptive_l1", "fista", 1.0, "admm", 5.7, "fista", 5.6),
    ("gamma", "scad", "admm", 1.0, "fista", 9.3, "fista", 9.1),
    ("gamma", "mcp", "fista_bb", 1.0, "admm", 9.1, "fista", 9.2),
    ("gamma", "group_lasso", "fista_bb", 1.0, "fista_bb", 5.2, "admm", 5.1),
    ("gamma", "group_scad", "fista", 1.0, "fista", 6.6, "fista", 6.6),
    ("gamma", "group_mcp", "admm", 1.0, "fista", 6.7, "fista", 6.6),
    ("inverse_gaussian", "none", "lbfgs", 1.7, "lbfgs", 9.5, "lbfgs", 9.5),
    ("inverse_gaussian", "l2", "lbfgs", 1.2, "lbfgs", 6.7, "lbfgs", 6.2),
    ("inverse_gaussian", "l1", "fista_bb", 0.9, "fista", 4.0, "fista", 4.2),
    ("inverse_gaussian", "elasticnet", "fista", 1.0, "fista_bb", 3.8, "fista", 4.0),
    ("inverse_gaussian", "adaptive_l1", "fista", 1.0, "admm", 4.8, "fista", 4.8),
]

# Remaining 25 cases from 100K run
with open("results/glm_solver_remaining_100K.json") as f:
    remaining = json.load(f)

# Build markdown
lines = [
    "# GLM Solver Benchmark: 7 Families x 10 Penalties x 3 Backends",
    "",
    "Date: 2026-06-23",
    "Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)",
    "Scale: N=100,000 x P=50",
    "Reference: numpy + fista solver (median of 3 runs)",
    "",
    "## Complete 3-Backend Table",
    "",
    "| Family | Penalty | numpy best | spd | cupy best | spd | torch best | spd |",
    "|--------|---------|------------|-----|-----------|-----|------------|-----|",
]

for row in FIRST45:
    fam, pen, nb, ns, cb, cs, tb, ts = row
    lines.append(f"| {fam} | {pen} | {nb} | {ns:.1f}x | {cb} | {cs:.1f}x | {tb} | {ts:.1f}x |")

for r in remaining:
    fam = r["family"]
    pen = r["penalty"]
    parts = []
    for bn in ["numpy", "cupy", "torch"]:
        b = r["backends"][bn]
        bst = b["best_solver"]
        spd = b["solvers"][bst]["speedup"] if bst else 0
        parts.extend([bst or "-", f"{spd:.1f}x"])
    lines.append(f"| {fam} | {pen} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} | {parts[5]} |")

# Top 10
lines.extend([
    "",
    "## Top 10 GPU Speedups (torch)",
    "",
    "| Rank | Family | Penalty | Solver | Speedup |",
    "|------|--------|---------|--------|---------|",
])

all_torch = []
for row in FIRST45:
    all_torch.append((row[0], row[1], row[6], row[7]))
for r in remaining:
    b = r["backends"]["torch"]
    bst = b["best_solver"]
    spd = b["solvers"][bst]["speedup"] if bst else 0
    all_torch.append((r["family"], r["penalty"], bst, spd))

all_torch.sort(key=lambda x: -x[3])
for i, (fam, pen, solver, spd) in enumerate(all_torch[:10]):
    lines.append(f"| {i+1} | {fam} | {pen} | {solver} | {spd:.1f}x |")

# Solver guide
lines.extend([
    "",
    "## Solver Selection Guide",
    "",
    "| Penalty Type | Best CPU Solver | Best GPU Solver | Typical GPU Speedup |",
    "|-------------|----------------|-----------------|--------------------|",
    "| Smooth (none/l2) | lbfgs/newton | newton/irls | 10-102x |",
    "| L1/ElasticNet | fista/fista_bb | fista/fista_bb | 3-7x |",
    "| Adaptive L1 | fista/fista_bb | fista_bb | 3-5x |",
    "| SCAD/MCP | fista | fista | 6-19x |",
    "| Group Lasso | admm | admm/fista | 3-5x |",
    "| Group SCAD/MCP | fista | fista | 5-7x |",
])

md = "\n".join(lines)
with open("results/glm_solver_benchmark_2026-06-23.md", "w") as f:
    f.write(md)
print(f"Saved: results/glm_solver_benchmark_2026-06-23.md")
print(f"Total: {len(FIRST45) + len(remaining)} cases")
