"""Merge new modules benchmark results into a single frontend-friendly file.

Combines:
1. Panel/GAM/ANOVA performance (3 backends, 3 scales)
2. GAM aligned precision (statgpu vs pygam)
3. External framework comparison (statgpu vs linearmodels/pygam/scipy)
"""
import json
from datetime import datetime

# Load source files
with open("results/new_modules_bench_2026-06-24.json") as f:
    perf = json.load(f)

with open("results/gam_aligned_2026-06-24.json") as f:
    gam_aligned = json.load(f)

with open("results/compare_external_v2_2026-06-24.json") as f:
    external = json.load(f)

# Build merged output
output = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "environment": perf.get("environment", {}),
    "modules": {}
}

# ===========================================================================
# Panel Data
# ===========================================================================
panel = {"performance": {}, "external_comparison": {}}

# Performance: extract from perf["benchmarks"]
for scale in ["small", "medium", "large"]:
    for estimator in ["PooledOLS", "PooledOLS_hac", "PanelOLS_entity", "PanelOLS_two_way",
                       "RandomEffects", "BetweenOLS", "FirstDifferenceOLS", "FamaMacBeth"]:
        key_prefix = f"panel_{scale}_{estimator}"
        entry = {}
        for backend in ["numpy", "cupy", "torch"]:
            k = f"{key_prefix}_{backend}"
            if k in perf["benchmarks"]:
                entry[backend] = perf["benchmarks"][k]
        if entry:
            if scale not in panel["performance"]:
                panel["performance"][scale] = {}
            panel["performance"][scale][estimator] = entry

# External comparison
for k, v in external.items():
    if k.startswith("panel_"):
        panel["external_comparison"][k] = v

output["modules"]["panel"] = panel

# ===========================================================================
# GAM
# ===========================================================================
gam = {"performance": {}, "precision_aligned": {}, "external_comparison": {}}

# Performance
for scale in ["small", "medium", "large"]:
    entry = {}
    for backend in ["numpy", "cupy", "torch"]:
        k = f"gam_{scale}_{backend}"
        if k in perf["benchmarks"]:
            entry[backend] = perf["benchmarks"][k]
    if entry:
        gam["performance"][scale] = entry

# Aligned precision
for k, v in gam_aligned.items():
    gam["precision_aligned"][k] = v

# External comparison
for k, v in external.items():
    if k.startswith("gam_"):
        gam["external_comparison"][k] = v

output["modules"]["gam"] = gam

# ===========================================================================
# ANOVA
# ===========================================================================
anova = {"performance": {}, "external_comparison": {}}

# Performance
for scale in ["small", "medium", "large"]:
    for func in ["f_oneway", "f_twoway", "f_welch", "tukey_hsd", "bonferroni"]:
        entry = {}
        for backend in ["numpy", "cupy", "torch"]:
            k = f"anova_{scale}_{backend}_{func}"
            if k in perf["benchmarks"]:
                entry[backend] = perf["benchmarks"][k]
        if entry:
            if scale not in anova["performance"]:
                anova["performance"][scale] = {}
            anova["performance"][scale][func] = entry

# External comparison
for k, v in external.items():
    if k.startswith("anova_"):
        anova["external_comparison"][k] = v

output["modules"]["anova"] = anova

# ===========================================================================
# Summary tables (for quick frontend rendering)
# ===========================================================================
summary = {
    "panel_large": {
        "estimators": {},
        "note": "100K obs, 20 vars, vs linearmodels"
    },
    "gam_large": {
        "backends": {},
        "note": "100K obs, 10 features, vs pygam (aligned)"
    },
    "anova_large": {
        "functions": {},
        "note": "100K/group, 20 groups, vs scipy"
    }
}

# Panel summary
for est in ["PooledOLS", "PooledOLS_hac", "PanelOLS_entity", "PanelOLS_two_way", "RandomEffects"]:
    entry = {}
    for be in ["numpy", "cupy", "torch"]:
        k = f"panel_large_{est}_{be}"
        if k in perf["benchmarks"]:
            entry[be] = perf["benchmarks"][k].get("time", None)
    # External
    ext_key = f"panel_large_{est.replace('PanelOLS_', 'PanelOLS').replace('PooledOLS_hac', 'PooledOLS')}"
    # Find matching external key
    for ek, ev in external.items():
        if f"panel_large_{est}" in ek or f"panel_large_{est.replace('PanelOLS_', '')}" in ek:
            if "external_time" in ev:
                entry["external_time"] = ev["external_time"]
                entry["external_name"] = "linearmodels"
                if "speedup" in ev:
                    entry["best_speedup"] = ev["speedup"]
                break
    summary["panel_large"]["estimators"][est] = entry

# GAM summary (aligned)
for be in ["numpy", "cupy", "torch"]:
    k = f"gam_large_{be}"
    if k in perf["benchmarks"]:
        entry = dict(perf["benchmarks"][k])
        # Add aligned precision
        aligned_key = f"gam_fixed_large_{be}"
        if aligned_key in gam_aligned:
            entry["aligned_pred_rel_diff"] = gam_aligned[aligned_key].get("pred_rel_diff", None)
        # Add external comparison
        ext_key = f"gam_large_{be}"
        if ext_key in external:
            entry["external_time"] = external[ext_key].get("external_time", None)
            entry["external_name"] = "pygam"
        summary["gam_large"]["backends"][be] = entry

# ANOVA summary
for func in ["f_oneway", "f_twoway"]:
    entry = {}
    for be in ["numpy", "cupy", "torch"]:
        k = f"anova_large_{be}_{func}"
        if k in perf["benchmarks"]:
            entry[be] = perf["benchmarks"][k]
    # External
    if func == "f_oneway":
        for ek, ev in external.items():
            if "f_oneway" in ek and "large" in ek and "external_time" in ev:
                entry["external_time"] = ev["external_time"]
                entry["external_name"] = "scipy"
                break
    summary["anova_large"]["functions"][func] = entry

output["summary"] = summary

# Save
with open("results/new_modules_full_2026-06-24.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"Saved: results/new_modules_full_2026-06-24.json")
print(f"Modules: {list(output['modules'].keys())}")
print(f"Summary keys: {list(output['summary'].keys())}")
