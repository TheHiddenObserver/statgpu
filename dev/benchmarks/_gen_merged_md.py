"""Generate merged markdown report."""
import json

with open('results/new_modules_full_2026-06-24.json') as f:
    data = json.load(f)

lines = [
    '# New Modules Benchmark (Complete)',
    '',
    'Date: 2026-06-24',
    'Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)',
    '',
    '## Panel Data',
    '',
    '### Performance (Large: 100K obs, 20 vars)',
    '',
    '| Estimator | numpy (s) | cupy (s) | torch (s) | cupy spd | torch spd |',
    '|-----------|-----------|----------|-----------|----------|-----------|',
]

panel_perf = data['modules']['panel']['performance'].get('large', {})
for est in ['PooledOLS', 'PooledOLS_hac', 'PanelOLS_entity', 'PanelOLS_two_way', 'RandomEffects', 'BetweenOLS', 'FirstDifferenceOLS', 'FamaMacBeth']:
    if est in panel_perf:
        entry = panel_perf[est]
        nt = entry.get('numpy', {}).get('time', 0)
        ct = entry.get('cupy', {}).get('time', 0)
        tt = entry.get('torch', {}).get('time', 0)
        cs = nt/ct if ct > 0 else 0
        ts = nt/tt if tt > 0 else 0
        lines.append(f'| {est} | {nt:.4f} | {ct:.4f} | {tt:.4f} | {cs:.1f}x | {ts:.1f}x |')

lines.extend([
    '',
    '### External Comparison (vs linearmodels)',
    '',
    '| Estimator | linearmodels (s) | statgpu torch (s) | Speedup | coef rel diff |',
    '|-----------|-----------------|-------------------|---------|---------------|',
])

for k, v in data['modules']['panel']['external_comparison'].items():
    if 'external_time' in v and 'statgpu_time' in v:
        ext_t = v['external_time']
        sg_t = v['statgpu_time']
        spd = v.get('speedup', 0)
        rel = v.get('coef_rel_diff', 'N/A')
        if isinstance(rel, float):
            rel = f'{rel:.2e}'
        lines.append(f'| {k} | {ext_t:.4f} | {sg_t:.4f} | {spd:.1f}x | {rel} |')

lines.extend([
    '',
    '## GAM',
    '',
    '### Performance (Large: 100K obs, 10 features)',
    '',
    '| Backend | statgpu (s) | pygam (s) | Speedup |',
    '|---------|------------|----------|---------|',
])

gam_perf = data['modules']['gam']['performance'].get('large', {})
gam_ext = data['modules']['gam']['external_comparison']
for be in ['numpy', 'cupy', 'torch']:
    if be in gam_perf:
        t = gam_perf[be].get('time', 0)
        ext_t = None
        for ek, ev in gam_ext.items():
            if f'gam_large_{be}' in ek and 'external_time' in ev:
                ext_t = ev['external_time']
                break
        if ext_t:
            spd = ext_t/t if t > 0 else 0
            lines.append(f'| {be} | {t:.4f} | {ext_t:.4f} | {spd:.1f}x |')
        else:
            lines.append(f'| {be} | {t:.4f} | N/A | - |')

lines.extend([
    '',
    '### Precision (Aligned: uniform knots, gamma=1.4, fixed lam=1.0)',
    '',
    '| Backend | pred rel diff |',
    '|---------|--------------|',
])

for k, v in data['modules']['gam']['precision_aligned'].items():
    if 'fixed' in k and 'large' in k:
        be = k.split('_')[-1]
        rel = v.get('pred_rel_diff', 'N/A')
        if isinstance(rel, float):
            rel = f'{rel:.2e}'
        lines.append(f'| {be} | {rel} |')

lines.extend([
    '',
    '## ANOVA',
    '',
    '### Performance (Large: 100K/group, 20 groups)',
    '',
    '| Function | numpy (ms) | cupy (ms) | torch (ms) | cupy spd | torch spd |',
    '|----------|-----------|----------|-----------|----------|-----------|',
])

anova_perf = data['modules']['anova']['performance'].get('large', {})
for func in ['f_oneway', 'f_twoway', 'f_welch', 'tukey_hsd', 'bonferroni']:
    if func in anova_perf:
        entry = anova_perf[func]
        nt = entry.get('numpy', {}).get('time', 0) * 1000
        ct = entry.get('cupy', {}).get('time', 0) * 1000
        tt = entry.get('torch', {}).get('time', 0) * 1000
        cs = nt/ct if ct > 0 else 0
        ts = nt/tt if tt > 0 else 0
        lines.append(f'| {func} | {nt:.2f} | {ct:.2f} | {tt:.2f} | {cs:.1f}x | {ts:.1f}x |')

lines.extend([
    '',
    '### External Comparison (vs scipy, f_oneway)',
    '',
    '| Backend | scipy (ms) | statgpu (ms) | Speedup | F rel diff |',
    '|---------|-----------|-------------|---------|------------|',
])

for k, v in data['modules']['anova']['external_comparison'].items():
    if 'f_oneway' in k and 'large' in k and 'external_time' in v:
        be = k.split('_')[-1]
        st = v.get('statgpu_time', 0) * 1000
        et = v.get('external_time', 0) * 1000
        spd = et/st if st > 0 else 0
        rel = v.get('f_rel_diff', 'N/A')
        if isinstance(rel, float):
            rel = f'{rel:.2e}'
        lines.append(f'| {be} | {et:.2f} | {st:.2f} | {spd:.1f}x | {rel} |')

md = '\n'.join(lines)
with open('results/new_modules_full_2026-06-24.md', 'w') as f:
    f.write(md)
print('Saved: results/new_modules_full_2026-06-24.md')
