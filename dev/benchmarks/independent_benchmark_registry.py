"""Source-only inventory and reviewed, bounded independent benchmark protocols.

The catalog records a historical audit, NOT a claim that those sources were run.
No numerical modules, benchmark modules, subprocesses or credential files are
loaded while planning. Admission is explicit code, never inferred from a CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).with_name('independent_benchmark_catalog.json')
TIERS = ('local-minimal', 'local-full', 'remote-full')
BACKENDS = ('numpy', 'cupy', 'torch')
COMPILE_CASES = ('glm_irls', 'lasso', 'elasticnet', 'scad', 'mcp', 'group_scad', 'group_mcp')
HYBRID_CASES = ('kde_bootstrap', 'bandwidth_nrd', 'bandwidth_nrd0', 'bandwidth_ucv',
                'bandwidth_bcv', 'bandwidth_sj', 'bandwidth_sj-ste', 'bandwidth_sj-dpi',
                'bandwidth_cv', 'knockoff_fixed', 'knockoff_model')
PANEL_C_CASES = ('pooled_nonrobust', 'pooled_hc3', 'pooled_cluster_two_way', 'pooled_dk_qs',
                 'panel_entity_nonrobust', 'panel_entity_hc3', 'panel_entity_dk',
                 'random_effects_nonrobust', 'random_effects_hc3')


def _adapter(module, args, contract, *, backends=BACKENDS, seconds=900, note='', correctness='runner-validated'):
    return {'module': 'dev.benchmarks.' + module, 'arguments': args.split(),
            'contract': contract, 'backends': list(backends),
            'budget': {'timeout_seconds': seconds, 'cpu_threads': 1, 'concurrent_processes': 1,
                       'minimum_gpu_free_bytes': 1024**3, 'minimum_host_available_bytes': 1024**3,
                       'memory_limit_note': 'Admission reserve, not a peak-memory or OOM guarantee; live cgroup/disk guards apply.'},
            'correctness_scope': correctness, 'limitations': note,
            'prerequisites': ['clean exact requested checkout', 'fresh external output directory',
                              'NumPy, SciPy, threadpoolctl and requested CUDA packages already installed',
                              'physical CUDA for every requested GPU backend; no fallback or installation']}


# Fixed pilot menus intentionally avoid arbitrary user-supplied commands/flags.
ADAPTERS = {
    'public-native-kde-bootstrap': _adapter('benchmark_public_native_kde_bootstrap',
        '--preset small --n 128 --queries 16 --resamples 30 --dimension 1 --weights unequal --kernel gaussian --bandwidth 0.4 --warmup 1 --repeats 3 --output {output}',
        'public-kde', note='One bounded numeric-bandwidth Gaussian pilot, not the complete kernel/selector/weight/dimension matrix.'),
    'native-kde-bootstrap': _adapter('benchmark_native_kde_bootstrap',
        '--preset small --n 128 --queries 16 --resamples 30 --weights unequal --warmup 1 --repeats 3 --output {output}',
        'controlled-kde', backends=('numpy', 'cupy'), note='Controlled indices exclude RNG; developer experiment, never public API timing.'),
    'native-kde-rng': _adapter('benchmark_native_kde_rng',
        '--preset small --n 128 --queries 16 --resamples 30 --weights unequal --warmup 1 --repeats 3 --output {output}',
        'rng-kde', backends=('numpy', 'cupy'), note='Historical experimental candidate versus live public baseline, not migration speedup.'),
    'hybrid-phases': _adapter('benchmark_hybrid_phases',
        '--preset tiny --backends numpy cupy torch --warmup 1 --repeats 3 --output {output}',
        'hybrid', note='Eleven phase protocols at tiny scale; cross-backend RNG is not identical.'),
    'cox-stability-review': _adapter('benchmark_cox_stability_review',
        '--n 512 --p 4 --warmups 1 --repeats 3 --output {output}', 'cox-stability',
        correctness='finite/converged ordinary Cox and extreme-predictor invariants; not external equivalence'),
    'exact-strata-count-scaling': _adapter('benchmark_exact_strata_count_scaling',
        '--strata-counts 3 32 --rows-per-stratum 8 --features 4 --repeats 3 --devices cpu cuda torch --output {output}',
        'exact-strata', correctness='Exact objective agreement against NumPy at registered tolerance'),
    'exact-ties-scaling': _adapter('benchmark_exact_ties_scaling',
        '--sizes 120 240 --features 4 --repeats 3 --largest-repeats 3 --devices cpu cuda torch --output {output}',
        'exact-ties', note='R alignment is a separate unexecuted prerequisite lane; this pilot does not claim external-R acceptance.',
        correctness='finite/converged Exact fits; no external reference requested'),
    'panel-stage-b-identity-overhead': _adapter('benchmark_panel_stage_b_identity_overhead',
        '--scales 1000x2 --backends cupy,torch --repeats 3 --expected-sha {expected_sha} --out {output}',
        'panel-b', backends=('cupy', 'torch'), correctness='unavailable: timing-only digest overhead, no numerical oracle'),
    'panel-stage-c-covariance': _adapter('benchmark_panel_stage_c_covariance',
        '--scales 1000x2 --high-t-scale 1200x2x60 --two-way-unbalanced-scale 2000x2x20 --backends cupy,torch --repeats 3 --expected-sha {expected_sha} --out {output}',
        'panel-c', backends=('cupy', 'torch'), correctness='unavailable: timing and backend-routing assertions, not covariance accuracy'),
    'fama-macbeth-scaling-gpu': _adapter('benchmark_fama_macbeth_scaling_gpu',
        '--backends cupy,torch --warmup 1 --repeats 3 --expected-sha {expected_sha} --validation-tier {validation_tier} --out {output}',
        'fama-macbeth', seconds=1800, note='Fixed micro/medium/large fixtures through 524288x16; not an unbounded scale sweep.',
        correctness='NumPy snapshot and inference-descriptor agreement asserted by runner'),
    'torch-compile-maintenance': _adapter('benchmark_torch_compile_maintenance',
        '--repeats 3 --output {output}', 'compile-maintenance', backends=('torch',), seconds=1800,
        note='Runner raw tier is historical metadata; actual orchestrator tier is authoritative in qualified sidecar.'),
    'torch-compile-scale': _adapter('benchmark_torch_compile_scale',
        '--preset quick --scales 1024x64 --repeats 3 --output {output}', 'compile-scale', backends=('torch',), seconds=1800,
        note='Seven cases at one pilot shape, cold and warm isolated; no universal crossover claim.'),
    'gpu-memory-cleanup': _adapter('benchmark_gpu_memory_cleanup',
        '--n_samples 1024 --n_features 8 --models linear,ridge,lasso,logistic,cox --warmup_runs 1 --repeats 2',
        'memory-cleanup', backends=('cupy',), correctness='unavailable: pool lifecycle snapshots, not numerical correctness',
        note='Capture wrapper records original _fit_one return dictionaries, including warmups. Pool snapshots are NOT peak VRAM or OOM capacity.'),
}

# Each other audited parameterized runner has an explicit admission decision.
BLOCK_REASONS = {
 'all-methods-large-scale': 'Cross-family large defaults and optional R references need per-case scale and completeness contracts.',
 'cox-cluster': 'Covariance and R-reference cells need exact expected-case and external-package contracts.',
 'cox-fit-timing': 'Stdout-only and PHReg includes inference while statgpu disables it: unequal timing estimands.',
 'cv-models': 'Reused by six-family-cv campaign; canonical/external protocol must not be counted as disjoint model families.',
 'distributions': 'Contains SSH/host-key auto-accept branch and 100M defaults; no remote branch admitted.',
 'external-frameworks': 'Optional Python/R oracle availability and objective alignment need explicit expected-cell validation.',
 'hc2-hc3-hac-gate': 'Covariance accuracy gate requires per-case external oracle and inference-target qualification.',
 'inference-backends': 'Hardcoded checkout output path and unparameterized inference sizes need output isolation and a bounded exact-case adapter.',
 'kde-vs-scipy': 'External SciPy bandwidth/weights/objective and default scale contract not yet admitted.',
 'kernel-regression-vs-statsmodels': 'External statsmodels bandwidth/regression-type comparability and scales not yet admitted.',
 'knockoff-fdr-calibration': 'Monte Carlo FDR replicate budget, stochastic uncertainty and calibration oracle require separate admission.',
 'lasso-cpu-gpu-tol': 'Stdout-only tolerance/convergence sweep has no qualified structured-output contract.',
 'lasso-inference-gpu-vs-cpu': 'Stdout-only inference comparison needs active-set and inference-method alignment.',
 'lassocv-impls': 'Implementation/CV-path overlap requires folds/grid/final-refit expected identity and bounded scale review.',
 'multitarget-sklearn-r': 'Optional sklearn/R target matrix requires exact external versions/objective mapping and expected cells.',
 'new-modules': 'Stdout/fixed artifact behavior and mixed-family defaults not admitted as safe bounded protocol.',
 'nonparametric-vs-r': 'R package/toolchain availability and bandwidth/objective alignment are unverified prerequisites.',
 'penalized-glm': 'Contains remote launch plumbing; local 70-combination objective/penalty oracle and scale review required.',
 'resampling-strategy-and-external': 'Hardcoded checkout output path and fixed resampling/external workload need output isolation and bounded adapter.',
 'survival-completion': 'Full-scale inference/formula/survival matrix needs per-case budgets and exact completeness validator.',
 'torch-vs-cupy-comprehensive': 'Broad overlapping matrix needs exact case identities and explicit no-fallback qualification.',
 'unsupervised': 'Base model suite overlaps external/extension matrices; default scales and expected cells not admitted.',
 'unsupervised-dbscan-cython': 'Compiled extension availability and exact implementation parity are separate prerequisite gates.',
 'unsupervised-external-matrix': 'Optional external/R cells and overlap with base unsupervised cases require exact-case reconciliation.',
 'unsupervised-phase2': 'Phase2 overlaps base/external matrices and requires exact solver/data/backend identity.',
 'unsupervised-phase3': 'Extension algorithms need individual bounded resource and accuracy contracts.',
 'unsupervised-phase3b': 'Phase3b overlaps base/phase2 and requires exact covariance/linkage/solver identity.',
 'unsupervised-phase3c': 'Extension algorithms and optional baselines need exact-case completeness and scale admission.',
}


# Additional local structured protocols are conditional on their existing
# Python/R/compiled dependencies. Missing dependencies are reported at runtime;
# nothing is installed and no optional SSH branch is reachable from these argv.
ADAPTERS.update({
 'kde-vs-scipy': _adapter('benchmark_kde_vs_scipy', '--n-samples 256 --n-eval 64 --dim 1 --bandwidth scott --weighted --repeats 3 --warmup 1 --json-out {output}', 'kde-scipy', backends=('numpy', 'cupy'), correctness='unavailable: recorded SciPy precision metrics, acceptance thresholds require review'),
 'kernel-regression-vs-statsmodels': _adapter('benchmark_kernel_regression_vs_statsmodels', '--n-samples 256 --n-eval 64 --dim 1 --bandwidth-abs 0.45 --kernel gaussian --regression nw --kernel-metric diagonal --repeats 3 --warmup 1 --json-out {output}', 'kernel-statsmodels', backends=('numpy', 'cupy'), correctness='unavailable: recorded statsmodels precision metrics, acceptance thresholds require review'),
 'hc2-hc3-hac-gate': _adapter('benchmark_hc2_hc3_hac_gate', '--model linear --n 512 --p 8 --cov-types hc2 hc3 hac --repeats 3 --json-out {output}', 'covariance-gate', correctness='unavailable: explicit external precision metrics retained; numerical thresholds are not invented'),
 'unsupervised': _adapter('benchmark_unsupervised', '--n 256 --p 16 --k 4 --pca-components 4 --pca-solver full --devices cpu,cuda,torch --methods pca,kmeans --warmup-runs 1 --repeats 3 --json-out {output}', 'unsupervised-base', correctness='unavailable: diagnostic clustering/PCA metrics, no universal accuracy threshold'),
 'unsupervised-dbscan-cython': _adapter('benchmark_unsupervised_dbscan_cython', '--devices cuda,torch --warmup-runs 1 --repeats 3 --json-out {output}', 'dbscan-cython', correctness='compiled/fallback/sklearn/GPU label partition and noise-mask parity', note='Five fixed fixtures, maximum 5000x8; compiled extension required, never built automatically.'),
 'unsupervised-external-matrix': _adapter('benchmark_unsupervised_external_matrix', '--n 256 --p 16 --k-data 4 --components 2,4 --ks 2,4 --devices cpu,cuda,torch --pca-solver full --methods pca,kmeans --warmup-runs 1 --repeats 3 --r-timeout 60 --json-out {output}', 'unsupervised-external', correctness='unavailable: external comparison diagnostics, no invented clustering equivalence'),
 'unsupervised-phase2': _adapter('benchmark_unsupervised_phase2', '--devices cpu,cuda,torch --warmup-runs 1 --repeats 3 --json-out {output}', 'unsupervised-phase2', correctness='unavailable: diagnostic metrics only', note='Fixed bounded fixtures DBSCAN600x8/GMM1200x12/NMF800x64/embedding400x16. R cluster/umap/openTSNE missing remains partial.'),
 'unsupervised-phase3': _adapter('benchmark_unsupervised_phase3', '--n 120 --p 12 --k 3 --components 3 --neighbors 8 --batch-size 32 --mb-iter 10 --manifold-epochs 15 --tsne-iter 250 --perplexity 15 --warmup 1 --repeats 3 --devices cpu,cuda,torch --json-out {output}', 'unsupervised-phase3', correctness='unavailable: extension model diagnostics, no accuracy certification'),
 'unsupervised-phase3b': _adapter('benchmark_unsupervised_phase3b', '--n 180 --p 4 --k 3 --gmm-iter 80 --warmup 1 --repeats 3 --devices cpu,cuda,torch --json-out {output}', 'unsupervised-phase3b', correctness='unavailable: covariance/linkage diagnostics, no accuracy certification'),
 'unsupervised-phase3c': _adapter('benchmark_unsupervised_phase3c', '--n 256 --p 12 --components 3 --batch-size 64 --max-iter 20 --warmup 1 --repeats 3 --devices cpu,cuda,torch --json-out {output}', 'unsupervised-phase3c', correctness='unavailable: incremental/batch diagnostics, no accuracy certification'),

 'nonparametric-vs-r': _adapter('benchmark_nonparametric_vs_r', '--n-samples 256 --n-eval 64 --n-ci-eval 32 --statgpu-backend cupy --repeats 3 --warmup 1 --ci-resamples 30 --ci-repeats 3 --ci-warmup 1 --json-out {output}', 'nonparametric-r', backends=('numpy', 'cupy'), correctness='unavailable: R/SciPy comparison metrics; method-specific precision thresholds need review'),
 'lassocv-impls': _adapter('benchmark_lassocv_impls', '--seeds 42 --n_samples 128 --n_features 16 --n_signal 4 --n_test_samples 64 --cv 3 --n_alphas 3 --max_iter 1000 --n_jobs 1 --shared_alpha_grid --shared_cv_splits --warmup 1 --repeats 3 --output {output}', 'lassocv', backends=('numpy', 'cupy'), correctness='unavailable: shared folds/grid diagnostic; selection/accuracy thresholds require review'),
})
# Source inspection found swallowed partial repeats and a false Torch label.
# A CuPy-only selection avoids the label, but cannot repair hidden repeat errors.
BLOCK_REASONS['new-modules'] = 'Unsupported output contract: Panel/GAM mislabel Torch as CUDA, lack GPU synchronization and retain partial successful repeats after later errors; numerical/reporting producer repair required.'

for _identifier in ADAPTERS:
    BLOCK_REASONS.pop(_identifier, None)
for _identifier in ('kernel-regression-vs-statsmodels', 'hc2-hc3-hac-gate'):
    ADAPTERS[_identifier]['required_python'] = ['statsmodels']
for _identifier in ('unsupervised', 'unsupervised-dbscan-cython', 'unsupervised-external-matrix', 'unsupervised-phase2', 'unsupervised-phase3', 'unsupervised-phase3b', 'unsupervised-phase3c', 'lassocv-impls'):
    ADAPTERS[_identifier]['required_python'] = ['sklearn']
for _identifier in ('nonparametric-vs-r', 'unsupervised-external-matrix', 'unsupervised-phase2', 'unsupervised-phase3', 'unsupervised-phase3b'):
    ADAPTERS[_identifier]['required_executables'] = ['Rscript']


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def number(token):
        result = float(token)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON number')
        return result
    def constant(token):
        raise ValueError('Nonfinite JSON constant: ' + token)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=constant)


def lane_id(path):
    return Path(path).stem.removeprefix('benchmark_').replace('_', '-')


def generate_catalog(audit_path, command_plan_path):
    """Rebuild deterministically from the two source-only audit artifacts, no imports."""
    audit, commands = _read(audit_path), _read(command_plan_path)
    inventory = audit['inventory']
    paths = [row['path'] for row in inventory]
    if len(paths) != 202 or len(set(paths)) != 202:
        raise ValueError('Expected the complete 202-file audit baseline')
    timing = [row for row in inventory if row['category'] == 'independent_parameterized_timing_runner']
    ids = {lane_id(row['path']) for row in timing}
    command_paths = {row['path'] for row in commands['independent_timing_runners']}
    if len(timing) != 41 or ids != set(ADAPTERS) | set(BLOCK_REASONS) or command_paths != {r['path'] for r in timing}:
        raise ValueError('Every one of the 41 independent runners needs exactly one reviewed disposition')
    if audit['target']['head_sha'] != commands['target']['head_sha']:
        raise ValueError('Audit inputs refer to different source commits')
    entries = []
    for row in inventory:
        category = row['category']
        disposition = ('registered-campaign' if category == 'campaign_timing_runner' else
                       'support-only' if category in ('helper', 'artifact_processor') else
                       'wrapper-only-no-double-count' if category == 'orchestrator' else
                       'blocked-unsafe' if category.startswith('legacy') or category == 'unsafe_remote_launcher' else
                       'separate-correctness-gate' if category in ('correctness_gate', 'correctness_collection_runner') else
                       'diagnostic-review-required')
        if category == 'independent_parameterized_timing_runner':
            identifier = lane_id(row['path'])
            if identifier in ADAPTERS:
                disposition = 'conditional-external-dependencies' if ADAPTERS[identifier].get('required_python') or ADAPTERS[identifier].get('required_executables') or identifier == 'unsupervised-dbscan-cython' else 'ready-bounded-adapter'
            else:
                disposition = 'deduplicated' if identifier == 'cv-models' else ('unsupported-contract' if identifier in ('cox-fit-timing', 'inference-backends', 'resampling-strategy-and-external', 'lasso-cpu-gpu-tol', 'lasso-inference-gpu-vs-cpu', 'new-modules') else 'blocked-admission')
        entries.append({k: row[k] for k in ('path', 'sha256', 'category', 'purpose', 'risks',
                       'wrapper_children', 'wrapped_by', 'case_lineage_or_reuse') if k in row} | {'disposition': disposition})
    return {'schema_version': 'statgpu-independent-catalog-v1',
            'audit_baseline_sha': audit['target']['head_sha'],
            'audit_inputs_sha256': {'independent-runners.json': _hash(audit_path), 'independent-command-plan.json': _hash(command_plan_path)},
            'provenance_note': 'Historical source-only audit. Hash drift is reported, never silently called execution evidence or a permanent current-source pin.',
            'regeneration': 'python -m dev.benchmarks.independent_benchmark_registry --regenerate-catalog AUDIT_JSON COMMAND_PLAN_JSON --output NEW_CATALOG_JSON',
            'counts': dict(sorted(Counter(row['category'] for row in inventory).items())),
            'deduplication': audit['deduplication'], 'files': entries}


def load_catalog(path=CATALOG):
    catalog = _read(path)
    files = catalog.get('files', [])
    if catalog.get('schema_version') != 'statgpu-independent-catalog-v1' or len(files) != 202 or len({r['path'] for r in files}) != 202:
        raise ValueError('Invalid complete audit catalog')
    for row in files:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or not row['path'].startswith('dev/benchmarks/'):
            raise ValueError('Invalid audited path')
        if len(row.get('sha256', '')) != 64:
            raise ValueError('Missing audited source hash')
    return catalog


def registry():
    result = {}
    for row in load_catalog()['files']:
        if row['category'] != 'independent_parameterized_timing_runner':
            continue
        identifier = lane_id(row['path'])
        result[identifier] = {'id': identifier, 'script': row['path'],
            'audit_sha256': row['sha256'], 'purpose': row['purpose'],
            'disposition': row['disposition'], 'wrapper_children': row.get('wrapper_children', []),
            'adapter': ADAPTERS.get(identifier), 'blocked_reason': BLOCK_REASONS.get(identifier),
            'execution_status': 'not_run'}
    if len(result) != 41 or set(result) != set(ADAPTERS) | set(BLOCK_REASONS):
        raise ValueError('Independent registry must account for all 41 audited runners')
    return result


def verify_catalog_sources(root=ROOT):
    changed, missing = [], []
    for row in load_catalog()['files']:
        source = Path(root) / row['path']
        if not source.is_file():
            missing.append(row['path'])
        elif _hash(source) != row['sha256']:
            changed.append(row['path'])
    return {'audit_baseline_sha': load_catalog()['audit_baseline_sha'],
            'changed_since_audit': changed, 'missing_since_audit': missing,
            'new_since_audit': sorted(p.relative_to(root).as_posix() for p in (Path(root) / 'dev/benchmarks').rglob('*.py') if '__pycache__' not in p.parts and p.relative_to(root).as_posix() not in {r['path'] for r in load_catalog()['files']}),
            'note': 'Not an execution source gate; current sources are fingerprinted before and after every run.'}


# Explicit parameter domains. These are resource guardrails, not empirical OOM
# bounds. New axes require code review, never --extra-args passthrough.
CONFIG_DEFAULTS = {
 'public-native-kde-bootstrap': dict(n=128, queries=16, resamples=30, dimension=1, weights='unequal', kernel='gaussian', bandwidth=0.4, seed=20261020, warmup=2, repeats=7),
 'native-kde-bootstrap': dict(n=128, queries=16, resamples=30, weights='unequal', seed=20261020, warmup=2, repeats=7),
 'native-kde-rng': dict(n=128, queries=16, resamples=30, weights='unequal', seed=20261020, warmup=2, repeats=7),
 'hybrid-phases': dict(n=32, p=2, queries=7, resamples=5, seed=20261020, cases=list(HYBRID_CASES), warmup=1, repeats=3),
 'cox-stability-review': dict(n=512, p=4, warmups=1, repeats=3),
 'exact-strata-count-scaling': dict(strata_counts=[3, 32], rows_per_stratum=8, features=4, seed=20260727, repeats=3),
 'exact-ties-scaling': dict(sizes=[120, 240], features=4, seed=88031, repeats=3, largest_repeats=3, scaling_scenario='right_censored', strata_count=3),
 'panel-stage-b-identity-overhead': dict(scales=['1000x2'], repeats=3),
 'panel-stage-c-covariance': dict(scales=['1000x2'], high_t_scale='1200x2x60', two_way_unbalanced_scale='2000x2x20', repeats=3),
 'fama-macbeth-scaling-gpu': dict(warmup=1, repeats=3),
 'torch-compile-maintenance': dict(repeats=3),
 'torch-compile-scale': dict(preset='quick', scales=['1024x64'], cases=list(COMPILE_CASES), repeats=3),
 'gpu-memory-cleanup': dict(n_samples=1024, n_features=8, seed=42, models=['linear', 'ridge', 'lasso', 'logistic', 'cox'], warmup_runs=1, repeats=2),
}
CHOICES = {
 'weights': ('none', 'unequal', 'sparse'),
 'kernel': ('gaussian', 'rectangular', 'triangular', 'epanechnikov', 'biweight', 'triweight', 'cosine', 'optcosine'),
 'bandwidth': ('scott', 'silverman', 'nrd', 'nrd0', 'ucv', 'bcv', 'sj', 'sj-ste', 'sj-dpi'),
 'scaling_scenario': ('right_censored', 'delayed_entry', 'strata', 'delayed_entry_strata'),
 'preset': ('quick', 'standard'),
}


def _shape(value, dimensions=2):
    if not isinstance(value, str):
        raise ValueError('Shape must be a string')
    parts = value.split('x')
    if len(parts) != dimensions or any(not p.isdigit() for p in parts):
        raise ValueError('Invalid shape')
    shape = tuple(map(int, parts))
    if shape[0] < 32 or shape[0] > 100000 or not 1 <= shape[1] <= 1024 or shape[0] * shape[1] > 20_000_000:
        raise ValueError('Shape exceeds admitted resource budget')
    if dimensions == 3 and not 2 <= shape[2] <= min(shape[0], 512):
        raise ValueError('Invalid or oversized panel time scale')
    return shape


def normalize_configuration(identifier, override=None):
    defaults = CONFIG_DEFAULTS.get(identifier, {})
    override = {} if override is None else override
    if not isinstance(override, dict) or set(override) - set(defaults):
        raise ValueError('Unknown configuration field for ' + identifier)
    config = json.loads(json.dumps(defaults))
    config.update(override)
    if identifier == 'torch-compile-scale' and override.get('preset') == 'standard' and 'scales' not in override:
        config['scales'] = ['1024x64', '4096x64', '16384x64', '4096x256', '4096x1024', '16384x1024']
    for key, value in config.items():
        if key == 'bandwidth':
            if isinstance(value, str):
                if value not in CHOICES[key]:
                    raise ValueError('Unknown bandwidth selector')
            elif type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 100:
                raise ValueError('Invalid numeric bandwidth')
        elif key in CHOICES:
            if value not in CHOICES[key]:
                raise ValueError('Invalid ' + key)
        elif key in ('scales', 'sizes', 'strata_counts', 'cases', 'models'):
            if not isinstance(value, list) or not value or len(value) != len(set(value)):
                raise ValueError('Expected nonempty distinct ' + key)
            if key == 'scales':
                if len(value) > 6:
                    raise ValueError('At most six shapes per protocol')
                for shape in value:
                    _shape(shape)
            elif key in ('sizes', 'strata_counts'):
                if len(value) > 4 or any(type(v) is not int or not 1 <= v <= (1920 if key == 'sizes' else 256) for v in value):
                    raise ValueError('Scale grid exceeds budget')
            else:
                allowed = (COMPILE_CASES if identifier == 'torch-compile-scale' else HYBRID_CASES) if key == 'cases' else ('linear', 'ridge', 'lasso', 'logistic', 'cox')
                if set(value) - set(allowed):
                    raise ValueError('Unknown ' + key)
        elif key in ('high_t_scale', 'two_way_unbalanced_scale'):
            _shape(value, 3)
        else:
            maximum = {'n': 4096, 'n_samples': 20000, 'queries': 512, 'resamples': 500,
                       'dimension': 8, 'p': 32, 'n_features': 64, 'features': 32,
                       'rows_per_stratum': 64, 'strata_count': 64,
                       'repeats': 10, 'largest_repeats': 10, 'warmup': 3,
                       'warmups': 3, 'warmup_runs': 3, 'seed': 2**32 - 1}[key]
            minimum = 0 if key in ('seed', 'warmup', 'warmups', 'warmup_runs') else 1
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError('Invalid or over-budget ' + key)
    if identifier == 'public-native-kde-bootstrap' and config['kernel'] in ('cosine', 'optcosine') and config['dimension'] != 1:
        raise ValueError('Cosine/optcosine require dimension=1')
    if config.get('n', 2) < 2:
        raise ValueError('n must be at least two')
    if 'queries' in config:
        n, q, count, dim = config['n'], config['queries'], config['resamples'], config.get('dimension', 1)
        if identifier == 'public-native-kde-bootstrap':
            work = (n * q * dim + n * dim**2 + dim**3) * (count + 1)
            total = work * 3 * (config['warmup'] + config['repeats'] + 5)
            selector_bytes = 8 * (20 * 8192 + 20 * 2048) if isinstance(config['bandwidth'], str) and config['bandwidth'] not in ('scott', 'silverman') else 0
            host = 8 * (4*n*count + 8*q*n*dim + 12*count*q + 20*n*dim + 32*dim**2) + selector_bytes
            if work > 1_000_000_000 or total > 20_000_000_000 or host > 512 * 1024**2:
                raise ValueError('Public KDE fixed work/host guard exceeded; no downscaling')
        elif n * q * count * dim * (config['repeats'] + config.get('warmup', 0) + 3) > 2_000_000_000:
            raise ValueError('Developer KDE/hybrid total work exceeds admitted guard')
    return config


def configured_adapter(identifier, config):
    adapter = json.loads(json.dumps(ADAPTERS[identifier]))
    args = adapter['arguments']
    # Replace only allowlisted flags; fixed output/source flags are untouchable.
    for key, value in config.items():
        flag = '--' + (key if identifier == 'gpu-memory-cleanup' else key.replace('_', '-'))
        if flag in args:
            start = args.index(flag)
            end = start + 1
            while end < len(args) and not args[end].startswith('--'):
                end += 1
            del args[start:end]
        if isinstance(value, list):
            comma = key in ('models', 'scales') or (key == 'cases' and identifier == 'torch-compile-scale')
            tokens = [','.join(map(str, value))] if comma else list(map(str, value))
        else:
            tokens = [str(value)]
        args.extend([flag, *tokens])
    return adapter


def expected_cells(identifier, config):
    """Protocol cells, not disjoint statistical model coverage."""
    if identifier == 'public-native-kde-bootstrap':
        return list(BACKENDS)
    if identifier == 'native-kde-bootstrap':
        return ['numpy-controlled', 'cupy-current-controlled', 'cupy-native-controlled']
    if identifier == 'native-kde-rng':
        return ['numpy-public-bootstrap', 'cupy-public-bootstrap', 'cupy-native-rng']
    if identifier == 'hybrid-phases':
        return [f'{b}/{c}' for b in BACKENDS for c in config['cases']]
    if identifier == 'torch-compile-scale':
        return [f'{c}/{s}/{m}' for c in config['cases'] for s in config['scales'] for m in ('disable', 'default')]
    if identifier == 'exact-strata-count-scaling':
        return [f'strata={s}/{b}' for s in config['strata_counts'] for b in ('cpu', 'cuda', 'torch')]
    if identifier == 'exact-ties-scaling':
        return [f'n={n}/{b}/{config["scaling_scenario"]}' for n in config['sizes'] for b in BACKENDS]
    if identifier not in ADAPTERS:
        return []
    if identifier == 'cox-stability-review':
        return [f'{s}/{t}/{d}' for s in ('continuous', 'heavy_ties', 'extreme') for t in ('breslow', 'efron') for d in ('cpu', 'cuda', 'torch')]
    if identifier == 'panel-stage-b-identity-overhead':
        return [f'{b}/{m}/{shape}' for b in ('cupy', 'torch') for m in ('PanelOLS', 'RandomEffects') for shape in config['scales']]
    if identifier == 'panel-stage-c-covariance':
        return ([f'{b}/{c}/base/{shape}' for b in ('cupy', 'torch') for c in PANEL_C_CASES for shape in config['scales']]
                + [f'{b}/{c}/high_t_qs/{config["high_t_scale"]}' for b in ('cupy', 'torch') for c in ('pooled_dk_qs', 'panel_entity_dk_qs')]
                + [f'{b}/panel_two_way_nonrobust/{config["two_way_unbalanced_scale"]}' for b in ('cupy', 'torch')])
    if identifier == 'fama-macbeth-scaling-gpu':
        return [f'{fixture}/{b}' for fixture in ('micro', 'medium', 'large') for b in BACKENDS]
    if identifier == 'torch-compile-maintenance':
        return [f'{case}/{mode}' for case in COMPILE_CASES for mode in ('disable', 'default')]
    if identifier == 'gpu-memory-cleanup':
        return [f'{m}/cleanup={c}/sample={i}' for m in config['models'] for c in (False, True) for i in range(config['warmup_runs'] + config['repeats'])]
    if identifier in ('kde-vs-scipy', 'kernel-regression-vs-statsmodels'):
        return ['statgpu_numpy', 'statgpu_cupy', 'scipy' if identifier == 'kde-vs-scipy' else 'statsmodels']
    if identifier == 'hc2-hc3-hac-gate':
        return [f'linear/{cov}/{b}' for cov in ('hc2', 'hc3', 'hac') for b in ('statsmodels', 'statgpu_cpu', 'statgpu_cupy_gpu', 'statgpu_torch_gpu')]
    if identifier == 'unsupervised':
        return [f'{m}/{b}' for m in ('PCA', 'KMeans') for b in ('cpu', 'cuda', 'torch', 'sklearn')]
    if identifier == 'unsupervised-dbscan-cython':
        return [f'{case}/{b}' for case in ('compact_medium', 'compact_2500', 'compact_large', 'variable', 'all_noise') for b in ('statgpu_cpu_cython', 'statgpu_cpu_fallback', 'sklearn', 'statgpu_cuda', 'statgpu_torch')]
    if identifier == 'unsupervised-external-matrix':
        return [f'{m}/{v}/{b}' for m in ('PCA', 'KMeans') for v in (2, 4) for b in (('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch', 'sklearn', 'statsmodels', 'R') if m == 'PCA' else ('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch', 'sklearn', 'R'))]
    if identifier == 'unsupervised-phase2':
        return ([f'{m}/{b}' for m in ('DBSCAN', 'GaussianMixture', 'NMF') for b in ('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch', 'sklearn')]
                + [f'AgglomerativeClustering/{b}' for b in ('statgpu_cpu', 'sklearn', 'scipy', 'R cluster')]
                + ['UMAP/umap-learn', 'TSNE/openTSNE'])
    if identifier == 'unsupervised-phase3':
        refs = {'TruncatedSVD': ('sklearn_cpu', 'statsmodels_cpu', 'R_cpu'), 'MiniBatchKMeans': ('sklearn_cpu',), 'UMAP': ('umap_learn_cpu',), 'TSNE': ('sklearn_cpu', 'openTSNE_cpu')}
        return [f'{m}/{b}' for m, external in refs.items() for b in ('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch', *external)]
    if identifier == 'unsupervised-phase3b':
        return ([f'GaussianMixture/{cov}/{b}' for cov in ('diag', 'spherical', 'tied', 'full') for b in ('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch', 'sklearn_cpu')]
                + [f'AgglomerativeClustering/{link}/{b}' for link in ('ward', 'complete', 'average', 'single') for b in ('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch', 'sklearn_cpu', 'scipy_cpu', 'R_cluster_agnes')])
    if identifier == 'unsupervised-phase3c':
        return [f'{m}/{b}' for m in ('IncrementalPCA', 'MiniBatchNMF') for b in ('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch', 'sklearn_cpu')]
    if identifier == 'nonparametric-vs-r':
        return [f'{m}/{b}' for m in ('kde', 'kernel_regression_nw', 'kernel_regression_local_linear') for b in ('statgpu', 'r')] + [f'kde_confidence_interval/{b}' for b in ('statgpu', 'r', 'scipy')]
    if identifier == 'lassocv-impls':
        return ['sklearn_lassocv_cpu/seed=42', 'statgpu_lassocv_cpu/seed=42', 'statgpu_lassocv_gpu/seed=42']
    raise ValueError('Missing reviewed expected-cell map for ' + identifier)



def build_plan(lanes=None, *, expected_sha=None, validation_tier='local-full', environment_id=None, configurations=None):
    if validation_tier not in TIERS:
        raise ValueError('Unknown validation tier')
    entries = registry()
    selected = sorted(entries) if lanes is None else list(lanes)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(entries):
        raise ValueError('Select distinct known independent lane IDs')
    configurations = {} if configurations is None else configurations
    if not isinstance(configurations, dict) or set(configurations) - set(selected):
        raise ValueError('Configurations must name selected lanes only')
    jobs = []
    for identifier in selected:
        entry = entries[identifier]
        variants = configurations.get(identifier, [{}])
        if not isinstance(variants, list) or not variants or len(variants) > 128:
            raise ValueError('Each lane needs 1 to 128 bounded configurations')
        for override in variants:
            configuration = normalize_configuration(identifier, override)
            adapter = configured_adapter(identifier, configuration) if identifier in ADAPTERS else None
            config = {'protocol_version': 1, 'lane_id': identifier, 'adapter': adapter,
                      'configuration': configuration, 'validation_tier': validation_tier,
                      'environment_id': environment_id, 'expected_sha': expected_sha}
            config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            jobs.append(dict(entry, adapter=adapter, config_id=config_id, configuration=configuration,
                             expected_cells=expected_cells(identifier, configuration)))
    if len({j['config_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate exact protocol configurations')
    return {'schema_version': 'statgpu-independent-plan-v1', 'mode': 'plan-only',
            'audit_baseline_sha': load_catalog()['audit_baseline_sha'],
            'expected_sha': expected_sha, 'validation_tier': validation_tier,
            'environment_id': environment_id, 'jobs': jobs,
            'counts': {'audited_files': 202, 'independent_runner_files': 41,
                       'selected_lanes': len(selected), 'selected_configurations': len(jobs), 'ready_selected': sum(j['adapter'] is not None for j in jobs),
                       'registered_ready_lanes': len(ADAPTERS), 'blocked_or_deduplicated_lanes': len(BLOCK_REASONS)},
            'coverage_status': 'partial: bounded admitted protocols, not full numerical coverage',
            'performance_eligible': False, 'canonical_dashboard_ready': False,
            'accounting': 'File counts are not unique models/cases; wrappers never execute here. Protocol config IDs include scope, arguments, budgets, tier, source and environment.'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, *, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0), label)


def _samples(values, count, label):
    _require(isinstance(values, list) and len(values) == count, label + ': missing repeat')
    for value in values:
        _number(value, label, positive=True)


def _rows(payload, field, key, expected):
    rows = payload.get(field)
    _require(isinstance(rows, list) and len(rows) == len(expected), field + ': wrong row count')
    identities = [key(row) for row in rows]
    _require(len(set(identities)) == len(identities) and set(identities) == set(expected), field + ': missing/duplicate/unexpected identities')
    return rows


def _validate_nested_thread_limits(value):
    _require(isinstance(value, dict) and set(value) == {'before', 'after'}, 'Missing nested workload thread evidence')
    for state in value.values():
        _require(state.get('cap_verified') is True and all(type(state.get(k)) is int and state[k] == 1 for k in ('intraop', 'interop', 'inductor_compile_threads')) and bool(state.get('pools')) and all(p.get('num_threads') == 1 for p in state['pools']), 'Nested workload thread cap unverified')


def validate_artifact(lane, payload, configuration=None):
    """Fail closed on exact pilot row identities; retain diagnostic-only semantics.

    Protocol completion is not automatically correctness, device proof or speed.
    The raw document is never modified by this validator.
    """
    _require(lane in ADAPTERS and isinstance(payload, dict), 'Unknown adapter or nonobject artifact')
    config = normalize_configuration(lane, configuration)
    repeats = config.get('repeats', 3)
    adapter = ADAPTERS[lane]
    contract = adapter['contract']
    _require(payload.get('source_changed') is not True, 'Runner reported source drift')
    if contract in ('public-kde', 'controlled-kde', 'rng-kde'):
        for key in ('n', 'queries', 'resamples', 'seed', 'weights', 'dimension', 'kernel', 'bandwidth'):
            if key in config:
                _require(payload.get('problem', {}).get(key) == config[key], 'Wrong configured KDE problem: ' + key)
    _require(payload.get('gate_failures', []) == [], 'Runner gate failures')
    correctness = 'unavailable' if adapter['correctness_scope'].startswith('unavailable:') else 'pass'
    if contract in ('public-kde', 'controlled-kde', 'rng-kde', 'hybrid'):
        _require(payload.get('schema_version') == 'statgpu-benchmark-v2', 'Wrong benchmark schema')
        if contract == 'public-kde':
            expected, key = BACKENDS, lambda r: r.get('backend')
        elif contract == 'controlled-kde':
            expected, key = ('numpy-controlled', 'cupy-current-controlled', 'cupy-native-controlled'), lambda r: r.get('variant')
        elif contract == 'rng-kde':
            expected, key = ('numpy-public-bootstrap', 'cupy-public-bootstrap', 'cupy-native-rng'), lambda r: r.get('variant')
        else:
            expected, key = [(b, c) for b in BACKENDS for c in config['cases']], lambda r: (r.get('backend'), r.get('case'))
        rows = _rows(payload, 'rows', key, expected)
        for row in rows:
            _require(row.get('status') == 'pass', 'Missing, unavailable or failed backend/variant')
            _require(row.get('correctness', {}).get('status') == 'pass', 'Missing row correctness evidence')
            if contract == 'hybrid':
                for field in ('n', 'p', 'queries', 'resamples', 'seed'):
                    _require(row.get('problem', {}).get(field) == config[field], 'Wrong hybrid configuration: ' + field)
            timing = row.get('timing', {})
            # Controlled/RNG experiments call their uninstrumented timing record performance.
            if not timing and isinstance(row.get('performance'), dict):
                timing = row['performance']
            _samples(timing.get('samples_seconds'), repeats, 'timing')
        _require(payload.get('validation', {}).get('status') in ('complete', 'pass'), 'Incomplete validation')
        if contract != 'hybrid':
            _require(payload.get('timing', {}).get('warmup') == config['warmup'] and payload.get('timing', {}).get('repeats') == repeats, 'Wrong timing configuration')
    elif contract == 'cox-stability':
        expected = {f'{scenario}:{ties}:{device}' for scenario in ('continuous', 'heavy_ties') for ties in ('breslow', 'efron') for device in ('cpu', 'cuda', 'torch')}
        _require(set(payload.get('cases', {})) == expected, 'Incomplete Cox case matrix')
        _require(payload.get('shape') == {'n': config['n'], 'p': config['p']}, 'Wrong Cox pilot shape')
        _require(payload.get('timing_contract', {}).get('warmups') == config['warmups'] and payload['timing_contract'].get('repeats') == repeats, 'Wrong Cox timing configuration')
        for row in payload['cases'].values():
            _require(row.get('all_finite') is True and row.get('all_converged') is True, 'Cox finite/convergence failure')
            _samples([r.get('seconds') for r in row.get('runs', [])], repeats, 'Cox samples')
        extreme = payload.get('extreme_predictor_cases', {})
        _require(set(extreme) == {f'{ties}:{device}' for ties in ('breslow', 'efron') for device in ('cpu', 'cuda', 'torch')}, 'Incomplete extreme cases')
        _require(all(r.get('finite') is True for r in extreme.values()), 'Extreme case finite/convergence failure')
    elif contract in ('exact-strata', 'exact-ties'):
        _require(payload.get('status') == 'complete', 'Exact protocol incomplete')
        fields = ('rows_per_stratum', 'features', 'seed') if contract == 'exact-strata' else ('features', 'seed', 'scaling_scenario')
        for field in fields:
            _require(payload.get(field) == config[field], 'Wrong Exact configuration: ' + field)
        if contract == 'exact-ties':
            _require(payload.get('strata_count_requested') == config['strata_count'], 'Wrong Exact strata count')
        expected = config['strata_counts'] if contract == 'exact-strata' else config['sizes']
        field = 'strata_count' if contract == 'exact-strata' else 'n'
        rows = _rows(payload, 'cases', lambda r: r.get(field), expected)
        for row in rows:
            if contract == 'exact-strata':
                _require(row.get('n') == row['strata_count'] * config['rows_per_stratum'], 'Wrong Exact effective size')
            backends = row.get('backends', {})
            _require(set(backends) == ({'cpu', 'cuda', 'torch'} if contract == 'exact-strata' else set(BACKENDS)), 'Exact backend missing')
            for value in backends.values():
                _require(value.get('all_finite') is True, 'Exact nonfinite')
                if contract == 'exact-ties':
                    _require(value.get('all_converged') is True, 'Exact nonconvergence')
                _samples(value.get('seconds'), config['largest_repeats'] if contract == 'exact-ties' and row['n'] == max(config['sizes']) else repeats, 'Exact timing')
                if contract == 'exact-strata':
                    diff = value.get('log_likelihood_abs_vs_cpu')
                    _number(diff, 'Exact precision missing')
                    _require(diff <= 1e-9, 'Exact objective precision failed')
    elif contract == 'panel-b':
        rows = _rows(payload, 'rows', lambda r: (r.get('backend'), r.get('model'), r.get('n_samples'), r.get('n_features')),
                     [(b, m, *_shape(s)) for b in ('cupy', 'torch') for m in ('PanelOLS', 'RandomEffects') for s in config['scales']])
        for row in rows:
            for field in ('with_digest_samples', 'without_digest_samples'):
                _samples(row.get(field), repeats, field)
    elif contract == 'panel-c':
        _require(payload.get('high_t_scale') == config['high_t_scale'] and payload.get('two_way_unbalanced_scale') == config['two_way_unbalanced_scale'], 'Wrong Panel time scale')
        expected = [(b, c, 'base', *_shape(s)) for b in ('cupy', 'torch') for c in PANEL_C_CASES for s in config['scales']]
        expected += [(b, c, 'high_t_qs', *_shape(config['high_t_scale'], 3)[:2]) for b in ('cupy', 'torch') for c in ('pooled_dk_qs', 'panel_entity_dk_qs')]
        expected += [(b, 'panel_two_way_nonrobust', 'two_way_unbalanced', *_shape(config['two_way_unbalanced_scale'], 3)[:2]) for b in ('cupy', 'torch')]
        rows = _rows(payload, 'rows', lambda r: tuple(r.get(k) for k in ('backend', 'case', 'scenario', 'n_samples', 'n_features')), expected)
        for row in rows:
            _samples(row.get('samples_seconds'), repeats, 'Panel C timing')
            expected_t = 20 if row['scenario'] == 'base' else _shape(config['high_t_scale'] if row['scenario'] == 'high_t_qs' else config['two_way_unbalanced_scale'], 3)[2]
            _require(row.get('n_times') == expected_t, 'Wrong effective Panel time dimension')
    elif contract == 'fama-macbeth':
        fixtures = payload.get('fixtures', {})
        _require(payload.get('status') == 'success' and set(fixtures) == {'micro', 'medium', 'large'}, 'Incomplete Fama-MacBeth fixtures')
        _require(payload.get('timing_protocol', {}).get('warmup') == config['warmup'] and payload['timing_protocol'].get('repeats') == repeats, 'Wrong Fama timing protocol')
        for name, row in fixtures.items():
            expected_shape = {'micro': (64, 128, 4), 'medium': (128, 1024, 8), 'large': (128, 4096, 16)}[name]
            _require(tuple(row.get('shape', {}).get(k) for k in ('n_times', 'observations_per_period', 'n_features')) == expected_shape, 'Wrong Fama fixture shape')
            _samples(row.get('numpy', {}).get('samples_seconds'), repeats, 'Fama numpy timing')
            _require(set(row.get('backends', {})) == {'cupy', 'torch'}, 'Missing Fama backend')
            for result in row['backends'].values():
                _samples(result.get('samples_seconds'), repeats, 'Fama timing')
                _require(bool(result.get('max_abs_differences_vs_numpy')), 'Missing Fama precision metrics')
    elif contract == 'compile-maintenance':
        precision = payload.get('precision_vs_external', {}).get('default_vs_disable', {})
        _require(set(precision) == set(COMPILE_CASES), 'Incomplete compile case set')
        _require(all(r.get('status') == 'pass' for r in precision.values()), 'Compile precision failed')
        modes = payload.get('compatibility_matrix', {})
        _require(set(modes) == {'disable', 'default'}, 'Missing compile mode')
        for mode, result in modes.items():
            _validate_nested_thread_limits(result.get('thread_limits'))
            _require(set(result.get('cases', {})) == set(COMPILE_CASES), 'Missing mode case')
            for value in result['cases'].values():
                _samples(value.get('fit_seconds'), repeats, 'Compile timing')
                _require(value.get('finite_prediction') is True and value.get('finite_coefficients') is True, 'Compile nonfinite')
                if mode == 'default':
                    _require(value.get('unique_graphs_delta', 0) > 0, 'No executed compile graph')
    elif contract == 'compile-scale':
        rows = _rows(payload, 'results', lambda r: (r.get('case'), r.get('n_samples'), r.get('n_features')), [(c, *_shape(s)) for c in config['cases'] for s in config['scales']])
        for row in rows:
            _require(row.get('precision', {}).get('status') == 'pass', 'Compile precision failed')
            _require(row.get('compile', {}).get('default', {}).get('unique_graphs_delta', 0) > 0, 'No compile graph')
            for mode in ('disable', 'default'):
                _validate_nested_thread_limits(row.get('thread_limits', {}).get(mode))
                _samples(row.get('raw_fit_seconds', {}).get(mode), repeats, 'Compile timing')
    elif contract == 'memory-cleanup':
        rows = _rows(payload, 'rows', lambda r: (r.get('model'), r.get('cleanup'), r.get('sample_index')),
                     [(m, c, i) for m in config['models'] for c in (False, True) for i in range(config['warmup_runs'] + repeats)])
        for row in rows:
            _number(row.get('fit_ms'), 'Memory timing', positive=True)
            for field in ('used_after_fit', 'total_after_fit', 'used_after_reset', 'total_after_reset'):
                _require(type(row.get(field)) is int and row[field] >= 0, 'Invalid pool snapshot')
            _require(row['used_after_fit'] <= row['total_after_fit'] and row['used_after_reset'] <= row['total_after_reset'], 'Inconsistent pool snapshots')
    elif contract in ('kde-scipy', 'kernel-statsmodels'):
        _require(payload.get('config', {}).get('n_samples') == 256 and payload['config'].get('n_eval') == 64, 'Wrong external pilot shape')
        reference = 'scipy' if contract == 'kde-scipy' else 'statsmodels'
        timing = payload.get('metrics', {}).get('timing_ms', {})
        precision = payload.get('metrics', {}).get('precision', {})
        _require(set(timing) == {reference, 'statgpu_numpy', 'statgpu_cupy', 'ratios'}, 'Missing external timing cell')
        _require(set(precision) == {'numpy_vs_' + reference, 'cupy_vs_' + reference, 'cupy_vs_numpy'}, 'Missing precision cell')
        for name in (reference, 'statgpu_numpy', 'statgpu_cupy'):
            _samples(timing[name].get('times_ms'), 3, name)
        for metrics in precision.values():
            _number(metrics.get('max_abs_diff'), 'Missing precision metric')
    elif contract == 'covariance-gate':
        rows = _rows(payload, 'results', lambda r: (r.get('model'), r.get('cov_type'), r.get('backend')),
                     [('linear', c, b) for c in ('hc2', 'hc3', 'hac') for b in ('statsmodels', 'statgpu_cpu', 'statgpu_cupy_gpu', 'statgpu_torch_gpu')])
        for row in rows:
            _require(not row.get('error'), 'Covariance row error')
            _number(row.get('time_ms'), 'Covariance timing', positive=True)
            if row['backend'] != 'statsmodels':
                for field in ('coef_max_diff', 'bse_max_diff', 'pvalue_max_diff'):
                    _number(row.get(field), 'Missing covariance comparison')
    elif contract == 'unsupervised-base':
        rows = _rows(payload, 'rows', lambda r: (r.get('method'), r.get('device')),
                     [(m, d) for m in ('PCA', 'KMeans') for d in ('cpu', 'cuda', 'torch', 'sklearn')])
        for row in rows:
            _require(row.get('status') == 'ok', 'Skipped/error unsupervised cell')
            _samples(row.get('fit_ms_all'), 3, 'Unsupervised timing')
    elif contract == 'dbscan-cython':
        _require(payload.get('extension_loaded') is True, 'Compiled DBSCAN extension unavailable')
        cases = ('compact_medium', 'compact_2500', 'compact_large', 'variable', 'all_noise')
        methods = [('statgpu_cpu_cython', 'cpu'), ('statgpu_cpu_fallback', 'cpu'), ('sklearn', 'cpu'), ('statgpu', 'cuda'), ('statgpu', 'torch')]
        rows = _rows(payload, 'rows', lambda r: (r.get('case'), r.get('framework'), r.get('backend')), [(c, f, b) for c in cases for f, b in methods])
        for row in rows:
            _require(row.get('status') == 'ok', 'DBSCAN skipped/error cell')
            _samples(row.get('fit_ms_all'), 3, 'DBSCAN timing')
            if row['framework'] != 'statgpu_cpu_cython':
                _require(row.get('noise_mask_match_vs_cython') is True and row.get('ari_vs_cython') == 1.0, 'DBSCAN partition disagreement')
    elif contract == 'unsupervised-external':
        expected = [(m, 'statgpu', b, v) for m in ('PCA', 'KMeans') for b in ('cpu', 'cuda', 'torch') for v in (2, 4)]
        expected += [(m, f, 'cpu', v) for m in ('PCA', 'KMeans') for f in (('sklearn', 'statsmodels', 'R') if m == 'PCA' else ('sklearn', 'R')) for v in (2, 4)]
        rows = _rows(payload, 'rows', lambda r: (r.get('method'), r.get('framework'), r.get('backend'), r.get('n_components') if r.get('method') == 'PCA' else r.get('k')), expected)
        _require(all(r.get('status') == 'ok' for r in rows), 'External baseline unavailable/error/timeout')
        for row in rows:
            _number(row.get('fit_ms'), 'External matrix timing', positive=True)
            _samples(row.get('fit_ms_all'), 3, 'External matrix repeats')
    elif contract == 'unsupervised-phase2':
        expected = [(m, 'statgpu', b) for m in ('DBSCAN', 'GaussianMixture', 'NMF') for b in ('cpu', 'cuda', 'torch')]
        expected += [(m, 'sklearn', 'cpu') for m in ('DBSCAN', 'GaussianMixture', 'NMF')]
        expected += [('AgglomerativeClustering', f, 'cpu') for f in ('statgpu', 'sklearn', 'scipy', 'R cluster')]
        expected += [('UMAP', 'umap-learn', 'cpu'), ('TSNE', 'openTSNE', 'cpu')]
        rows = _rows(payload, 'rows', lambda r: (r.get('method'), r.get('framework'), r.get('backend')), expected)
        _require(all(r.get('status') == 'ok' for r in rows), 'Phase2 external dependencies/cells incomplete')
        for row in rows:
            _number(row.get('fit_ms'), 'Phase2 timing', positive=True)
            _samples(row.get('fit_ms_all'), 3, 'Phase2 repeats')
    elif contract in ('unsupervised-phase3', 'unsupervised-phase3b', 'unsupervised-phase3c'):
        models = payload.get('models', {})
        expected = {'unsupervised-phase3': {'TruncatedSVD', 'MiniBatchKMeans', 'UMAP', 'TSNE'},
                    'unsupervised-phase3b': {'GaussianMixture', 'AgglomerativeClustering'},
                    'unsupervised-phase3c': {'IncrementalPCA', 'MiniBatchNMF'}}[contract]
        _require(set(models) == expected, 'Missing extension model')
        entries = []
        if contract == 'unsupervised-phase3b':
            _require(set(models['GaussianMixture']) == {'diag', 'spherical', 'tied', 'full'}, 'Missing GMM covariance')
            _require(set(models['AgglomerativeClustering']) == {'ward', 'complete', 'average', 'single'}, 'Missing linkage')
            entries = [(v, {'sklearn_cpu'} if name == 'GaussianMixture' else {'sklearn_cpu', 'scipy_cpu', 'R_cluster_agnes'}) for name, model in models.items() for v in model.values()]
        elif contract == 'unsupervised-phase3':
            references = {'TruncatedSVD': {'sklearn_cpu', 'statsmodels_cpu', 'R_cpu', 'cuml_gpu'}, 'MiniBatchKMeans': {'sklearn_cpu', 'cuml_gpu'}, 'UMAP': {'umap_learn_cpu', 'cuml_gpu'}, 'TSNE': {'sklearn_cpu', 'openTSNE_cpu', 'cuml_gpu'}}
            entries = [(v, references[name]) for name, v in models.items()]
        else:
            entries = [(v, {'sklearn_cpu'}) for v in models.values()]
        for values, references in entries:
            _require(set(values) == {'statgpu_cpu', 'statgpu_cuda', 'statgpu_torch'} | references, 'Missing/unexpected extension backend or reference')
            for name, row in values.items():
                # cuML entries explicitly not implemented by these source
                # protocols are uncovered, never additional accepted cells.
                if name == 'cuml_gpu':
                    continue
                _require(isinstance(row, dict) and row.get('status') == 'ok', 'Extension or external reference skipped/error')
                if name.startswith('statgpu_'):
                    value = row.get('fit_ms_mean', row.get('timing', {}).get('warm_mean_ms'))
                    _number(value, 'Extension timing missing', positive=True)
    elif contract == 'nonparametric-r':
        metrics = payload.get('metrics', {})
        _require(set(metrics) == {'kde', 'kernel_regression_nw', 'kernel_regression_local_linear', 'kde_confidence_interval'}, 'Missing nonparametric oracle method')
        configuration = payload.get('config', {})
        _require(all(configuration.get(k) == v for k, v in {'n_samples': 256, 'n_eval': 64, 'n_ci_eval': 32, 'statgpu_backend': 'cupy', 'repeats': 3, 'ci_repeats': 3, 'ci_resamples': 30}.items()), 'Wrong R comparison configuration')
        def precision(value):
            _require(set(value) == {'max_abs_diff', 'mean_abs_diff', 'l2_rel_diff'}, 'Missing exact precision metrics')
            for metric in value.values(): _number(metric, 'Invalid precision metric')
        def timing(value, samples):
            for key in ('time_ms_mean', 'time_ms_std', 'time_ms_min', 'time_ms_max'):
                _number(value.get(key), 'Missing timing summary: ' + key)
            if samples:
                _samples(value.get('times_ms'), 3, 'Nonparametric timing')
        for method, value in metrics.items():
            ci = method == 'kde_confidence_interval'
            expected_timing = {'statgpu', 'r', 'ratio_statgpu_over_r'} | ({'scipy', 'ratio_statgpu_over_scipy'} if ci else set())
            _require(set(value.get('timing_ms', {})) == expected_timing, 'Missing exact R/SciPy timing cells')
            timing(value['timing_ms']['statgpu'], True)
            timing(value['timing_ms']['r'], False)  # R producer retains summaries, not individual samples.
            if ci:
                timing(value['timing_ms']['scipy'], True)
                for reference in ('precision_vs_r', 'precision_vs_scipy'):
                    _require(set(value.get(reference, {})) == {'estimate', 'lower', 'upper'}, 'Missing CI oracle outputs')
                    for metric in value[reference].values(): precision(metric)
            else:
                precision(value.get('precision', {}))
    elif contract == 'lassocv':
        methods = payload.get('methods', {})
        _require(set(methods) == {'sklearn_lassocv_cpu', 'statgpu_lassocv_cpu', 'statgpu_lassocv_gpu'}, 'Missing LassoCV implementation')
        _require(payload.get('config', {}).get('shared_alpha_grid') is True and payload['config'].get('shared_cv_splits') is True, 'CV grids/folds not aligned')
        _require(all(payload['config'].get(k) == v for k, v in {'seeds': [42], 'n_samples': 128, 'n_features': 16, 'n_test_samples': 64, 'cv': 3, 'n_alphas': 3, 'repeats': 3}.items()), 'Wrong CV configuration')
        for value in methods.values():
            _require(isinstance(value, dict) and len(value.get('runs', [])) == 1 and value.get('aggregate', {}).get('n_runs') == 1, 'Missing CV seed/aggregate')
            row = value['runs'][0]
            _require(row.get('seed') == 42, 'Wrong CV seed')
            for field in ('time_ms', 'alpha', 'train_mse', 'test_mse'):
                _number(row.get(field), 'Missing CV metric: ' + field)
        _require(not any(payload.get('errors', {}).values()), 'CV dependency error')
    else:
        raise ValueError('Unreviewed output contract')
    # Do not mistake backend names, availability probes, output residency or a
    # successful process for positive executed-device evidence.
    return {'protocol_status': 'complete', 'correctness_status': correctness,
            'correctness_scope': adapter['correctness_scope'],
            'device_claim': 'recorded-scoped-evidence-requires-review' if contract in ('public-kde', 'controlled-kde', 'rng-kde', 'hybrid') and any(isinstance(r.get('operation_evidence', r.get('backend_evidence')), dict) for r in payload.get('rows', [])) else 'unavailable: independent adapter does not certify executed numerical device',
            'performance_eligible': False, 'canonical_dashboard_ready': False,
            'limitations': adapter['limitations']}


def torch_child_thread_state(torch_module, *, pool_info=None, compile_config=None):
    """Inspect the actual nested workload process, never its idle parent.

    Optional dependency injection is solely for CPU-only protocol unit tests.
    Numerical imports happen only when this execution helper is explicitly used.
    """
    if pool_info is None:
        from threadpoolctl import threadpool_info
        pool_info = threadpool_info
    if compile_config is None:
        import importlib
        compile_config = importlib.import_module('torch._inductor.config')
    pools = pool_info()
    result = {'intraop': torch_module.get_num_threads(),
              'interop': torch_module.get_num_interop_threads(),
              'inductor_compile_threads': compile_config.compile_threads,
              'pools': pools, 'cap_verified': False}
    if not pools or any(type(p.get('num_threads')) is not int or p['num_threads'] != 1 for p in pools):
        raise RuntimeError('Nested compile child numerical pool is not serial')
    if any(result[k] != 1 for k in ('intraop', 'interop', 'inductor_compile_threads')):
        raise RuntimeError('Nested compile child Torch/Inductor thread budget exceeded')
    result['cap_verified'] = True
    return result


def configure_torch_child_threads(torch_module, *, pool_info=None, compile_config=None):
    """Set serial limits before any work in a fresh compile subprocess."""
    import os
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['MAX_JOBS'] = '1'
    os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] = '1'
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    if compile_config is None:
        import importlib
        compile_config = importlib.import_module('torch._inductor.config')
    compile_config.compile_threads = 1
    return torch_child_thread_state(torch_module, pool_info=pool_info, compile_config=compile_config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate-catalog', nargs=2, metavar=('AUDIT_JSON', 'COMMAND_PLAN_JSON'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.regenerate_catalog:
        if args.output is None:
            parser.error('--output is required; existing files are never replaced')
        document = generate_catalog(*args.regenerate_catalog)
        with args.output.open('x') as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    else:
        print(json.dumps(build_plan(), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
