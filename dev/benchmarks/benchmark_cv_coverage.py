#!/usr/bin/env python3
"""Scoped six-family CV candidate benchmark, reusing the maintained CV runner.

Unlike the immutable canonical-source schema, this adapter permits a CPU-only
run. Diagnostic profiling and clean synchronized public-fit samples are separate.
It never registers the output as canonical. CUDA timings require observations
of native numerical arrays inside both diagnostic selector and final-refit
regions on concrete device 0. Missing proof fails the
case; absent GPU packages are unavailable, never silently replaced by CPU.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import statgpu
from dev.benchmarks import benchmark_cv_models as legacy
from dev.benchmarks import cv_external

EXPECTED_MODELS = {s.model_id for s in legacy.CASE_SPECS}
TIMINGS = ('cv_evaluation_ms', 'final_refit_ms', 'total_fit_ms')


def source_identity():
    files = sorted((ROOT/'statgpu').rglob('*.py')) + [Path(__file__).resolve()]
    files += [Path(legacy.__file__), Path(cv_external.__file__), Path(__file__).with_name('cv_source.py'), Path(__file__).with_name('cv_source_schema.json')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    try:
        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT))
        tracked = subprocess.run(['git','ls-files','--error-unmatch',str(Path(__file__).relative_to(ROOT))],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.CalledProcessError):
        head, dirty, tracked = None, True, False
    return {'git_commit':head, 'working_tree':'dirty' if dirty or not tracked else 'clean',
            'runner_overlay':not tracked, 'statgpu_version':statgpu.__version__,
            'statgpu_import_path':str(Path(statgpu.__file__).resolve()),
            'files_sha256':hashes, 'content_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}


def validate_scoped_source(source, backends, repeats):
    """Validate measured records, without fabricating omitted backend rows."""
    cases = source.get('cases', [])
    if len(cases) != len(EXPECTED_MODELS) or {c.get('model_id') for c in cases} != EXPECTED_MODELS:
        raise ValueError('Expected exactly the six maintained CV model families')
    if len({c.get('case_id') for c in cases}) != len(cases):
        raise ValueError('Duplicate CV case identity')
    for case in cases:
        runs = case.get('runs', [])
        measured = [r for r in runs if r.get('framework') == 'statgpu']
        if sorted(r.get('backend') for r in measured) != sorted(backends):
            raise ValueError('Missing or duplicate requested backend disposition')
        for run in runs:
            if run.get('status') != 'success':
                if not run.get('reason'):
                    raise ValueError('Non-success row needs reason')
                if run.get('timing') is not None:
                    raise ValueError('Non-success row cannot retain timing')
                continue
            if not run.get('selected_parameters'):
                raise ValueError('Successful CV row lacks selected parameter')
            timing = run.get('timing') or {}
            if any(not isinstance(timing.get(k),(int,float)) or not np.isfinite(timing[k]) or timing[k] < 0 for k in TIMINGS):
                raise ValueError('Invalid CV/refit/total duration')
            if timing['total_fit_ms'] < max(timing[k] for k in TIMINGS[:2]):
                raise ValueError('CV component exceeds total duration')
            samples=run.get('repeat_samples', [])
            if len(samples) != repeats:
                raise ValueError('Incorrect raw repeat count')
            for sample in samples:
                if any(not np.isfinite(sample.get(k, np.nan)) or sample[k] < 0 for k in TIMINGS):
                    raise ValueError('Invalid raw duration')
            convergence=run.get('convergence') or {}
            if (convergence.get('fold_count') != case['cv']['fold_count']
                    or convergence.get('candidate_count') != case['grid']['candidate_count']
                    or not convergence.get('final_refit_converged')
                    or convergence.get('failed_candidates') != 0 or convergence.get('failed_folds') != 0):
                raise ValueError('CV/refit convergence metadata invalid')
            if any(not np.isfinite(run.get('scores',{}).get(k,np.nan)) for k in ('validation_score','final_score')):
                raise ValueError('Nonfinite CV or final score')


def convergence_errors():
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergence
    from statgpu.solvers._convergence import ConvergenceWarning as SolverConvergence
    from statgpu.glm_core._solver_utils import ConvergenceWarning as GLMConvergence
    for category in (SklearnConvergence,SolverConvergence,GLMConvergence):
        warnings.simplefilter('error', category)


def direct_refit_check(spec, seed, n, p):
    """A fresh unfitted direct estimator must reproduce selected CV refit."""
    generator={'regression':legacy._regression_data,'classification':legacy._classification_data,'survival':legacy._survival_data}[spec.task]
    X,y,Xt,yt=generator(seed,n,p)
    cls=getattr(importlib.import_module(spec.import_module),spec.import_name)
    model=legacy._construct(cls,legacy._model_kwargs(spec,'numpy',seed))
    legacy._clear_cv_caches(strict=True)
    with warnings.catch_warnings():
        convergence_errors()
        model.fit(*legacy._fit_args(spec,X,y))
        final=model.estimator_
        fresh=legacy._construct(type(final), final.get_params(deep=False))
        fresh.fit(*legacy._fit_args(spec,X,y))
        legacy._validate_convergence(model)
        legacy._validate_convergence(fresh)
    left=legacy._to_numpy(model.coef_)
    right=legacy._to_numpy(fresh.coef_)
    score_a=legacy._score_model(model,spec,Xt,yt)
    score_b=legacy._score_model(fresh,spec,Xt,yt)
    np.testing.assert_allclose(left,right,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(score_a,score_b,rtol=2e-5,atol=2e-6)
    # Independent repeat/refit contract, not an external Cox statistical oracle.
    return {'status':'pass','reference_type':'trusted-statgpu',
            'reference':'fresh direct estimator under selected CV parameters',
            'coefficient_max_abs_diff':float(np.max(np.abs(left-right))),
            'score_abs_diff':abs(score_a-score_b),
            'thresholds':{'atol':2e-6,'rtol':2e-5},
            'selected_parameters':legacy._selected_parameters(model),
            'reference_validation_score':legacy._validation_score(model,spec), 'reference_final_score':score_a}


def check_external_alignment(case):
    runs=case['runs']
    native=next((r for r in runs if r['framework']=='statgpu' and r['status']=='success'),None)
    reference=next((r for r in runs if r['framework']=='sklearn'),None)
    if native is None or reference is None:
        return {'status':'not_applicable','reason':'No equivalent sklearn reference in maintained six-family runner'}
    if reference['status']!='success':
        raise ValueError('Equivalent sklearn reference failed or was unavailable')
    if native['selected_parameters'] != reference['selected_parameters']:
        raise ValueError('Selected parameter disagrees with aligned sklearn grid')
    errors={}
    for key in ('validation_score','final_score'):
        a,b=native['scores'][key],reference['scores'][key]
        np.testing.assert_allclose(a,b,rtol=3e-3,atol=3e-4)
        errors[key+'_abs_diff']=abs(a-b)
    return {'status':'pass','reference_type':'external-python','reference':'aligned sklearn explicit CV grid',
            'thresholds':{'atol':3e-4,'rtol':3e-3,'scope':'well-conditioned generated six-family fixtures; different optimizer tolerances'},
            **errors}


# Computed intermediates, not input arrays or a fitted backend label. Keep this
# inventory tied to actual numerical implementations; adding all linear_model
# frames would allow a GPU-labelled wrapper around a CPU solve to pass.
_NUMERICAL_GROUPS = {
    ('statgpu.linear_model.cv._ridge_cv', '_solve_ridge_path_gpu_from_gram_eig'):
        (('eigvals', 'QTXty', 'coefs'),),
    ('statgpu.linear_model.cv._ridge_cv', '_solve_ridge_path_native_from_gram'):
        (('system', 'coef'),),
    ('statgpu.linear_model.cv._logistic_cv', '_solve_logistic_path_gpu_from_batch'):
        (('XtWX', 'Xtz', 'params'),),
    ('statgpu.linear_model.wrappers._lasso', '_solve_lasso_path_gpu_fista_batched_from_gram'):
        (('grad', 'coef_new', 'coefs'),),
    ('statgpu.linear_model.wrappers._lasso', '_solve_lasso_path_gpu_fista_multi_fold_from_gram'):
        (('grad', 'coef_candidate', 'coefs'),),
    ('statgpu.linear_model.wrappers._lasso', '_solve_lasso_path_gpu_fista_multi_fold_from_gram_torch'):
        (('grad', 'coef_candidate', 'coefs'),),
    ('statgpu.linear_model.penalized._fit_mixin', '_fit_gpu_backend'):
        (('xtx_y', 'coef'), ('grad', 'coef')),
    ('statgpu.linear_model.penalized._fit_mixin', '_solve_exact_cupy'):
        (('A', '_returned_solution'),),
    ('statgpu.linear_model.penalized._fit_mixin', '_solve_exact_torch'):
        (('A', '_returned_solution'),),
    ('statgpu.linear_model.wrappers._logistic', '_fit_gpu'):
        (('XtWX', 'Xtz', 'params'),),
    ('statgpu.linear_model.wrappers._logistic', '_fit_torch'):
        (('XtWX', 'Xtz', 'params'),),
    ('statgpu.linear_model.penalized._penalized_cv', '_ridge_eig_single_backend'):
        (('gram', 'cross', 'eigvals', 'coef'),),
    ('statgpu.solvers._newton', 'newton_solver'):
        (('grad', 'hess', 'params'),),
    ('statgpu.solvers._fista', 'fista_solver'):
        (('grad', 'coef'),),
    ('statgpu.solvers._fista_bb', 'fista_bb_solver'):
        (('grad', 'coef'),),
    ('statgpu.losses._cox_ph', '_full_objective_from_eta_backend'):
        (('score', 'information', 'weighted_second'),),
}


def _numerical_intermediates(frame, event, result):
    key = (str(frame.f_globals.get('__name__', '')), frame.f_code.co_name)
    groups = _NUMERICAL_GROUPS.get(key, ())
    if not groups or event == 'call':
        return {}
    values = dict(frame.f_locals)
    if event == 'return' and result is not None:
        values['_returned_solution'] = result
    for names in groups:
        if all(name in values for name in names):
            return {name: values[name] for name in names}
    return {}


def _cpu_numerical_dispatch(frame):
    module = str(frame.f_globals.get('__name__', ''))
    name = frame.f_code.co_name
    if module.startswith('statgpu.'):
        return (name == '_fit_cpu' or name == '_solve_exact_numpy'
                or (module == 'statgpu.linear_model.penalized._penalized_cv'
                    and name in {'_ridge_eig_single', '_ridge_eig_batch'})
                or (module == 'statgpu.linear_model.wrappers._lasso'
                    and name.startswith('_solve_lasso_path_cpu')))
    return (module.startswith(('numpy.linalg', 'scipy.linalg'))
            and name in {'solve', 'lstsq', 'pinv', 'pinvh', 'inv', 'eigh',
                         'eigvalsh', 'svd', 'cholesky', 'cho_solve', 'solve_triangular'})


@contextmanager
def observed_backend(backend):
    """Observe source-specific native computations in diagnostic CV regions only.

    This is diagnostic instrumentation, not a proof from device-resident input
    or a model label. Intermediate/C-call events are necessary: some optimized
    solvers delete numerical temporaries before their Python return event.
    """
    if backend == 'numpy':
        yield []
        return
    if backend == 'cupy':
        import cupy as xp
        context = xp.cuda.Device(0)
        synchronize = lambda: xp.cuda.Device(0).synchronize()
        def identity(value):
            return int(value.device.id) if isinstance(value, xp.ndarray) else None
    else:
        import torch as xp
        context = xp.cuda.device(0)
        synchronize = lambda: xp.cuda.synchronize(device=0)
        def identity(value):
            return int(value.device.index) if isinstance(value, xp.Tensor) and value.is_cuda else None
    original_profiler, original_sync = legacy.RegionProfiler, legacy._synchronizer
    proofs = []

    class ObservedProfiler(original_profiler):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.observed = {'selector': {}, 'final_refit': {}}
            self.failure = None
            self.rejected_observations = []

        def _observe(self, frame, event, result):
            values = _numerical_intermediates(frame, event, result)
            if not values:
                return
            call = str(frame.f_globals.get('__name__', '')) + '.' + frame.f_code.co_name
            devices = {name: identity(value) for name, value in values.items()}
            dtypes = {name: str(getattr(value, 'dtype', 'unknown')).replace('torch.', '')
                      for name, value in values.items()}
            error = None
            if any(device is None for device in devices.values()):
                error = f'CPU/non-native numerical intermediate in requested CUDA CV region: {call}'
            elif any(device != 0 for device in devices.values()):
                error = f'CV numerical array executed on unexpected CUDA device: {call}'
            elif any(dtype != 'float64' for dtype in dtypes.values()):
                error = f'CV float64 baseline executed a different compute dtype: {call}: {dtypes}'
            if error:
                self.rejected_observations.append({
                    'call': call, 'devices': devices, 'computed_array_dtypes': dtypes,
                    'stage': 'selector' if self.selector_stack else 'final_refit',
                    'reason': error,
                })
                raise RuntimeError(error)
            observation = {'device': 'cuda:0', 'computed_array_dtypes': dtypes}
            if self.selector_stack:
                self.observed['selector'][call] = observation
            # A nested full-row fit during selection is not final-refit proof.
            elif self.refit_stack:
                self.observed['final_refit'][call] = observation

        def _profile(self, frame, event, arg):
            if self.selector_stack or self.refit_stack:
                try:
                    if event == 'call' and _cpu_numerical_dispatch(frame):
                        raise RuntimeError('CPU fit dispatch/numerical solve encountered inside requested CUDA CV region: '
                                           + str(frame.f_globals.get('__name__', '')) + '.' + frame.f_code.co_name)
                    self._observe(frame, event, arg)
                    # A Python child call can expose the parent's freshly
                    # computed temporaries before cleanup (notably Logistic).
                    caller = getattr(frame, 'f_back', None)
                    if caller is not None and event in {'call', 'return'}:
                        self._observe(caller, 'intermediate', None)
                except Exception as exc:
                    self.failure = str(exc)
                    raise
            return super()._profile(frame, event, arg)

        def __exit__(self, typ, value, tb):
            result = super().__exit__(typ, value, tb)
            complete = all(self.observed.values())
            error = self.failure or (str(value) if value is not None else None)
            if typ is None and not complete:
                error = 'Cannot establish native numerical-array execution in both CV selector and final refit'
            proofs.append({
                'model': type(self.outer_estimator).__name__, 'backend': backend,
                'device': 'cuda:0', 'synchronized_device': 'cuda:0',
                'status': 'success' if typ is None and complete and not error else 'failed',
                'reason': error, 'invocation': getattr(self, 'invocation', None),
                'input_dtypes': getattr(self, 'input_dtypes', None),
                'workload_identity': getattr(self, 'workload_identity', None),
                'observed_numerical_calls': {k: sorted(v) for k, v in self.observed.items()},
                'numerical_observations': self.observed,
                'rejected_numerical_observations': self.rejected_observations,
                'stage_observed': {k: bool(v) for k, v in self.observed.items()},
                'stage_active_at_failure': {'selector': bool(self.selector_stack),
                                            'final_refit': bool(self.refit_stack)},
                'exception_chain': (''.join(traceback.format_exception(typ, value, tb))
                                    if typ is not None else error),
            })
            if typ is None and error:
                raise RuntimeError(error)
            return result

    try:
        legacy.RegionProfiler = ObservedProfiler
        legacy._synchronizer = lambda requested: synchronize if requested == backend else original_sync(requested)
        with context:
            yield proofs
    finally:
        legacy.RegionProfiler = original_profiler
        legacy._synchronizer = original_sync


def _source_key(source):
    return {key: source[key] for key in ("git_commit", "content_sha256")}


def _matched_execution_device(backend, diagnostics, proofs):
    """Bind untimed native-array proof to the exact clean-repeat workload."""
    if backend == "numpy":
        return "cpu"
    matching = [proof for proof in proofs
                if (proof.get("invocation") or {}).get("kind") == "repeat"]
    if len(matching) != len(diagnostics):
        raise RuntimeError("Clean CV timing requires one native proof per diagnostic repeat")
    devices = set()
    for diagnostic, proof in zip(diagnostics, matching):
        if (proof.get("status") != "success" or proof.get("backend") != backend
                or proof.get("workload_identity") != diagnostic["workload_identity"]):
            raise RuntimeError("Native proof does not match diagnostic CV workload")
        device = proof.get("device")
        if (not isinstance(device, str) or not device.startswith("cuda:")
                or not device[5:].isdigit() or proof.get("synchronized_device") != device):
            raise RuntimeError("Native proof lacks concrete synchronized executed device")
        for stage in ("selector", "final_refit"):
            observations = (proof.get("numerical_observations") or {}).get(stage, {})
            if not observations or not (proof.get("stage_observed") or {}).get(stage):
                raise RuntimeError("Native proof does not cover both CV stages")
            for observation in observations.values():
                dtypes = observation.get("computed_array_dtypes", {})
                if (observation.get("device") != device or not dtypes
                        or any(dtype != "float64" for dtype in dtypes.values())):
                    raise RuntimeError("Native CV proof device/compute dtype mismatch")
        devices.add(device)
    if len(devices) != 1:
        raise RuntimeError("Native diagnostic CV repeats changed executed device")
    return devices.pop()


def clean_public_timing(spec, backend, args, n, diagnostics, proofs, source):
    """A separate synchronized E2E series, never selector time by subtraction."""
    section = {
        "status": "failed", "reason": None, "timing": None,
        "scope": "public-fit-E2E; device-resident inputs; constructor/transfers/scoring excluded",
        "instrumented": False, "repeats": args.repeats, "warmup": args.warmup,
        "statistic": "median", "raw_seconds": [], "repeat_samples": [],
        "performance_eligible": False, "performance_status": "UNVERIFIED_EXCLUSIVITY",
        "comparisons": [],
        "provenance_basis": "separate matched diagnostic execution, not instrumentation during clean fit",
    }
    try:
        if len(diagnostics) != args.repeats:
            raise RuntimeError("Missing complete diagnostic CV repeat series")
        executed_device = _matched_execution_device(backend, diagnostics, proofs)
        section["executed_device"] = executed_device
        section["synchronized_device"] = None if backend == "numpy" else executed_device
        expected_source = _source_key(source)
        def check_source():
            current = _source_key(source_identity())
            if current != expected_source:
                raise RuntimeError("Source changed between diagnostic and clean CV execution")
            return current
        with warnings.catch_warnings():
            convergence_errors()
            for _ in range(args.warmup):
                legacy._run_clean_once(spec, backend, args.seed, n, args.features,
                                       diagnostic=diagnostics[0], executed_device=executed_device,
                                       check_source=check_source)
            samples = [legacy._run_clean_once(
                spec, backend, args.seed, n, args.features,
                diagnostic=diagnostic, executed_device=executed_device, check_source=check_source,
            ) for diagnostic in diagnostics]
        seconds = [sample["total_fit_ms"] / 1000.0 for sample in samples]
        section.update(status="pass", raw_seconds=seconds, repeat_samples=samples,
                       timing={"median_seconds": float(np.median(seconds))},
                       source=expected_source,
                       workload_identity=diagnostics[0]["workload_identity"],
                       correctness={"status": "pass", "reference": "matched diagnostic public fit",
                                    "selection": "identical", "atol": 2e-6, "rtol": 2e-5,
                                    "checks": ["validation_score", "final_score", "final_refit_coefficients", "coefficient_dtype"]})
    except Exception as exc:
        section["reason"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return section


def _invalidate_clean(section, reason):
    """Never retain usable timing from a failed/partial correctness series."""
    section.update(status="failed", reason=reason, timing=None, raw_seconds=[], repeat_samples=[],
                   performance_eligible=False)


def run(args):
    if Path(statgpu.__file__).resolve().parent != ROOT/'statgpu':
        raise RuntimeError('statgpu imported from outside benchmark checkout')
    before=source_identity()
    results=[]
    for n in args.sizes:
        for backend in args.backends:
            available,_,reason=legacy._backend_status(backend)
            if not available:
                for spec in legacy.CASE_SPECS:
                    results.append({'model_id':spec.model_id,'backend':backend,'n_samples':n,
                        'status':'unavailable','reason':reason,'timing':None})
                continue
            diagnostic_samples = []
            options=SimpleNamespace(backends=backend,seed=args.seed,repeats=args.repeats,warmup=args.warmup,
                diagnostic_samples=diagnostic_samples,
                n_samples=n,n_features=args.features,include_sklearn=True,env_id='scoped-cv-candidate')
            proofs = []
            try:
                with warnings.catch_warnings(), observed_backend(backend) as proofs:
                    convergence_errors()
                    source=legacy.build_source(options)
                validate_scoped_source(source,[backend],args.repeats)
            except Exception as exc:
                for spec in legacy.CASE_SPECS:
                    results.append({'model_id':spec.model_id,'backend':backend,'n_samples':n,
                        'status':'failed','reason':'Source construction/validation: ' + ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)), 'timing':None,
                        'execution_provenance':[proof for proof in proofs if proof['model']==spec.import_name]})
                continue
            source['protocol']['instrumentation']='DIAGNOSTIC ONLY: RegionProfiler/source-specific numerical-array proof runs separately from clean_public_fit. Profiling overhead INCLUDED in compatibility stage timings; never use these stage/total durations as clean performance samples.'
            source['protocol']['precision_policy']='Explicit float64 accuracy baseline: gpu_cv_mixed_precision=False where exposed. Input and actual computed intermediate dtypes recorded separately in CUDA execution provenance; NumPy compute dtype is not runtime-instrumented.'
            specs={spec.model_id:spec for spec in legacy.CASE_SPECS}
            for case in source['cases']:
                spec=specs[case['model_id']]
                row={'model_id':spec.model_id,'backend':backend,'n_samples':n,'n_features':args.features,
                     'case':case,'environment':source['environment'],'protocol':source['protocol'],
                     'execution_provenance':[proof for proof in proofs if proof['model']==spec.import_name] if backend!='numpy' else {'device':'cpu','synchronized_device':None},
                     'performance_valid':False, 'performance_status':'UNVERIFIED_EXCLUSIVITY',
                     'diagnostic_timing':{'instrumented':True,'eligible_for_comparison':False}}
                try:
                    if any(r['status']!='success' for r in case['runs']):
                        raise ValueError('A required measured/reference run did not succeed: ' + '; '.join(
                            f"{r['framework']}/{r.get('backend')}: {r.get('reason')}"
                            for r in case['runs'] if r['status'] != 'success'))
                    row['correctness']={'direct_refit':direct_refit_check(spec,args.seed,n,args.features),
                                        'external':check_external_alignment(case)}
                    native=next(r for r in case['runs'] if r['framework']=='statgpu')
                    ref=row['correctness']['direct_refit']
                    if native['selected_parameters']!=ref['selected_parameters']:
                        raise ValueError('Measured CV selection differs from independently rerun CPU reference')
                    np.testing.assert_allclose(native['scores']['final_score'],ref['reference_final_score'],rtol=3e-3,atol=3e-4)
                    np.testing.assert_allclose(native['scores']['validation_score'],ref['reference_validation_score'],rtol=3e-3,atol=3e-4)
                    diagnostics = [record['sample'] for record in diagnostic_samples
                                   if record['model_id'] == spec.model_id and record['backend'] == backend]
                    row['clean_public_fit'] = clean_public_timing(
                        spec, backend, args, n, diagnostics,
                        [proof for proof in proofs if proof['model'] == spec.import_name], before)
                    if row['clean_public_fit']['status'] != 'pass':
                        raise RuntimeError('Clean public fit: ' + row['clean_public_fit']['reason'])
                    row['status']='pass'
                except Exception as exc:
                    row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                    for measured in case['runs']:
                        measured['timing']=None
                        measured['repeat_samples']=[]
                    if 'clean_public_fit' not in row:
                        row['clean_public_fit'] = {'status':'not_run', 'reason':'Required diagnostic/reference correctness failed',
                            'timing':None, 'raw_seconds':[], 'repeat_samples':[], 'instrumented':False,
                            'performance_eligible':False}
                    row['performance_valid']=False
                results.append(row)
    after=source_identity()
    changed=before['git_commit']!=after['git_commit'] or before['content_sha256']!=after['content_sha256']
    if changed:
        for row in results:
            if 'clean_public_fit' in row:
                _invalidate_clean(row['clean_public_fit'], 'Source changed during benchmark execution')
            if row['status'] == 'pass':
                row.update(status='failed', reason='Source changed during benchmark execution')
                for measured in row.get('case', {}).get('runs', []):
                    measured['timing'] = None
                    measured['repeat_samples'] = []
    failed=changed or any(r['status']=='failed' for r in results)
    partial=any(r['status'] in {'unavailable','blocked'} for r in results)
    return {'schema_version':'statgpu-scoped-cv-candidate-v2','source':before,'source_after':after,
            'benchmark':{'name':'six-family-scoped-cv','command':[sys.executable,*sys.argv]},
            'coverage':{'active_models':sorted(EXPECTED_MODELS),'fold_count':3,'dtype':'float64',
                        'grids':{s.model_id:s.grid_parameters for s in legacy.CASE_SPECS},
                        'exhaustive_loss_penalty_cv':False},
            'results':results,'validation':{'status':'failed' if failed else 'partial' if partial else 'complete',
            'tier':'CPU-harness-smoke' if args.smoke else 'benchmark-candidate','canonical_ready':False,
            'limitations':['Not canonical cv_source schema; unrequested backend rows are not fabricated.',
                'Clean E2E repeats are separate from profiled diagnostic stage timings; no selector/refit clean decomposition is claimed.',
                'All timing remains diagnostic unless hardware exclusivity is independently verified; no speedup comparisons emitted.',
                'Three folds/fixed grids/explicit float64 accuracy baseline (not mixed-precision defaults); sizes configurable; no complete grid-fold-weight sweep.',
                'Cox and PenalizedGLM use fresh direct-refit invariants, not independent external statistical certification.',
                'CUDA candidates fail if native numerical arrays are not observed in both selector/refit regions; only concrete device 0 supported. No physical CUDA run claimed.'],
            'source_changed':changed,
            'performance_eligible':False, 'performance_status':'UNVERIFIED_EXCLUSIVITY',
            'comparisons':[]}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backends',nargs='+',choices=['numpy','cupy','torch'],default=['numpy','cupy','torch'])
    p.add_argument('--sizes',nargs='+',type=int,default=[240,1024])
    p.add_argument('--features',type=int,default=16)
    p.add_argument('--seed',type=int,default=20260807)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--require-backends',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.smoke:
        args.backends,args.sizes,args.features,args.repeats,args.warmup=['numpy'],[60],4,1,0
    if min(args.sizes)<30 or args.features<1 or args.repeats<1 or args.warmup<0 or args.features>=min(args.sizes):
        p.error('Require n>=30, 1<=features<n, repeats>=1, warmup>=0')
    result=run(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    status=result['validation']['status']
    print(json.dumps({'status':status,'rows':len(result['results']),'output':str(args.output)}))
    return 2 if status=='failed' or (args.require_backends and status=='partial') else 0


if __name__=='__main__':
    raise SystemExit(main())
