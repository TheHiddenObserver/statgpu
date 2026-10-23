#!/usr/bin/env python3
"""Scoped six-family CV candidate benchmark, reusing the maintained CV runner.

Unlike the immutable canonical-source schema, this adapter permits a CPU-only
run. It never registers the output as canonical. CUDA timings require runtime observations of native numerical arrays inside both
selector and final-refit regions on concrete device 0. Missing proof fails the
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
    with warnings.catch_warnings():
        convergence_errors()
        model.fit(*legacy._fit_args(spec,X,y))
        final=model.estimator_
        fresh=legacy._construct(type(final), final.get_params(deep=False))
        fresh.fit(*legacy._fit_args(spec,X,y))
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


@contextmanager
def observed_backend(backend):
    """Observe, but never modify, numerical arrays in the existing CV regions."""
    if backend == 'numpy':
        yield []
        return
    if backend == 'cupy':
        import cupy as xp
        context=xp.cuda.Device(0)
        synchronize=lambda: xp.cuda.Device(0).synchronize()
        def identity(value):
            return int(value.device.id) if isinstance(value,xp.ndarray) else None
    else:
        import torch as xp
        context=xp.cuda.device(0)
        synchronize=lambda: xp.cuda.synchronize(device=0)
        def identity(value):
            return int(value.device.index) if isinstance(value,xp.Tensor) and value.is_cuda else None
    original_profiler,original_sync=legacy.RegionProfiler,legacy._synchronizer
    proofs=[]
    class ObservedProfiler(original_profiler):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.observed={'selector':set(),'final_refit':set()}
        def _profile(self,frame,event,arg):
            if event=='call':
                module=str(frame.f_globals.get('__name__',''))
                name=frame.f_code.co_name
                if module.startswith('statgpu.') and name == '_fit_cpu' and (self.selector_stack or self.refit_stack):
                    raise RuntimeError('CPU fit dispatch encountered inside requested CUDA CV region')
                if module.startswith('statgpu.solvers.') and name.endswith('_solver'):
                    design=frame.f_locals.get('X')
                    if isinstance(design,np.ndarray):
                        raise RuntimeError('CPU numerical solver design encountered inside requested CUDA CV region')
                numerical=(module.startswith(('statgpu.solvers.','statgpu.glm_core.'))
                           or (module.startswith('statgpu.') and name in {'_fit_gpu','_fit_torch'})
                           or (module.startswith('statgpu.survival.') and name not in {'fit','__init__'}))
                if numerical:
                    devices=[identity(v) for v in frame.f_locals.values()]
                    devices=[d for d in devices if d is not None]
                    if any(d != 0 for d in devices):
                        raise RuntimeError('CV numerical array executed on unexpected CUDA device')
                    if devices:
                        if self.selector_stack:self.observed['selector'].add(module+'.'+name)
                        if self.refit_stack:self.observed['final_refit'].add(module+'.'+name)
            return super()._profile(frame,event,arg)
        def __exit__(self,typ,value,tb):
            result=super().__exit__(typ,value,tb)
            if typ is None:
                if not all(self.observed.values()):
                    raise RuntimeError('Cannot establish native numerical-array execution in both CV selector and final refit')
                proofs.append({'model':type(self.outer_estimator).__name__,'backend':backend,'device':'cuda:0','synchronized_device':'cuda:0',
                               'observed_numerical_calls':{k:sorted(v) for k,v in self.observed.items()}})
            return result
    try:
        legacy.RegionProfiler=ObservedProfiler
        legacy._synchronizer=lambda requested: synchronize if requested==backend else original_sync(requested)
        with context:
            yield proofs
    finally:
        legacy.RegionProfiler=original_profiler
        legacy._synchronizer=original_sync


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
            options=SimpleNamespace(backends=backend,seed=args.seed,repeats=args.repeats,warmup=args.warmup,
                n_samples=n,n_features=args.features,include_sklearn=True,env_id='scoped-cv-candidate')
            try:
                with warnings.catch_warnings(), observed_backend(backend) as proofs:
                    convergence_errors()
                    source=legacy.build_source(options)
                validate_scoped_source(source,[backend],args.repeats)
            except Exception as exc:
                for spec in legacy.CASE_SPECS:
                    results.append({'model_id':spec.model_id,'backend':backend,'n_samples':n,
                        'status':'failed','reason':f'Source construction/validation: {type(exc).__name__}: {exc}','timing':None})
                continue
            source['protocol']['instrumentation']='RegionProfiler observes Python selector/refit boundaries; GPU runtime also inspects numerical arrays. Profiling overhead INCLUDED; diagnostic stage timings, not clean uninstrumented fits.'
            specs={spec.model_id:spec for spec in legacy.CASE_SPECS}
            for case in source['cases']:
                spec=specs[case['model_id']]
                row={'model_id':spec.model_id,'backend':backend,'n_samples':n,'n_features':args.features,
                     'case':case,'environment':source['environment'],'protocol':source['protocol'],
                     'execution_provenance':[proof for proof in proofs if proof['model']==spec.import_name] if backend!='numpy' else {'device':'cpu','synchronized_device':None}}
                try:
                    if any(r['status']!='success' for r in case['runs']):
                        raise ValueError('A required measured/reference run did not succeed')
                    row['correctness']={'direct_refit':direct_refit_check(spec,args.seed,n,args.features),
                                        'external':check_external_alignment(case)}
                    native=next(r for r in case['runs'] if r['framework']=='statgpu')
                    ref=row['correctness']['direct_refit']
                    if native['selected_parameters']!=ref['selected_parameters']:
                        raise ValueError('Measured CV selection differs from independently rerun CPU reference')
                    np.testing.assert_allclose(native['scores']['final_score'],ref['reference_final_score'],rtol=3e-3,atol=3e-4)
                    np.testing.assert_allclose(native['scores']['validation_score'],ref['reference_validation_score'],rtol=3e-3,atol=3e-4)
                    row['status']='pass'
                except Exception as exc:
                    row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                    for measured in case['runs']:
                        measured['timing']=None
                        measured['repeat_samples']=[]
                    row['performance_valid']=False
                results.append(row)
    after=source_identity()
    changed=before['git_commit']!=after['git_commit'] or before['content_sha256']!=after['content_sha256']
    failed=changed or any(r['status']=='failed' for r in results)
    partial=any(r['status'] in {'unavailable','blocked'} for r in results)
    return {'schema_version':'statgpu-scoped-cv-candidate-v1','source':before,'source_after':after,
            'benchmark':{'name':'six-family-scoped-cv','command':[sys.executable,*sys.argv]},
            'coverage':{'active_models':sorted(EXPECTED_MODELS),'fold_count':3,'dtype':'float64',
                        'grids':{s.model_id:s.grid_parameters for s in legacy.CASE_SPECS},
                        'exhaustive_loss_penalty_cv':False},
            'results':results,'validation':{'status':'failed' if failed else 'partial' if partial else 'complete',
            'tier':'CPU-harness-smoke' if args.smoke else 'benchmark-candidate','canonical_ready':False,
            'limitations':['Not canonical cv_source schema; unrequested backend rows are not fabricated.',
                'Three folds/fixed grids/float64; sizes configurable; no complete grid-fold-weight sweep.',
                'Cox and PenalizedGLM use fresh direct-refit invariants, not independent external statistical certification.',
                'CUDA candidates fail if native numerical arrays are not observed in both selector/refit regions; only concrete device 0 supported. No physical CUDA run claimed.'],
            'source_changed':changed}}


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
