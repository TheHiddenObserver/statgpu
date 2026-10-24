#!/usr/bin/env python3
"""Correctness-gated diagnostics, selection, multi-target and formula benchmarks.

Run from repository root: python -m dev.benchmarks.benchmark_workflow_coverage
--smoke --output results/workflow-smoke.json. This produces candidate evidence,
not a canonical dashboard source. Diagnostics execute on CPU even when the fitted
model was on CUDA. Formula preprocessing executes on CPU. All times are public
API scopes, not pure kernels; fit instrumentation and internal conversions remain
included. External references are correctness oracles, not timed competitors.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import warnings
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.benchmarks import benchmark_release_refresh as common
from statgpu.diagnostics import RegressionDiagnostics
from statgpu.feature_selection import StepwiseSelector

CASES = ('diagnostics', 'stepwise', 'multitarget', 'formula')


def source_identity():
    return common.source_identity(extra_paths=(Path(__file__),))


def close(actual, expected, label, atol=2e-7, rtol=2e-7):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f'{label}: nonfinite or mismatched shapes {actual.shape}/{expected.shape}')
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f'{label}: max absolute error {np.max(np.abs(actual-expected))}')
    return float(np.max(np.abs(actual-expected))) if actual.size else 0.0


def measure(call, check, runtime, warmup, repeats, scope):
    # Correctness precedes timing even with zero configured warmups.
    check(call())
    for _ in range(warmup):
        check(call())
    samples = []
    last = None
    for _ in range(repeats):
        last, duration = common.time_call(call, runtime, runtime.device)
        check(last)
        samples.append(duration)
    host = scope.startswith(('host-', 'fresh-host-', 'cached-host-'))
    placement = 'host-cpu' if host else 'mixed-host-device' if scope.startswith(('formula-', 'complete-stepwise-')) else 'selected-backend-with-host-reporting'
    return last, {'scope': scope, 'compute_placement': placement,
                  'instrumented': ('fit' in scope or 'stepwise' in scope),
                  'instrumentation': 'fit ingress observation included when fitting; not whole-computation proof',
                  'correctness_preflight_calls': 1, 'raw_seconds': samples, 'median_seconds': statistics.median(samples),
                  'warmup': warmup, 'repeats': repeats, 'synchronized_device': runtime.device}


def model(runtime, *, inference=False, intercept=True):
    return common.ObservedOLS(device={'numpy':'cpu','cupy':'cuda','torch':'torch'}[runtime.name],
                              compute_inference=inference, fit_intercept=intercept)


def fixture(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p); beta[:min(2,p)] = [2.0, -1.0][:min(2,p)]
    y = 0.5 + X @ beta + rng.normal(scale=0.5, size=n)
    return X, y


def require_module(name):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise common.BackendUnavailable(f'Optional reference dependency {name} unavailable: {exc}') from exc


def diagnostics(runtime, X, y, args):
    sm = require_module('statsmodels.api')
    outliers = require_module('statsmodels.stats.outliers_influence')
    D = np.column_stack((np.ones(len(y)), X))
    fit = model(runtime, inference=True).fit(runtime.array(X), runtime.array(y))
    common.verify_route(fit, runtime)
    ref = sm.OLS(y, D).fit(); influence = ref.get_influence()
    expected = {'residuals': ref.resid, 'fitted_values': ref.fittedvalues,
                'standardized_residuals': ref.resid/np.sqrt(ref.scale),
                'studentized_residuals': influence.resid_studentized_internal,
                'externally_studentized_residuals': influence.resid_studentized_external,
                'leverage': influence.hat_matrix_diag, 'cooks_distance': influence.cooks_distance[0],
                'vif': np.array([outliers.variance_inflation_factor(D,j) for j in range(1,D.shape[1])])}
    errors = {}
    def snapshot_check(diag):
        errors['snapshot_residuals'] = close(diag.residuals, ref.resid, 'snapshot residuals')
    _, snapshot_t = measure(lambda: RegressionDiagnostics(fit), snapshot_check, runtime,
                            args.warmup, args.repeats, 'host-diagnostic-snapshot-including-device-to-host-if-needed')
    def compute():
        diag = RegressionDiagnostics(fit)
        return {k: diag.vif() if k=='vif' else getattr(diag,k) for k in expected}
    def check(value):
        for k in expected: errors[k] = close(value[k],expected[k], k)
    _, operations_t = measure(compute, check, runtime, args.warmup,args.repeats,
                              'fresh-host-snapshot-plus-all-diagnostic-operations')
    cached = RegressionDiagnostics(fit); cached.leverage
    _, cache_t = measure(lambda: cached.leverage, lambda v: close(v,expected['leverage'],'cached leverage'),
                         runtime,args.warmup,args.repeats,'cached-host-leverage-access')
    return {'correctness': {'reference':'statsmodels.OLSInfluence and variance_inflation_factor',
                            'metrics':errors,'status':'pass'},
            'execution': {'fit_backend':runtime.name,'diagnostics_backend':'numpy-cpu',
                          'not_gpu_native':True}, 'timing': [snapshot_t,operations_t,cache_t]}


def gaussian_score(X, y, subset):
    D = np.column_stack((np.ones(len(y)), X[:,list(subset)]))
    beta = np.linalg.lstsq(D,y,rcond=None)[0]
    sse = np.sum((y-D@beta)**2); n=len(y); k=np.linalg.matrix_rank(D)
    minus2ll = n*(np.log(2*np.pi)+1+np.log(sse/n))
    return {'aic':float(minus2ll+2*k),'bic':float(minus2ll+k*np.log(n))}


def selection_reference(X,y,direction,criterion):
    """Independent host least-squares scoring of every visited proposal."""
    selected = tuple(range(X.shape[1])) if direction=='backward' else ()
    history=[(selected,gaussian_score(X,y,selected))]
    while True:
        candidates=[]
        if direction in ('forward','both'):
            for j in set(range(X.shape[1]))-set(selected):
                s=tuple(sorted((*selected,j)));candidates.append((gaussian_score(X,y,s)[criterion],1,j,s))
        if direction in ('backward','both'):
            for j in selected:
                s=tuple(k for k in selected if k!=j);candidates.append((gaussian_score(X,y,s)[criterion],0,j,s))
        if not candidates: break
        score,_,_,s=min(candidates)
        current=history[-1][1][criterion]
        if not score < current-1e-12*max(1,abs(current)): break
        selected=s;history.append((selected,gaussian_score(X,y,selected)))
    return history


def stepwise(runtime,X,y,args):
    dx,dy=runtime.array(X),runtime.array(y)
    times=[];checks=[]
    for direction in ('forward','backward','both'):
        for criterion in ('aic','bic'):
            reference=selection_reference(X,y,direction,criterion)
            execution=[]
            class CheckedOLS(common.ObservedOLS):
                def fit(self,*a,**kw):
                    result=super().fit(*a,**kw)
                    common.verify_route(self,runtime)
                    execution.append(self.benchmark_fit_ingress)
                    return result
            def fit():
                return StepwiseSelector(CheckedOLS,criterion=criterion,direction=direction,n_jobs=1,
                    device={'numpy':'cpu','cupy':'cuda','torch':'torch'}[runtime.name],compute_inference=True).fit(dx,dy)
            def check(selector):
                actual=selector.selection_history_
                if len(actual)!=len(reference):raise AssertionError('selection path length differs')
                for a,(features,scores) in zip(actual,reference):
                    if tuple(a['features'])!=features:raise AssertionError('selected feature path differs')
                    close([a['aic'],a['bic']],[scores['aic'],scores['bic']],'selection AIC/BIC')
                if tuple(selector.selected_features_)!=reference[-1][0]:raise AssertionError('selected set differs')
                subset=list(reference[-1][0]);D=np.column_stack((np.ones(len(y)),X[:,subset]))
                expected=D@np.linalg.lstsq(D,y,rcond=None)[0]
                predicted=selector.predict(dx)
                if common.array_identity(predicted)['device']!=runtime.device:raise AssertionError('selection prediction on wrong device')
                close(runtime.host(predicted),expected,'selection final-refit prediction')
            selected,t=measure(fit,check,runtime,args.warmup,args.repeats,'complete-stepwise-search-with-candidate-inference-and-final-refit')
            t.update(direction=direction,criterion=criterion);times.append(t)
            checks.append({'direction':direction,'criterion':criterion,'selected_features':selected.selected_features_,
                           'accepted_steps':len(selected.selection_history_),'candidate_cache_size':len(selected._score_cache),
                           'observed_fit_calls_all_repetitions':len(execution)})
    return {'correctness':{'reference':'independent NumPy lstsq Gaussian loglikelihood greedy candidate path',
                           'status':'pass','selection':checks},'timing':times,
            'limitations':['Finite-score fixture only; does not certify known nonfinite-score stopping/cap limitations',
                           'Selection is not exhaustive best-subset or FDR control; no post-selection inference claim']}


def multitarget(runtime,X,y,args):
    rng=np.random.default_rng(args.seed+1)
    Y=y[:,None]+X@rng.normal(size=(X.shape[1],args.targets))
    query=rng.normal(size=(max(8,len(y)//4),X.shape[1]))
    D=np.column_stack((np.ones(len(y)),X));params=np.linalg.lstsq(D,Y,rcond=None)[0]
    expected=np.column_stack((np.ones(len(query)),query))@params
    dx,dy,dq=runtime.array(X),runtime.array(Y),runtime.array(query)
    errors={}
    def check(fit):
        common.verify_route(fit,runtime)
        errors['coefficients']=close(fit.coef_,params[1:].T,'multi-target coefficients')
        errors['intercepts']=close(np.atleast_1d(fit.intercept_),params[0],'multi-target intercept')
    fit,fit_t=measure(lambda:model(runtime).fit(dx,dy),check,runtime,args.warmup,args.repeats,'multi-target-fit-only-public-api')
    def prediction_check(value):
        if common.array_identity(value)['device']!=runtime.device:raise AssertionError('prediction executed on wrong device')
        errors['prediction']=close(runtime.host(value),expected,'multi-target prediction')
    _,pred_t=measure(lambda:fit.predict(dq),prediction_check,runtime,args.warmup,args.repeats,'multi-target-predict-only-public-api')
    return {'correctness':{'reference':'NumPy float64 augmented-design least squares','status':'pass','metrics':errors},
            'timing':[fit_t,pred_t],'problem_extra':{'n_targets':args.targets,'target_query_seed':args.seed+1,'targets_query_sha256':hashlib.sha256(Y.tobytes()+query.tobytes()).hexdigest()},
            'limitations':['Inference intentionally disabled: GPU multi-output inference is unsupported']}


def formula(runtime,X,y,args):
    pd=require_module('pandas');patsy=require_module('patsy')
    from statgpu.core.formula import FormulaParser
    frame=pd.DataFrame({'y':y,'x':X[:,0],'z':X[:,1], 'group':np.where(np.arange(len(y))%2,'b','a')})
    formula_text='y ~ x + z + C(group) + x:z'
    # A missing predictor exercises retained-row identity, not silent row drift.
    frame.loc[3,'z']=np.nan
    ry,rX=patsy.dmatrices(formula_text,frame,return_type='dataframe')
    refy=np.asarray(ry).ravel();refX=np.asarray(rX);params=np.linalg.lstsq(refX,refy,rcond=None)[0]
    errors={}
    def parse():
        parser=FormulaParser(formula_text); response,design,info=parser.eval(frame)
        return parser,response,design,info
    def check_parse(value):
        parser,response,design,info=value
        errors['design']=close(design,refX,'formula design')
        errors['response']=close(np.asarray(response).ravel(),refy,'formula response')
        if list(info.column_names)!=list(rX.columns):raise AssertionError('formula column order differs')
        if list(parser.row_positions)!=list(rX.index):raise AssertionError('formula row positions differ')
    cpu=common.Runtime('numpy')
    _,parse_t=measure(parse,check_parse,cpu,args.warmup,args.repeats,'host-formula-model-matrix-preparation-only')
    def check_fit(fit):
        common.verify_route(fit,runtime)
        beta=np.r_[fit.intercept_,fit.coef_]
        errors['params']=close(beta,params,'formula fit parameters')
    fitted,fit_t=measure(lambda:model(runtime).fit(formula=formula_text,data=frame),check_fit,runtime,
                        args.warmup,args.repeats,'formula-parse-plus-fit-including-required-host-to-device-transfer')
    dx,dy=runtime.array(refX[:,1:]),runtime.array(refy)
    baseline,baseline_t=measure(lambda:model(runtime).fit(dx,dy),check_fit,runtime,args.warmup,args.repeats,'resident-array-fit-only-same-formula-design')
    errors['formula-array-parity']=close(np.r_[fitted.intercept_,fitted.coef_],np.r_[baseline.intercept_,baseline.coef_],'formula/array parity')
    return {'correctness':{'reference':'patsy design/row identity and NumPy least squares','status':'pass','metrics':errors},
            'timing':[parse_t,fit_t,baseline_t],'formula':{'formula':formula_text,'columns':list(rX.columns),
                 'retained_rows':list(map(int,rX.index)),'missing_rows':[3]},
            'limitations':['One categorical/interactions/missing-data design; not exhaustive formula syntax certification']}


IMPLEMENTATIONS=dict(diagnostics=diagnostics,stepwise=stepwise,multitarget=multitarget,formula=formula)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--backends',nargs='+',choices=['numpy','cupy','torch'],default=['numpy','cupy','torch'])
    p.add_argument('--sizes',nargs='+',type=int,default=[256,1024])
    p.add_argument('--features',type=int,default=6);p.add_argument('--targets',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261020);p.add_argument('--cuda-device',type=int,default=0)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--require-backends',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.smoke:args.backends,args.sizes,args.features,args.targets,args.warmup,args.repeats=['numpy'],[48],4,3,0,1
    if min(args.sizes)<=args.features+3 or args.features<2 or args.targets<1 or args.warmup<0 or args.repeats<1 or args.cuda_device<0:p.error('Require n>p+3, p>=2, targets>=1, nonnegative warmup/device and positive repeats')
    if Path(sys.modules['statgpu'].__file__).resolve().parent!=ROOT/'statgpu':raise RuntimeError('Wrong statgpu import source')
    source=source_identity(); rows=[]; env=common.environment();env['optional_packages']={}
    for name in ['statsmodels','pandas','patsy']:
        try:env['optional_packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:env['optional_packages'][name]=None
    for backend in args.backends:
        try:runtime=common.Runtime(backend,args.cuda_device)
        except common.BackendUnavailable as exc:
            for case in args.cases:rows.append({'case':case,'backend':backend,'status':'unavailable','reason':str(exc)})
            continue
        except Exception as exc:
            rows.append({'backend':backend,'status':'failed','reason':str(exc)});continue
        for n in args.sizes:
            X,y=fixture(n,args.features,args.seed)
            for case in args.cases:
                row={'case':case,'backend':backend,'executed_device':runtime.device,'hardware':runtime.hardware,
                     'problem':{'n':n,'p':args.features,'dtype':'float64','seed':args.seed,
                        'data_id':hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()}}
                try:
                    with runtime.context(),warnings.catch_warnings():
                        warnings.simplefilter('error')
                        row.update(IMPLEMENTATIONS[case](runtime,X,y,args))
                    row['correctness']['thresholds']={'atol':2e-7,'rtol':2e-7,'source':'workflow_coverage v1 finite full-rank Gaussian fixture; not universal tolerance'}
                    row['status']='passed'
                except common.BackendUnavailable as exc:row.update(status='unavailable',reason=str(exc))
                except Exception as exc:row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                rows.append(row)
    after=source_identity()
    if after['worktree_fingerprint']!=source['worktree_fingerprint']:rows.append({'status':'failed','reason':'source changed during execution'})
    missing=any(r['status']=='unavailable' for r in rows)
    failed=any(r['status']=='failed' for r in rows) or (args.require_backends and missing) or not any(r['status']=='passed' for r in rows)
    status='failed' if failed else ('partial' if missing else 'complete')
    env['environment_id']=hashlib.sha256(json.dumps({'software_host':env,'devices':[{k:r.get(k) for k in ('backend','executed_device','hardware')} for r in rows]},sort_keys=True).encode()).hexdigest()
    result={'schema_version':'statgpu-benchmark-v2','benchmark':{'name':'workflow-coverage','command':[sys.executable,*sys.argv],
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},'source':source,'source_after':after,
            'environment':env,'rows':rows,'validation':{'status':status,'tier':'local-minimal' if args.smoke else 'candidate-benchmark',
            'canonical_dashboard_registered':False,'limitations':['No speedup ratios or external timing','No physical CUDA certification from CPU smoke',
            'Finite full-rank fixtures only; broad operations, not exhaustive statistical/backend product']}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'{status}: {len(rows)} rows; {args.output}')
    return 2 if failed else 0

if __name__=='__main__':raise SystemExit(main())
