#!/usr/bin/env python3
"""Nonparametric/learning benchmark adapters with explicit coverage boundaries.

CPU harness: python -m dev.benchmarks.benchmark_learning_coverage --smoke --output results/learning.json
No remote access or publication. This is not an exhaustive Cartesian matrix.
Invariant-only stochastic cases are diagnostic timings, never speedup evidence.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import statgpu
from dev.benchmarks.benchmark_release_refresh import Runtime, BackendUnavailable, array_identity, environment as base_environment

CASES = ('kde','kernel_regression_nw','kernel_regression_local_linear','kernel_ridge',
         'kernel_pca','nystroem','spline','gam','pca','incremental_pca','truncated_svd',
         'kmeans','minibatch_kmeans','dbscan','agglomerative','gaussian_mixture','nmf',
         'minibatch_nmf','tsne','umap','knockoff_fixed','knockoff_model','kernel_ridge_cv','kde_bootstrap',
         'bspline_basis','natural_spline','cyclic_spline','thin_plate_spline',
         'bandwidth_scott','bandwidth_silverman','bandwidth_nrd','bandwidth_nrd0',
         'bandwidth_ucv','bandwidth_bcv','bandwidth_sj','bandwidth_sj-ste','bandwidth_sj-dpi','bandwidth_cv','smoothing_helpers','bandwidth_helpers','pairwise_kernels','chi2_kernel','kde_normal_ci','umap_nndescent')
LEARNED = ('components_', 'cluster_centers_', 'embedding_', 'means_', 'dual_coef_',
           'labels_', 'coef_', 'covariances_', 'eigenvectors_')

from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, learning_stages, estimator_stages, exception_chain, identities

ESTIMATOR_TRACE_CASES = {'kde', 'kernel_regression_nw', 'kernel_regression_local_linear',
                         'spline', 'kernel_pca', 'nystroem', 'gam', 'agglomerative'}

TRACE_CASES = {c for c in CASES if c.startswith(('bandwidth', 'knockoff'))} | {'kde_bootstrap', 'kde_normal_ci'}

class Blocked(RuntimeError):
    pass


class UnsupportedCapability(Blocked):
    """A production support boundary, distinct from missing observation."""


def host(x):
    module = type(x).__module__.split('.')[0]
    if module == 'cupy':
        import cupy
        return cupy.asnumpy(x)
    if module == 'torch':
        return x.detach().cpu().numpy()
    return np.asarray(x)


def source_identity():
    paths = sorted((ROOT/'statgpu').rglob('*.py')) + [Path(__file__), ROOT/'dev/benchmarks/benchmark_release_refresh.py', ROOT/'dev/benchmarks/learning_execution_trace.py']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def git(*a):
        return subprocess.check_output(['git',*a],cwd=ROOT,stderr=subprocess.DEVNULL).decode().strip()
    try:
        sha=git('rev-parse','HEAD');dirty=bool(git('status','--porcelain'))
        runner_paths=[Path(__file__),ROOT/'dev/benchmarks/benchmark_release_refresh.py',ROOT/'dev/benchmarks/learning_execution_trace.py']
        tracked=subprocess.run(['git','ls-files','--error-unmatch',*[str(p.relative_to(ROOT)) for p in runner_paths]],cwd=ROOT,capture_output=True).returncode==0
    except (OSError,subprocess.CalledProcessError):
        sha=None;dirty=True;tracked=False
    return {'git_commit':sha,'working_tree':'dirty' if dirty or not tracked else 'clean',
            'runner_overlay':not tracked,'statgpu_version':statgpu.__version__,
            'source_content_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            'script_sha256':hashes[str(Path(__file__).relative_to(ROOT))],
            'fingerprint_scope':'all statgpu Python files, this runner and imported release_refresh/learning_execution_trace helpers',
            'import_path':str(Path(statgpu.__file__).resolve())}


def data(n,p,seed):
    rng=np.random.default_rng(seed)
    labels=np.arange(n)%2
    X=rng.normal(scale=.2,size=(n,p))+(labels[:,None]*2-1)*2
    y=.3+X@np.linspace(.1,.4,p)+rng.normal(scale=.03,size=n)
    return X,y,labels


def estimator(case,device,seed,n):
    from statgpu import nonparametric as N, unsupervised as U
    from statgpu.semiparametric import GAM
    kw={'device':device}
    models={
      'kde':lambda:N.KernelDensityEstimator(bandwidth=.4,**kw),
      'kernel_regression_nw':lambda:N.KernelRegression(bandwidth=.4,regression='nw',kernel_metric='diagonal',**kw),
      'kernel_regression_local_linear':lambda:N.KernelRegression(bandwidth=.4,regression='local_linear',kernel_metric='diagonal',**kw),
      'kernel_ridge_cv':lambda:N.KernelRidgeCV(alphas=[.05,.2,1.],cv=3,kernel='rbf',gamma=.5,random_state=seed,**kw),
      'kernel_ridge':lambda:N.KernelRidge(alpha=.2,kernel='rbf',gamma=.5,**kw),
      'kernel_pca':lambda:N.KernelPCA(n_components=2,kernel='rbf',gamma=.5,**kw),
      'nystroem':lambda:N.Nystroem(n_components=min(12,n),kernel='rbf',gamma=.5,random_state=seed,**kw),
      'spline':lambda:N.SplineTransformer(n_knots=5,degree=3,**kw),
      'gam':lambda:GAM(n_splines=6,lam=.1,**kw),
      'pca':lambda:U.PCA(n_components=2,svd_solver='full',**kw),
      'incremental_pca':lambda:U.IncrementalPCA(n_components=2,batch_size=n,**kw),
      'truncated_svd':lambda:U.TruncatedSVD(n_components=2,algorithm='randomized',n_iter=5,random_state=seed,**kw),
      'kmeans':lambda:U.KMeans(n_clusters=2,n_init=1,max_iter=100,random_state=seed,**kw),
      'minibatch_kmeans':lambda:U.MiniBatchKMeans(n_clusters=2,n_init=1,batch_size=n,max_iter=100,random_state=seed,**kw),
      'dbscan':lambda:U.DBSCAN(eps=1.,min_samples=3,**kw),
      'agglomerative':lambda:U.AgglomerativeClustering(n_clusters=2,linkage='ward',**kw),
      'gaussian_mixture':lambda:U.GaussianMixture(n_components=1,covariance_type='diag',random_state=seed,max_iter=100,**kw),
      'nmf':lambda:U.NMF(n_components=2,max_iter=300,random_state=seed,**kw),
      'minibatch_nmf':lambda:U.MiniBatchNMF(n_components=2,batch_size=n,max_iter=300,random_state=seed,**kw),
      'tsne':lambda:U.TSNE(n_components=2,perplexity=min(10,n//4),max_iter=250,random_state=seed,**kw),
      'umap_nndescent':lambda:U.UMAP(n_components=2,n_neighbors=min(10,n-1),n_epochs=100,random_state=seed,nn_method='nndescent',**kw),
      'umap':lambda:U.UMAP(n_components=2,n_neighbors=min(10,n-1),n_epochs=100,random_state=seed,**kw),
    }
    return models[case]()


def operation(case,model,X):
    if case in ('tsne','umap','umap_nndescent'):
        return model.embedding_
    if case in ('dbscan','agglomerative'):
        return model.labels_
    if case in ('kde','kernel_regression_nw','kernel_regression_local_linear','kernel_ridge','kernel_ridge_cv','gam','kmeans','minibatch_kmeans','gaussian_mixture'):
        return model.predict(X)
    return model.transform(X)


def assert_close(a,b,atol=2e-7,rtol=2e-7):
    np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=rtol)


def validate(case,m,X,y,z,labels):
    """Independent references where available; named weaker invariants otherwise."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.pairwise import rbf_kernel
    z=host(z)
    if not np.isfinite(z).all():
        raise AssertionError('nonfinite output')
    ref='';strength='external-or-analytic';metrics={}
    if case=='kde':
        from scipy.stats import gaussian_kde
        refm=gaussian_kde(X.T,bw_method=.4)
        expected=refm(X.T)
        assert_close(z.reshape(-1),expected)
        ref='scipy.stats.gaussian_kde bw_method=0.4 (covariance bandwidth factor)'
    elif case.startswith('kernel_regression'):
        variance=float(np.var(X[:,0],ddof=1))*.4**2
        diff=X[:,0][None,:]-X[:,0][:,None];weights=np.exp(-.5*diff**2/variance)
        if case=='kernel_regression_nw':expected=(weights@y)/weights.sum(axis=1)
        else:
            expected=[]
            for j in range(len(X)):
                design=np.column_stack([np.ones(len(X)),diff[j]])
                sw=np.sqrt(weights[j]);expected.append(np.linalg.lstsq(design*sw[:,None],y*sw,rcond=None)[0][0])
        assert_close(z.reshape(-1),expected,atol=3e-6)
        ref='independent Gaussian covariance-factor weights and weighted local least-squares/nonparametric mean'
    elif case=='kernel_ridge':
        K=rbf_kernel(X,gamma=.5);expected=K@np.linalg.solve(K+.2*np.eye(len(X)),y)
        assert_close(z.reshape(-1),expected);ref='analytic (K+alpha I)^-1 y; summed-loss kernel ridge'
    elif case=='kernel_ridge_cv':
        K=rbf_kernel(X,gamma=.5);indices=np.arange(len(X));np.random.RandomState(m.random_state).shuffle(indices)
        folds=np.array_split(indices,3);scores=[]
        for alpha in [.05,.2,1.]:
            fold_scores=[]
            for test in folds:
                train=indices[~np.isin(indices,test)]
                pred=K[np.ix_(test,train)]@np.linalg.solve(K[np.ix_(train,train)]+alpha*np.eye(len(train)),y[train])
                fold_scores.append(np.mean((pred-y[test])**2))
            scores.append(fold_scores)
        scores=np.asarray(scores);chosen=[.05,.2,1.][int(np.argmin(scores.mean(axis=1)))]
        assert_close(host(m.cv_results_['mse_table']).squeeze(),scores)
        assert_close(m.alpha_,chosen)
        assert_close(z.reshape(-1),K@np.linalg.solve(K+chosen*np.eye(len(X)),y))
        ref='independent seeded folds + NumPy KRR solves, score table, selection and full-data refit'
    elif case=='kernel_pca':
        K=rbf_kernel(X,gamma=.5);K-=K.mean(axis=0)[None,:];K-=K.mean(axis=1)[:,None]
        e,v=np.linalg.eigh(K);expected=(v[:,-2:]*e[-2:])@v[:,-2:].T
        assert_close(z@z.T,expected,atol=3e-6);ref='NumPy eigensystem of centered RBF Gram, rotation-invariant Gram comparison'
    elif case=='nystroem':
        C=host(m.components_);Kxc=rbf_kernel(X,C,gamma=.5);Kcc=rbf_kernel(C,gamma=.5)
        expected=Kxc@np.linalg.pinv(Kcc)@Kxc.T
        assert_close(z@z.T,expected,atol=2e-5,rtol=2e-5);ref='Nyström Gram Kxc pinv(Kcc) Kcx'
    elif case=='spline':
        from scipy.interpolate import BSpline
        knots=host(m.knots_[0]);t=np.r_[np.repeat(knots[0],3),knots,np.repeat(knots[-1],3)]
        expected=BSpline(t,np.eye(len(t)-4),3)(X[:,0])
        assert_close(z,expected);ref='SciPy BSpline with explicit clamped knot convention (not sklearn extended knots)'
    elif case=='gam':
        from scipy.interpolate import BSpline
        knots=np.percentile(X[:,0],np.linspace(0,100,4)[1:-1])
        t=np.r_[np.repeat(X.min(),4),knots,np.repeat(X.max(),4)]
        B=BSpline(t,np.eye(len(t)-4),3)(X[:,0]);B-=B.mean(axis=0)
        B=np.column_stack([np.ones(len(X)),B]);D=np.diff(np.eye(B.shape[1]-1),n=2,axis=0)
        P=np.column_stack([np.zeros(len(D)),D])
        beta=np.linalg.lstsq(np.vstack([B,np.sqrt(.1)*P]),np.r_[y,np.zeros(len(P))],rcond=None)[0]
        assert_close(z.reshape(-1),B@beta,atol=3e-6)
        ref='SciPy fixed quantile B-spline basis + NumPy augmented least squares with second-difference penalty; lambda=.1'

    elif case in ('pca','incremental_pca','truncated_svd'):
        A=X-X.mean(axis=0) if case!='truncated_svd' else X
        _,s,v=np.linalg.svd(A,full_matrices=False)
        rec=z@host(m.components_)
        optimum=np.linalg.norm(A-A@v[:2].T@v[:2])
        error=np.linalg.norm(A-rec);metrics={'reconstruction_error':float(error),'optimal_rank2_error':float(optimum)}
        if error > optimum+2e-5*(1+np.linalg.norm(A)):raise AssertionError('decomposition exceeds rank-2 optimal residual')
        ref='NumPy optimal rank-2 SVD reconstruction'
    elif case in ('kmeans','minibatch_kmeans','dbscan','agglomerative'):
        ari=adjusted_rand_score(labels,z.reshape(-1));metrics={'adjusted_rand_index':float(ari)}
        if ari<.99:raise AssertionError('separated-cluster label recovery failed')
        ref='known separated two-cluster fixture, permutation-invariant ARI >= .99'
    elif case=='gaussian_mixture':
        assert_close(host(m.means_)[0],X.mean(axis=0),atol=2e-6)
        assert_close(host(m.covariances_)[0],X.var(axis=0)+1e-6,atol=2e-6)
        ref='analytic single-component Gaussian MLE mean and diagonal covariance'
    elif case in ('nmf','minibatch_nmf'):
        H=host(m.components_);err=np.linalg.norm(X-z@H)
        if (z<0).any() or (H<0).any() or err>np.linalg.norm(X):raise AssertionError('nonnegative reconstruction invariant failed')
        ref='nonnegativity and reconstruction improves zero predictor';strength='invariant-only-no-speedup';metrics={'reconstruction_error':float(err)}
    elif case in ('tsne','umap','umap_nndescent'):
        from sklearn.manifold import trustworthiness
        if z.shape!=(len(X),2):raise AssertionError('embedding shape mismatch')
        trust=float(trustworthiness(X,z,n_neighbors=min(5,(len(X)-1)//2)))
        if not np.isfinite(trust) or trust<.5:raise AssertionError('embedding loses basic neighborhood structure')
        metrics={'trustworthiness':trust};ref='sklearn trustworthiness diagnostic >= .5; not optimizer equivalence';strength='invariant-only-no-speedup'
    else:raise ValueError(case)
    return {'status':'pass','reference':ref,'strength':strength,'metrics':metrics,
            'scope':'named fixture only; not all hyperparameter/degeneracy regimes'}


def _legacy_ownership(model,out,runtime):
    if runtime.name=='numpy':return {'observed_array_device':'cpu','compute_placement':'explicit CPU adapter','proof':'CPU inputs and explicit CPU estimator'}
    native=[]
    for name in LEARNED:
        value=getattr(model,name,None)
        if value is not None:
            try:ident=array_identity(value)
            except RuntimeError:continue
            if ident['backend']==runtime.name and ident['device']==runtime.device:native.append(name)
    try:out_id=array_identity(out)
    except RuntimeError:out_id=None
    if not native or out_id is None or out_id['backend']!=runtime.name or out_id['device']!=runtime.device:
        raise Blocked('GPU fitted/output ownership not jointly verifiable for this public adapter; no CUDA timing emitted')
    return {'observed_array_device':runtime.device,'native_fitted_fields':native,'output':out_id,
            'compute_placement':'unverified: native array ownership does not prove numerical execution placement',
            'scope':'ownership-only diagnostic with possible host work; not physical acceleration evidence'}


# Public host reports are intentionally distinct from retained native numerical
# state. These fields validate the API boundary only; stage traces prove selected
# arithmetic, and ownership alone cannot satisfy the execution gate.
FIELD_CONTRACTS = {
    'kde': (('samples_', 'weights_', 'covariance_', 'inv_covariance_'), (), 'native'),
    'kernel_regression_nw': (('samples_', 'targets_', 'weights_', 'inv_covariance_'), (), 'native'),
    'kernel_regression_local_linear': (('samples_', 'targets_', 'weights_', 'inv_covariance_'), (), 'native'),
    'spline': (('knots_', 'boundary_lo_', 'boundary_hi_'), (), 'native'),
    'kernel_pca': ((), ('lambdas_', 'alphas_', 'X_fit_'), 'native'),
    'nystroem': (('_landmarks',), ('components_', 'normalization_', 'eigenvalues_'), 'native'),
    'gam': (('coef_', 'knots_', '_basis_mean_'), (), 'host-report'),
    'agglomerative': ((), ('labels_', 'children_', 'distances_'), 'host-report'),
}


def verify_device(model, out, runtime, case=None):
    if case not in FIELD_CONTRACTS:
        return _legacy_ownership(model, out, runtime)
    native_fields, host_fields, output_role = FIELD_CONTRACTS[case]
    fields = {}
    for names, backend, device, role in (
        (native_fields, runtime.name, runtime.device, 'retained-numerical-state'),
        (host_fields, 'numpy', 'cpu', 'public-host-report')):
        for name in names:
            values = identities(getattr(model, name, None))
            if not values or any(v.get('kind') != 'array' or v['backend'] != backend
                                 or v['device'] != device for v in values):
                raise Blocked('Fitted field ownership mismatch: ' + name
                              + '; expected ' + backend + '/' + device)
            dtype = 'int64' if name in ('labels_', 'children_') else 'float64'
            if any(v['dtype'] != dtype for v in values):
                raise Blocked('Fitted field dtype mismatch: ' + name + '; expected ' + dtype)
            fields[name] = {'role': role, 'expected_dtype': dtype, 'arrays': values}
    output = identities(out)
    backend, device = (('numpy', 'cpu') if output_role == 'host-report'
                       else (runtime.name, runtime.device))
    if not output or any(v.get('kind') != 'array' or v['backend'] != backend
                         or v['device'] != device for v in output):
        raise Blocked('Output ownership mismatch: expected ' + backend + '/' + device)
    output_dtype = 'int64' if case == 'agglomerative' else 'float64'
    if any(v['dtype'] != output_dtype for v in output):
        raise Blocked('Output dtype mismatch: expected ' + output_dtype)
    return {'observed_array_device': runtime.device, 'fitted_fields': fields,
            'output': {'role': output_role, 'expected_dtype': output_dtype, 'arrays': output},
            'compute_placement': 'separately gated by algorithm-specific numerical stages',
            'scope': 'API ownership/report boundary only; not numerical execution evidence'}


def timed(fn,rt):
    rt.sync(rt.device);t=time.perf_counter();value=fn();rt.sync(rt.device)
    return value,time.perf_counter()-t


def knockoff(case,rt,X,y,seed,q=.2):
    from statgpu.feature_selection import fixed_x_knockoff_filter,model_x_knockoff_filter
    fn=fixed_x_knockoff_filter if case=='knockoff_fixed' else model_x_knockoff_filter
    return fn(rt.array(X),rt.array(y),q=q,method='corr_diff',backend=rt.name,random_state=seed)


def validate_knockoff(result,truth):
    W=np.asarray(result.W);candidates=np.sort(np.unique(np.abs(W[W!=0])))
    expected=np.inf
    for threshold in candidates:
        ratio=(1+np.sum(W<=-threshold))/max(1,np.sum(W>=threshold))
        if ratio<=result.q:expected=threshold;break
    assert_close(result.threshold,expected)
    selected=np.asarray(result.selected_features,dtype=int)
    np.testing.assert_array_equal(selected,np.flatnonzero(W>=expected))
    if not np.isfinite(W).all():raise AssertionError('nonfinite knockoff statistic')
    false=int(np.sum(~np.isin(selected,truth)));power=float(np.sum(np.isin(truth,selected))/len(truth))
    return {'status':'pass','reference':'independent knockoff+ threshold enumeration',
            'strength':'threshold-contract-only-no-FDR-certification',
            'metrics':{'realized_fdp':false/max(1,len(selected)),'realized_power':power,'selected_count':len(selected)},
            'scope':'one seeded trial; empirical FDR calibration requires many independent seeds'}


@contextmanager
def retain_partial_trace(traces):
    """Keep observed preflight work if a later oracle or timed call raises."""
    try:
        yield
    except Exception as exc:
        if not hasattr(exc, 'execution_trace'):
            exc.execution_trace = list(traces)
        raise


def traced_call(case, call, rt, traces, *, phase=None, allow_incomplete=False):
    if phase is None and case not in TRACE_CASES:
        return call()
    specs = (estimator_stages(case, rt.name, phase) if phase is not None
             else learning_stages(case, rt.name))
    if not specs:
        return call()
    observer = NumericalTrace(rt, specs)
    try:
        with observer:
            result = call()
    except Exception as exc:
        traces.append(observer.snapshot(reason=exc))
        exc.execution_trace = list(traces)
        raise
    try:
        traces.append(observer.report())
    except TraceIncomplete as exc:
        traces.append(exc.execution_trace)
        if not allow_incomplete:
            blocked = Blocked(str(exc))
            blocked.execution_trace = list(traces)
            raise blocked from exc
    return result


def add_trace(row, traces, *, instrumented=True):
    if traces:
        row['execution_trace'] = traces
        row['performance_eligible'] = False
        incomplete = [trace for trace in traces if trace['status'] != 'observed']
        if incomplete:
            row.update(status='blocked', blocked_kind='incomplete-execution-evidence',
                       reason='; '.join(trace['reason'] for trace in incomplete))
            row.pop('timing', None)
        elif 'timing' in row:
            row['timing']['instrumentation'] = (
                'Includes stage profiler/native primitive observation overhead; diagnostic only'
                if instrumented else
                'Algorithm-specific primitive tracing in untimed preflight only; timed calls unprofiled')
            row['timing']['transfer_policy'] = 'Public-call internal host/device transfers included'
    return row


def helper_case(case,rt,n,seed,warmup,repeats):
    from statgpu import nonparametric as N
    from scipy.stats import gaussian_kde,norm
    from sklearn.metrics.pairwise import pairwise_kernels as sk_kernels
    rng=np.random.default_rng(seed);X=rng.normal(size=(n,1));y=np.sin(X[:,0]);Q=np.linspace(-1,1,9)[:,None]
    def call():
        dx,dq,dy=rt.array(X),rt.array(Q),rt.array(y)
        if case=='smoothing_helpers':
            a=N.fit_kde(dx,bandwidth=.4,backend=rt.name).pdf(dq)
            b=N.kde_pdf(dx,dq,bandwidth=.4,backend=rt.name)
            c=N.fit_kernel_regression(dx,dy,bandwidth=.4,backend=rt.name).predict(dq)
            d=N.kernel_regression_predict(dx,dy,dq,bandwidth=.4,backend=rt.name)
            return a,b,c,d
        if case=='bandwidth_helpers':
            kw={'n_eff':n,'n_features':1,'samples_2d':dx,'weights_1d':rt.array(np.ones(n)/n),'data_cov':rt.array(np.array([[X.var(ddof=1)]])),'xp':rt.module}
            return N.select_bandwidth('scott',**kw),N.select_bandwidth_factor('scott',**kw)
        if case=='kde_normal_ci':
            return N.kde_confidence_interval(dx,dq,bandwidth=.4,backend=rt.name,method='normal')
        if case=='chi2_kernel':return N.chi2_kernel(rt.array(np.abs(X)+.1),rt.array(np.abs(Q)+.1),gamma=.5,xp=rt.module)
        return {name:N.pairwise_kernels(dx,dq,metric=name,xp=rt.module,**params) for name,params in [('linear',{}),('rbf',{'gamma':.5}),('polynomial',{'gamma':.5,'degree':3,'coef0':1.}),('laplacian',{'gamma':.5}),('cosine',{}),('sigmoid',{'gamma':.5,'coef0':1.})]}
    def check(value):
        if case=='bandwidth_helpers':
            assert_close(value[0].factor,n**(-.2));assert_close(value[1],n**(-.2));return
        if case=='kde_normal_ci':
            estimate=gaussian_kde(X.T,bw_method=.4)(Q.T);h=X.std(ddof=1)*.4
            se=np.sqrt(estimate/(2*np.sqrt(np.pi)*n*h))
            assert_close(value.estimate,estimate);assert_close(value.lower,np.maximum(estimate-norm.ppf(.975)*se,0));assert_close(value.upper,estimate+norm.ppf(.975)*se);return
        arrays=list(value.values()) if isinstance(value,dict) else (value if isinstance(value,tuple) else [value])
        for a in arrays:
            ident=array_identity(a)
            if ident['backend']!=rt.name or ident['device']!=rt.device:raise Blocked('helper output device mismatch')
        if case=='smoothing_helpers':
            density=gaussian_kde(X.T,bw_method=.4)(Q.T)
            w=np.exp(-.5*(Q-X.T)**2/(X.var(ddof=1)*.16));prediction=w@y/w.sum(axis=1)
            for a in value[:2]:assert_close(host(a).ravel(),density)
            for a in value[2:]:assert_close(host(a).ravel(),prediction)
        elif case=='chi2_kernel':
            xx=np.abs(X)+.1;qq=np.abs(Q)+.1
            expected=np.exp(-.5*(xx-qq.T)**2/(xx+qq.T));assert_close(host(value),expected)
        else:
            for name,a in value.items():
                params={} if name in ('linear','cosine') else {'gamma':.5}
                if name=='polynomial':params.update(degree=3,coef0=1.)
                if name=='sigmoid':params.update(coef0=1.)
                assert_close(host(a),sk_kernels(X,Q,metric=name,**params))
    values=[];traces=[]
    observed=lambda:traced_call(case,call,rt,traces,allow_incomplete=True)
    with rt.context(), retain_partial_trace(traces):
        check(observed())
        for i in range(warmup+repeats):
            result,t=timed(observed,rt);check(result)
            if i>=warmup:values.append(t)
    row={'case':case,'backend':rt.name,'status':'pass','performance_eligible':True,'problem':{'n':n,'p':1,'seed':seed,'dtype':'float64','data_id':hashlib.sha256(X.tobytes()+Q.tobytes()).hexdigest()},'correctness':{'status':'pass','strength':'external-or-analytic','reference':'SciPy Gaussian KDE/normal CI, analytic NW/Scott/chi2, sklearn pairwise kernels; matching helper operation'},'timing':{'scope':'combined public helper operations with input conversion','samples_seconds':values,'median_seconds':statistics.median(values),'warmup':warmup,'repeats':repeats,'synchronized_device':rt.device}}
    return add_trace(row,traces)


def special_case(case,rt,n,seed,warmup,repeats):
    from statgpu import nonparametric as N
    from scipy.interpolate import BSpline
    from scipy.stats import gaussian_kde
    rng=np.random.default_rng(seed);x=np.linspace(0,1,n);knots=np.linspace(.2,.8,4)
    samples=rng.normal(size=(n,1));points=np.linspace(-1,1,9)
    row={'case':case,'backend':rt.name,'status':'failed','problem':{'n':n,'p':1,'seed':seed,'dtype':'float64','data_id':hashlib.sha256(x.tobytes()+samples.tobytes()).hexdigest()}}
    def call():
        if case=='kde_bootstrap':
            return N.kde_bootstrap_confidence_interval(rt.array(samples),rt.array(points),bandwidth=.4,n_resamples=20,random_state=seed,return_bootstrap_samples=True,backend=rt.name)
        if case.startswith('bandwidth'):
            method=case[len('bandwidth_'):];device={'numpy':'cpu','cupy':'cuda','torch':'torch'}[rt.name]
            if method=='cv':return N.KernelRegression(bandwidth='cv',device=device).fit(rt.array(samples),rt.array(np.sin(samples[:,0])))
            return N.KernelDensityEstimator(bandwidth=method,device=device).fit(rt.array(samples))
        dx=rt.array(x);dk=rt.array(knots)
        fn={'bspline_basis':N.bspline_basis,'natural_spline':N.natural_cubic_spline_basis,'cyclic_spline':N.cyclic_cubic_spline_basis,'thin_plate_spline':N.thin_plate_spline_basis}[case]
        return fn(dx,dk,xp=rt.module)
    def check(value):
        strength='external-or-analytic';metrics={}
        if case=='kde_bootstrap':
            draws=np.asarray(value.bootstrap_samples)
            assert_close(value.lower,np.quantile(draws,.025,axis=0));assert_close(value.upper,np.quantile(draws,.975,axis=0))
            assert_close(value.estimate,gaussian_kde(samples.T,bw_method=.4)(points))
            if draws.shape!=(20,9) or not np.isfinite(draws).all():raise AssertionError('invalid bootstrap sample matrix')
            ref='stored replicate quantiles and SciPy density; 20 replicates only, not interval coverage certification'
            strength='bootstrap-quantile-contract-no-coverage-certification'
        elif case.startswith('bandwidth'):
            method=case[len('bandwidth_'):];factor=float(value.bandwidth_factor_)
            if not np.isfinite(factor) or factor<=0:raise AssertionError('invalid selected factor')
            if method=='scott':assert_close(factor,n**(-1/5))
            elif method=='silverman':assert_close(factor,(n*3/4)**(-1/5))
            elif method in ('nrd','nrd0'):
                std=float(samples.std(ddof=1));scale=min(std,float(np.subtract(*np.percentile(samples,[75,25])))/1.34)
                expected=(1.06 if method=='nrd' else .9)*scale*n**(-.2)/std
                assert_close(factor,expected,atol=2e-6)
            else:strength='selector-diagnostic-no-optimality-certification'
            if method!='cv':assert_close(host(value.predict(rt.array(points))),gaussian_kde(samples.T,bw_method=factor)(points))
            else:
                pred=host(value.predict(rt.array(samples)));
                if not np.isfinite(pred).all():raise AssertionError('invalid CV smoother predictions')
            metrics={'selected_factor':factor};ref='Scott/Silverman/NRD analytic rule where applicable; density at selected factor vs SciPy; other selector optimization not independently certified'
        else:
            ident=array_identity(value)
            if ident['backend']!=rt.name or ident['device']!=rt.device:raise Blocked('specialized basis output not on requested concrete device')
            B=host(value)
            if not np.isfinite(B).all():raise AssertionError('nonfinite basis')
            if case=='thin_plate_spline':
                expected=np.column_stack([np.abs(x[:,None]-knots[None,:])**3,np.ones(n),x]);assert_close(B,expected)
                ref='analytic 1D cubic radial basis plus intercept/linear polynomial'
            else:
                t=np.r_[np.zeros(4),knots,np.ones(4)];spl=BSpline(t,np.eye(len(t)-4),3);base=spl(x)
                if case=='bspline_basis':assert_close(B,base);ref='SciPy clamped cubic BSpline'
                else:
                    transform=np.linalg.lstsq(base,B,rcond=None)[0];assert_close(base@transform,B,atol=2e-6)
                    if case=='natural_spline':error=float(np.max(np.abs(spl.derivative(2)([0,1])@transform)))
                    else:error=max(float(np.max(np.abs((spl.derivative(d)(0)-spl.derivative(d)(1))@transform))) for d in (0,1,2))
                    metrics={'boundary_constraint_max_abs_error':error}
                    if error>2e-3:raise AssertionError(f'{case} boundary constraint violation: max error {error:.6g}; no timing eligible')
                    ref='SciPy reconstructed spline boundary derivatives; unit interval only'
        return {'status':'pass','reference':ref,'strength':strength,'metrics':metrics}
    traces=[];observed=lambda:traced_call(case,call,rt,traces,allow_incomplete=True)
    with rt.context(), retain_partial_trace(traces):
        value=observed();correct=check(value);values=[]
        for i in range(warmup+repeats):
            value,t=timed(observed,rt);correct=check(value)
            if i>=warmup:values.append(t)
    row.update(status='pass',performance_eligible=correct['strength']=='external-or-analytic',correctness=correct,timing={'scope':'selection+fit' if case.startswith('bandwidth') else 'public-operation','samples_seconds':values,'median_seconds':statistics.median(values),'warmup':warmup,'repeats':repeats,'synchronized_device':rt.device,'transfer_policy':'specialized basis input conversion included; CPU bootstrap/selector public workflow included'})
    return add_trace(row,traces)


def classify_placement(row,rt):
    if row.get('execution_trace') and row.get('status')=='pass':
        row['performance_eligible']=False
        row['execution_qualification']={
            'array_device':rt.device,
            'compute_placement':'host-scalar' if row['case']=='bandwidth_helpers' else ('host' if rt.name=='numpy' else ('observed-native-arithmetic-with-host-control-and-api-boundaries' if row['case']=='kde_bootstrap' or (row['case'].startswith('bandwidth_') and row['case']!='bandwidth_cv') else 'mixed-host-and-observed-native-arithmetic')),
            'evidence':'Per-invocation source-identified numerical stages and native arithmetic observations; not full GPU residency',
            'timing_use':'Diagnostic public-call duration only, not canonical speedup evidence; instrumentation scope is recorded in timing'}
        return row
    if rt.name!='numpy' and row.get('status')=='pass':
        row['performance_eligible']=False
        row['execution_qualification']={
            'array_device':rt.device,
            'compute_placement':'unverified',
            'evidence':'Native inputs/outputs or fitted fields establish ownership only; computation may include host work or copies.',
            'timing_use':'diagnostic synchronized public-call duration only; not canonical GPU timing or speedup evidence'}
    return row


def run_case(case,rt,n,p,seed,warmup,repeats):
    X,y,labels=data(n,p,seed)
    if case in ('kde','kernel_regression_nw','kernel_regression_local_linear','spline','gam'):X=X[:,:1]
    if case in ('kernel_regression_nw','kernel_regression_local_linear','gam'):y=np.sin(X[:,0])+.2*X[:,0]
    if case in ('nmf','minibatch_nmf'):X=np.abs(X)
    row={'case':case,'backend':rt.name,'problem':{'n':n,'p':X.shape[1],'dtype':str(X.dtype),'seed':seed,
           'data_id':hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()},'status':'failed'}
    if case in ('smoothing_helpers','bandwidth_helpers','pairwise_kernels','chi2_kernel','kde_normal_ci'):
        return classify_placement(helper_case(case,rt,n,seed,warmup,repeats),rt)
    if case.startswith('bandwidth') or case in ('kde_bootstrap','bspline_basis','natural_spline','cyclic_spline','thin_plate_spline'):
        return classify_placement(special_case(case,rt,n,seed,warmup,repeats),rt)
    if case.startswith('knockoff'):
        X=np.random.default_rng(seed).normal(size=(n,p));truth=np.arange(max(1,min(6,p//3)));y=X[:,truth].sum(axis=1)*3+np.random.default_rng(seed+1).normal(size=n)
        row['problem']['true_signals']=truth.tolist();row['problem']['q']=.2;row['problem']['positive_statistics_needed_for_knockoff_plus']=5
        row['problem']['data_id']=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
        vals=[];traces=[]
        with rt.context(), retain_partial_trace(traces):
          for i in range(warmup+repeats):
            result,t=timed(lambda:traced_call(case,lambda:knockoff(case,rt,X,y,seed),rt,traces,allow_incomplete=True),rt);check=validate_knockoff(result,truth)
            if i>=warmup:vals.append(t)
        row.update(status='pass',performance_eligible=False,correctness=check,timing={'scope':'construction+statistics+selection','samples_seconds':vals,'median_seconds':statistics.median(vals),'warmup':warmup,'repeats':repeats,'synchronized_device':rt.device})
        return classify_placement(add_trace(row,traces),rt)
    device={'numpy':'cpu','cupy':'cuda','torch':'torch'}[rt.name]
    traces=[]
    with rt.context(), retain_partial_trace(traces):
        dx,dy=rt.array(X),rt.array(y)
        def fitted():
            m=estimator(case,device,seed,n)
            if case in ('kernel_regression_nw','kernel_regression_local_linear','kernel_ridge','kernel_ridge_cv','gam'):m.fit(dx,dy)
            else:m.fit(dx)
            return m
        if case in ESTIMATOR_TRACE_CASES:
            m=traced_call(case,fitted,rt,traces,phase='fit',allow_incomplete=True)
            out=traced_call(case,lambda:operation(case,m,dx),rt,traces,
                            phase='operation',allow_incomplete=True)
        else:
            m=fitted();out=operation(case,m,dx)
        # Correctness remains independently visible when observation is incomplete.
        check=validate(case,m,X,y,out,labels)
        row['correctness']=check
        try:
            proof=verify_device(m,out,rt,case)
        except Blocked as exc:
            row.update(status='blocked', performance_eligible=False, reason=str(exc),
                       blocked_kind='ownership-boundary-mismatch', execution_trace=traces)
            return row
        row['execution']=proof
        if any(trace['status']!='observed' for trace in traces):
            return add_trace(row,traces,instrumented=False)

        fits=[];ops=[]
        for i in range(warmup+repeats):
            m,t=timed(fitted,rt);out,t_op=timed(lambda:operation(case,m,dx),rt)
            verify_device(m,out,rt,case);check=validate(case,m,X,y,out,labels)
            if i>=warmup:fits.append(t);ops.append(t_op)
        operation_name='stored-output-access' if case in ('tsne','umap','umap_nndescent','dbscan','agglomerative') else ('predict' if case in ('kde','kernel_regression_nw','kernel_regression_local_linear','kernel_ridge','kernel_ridge_cv','gam','kmeans','minibatch_kmeans','gaussian_mixture') else 'transform')
        row.update(status='pass',performance_eligible=check['strength']=='external-or-analytic',correctness=check,execution=proof,timing={'fit':{'scope':'public-fit','samples_seconds':fits,'median_seconds':statistics.median(fits)},operation_name:{'scope':operation_name,'samples_seconds':ops,'median_seconds':statistics.median(ops)},'warmup':warmup,'repeats':repeats,'synchronized_device':proof['observed_array_device'],'transfer_policy':'input transfer excluded; internal casts/report transfers included'})
        return classify_placement(add_trace(row,traces,instrumented=False),rt)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));ap.add_argument('--backends',nargs='+',choices=['numpy','cupy','torch'],default=['numpy','cupy','torch'])
    ap.add_argument('--n',type=int,default=256);ap.add_argument('--p',type=int,default=4);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--knockoff-p',type=int,default=20,help='Separate feature dimension for selection trials; n must be >=2*knockoff-p')
    ap.add_argument('--max-pairwise-elements',type=int,default=25000000,help='Resource safety cap for n*n reference allocations; oversized cases are blocked, not silently shrunk')
    ap.add_argument('--selection-seeds',nargs='+',type=int,default=None,help='Independent Knockoff trials only; default uses --seed; repeats within a trial are timing repetitions, not independent FDR trials')
    ap.add_argument('--warmup',type=int,default=1);ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--require-backends',action='store_true');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.smoke:a.n=48;a.p=4;a.warmup=0;a.repeats=1;a.backends=['numpy']
    if a.n<16 or a.p<2 or a.n<2*a.p or a.warmup<0 or a.repeats<1 or a.device<0:ap.error('require n>=max(16,2p), p>=2, warmup>=0, repeats>=1, device>=0')
    if Path(statgpu.__file__).resolve().parent!=ROOT/'statgpu':raise RuntimeError('wrong statgpu import source')
    if len(set(a.backends))!=len(a.backends) or len(set(a.cases))!=len(a.cases):ap.error('duplicate backends/cases are not allowed')
    if a.selection_seeds and len(set(a.selection_seeds))!=len(a.selection_seeds):ap.error('selection-seeds must be unique independent trials')
    if a.knockoff_p<2 or a.max_pairwise_elements<1:ap.error('knockoff-p>=2 and max-pairwise-elements>=1 required')
    source=source_identity();rows=[];env={}
    trial_cases=[(case,seed) for case in a.cases for seed in (a.selection_seeds if case.startswith('knockoff') and a.selection_seeds else [a.seed])]
    for backend in a.backends:
        try:rt=Runtime(backend,a.device)
        except BackendUnavailable as e:
            rows.extend({'case':case,'backend':backend,'seed':seed,'status':'skipped','reason':str(e)} for case,seed in trial_cases);continue
        except Exception as e:
            rows.extend({'case':case,'backend':backend,'seed':seed,'status':'failed','reason':str(e)} for case,seed in trial_cases);continue
        env[backend]={'device':rt.device,**rt.hardware}
        for case,trial_seed in trial_cases:
            try:
                if a.n*a.n>a.max_pairwise_elements:raise Blocked('n*n exceeds configured resource cap; increase --max-pairwise-elements only with adequate memory')
                p=a.knockoff_p if case.startswith('knockoff') else a.p
                if a.n<2*p:raise Blocked('n must be >=2*p for this selection fixture')
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');row=run_case(case,rt,a.n,p,trial_seed,a.warmup,a.repeats)
                row['warnings']=[str(w.message) for w in caught]
                convergence=[w for w in caught if 'converg' in str(w.message).lower() or 'converg' in w.category.__name__.lower()]
                if convergence:
                    row['performance_eligible']=False
                    row.pop('timing',None)
                    row['status']='blocked';row['reason']='Convergence warning: '+str(convergence[0].message)
            except UnsupportedCapability as e:row={'case':case,'backend':backend,'seed':trial_seed,'status':'blocked','blocked_kind':'unsupported-capability','reason':str(e)}
            except Blocked as e:row={'case':case,'backend':backend,'seed':trial_seed,'status':'blocked','reason':str(e),'exception':exception_chain(e),'execution_trace':getattr(e,'execution_trace',[])}
            except ImportError as e:row={'case':case,'backend':backend,'seed':trial_seed,'status':'skipped','reason':str(e)}
            except Exception as e:row={'case':case,'backend':backend,'seed':trial_seed,'status':'failed','reason':f'{type(e).__name__}: {e}','exception':exception_chain(e),'execution_trace':getattr(e,'execution_trace',[])}
            rows.append(row)
    after=source_identity();changed=after!=source
    failed=changed or any(x['status']=='failed' for x in rows) or (a.require_backends and any(x['status']!='pass' for x in rows))
    environment=base_environment()
    for package in ('scikit-learn','statsmodels'):
        try:environment['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:environment['packages'][package]=None
    environment['backends']=env
    environment['environment_id']=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()
    summaries=[]
    for case in ('knockoff_fixed','knockoff_model'):
      for backend in a.backends:
        trials=[r for r in rows if r.get('case')==case and r.get('backend')==backend and r['status']=='pass']
        if trials:
            fdp=[r['correctness']['metrics']['realized_fdp'] for r in trials];power=[r['correctness']['metrics']['realized_power'] for r in trials]
            summaries.append({'case':case,'backend':backend,'independent_trial_count':len(trials),'seeds':[r['problem']['seed'] for r in trials],'mean_fdp':float(np.mean(fdp)),'mean_power':float(np.mean(power)),'fdp_mc_standard_error':float(np.std(fdp,ddof=1)/np.sqrt(len(fdp))) if len(fdp)>1 else None,'interpretation':'fixture-specific Monte Carlo diagnostic, not a proof of target FDR control'})
    result={'schema_version':'statgpu-benchmark-v2','benchmark':{'name':'learning-coverage','script':str(Path(__file__).relative_to(ROOT)),'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}},'source':source,'source_after':after,
      'environment':environment,'results':rows,'selection_summary':summaries,
      'validation':{'status':'failed' if failed else ('partial' if any(x['status']!='pass' for x in rows) else 'complete'),'tier':'local-minimal' if a.smoke else 'candidate-not-canonical','canonical_ready':False,'uncovered':['not exhaustive hyperparameter/degeneracy/CV/bootstrap matrix','invariant-only stochastic cases are diagnostic timings, not speedup evidence','physical GPU not established by CPU smoke'],'source_changed':changed}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['validation']['status'],'rows':len(rows),'failed':[x['case'] for x in rows if x['status']=='failed']}))
    return 2 if failed else 0

if __name__=='__main__':raise SystemExit(main())
