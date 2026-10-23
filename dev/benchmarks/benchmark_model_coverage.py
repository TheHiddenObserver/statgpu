#!/usr/bin/env python3
"""Correctness-gated supervised family coverage, not a Cartesian certification.

python -m dev.benchmarks.benchmark_model_coverage --smoke --output results/models.json
CV families are intentionally delegated to the existing benchmark_cv_models.py.
GPU timings require observed numerical arrays, not merely a requested device.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import statgpu
from statgpu import linear_model as lm, panel, survival
from dev.benchmarks.benchmark_release_refresh import Runtime, BackendUnavailable, array_identity, environment

GLMS = ('gaussian','binomial','poisson','gamma','inverse_gaussian','negative_binomial','tweedie')
PENALTIES = ('l1','l2','elasticnet','scad','mcp','group_lasso','group_scad','group_mcp')
PANELS = ('pooled','within','between','first_difference','random_effects','fama_macbeth')
WRAPPERS = {
 'LinearRegression':'glm:gaussian','Ridge':'penalty:l2','Lasso':'penalty:l1','ElasticNet':'penalty:elasticnet',
 'AdaptiveLasso':'penalty:l1','SCADRegression':'penalty:scad','MCPRegression':'penalty:mcp',
 'LogisticRegression':'glm:binomial','PoissonRegression':'glm:poisson','GammaRegression':'glm:gamma',
 'InverseGaussianRegression':'glm:inverse_gaussian','NegativeBinomialRegression':'glm:negative_binomial',
 'TweedieRegression':'glm:tweedie','QuantileRegression':'quantile:0.5',
 'PenalizedLinearRegression':'penalty:l2','PenalizedLogisticRegression':'penalized_glm:logistic',
 'PenalizedPoissonRegression':'penalized_glm:poisson','PenalizedGammaRegression':'penalized_glm:gamma',
 'PenalizedInverseGaussianRegression':'penalized_glm:inverse_gaussian',
 'PenalizedNegativeBinomialRegression':'penalized_glm:negative_binomial',
 'PenalizedTweedieRegression':'penalized_glm:tweedie'}
CASES = tuple(['glm:'+x for x in GLMS] + ['penalty:'+x for x in PENALTIES]
              + ['penalized_glm:'+x for x in ('logistic','poisson','gamma','inverse_gaussian','negative_binomial','tweedie')]
              + ['robust:'+x for x in ('huber','fair','bisquare')]
              + ['quantile:'+x for x in ('0.25','0.5','0.75')]
              + ['cox:'+x for x in ('breslow','efron','penalized_l2')]
              + ['panel:'+x for x in PANELS]
              + ['glm_inference:'+f+':'+c for f in GLMS for c in ('nonrobust','hc0')]
              + ['inference:'+x for x in ('nonrobust','hc0','hc3')] + ['wrapper:'+x for x in WRAPPERS])
DEVICE = {'numpy':'cpu','cupy':'cuda','torch':'torch'}


def source_identity():
    paths = sorted((ROOT/'statgpu').rglob('*.py')) + [Path(__file__), ROOT/'dev/benchmarks/benchmark_release_refresh.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.DEVNULL).decode().strip()
    try:
        head, dirty = git('rev-parse','HEAD'), bool(git('status','--porcelain'))
        tracked = bool(git('ls-files',str(Path(__file__).relative_to(ROOT))))
    except (OSError,subprocess.CalledProcessError):
        head, dirty, tracked = None,True,False
    return {'git_commit':head,'working_tree':'dirty' if dirty or not tracked else 'clean',
            'runner_overlay':not tracked,'statgpu_version':statgpu.__version__,
            'statgpu_import_path':str(Path(statgpu.__file__).resolve()),
            'file_hashes':hashes,'worktree_fingerprint':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}


def orthogonal_data(n,p,seed):
    rng=np.random.default_rng(seed)
    raw=rng.normal(size=(n,p)); raw-=raw.mean(0)
    X=np.linalg.qr(raw)[0]*np.sqrt(n)
    y=.3+X@np.linspace(-.6,.8,p)+rng.normal(scale=.15,size=n)
    return X,y,rng


def scalar_penalty(t,a,name):
    t=abs(float(t))
    if name in ('l1','group_lasso'):return a*t
    if name=='l2':return a*t*t/2
    if name=='elasticnet':return a*(.5*t+.25*t*t)
    if name in ('mcp','group_mcp'):return a*t-t*t/6 if t<=3*a else 1.5*a*a
    if name in ('scad','group_scad'):
        return a*t if t<=a else ((-t*t+7.4*a*t-a*a)/5.4 if t<=3.7*a else 2.35*a*a)
    raise ValueError(name)


def orthogonal_oracle(X,y,alpha,name):
    """Independent piecewise scalar minimization on an orthonormal design."""
    from scipy.optimize import minimize_scalar
    z=X.T@(y-y.mean())/len(y); beta=np.zeros_like(z)
    groups=[np.arange(i,min(i+2,len(z))) for i in range(0,len(z),2)] if name.startswith('group_') else [np.array([i]) for i in range(len(z))]
    for indices in groups:
        radius=np.linalg.norm(z[indices]); a=alpha*np.sqrt(len(indices)) if name.startswith('group_') else alpha
        objective=lambda t:.5*(t-radius)**2+scalar_penalty(t,a,name)
        knots=sorted(set([0.,a,3*a,3.7*a,max(radius+2*a,4*a,1e-5)]))
        candidates=knots[:]
        for lo,hi in zip(knots[:-1],knots[1:]):
            candidates.append(minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-13}).x)
        optimum=min(candidates,key=objective)
        if radius:beta[indices]=z[indices]*optimum/radius
    return np.r_[y.mean(),beta]


def cox_objective(beta,X,t,e,alpha=0.,ties='breslow'):
    from scipy.special import logsumexp
    eta=X@beta; out=0.
    for ti in np.unique(t[e==1]):
        events=(t==ti)&(e==1); d=int(events.sum()); risk=t>=ti
        out-=eta[events].sum()
        m=float(eta[risk].max()); total=np.exp(eta[risk]-m).sum(); fail=np.exp(eta[events]-m).sum()
        for j in range(d):out+=m+np.log(total-(j/d)*fail) if ties=='efron' else logsumexp(eta[risk])
    return float(out/len(t)+alpha*np.dot(beta,beta)/2)


def make_case(case,n,p,seed,backend):
    """Return factory, host fit arguments, independent target and contract."""
    if case.startswith('glm_inference:'):
        import statsmodels.api as sm
        _,family,cov=case.split(':')
        _,args,fitkw,target,contract,_=make_case('glm:'+family,n,p,seed,backend)
        X,y=args
        fam={'gaussian':sm.families.Gaussian(),'binomial':sm.families.Binomial(),
             'poisson':sm.families.Poisson(),'gamma':sm.families.Gamma(link=sm.families.links.Log()),
             'inverse_gaussian':sm.families.InverseGaussian(link=sm.families.links.Log()),
             'negative_binomial':sm.families.NegativeBinomial(alpha=1.),
             'tweedie':sm.families.Tweedie(var_power=1.5,link=sm.families.links.Log())}[family]
        ref=sm.GLM(y,sm.add_constant(X),family=fam).fit(maxiter=1000,tol=1e-10,cov_type='nonrobust' if cov=='nonrobust' else 'HC0')
        infer={'bse':np.asarray(ref.bse),'pvalues':np.asarray(ref.pvalues),'conf_int':np.asarray(ref.conf_int()),
               'zvalues':np.asarray(ref.tvalues),'covariance':np.asarray(ref.cov_params())}
        contract.update(scope='fit+inference-public-api',reference='statsmodels.GLM coefficients/Fisher or HC0 covariance/normal inference')
        contract['parameters'].update(case=case,cov_type=cov)
        return lambda:lm.GeneralizedLinearModel(family=family,C=1e12,max_iter=1000,tol=1e-9,compute_inference=True,cov_type=cov,device=DEVICE[backend]),args,fitkw,np.asarray(ref.params),contract,infer
    if case.startswith('wrapper:'):
        name=case.split(':',1)[1];base=WRAPPERS[name]
        _,args,fitkw,target,contract,infer=make_case(base,n,p,seed,backend)
        params={'device':DEVICE[backend],'compute_inference':False}
        if name!='LinearRegression':params.update(max_iter=3000,tol=1e-9)
        if name in ('Ridge','Lasso','ElasticNet','AdaptiveLasso','SCADRegression','MCPRegression'):
            params['alpha']=.08
        elif name.startswith('Penalized'):
            params.update(penalty='l2',alpha=.08,solver='lbfgs')
        elif name not in ('LinearRegression','QuantileRegression'):
            params['C']=1e12
        if name=='AdaptiveLasso':
            X,y=args;z=X.T@(y-y.mean())/len(y);weights=1/(np.abs(z)+1e-8);weights/=weights.mean()
            target=np.r_[y.mean(),np.sign(z)*np.maximum(np.abs(z)-.08*weights,0)]
            contract.update(reference='orthogonal adaptive Lasso soft threshold with normalized inverse OLS weights',objective_scaling='mean squared loss /2 + .08*sum(normalized_weights*abs(coef)); nu1 eps1e-8')
        contract['parameters']['case']=case;contract['parameters']['public_class']=name
        return lambda:getattr(lm,name)(**params),args,fitkw,target,contract,infer
    X,y,rng=orthogonal_data(n,p,seed); kind,variant=case.split(':',1)
    kw={'device':DEVICE[backend]}; tol=2e-4; alpha=.08
    contract={'reference_type':'analytic','reference':'independent NumPy/SciPy calculation',
              'scope':'fit-only-public-api','coef_atol':tol,'coef_rtol':tol,
              'threshold_source':'fixed well-conditioned fixture tolerances in benchmark_model_coverage.py; not universal model acceptance',
              'parameters':{'case':case,'n':n,'p':p,'seed':seed,'dtype':'float64'}}
    args=(X,y); fitkw={}; infer=None
    if kind=='glm':
        import statsmodels.api as sm
        eta=.2+X@np.linspace(-.2,.25,p); mu=np.exp(eta)
        fam={'gaussian':sm.families.Gaussian(),'binomial':sm.families.Binomial(),
             'poisson':sm.families.Poisson(),'gamma':sm.families.Gamma(link=sm.families.links.Log()),
             'inverse_gaussian':sm.families.InverseGaussian(link=sm.families.links.Log()),
             'negative_binomial':sm.families.NegativeBinomial(alpha=1.),
             'tweedie':sm.families.Tweedie(var_power=1.5,link=sm.families.links.Log())}[variant]
        if variant=='binomial': y=rng.binomial(1,1/(1+np.exp(-eta))).astype(float)
        elif variant in ('poisson','negative_binomial'):y=rng.poisson(mu).astype(float)
        elif variant not in ('gaussian',):y=mu*rng.gamma(4,.25,size=n)
        ref=sm.GLM(y,sm.add_constant(X),family=fam).fit(maxiter=1000,tol=1e-10)
        target=np.asarray(ref.params);args=(X,y)
        factory=lambda:lm.GeneralizedLinearModel(family=variant,C=1e12,max_iter=1000,tol=1e-9,compute_inference=False,**kw)
        contract.update(reference='statsmodels.GLM',reference_type='external-python',objective_scaling='unpenalized; C=1e12 disables material binomial regularization',reference_version=importlib.metadata.version('statsmodels'))
    elif kind=='penalty':
        target=orthogonal_oracle(X,y,alpha,variant)
        penalty_kw={'groups':[list(range(i,min(i+2,p))) for i in range(0,p,2)]} if variant.startswith('group_') else None
        factory=lambda:lm.PenalizedGeneralizedLinearModel(loss='squared_error',penalty=variant,alpha=alpha,penalty_kwargs=penalty_kw,solver='fista',lla=False,max_iter=3000,tol=1e-9,compute_inference=False,**kw)
        contract['objective_scaling']='mean squared error / 2 + public penalty; group lambda=alpha*sqrt(group_size); alpha=.08; EN l1_ratio=.5; SCAD a=3.7; MCP gamma=3'
    elif kind=='penalized_glm':
        from scipy.optimize import minimize
        eta=.2+X@np.linspace(-.2,.25,p);mu=np.exp(eta)
        if variant=='logistic':y=rng.binomial(1,1/(1+np.exp(-eta))).astype(float)
        elif variant in ('poisson','negative_binomial'):y=rng.poisson(mu).astype(float)
        else:y=mu*rng.gamma(4,.25,size=n)
        D=np.column_stack([np.ones(n),X]);args=(X,y)
        def objective(b):
            z=D@b
            if variant=='logistic':v=np.logaddexp(0,z)-y*z
            elif variant=='poisson':v=np.exp(z)-y*z
            elif variant=='gamma':v=y*np.exp(-z)+z
            elif variant=='inverse_gaussian':v=.5*y*np.exp(-2*z)-np.exp(-z)
            elif variant=='negative_binomial':v=(y+1)*np.logaddexp(0,z)-y*z
            else:v=2*y*np.exp(-.5*z)+2*np.exp(.5*z)
            return float(v.mean()+alpha*np.dot(b[1:],b[1:])/2)
        opt=minimize(objective,np.zeros(p+1),method='BFGS',options={'gtol':1e-8,'maxiter':2000})
        if not np.isfinite(opt.fun) or np.max(np.abs(opt.jac))>2e-5:raise RuntimeError('independent penalized GLM optimizer failed stationarity')
        target=opt.x
        factory=lambda:lm.PenalizedGeneralizedLinearModel(loss=variant,penalty='l2',alpha=alpha,solver='lbfgs',tol=1e-9,max_iter=2000,compute_inference=False,**kw)
        contract.update(reference='independent mean GLM loss + scipy BFGS',objective_scaling='mean loss + .08*||slopes||²/2; log links; NB dispersion1, Tweedie power1.5')
    elif kind=='robust':
        from scipy.optimize import minimize
        y=y.copy();y[::11]+=2.;args=(X,y); D=np.column_stack([np.ones(n),X]);delta=1.
        def objective(b):
            r=y-D@b;u=np.abs(r)
            if variant=='huber':loss=np.where(u<=delta,.5*r*r,delta*(u-.5*delta))
            elif variant=='fair':loss=delta**2*(u/delta-np.log1p(u/delta))
            else:loss=np.where(u<=delta,delta**2/6*(1-(1-(r/delta)**2)**3),delta**2/6)
            return float(loss.mean()+alpha*.5*np.dot(b[1:],b[1:]))
        opt=minimize(objective,np.linalg.lstsq(D,y,rcond=None)[0],method='BFGS',options={'gtol':1e-8,'maxiter':3000})
        if not np.isfinite(opt.fun) or np.max(np.abs(opt.jac))>2e-5:raise RuntimeError('independent robust optimizer failed stationarity')
        target=opt.x
        factory=lambda:lm.PenalizedRobustRegression(loss=variant,penalty='l2',alpha=alpha,loss_kwargs={'delta':delta},solver='lbfgs',max_iter=3000,tol=1e-8,compute_inference=False,**kw)
        contract.update(reference='independent fixed-delta robust objective + scipy BFGS',objective_scaling='mean rho(residual;delta=1) + .08*||slopes||²/2; nonconvex bisquare local stationary reference')
    elif kind=='quantile':
        from scipy.optimize import linprog
        if n>2000:raise RuntimeError('quantile independent dense LP oracle budget is n<=2000; use maintained scalable quantile validator for larger n')
        q=float(variant);D=np.column_stack([np.ones(n),X]);k=p+1
        A=np.column_stack([D,np.eye(n),-np.eye(n)])
        opt=linprog(np.r_[np.zeros(k),np.repeat(q/n,n),np.repeat((1-q)/n,n)],A_eq=A,b_eq=y,bounds=[(None,None)]*k+[(0,None)]*(2*n),method='highs')
        if not opt.success:raise RuntimeError('quantile LP oracle failed: '+opt.message)
        target=opt.x[:k]
        factory=lambda:lm.PenalizedQuantileRegression(quantile=q,penalty='none',alpha=0.,solver='auto',max_iter=10000,tol=1e-8,compute_inference=False,**kw)
        contract.update(reference='scipy.optimize.linprog exact pinball LP',objective_scaling=f'mean pinball loss q={q}; no penalty',coef_atol=2e-3,coef_rtol=2e-3)
    elif kind=='cox':
        from scipy.optimize import minimize
        t=rng.exponential(np.exp(-X@np.linspace(-.15,.2,p)))+.1;t=np.round(t,1)+.1
        e=(rng.random(n)>.25).astype(float); ties='efron' if variant=='efron' else 'breslow';a=alpha if variant=='penalized_l2' else 0.
        opt=minimize(cox_objective,np.zeros(p),args=(X,t,e,a,ties),method='BFGS',options={'gtol':1e-8,'maxiter':1000})
        if not np.isfinite(opt.fun) or np.max(np.abs(opt.jac))>2e-5:raise RuntimeError('independent Cox optimizer failed stationarity')
        target=opt.x;args=(X,t,e)
        if variant=='penalized_l2':
            args=(X,np.column_stack([t,e]))
            factory=lambda:lm.PenalizedCoxPHModel(penalty='l2',alpha=a,ties=ties,solver='lbfgs',tol=1e-9,max_iter=1000,compute_inference=False,**kw)
        else:factory=lambda:survival.CoxPH(ties=ties,tol=1e-9,max_iter=1000,compute_inference=False,compute_cindex=False,**kw)
        contract.update(reference='independent risk-set log-sum-exp objective + scipy BFGS',objective_scaling='negative summed partial log likelihood / n + alpha*||coef||²/2',parameter_layout='slopes-only')
    elif kind=='panel':
        # Balanced panel with enough entities and periods for every auxiliary fit.
        T=max(5,p+2);N=max(p+3,n//T);n=N*T
        X=rng.normal(size=(n,p));entity=np.repeat(np.arange(N),T);times=np.tile(np.arange(T),N)
        y=X@np.linspace(-.4,.6,p)+rng.normal(size=N)[entity]+rng.normal(scale=.2,size=n)
        Xm=X.reshape(N,T,p).mean(1);ym=y.reshape(N,T).mean(1)
        def ols(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
        if variant=='pooled':
            target=ols(np.column_stack([np.ones(n),X]),y);factory=lambda:panel.PooledOLS(**kw);fitkw={'entity_ids':entity}
        elif variant=='within':
            target=ols(X-Xm[entity],y-ym[entity]);factory=lambda:panel.PanelOLS(entity_effects=True,**kw);fitkw={'entity_ids':entity,'time_ids':times}
        elif variant=='between':
            target=ols(np.column_stack([np.ones(N),Xm]),ym);factory=lambda:panel.BetweenOLS(**kw);fitkw={'entity_ids':entity}
        elif variant=='first_difference':
            target=ols(np.diff(X.reshape(N,T,p),axis=1).reshape(-1,p),np.diff(y.reshape(N,T),axis=1).ravel());factory=lambda:panel.FirstDifferenceOLS(**kw);fitkw={'entity_ids':entity,'time_ids':times}
        elif variant=='fama_macbeth':
            target=np.mean([ols(np.column_stack([np.ones(N),X[times==t]]),y[times==t]) for t in range(T)],axis=0);factory=lambda:panel.FamaMacBeth(cov_type='nonrobust',**kw);fitkw={'time_ids':times}
        else:
            Xw=X-Xm[entity];yw=y-ym[entity];se=np.sum((yw-Xw@ols(Xw,yw))**2)/(n-p-N+1)
            sb=np.sum((ym-Xm@ols(Xm,ym))**2)/(N-p);sa=max(0.,(sb-se)/T);theta=1-np.sqrt(se/(se+T*sa))
            target=ols(X-theta*Xm[entity],y-theta*ym[entity]);factory=lambda:panel.RandomEffects(**kw);fitkw={'entity_ids':entity,'time_ids':times}
        args=(X,y);contract.update(reference='independent balanced-panel transform and least squares',parameter_layout='slopes-only',scope='fit+inference-public-api')
        contract['parameters'].update(n=n,entities=N,periods=T,cov_type='nonrobust')
    elif kind=='inference':
        import statsmodels.api as sm
        ref=sm.OLS(y,sm.add_constant(X)).fit(cov_type='nonrobust' if variant=='nonrobust' else variant.upper())
        target=np.asarray(ref.params);factory=lambda:lm.LinearRegression(compute_inference=True,cov_type=variant,**kw)
        infer={'bse':np.asarray(ref.bse),'pvalues':np.asarray(ref.pvalues),'conf_int':np.asarray(ref.conf_int())}
        contract.update(reference='statsmodels.OLS',reference_type='external-python',scope='fit+inference-public-api')
    else:raise ValueError(case)
    contract['data_id']=hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in args)).hexdigest()
    return factory,args,fitkw,np.asarray(target),contract,infer


class Observe:
    """Record arrays at actual numerical dispatch calls, not constructor hints."""
    def __init__(self):self.rows=[];self.previous=None
    def trace(self,frame,event,arg):
        if event!='call':return
        module=str(frame.f_globals.get('__name__',''));name=frame.f_code.co_name
        if not module.startswith('statgpu.'):return
        numerical=(module.startswith(('statgpu.solvers.','statgpu.glm_core.')) or name in ('_fit_cpu','_fit_gpu','_fit_torch'))
        if not numerical:return
        for k in ('X','X_arr','coef','w','beta'):
            a=frame.f_locals.get(k)
            if a is None:continue
            try:identity=array_identity(a)
            except RuntimeError:continue
            r={'function':module+'.'+name,'argument':k,**identity}
            if r not in self.rows:self.rows.append(r)
    def __enter__(self):self.previous=sys.getprofile();sys.setprofile(self.trace);return self
    def __exit__(self,*exc):sys.setprofile(self.previous)


def check_fit(model,target,contract,inference):
    coef=np.asarray(model.coef_,dtype=float).ravel()
    if contract.get('parameter_layout')=='slopes-only':actual=coef
    else:actual=np.r_[float(model.intercept_),coef]
    if actual.shape!=target.shape or not np.isfinite(actual).all():raise AssertionError('nonfinite or wrong-shape parameters')
    metrics={'coef_max_abs_error':float(np.max(np.abs(actual-target)))}
    if not np.allclose(actual,target,atol=contract['coef_atol'],rtol=contract['coef_rtol']):raise AssertionError(f'parameter oracle mismatch {metrics}')
    if inference:
        for k,attr in [('bse','_bse'),('pvalues','_pvalues'),('conf_int','_conf_int')]:
            value=np.asarray(getattr(model,attr)); expected=inference[k]
            if value.shape!=expected.shape or not np.allclose(value,expected,atol=1e-6,rtol=1e-5):raise AssertionError('inference oracle mismatch '+k+' max_abs='+str(float(np.max(np.abs(value-expected)))))
            metrics[k+'_max_abs_error']=float(np.max(np.abs(value-expected)))
        if 'zvalues' in inference:
            actual_z=np.asarray(model._zvalues)
            if not np.allclose(actual_z,inference['zvalues'],atol=1e-5,rtol=2e-5):raise AssertionError('inference z-statistic mismatch')
            metrics['zvalues_max_abs_error']=float(np.max(np.abs(actual_z-inference['zvalues'])))
        if 'covariance' in inference:
            # GLM publishes marginal arrays; verify the full covariance returned
            # by its inference engine separately, outside the measured region.
            # ParameterInferenceResult.apply_to stores reporting parameters on
            # the host, while the retained inference design stays native.
            # Re-align this extra oracle call without mutating fitted state.
            from statgpu.inference._sandwich import m_estimation_inference
            from statgpu.backends import _resolve_backend, _to_numpy
            from statgpu.backends._utils import _get_xp, xp_asarray
            xp = _get_xp(_resolve_backend('auto', model._X_design))
            params_native = xp_asarray(
                model._params, dtype=model._X_design.dtype,
                xp=xp, ref_arr=model._X_design,
            )
            actual_cov=m_estimation_inference(model._loss,model._X_design,model._y_inf,params_native,cov_type=model._cov_type,
                penalty_curvature_diag=model._fit_metadata.get('penalty_curvature_diag'),sample_weight=model._sample_weight_inf)['cov_params']
            actual_cov = np.asarray(_to_numpy(actual_cov))
            if not np.allclose(actual_cov,inference['covariance'],atol=1e-6,rtol=2e-5):raise AssertionError('full GLM covariance oracle mismatch')
            metrics['covariance_max_abs_error']=float(np.max(np.abs(actual_cov-inference['covariance'])))

    return metrics


def verify_execution(observed,backend,device):
    if backend=='numpy':return
    matches=[r for r in observed if r['backend']==backend and r['device']==device]
    contradictory=[r for r in observed if r['backend']!=backend or r['device']!=device]
    if not matches:raise RuntimeError('GPU numerical-dispatch arrays could not be verified')
    if contradictory:raise RuntimeError('mixed/contradictory numerical execution observed; no pure backend timing claim')


def run_case(case,runtime,n,p,seed,warmup,repeats):
    factory,args,fitkw,target,contract,inference=make_case(case,n,p,seed,runtime.name)
    row={'case':case,'backend':runtime.name,'device':runtime.device,'problem':contract['parameters'],'data_id':contract['data_id'],'contract':contract}
    with runtime.context():
        native=tuple(runtime.array(v) for v in args)
        # IDs are metadata; panel implementations own their documented conversion.
        fitargs=fitkw
        samples=[];last_metrics=None;observed=[]
        for i in range(warmup+repeats+1):
            model=factory();runtime.sync(runtime.device)
            # Route profiling is an untimed preflight only: do not contaminate
            # CPU/GPU comparisons with Python call-tracing overhead.
            with warnings.catch_warnings(record=True) as caught, (Observe() if i==0 else nullcontext()) as observer:
                warnings.simplefilter('always');t0=time.perf_counter();model.fit(*native,**fitargs);runtime.sync(runtime.device);elapsed=time.perf_counter()-t0
            bad=[w for w in caught if 'converg' in str(w.message).lower() or 'converg' in w.category.__name__.lower()]
            if bad:raise AssertionError('convergence warning: '+str(bad[0].message))
            if i==0:
                observed=observer.rows
                if runtime.name!='numpy':
                    try:verify_execution(observed,runtime.name,runtime.device)
                    except RuntimeError as exc:
                        row.update(status='blocked',reason=str(exc)+'; no CUDA timing emitted',backend_provenance=observed);return row
            if runtime.name!='numpy':
                selected=getattr(model,'_selected_backend_name',None)
                if selected is not None and selected!=runtime.name:raise RuntimeError('backend changed after preflight')
                selected_device=getattr(model,'_selected_backend_device',None)
                if selected_device is not None and selected_device!=runtime.device:raise RuntimeError('device changed after preflight')
            try:last_metrics=check_fit(model,target,contract,inference)
            except AssertionError as exc:
                row.update(status='failed',correctness={'status':'fail','reason':str(exc),'reference':contract['reference']})
                return row
            if i>warmup:samples.append(elapsed)
        row.update(status='passed',correctness={'status':'pass','metrics':last_metrics,'reference':contract['reference']},
                   solver={'requested':str(getattr(model,'solver','not exposed')),'resolved':str(getattr(model,'_selected_solver','not exposed')),'n_iter':str(getattr(model,'n_iter_','not exposed'))},
                   backend_provenance={'untimed_preflight_numerical_dispatch_arrays':observed,'compute_dtype':'not inferred from input dtype'},
                   timing={'scope':contract['scope'],'raw_seconds':samples,'median_seconds':statistics.median(samples),'warmup':warmup,'repeats':repeats,
                           'correctness_preflight':1,'synchronized_device':None if runtime.name=='numpy' else runtime.device,
                           'transfer_policy':'native allocation excluded; public internal conversion/reporting included',
                           'instrumentation':'numerical-dispatch profiler only in untimed preflight; timed fits unprofiled; same factory, native arrays, device context; reporting included'})
        return row


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--backends',nargs='+',choices=DEVICE,default=['numpy','cupy','torch'])
    p.add_argument('--n',type=int,default=500);p.add_argument('--p',type=int,default=8);p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--device',type=int,default=0)
    p.add_argument('--smoke',action='store_true');p.add_argument('--require-backends',action='store_true');p.add_argument('--output',type=Path,required=True)
    return p


def main(argv=None):
    ap=parser();a=ap.parse_args(argv)
    if a.smoke:a.n,a.p,a.backends,a.warmup,a.repeats=48,3,['numpy'],0,1
    if a.n<=a.p+2 or a.p<1 or a.repeats<1 or a.warmup<0 or a.device<0:ap.error('Require n>p+2, p/repeats>=1 and warmup/device>=0')
    if Path(statgpu.__file__).resolve().parent!=ROOT/'statgpu':raise RuntimeError('wrong statgpu import source')
    source=source_identity();rows=[];backends={}
    for backend in a.backends:
        try:rt=Runtime(backend,a.device)
        except BackendUnavailable as exc:
            rows.extend({'case':c,'backend':backend,'status':'skipped','reason':str(exc)} for c in a.cases);continue
        except Exception as exc:
            rows.extend({'case':c,'backend':backend,'status':'failed','reason':str(exc)} for c in a.cases);continue
        backends[backend]={'device':rt.device,**rt.hardware}
        for case in a.cases:
            try:rows.append(run_case(case,rt,a.n,a.p,a.seed,a.warmup,a.repeats))
            except ImportError as exc:rows.append({'case':case,'backend':backend,'status':'skipped','reason':f'reference dependency unavailable: {exc}'})
            except Exception as exc:rows.append({'case':case,'backend':backend,'status':'failed','reason':f'{type(exc).__name__}: {exc}'})
    after=source_identity();changed=source['worktree_fingerprint']!=after['worktree_fingerprint'] or source['git_commit']!=after['git_commit']
    failed=changed or any(r['status'] in ('failed','blocked') for r in rows) or (a.require_backends and any(r['status']=='skipped' for r in rows))
    status='failed' if failed else ('partial' if any(r['status']=='skipped' for r in rows) else 'complete')
    result={'schema_version':'statgpu-benchmark-v2','benchmark':{'name':'supervised-family-coverage','script':str(Path(__file__).relative_to(ROOT)),'command':[sys.executable,*sys.argv]},
            'source':source,'source_after':after,'environment':environment(),'backends':backends,'rows':rows,
            'validation':{'status':status,'tier':'local-minimal' if a.smoke else 'benchmark-candidate','canonical_ready':False,
                          'uncovered_reasons':['CV delegated to benchmark_cv_models.py; not an exhaustive loss/penalty/solver Cartesian matrix',
                                               'No external-library timing or new GPU physical evidence implied','GPU route without observed numerical arrays is blocked, not timed']}}
    for package in ('scipy','scikit-learn','statsmodels'):
        try:result['environment'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:result['environment'][package]=None
    result['environment']['environment_id']=hashlib.sha256(json.dumps({'environment':result['environment'],'backends':backends},sort_keys=True).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':status,'rows':len(rows),'passed':sum(r['status']=='passed' for r in rows),'output':str(a.output)}))
    return 2 if failed else 0

if __name__=='__main__':raise SystemExit(main())
