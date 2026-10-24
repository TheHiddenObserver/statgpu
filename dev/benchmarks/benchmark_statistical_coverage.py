#!/usr/bin/env python3
"""Executable statistical-family benchmark matrix, not canonical evidence.

Run --smoke for CPU harness validation; --list-cases enumerates actual adapters.
Resident inputs; operation/fit scope includes internal casts and public reporting.
No speedups, Monte Carlo coverage claim, or exhaustive parameter-grid claim.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import warnings
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import statgpu
from dev.benchmarks.benchmark_release_refresh import Runtime, BackendUnavailable, array_identity, environment

DISTRIBUTIONS = {'norm': {}, 't': {'df': 10}, 'chi2': {'df': 5},
    'gamma': {'a': 2.0}, 'beta': {'a': 2.0, 'b': 3.0}, 'f': {'dfn': 3, 'dfd': 20},
    'uniform': {}, 'expon': {}, 'cauchy': {}, 'laplace': {}, 'logistic': {},
    'weibull_min': {'c': 2.0}, 'lognorm': {'s': 1.0}, 'poisson': {'mu': 5},
    'binom': {'n': 20, 'p': .3}}
COVARIANCES = ('EmpiricalCovariance','LedoitWolf','OAS','ShrunkCovariance','MinCovDet','GraphicalLasso','GraphicalLassoCV')
ADJUST = {'bh':'fdr_bh','by':'fdr_by','holm':'holm','bonferroni':'bonferroni','hochberg':'simes-hochberg'}
ANOVA = ('f_oneway','f_welch','f_twoway','tukey_hsd','bonferroni','cohens_f','partial_eta_squared')
FAMILIES = ('distributions','multiple-testing','resampling','covariance','anova','ordered','panel-primitives')
PANEL_PRIMITIVES = ('clustered_covariance','two_way_clustered_covariance','hac_covariance','driscoll_kraay_covariance','hausman_test','pooling_f_test','breusch_pagan_lm_test')

def case_ids():
    ids=[]
    for name in DISTRIBUTIONS:
        ids += [f'distributions/{name}/{m}' for m in ('cdf','sf','ppf','isf','pmf' if name in ('poisson','binom') else 'pdf','rvs')]
        if name in ('norm','t'):ids += [f'distributions/{name}/{m}' for m in ('two_sided_pvalue','two_sided_critical_value')]
    ids += [f'multiple-testing/{api}/{m}' for api in ('adjust','multipletests') for m in ADJUST]
    ids += [f'multiple-testing/combine/{m}' for m in ('fisher','stouffer','cauchy')]
    ids += [f'resampling/bootstrap/{s}' for s in ('iid','stratified','cluster','block')]
    ids += [f'resampling/permutation/{s}' for s in ('iid','stratified','grouped')]
    ids += [f'covariance/{c}' for c in COVARIANCES]
    ids += [f'anova/{a}' for a in ANOVA]
    ids += [f'ordered/{s}' for s in ('logit','probit','base-logit')]
    ids += [f'panel-primitives/{name}' for name in PANEL_PRIMITIVES]
    return ids

def source_identity():
    paths=sorted((ROOT/'statgpu').rglob('*.py'))+[Path(__file__).resolve(), ROOT/'dev/benchmarks/benchmark_release_refresh.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.DEVNULL)
    try:
        head=git('rev-parse','HEAD').decode().strip(); dirty=bool(git('status','--porcelain')); diff=git('diff','HEAD','--binary')
        tracked=all(subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(ROOT))],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 for p in paths[-2:])
    except (OSError,subprocess.CalledProcessError):head=None;dirty=True;tracked=False;diff=b'git-unavailable'
    return {'git_commit':head,'working_tree':'dirty' if dirty or not tracked else 'clean',
        'runner_overlay':not tracked,'statgpu_version':statgpu.__version__,
        'statgpu_import_path':str(Path(statgpu.__file__).resolve()),'file_sha256':hashes,
        'worktree_fingerprint':hashlib.sha256(diff+json.dumps(hashes,sort_keys=True).encode()).hexdigest()}

def host(x):
    if hasattr(x,'detach'):return x.detach().cpu().numpy()
    if hasattr(x,'get'):return x.get()
    return np.asarray(x)

def native(x, runtime):
    ident=({'backend':'numpy','device':'cpu','dtype':str(np.asarray(x).dtype)} if runtime.name=='numpy' and np.isscalar(x) else array_identity(x))
    if ident['backend']!=runtime.name or ident['device']!=runtime.device:
        raise AssertionError(f'Operational output misplaced: {ident}, expected {runtime.name}/{runtime.device}')
    return ident

class ComparisonError(AssertionError):
    """A rejected comparison with machine-readable, full-precision evidence."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(json.dumps(diagnostics, sort_keys=True, allow_nan=False))


def _diagnostic_value(value):
    # JSON cannot represent NaN/Inf. Keep their identity explicit rather than
    # dropping a failed comparison or emitting invalid JSON.
    if isinstance(value, dict):
        return {key: _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    return value


def close(actual, expected, atol=1e-7, rtol=1e-6, *, label=None, context=None):
    a,b=np.asarray(host(actual)),np.asarray(expected)
    try:
        if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError(f'Nonfinite or shape mismatch: {a.shape}/{b.shape}')
        np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
    except AssertionError:
        if label is None:
            raise
        diagnostic = {'quantity':label, 'actual':a.tolist(), 'expected':b.tolist(),
            'actual_shape':list(a.shape), 'expected_shape':list(b.shape),
            'actual_dtype':str(a.dtype), 'expected_dtype':str(b.dtype),
            'atol':atol, 'rtol':rtol, 'context':context or {}}
        if a.shape == b.shape:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                error = np.abs(a.astype(float)-b.astype(float))
                diagnostic['max_abs_error'] = float(np.max(error, initial=0))
                diagnostic['max_relative_error'] = float(np.max(error/np.abs(b), initial=0))
        raise ComparisonError(_diagnostic_value(diagnostic)) from None
    return float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0))

def prepare(case, runtime, n, p, seed, resamples):
    """Return callable, validator, problem metadata, correctness reference."""
    from scipy import stats
    from statgpu import inference
    rng=np.random.default_rng(seed); parts=case.split('/'); family=parts[0]
    problem={'n':n,'p':p,'seed':seed,'input_dtype':'float64'}
    arrays=[]
    if family=='distributions':
        _,name,op=parts; kw=DISTRIBUTIONS[name]; ref=getattr(stats,name)
        q=np.linspace(.02,.98,n); x=q if op in ('ppf','isf','two_sided_critical_value') else ref.ppf(q,**kw)
        if op=='two_sided_pvalue':x=np.abs(x)
        if name in ('poisson','binom') and op not in ('ppf','isf'):x=np.floor(x)
        if op=='two_sided_critical_value':x=np.asarray(.05);problem['operation_elements']=1
        dx=runtime.array(x); arrays=[x]
        obj=inference.get_distribution(name,backend=runtime.name,device=runtime.device if runtime.name=='torch' else None)
        if op=='rvs':
            def call():return obj.rvs(size=n,**kw)
            call.reset_seed=lambda:np.random.seed(seed)
            def check(out):
                ident=native(out,runtime); sample=np.sort(host(out));assert sample.shape==(n,) and np.isfinite(sample).all()
                lo,hi=ref.support(**kw);assert np.all(sample>=lo) and np.all(sample<=hi)
                if name in ('poisson','binom'):
                    assert np.all(sample==np.floor(sample))
                    points=np.unique(sample)
                    error=max(np.max(np.abs(np.searchsorted(sample,points,side='right')/n-ref.cdf(points,**kw))),np.max(np.abs(np.searchsorted(sample,points,side='left')/n-ref.cdf(points-1,**kw))))
                else:
                    prob=ref.cdf(sample,**kw); error=max(np.max(np.arange(1,n+1)/n-prob),np.max(prob-np.arange(n)/n))
                bound=float(np.sqrt(np.log(2/1e-8)/(2*n)));assert error<=bound
                return {'empirical_cdf_sup_error':float(error),'DKW_bound_alpha_1e_minus_8':bound,'output':ident,'execution':'host NumPy RNG followed by output backend conversion'}
            problem['rng_seed']=seed
            reference={'name':f'scipy.stats.{name} support/CDF, DKW goodness-of-fit bound','type':'numerical-invariant','parameters':kw,'limitation':'Fixed-seed distribution diagnostic, not RNG quality certification; host generation and conversion are timed.'}
        else:
            expected=(2*ref.sf(x,**kw) if op=='two_sided_pvalue' else ref.isf(x/2,**kw) if op=='two_sided_critical_value' else getattr(ref,op)(x,**kw))
            def call():return getattr(obj,op)(float(x) if op=='two_sided_critical_value' else dx,**kw)
            def check(out):
                ident=native(out,runtime)
                return {'max_abs_error':close(out,expected,atol=2e-6,rtol=2e-5),'output':ident}
            reference={'name':f'scipy.stats.{name}.{op}','type':'external-python','parameters':kw,'atol':2e-6,'rtol':2e-5,'grid':'central quantiles .02..98; not extreme-tail acceptance'}
    elif family=='multiple-testing':
        vals=rng.uniform(.001,.95,size=n); vals[::9]=.001; dv=runtime.array(vals); arrays=[vals]
        if parts[1] in ('adjust','multipletests'):
            from statsmodels.stats.multitest import multipletests
            method=parts[2]; reject,adjusted,*_=multipletests(vals,method=ADJUST[method]); expected=(reject,adjusted)
            def call():return getattr(inference,'adjust_pvalues' if parts[1]=='adjust' else 'multipletests')(dv,method=method,backend=runtime.name)
        else:
            method=parts[2]
            if method=='cauchy':
                statistic=float(np.mean(np.tan((.5-vals)*np.pi))); expected=(statistic,.5-np.arctan(statistic)/np.pi)
            else:expected=tuple(stats.combine_pvalues(vals,method=method))
            def call():return inference.combine_pvalues(dv,method=method,backend=runtime.name)
        def check(out):
            # Public combination outputs may be native scalar arrays; verify both.
            identities=[native(v,runtime) for v in out]
            return {'max_abs_error':max(close(a,b,atol=1e-7,rtol=1e-6) for a,b in zip(out,expected)),'outputs':identities}
        reference={'name':'statsmodels multipletests / scipy combine_pvalues / analytic equal-weight Cauchy','type':'external-python/analytic','method':parts[2]}
    elif family=='resampling':
        strategy=parts[2]; x=np.linspace(-1,2,n); y=rng.normal(size=n); dx,dy=runtime.array(x),runtime.array(y); arrays=[x,y]
        labels=np.arange(n)%3; kw={}
        if strategy=='stratified':kw['strata']=labels
        if strategy=='cluster':kw['clusters']=np.arange(n)//3
        if strategy=='block':kw['block_size']=3
        if strategy=='grouped':kw['groups']=labels
        def mean(v):return v.mean(axis=-1) if runtime.name!='torch' else v.mean(dim=-1)
        def dot(a,b):return (a*b).mean(axis=-1) if runtime.name!='torch' else (a*b).mean(dim=-1)
        if parts[1]=='bootstrap':
            def call():return inference.bootstrap_statistic(mean,dx,n_resamples=resamples,strategy=strategy,random_state=seed,backend=runtime.name,**kw)
            def check(out):
                ident=native(out.samples,runtime); sample=host(out.samples)
                close(out.observed,x.mean()); assert sample.shape==(resamples,)
                assert np.all(np.isfinite(sample)) and np.min(sample)>=x.min()-1e-12 and np.max(sample)<=x.max()+1e-12
                close(out.confidence_interval,np.quantile(sample,[.025,.975]))
                return {'mean_bounds_and_percentile_interval':True,'samples':ident}
        else:
            def call():return inference.permutation_test(dot,dx,dy,n_resamples=resamples,strategy=strategy,random_state=seed,backend=runtime.name,**kw)
            def check(out):
                ident=native(out.samples,runtime); sample=host(out.samples); close(out.observed,np.mean(x*y)); assert sample.shape==(resamples,) and np.isfinite(sample).all()
                expected=(1+np.count_nonzero(np.abs(sample)>=abs(out.observed)))/(resamples+1)
                close(out.pvalue,expected); return {'plus_one_randomization_pvalue':True,'samples':ident}
        problem['resamples']=resamples
        reference={'name':'observed statistic + resample bounds + percentile interval / plus-one p-value identity','type':'numerical-invariant','limitation':'Does not establish RNG-distribution equivalence, MC interval coverage, or permutation exchangeability.'}
    elif family=='covariance':
        import statgpu.covariance as cov
        from sklearn import covariance as skcov
        name=parts[1]; X=rng.normal(size=(n,p)); X[:,0]+=.2*X[:,-1]; dx=runtime.array(X); arrays=[X]
        kwargs={}
        if name=='ShrunkCovariance':kwargs={'shrinkage':.2}
        if name=='MinCovDet':kwargs={'random_state':seed}
        if name=='GraphicalLasso':kwargs={'alpha':.1,'tol':1e-7,'max_iter':200}
        if name=='GraphicalLassoCV':kwargs={'alphas':[.05,.2],'cv':3,'random_state':seed,'tol':1e-7,'max_iter':200}
        def call():return getattr(cov,name)(device={'numpy':'cpu','cupy':'cuda','torch':'torch'}[runtime.name],**kwargs).fit(dx)
        expected=None
        if name not in ('MinCovDet','GraphicalLassoCV'):expected=getattr(skcov,name)(**kwargs).fit(X).covariance_
        def check(out):
            ident=native(out.covariance_,runtime); actual=host(out.covariance_);close(actual,actual.T)
            assert np.linalg.eigvalsh(actual).min()>0
            close(actual@host(out.precision_),np.eye(p),atol=1e-4,rtol=1e-4)
            if name=='MinCovDet':
                support=host(out.support_).astype(bool); subset=X[support]; assert len(subset)>p
                factor=.975/stats.chi2.cdf(stats.chi2.ppf(.975,p),p+2)
                err=close(actual,np.cov(subset,rowvar=False,bias=True)*factor,atol=1e-5,rtol=1e-5)
            elif name=='GraphicalLassoCV':
                scores=[row['mean_score'] for row in out.cv_results_]; selected=[.05,.2][int(np.argmax(scores))]
                close(out.alpha_,selected)
                ref=skcov.GraphicalLasso(alpha=selected,tol=1e-7,max_iter=200).fit(X).covariance_
                err=close(actual,ref,atol=2e-5,rtol=2e-4)
            else:err=close(actual,expected,atol=2e-5,rtol=2e-4)
            return {'covariance_max_abs_error':err,'native_covariance':ident}
        reference={'name':'sklearn covariance; MinCovDet independent support-reweighting identity; GraphicalLassoCV selected-refit sklearn comparison','type':'external-python/numerical-invariant','parameters':kwargs,'limitation':'MCD optimum and CV fold scores not independently certified by these checks.'}
    elif family=='anova':
        from statgpu import anova
        op=parts[1]; groups=[rng.normal(loc=i*.2,scale=1+i*.15,size=n) for i in range(3)]; arrays=groups; dg=[runtime.array(g) for g in groups]
        observed=[]
        def tracked(value):
            try: ident=array_identity(value)
            except RuntimeError:ident=None
            if ident and ident['backend']==runtime.name and ident['device']==runtime.device:observed.append(ident)
            from statgpu.backends import _to_float_scalar
            return _to_float_scalar(value)
        modules=['statgpu.anova._oneway','statgpu.anova._welch','statgpu.anova._twoway','statgpu.anova._posthoc']
        def call():
            observed.clear()
            with ExitStack() as stack:
                for mod in modules:stack.enter_context(patch.object(importlib.import_module(mod),'_to_float_scalar',tracked))
                if op=='partial_eta_squared':return anova.partial_eta_squared(3.,7.)
                if op=='f_twoway':return anova.f_twoway(device_cells,backend=runtime.name)
                return getattr(anova,op)(*dg,backend=runtime.name)
        if op=='f_twoway':
            cells=rng.normal(size=(2,3,n))+np.arange(2)[:,None,None]*.2;arrays=[cells]
            device_cells=[[runtime.array(cells[i,j]) for j in range(3)] for i in range(2)]
        grand=np.mean(np.concatenate(groups));ssb=sum(len(g)*(g.mean()-grand)**2 for g in groups);ssw=sum(np.sum((g-g.mean())**2) for g in groups)
        def check(out):
            if runtime.name!='numpy' and op!='partial_eta_squared' and not observed:raise AssertionError('No native device reduction observed')
            f_diagnostics = None
            if op in ('f_oneway', 'f_welch'):
                if op == 'f_oneway':
                    ref = stats.f_oneway(*groups)
                    ref_df = (len(groups)-1, sum(len(g) for g in groups)-len(groups))
                else:
                    from statsmodels.stats.oneway import anova_oneway
                    ref = anova_oneway(groups, use_var='unequal')
                    ref_df = ref.df
                f_diagnostics = {'statistic':out.statistic, 'pvalue':out.pvalue,
                    'dfn':out.df_between, 'dfd':out.df_within,
                    'reference_statistic':float(ref.statistic),
                    'reference_pvalue':float(ref.pvalue),
                    'reference_dfn':float(ref_df[0]), 'reference_dfd':float(ref_df[1]),
                    'backend':runtime.name, 'device':runtime.device,
                    'input_dtype':str(groups[0].dtype),
                    'native_reductions':list(observed)}
                err = max(close(out.statistic, ref.statistic, label='statistic', context=f_diagnostics),
                          close(out.pvalue, ref.pvalue, label='pvalue', context=f_diagnostics))
                close(out.df_between, ref_df[0], label='dfn', context=f_diagnostics)
                close(out.df_within, ref_df[1], label='dfd', context=f_diagnostics)
                if op == 'f_oneway':
                    close(out.eta_squared, ssb/(ssb+ssw), label='eta_squared', context=f_diagnostics)
            elif op=='cohens_f':err=close(out,np.sqrt(ssb/ssw))
            elif op=='partial_eta_squared':err=close(out,.3)
            elif op in ('tukey_hsd','bonferroni'):
                errors=[]; tr=stats.tukey_hsd(*groups) if op=='tukey_hsd' else None
                for c in out.comparisons:
                    i,j=c.group_i,c.group_j;pv=tr.pvalue[i,j] if tr is not None else stats.ttest_ind(groups[i],groups[j],equal_var=False).pvalue
                    assert c.reject == bool(pv < (.05 if tr is not None else .05/3))
                    errors += [close(c.mean_diff,groups[i].mean()-groups[j].mean()),close(c.pvalue,pv,atol=1e-6)]
                err=max(errors)
            else:
                mean=cells.mean(); ma=cells.mean(axis=(1,2));mb=cells.mean(axis=(0,2)); mc=cells.mean(axis=2)
                se=np.sum((cells-mc[:,:,None])**2); mse=se/(6*(n-1));sa=3*n*np.sum((ma-mean)**2); sb=2*n*np.sum((mb-mean)**2);si=n*np.sum((mc-ma[:,None]-mb[None,:]+mean)**2)
                err=max(close(out.factor_a_statistic,sa/mse),close(out.factor_b_statistic,sb/2/mse),close(out.interaction_statistic,si/2/mse),close(out.ss_within,se))
            metrics = {'max_abs_error':err,'native_reduction_observed':bool(observed),'host_reporting':True}
            if f_diagnostics is not None:
                metrics['f_distribution'] = f_diagnostics
            return metrics
        reference={'name':'SciPy/statsmodels ANOVA and independent balanced sums-of-squares/effect identities','type':'external-python/analytic','limitation':'Timing includes reduction observation and host scalar distribution reporting.'}
    elif family=='panel-primitives':
        from statgpu import panel
        op=parts[1]; n=4*(n//4);problem['n']=n
        entity=np.repeat(np.arange(n//4),4);period=np.tile(np.arange(4),n//4)
        X=rng.normal(size=(n,p));y=X@np.linspace(-.3,.5,p)+rng.normal(size=n//4)[entity]+rng.normal(scale=.4,size=n)
        resid=y-X@np.linalg.lstsq(X,y,rcond=None)[0];dx,dr=runtime.array(X),runtime.array(resid);arrays=[X,y,entity,period]
        bread=np.linalg.inv(X.T@X);score=X*resid[:,None]
        def clustered(labels):
            grouped=np.stack([score[labels==label].sum(axis=0) for label in np.unique(labels)])
            return bread@(grouped.T@grouped)@bread
        def hac_scores(scores,bandwidth=2):
            meat=scores.T@scores
            for lag in range(1,min(len(scores),bandwidth+1)):
                gamma=scores[lag:].T@scores[:-lag];meat+=(1-lag/(bandwidth+1))*(gamma+gamma.T)
            return bread@meat@bread
        if op=='clustered_covariance':
            expected=clustered(entity)
            def call():return panel.clustered_covariance(dx,dr,entity,xp=runtime.module)
        elif op=='two_way_clustered_covariance':
            expected=clustered(entity)+clustered(period)-clustered(entity*4+period)
            def call():return panel.two_way_clustered_covariance(dx,dr,entity,period,xp=runtime.module)
        elif op=='hac_covariance':
            expected=hac_scores(score)
            def call():return panel.hac_covariance(dx,dr,bandwidth=2,xp=runtime.module)
        elif op=='driscoll_kraay_covariance':
            grouped=np.stack([score[period==j].sum(axis=0) for j in range(4)])
            expected=hac_scores(grouped)*n/(n-p)
            def call():return panel.driscoll_kraay_covariance(dx,dr,period,bandwidth=2,xp=runtime.module)
        else:
            # These public accessors compute from retained host-side model state.
            # Their model preparation is deliberately outside the timed region.
            if op in ('pooling_f_test','hausman_test'):
                fe=panel.PanelOLS(entity_effects=True,device='cpu').fit(X,y,entity_ids=entity)
            if op=='pooling_f_test':
                Xw=X-np.repeat(X.reshape(-1,4,p).mean(axis=1),4,axis=0);yw=y-np.repeat(y.reshape(-1,4).mean(axis=1),4)
                rssfe=np.sum((yw-Xw@np.linalg.lstsq(Xw,yw,rcond=None)[0])**2)
                design=np.column_stack((np.ones(n),X));rsspool=np.sum((y-design@np.linalg.lstsq(design,y,rcond=None)[0])**2)
                dfden=n-p-n//4;dfnum=n//4-1;expected=((rsspool-rssfe)/dfnum)/(rssfe/dfden)
                def call():return panel.pooling_f_test(fe)
            elif op=='breusch_pagan_lm_test':
                pooled=panel.PooledOLS(device='cpu').fit(X,y,entity_ids=entity)
                design=np.column_stack((np.ones(n),X));rr=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
                sums=np.array([rr[entity==g].sum() for g in np.unique(entity)]);a1=sums@sums/(rr@rr)-1;expected=(n*np.sqrt(1/(2*(4*n-n)))*a1)**2
                def call():return panel.breusch_pagan_lm_test(pooled)
            else:
                re=panel.RandomEffects(device='cpu').fit(X,y,entity_ids=entity)
                difference=np.asarray(fe.coef_)[-p:]-np.asarray(re.coef_)[-p:]
                covdiff=np.asarray(fe._panel_cov_params)[-p:,-p:]-np.asarray(re._panel_cov_params)[-p:,-p:]
                eigen=np.linalg.eigvalsh(covdiff);positive=np.min(eigen)>1e-10
                expected=float(difference@np.linalg.solve(covdiff,difference)) if positive else None
                def call():return panel.hausman_test(fe,re)
        if op.endswith('covariance'):
            def check(out):return {'max_abs_error':close(out,expected,atol=1e-8,rtol=1e-6),'native_covariance':native(out,runtime)}
        else:
            def check(out):
                if op=='hausman_test' and expected is None:
                    # An indefinite covariance difference has no classical chi2 statistic.
                    if not out.applicable:
                        assert out.statistic is None and out.pvalue is None and out.reason
                        return {'applicable':False,'reason':out.reason,'statistic':None,'pvalue':None}
                    # Singular PSD extensions require separate range checks, not guessed df.
                    raise AssertionError('Unexpected Hausman applicability for unresolved covariance spectrum')
                assert out.applicable
                err=close(out.statistic,expected,atol=1e-7,rtol=1e-6)
                pv=stats.f.sf(expected,*out.df) if op=='pooling_f_test' else stats.chi2.sf(expected,out.df)
                close(out.pvalue,pv,atol=1e-7,rtol=1e-6)
                return {'applicable':True,'statistic_max_abs_error':err,'public_result':'host metadata diagnostic'}
        reference={'name':'independent sandwich/group-score/HAC/DK equations; fitted-model poolability, BP-LM and Hausman quadratic identities','type':'analytic','limitation':'Classical balanced fixture; diagnostic fit preparation excluded, no claim on singular-PSD extensions.'}
    elif family=='ordered':
        from statgpu.linear_model import OrderedLogitRegression,OrderedProbitRegression
        from statgpu.linear_model._glm_base import OrderedGeneralizedLinearModel
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        kind=parts[1]; link='probit' if kind=='probit' else 'logit'; X=rng.normal(size=(n,p)); latent=X@np.linspace(-.4,.5,p)+(rng.logistic(size=n) if link=='logit' else rng.normal(size=n)); y=np.digitize(latent,[-.5,.6]); dx,dy=runtime.array(X),runtime.array(y);arrays=[X,y]
        base=OrderedGeneralizedLinearModel if kind=='base-logit' else OrderedLogitRegression if kind=='logit' else OrderedProbitRegression
        class Observed(base):
            def _fit_ordered_newton_impl(self,X,y,*a,**kw):
                self.benchmark_input=native(X,runtime);return super()._fit_ordered_newton_impl(X,y,*a,**kw)
        def call():return Observed(n_categories=3,device={'numpy':'cpu','cupy':'cuda','torch':'torch'}[runtime.name],max_iter=300,tol=1e-8,compute_inference=False).fit(dx,dy)
        ref=OrderedModel(y,X,distr=link).fit(method='bfgs',disp=False,maxiter=500,gtol=1e-9)
        expected=ref.model.predict(ref.params,exog=X)
        def check(out):
            assert hasattr(out,'benchmark_input');pred=out.predict_proba(dx)
            err=close(pred,expected,atol=2e-4,rtol=2e-4);close(pred.sum(axis=1),np.ones(n));assert np.all(np.diff(out.thresholds_)>0)
            return {'probability_max_abs_error':err,'fit_ingress':out.benchmark_input,'iterations':int(out.n_iter_)}
        reference={'name':'statsmodels OrderedModel unpenalized maximum likelihood','type':'external-python','parameters':{'distr':link},'atol':2e-4,'rtol':2e-4}
    else:raise ValueError(case)
    dig=hashlib.sha256()
    for arr in arrays:dig.update(str((arr.shape,str(arr.dtype))).encode());dig.update(arr.tobytes())
    problem['data_id']=dig.hexdigest()
    return call,check,problem,reference

def run_case(case,runtime,n,p,seed,resamples,warmup,repeats):
    if (case=='anova/partial_eta_squared' or (case.startswith('panel-primitives/') and case.endswith('_test'))) and runtime.name!='numpy':
        return {'case_id':case,'backend':runtime.name,'status':'not-applicable','reason':'Host-side public diagnostic/scalar API; CPU row covers this operation, estimator backend fit covered separately'}
    with runtime.context(),warnings.catch_warnings():
        warnings.simplefilter('error')
        call,check,problem,reference=prepare(case,runtime,n,p,seed,resamples)
        if hasattr(call,'reset_seed'):call.reset_seed()
        checked=check(call()); samples=[]
        for i in range(warmup+repeats):
            if hasattr(call,'reset_seed'):call.reset_seed()
            runtime.sync(runtime.device);start=__import__('time').perf_counter();out=call();runtime.sync(runtime.device);elapsed=__import__('time').perf_counter()-start
            checked=check(out)
            if i>=warmup:samples.append(elapsed)
    return {'case_id':case,'backend':runtime.name,'device':runtime.device,'status':'passed','problem':problem,
        'correctness':{'status':'pass','reference':reference,'metrics':checked},
        'timing':{'scope':'host-diagnostic-accessor-excluding-model-fit' if case.startswith('panel-primitives/') and case.endswith('_test') else 'host-rng+backend-output-transfer' if case.endswith('/rvs') else 'construct+fit' if case.startswith(('covariance/','ordered/')) else 'public-operation',
            'transfer_policy':'input transfer excluded; internal casts/public host reporting included','warmup':warmup,'repeats':repeats,
            'seconds':samples,'median_seconds':statistics.median(samples),'synchronized_device':runtime.device,'statistic':'median'},
        'performance_claim_eligible':False}

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=['all'],help='Family selectors or exact case IDs; see --list-cases')
    p.add_argument('--list-cases',action='store_true');p.add_argument('--backends',nargs='+',choices=['numpy','cupy','torch'],default=['numpy','cupy','torch'])
    p.add_argument('--sizes',nargs='+',type=int,default=[128,1024]);p.add_argument('--features',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261020);p.add_argument('--resamples',type=int,default=100)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=5);p.add_argument('--cuda-device',type=int,default=0)
    p.add_argument('--smoke',action='store_true');p.add_argument('--require-backends',action='store_true');p.add_argument('--output',type=Path)
    return p

def main(argv=None):
    par=parser();args=par.parse_args(argv)
    if args.list_cases:print(json.dumps(case_ids(),indent=2));return 0
    if args.output is None:par.error('--output is required')
    if args.smoke:args.backends=['numpy'];args.sizes=[48];args.features=3;args.resamples=12;args.warmup=0;args.repeats=1
    if min(args.sizes)<=args.features+2 or args.features<2 or args.repeats<1 or args.warmup<0 or args.resamples<2:par.error('Require sizes>features+2, features>=2, repeats>=1, warmup>=0,resamples>=2')
    known=case_ids();selected=[c for c in known if 'all' in args.cases or c in args.cases or c.split('/')[0] in args.cases]
    if any(c not in ('all',*FAMILIES,*known) for c in args.cases):par.error('Unknown case selector')
    if Path(statgpu.__file__).resolve().parent!=ROOT/'statgpu':raise RuntimeError('Wrong statgpu import source')
    source=source_identity();rows=[];backends={};failed=False
    for name in args.backends:
        try:runtime=Runtime(name,args.cuda_device)
        except BackendUnavailable as exc:
            backends[name]={'status':'unavailable','reason':str(exc)};failed|=args.require_backends
            rows.extend({'case_id':c,'backend':name,'n':n,'status':'skipped','reason':str(exc)} for n in args.sizes for c in selected);continue
        except Exception as exc:
            backends[name]={'status':'failed','reason':f'{type(exc).__name__}: {exc}'};failed=True;continue
        backends[name]={'status':'available','device':runtime.device,**runtime.hardware}
        for n in args.sizes:
            for case in selected:
                try:row=run_case(case,runtime,n,args.features,args.seed,args.resamples,args.warmup,args.repeats)
                except ModuleNotFoundError as exc:
                    missing=(exc.name or '').split('.')[0]
                    if missing in ('sklearn','statsmodels'):
                        row={'case_id':case,'backend':name,'n':n,'status':'skipped','reason':f'Reference dependency unavailable: {exc.name}'}
                    else:
                        row={'case_id':case,'backend':name,'n':n,'status':'failed','reason':f'{type(exc).__name__}: {exc}'};failed=True
                except Exception as exc:
                    row={'case_id':case,'backend':name,'n':n,'status':'failed','reason':f'{type(exc).__name__}: {exc}'};failed=True
                    if isinstance(exc, ComparisonError):
                        row['correctness'] = {'status':'fail', 'comparison':exc.diagnostics}
                if case=='distributions/f/rvs':row['known_issue']='https://github.com/TheHiddenObserver/statgpu/issues/173'
                rows.append(row)
    after=source_identity()
    if after!=source:failed=True
    env=environment()
    from importlib.metadata import version, PackageNotFoundError
    for package in ('scikit-learn','statsmodels'):
        try:env['packages'][package]=version(package)
        except PackageNotFoundError:env['packages'][package]=None
    env['environment_id']=hashlib.sha256(json.dumps({'environment':env,'backends':backends},sort_keys=True).encode()).hexdigest()
    result={'schema_version':'statgpu-benchmark-v2','benchmark':{'name':'statistical-family-coverage','command':[sys.executable,*sys.argv],'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},'source':source,'source_after':after,'environment':env,'backends':backends,'results':rows,
        'validation':{'status':'failed' if failed else ('partial' if any(x['status']=='skipped' for x in rows) else 'complete'), 'tier':'local-minimal' if args.smoke else 'benchmark-candidate','canonical_registered':False,
            'source_changed':after!=source,'failure_reasons':['Source identity changed during execution'] if after!=source else [],
            'uncovered':['No extreme-tail/parameter sweep or exhaustive RNG calibration','Resampling identity checks do not certify statistical coverage','No external timing/speedup, memory peaks, inference-CI or dashboard parser','GPU paths require physical execution; CPU smoke is not CUDA acceptance']}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['validation']['status'],'cases':len(rows),'failed':sum(x['status']=='failed' for x in rows)}))
    return 2 if failed else 0
if __name__=='__main__':raise SystemExit(main())
