#!/usr/bin/env python3
"""Phase 11: cost-ledgered published-method components on *synthetic* safety outcomes.

DO NOT describe this as a faithful full AM-PPI or MultiPPI author implementation,
or as validation of latent Y in the real reviewer database. No raw real cases read.
Only `active_seconds` duration observations are read from the optional SQLite DB.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sqlite3
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta

METHODS = ('whole_srs','allowed_srs','ppi_plus_tuned','stratppi_mean','multipppi_subset_gls','active_aipw_restricted')
SCENARIOS=('perfect', 'independent_errors','correlated_errors')
SUBSETS=((0,),(0,1),(0,2),(0,1,2),(1,),(2,),(1,2))

@dataclass
class Population:
    y:np.ndarray
    allowed:np.ndarray
    f0:np.ndarray
    f1:np.ndarray
    shared_flip:np.ndarray
    @property
    def N(self):return len(self.y)
    @property
    def idx_allowed(self):return np.flatnonzero(self.allowed)
    @property
    def q(self):return self.allowed.mean()
    def z_expect(self,scenario):
        if scenario=='perfect':return self.y.astype(float)
        base=np.where(self.y==1,0.80,0.04)
        if scenario=='correlated_errors':return np.where(self.shared_flip,1-self.y,base)
        return base


def make_population(seed:int, N:int=1687, M:int=1557):
    rng=np.random.default_rng(seed)
    y=np.zeros(N,dtype=np.int8)
    # Fixed 34/1557 and 100/4000 missed synthetic harms, not archival truth.
    harm_allow=34 if (N,M)==(1687,1557) else max(1,round(.025*M))
    harm_block=max(1,round(.13*(N-M)))
    y[rng.choice(M,harm_allow,replace=False)]=1
    y[M+rng.choice(N-M,harm_block,replace=False)]=1
    # Latent difficult cases cause *correlated proxy errors*.
    difficulty=rng.normal(0,1,N)
    shared=rng.normal(0,1,N)
    logit0=-3.1+4.1*y+1.15*difficulty+0.65*shared+rng.normal(0,1,N)
    logit1=-4.0+5.7*y+0.55*difficulty+0.35*shared+rng.normal(0,.75,N)
    sig=lambda x:1/(1+np.exp(-np.clip(x,-35,35)))
    allowed=np.arange(N)<M
    return Population(y,allowed,sig(logit0),sig(logit1),rng.random(N)<.045)


def read_review_durations(db:Path|None):
    if db is None:
        # Deterministic illustrative fallback only, not empirical durations.
        return np.array([20.,27.,35.,42.,58.,75.,101.,143.],dtype=float),'ILLUSTRATIVE_NOT_EMPIRICAL'
    if not db.exists():raise FileNotFoundError(f'Reviewer DB missing: {db}')
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as cx:
        cx.execute('PRAGMA query_only=ON')
        cols={r[1] for r in cx.execute('PRAGMA table_info(ratings)')}
        if not {'active_seconds','rater'}.issubset(cols):raise ValueError('ratings schema missing time columns')
        rows=cx.execute('SELECT active_seconds FROM ratings').fetchall()
    d=np.array([float(t[0]) for t in rows],float)
    if len(d)!=336 or np.any(~np.isfinite(d)) or np.any(d<=0):
        raise ValueError('Expected 336 finite positive active-time records; no silent dropping')
    return d,'OBSERVED_336_BROWSER_ACTIVE_SECONDS_RESAMPLED'

class Ledger:
    """Hard cap on explicitly declared human-seconds-equivalent budget units.
    Human and compute costs always reported separately. A new human action is
    reserved against the maximum *observed* time; no realized overspending.
    """
    def __init__(self,budget,durations,cheap,expensive,rng):
        self.budget=float(budget);self.durations=durations
        self.cheap=float(cheap);self.expensive=float(expensive);self.rng=rng
        self.review_sec=0.;self.proxy_cost=0.;self.cheap_calls=0;self.exp_calls=0
        self.calibration_reviews=0;self.inference_reviews=0
        self.calibration_human_sec=0.;self.inference_human_sec=0.
        self.calibration_proxy_cost=0.;self.inference_proxy_cost=0.
        self.actions=0;self.failed_attempts=0
    @property
    def spent(self):return self.review_sec+self.proxy_cost
    @property
    def remaining(self):return self.budget-self.spent
    @property
    def max_review(self):return float(np.max(self.durations))
    def can(self,p0=0,p1=0,human=False):
        return self.remaining+1e-8 >= p0*self.cheap+p1*self.expensive+(self.max_review if human else 0)
    def buy(self,p0=0,p1=0,human=False,calibration=False):
        if not self.can(p0,p1,human):
            self.failed_attempts+=1;return False
        cost=p0*self.cheap+p1*self.expensive
        self.cheap_calls+=int(p0);self.exp_calls+=int(p1)
        self.proxy_cost+=cost
        if calibration:self.calibration_proxy_cost+=cost
        else:self.inference_proxy_cost+=cost
        if human:
            t=float(self.rng.choice(self.durations));self.review_sec+=t
            if calibration:self.calibration_reviews+=1;self.calibration_human_sec+=t
            else:self.inference_reviews+=1;self.inference_human_sec+=t
            self.actions+=1
        if self.spent>self.budget+1e-7:raise AssertionError('Hard budget violated')
        return True
    def report(self):
        return {k:round(v,7) if isinstance(v,float) else v for k,v in {
            'budget_equivalent_units':self.budget,'actual_total_equivalent_units':self.spent,
            'human_active_seconds':self.review_sec,'calibration_human_active_seconds':self.calibration_human_sec,
            'estimation_human_active_seconds':self.inference_human_sec,
            'proxy_cost_equivalent_units':self.proxy_cost,'calibration_proxy_cost_units':self.calibration_proxy_cost,
            'estimation_proxy_cost_units':self.inference_proxy_cost,
            'cheap_proxy_calls':self.cheap_calls,'expensive_proxy_calls':self.exp_calls,
            'calibration_review_actions':self.calibration_reviews,
            'estimation_review_actions':self.inference_reviews,
            'hard_budget_ok':bool(self.spent<=self.budget+1e-7),'unused_equivalent_units':max(0,self.remaining)
        }.items()}

def review(pop,i,scenario,rng):
    if scenario=='perfect':return float(pop.y[i])
    if scenario=='correlated_errors' and pop.shared_flip[i]:return float(1-pop.y[i])
    p=.80 if pop.y[i] else .04
    return float(rng.random()<p)


def result(name,mu,se,pop,scenario,ledger,n,ci_half=None,ci_low=None,ci_high=None,ci_name="Wald_diagnostic_only",**extra):
    if not np.isfinite(mu) or not np.isfinite(se) or n<2:
        return {'method':name,'valid':False,'reason':'insufficient samples / invalid variance','ledger':ledger.report()}
    q=pop.q
    # mu,se here prevalence in FULL N scale (already scaled for allowed-only methods).
    if ci_low is not None and ci_high is not None:
        lo=float(ci_low);hi=float(ci_high)
    else:
        half=1.95996398454005*se if ci_half is None else float(ci_half)
        lo=mu-half;hi=mu+half
    target_Y=float(np.mean(pop.y*pop.allowed));target_Z=float(np.mean(pop.z_expect(scenario)*pop.allowed))
    return {'method':name,'valid':True,'estimate':float(mu),'se':float(se),
      'ci_low':float(lo),'ci_high':float(hi),'ci_width':float(hi-lo),
      'target_latent_Y':target_Y,'target_reviewer_Z':target_Z,
      'squared_error_Y':float((mu-target_Y)**2),'squared_error_Z':float((mu-target_Z)**2),
      'covers_Y_diagnostic':bool(lo<=target_Y<=hi),'covers_reviewer_Z':bool(lo<=target_Z<=hi),
      'sampled_inference_draws':int(n),'ci_procedure':ci_name,'ledger':ledger.report(),**extra}

def make_ledger(budget,durations,c0,c1,rng):return Ledger(budget,durations,c0,c1,rng)

def srs(pop,scenario,ledger,rng,whole=False):
    indices=np.arange(pop.N) if whole else pop.idx_allowed
    observations=[]
    while ledger.can(human=True):
        i=int(rng.choice(indices))
        if not ledger.buy(human=True):break
        observations.append(review(pop,i,scenario,rng)*(float(pop.allowed[i]) if whole else 1.))
    if len(observations)<2:return result('whole_srs' if whole else 'allowed_srs',np.nan,np.nan,pop,scenario,ledger,len(observations))
    vals=np.array(observations);q=1 if whole else pop.q
    k=int(round(vals.sum()));n=len(vals)
    lower=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
    upper=1. if k==n else float(beta.ppf(.975,k+1,n-k))
    return result('whole_srs' if whole else 'allowed_srs',q*vals.mean(),q*vals.std(ddof=1)/math.sqrt(n),pop,scenario,ledger,n,
                  ci_low=q*lower,ci_high=q*upper,ci_name='Clopper_Pearson_95')

def fit_lambda(y,f):
    if len(y)<3 or np.var(f)<1e-10:return 0.
    return float(np.clip(np.cov(y,f,ddof=1)[0,1]/np.var(f,ddof=1),-2,3))

def ppi_plus(pop,scenario,ledger,rng,pilot=12):
    ids=pop.idx_allowed
    if not ledger.buy(p0=len(ids)):return result('ppi_plus_tuned',np.nan,np.nan,pop,scenario,ledger,0)
    ys=[];fs=[]
    for _ in range(pilot):
        if not ledger.buy(human=True,calibration=True):break
        i=int(rng.choice(ids));ys.append(review(pop,i,scenario,rng));fs.append(pop.f0[i])
    if len(ys)<5:return result('ppi_plus_tuned',np.nan,np.nan,pop,scenario,ledger,0)
    lam=fit_lambda(np.array(ys),np.array(fs))
    residual=[]
    while ledger.can(human=True):
        i=int(rng.choice(ids));ledger.buy(human=True)
        residual.append(review(pop,i,scenario,rng)-lam*pop.f0[i])
    if len(residual)<2:return result('ppi_plus_tuned',np.nan,np.nan,pop,scenario,ledger,len(residual))
    a=np.array(residual);q=pop.q
    hb=q*(1+abs(lam))*math.sqrt(math.log(40)/(2*len(a)))
    return result('ppi_plus_tuned',q*(lam*pop.f0[ids].mean()+a.mean()),q*a.std(ddof=1)/math.sqrt(len(a)),pop,scenario,ledger,len(a),
                  ci_half=hb,ci_name='Conditional_pilot_Hoeffding_95',lambda_tuned=lam)

def stratppi(pop,scenario,ledger,rng,pilot_each=7):
    ids=pop.idx_allowed;N=len(ids)
    if not ledger.buy(p0=N):return result('stratppi_mean',np.nan,np.nan,pop,scenario,ledger,0)
    # Frozen proxy-threshold strata, no response labels needed to define bins.
    threshold=float(np.quantile(pop.f0[ids],.85))
    strata=[ids[pop.f0[ids]<threshold],ids[pop.f0[ids]>=threshold]]
    weights=np.array([len(s)/N for s in strata]);lam=[];sd=[]
    for s in strata:
        y=[];f=[]
        for _ in range(pilot_each):
            if not ledger.buy(human=True,calibration=True):break
            i=int(rng.choice(s));y.append(review(pop,i,scenario,rng));f.append(pop.f0[i])
        if len(y)<4:return result('stratppi_mean',np.nan,np.nan,pop,scenario,ledger,0)
        l=fit_lambda(np.array(y),np.array(f));lam.append(l)
        res=np.array(y)-l*np.array(f)
        sd.append(max(.04,float(res.std(ddof=1))))
    alpha=weights*np.array(sd);alpha=alpha/alpha.sum()
    residual=[[],[]]
    # Prevent zero sampling in the smaller but higher-risk stratum.
    for h in (0,1):
        for _ in range(2):
            if not ledger.buy(human=True):break
            i=int(rng.choice(strata[h]));residual[h].append(review(pop,i,scenario,rng)-lam[h]*pop.f0[i])
    while ledger.can(human=True):
        h=int(rng.choice(2,p=alpha));i=int(rng.choice(strata[h]))
        if not ledger.buy(human=True):break
        residual[h].append(review(pop,i,scenario,rng)-lam[h]*pop.f0[i])
    if min(len(x) for x in residual)<2:return result('stratppi_mean',np.nan,np.nan,pop,scenario,ledger,0)
    means=[];v=0.
    for h in range(2):
        arr=np.array(residual[h]);means.append(lam[h]*pop.f0[strata[h]].mean()+arr.mean())
        v+=weights[h]**2*arr.var(ddof=1)/len(arr)
    q=pop.q
    hb=q*sum(weights[h]*(1+abs(lam[h]))*math.sqrt(math.log(80)/(2*len(residual[h]))) for h in range(2))
    return result('stratppi_mean',q*float(weights@means),q*math.sqrt(max(0,v)),pop,scenario,ledger,sum(map(len,residual)),
                  ci_half=hb,ci_name='Stratum_Hoeffding_Bonferroni_95',lambda_by_stratum=lam,threshold_proxy=threshold)

def ridge_predict(X,y,lamb=.6):
    X=np.asarray(X);y=np.asarray(y)
    A=np.column_stack([np.ones(len(X)),X]);reg=np.eye(A.shape[1])*lamb;reg[0,0]=.01
    coef=np.linalg.solve(A.T@A+reg,A.T@y)
    return coef

def apply_coef(X,c):return np.column_stack([np.ones(len(X)),np.asarray(X)])@c

def covariance(X):
    C=np.cov(X,rowvar=False,ddof=1)
    if np.ndim(C)!=2:raise ValueError('Expected pilot with three variables')
    # Nonzero shrinkage recorded and fixed before trial; stabilizes tiny pilot.
    return .80*C+.20*np.diag(np.diag(C))+.02*np.eye(3)

def multi_matrices(Sigma):
    mats=[]
    for subset in SUBSETS:
        idx=np.array(subset,int);P=np.eye(3)[idx,:]
        cov=Sigma[np.ix_(idx,idx)]
        Q=P.T@np.linalg.solve(cov,P)
        mats.append(Q)
    return mats

def optimize_multi(Sigma,costs,seed=0):
    mats=multi_matrices(Sigma);a=np.array([1.,0,0]);cost=np.array(costs,float)
    def objective(frac):
        M=sum((frac[j]/cost[j])*mats[j] for j in range(len(mats)))+1e-8*np.eye(3)
        try:return float(a@np.linalg.solve(M,a))
        except np.linalg.LinAlgError:return 1e15
    cons=[{'type':'eq','fun':lambda u:float(np.sum(u)-1)},
          {'type':'ineq','fun':lambda u:float(np.sum(u[:4])-.12)}]
    starts=[np.array([1.,0,0,0,0,0,0]),
            np.array([.4,.1,.05,.2,.1,.05,.1]),
            np.array([.25,.25,.15,.15,.08,.06,.06])]
    best=None
    for x in starts:
        sol=minimize(objective,x,bounds=[(0,1)]*7,constraints=cons,method='SLSQP',
                     options={'maxiter':170,'ftol':1e-9})
        if sol.success and np.all(sol.x>=-1e-5) and (best is None or sol.fun<best.fun):best=sol
    if best is None:return starts[0],False
    v=np.maximum(0,best.x);return v/v.sum(),True

def pilot_all(pop,scenario,ledger,rng,n=16):
    items=[];ids=pop.idx_allowed
    for _ in range(n):
        if not ledger.buy(p0=1,p1=1,human=True,calibration=True):break
        i=int(rng.choice(ids))
        items.append((review(pop,i,scenario,rng),pop.f0[i],pop.f1[i]))
    return np.array(items,float).reshape(-1,3)

def multipppi(pop,scenario,ledger,rng,pilot=16):
    train=pilot_all(pop,scenario,ledger,rng,pilot)
    if len(train)<9:return result('multipppi_subset_gls',np.nan,np.nan,pop,scenario,ledger,0)
    Sigma=covariance(train)
    # Full subset acquisition + constrained covariance-information allocation:
    # MultiPPI 2026 Sec.4 / Eq.(7), with explicit pilot charge and integral rounding.
    expected=[(ledger.durations.mean() if 0 in s else 0)+int(1 in s)*ledger.cheap+int(2 in s)*ledger.expensive for s in SUBSETS]
    fractions,solved=optimize_multi(Sigma,expected)
    planned=np.floor(ledger.remaining*fractions/np.array(expected)).astype(int)
    # Guarantee at least a few gold observations under integer rounding.
    if np.sum(planned[:4])<3:planned[0]=max(3,planned[0])
    index_pool=pop.idx_allowed
    observed=[[] for _ in SUBSETS]
    for j in np.argsort(-fractions):
        sub=SUBSETS[j]
        for _ in range(int(planned[j])):
            if not ledger.can(int(1 in sub),int(2 in sub),0 in sub):break
            i=int(rng.choice(index_pool))
            if not ledger.buy(int(1 in sub),int(2 in sub),0 in sub):break
            vec=np.array([review(pop,i,scenario,rng) if 0 in sub else np.nan,pop.f0[i],pop.f1[i]])
            observed[j].append(vec[list(sub)])
    # Remaining available budget is used for the gold-only arm (no hidden oracle).
    while ledger.can(human=True):
        i=int(rng.choice(index_pool));ledger.buy(human=True)
        observed[0].append(np.array([review(pop,i,scenario,rng)]))
    M=np.zeros((3,3));b=np.zeros(3);n_y=0
    for j,obs in enumerate(observed):
        if len(obs)==0:continue
        s=SUBSETS[j];P=np.eye(3)[list(s),:];C=Sigma[np.ix_(s,s)]
        I=np.linalg.solve(C,np.eye(len(s)))
        M+=len(obs)*P.T@I@P
        b+=len(obs)*P.T@I@np.array(obs).mean(axis=0)
        if 0 in s:n_y+=len(obs)
    if n_y<2:return result('multipppi_subset_gls',np.nan,np.nan,pop,scenario,ledger,0)
    Mi=np.linalg.pinv(M,rcond=1e-11)
    mu_est=float((Mi@b)[0]);se=math.sqrt(max(0,float(Mi[0,0])))
    a=np.array([1.,0.,0.]);projection=np.zeros(3);h_sumsq=0.
    for j,obs in enumerate(observed):
        if not obs:continue
        subset=SUBSETS[j];P=np.eye(3)[list(subset),:];S=Sigma[np.ix_(subset,subset)]
        w=a@Mi@(len(obs)*P.T@np.linalg.inv(S))
        projection+=w@P
        h_sumsq+=(np.sum(np.abs(w))**2)/len(obs)
    if np.linalg.norm(projection-a)>1e-5:
        return result('multipppi_subset_gls',np.nan,np.nan,pop,scenario,ledger,n_y,
                      ci_name='Invalid_MultiPPI_unbiasedness_constraint')
    hb=pop.q*math.sqrt(math.log(40)/2*h_sumsq)
    return result('multipppi_subset_gls',pop.q*mu_est,pop.q*se,pop,scenario,ledger,n_y,ci_half=hb,
        ci_name='Conditional_pilot_multiarm_Hoeffding_95',
        burnin_covariance_shrinkage=.20,optimizer_success=bool(solved),subset_counts={''.join('YCE'[k] for k in s):len(observed[j]) for j,s in enumerate(SUBSETS)})

def active_aipw(pop,scenario,ledger,rng,pilot=16):
    ids=pop.idx_allowed;N=len(ids)
    train=pilot_all(pop,scenario,ledger,rng,pilot)
    if len(train)<9:return result('active_aipw_restricted',np.nan,np.nan,pop,scenario,ledger,0)
    if not ledger.buy(p0=N):return result('active_aipw_restricted',np.nan,np.nan,pop,scenario,ledger,0)
    b0=ridge_predict(train[:,1:2],train[:,0]);b1=ridge_predict(train[:,1:3],train[:,0])
    m0=np.clip(apply_coef(pop.f0[ids,None],b0),0,1)
    # Do not inspect expensive scores before selecting and paying for those cases.
    r0=max(.01,float(np.mean((train[:,0]-np.clip(apply_coef(train[:,1:2],b0),0,1))**2)))
    r1=max(.01,float(np.mean((train[:,0]-np.clip(apply_coef(train[:,1:3],b1),0,1))**2)))
    # Prior-cost proxy acquisition is conditioned on only the already observed cheap proxy.
    # The full AM-PPI paper solves Eq(5),(7),(8),(9) jointly. This is a RESTRICTED,
    # fixed-cost heuristic routing + Hansen-Hurwitz AIPW implementation, NOT full AM-PPI.
    affordable_exp=max(0,min(N,int(ledger.remaining*.18/max(ledger.expensive,1e-8))))
    risk=np.abs(pop.f0[ids]-.5)
    expensive_pos=np.argsort(risk,kind='mergesort')[:affordable_exp]
    mask=np.zeros(N,dtype=bool);mask[expensive_pos]=True
    if affordable_exp and not ledger.buy(p1=affordable_exp):return result('active_aipw_restricted',np.nan,np.nan,pop,scenario,ledger,0)
    pred=m0.copy()
    if affordable_exp:
        pos=np.flatnonzero(mask)
        pred[pos]=np.clip(apply_coef(np.column_stack([pop.f0[ids[pos]],pop.f1[ids[pos]]]),b1),0,1)
    risk_scale=1+2*np.exp(-np.abs(pop.f0[ids]-.5)*8)
    residual_scale=np.where(mask,r1,r0)*risk_scale
    p=np.sqrt(residual_scale);p=.5*p/p.sum()+.5/N
    terms=[]
    while ledger.can(human=True):
        j=int(rng.choice(N,p=p));i=int(ids[j]);ledger.buy(human=True)
        terms.append((review(pop,i,scenario,rng)-pred[j])/(N*p[j]))
    if len(terms)<2:return result('active_aipw_restricted',np.nan,np.nan,pop,scenario,ledger,len(terms))
    ts=np.asarray(terms);q=pop.q
    range_t=2/(N*np.min(p))
    hb=q*range_t*math.sqrt(math.log(40)/(2*len(ts)))
    return result('active_aipw_restricted',q*(pred.mean()+ts.mean()),q*ts.std(ddof=1)/math.sqrt(len(ts)),pop,scenario,ledger,len(ts),
                  ci_half=hb,ci_name='Conditional_pilot_importance_Hoeffding_95',
                  expensive_routed_cases=int(affordable_exp),aipw_design='iid_with_replacement_importance_sampling_not_paper_Bernoulli')

METHOD_FUN={'whole_srs':lambda p,s,l,r: srs(p,s,l,r,True),
            'allowed_srs':lambda p,s,l,r:srs(p,s,l,r,False),
            'ppi_plus_tuned':ppi_plus,'stratppi_mean':stratppi,
            'multipppi_subset_gls':multipppi,'active_aipw_restricted':active_aipw}

def run_trial(pop,scenario,budget,durations,c0,c1,seed):
    items=[]
    # Separate independent per-method streams; common population and review assumptions.
    for j,name in enumerate(METHODS):
        rng=np.random.default_rng(seed+193*j)
        led=Ledger(budget,durations,c0,c1,rng)
        try:
            x=METHOD_FUN[name](pop,scenario,led,rng)
        except Exception as e:
            raise RuntimeError(f'Method {name} failed for seed={seed}: {e}') from e
        if led.spent>budget+1e-7:raise AssertionError('Budget violation')
        x['scenario']=scenario
        items.append(x)
    return items

def summarise(raw,pop_count,reps):
    out={}; rows=[]
    for scenario in SCENARIOS:
        out[scenario]={}
        for name in METHODS:
            v=[x for x in raw if x['scenario']==scenario and x['method']==name]
            good=[x for x in v if x['valid']]
            if not good:
                out[scenario][name]={'n_runs':len(v),'valid_runs':0,'failed_runs':len(v)};continue
            stat=lambda key:float(np.mean([x[key] for x in good]))
            ls=[x['ledger'] for x in good]
            ave=lambda key:float(np.mean([x[key] for x in ls]))
            record={'n_runs':len(v),'valid_runs':len(good),'failed_runs':len(v)-len(good),
                    'rmse_vs_synthetic_latent_Y':math.sqrt(stat('squared_error_Y')),
                    'rmse_vs_reviewer_endpoint_Z':math.sqrt(stat('squared_error_Z')),
                    'mean_ci_width':stat('ci_width'),
                    'empirical_ci_coverage_reviewer_Z':stat('covers_reviewer_Z'),
                    'empirical_ci_coverage_latent_Y_DIAGNOSTIC_ONLY':stat('covers_Y_diagnostic'),
                    'mean_total_equivalent_cost':ave('actual_total_equivalent_units'),
                    'max_total_equivalent_cost':max(x['actual_total_equivalent_units'] for x in ls),
                    'mean_human_active_seconds':ave('human_active_seconds'),
                    'mean_calibration_human_seconds':ave('calibration_human_active_seconds'),
                    'mean_review_actions':ave('calibration_review_actions')+ave('estimation_review_actions'),
                    'mean_calibration_review_actions':ave('calibration_review_actions'),
                    'mean_proxy_cost_units':ave('proxy_cost_equivalent_units'),
                    'mean_cheap_proxy_calls':ave('cheap_proxy_calls'),
                    'mean_expensive_proxy_calls':ave('expensive_proxy_calls'),
                    'mean_unused_units':ave('unused_equivalent_units'),
                    'hard_budget_violation_count':sum(not x['hard_budget_ok'] for x in ls)}
            out[scenario][name]=record;rows.append((scenario,name,record))
    # Paired bootstrap over genuinely independently generated populations, NOT trials.
    # The paired difference is RMSE(method)-RMSE(allowed SRS), negative favors method.
    rng_boot=np.random.default_rng(994442)
    for s in SCENARIOS:
        base={(v['population_index'],v['rep_index']):v for v in raw
              if v['scenario']==s and v['method']=='allowed_srs' and v['valid']}
        for name in METHODS:
            if name=='allowed_srs' or not out[s][name].get('valid_runs'):continue
            other={(v['population_index'],v['rep_index']):v for v in raw
                   if v['scenario']==s and v['method']==name and v['valid']}
            common=base.keys() & other.keys()
            paired_pops=sorted({k[0] for k in common})
            paired_data={p:([other[k]['squared_error_Y'] for k in sorted(common) if k[0]==p],
                            [base[k]['squared_error_Y'] for k in sorted(common) if k[0]==p])
                         for p in paired_pops}
            if len(paired_pops)<2:continue
            def rmse_delta(pop_sequence):
                aa=[];bb=[]
                for pp in pop_sequence:
                    ap,bp=paired_data[pp];aa.extend(ap);bb.extend(bp)
                return float(math.sqrt(np.mean(aa))-math.sqrt(np.mean(bb)))
            delta=rmse_delta(paired_pops)
            boot=np.array([rmse_delta(rng_boot.choice(paired_pops,size=len(paired_pops),replace=True))
                           for _ in range(800)])
            out[s][name]['paired_rmse_diff_Y_vs_allowed_srs']=delta
            out[s][name]['paired_population_bootstrap95_diff_Y']=[float(x) for x in np.quantile(boot,[.025,.975])]
            out[s][name]['paired_populations']=len(paired_pops)
            out[s][name]['population_bootstrap_repetitions']=800
    return out,rows

def render_md(meta,stats):
    lines=['# Phase 11 — Fair-budget estimator comparison (synthetic)','',
     '**Status:** Simulation using synthetic latent Y and simulated reviewer votes, **not** verified real harmfulness.','',
     'This compares mean specializations and a restricted AM-PPI-style estimator; it is **not** a complete official replication of all published algorithms.','',
     '## Design and costs','',f"- Population: {meta['N']} historical-geometry-shaped synthetic cases; allowed {meta['allowed_N']}.",
     f"- Populations: {meta['populations']}; repetitions/population: {meta['repetitions_per_population']}.",
     f"- Hard **normalized human-active-second-equivalent budget**: {meta['budget_units']}; each human review charged observed resampled active time.",
     f"- Review time source: {meta['duration_source']}; observed max {meta['max_recorded_review_sec']:.2f}s.",
     f"- Cheap proxy unit price: {meta['cheap_unit_equivalent']}; expensive: {meta['expensive_unit_equivalent']} (**illustrative conversion assumptions, not directly measured inference costs**).",
     '- Calibration human judgments, proxy acquisitions on calibration cases, full-population proxy passes, and inference-time human judgments are all recorded separately. The fee for generating the common prompt/response pool is excluded for all methods.','',
     '## Results', '',
     '| Scenario | Method | RMSE vs Y | RMSE vs reviewer Z | Mean CI width | Z coverage | Mean spend | Human sec | Cheap calls | Expensive calls |',
     '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SCENARIOS:
        for name in METHODS:
            r=stats[s][name]
            if not r.get('valid_runs'):
                lines.append(f'| {s} | {name} | insufficient | - | - | - | - | - | - | - |');continue
            lines.append(f"| {s} | {name} | {r['rmse_vs_synthetic_latent_Y']:.4f} | {r['rmse_vs_reviewer_endpoint_Z']:.4f} | {r['mean_ci_width']:.4f} | {r['empirical_ci_coverage_reviewer_Z']:.3f} | {r['mean_total_equivalent_cost']:.0f} | {r['mean_human_active_seconds']:.0f} | {r['mean_cheap_proxy_calls']:.0f} | {r['mean_expensive_proxy_calls']:.0f} |")
    lines+=['','## Paired population-bootstrap comparisons (RMSE vs synthetic latent Y)','',
      '| Scenario | Method vs allowed SRS | ΔRMSE | 95% between-population bootstrap interval |',
      '|---|---|---:|---:|']
    for s in SCENARIOS:
        for name in METHODS:
            rr=stats[s][name]
            if 'paired_rmse_diff_Y_vs_allowed_srs' in rr:
                ci=rr['paired_population_bootstrap95_diff_Y']
                lines.append(f"| {s} | {name} | {rr['paired_rmse_diff_Y_vs_allowed_srs']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines+=['','## Method fidelity and evidence limits','',
     '- **SRS**: iid-with-replacement finite-pool mean and **Clopper-Pearson** 95% interval for reviewer-defined binary endpoint.',
     '- **PPI++**: linear mean specialization with independent, fully charged pilot tuning lambda; mean cheap-proxy census acquired; **conditional Hoeffding** 95% CI.',
     '- **StratPPI**: stratified mean-specialization, known stratum masses, stratum-specific pilot-tuned lambdas and pilot-Neyman allocation; **Bonferroni-Hoeffding** reviewer-endpoint 95% CI. Not full author paper M-estimator/CI implementation.',
     '- **MultiPPI subset-GLS**: covariance-information allocation across all seven nonempty subsets of {human, cheap, expensive}, independent subset samples and a charged covariance pilot, based on Cowen-Breen et al. Eq.(7). A conditional, multiarm **Hoeffding** CI is applied after checking the unbiasedness linear constraint; not the official paper CI. Numerical SLSQP integral rounding and tiny-pilot shrinkage are adaptations; **author-code and optimum certification are not established**.',
     '- **Active AIPW restricted**: predictor-subset heuristic based on cheap scores and budget, residual-importance sampling with replacement, charged expensive proxy acquisition, and AIPW mean correction; **conditional Hoeffding** CI. **NOT the full AM-PPI instance-specific Bernoulli, Eq.(5)/(7)/(8)/(9) joint KKT optimization or its published CI**.',
     '- **Latent truth**: known only inside synthetic generation. All real archived reviewer judgments remain unverified; reviewer-endpoint intervals DO NOT cover latent Y under unbounded label noise. The reported methods are finite-sample conservative for reviewer Z under iid-with-replacement sampling, bounded measurements, independent calibration, cost/Y independence and the fixed-population simulation; no real deployment coverage theorem is claimed.',
     '- **No full economic parity claim**: proxy costs use explicit assumed human-second-equivalent rates; they are not observed model-call prices or measured device latency. Recorded active human seconds are not economic wages. Response-generation costs, adjudication and training were not incurred by this simulated experiment; zero is not imputed as measured cost.',
     '- **Reviewer cost independence**: active durations are resampled independently of response difficulty; unverified assumption. Hard cap uses observed maximum per next action, so there are zero budget overspends by construction.','',
     '## Requirements for research claims','',
     '1. Run with verified actual reviewer-time DB (read-only) for empirical duration resampling.',
     '2. Measure cheap/expensive predictor inference costs on deployment hardware and declare explicit conversions to normalized economic budget units.',
     '3. Verify full author implementations / published confidence intervals for MultiPPI and AM-PPI; present restricted adaptations honestly until then.',
     '4. Independently collect qualified, policy-frozen response-level reference outcomes under a probability design for native prospective evaluation, or limit claims to reviewer-defined development endpoints.',
     '5. Freeze code, hashes, assumptions, and prospective measurements before making confirmatory claims.',
     '', '## Primary references',
     '- Fisch et al., NeurIPS 2024: https://arxiv.org/abs/2406.04291',
     '- Cowen-Breen et al., ICLR 2026: https://arxiv.org/abs/2603.27414',
     '- Brawand et al., 2026: https://arxiv.org/abs/2605.08429',
     '- Feng et al., ICLR 2026: https://arxiv.org/abs/2601.20913','']
    return '\n'.join(lines)

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',type=Path,help='Read-only private SQLite for active_seconds only; required for empirical-cost run')
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--populations',type=int,default=8)
    ap.add_argument('--repetitions',type=int,default=25)
    ap.add_argument('--budget',type=float,default=4800)
    ap.add_argument('--cheap-unit-equivalent',type=float,default=.08)
    ap.add_argument('--expensive-unit-equivalent',type=float,default=2.5)
    ap.add_argument('--seed',type=int,default=20261112)
    ap.add_argument('--geometry',choices=['development_1687','phase6_5000'],default='development_1687')
    args=ap.parse_args(argv)
    if args.populations<1 or args.repetitions<1 or args.budget<20 or args.cheap_unit_equivalent<0 or args.expensive_unit_equivalent<=0:
        ap.error('Invalid positive population, replication, budget, proxy parameters')
    if args.out_dir.exists():raise FileExistsError(f'Output directory exists; refusing overwrite: {args.out_dir}')
    durations,duration_source=read_review_durations(args.db)
    N,M=(1687,1557) if args.geometry=='development_1687' else (5000,4000)
    meta={'study':'PHASE11_PUBLISHED_METHOD_COMPONENT_FAIR_BUDGET_SIMULATION',
          'created_utc':datetime.now(timezone.utc).isoformat(),
          'data_scope':'SYNTHETIC_Y_AND_VOTES_ONLY', 'duration_source':duration_source,
          'N':N,'allowed_N':M,'populations':args.populations,
          'repetitions_per_population':args.repetitions,'budget_units':args.budget,
          'cheap_unit_equivalent':args.cheap_unit_equivalent,
          'expensive_unit_equivalent':args.expensive_unit_equivalent,
          'duration_n':len(durations),'mean_review_time_sec':float(durations.mean()),
          'max_recorded_review_sec':float(durations.max()),'seed':args.seed,
          'pilot_labels_charged':True,'full_author_AM_PPI_replication':False,
          'official_MultiPPI_solver_certified':False}
    raw=[]
    for p in range(args.populations):
        pop=make_population(args.seed+1009*p+101,N,M)
        for r in range(args.repetitions):
            for sidx,scenario in enumerate(SCENARIOS):
                completed=run_trial(pop,scenario,args.budget,durations,
                         args.cheap_unit_equivalent,args.expensive_unit_equivalent,
                         seed=args.seed+100000*p+1000*r+13*sidx+71)
                for item in completed:
                    item['population_index']=p;item['rep_index']=r
                raw.extend(completed)
        print(f"Population {p+1}/{args.populations} finished; total method-trials={len(raw)}",flush=True)
    stats,_=summarise(raw,args.populations,args.repetitions)
    args.out_dir.mkdir(parents=True)
    payload={'metadata':meta,'methods':list(METHODS),'scenarios':list(SCENARIOS),'summary':stats,
             'privacy':'Only aggregate scalar results exported; no private cases, individual human durations, or user-provided human labels'}
    (args.out_dir/'phase11_aggregate.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    (args.out_dir/'phase11_aggregate.md').write_text(render_md(meta,stats))
    # Protected: raw contains no private human case or review records; no identifiers, only synthetic scalars.
    # Do not export raw method trials; archive aggregate only.
    print('COMPLETE',args.out_dir,flush=True)
    for scenario in SCENARIOS:
        print('SCENARIO',scenario)
        for name in METHODS:
            d=stats[scenario][name]
            print(name,'RMSE_Y',round(d.get('rmse_vs_synthetic_latent_Y',float('nan')),5),
                         'cost',round(d.get('mean_total_equivalent_cost',float('nan')),1),
                         'valid',d.get('valid_runs',0))

if __name__=='__main__':main()
