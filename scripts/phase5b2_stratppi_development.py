#!/usr/bin/env python3
"""Phase 5B2: restricted mean-estimation StratPPI comparison (development only).

Implements the 1-D stratified prediction-powered mean estimator of Fisch et al.
(NeurIPS 2024) for A=P(Y=1,D=0), B=P(Y=1,D=1), and FNR=A/(A+B).
This is a finite-pool, post-hoc, self-contained estimator benchmark, NOT a
full reproduction of the paper's general M-estimation / CI method, and NOT
Active MultiPPI. Only the previously authorized development parquet is read.

One inexpensive *already available* historical cross-fitted base_score is
used as proxy f for response-level Y. No optional monitor is queried or used.
The fixed-weight lambda=1 arms spend zero tuning labels. Pilot-calibrated
lambda arms reserve 20% of each human-review budget for independent within-D
pilot labels and count them in the total; evaluation draws exclude the pilot.
All estimators of A/B are design-unbiased conditional on the sampled pilot
(when present); FNR is a ratio and does not inherit exact unbiasedness.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BRANCH = 'stage-a-paired-transport-20260921'
SOURCE = 'reports/decision_value_real_data/cross_fitted_decision_value_targets.parquet'
FOLDS = 'reports/decision_value_real_data/development_outer_fold_assignments.csv'
PROTOCOL = 'configs/decision_value_real_data_protocol_v1.json'
BOUNDARY = 'configs/exact_cost_risk_cascade_protocol_v2.json'
HASHES = {
    SOURCE: '0041158b10e7ee0dd1189a0611c5b07f18a44d88355e3c150ac044e4112fe6ec',
    FOLDS: '62a2d8df0ba55b047d7e4685488b4d4d0ef77d480bcdb26cc34e5586e935d2dd',
    PROTOCOL: '7c82846f9de712a56260d74fbc1901b084262f2a7a745007547a75eab3d5f914',
    BOUNDARY: '96a0bfdf1a0954d9313ecd7a2ae1272a07f0df4ba6197eede2f0e0afd9f1c1c7',
}
OUT = 'results/phase5b2_stratppi_mean_development_v1'
ARTIFACT = 'phase5b2_stratppi_mean_development_v1'
BUDGETS = (40,80,160)
REPETITIONS = 400
SEED = 20261014
METHODS = (
    'uniform_srs',
    'allowed_only_srs',
    'proportional_labels_only',
    'allowed70_labels_only',
    'ppi_uniform_fixed1',
    'stratppi_proportional_fixed1',
    'stratppi_allowed70_fixed1',
    'stratppi_proportional_pilot_lambda',
    'stratppi_allowed70_pilot_lambda',
)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20), b''): h.update(x)
    return h.hexdigest()


def downloads():
    try:
        w=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
            '[Environment]::GetFolderPath("UserProfile")'],text=True,timeout=8).strip().replace('\r','')
        q=subprocess.check_output(['wslpath','-u',w],text=True,timeout=5).strip()
        p=Path(q)/'Downloads'
        if p.is_dir(): return p
    except (OSError,ValueError,subprocess.SubprocessError): pass
    p=Path('/mnt/c/Users/user/Downloads')
    return p if p.is_dir() else None


def quotas(b,N0,N1,kind):
    if kind == 'proportional': n0=round(b*N0/(N0+N1))
    elif kind == 'allowed70': n0=round(0.70*b)
    else: raise ValueError('unknown allocation')
    n0=max(2,min(N0,n0)); n1=b-n0
    if n1>N1: n1=N1; n0=b-n1
    if n1<2: n1=2; n0=b-n1
    if min(n0,n1)<2 or n0>N0 or n1>N1 or n0+n1!=b:
        raise ValueError('allocation infeasible')
    return (int(n0),int(n1))


def _component(yy,ff,whole,pilot,eval_indices,lam):
    """Return finite-population total and conditional design variance estimate.

    pilot labels are known; a disjoint SRS among remaining members estimates
    their residual sum. All f are known on the full permitted development pool.
    """
    rem=np.setdiff1d(whole,pilot,assume_unique=True)
    if not len(rem):
        return float(yy[pilot].sum()),0.0
    if len(eval_indices)<2 or len(np.intersect1d(pilot,eval_indices)):
        raise ValueError('two fresh evaluation labels required and no pilot overlap')
    n=len(eval_indices);Nr=len(rem)
    resid=yy[eval_indices]-lam*ff[eval_indices]
    total=float(yy[pilot].sum() + Nr*(lam*float(np.mean(ff[rem])) + float(np.mean(resid))))
    variance=0.0 if n==Nr else (Nr**2)*(1-n/Nr)*float(np.var(resid,ddof=1))/n
    return total,variance


def lambda_from_pilot(y,f,indices):
    """Pilot-only variance reducing coefficient; clipping disclosed.

    The general StratPPI treatment estimates power weights; this one-dimensional
    finite-pool analogue uses a pilot regression slope with no held-out labels.
    """
    x=f[indices].astype(float);v=y[indices].astype(float)
    if len(x)<4 or np.var(x,ddof=1)<1e-12: return 0.0
    slope=float(np.cov(v,x,ddof=1)[0,1]/np.var(x,ddof=1))
    return float(np.clip(slope,-3,3))


def estimate(y,d,f,method,budget,rng):
    N=len(y); groups=(np.flatnonzero(d==0),np.flatnonzero(d==1));N0,N1=map(len,groups)
    a=b=va=vb=0.0
    cal=0
    lams=(0.,0.)
    if method in ('uniform_srs','ppi_uniform_fixed1'):
        selected=rng.choice(N,size=budget,replace=False)
        z=y*(1-d);w=y*d
        fz=f*(1-d);fw=f*d
        lam=float(method=='ppi_uniform_fixed1')
        for outcome,proxy,slot in ((z,fz,0),(w,fw,1)):
            residual=outcome[selected]-lam*proxy[selected]
            val=lam*float(np.mean(proxy))+float(np.mean(residual))
            var=(1-budget/N)*float(np.var(residual,ddof=1))/budget
            if slot==0: a,va=val,var
            else: b,vb=val,var
        lams=(lam,lam)
    elif method=='allowed_only_srs':
        selected=rng.choice(groups[0],size=budget,replace=False)
        a=(N0/N)*float(np.mean(y[selected]))
        va=(N0/N)**2*(1-budget/N0)*float(np.var(y[selected],ddof=1))/budget
        # B intentionally left undefined: cannot infer FNR from allowed labels.
        b=float('nan'); vb=float('nan')
    else:
        if 'allowed70' in method: allocation='allowed70'
        elif 'proportional' in method: allocation='proportional'
        else: raise ValueError('unknown stratification method')
        fixed='fixed1' in method
        calibrated='pilot_lambda' in method
        label_only='labels_only' in method
        if not (fixed or calibrated or label_only): raise ValueError('unknown estimator')
        if calibrated:
            # Every pilot label costs one human review; pilot and estimation
            # labels disjoint. These pilot quotas are fixed ex ante.
            cal=max(8,budget//5)
            cp0=cal//2;cp1=cal-cp0
            if cp1>=N1 or cp0>=N0: raise ValueError('too many pilot labels')
            pilot=(rng.choice(groups[0],size=cp0,replace=False),
                   rng.choice(groups[1],size=cp1,replace=False))
            remaining=(np.setdiff1d(groups[0],pilot[0],assume_unique=True),
                       np.setdiff1d(groups[1],pilot[1],assume_unique=True))
            n0,n1=quotas(budget-cal,len(remaining[0]),len(remaining[1]),allocation)
            lams=(lambda_from_pilot(y,f,pilot[0]),lambda_from_pilot(y,f,pilot[1]))
        else:
            pilot=(np.empty(0,dtype=int),np.empty(0,dtype=int))
            remaining=groups
            n0,n1=quotas(budget,N0,N1,allocation)
            lams=(0.,0.) if label_only else (1.,1.)
        sample=(rng.choice(remaining[0],size=n0,replace=False),
                rng.choice(remaining[1],size=n1,replace=False))
        # Group-specific outcomes are Y on their own decision group, and zero
        # on the other. Therefore separate A/B total estimates are natural.
        a_total,va_total=_component(y,f,groups[0],pilot[0],sample[0],lams[0])
        b_total,vb_total=_component(y,f,groups[1],pilot[1],sample[1],lams[1])
        a,b=a_total/N,b_total/N
        va,vb=va_total/N**2,vb_total/N**2
    s=a+b
    fnr=a/s if math.isfinite(s) and s>1e-12 else float('nan')
    return {'A':a,'B':b,'FNR':fnr,'var_A':va,'var_B':vb,
            'calibration_labels':cal,'main_labels':budget-cal,'lambda_allowed':lams[0],
            'lambda_blocked':lams[1]}


def one_setup(df,setup,repetitions=REPETITIONS,budgets=BUDGETS):
    g=df.sort_values('example_id'); ids=g.example_id.astype(str).to_numpy()
    y=g.y.astype(int).to_numpy(); d=g.base_prediction.astype(int).to_numpy()
    f=g.base_score.astype(float).to_numpy()
    N=len(y)
    if N!=1687 or len(set(ids))!=N or int(y.sum())!=291:
        raise ValueError('development case count mismatches frozen protocol')
    if not np.isfinite(f).all(): raise ValueError('nonfinite cheap proxy')
    if not set(np.unique(y)).issubset({0,1}) or not set(np.unique(d)).issubset({0,1}):
        raise ValueError('nonbinary decisions/labels')
    truthA=float(np.mean(y*(1-d)));truthB=float(np.mean(y*d));truthF=truthA/(truthA+truthB)
    truths={'A':truthA,'B':truthB,'FNR':truthF}
    out=[];setup_code=0 if setup=='compact_after_rule' else 1
    for budget in budgets:
        for repetition in range(repetitions):
            for k,method in enumerate(METHODS):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,setup_code,budget,repetition,k]))
                e=estimate(y,d,f,method,budget,rng)
                row={'setup_id':setup,'budget':budget,'repetition':repetition,'method':method,
                     'calibration_labels':e['calibration_labels'],'main_labels':e['main_labels'],
                     'optional_monitor_calls':0,
                     'lambda_allowed':e['lambda_allowed'],'lambda_blocked':e['lambda_blocked']}
                for t in ('A','B','FNR'):
                    val=e[t]
                    row[t+'_error']=float(val-truths[t]) if math.isfinite(val) else float('nan')
                    row[t+'_estimable']=bool(math.isfinite(val))
                out.append(row)
    return out,{'N':N,'harmful_count':int(y.sum()),'allowed_count':int((d==0).sum()),
                'blocked_count':int((d==1).sum()),'A':truthA,'B':truthB,'FNR':truthF,
                'cheap_proxy_min':float(f.min()),'cheap_proxy_max':float(f.max())}


def summarize(allrows):
    rows=[]
    for (setup,method,budget),g in allrows.groupby(['setup_id','method','budget'],sort=True):
        rec={'setup_id':setup,'method':method,'human_budget':int(budget),'repetitions':len(g),
             'calibration_labels':int(g.calibration_labels.iloc[0]),
             'main_labels':int(g.main_labels.iloc[0]),'optional_monitor_calls':0}
        for name in ('A','B','FNR'):
            err=g[name+'_error'].to_numpy(dtype=float);valid=np.isfinite(err)
            rec[name+'_estimable_fraction']=float(valid.mean())
            rec[name+'_RMSE']=float(np.sqrt(np.mean(err[valid]**2))) if valid.any() else float('nan')
            rec[name+'_bias']=float(np.mean(err[valid])) if valid.any() else float('nan')
        rows.append(rec)
    return pd.DataFrame(rows)


def self_test():
    y=np.array([0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1],dtype=int)
    d=np.array([0]*14+[1]*10,dtype=int)
    f=np.array([0.1,0.7,0.8,0.2,0.65,0.1,0.8,0.3,0.9,0.7,0.15,0.82,0.3,0.91,
                .2,.65,.3,.8,.1,.75,.2,.9,.5,.8])
    allindices=np.arange(len(y));empty=np.empty(0,dtype=int)
    for group in (np.flatnonzero(d==0),np.flatnonzero(d==1)):
        total,v=_component(y,f,group,empty,group,1.0)
        assert abs(total-float(y[group].sum()))<1e-12 and abs(v)<1e-12
    assert quotas(10,14,10,'proportional')==(6,4)
    assert quotas(10,14,10,'allowed70')==(7,3)
    assert 0<=lambda_from_pilot(y,f,np.arange(10))<=3
    for method in ('uniform_srs','allowed_only_srs','proportional_labels_only',
                   'allowed70_labels_only','ppi_uniform_fixed1',
                   'stratppi_proportional_fixed1','stratppi_allowed70_fixed1'):
        e=estimate(y,d,f,method,10,np.random.default_rng(4))
        assert np.isfinite(e['A'])
        assert e['calibration_labels']==0
        assert e['main_labels']==10
    # Pilot labels are accounted for separately, not re-used in estimator.
    for method in ('stratppi_proportional_pilot_lambda','stratppi_allowed70_pilot_lambda'):
        e=estimate(y,d,f,method,12,np.random.default_rng(4))
        assert e['calibration_labels']==8 and e['main_labels']==4
        assert np.isfinite(e['A']) and np.isfinite(e['B'])
    # Exhaustive unbiasedness conditional on fixed pilot and fixed lambda.
    import itertools
    yy=np.array([1.,0.,1.,1.,0.]);ff=np.array([.9,.2,.8,.6,.1]);whole=np.arange(5)
    pilot=np.array([0]);rem=np.array([1,2,3,4])
    for lam in (0.,1.,.7):
        vals=[_component(yy,ff,whole,pilot,np.array(a),lam)[0]
              for a in itertools.combinations(rem,2)]
        assert abs(float(np.mean(vals))-float(yy.sum()))<1e-12
    print('PHASE5B2_SELF_TEST_PASS')


def synthetic_smoke_test():
    rng=np.random.default_rng(10)
    n=1687
    f=rng.uniform(0,1,size=n)
    d=(f>0.91).astype(int)
    # exactly 291 old-style positives, independent of selection logic
    y=np.zeros(n,dtype=int);y[rng.choice(n,size=291,replace=False)]=1
    df=pd.DataFrame({'example_id':[f'e{i:05d}' for i in range(n)],'y':y,
                     'base_prediction':d,'base_score':f})
    z,t=one_setup(df,'compact_after_rule',repetitions=3,budgets=(40,80))
    assert len(z)==len(METHODS)*3*2
    a=summarize(pd.DataFrame(z))
    assert len(a)==len(METHODS)*2
    assert (a.calibration_labels+a.main_labels==a.human_budget).all()
    assert (a.optional_monitor_calls==0).all()
    print('PHASE5B2_SYNTHETIC_PASS',len(z),'rows',len(a),'aggregate rows')


def main():
    self_test()
    if '--self-test' in sys.argv:
        synthetic_smoke_test();return 0
    root=Path.cwd().resolve()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,timeout=8).strip()
    if branch!=BRANCH:
        print('STOP: wrong branch',branch);return 2
    out=root/OUT
    if out.exists():
        print('STOP: output folder exists, preserved',out);return 2
    for p,checksum in HASHES.items():
        if not (root/p).is_file():
            print('STOP: input missing',p);return 2
        if sha256(root/p)!=checksum:
            raise ValueError('frozen source/protocol SHA mismatch: '+p)
    boundary=json.loads((root/BOUNDARY).read_text(encoding='utf-8'))
    allowed=set(boundary.get('data_boundary',{}).get('permitted_existing_development_artifacts',[]))
    if not {SOURCE,FOLDS}.issubset(allowed): raise ValueError('data boundary disallows source')
    protocol=json.loads((root/PROTOCOL).read_text(encoding='utf-8'))
    if protocol['scope']['expected_development_rows']!=1687:
        raise ValueError('wrong development cohort')
    cols=['example_id','setup_id','outer_fold','y','base_prediction','base_score']
    df=pd.read_parquet(root/SOURCE,columns=cols)
    folds=pd.read_csv(root/FOLDS,usecols=['example_id','outer_fold'],keep_default_na=False)
    if len(folds)!=1687 or folds.example_id.nunique()!=1687:
        raise ValueError('wrong frozen fold CSV')
    expected={q['setup_id'] for q in protocol['optional_monitor_setups']}
    if set(df.setup_id)!=expected or len(df)!=2*1687:
        raise ValueError('wrong two-setup development parquet')
    if df.groupby('example_id').y.nunique().max()!=1:
        raise ValueError('historical response labels disagree between setups')
    allrows=[];truths={}
    for setup in sorted(expected):
        subset=df.loc[df.setup_id==setup]
        jj=subset[['example_id','outer_fold']].merge(folds,on='example_id',validate='one_to_one',suffixes=('_data','_frozen'))
        if len(jj)!=1687 or not (jj.outer_fold_data.astype(str)==jj.outer_fold_frozen.astype(str)).all():
            raise ValueError('frozen folds changed')
        print('RUN',setup,'400 repetitions x 3 budgets x',len(METHODS),'methods',flush=True)
        r,t=one_setup(subset,setup);allrows.extend(r);truths[setup]=t
    frame=pd.DataFrame(allrows);summary=summarize(frame)
    if len(frame)!=len(expected)*len(BUDGETS)*REPETITIONS*len(METHODS):
        raise ValueError('wrong count')
    # verify enough valid FNR; the allowed-only arm is intentionally N/A
    bad=summary[(summary.method!='allowed_only_srs')&(summary.FNR_estimable_fraction<0.9)]
    if len(bad): raise ValueError('Too many invalid FNR denominators: '+str(bad.method.tolist()))
    out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'per_repetition_errors.csv',index=False,float_format='%.12g',lineterminator='\n')
    summary.to_csv(out/'stratppi_comparison.csv',index=False,float_format='%.12g',lineterminator='\n')
    manifest={'artifact_id':ARTIFACT,'status':'development_only_stratppi_mean_specialization',
       'research_baseline':'Fisch et al., NeurIPS 2024, StratPPI 1-dimensional mean case, not general M-estimator or full published CIs',
       'primary_estimand':'A=P(Y=1,D=0)','secondary':'FNR=A/(A+B)',
       'score_used':'base_score available on all development cases; not asserted calibrated',
       'historical_decision':'cross-fitted base_prediction, NOT deployed native guard',
       'historical_Y':'historical response labels not independently validated',
       'budgets':list(BUDGETS),'repetitions':REPETITIONS,'random_seed':SEED,
       'methods':list(METHODS),'lambda_fixed':1,'calibration_fraction_pilot':0.2,
       'pilot_allocation':'half to each D group, randomly sampled, disjoint from evaluation',
       'pilot_lambda_estimator':'sample covariance(y,base_score)/variance(base_score); clipped [-3,3]',
       'total_labels_accounting':'each pilot uses 8, 16 or 32 human labels included in the stated budget; not free',
       'optional_monitor_calls':0,'no_new_inference_timing':True,'no_human_review_time':True,
       'CIs':'not reported: full paper CI / tuning assumptions need separate implementation',
       'unbiasedness':'A and B design-unbiased conditional on pilot, FNR ratio not unbiased',
       'protected_rows_opened':False,'raw_records_exported':False,
       'source_branch':branch,'source_sha256':{k:sha256(root/k) for k in HASHES},
       'development_reference_truth':truths,
       'claim_limit':'exploratory old development. This is NOT Active MultiPPI and no prospective safety guarantee'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    notes='''# Phase 5B2 — StratPPI mean-estimation comparator, development only

This is the 1-D **mean specialization** of the StratPPI estimator from Fisch et al.
(NeurIPS 2024); it is NOT their full general estimator, proof or CI implementation.
The full source citation is https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract.html.

For D=0 and D=1 separate finite strata, use known cheap proxy f and independent
within-stratum probability samples to estimate each stratum harm prevalence as
`lambda*mean_all(f)+mean_reviewed(y-lambda*f)`. Combine known stratum weights
for A=allowed harmful responses and B=blocked harmful responses; report FNR=A/(A+B).

Fixed weights lambda=1 spend zero calibration labels. Independently tuned weights
spend 20% of TOTAL human budget on separate randomized pilot labels (balanced
between D strata) and only use the remaining labels for residual estimation.
The pilot labels contribute their *known* outcomes to the finite-population total;
this avoids throwing pilot observations away. All data/decisions are historical,
not newly annotated, and no optional Qwen scores are acquired.

**Scientific cautions:** Unstratified PPI on Y(1-D) is a comparison against the
stratified specialization. The primary A can be estimated from allowed-only
sampling; the latter cannot estimate FNR. FNR is a ratio and is not exactly
unbiased. The paper's asymptotic CI guarantee is not claimed here. Pilot-tuned
lambda is a project-specific finite-pool adaptation, not an exact replication
of every author experiment. This step cannot close the professor's Active
MultiPPI, independent review, cost or held-out validation requests.
'''
    (out/'README.md').write_text(notes,encoding='utf-8')
    archive=out/(ARTIFACT+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for q in ('stratppi_comparison.csv','per_repetition_errors.csv','manifest.json','README.md'):
            z.write(out/q,arcname=q)
    target=downloads()
    if target:
        destination=target/archive.name
        if not destination.exists():
            shutil.copy2(archive,destination)
            print('WINDOWS_DOWNLOADS',destination)
        else:print('PRESERVED old ZIP in Windows Downloads; new ZIP is',archive)
    else:print('ZIP saved under',archive)
    print('PHASE5B2_SUCCESS',len(frame),'repetition records',len(summary),'summary rows')
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        print('PHASE5B2_STOP',type(exc).__name__,str(exc));sys.exit(2)