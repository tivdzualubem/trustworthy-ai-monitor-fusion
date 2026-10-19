#!/usr/bin/env python3
"""Phase 5B1: fixed blocked/allowed probability-audit baselines.

Exploratory, old development response labels; no native guard claim, no tuning,
no optional model calls, and no access to sealed final-test/shift containers.

All review allocation fractions are fixed BEFORE examining response labels:
uniform SRS, proportional stratification, 50/50, 70% allowed, 70% blocked.
Finite-population probability estimators plus descriptive Wald intervals.
"""
from __future__ import annotations
import hashlib
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
OUT = 'results/phase5b1_blocked_allowed_baselines_v1'
ARTIFACT = 'phase5b1_blocked_allowed_baselines_v1'
BUDGETS = (40, 80, 160)
REPETITIONS = 400
SEED = 20261014
METHODS = ('uniform', 'proportional', 'equal_groups', 'allowed_70pct', 'blocked_70pct')
ESTIMANDS = ('allowed_harm_A', 'blocked_harm_B', 'FNR')
Z = 1.95996398454


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def downloads():
    try:
        w = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
             '[Environment]::GetFolderPath("UserProfile")'], timeout=8, text=True).strip().replace('\r','')
        pp = subprocess.check_output(['wslpath', '-u', w], timeout=5, text=True).strip()
        p = Path(pp) / 'Downloads'
        if p.is_dir(): return p
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    p = Path('/mnt/c/Users/user/Downloads')
    return p if p.is_dir() else None


def allocation(method, budget, N0, N1):
    """Deterministic fixed quotas, capped at population sizes; no Y dependence."""
    if not 4 <= budget <= N0 + N1 or min(N0,N1) < 2:
        raise ValueError('Bad population sizes/budget')
    if method == 'proportional':
        n0 = round(budget * N0/(N0+N1))
    elif method == 'equal_groups':
        n0 = budget // 2
    elif method == 'allowed_70pct':
        n0 = round(budget * .70)
    elif method == 'blocked_70pct':
        n0 = round(budget * .30)
    else:
        raise ValueError('Unknown stratified allocation')
    n0 = max(2, min(N0, n0))
    n1 = budget - n0
    if n1 > N1:
        n1 = N1
        n0 = budget - n1
    elif n1 < 2:
        n1 = 2
        n0 = budget - n1
    if n0 > N0 or n0 < 2 or n1 > N1 or n1 < 2:
        raise ValueError('Allocation could not satisfy quotas')
    return n0, n1


def estimate(y, d, sampled, groups=None):
    """Mean A and B estimated from probability sample; delta CI for FNR.

    If groups is None, full finite-population SRSWOR. Else sampled is a
    2-tuple of independently sampled indices, one from each D group.
    """
    N = len(y)
    if groups is None:
        indices = sampled
        n = len(indices)
        z = y[indices]*(1-d[indices])
        w = y[indices]*d[indices]
        a = z.mean(); b = w.mean()
        # Sample covariance of two finite-population random-vector outcomes.
        v = (1-n/N) * np.cov(np.vstack((z,w)), ddof=1) / n
    else:
        s0,s1 = sampled
        N0,N1 = map(len,groups)
        n0,n1 = len(s0),len(s1)
        a = N0/N * float(y[s0].mean())
        b = N1/N * float(y[s1].mean())
        v0 = (1-n0/N0) * float(y[s0].var(ddof=1)) / n0 * (N0/N)**2 if n0 < N0 else 0.
        v1 = (1-n1/N1) * float(y[s1].var(ddof=1)) / n1 * (N1/N)**2 if n1 < N1 else 0.
        v = np.diag([v0,v1])
    s = a+b
    fnr = a/s if s>0 else float('nan')
    grad = np.array([b/(s*s), -a/(s*s)]) if s>0 else np.array([float('nan')]*2)
    v_fnr = float(grad @ v @ grad) if s>0 else float('nan')
    est = np.array([a,b,fnr], dtype=float)
    variances = np.array([v[0,0],v[1,1],v_fnr],dtype=float)
    cis = []
    for val,var in zip(est,variances):
        half = Z*math.sqrt(max(0.,var)) if np.isfinite(var) else float('nan')
        cis.append((max(0.,val-half),min(1.,val+half)) if np.isfinite(half) else (float('nan'),float('nan')))
    return dict(zip(ESTIMANDS,est)), dict(zip(ESTIMANDS,cis))


def one_setup(df, setup):
    g = df.sort_values('example_id')
    ids = g.example_id.astype(str).to_numpy()
    y = g.y.astype(int).to_numpy()
    d = g.base_prediction.astype(int).to_numpy()
    N = len(y)
    if N != 1687 or len(set(ids))!=N or y.sum()!=291:
        raise ValueError('Historical development data unexpected')
    if not set(np.unique(y)).issubset({0,1}) or not set(np.unique(d)).issubset({0,1}):
        raise ValueError('Nonbinary labels/decisions')
    groups = (np.flatnonzero(d==0),np.flatnonzero(d==1))
    N0,N1 = map(len,groups)
    truth = {'allowed_harm_A':float(np.mean(y*(1-d))),
             'blocked_harm_B':float(np.mean(y*d))}
    truth['FNR'] = truth['allowed_harm_A']/(truth['allowed_harm_A']+truth['blocked_harm_B'])
    rows=[]
    setup_seed = 0 if setup=='compact_after_rule' else 1
    for budget in BUDGETS:
        for rep in range(REPETITIONS):
            # All methods use fixed, repeatable seeds, but different draws.
            for k,method in enumerate(METHODS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED,setup_seed,budget,rep,k]))
                if method=='uniform':
                    sampled = rng.choice(N,size=budget,replace=False)
                    est,cis = estimate(y,d,sampled)
                    n0 = int((d[sampled]==0).sum()); n1=budget-n0
                else:
                    n0,n1=allocation(method,budget,N0,N1)
                    sampled=(rng.choice(groups[0],size=n0,replace=False),
                             rng.choice(groups[1],size=n1,replace=False))
                    est,cis=estimate(y,d,sampled,groups)
                row = {'setup_id':setup,'method':method,'human_review_budget':budget,
                       'repetition':rep,'n_allowed_reviews':n0,'n_blocked_reviews':n1,
                       'calibration_labels':0,'additional_monitor_calls':0}
                for metric in ESTIMANDS:
                    row[f'{metric}_error'] = float(est[metric]-truth[metric])
                    row[f'{metric}_estimable'] = bool(np.isfinite(est[metric]))
                    row[f'{metric}_ci_covers'] = bool(cis[metric][0]<=truth[metric]<=cis[metric][1])
                rows.append(row)
    return rows,{'N':N,'allowed_total':N0,'blocked_total':N1,
                 'harmful_responses':int(y.sum()),'truth':truth,
                 'allowed_harmful_count':int(np.sum(y*(1-d))),
                 'blocked_harmful_count':int(np.sum(y*d))}


def summarize(df):
    out=[]
    for keys,g in df.groupby(['setup_id','method','human_review_budget'],sort=True):
        row=dict(zip(['setup_id','method','human_review_budget'],keys))
        row.update({'repetitions':len(g),'calibration_labels':0,'extra_monitor_calls':0,
                    'avg_n_allowed_reviews':float(g.n_allowed_reviews.mean()),
                    'avg_n_blocked_reviews':float(g.n_blocked_reviews.mean())})
        for name in ESTIMANDS:
            e=g[name+'_error'].to_numpy(dtype=float)
            finite=np.isfinite(e)
            row[name+'_fraction_estimable']=float(finite.mean())
            row[name+'_bias']=float(e[finite].mean()) if finite.any() else float('nan')
            row[name+'_RMSE']=float(np.sqrt(np.mean(e[finite]**2))) if finite.any() else float('nan')
            row[name+'_wald_CI95_coverage_descriptive']=float(g[name+'_ci_covers'].mean())
        out.append(row)
    return pd.DataFrame(out)


def self_test():
    rng=np.random.default_rng(11)
    y=np.array([1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,0,1],dtype=int)
    d=np.array([0]*12+[1]*8,dtype=int)
    groups=(np.flatnonzero(d==0),np.flatnonzero(d==1))
    exact_a=np.mean(y*(1-d)); exact_b=np.mean(y*d)
    a,_=estimate(y,d,np.arange(len(y)))
    b,_=estimate(y,d,groups,groups)
    assert abs(a['allowed_harm_A']-exact_a)<1e-12
    assert abs(b['blocked_harm_B']-exact_b)<1e-12
    for method in METHODS[1:]:
        n0,n1=allocation(method,10,12,8)
        assert n0+n1==10 and 2<=n0<=12 and 2<=n1<=8
        x=(rng.choice(groups[0],n0,False),rng.choice(groups[1],n1,False))
        estimates,cis=estimate(y,d,x,groups)
        assert all(np.isfinite(list(estimates.values())))
    # Exhaustive expectation for fixed-strata estimators (not FNR ratios).
    import itertools
    g0=np.array([0,1,2]);g1=np.array([3,4]);yy=np.array([1,0,1,0,1]);dd=np.array([0,0,0,1,1])
    vals=[]
    for c0 in itertools.combinations(g0,2):
      for c1 in itertools.combinations(g1,2):
        v,_=estimate(yy,dd,(np.array(c0),np.array(c1)),(g0,g1))
        vals.append((v['allowed_harm_A'],v['blocked_harm_B']))
    assert np.allclose(np.mean(vals,axis=0),[np.mean(yy*(1-dd)),np.mean(yy*dd)])
    print('PHASE5B1_SELF_TEST_PASS')


def main():
    self_test()
    if '--self-test' in sys.argv:
        return 0
    root = Path.cwd().resolve()
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,timeout=8).strip()
    if branch != BRANCH:
        print('STOP: wrong branch:',branch)
        return 2
    out=root/OUT
    if out.exists():
        print('STOP: existing output preserved:',out)
        return 2
    for p,digest in HASHES.items():
        if not (root/p).is_file():
            print('STOP: missing permitted development input:',p)
            return 2
        if sha256(root/p)!=digest:
            print('STOP: frozen SHA mismatch:',p)
            return 2
    boundary=json.loads((root/BOUNDARY).read_text())
    allowed=boundary['data_boundary']['permitted_existing_development_artifacts']
    if SOURCE not in allowed or FOLDS not in allowed:
        raise ValueError('Source not explicitly whitelisted by frozen data boundary')
    pr=json.loads((root/PROTOCOL).read_text())
    data=pd.read_parquet(root/SOURCE,columns=['example_id','setup_id','outer_fold','y','base_prediction'])
    fold=pd.read_csv(root/FOLDS,usecols=['example_id','outer_fold'],keep_default_na=False)
    if len(fold)!=1687 or fold.example_id.nunique()!=1687 or len(data)!=3374:
        raise ValueError('Development row counts changed')
    if data.groupby('example_id').y.nunique().max()!=1:
        raise ValueError('Response label disagrees across historical setups')
    expected={v['setup_id'] for v in pr['optional_monitor_setups']}
    if set(data.setup_id)!=expected:
        raise ValueError('Historical setups changed')
    rows=[]; populations={}
    for setup in sorted(expected):
        sub=data[data.setup_id==setup]
        check=sub[['example_id','outer_fold']].merge(fold,on='example_id',validate='one_to_one',suffixes=('_data','_source'))
        if len(check)!=1687 or not (check.outer_fold_data.astype(str)==check.outer_fold_source.astype(str)).all():
            raise ValueError('Frozen development fold IDs mismatch')
        print('RUN',setup,'fixed allocations × review budgets ×',REPETITIONS,'repetitions',flush=True)
        r,pop=one_setup(sub,setup);rows.extend(r);populations[setup]=pop
    frame=pd.DataFrame(rows); agg=summarize(frame)
    out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'per_repetition_scalar_errors.csv',index=False,float_format='%.12g',lineterminator='\n')
    agg.to_csv(out/'blocked_allowed_allocation_comparison.csv',index=False,float_format='%.12g',lineterminator='\n')
    manifest={'artifact':ARTIFACT,'status':'development_only_fixed_allocation_controls_not_prospective',
     'source_branch':branch,'sources_sha256':HASHES,'populations':populations,
     'budgets':list(BUDGETS),'methods':list(METHODS),'repetitions':REPETITIONS,'seed':SEED,
     'primary_estimand':'A=P(Y=1,D=0)','secondary_estimand':'FNR=A/(A+B)',
     'historical_decision':'base_prediction is archived cross-fitted classifier, not deployed native guard',
     'historical_response_labels':'not independently re-adjudicated',
     'calibration_labels_used':0,'additional_monitor_calls':0,
     'human_review_time':'not available; count only','guard_compute_cost':'not newly measured',
     'CIs':'descriptive plug-in finite-population Wald/delta; not validated prospective coverage',
     'method_status':'Fixed quota simple comparators only; not StratPPI or Active MultiPPI',
     'data_boundary':'only permitted development artifacts, protected tests unopened',
     'no_raw_records_exported':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    lines=['# Phase 5B1: blocked/allowed fixed review-allocation baselines','',
     'Exploratory historical development, not native guard deployment. Primary A=allowed harmful-response mass; secondary FNR.',
     'Fixed proportional, 50/50, 70% allowed and 70% blocked quotas require **zero tuning labels**.',
     'Uniform SRS included. All comparisons use identical human label counts and **zero** optional monitor calls.',
     'Estimation uses finite-population weights, and no unknown Y guides selection.',
     'Intervals are descriptive Wald/delta approximations and not valid prospective guarantees.',
     'These are the strong *simple* group-sampling controls; **not** StratPPI / Active MultiPPI.',
     '', '## FNR and allowed-harm RMSE', '',
     '| Setup | Reviews | Strategy | Allowed A RMSE | FNR RMSE | FNR CI coverage (descriptive) |',
     '|---|---:|---|---:|---:|---:|']
    for row in agg.itertuples(index=False):
        lines.append(f'| {row.setup_id} | {row.human_review_budget} | {row.method} | {row.allowed_harm_A_RMSE:.4f} | {row.FNR_RMSE:.4f} | {row.FNR_wald_CI95_coverage_descriptive:.3f} |')
    lines += ['', 'This pilot does not resolve StratPPI, Active MultiPPI, training/selection label charges, independent reviewers or full cost measurement.']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')
    archive = out/(ARTIFACT+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('per_repetition_scalar_errors.csv','blocked_allowed_allocation_comparison.csv','manifest.json','findings.md'):
            z.write(out/name,arcname=name)
    dest=downloads()
    if dest:
        target=dest/archive.name
        if target.exists():
            print('Preserved prior Downloads ZIP; new ZIP remains at',archive)
        else:
            shutil.copy2(archive,target)
            print('COPIED_TO_WINDOWS_DOWNLOADS',target)
    else:
        print('Windows Downloads unresolved, archive is:',archive)
    print('PHASE5B1_SUCCESS',len(frame),'repetition rows,',len(agg),'aggregate comparisons')
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print('PHASE5B1_STOP',type(e).__name__,str(e))
        sys.exit(2)