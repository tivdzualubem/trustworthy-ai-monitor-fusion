#!/usr/bin/env python3
"""Development-only, read-only, design-based safety audit feasibility pilot.

No sealed rows; no new model inference; no row-level exports. Archived cross-fitted
base_prediction is a *proxy decision*, NOT a deployed frozen native guard policy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

REPO_BRANCH = 'stage-a-paired-transport-20260921'
DATA_REL = 'reports/decision_value_real_data/cross_fitted_decision_value_targets.parquet'
FOLD_REL = 'reports/decision_value_real_data/development_outer_fold_assignments.csv'
DV_PROTOCOL = 'configs/decision_value_real_data_protocol_v1.json'
BOUNDARY_PROTOCOL = 'configs/exact_cost_risk_cascade_protocol_v2.json'
OUT_REL = 'results/phase4_dev_response_audit_sampling_v1'
ARTIFACT = 'phase4_dev_response_audit_sampling_v1'
SEED = 20261009
BUDGETS = (40, 80, 160)
REPETITIONS = 200
METHODS = ('uniform', 'proportional_stratified', 'score_tilted_stratified', 'allowed_only')
Z975 = 1.959963984540054


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def windows_downloads():
    try:
        home = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
                                        '[Environment]::GetFolderPath("UserProfile")'],
                                       text=True, timeout=8).strip().replace('\r', '')
        path = Path(subprocess.check_output(['wslpath', '-u', home], text=True, timeout=5).strip()) / 'Downloads'
        if path.is_dir():
            return path
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    fallback = Path('/mnt/c/Users/user/Downloads')
    return fallback if fallback.is_dir() else None


def allocate(N, budget, weights):
    """Deterministic, capacity-respecting allocation, min 2 per available stratum."""
    N = np.asarray(N, dtype=int)
    weights = np.asarray(weights, dtype=float)
    assert np.all(N > 0) and np.all(np.isfinite(weights)) and np.all(weights > 0)
    min_each = np.minimum(N, 2)
    if budget < int(min_each.sum()) or budget > int(N.sum()):
        raise ValueError('Invalid audit budget for strata')
    alloc = min_each.copy()
    remaining = budget - int(alloc.sum())
    while remaining:
        candidates = np.flatnonzero(alloc < N)
        # Progressive target allocation uses only population strata sizes and cheap scores.
        score = weights[candidates] / (alloc[candidates] + 1.0)
        j = int(candidates[np.argmax(score)])
        alloc[j] += 1
        remaining -= 1
    assert int(alloc.sum()) == budget
    return alloc


def cheap_strata(D, uncertainty, ids):
    """At most four nonempty strata defined without consulting harm labels."""
    D = np.asarray(D, dtype=int)
    u = np.asarray(uncertainty, dtype=float)
    ids = np.asarray(ids, dtype=str)
    if not np.isfinite(u).all():
        raise ValueError('Nonfinite cheap uncertainty')
    h = np.zeros(len(D), dtype=int)
    for d in (0, 1):
        at = np.flatnonzero(D == d)
        if len(at) == 0:
            continue
        # Deterministic tie break even when all uncertainty values are identical.
        order = np.lexsort((ids[at], u[at]))
        h[at[order[len(at) // 2:]]] = 1
    labels = 2 * D + h
    return [np.flatnonzero(labels == x) for x in range(4) if np.any(labels == x)], labels


def estimate_stratified(y, D, groups, allocation, rng, N_total):
    """Design-based HT totals (stratified SRSWOR); ratio FNR/FPR are not unbiased."""
    total_z = total_y = total_b = total_fp = 0.0
    var_zmean = 0.0
    audited = 0
    for g, nh in zip(groups, allocation):
        Nh = len(g)
        draw = rng.choice(g, size=int(nh), replace=False)
        audited += int(nh)
        yy = y[draw]
        dd = D[draw]
        zz = yy * (1 - dd)
        fp = (1 - yy) * dd
        w = Nh / int(nh)
        total_z += w * float(zz.sum())
        total_y += w * float(yy.sum())
        total_b += w * float((1 - yy).sum())
        total_fp += w * float(fp.sum())
        if nh > 1:
            s2 = float(np.var(zz, ddof=1))
            var_zmean += (Nh / N_total)**2 * (1 - nh / Nh) * s2 / nh
    p = total_z / N_total
    halfwidth = Z975 * math.sqrt(max(0.0, var_zmean))
    return {'audits': audited, 'miss_est': p, 'harm_est': total_y / N_total,
            'fnr_est': total_z / total_y if total_y > 0 else float('nan'),
            'fpr_est': total_fp / total_b if total_b > 0 else float('nan'),
            'miss_ci_lo': max(0.0, p-halfwidth), 'miss_ci_hi': min(1.0, p+halfwidth)}


def estimate_allowed_only(y, D, budget, rng):
    """Only missed-harm *prevalence* is identifiable from this sampled subset."""
    N = len(y)
    allow = np.flatnonzero(D == 0)
    if not len(allow):
        return {'audits':0, 'miss_est':0.0, 'harm_est':float('nan'),
                'fnr_est':float('nan'), 'fpr_est':float('nan'),
                'miss_ci_lo':0.0, 'miss_ci_hi':0.0}
    n = min(budget, len(allow))
    draw = rng.choice(allow, size=n, replace=False)
    zz = y[draw].astype(float)
    w = len(allow)/N
    p = w*float(zz.mean())
    variance = w**2 * (1-n/len(allow)) * (float(np.var(zz,ddof=1))/n if n>1 else 0)
    halfwidth = Z975*math.sqrt(max(variance, 0.0))
    return {'audits':n, 'miss_est':p, 'harm_est':float('nan'),
            'fnr_est':float('nan'), 'fpr_est':float('nan'),
            'miss_ci_lo':max(0.0,p-halfwidth), 'miss_ci_hi':min(1.0,p+halfwidth)}


def summarize_simulation(y, D, scores, ids, budget, repetitions, seed):
    """Returns aggregate results only. `y` never determines acquisition or strata."""
    N = len(y)
    assert N == len(D) == len(scores) == len(ids)
    groups, group_labels = cheap_strata(D, scores, ids)
    truth_z = y*(1-D)
    true_miss = float(truth_z.mean())
    true_harm = float(y.mean())
    true_fnr = float(truth_z.sum()/y.sum()) if y.sum() else float('nan')
    true_fpr = float(((1-y)*D).sum()/(1-y).sum()) if (1-y).sum() else float('nan')
    rows = []
    group_sizes = np.array([len(g) for g in groups], dtype=int)
    cheap_priority_weights = []
    for g in groups:
        # Multipliers rely on observed no-charge D and uncertainty rank *only*.
        multiplier = (2.0 if D[g[0]] == 0 else 0.6)
        multiplier *= (1.4 if group_labels[g[0]]%2 == 1 else 1.0)
        cheap_priority_weights.append(len(g)*multiplier)
    for midx, method in enumerate(METHODS):
        if method == 'proportional_stratified':
            alloc = allocate(group_sizes, budget, group_sizes)
        elif method == 'score_tilted_stratified':
            alloc = allocate(group_sizes, budget, cheap_priority_weights)
        elif method == 'uniform':
            alloc = np.array([budget], dtype=int)
        elif method == 'allowed_only':
            alloc = None
        estimates=[]
        for rep in range(repetitions):
            # Distinct deterministic streams; no label-dependent tuning.
            rng = np.random.default_rng(np.random.SeedSequence([seed, budget, midx, rep]))
            if method == 'allowed_only':
                e = estimate_allowed_only(y,D,budget,rng)
            else:
                g = [np.arange(N)] if method == 'uniform' else groups
                e = estimate_stratified(y,D,g,alloc,rng,N)
            estimates.append(e)
        df = pd.DataFrame(estimates)
        p = df['miss_est'].to_numpy(float)
        f = df['fnr_est'].to_numpy(float)
        valid_fn = np.isfinite(f) & math.isfinite(true_fnr)
        rows.append({
            'method':method, 'budget_requested':budget, 'repetitions':repetitions,
            'avg_human_reviews':float(df['audits'].mean()),
            'N_development':N,'N_allowed':int((D==0).sum()),'N_blocked':int((D==1).sum()),
            'true_miss_prevalence_dev':true_miss,'true_harm_prevalence_dev':true_harm,
            'true_FNR_dev':true_fnr,'true_FPR_dev':true_fpr,
            'miss_est_mean':float(p.mean()),'miss_bias':float(p.mean()-true_miss),
            'miss_MAE':float(np.abs(p-true_miss).mean()),
            'miss_RMSE':float(np.sqrt(np.mean((p-true_miss)**2))),
            'miss_95pct_normal_CI_coverage':float(np.mean((df['miss_ci_lo'] <= true_miss)&(true_miss <=df['miss_ci_hi']))),
            'miss_95pct_CI_mean_width':float((df['miss_ci_hi']-df['miss_ci_lo']).mean()),
            'FNR_est_mean':float(f[valid_fn].mean()) if valid_fn.any() else None,
            'FNR_RMSE':float(np.sqrt(np.mean((f[valid_fn]-true_fnr)**2))) if valid_fn.any() else None,
            'FNR_fraction_estimable':float(valid_fn.mean()),
            'FPR_fraction_estimable':float(np.isfinite(df['fpr_est']).mean()),
        })
    return rows


def validate_input(df, assign, protocol):
    needed={'example_id','setup_id','outer_fold','y','base_score','base_prediction','base_uncertainty'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing cross-fitted columns: {needed-set(df.columns)}')
    setups={x['setup_id'] for x in protocol['optional_monitor_setups']}
    if set(df['setup_id'])!=setups:
        raise ValueError('Setup mismatch with original protocol')
    n=protocol['scope']['expected_development_rows']
    assign2=assign[['example_id','outer_fold']].copy()
    if len(assign2)!=n or assign2.example_id.nunique()!=n:
        raise ValueError('Development fold assignment size/uniqueness failed')
    for setup in sorted(setups):
        x=df.loc[df['setup_id']==setup]
        if len(x)!=n or x.example_id.nunique()!=n:
            raise ValueError('One row per example per setup required')
        if not set(x['y']).issubset({0,1}) or not set(x['base_prediction']).issubset({0,1}):
            raise ValueError('Nonbinary y or D')
        if not np.isfinite(pd.to_numeric(x['base_uncertainty'],errors='coerce')).all():
            raise ValueError('Nonfinite pre-review uncertainty')
        j=x[['example_id','outer_fold']].merge(assign2,on='example_id',validate='one_to_one',suffixes=('_data','_assign'))
        if len(j)!=n or (j['outer_fold_data'].astype(str)!=j['outer_fold_assign'].astype(str)).any():
            raise ValueError('Fold assignments do not match')
    if df.groupby('example_id')['y'].nunique().max()!=1:
        raise ValueError('Response-level y differs between setups')
    if df.groupby('example_id')['outer_fold'].nunique().max()!=1:
        raise ValueError('Fold differs between setups')
    return sorted(setups)


def self_test():
    a=np.array([3,5,8]); al=allocate(a,10,a); assert sum(al)==10 and np.all(al<=a) and np.all(al>=2)
    y=np.array([1,0,1,0,1,0,0,1,0,0,1,0],dtype=int)
    D=np.array([0,0,0,0,0,0,1,1,1,1,1,1],dtype=int)
    u=np.linspace(.1,.9,len(y))
    ids=np.array([f'x{i}' for i in range(len(y))])
    g,_=cheap_strata(D,u,ids)
    full=estimate_stratified(y,D,g,np.array([len(x) for x in g]),np.random.default_rng(1),len(y))
    assert abs(full['miss_est']-float((y*(1-D)).mean()))<1e-12
    assert abs(full['fnr_est']-float((y*(1-D)).sum()/y.sum()))<1e-12
    assert full['miss_ci_lo']==full['miss_ci_hi']
    a=estimate_allowed_only(y,D,6,np.random.default_rng(2))
    assert abs(a['miss_est']-float((y*(1-D)).mean()))<1e-12
    assert math.isnan(a['fnr_est'])
    sm=summarize_simulation(y,D,u,ids,8,7,11)
    assert len(sm)==4 and all(x['repetitions']==7 for x in sm)
    print('SELF_TEST_PASS: strata, fixed-budget sampling, HT totals, allow-only identifiability, zero-variance census, seeded simulation')


def main():
    root=Path.cwd().resolve()
    try:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):
        branch='unknown'
    if branch!=REPO_BRANCH:
        print(f'Not running: expected branch {REPO_BRANCH}, observed {branch}')
        return 2
    out=root/OUT_REL
    if out.exists():
        print(f'Not running: existing results preserved at {out}')
        return 2
    required=[DATA_REL,FOLD_REL,DV_PROTOCOL,BOUNDARY_PROTOCOL]
    if any(not (root/f).is_file() for f in required):
        print('Not running: a required permitted development file is missing')
        return 2
    dv=json.loads((root/DV_PROTOCOL).read_text('utf8'))
    boundary=json.loads((root/BOUNDARY_PROTOCOL).read_text('utf8'))
    allowed=boundary.get('data_boundary',{}).get('permitted_existing_development_artifacts',[])
    if not set((DATA_REL,FOLD_REL)).issubset(set(allowed)):
        print('Not running: source paths not on protocol whitelist')
        return 2
    self_test()
    df=pd.read_parquet(root/DATA_REL, columns=['example_id','setup_id','outer_fold','y','base_score','base_prediction','base_uncertainty'])
    assign=pd.read_csv(root/FOLD_REL, usecols=['example_id','outer_fold'],keep_default_na=False)
    setups=validate_input(df,assign,dv)
    rows=[]
    for setup in setups:
        d=df.loc[df.setup_id==setup].sort_values('example_id')
        ids=d.example_id.astype(str).to_numpy()
        y=d.y.astype(int).to_numpy()
        D=d.base_prediction.astype(int).to_numpy()
        u=d.base_uncertainty.astype(float).to_numpy()
        for budget in BUDGETS:
            if budget>=len(d):
                raise ValueError('Budget must be smaller than pool')
            for a in summarize_simulation(y,D,u,ids,budget,REPETITIONS,SEED):
                a['setup_id']=setup
                rows.append(a)
    table=pd.DataFrame(rows)
    # Report only aggregate data and protocol hashes, never row-level ids/labels/scores.
    provenance={
        'artifact_id':ARTIFACT,'status':'development_only_exploratory_sampling_feasibility',
        'source_branch':branch, 'data_source':DATA_REL,'source_sha256':sha256(root/DATA_REL),
        'fold_source':FOLD_REL,'fold_source_sha256':sha256(root/FOLD_REL),
        'decision_value_protocol_sha256':sha256(root/DV_PROTOCOL),
        'boundary_protocol_sha256':sha256(root/BOUNDARY_PROTOCOL),
        'development_rows_per_setup':dv['scope']['expected_development_rows'],
        'setups':setups,'seed':SEED,'budgets':BUDGETS,'repetitions':REPETITIONS,
        'methods':METHODS,'python':sys.version.split()[0], 'numpy':np.__version__,'pandas':pd.__version__,
        'sampling_unit':'development_example_id',
        'observed_D':'archived cross-fitted base_prediction (trained model; NOT a single frozen native guard)',
        'Y':'historical development response-level label under older audit provenance, not newly independent truth',
        'primary_estimand':'development finite-pool p_miss = mean(Y*(1-D))',
        'secondary_estimands':'FNR and FPR are ratios; allowed-only design cannot estimate them',
        'risk_boundary':'No fresh W0 data, sealed rows, model inference, prompt-only Stage-A, or confirmatory inference',
        'variance_note':'Normal design-based finite-population CI may undercover for rare outcomes; coverage is empirical, not a certificate',
        'label_cost':'one human audit per selected unique example; original historical labels simulated as hidden until sampled',
        'optional_score_use':'NONE; no unpurchased optional-monitor score informs sampling',
        'scope_limit':'No native-guard deployment FNR claims. Existing labels and scores may have historical selection/provenance limitations.',
        'not_implemented':'Published active-testing algorithms and optional-monitor acquisition are later comparisons, NOT claimed here.',
        'raw_rows_exported':False,'protected_split_rows_read':False}
    # Guard against accidental overwrite while building.
    out.mkdir(parents=True,exist_ok=False)
    table.to_csv(out/'audit_method_comparison.csv',index=False,lineterminator='\n')
    (out/'audit_pilot_manifest.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n',encoding='utf8')
    report=[
        '# Development-only sampling pilot (exploratory)',
        '', 'No deployed native-guard result, causal inference, FNR guarantee, or new safety certificate.',
        'All estimators use historical response-level development labels. No sealed data were accessed.',
        'The decision column is archived cross-fitted `base_prediction`, not a single fixed deployed guard.',
        'Uniform, proportional stratification, score-tilted stratification, and allow-only sampling use only',
        'cost-free pre-review fields (D, base_uncertainty) for audit selection. Optional scores are unused.',
        '', '## Results', '',
    ]
    for setup in setups:
        report+=['### '+setup,'','| Budget | Method | Miss RMSE | Miss bias | 95% CI coverage | FNR RMSE |',
                 '|---:|---|---:|---:|---:|---:|']
        subset=table[table.setup_id==setup]
        for _,z in subset.sort_values(['budget_requested','method']).iterrows():
            fm='n/a' if pd.isna(z['FNR_RMSE']) else f"{z['FNR_RMSE']:.4f}"
            report.append(f"| {int(z['budget_requested'])} | {z['method']} | {z['miss_RMSE']:.4f} | {z['miss_bias']:+.4f} | {z['miss_95pct_normal_CI_coverage']:.3f} | {fm} |")
        report+=['',f"Development finite-pool miss prevalence: {subset.iloc[0]['true_miss_prevalence_dev']:.5f}", '']
    report+=['## Interpretation warnings','','- The allow-only strategy identifies miss prevalence, **not FNR** because its audit never labels blocked outcomes.',
             '- The normal confidence intervals are an exploratory design-based approximation, not a validated rare-event bound.',
             '- Historical development labels are already known to the researcher and may include label errors or source effects.',
             '- Scores and cross-fitted decisions derive from historically fitted models; this is no clean prospective deployment test.',
             '- No nested strategy tuning on full labels and no published active-testing implementation is claimed.',
             '- Next: verify native prompt-response guard D, incorporate optional-monitor measurement cost, and then evaluate published active designs.', '']
    (out/'interpretation.md').write_text('\n'.join(report),encoding='utf8')
    archive=out/(ARTIFACT+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for n in ['audit_method_comparison.csv','audit_pilot_manifest.json','interpretation.md']:
            zf.write(out/n,arcname=n)
    dest=windows_downloads()
    if dest is not None:
        target=dest/archive.name
        if target.exists():
            print('Existing Windows file preserved; fresh archive remains:',archive)
        else:
            shutil.copy2(archive,target)
            print('Windows Downloads:',target)
    else:
        print('Windows Downloads not located; archive saved:',archive)
    print('SUCCESS: 2 setups x 3 budgets x 4 methods, 200 repetitions each; no source files modified')
    print('All reported results are aggregate only. Upload ZIP for analysis.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
