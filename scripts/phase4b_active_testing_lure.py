#!/usr/bin/env python3
"""Development-only Active Testing (Kossen et al., ICML 2021) LURE pilot.

Uses ONLY explicitly permitted pre-existing development data, not sealed/test
rows; writes aggregate output only. Label values are hidden from acquisition
except for earlier sampled labels. Archived base_prediction is NOT a native guard.

LURE weights reproduce Eq. (4) of Kossen et al. (2021) and the authors'
FancyUnbiasedRiskEstimator code. The acquisition surrogates here are simple,
stated adaptations, NOT faithful reproductions of their full neural-surrogate
experiments. No claim of unbiased FNR ratio or confidence interval coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

BRANCH = 'stage-a-paired-transport-20260921'
SOURCE = 'reports/decision_value_real_data/cross_fitted_decision_value_targets.parquet'
FOLDS = 'reports/decision_value_real_data/development_outer_fold_assignments.csv'
DV_PROTOCOL = 'configs/decision_value_real_data_protocol_v1.json'
BOUNDARY = 'configs/exact_cost_risk_cascade_protocol_v2.json'
PRIOR_DIR = 'results/phase4_dev_response_audit_sampling_v1'
OUT_DIR = 'results/phase4b_active_testing_lure_v1'
ARTIFACT = 'phase4b_active_testing_lure_v1'
M = (40, 80, 160)
REPS = 200
SEED = 20261009
MIX_UNIFORM = 0.20
METHODS = ('lure_fixed_cheap_rank', 'lure_adaptive_beta')
PAPER = 'https://proceedings.mlr.press/v139/kossen21a.html'
AUTHORS = 'https://github.com/jlko/active-testing/blob/main/activetesting/risk_estimators.py'


def digest(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def downloads():
    try:
        home = subprocess.check_output(['powershell.exe','-NoProfile','-Command',
                                        '[Environment]::GetFolderPath("UserProfile")'],
                                       text=True,timeout=8).strip().replace('\r','')
        dest = Path(subprocess.check_output(['wslpath','-u',home],
                                            text=True,timeout=5).strip()) / 'Downloads'
        if dest.is_dir():
            return dest
    except (OSError, subprocess.SubprocessError):
        pass
    fallback = Path('/mnt/c/Users/user/Downloads')
    return fallback if fallback.is_dir() else None


def midrank01(a: np.ndarray) -> np.ndarray:
    """Rank existing model scores without treating them as calibrated probabilities."""
    s = pd.Series(np.asarray(a, dtype=float))
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite pretrained cheap score')
    return ((s.rank(method='average') - 0.5) / len(s)).to_numpy(dtype=float)


def stratum_membership(D: np.ndarray, u: np.ndarray, ids: np.ndarray):
    """Four D x within-D median uncertainty strata, tie-breaking on ID."""
    labels = np.zeros(len(D), dtype=np.int8)
    for d in (0, 1):
        ix = np.flatnonzero(D == d)
        if not len(ix):
            continue
        order = np.lexsort((ids[ix], u[ix]))
        labels[ix[order[len(ix)//2:]]] = 1
    return (2 * D + labels).astype(np.int8)


def selection_distribution(remaining, D, ranks, strata, counts, successes, kind):
    k = len(remaining)
    if kind == 'lure_fixed_cheap_rank':
        # Fixed *uncalibrated* expected missed-harm proxy from available base_score.
        intensity = (0.15 + ranks[remaining]) * np.where(D[remaining] == 0, 1.0, 0.15)
    elif kind == 'lure_adaptive_beta':
        # Beta(2,8) prior per pre-review D x uncertainty stratum. Only previously
        # purchased y labels update counts/successes; no pool-label peeking.
        h = strata[remaining]
        posterior = (2.0 + successes[h]) / (10.0 + counts[h])
        intensity = 0.025 + posterior * np.where(D[remaining] == 0, 1.0, 0.25)
    else:
        raise ValueError('Unknown LURE proposal '+kind)
    if not np.isfinite(intensity).all() or np.any(intensity <= 0):
        raise ValueError('Proposal intensities require positive finite support')
    probs = MIX_UNIFORM / k + (1-MIX_UNIFORM) * intensity / intensity.sum()
    assert np.all(probs > 0) and np.isclose(probs.sum(), 1.0, atol=1e-12)
    return probs


def lure_weights(N: int, M_budget: int, sampled_q: np.ndarray):
    """Equation 4, Kossen et al. 2021, 1-indexed steps m=1,...,M.

    v_m=1+(N-M)/(N-m) * [ 1/((N-m+1)q_m) -1 ], for M < N.
    """
    if not 1 <= M_budget <= N or len(sampled_q) != M_budget:
        raise ValueError('Invalid budget/probability length')
    p = np.asarray(sampled_q, dtype=float)
    if np.any(p <= 0) or np.any(~np.isfinite(p)):
        raise ValueError('LURE requires known, positive proposal probabilities')
    if M_budget == N:
        return np.ones(N)
    m = np.arange(1, M_budget + 1, dtype=float)
    v = 1.0 + (N-M_budget)/(N-m) * (1/((N-m+1)*p)-1)
    if not np.isfinite(v).all() or np.any(v < -1e-9):
        raise ValueError('Invalid LURE weights')
    return v


def estimate_prefix(N, Dobs, Yobs, q, M_budget):
    """Linear finite-pool totals are unbiased; FNR/FPR ratios are NOT."""
    w = lure_weights(N, M_budget, q[:M_budget]) / M_budget
    y = Yobs[:M_budget]
    d = Dobs[:M_budget]
    miss = float(np.dot(w, y*(1-d)))
    harm = float(np.dot(w, y))
    false_alarm = float(np.dot(w, (1-y)*d))
    benign = float(np.dot(w, 1-y))
    return {'miss_est': miss,'harm_est': harm,
            'FNR_est': miss/harm if harm > 1e-12 else float('nan'),
            'FPR_est': false_alarm/benign if benign > 1e-12 else float('nan'),
            'weight_max': float(np.max(w * M_budget)),
            'weight_min': float(np.min(w * M_budget)),
            'weight_ess': float(np.sum(w)**2 / np.sum(w*w))}


def audit_path(y, D, score, u, ids, method, seed, n_draws):
    """Acquire at most max(M) labels sequentially; other labels stay hidden."""
    N = len(y)
    rank = midrank01(score)
    strata = stratum_membership(D, u, ids)
    n_groups = 4
    counts = np.zeros(n_groups, dtype=int)
    successes = np.zeros(n_groups, dtype=int)
    rng = np.random.default_rng(seed)
    remaining = np.arange(N)
    chosen_idx = np.empty(n_draws, dtype=int)
    chosen_q = np.empty(n_draws, dtype=float)
    for m in range(n_draws):
        q = selection_distribution(remaining, D, rank, strata, counts, successes, method)
        i = min(int(np.searchsorted(np.cumsum(q), rng.random(), side='right')), len(q)-1)
        idx = int(remaining[i])
        chosen_idx[m], chosen_q[m] = idx, q[i]
        # Observe y only AFTER acquisition. This is the sole adaptive label input.
        h = int(strata[idx])
        counts[h] += 1
        successes[h] += int(y[idx])
        remaining = np.delete(remaining, i)
    return chosen_idx, chosen_q


def finite_truth(y, D):
    harm = y.sum()
    benign = len(y)-harm
    return {'miss': float(np.mean(y*(1-D))),
            'harm': float(np.mean(y)),
            'FNR': float(np.sum(y*(1-D))/harm) if harm else float('nan'),
            'FPR': float(np.sum((1-y)*D)/benign) if benign else float('nan')}


def summarize(y,D,score,u,ids,setup,replications=REPS,budgets=M,seed=SEED):
    truth=finite_truth(y,D)
    results = {(method,b):[] for method in METHODS for b in budgets}
    for midx,method in enumerate(METHODS):
        for rep in range(replications):
            # Recorded deterministic streams; no label-conditioned method choice.
            draw_seed=np.random.SeedSequence([seed,midx,rep,0 if setup=='compact_after_rule' else 1])
            ix, probabilities=audit_path(y,D,score,u,ids,method,draw_seed,max(budgets))
            for b in budgets:
                e=estimate_prefix(len(y),D[ix],y[ix],probabilities,b)
                results[(method,b)].append(e)
    rows=[]
    for (method,b),values in results.items():
        def stats(k, reference):
            a=np.array([r[k] for r in values],dtype=float)
            usable=np.isfinite(a)
            return {'mean':float(np.mean(a[usable])) if usable.any() else None,
                    'bias':float(np.mean(a[usable]-reference)) if usable.any() else None,
                    'RMSE':float(np.sqrt(np.mean((a[usable]-reference)**2))) if usable.any() else None,
                    'fraction_estimable':float(usable.mean())}
        miss=stats('miss_est',truth['miss'])
        fnr=stats('FNR_est',truth['FNR'])
        fpr=stats('FPR_est',truth['FPR'])
        harm=stats('harm_est',truth['harm'])
        rows.append({'setup_id':setup,'method':method,'budget_requested':b,
                     'avg_human_reviews':b,'repetitions':replications,
                     'N_development':len(y),'true_miss_prevalence_dev':truth['miss'],
                     'true_harm_prevalence_dev':truth['harm'],
                     'true_FNR_dev':truth['FNR'],'true_FPR_dev':truth['FPR'],
                     'miss_est_mean':miss['mean'],'miss_bias':miss['bias'],
                     'miss_RMSE':miss['RMSE'],'harm_RMSE':harm['RMSE'],
                     'FNR_est_mean':fnr['mean'],'FNR_RMSE':fnr['RMSE'],
                     'FNR_fraction_estimable':fnr['fraction_estimable'],
                     'FPR_RMSE':fpr['RMSE'],
                     'FPR_fraction_estimable':fpr['fraction_estimable'],
                     'median_weight_ESS':float(np.median([v['weight_ess'] for v in values])),
                     'max_weight_observed':float(np.max([v['weight_max'] for v in values])),
                     'min_weight_observed':float(np.min([v['weight_min'] for v in values])),
                     'uncertainty_interval': 'not_calculated: no validated adaptive LURE CI'})
    return rows


def validate(df, assignment, protocol):
    req={'example_id','setup_id','outer_fold','y','base_score','base_prediction','base_uncertainty'}
    if not req.issubset(df.columns):
        raise ValueError('Missing columns: '+str(req-set(df.columns)))
    setups=sorted(x['setup_id'] for x in protocol['optional_monitor_setups'])
    n=int(protocol['scope']['expected_development_rows'])
    if assignment.example_id.nunique()!=n or len(assignment)!=n:
        raise ValueError('Development assignment count/unique mismatch')
    if len(df)!=n*len(setups) or set(df.setup_id)!=set(setups):
        raise ValueError('Development setup row count mismatch')
    if df.groupby('example_id').y.nunique().max()!=1:
        raise ValueError('Response labels vary between setups')
    for setup in setups:
        a=df[df.setup_id==setup]
        if len(a)!=n or a.example_id.nunique()!=n:
            raise ValueError('Each setup must have exactly one row per example')
        if not set(a.y).issubset({0,1}) or not set(a.base_prediction).issubset({0,1}):
            raise ValueError('Y and D must be binary')
        for field in ('base_score','base_uncertainty'):
            if not np.isfinite(pd.to_numeric(a[field],errors='coerce')).all():
                raise ValueError('Cheap score is not finite: '+field)
        check=a[['example_id','outer_fold']].merge(assignment,on='example_id',
                    how='inner',validate='one_to_one',suffixes=('_data','_frozen'))
        if len(check)!=n or not (check.outer_fold_data.astype(str)==check.outer_fold_frozen.astype(str)).all():
            raise ValueError('Outer fold mismatch')
    if df.drop_duplicates('example_id').y.sum()!=int(protocol['scope']['expected_positive_n']):
        raise ValueError('Historical development positive count mismatch')
    return setups


def test_math():
    y=np.array([1,0,1,0],dtype=int)
    d=np.array([0,0,1,0],dtype=int)
    qfirst=np.array([.1,.2,.3,.4])
    # Enumerate every possible length-2 adaptive trajectory to independently
    # check unbiasedness for both missed-harm and all-harm linear estimands.
    expected=np.zeros(2)
    for first in range(4):
        remaining=[z for z in range(4) if z!=first]
        qnext=np.array([.13,.31,.56]) if y[first] else np.array([.50,.30,.20])
        for j,second in enumerate(remaining):
            obs=np.array([first,second])
            weights=lure_weights(4,2,np.array([qfirst[first],qnext[j]])) / 2
            expected+=qfirst[first]*qnext[j]*np.array([
                float(np.dot(weights,y[obs]*(1-d[obs]))),
                float(np.dot(weights,y[obs]))])
    assert np.allclose(expected,[.25,.50],atol=1e-12),expected
    for n in (5,15):
        for m in (1,n//2,n):
            weights=lure_weights(n,m,np.array([1/(n-k) for k in range(m)]))
            assert np.allclose(weights,1),weights
    vals=estimate_prefix(4,d,y,np.array([1/4,1/3,1/2,1]),4)
    assert vals['miss_est']==.25 and vals['harm_est']==.50 and vals['FNR_est']==.50
    score=np.array([.1,.2,.3,.4]);unc=np.array([.2,.5,.7,.9])
    ids=np.array(['a','b','c','d'])
    for method in METHODS:
        i,q=audit_path(y,d,score,unc,ids,method,21,4)
        assert sorted(i.tolist())==[0,1,2,3] and all(q>0)
    # Extreme all-negative mini-population: ratio estimator not identifiable.
    e=estimate_prefix(4,np.array([0,1,0,1]),np.zeros(4),
                      np.array([.25,1/3,.5,1]),4)
    assert math.isnan(e['FNR_est'])
    print('SELF_TEST_PASS: enumerated adaptive LURE unbiasedness, uniform weights, full census, sampling support, FNR missing-denominator handling')


def write_results(root, rows, legacy, provenance, allow_copy=True):
    out=root / OUT_DIR
    if out.exists():
        raise FileExistsError('Existing Phase4B output left untouched: '+str(out))
    frame=pd.DataFrame(rows).sort_values(['setup_id','budget_requested','method'])
    check=legacy[['setup_id','method','budget_requested','miss_RMSE','FNR_RMSE']].copy()
    match=frame[['setup_id','method','budget_requested','miss_RMSE','FNR_RMSE']]
    combined=pd.concat([check,match],ignore_index=True).sort_values(['setup_id','budget_requested','method'])
    expected=2*3*(4+len(METHODS))
    if len(combined)!=expected:
        raise ValueError(f'Expected {expected} benchmark rows; got {len(combined)}')
    out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'lure_active_testing_metrics.csv',index=False,lineterminator='\n')
    combined.to_csv(out/'matched_baseline_comparison.csv',index=False,lineterminator='\n')
    (out/'active_testing_manifest.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n',encoding='utf8')
    report=['# Phase 4B: Active Testing LURE benchmark (development-only)',
            '', 'Source: Kossen et al., ICML 2021, Eq. 4, with authors\' FancyUnbiasedRiskEstimator.',
            'Paper: '+PAPER, 'Authors\' estimator: '+AUTHORS,
            '', '**Scope:** historical development Y and archived cross-fitted classifier D; NOT native-guard deployment.',
            'Positive-probability sequential active acquisition plus LURE unbiased *linear* mean estimates.',
            'FNR/FPR ratios are not unbiased; intervals not reported because they need separate validation.',
            'Risk-proxy strategies are project-specific adaptations, not exact authors\' experimental surrogates.',
            'All proposals use pre-review cheap scores and/or labels already purchased during that repetition.',
            '', '## Results and matched previous baselines','']
    for setup in provenance['setups']:
        report += ['### '+setup,'','| Budget | Method | Miss RMSE | FNR RMSE |', '|---:|---|---:|---:|']
        for _,x in combined.loc[combined.setup_id==setup].sort_values(['budget_requested','method']).iterrows():
            f='n/a' if pd.isna(x.FNR_RMSE) else f'{x.FNR_RMSE:.4f}'
            report.append(f'| {int(x.budget_requested)} | {x.method} | {x.miss_RMSE:.4f} | {f} |')
        report+=['']
    report += ['## Interpretation safeguards','',
      '- LURE is unbiased for finite-pool *means* of per-example losses under correct recorded nonzero q. Empirical Monte Carlo bias can be nonzero.',
      '- FNR uses estimated numerator / estimated harm prevalence; biased in finite samples, and may be undefined when estimated harm=0.',
      '- Existing Phase4A sampling controls use a different variance structure; comparison is same development pool, budgets and 200 repeats, not a randomized clinical superiority test.',
      '- `base_score` ranked for proposal construction, never assumed a calibrated probability.',
      '- Beta(2,8) updated exclusively from sampled audited labels. Fixed rank proposal uses no audited labels.',
      '- Sampling spends no optional Qwen score. This is NOT yet a priced extra-monitor selection study.',
      '- The original source labels and archived decisions can be imperfect; this is exploratory only.',
      '- Previous pilot calculated approximate CI coverage; Phase4B does not assert any confidence bound or matched CIs.',
      '- Original Stage-A 0/4 readout-transfer screening and all existing files remain untouched.',
      '', '## Stop/go', '',
      'Compare estimation error, variance and feasibility by method, setup and budget. If no consistent reduction over uniform or stratified auditing, report a negative result and prioritize valid labels/decision provenance before new methods.','']
    (out/'interpretation.md').write_text('\n'.join(report),encoding='utf8')
    archive=out/(ARTIFACT+'.zip')
    names=['lure_active_testing_metrics.csv','matched_baseline_comparison.csv','active_testing_manifest.json','interpretation.md']
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            zf.write(out/n,arcname=n)
    if allow_copy:
        dest=downloads()
        if dest:
            target=dest/archive.name
            if target.exists():
                print('Existing Windows archive preserved. New ZIP is in WSL:',archive)
            else:
                shutil.copy2(archive,target)
                print('Windows Downloads:',target)
        else:
            print('Windows Downloads unavailable. ZIP saved in WSL:',archive)
    print('PHASE4B_SUCCESS:',len(frame),'new rows,',len(combined),'total comparator rows;',
          'no raw examples exported or old files changed')
    return archive


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true',help='Check estimator mathematics without opening repository data')
    args=parser.parse_args()
    test_math()
    if args.self_test:
        return 0
    root=Path.cwd().resolve()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,timeout=6).strip()
    if branch != BRANCH:
        print('Not running: expected branch',BRANCH,'found',branch)
        return 2
    if (root/OUT_DIR).exists():
        print('Not running: new output directory already exists; no overwrite:',OUT_DIR)
        return 2
    required=[SOURCE,FOLDS,DV_PROTOCOL,BOUNDARY,
              PRIOR_DIR+'/audit_method_comparison.csv',
              PRIOR_DIR+'/audit_pilot_manifest.json']
    missing=[str(x) for x in required if not (root/x).is_file()]
    if missing:
        print('Not running: missing permitted files:',missing)
        return 2
    protocol=json.loads((root/DV_PROTOCOL).read_text(encoding='utf8'))
    boundary=json.loads((root/BOUNDARY).read_text(encoding='utf8'))
    allowed=boundary.get('data_boundary',{}).get('permitted_existing_development_artifacts',[])
    if SOURCE not in allowed or FOLDS not in allowed:
        print('Not running: development inputs not explicitly whitelisted by protocol')
        return 2
    old_manifest=json.loads((root/(PRIOR_DIR+'/audit_pilot_manifest.json')).read_text(encoding='utf8'))
    required_hashes={'source_sha256':digest(root/SOURCE),
                     'fold_source_sha256':digest(root/FOLDS),
                     'decision_value_protocol_sha256':digest(root/DV_PROTOCOL),
                     'boundary_protocol_sha256':digest(root/BOUNDARY)}
    for k,hash_value in required_hashes.items():
        if old_manifest.get(k)!=hash_value:
            print('Not running: frozen provenance differs from Phase 4A:',k)
            return 2
    if old_manifest.get('budgets') != list(M) or old_manifest.get('repetitions') != REPS:
        print('Not running: unmatched prior audit budget/repetition protocol')
        return 2
    columns=['example_id','setup_id','outer_fold','y','base_score','base_prediction','base_uncertainty']
    df=pd.read_parquet(root/SOURCE,columns=columns)
    assignment=pd.read_csv(root/FOLDS,usecols=['example_id','outer_fold'],keep_default_na=False)
    setups=validate(df,assignment,protocol)
    old=pd.read_csv(root/(PRIOR_DIR+'/audit_method_comparison.csv'))
    if len(old)!=2*3*4 or set(old.setup_id)!=set(setups):
        raise ValueError('Old pilot method-comparison records do not match the expected scope')
    if set(old.method)!={'uniform','proportional_stratified','score_tilted_stratified','allowed_only'}:
        raise ValueError('Old pilot method set changed')
    new=[]
    for setup in setups:
        g=df.loc[df.setup_id==setup].sort_values('example_id')
        ids=g.example_id.astype(str).to_numpy()
        y=g.y.to_numpy(dtype=int)
        d=g.base_prediction.to_numpy(dtype=int)
        score=g.base_score.to_numpy(dtype=float)
        u=g.base_uncertainty.to_numpy(dtype=float)
        # Refuse misleading comparison if the previous run's finite-pool truths differ.
        truth=finite_truth(y,d)
        old_setup=old.loc[old.setup_id==setup]
        if set(old_setup.budget_requested.astype(int))!=set(M) or len(old_setup)!=len(M)*4:
            raise ValueError('Old audit budgets do not match for '+setup)
        for field,ref in (('true_miss_prevalence_dev',truth['miss']),
                          ('true_harm_prevalence_dev',truth['harm']),
                          ('true_FNR_dev',truth['FNR']),('true_FPR_dev',truth['FPR'])):
            vals=old_setup[field].astype(float).to_numpy()
            if not np.allclose(vals,ref,atol=1e-12,rtol=0):
                raise ValueError('Prior pilot uses a different population/decision: '+setup+' '+field)
        new.extend(summarize(y,d,score,u,ids,setup))
    provenance={'artifact_id':ARTIFACT,'status':'development_only_exploratory_published_estimator_adaptation',
                'source_branch':branch,'setups':setups,'budgets':list(M),'repetitions':REPS,
                'seed':SEED,'methods':list(METHODS),'probability_mixture_uniform':MIX_UNIFORM,
                'sampling_unit':'development response example, once per setup',
                'Y':'historical development response-level harmfulness, simulated labels revealed only after review',
                'D':'historical cross-fitted base_prediction NOT a single native guard decision',
                'estimator':'Kossen et al. 2021 LURE Eq. 4 adapted to z=Y*(1-D), Y and (1-Y)*D; finite-pool linear means',
                'proposal_fixed':'score rank x allowed/blocked multiplier; rank NOT a calibrated probability',
                'proposal_adaptive':'four D/uncertainty strata, Beta(2,8) with purchased-label updates only',
                'positive_q':True,'optional_monitor_scores_used':False,
                'limitations':'FNR/FPR plug-in ratios generally biased; LURE normal CIs not supplied; no new independent response annotations',
                'sealed_rows_accessed':False,'raw_examples_exported':False,
                'source_paths':[SOURCE,FOLDS],
                'prior_aggregate_baseline':PRIOR_DIR+'/audit_method_comparison.csv',
                'prior_aggregate_sha256':digest(root/(PRIOR_DIR+'/audit_method_comparison.csv')),
                'library_reference':PAPER,'authors_reference_implementation':AUTHORS,
                'version':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__},
                'input_sha256':required_hashes}
    write_results(root,new,old,provenance)
    return 0

if __name__=='__main__':
    sys.exit(main())
